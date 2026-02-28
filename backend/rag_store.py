# backend/rag_store.py
# ==============================================================================
# Modulo de almacenamiento vectorial y ingestion de documentos.
#
# Responsabilidades:
#   - Extraer texto de PDFs (nativo con pdfplumber o OCR con EasyOCR)
#   - Limpiar y normalizar texto extraido
#   - Dividir en chunks con RecursiveCharacterTextSplitter
#   - Generar embeddings con HuggingFace all-MiniLM-L6-v2
#   - Almacenar en ChromaDB con metadata enriquecida
#   - Proveer acceso al vector store para el grafo RAG
#
# Metadata por chunk:
#   source, page, marca, modelo, doc_id, chunk_id, chunk_index, ocr
# ==============================================================================
from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import List

import pdfplumber
import easyocr
from PIL import Image
import numpy as np

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import TextLoader

logger = logging.getLogger(__name__)

# ── Configuracion ────────────────────────────────────────────────────────────
# Directorio de persistencia de ChromaDB (relativo a backend/)
PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
# Nombre de la coleccion en ChromaDB
COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_collection")

# Minimo de caracteres para considerar que una pagina tiene texto nativo.
# Paginas con menos de 50 chars se procesan con OCR.
MIN_CHARS_PARA_TEXTO = 50

# Lector OCR compartido — se inicializa una sola vez (singleton)
_lector_ocr: easyocr.Reader | None = None


def _obtener_lector_ocr() -> easyocr.Reader:
    """Obtiene o inicializa el lector OCR (singleton).

    Primera invocacion descarga modelos de EasyOCR (~100 MB).
    Soporta espanol e ingles, sin GPU.

    Returns:
        Instancia de easyocr.Reader lista para usar.
    """
    global _lector_ocr
    if _lector_ocr is None:
        logger.info("Inicializando EasyOCR (primera vez descarga modelos ~100 MB)...")
        _lector_ocr = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
        logger.info("EasyOCR listo.")
    return _lector_ocr


def _doc_id_desde_pdf(ruta_pdf: Path) -> str:
    """Genera un identificador estable de documento desde el nombre del archivo PDF.

    Normaliza a minusculas y reemplaza caracteres no alfanumericos por '_'.
    Ejemplo: 'Ficha-Tecnica-Hilux-2025.pdf' → 'ficha_tecnica_hilux_2025'

    Args:
        ruta_pdf: Ruta al archivo PDF.

    Returns:
        Identificador normalizado del documento.
    """
    stem = ruta_pdf.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    return stem.strip("_")


def _doc_id_desde_path(path: Path) -> str:
    """Genera doc_id desde cualquier ruta de archivo (PDF o TXT).

    Misma logica que _doc_id_desde_pdf pero para rutas genericas.

    Args:
        path: Ruta al archivo.

    Returns:
        Identificador normalizado del documento.
    """
    stem = path.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    return stem.strip("_")


def _inferir_modelo(ruta_pdf: Path) -> str:
    """Deriva un nombre de modelo legible a partir del nombre del archivo PDF.

    Proceso de limpieza:
    1. Elimina prefijos comunes ('ficha-tecnica-', 'ft-', etc.)
    2. Elimina sufijos de version/fecha ('-v2-26', '-my2025', '_202511')
    3. Elimina sufijos de procesamiento ('compressed', 'copy', 'vf')
    4. Reemplaza separadores por espacios y capitaliza

    Ejemplo: 'ficha-tecnica-mazda-cx-5-v2-26.pdf' → 'Mazda Cx 5'

    Args:
        ruta_pdf: Ruta al archivo PDF.

    Returns:
        Nombre de modelo limpio y legible.
    """
    nombre = ruta_pdf.stem.lower()

    # Eliminar prefijos comunes de fichas tecnicas
    for prefijo in [
        "ficha-tecnica-", "ficha_tecnica_", "fichatecnica_",
        "ftlandcruiser-", "ft-", "ft_", "ficha-",
    ]:
        if nombre.startswith(prefijo):
            nombre = nombre[len(prefijo):]
            break

    # Eliminar sufijos de version y fecha: -v2-26, -v-09-25, -my2025, _202511
    nombre = re.sub(r"[-_](v[-_]?\d+[-_]?\d*|my\d+|20\d{2,4}.*)", "", nombre)

    # Eliminar sufijos de procesamiento
    nombre = re.sub(r"[-_](compressed|copy\d*|vf|pa|ipm\d+)$", "", nombre)

    # Reemplazar separadores por espacios y capitalizar cada palabra
    nombre = nombre.replace("-", " ").replace("_", " ")
    nombre = " ".join(word.capitalize() for word in nombre.split())

    return nombre if nombre else ruta_pdf.stem


def _limpiar_texto(texto: str) -> str:
    """Limpia texto extraido de PDF eliminando ruido tipico.

    - Colapsa espacios multiples (excepto newlines) a un solo espacio
    - Reduce 3+ newlines consecutivos a 2
    - Elimina lineas que solo contienen guiones, puntos o underscores

    Args:
        texto: Texto crudo extraido del PDF.

    Returns:
        Texto limpio y normalizado.
    """
    texto = re.sub(r"[^\S\n]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"^[\s\-_\.]{3,}$", "", texto, flags=re.MULTILINE)
    return texto.strip()


def _extraer_paginas_pdf(ruta_pdf: Path) -> List[Document]:
    """Extrae texto de cada pagina del PDF con metadata enriquecida.

    Estrategia dual:
    - Paginas con texto nativo (>= 50 chars): pdfplumber directo
    - Paginas escaneadas (< 50 chars): OCR con EasyOCR a 200 DPI

    Cada pagina se convierte en un Document con metadata:
    source (nombre archivo), page, marca (carpeta padre), modelo (inferido),
    doc_id (normalizado), ocr (bool).

    Args:
        ruta_pdf: Ruta al archivo PDF a procesar.

    Returns:
        Lista de Documents (uno por pagina con texto) con metadata.
    """
    documentos: List[Document] = []

    try:
        with pdfplumber.open(ruta_pdf) as pdf:
            for i, pagina in enumerate(pdf.pages):
                texto_nativo = pagina.extract_text() or ""

                if len(texto_nativo.strip()) >= MIN_CHARS_PARA_TEXTO:
                    texto = texto_nativo
                    uso_ocr = False
                else:
                    logger.info(f"  OCR página {i + 1}: {ruta_pdf.name}")
                    imagen: Image.Image = pagina.to_image(resolution=200).original
                    lector = _obtener_lector_ocr()
                    resultados = lector.readtext(np.array(imagen), detail=0, paragraph=True)
                    texto = "\n".join(resultados)
                    uso_ocr = True

                texto_limpio = _limpiar_texto(texto)
                if texto_limpio:
                    documentos.append(
                        Document(
                            page_content=texto_limpio,
                            metadata={
                                "source": ruta_pdf.name,
                                "page":   i + 1,
                                "marca":  ruta_pdf.parent.name,
                                "modelo": _inferir_modelo(ruta_pdf),
                                "doc_id": _doc_id_desde_pdf(ruta_pdf),
                                "ocr":    uso_ocr,
                            },
                        )
                    )
    except Exception as e:
        logger.error(f"Error procesando {ruta_pdf.name}: {e}")

    return documentos


def get_vector_store() -> Chroma:
    """Retorna instancia de ChromaDB con embeddings HuggingFace.

    Usa el modelo all-MiniLM-L6-v2 para generar embeddings (384 dimensiones).
    La coleccion se persiste en disco en PERSIST_DIR (default: ./chroma_db).

    Returns:
        Instancia de Chroma lista para similarity_search y add_documents.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def load_files(data_dir: str) -> List[Document]:
    """Carga documentos PDF y TXT desde un directorio organizado por marca.

    Estructura esperada:
        data_dir/
        ├── Toyota/
        │   └── ficha-tecnica-hilux.pdf
        ├── Mazda/
        │   └── ficha-tecnica-cx-5.pdf
        └── ...

    - PDFs: extrae texto pagina por pagina (nativo o OCR)
    - TXT: carga con TextLoader y enriquece metadata
    La marca se infiere del nombre de la carpeta padre.

    Args:
        data_dir: Ruta al directorio raiz de datos.

    Returns:
        Lista de Documents (uno por pagina) con metadata enriquecida.

    Raises:
        ValueError: Si el directorio no existe.
    """
    p = Path(data_dir)
    if not p.exists():
        raise ValueError(f"data_dir does not exist: {data_dir}")

    docs: List[Document] = []

    # Procesar PDFs recursivamente
    pdfs = sorted(p.rglob("*.pdf"))
    logger.info(f"PDFs encontrados: {len(pdfs)} en {p.resolve()}")
    for fp in pdfs:
        logger.info(f"Procesando: {fp.name}")
        docs.extend(_extraer_paginas_pdf(fp))

    # Procesar TXTs recursivamente
    for fp in p.rglob("*.txt"):
        txt_docs = TextLoader(str(fp), encoding="utf-8").load()
        for d in txt_docs:
            meta = d.metadata or {}
            meta["source"] = meta.get("source") or fp.name
            meta["page"] = meta.get("page") or 1
            meta["marca"] = meta.get("marca") or fp.parent.name
            meta["modelo"] = meta.get("modelo") or _inferir_modelo(fp)
            meta["doc_id"] = meta.get("doc_id") or _doc_id_desde_path(fp)
            d.metadata = meta
        docs.extend(txt_docs)

    return docs


def ingest(data_dir: str) -> dict:
    """Pipeline completo de ingestion: carga → split → enriquecer → embeddings → ChromaDB.

    Proceso:
    1. Carga documentos con load_files() (PDF nativo/OCR + TXT)
    2. Divide en chunks con RecursiveCharacterTextSplitter (1000 chars, 150 overlap)
    3. Asigna chunk_id unico por chunk: {doc_id}_p{page}_c{indice}
    4. Genera embeddings y almacena en ChromaDB

    Args:
        data_dir: Ruta al directorio con PDFs organizados por marca.

    Returns:
        Dict con estadisticas: files_dir, raw_docs, chunks, ids_added.

    Raises:
        ValueError: Si no se encuentran documentos o se producen 0 chunks.
    """
    raw_docs = load_files(data_dir)
    if not raw_docs:
        raise ValueError(
            f"No documents loaded from data_dir={data_dir}. "
            "Put .pdf (or .txt) files there."
        )

    # Dividir documentos en chunks solapados
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(raw_docs)
    if not splits:
        raise ValueError("Loaded documents but produced 0 chunks. Are files empty?")

    # Asignar chunk_id unico por documento: {doc_id}_p{page}_c{indice}
    chunk_counter: dict[str, int] = {}
    for chunk in splits:
        meta = chunk.metadata or {}
        source = meta.get("source", "unknown")
        doc_id = meta.get("doc_id") or _doc_id_desde_path(Path(str(source)))
        page = meta.get("page", 1)
        idx = chunk_counter.get(doc_id, 0)

        meta["doc_id"] = doc_id
        meta["page"] = page
        meta["chunk_index"] = idx
        meta["chunk_id"] = f"{doc_id}_p{page}_c{idx}"
        chunk.metadata = meta

        chunk_counter[doc_id] = idx + 1

    # Generar embeddings y almacenar en ChromaDB
    vs = get_vector_store()
    ids = vs.add_documents(documents=splits)
    return {"files_dir": data_dir, "raw_docs": len(raw_docs), "chunks": len(splits), "ids_added": len(ids)}
