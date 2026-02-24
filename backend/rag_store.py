# backend/rag_store.py
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

PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_collection")

# Mínimo de caracteres para considerar que una página tiene texto nativo
MIN_CHARS_PARA_TEXTO = 50

# Lector OCR compartido — se inicializa una sola vez
_lector_ocr: easyocr.Reader | None = None


def _obtener_lector_ocr() -> easyocr.Reader:
    global _lector_ocr
    if _lector_ocr is None:
        logger.info("Inicializando EasyOCR (primera vez descarga modelos ~100 MB)...")
        _lector_ocr = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
        logger.info("EasyOCR listo.")
    return _lector_ocr


def _inferir_modelo(ruta_pdf: Path) -> str:
    """Deriva un nombre de modelo legible a partir del nombre del archivo PDF."""
    nombre = ruta_pdf.stem.lower()

    # Eliminar prefijos comunes de fichas técnicas
    for prefijo in [
        "ficha-tecnica-", "ficha_tecnica_", "fichatecnica_",
        "ftlandcruiser-", "ft-", "ft_", "ficha-",
    ]:
        if nombre.startswith(prefijo):
            nombre = nombre[len(prefijo):]
            break

    # Eliminar sufijos de versión y fecha: -v2-26, -v-09-25, -my2025, _202511
    nombre = re.sub(r"[-_](v[-_]?\d+[-_]?\d*|my\d+|20\d{2,4}.*)", "", nombre)

    # Eliminar sufijos de procesamiento
    nombre = re.sub(r"[-_](compressed|copy\d*|vf|pa|ipm\d+)$", "", nombre)

    # Reemplazar separadores por espacios y capitalizar cada palabra
    nombre = nombre.replace("-", " ").replace("_", " ")
    nombre = " ".join(word.capitalize() for word in nombre.split())

    return nombre if nombre else ruta_pdf.stem


def _limpiar_texto(texto: str) -> str:
    texto = re.sub(r"[^\S\n]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"^[\s\-_\.]{3,}$", "", texto, flags=re.MULTILINE)
    return texto.strip()


def _extraer_paginas_pdf(ruta_pdf: Path) -> List[Document]:
    """
    Extrae texto de cada página del PDF.
    - Páginas con texto nativo: pdfplumber directo.
    - Páginas escaneadas (imagen): EasyOCR.
    Retorna una lista de Documents con metadatos source + page.
    """
    documentos: List[Document] = []

    try:
        with pdfplumber.open(ruta_pdf) as pdf:
            for i, pagina in enumerate(pdf.pages):
                texto_nativo = pagina.extract_text() or ""

                if len(texto_nativo.strip()) >= MIN_CHARS_PARA_TEXTO:
                    texto = texto_nativo
                else:
                    logger.info(f"  OCR página {i + 1}: {ruta_pdf.name}")
                    imagen: Image.Image = pagina.to_image(resolution=200).original
                    lector = _obtener_lector_ocr()
                    resultados = lector.readtext(np.array(imagen), detail=0, paragraph=True)
                    texto = "\n".join(resultados)

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
                            },
                        )
                    )
    except Exception as e:
        logger.error(f"Error procesando {ruta_pdf.name}: {e}")

    return documentos

"""def get_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
"""

def get_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

def load_files(data_dir: str) -> List[Document]:
    p = Path(data_dir)
    if not p.exists():
        raise ValueError(f"data_dir does not exist: {data_dir}")

    docs: List[Document] = []

    pdfs = sorted(p.rglob("*.pdf"))
    logger.info(f"PDFs encontrados: {len(pdfs)} en {p.resolve()}")
    for fp in pdfs:
        logger.info(f"Procesando: {fp.name}")
        docs.extend(_extraer_paginas_pdf(fp))

    for fp in p.rglob("*.txt"):
        docs.extend(TextLoader(str(fp), encoding="utf-8").load())

    return docs

def ingest(data_dir: str) -> dict:
    raw_docs = load_files(data_dir)
    if not raw_docs:
        raise ValueError(
            f"No documents loaded from data_dir={data_dir}. "
            "Put .pdf (or .txt) files there."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(raw_docs)
    if not splits:
        raise ValueError("Loaded documents but produced 0 chunks. Are files empty?")

    vs = get_vector_store()
    ids = vs.add_documents(documents=splits)
    return {"files_dir": data_dir, "raw_docs": len(raw_docs), "chunks": len(splits), "ids_added": len(ids)}