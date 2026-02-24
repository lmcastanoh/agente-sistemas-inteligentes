# backend/tools.py
from __future__ import annotations

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from rag_store import get_vector_store


def _get_llm():
    return ChatOpenAI(model="gpt-5-nano", temperature=0)


@tool
def listar_modelos_disponibles(marca: str = "") -> str:
    """
    Retorna el catálogo de modelos indexados en la base de conocimiento.
    Si se indica una marca, filtra solo los modelos de esa marca.
    Usa esta tool cuando el usuario pregunte qué modelos o vehículos están disponibles.

    Args:
        marca: Nombre de la marca a filtrar (ej: 'Toyota', 'Mazda'). Opcional.
    """
    vs = get_vector_store()

    where = {"marca": marca} if marca else None
    result = vs._collection.get(where=where, include=["metadatas"])

    modelos_por_marca: dict[str, set[str]] = {}
    for meta in result["metadatas"]:
        m = meta.get("marca", "Desconocida")
        mod = meta.get("modelo", "Desconocido")
        modelos_por_marca.setdefault(m, set()).add(mod)

    if not modelos_por_marca:
        return "No se encontraron modelos en el catálogo."

    lineas = []
    for m in sorted(modelos_por_marca):
        for mod in sorted(modelos_por_marca[m]):
            lineas.append(f"- {m}: {mod}")

    return "Modelos disponibles:\n" + "\n".join(lineas)


@tool
def buscar_especificacion(especificacion: str, modelo: str) -> str:
    """
    Busca un dato técnico puntual (potencia, torque, autonomía, consumo, dimensiones, etc.)
    para un modelo específico.
    Usa esta tool cuando el usuario pregunte por una característica técnica concreta de un vehículo.

    Args:
        especificacion: El dato técnico buscado (ej: 'potencia', 'torque', 'autonomía').
        modelo: El nombre del modelo del vehículo (ej: 'Hilux', 'CX-5').
    """
    vs = get_vector_store()

    results = vs.similarity_search(
        f"{especificacion} {modelo}",
        k=6,
    )

    if not results:
        return f"No se encontró información sobre '{especificacion}' para el modelo '{modelo}'."

    fragmentos = "\n---\n".join(
        f"[{d.metadata.get('source', '')} p.{d.metadata.get('page', '')}]\n{d.page_content}"
        for d in results
    )
    return fragmentos


@tool
def buscar_por_marca(marca: str) -> str:
    """
    Recupera información general de todos los modelos de una marca específica.
    Usa esta tool cuando el usuario pregunte por una marca en general o quiera
    comparar modelos de la misma marca.

    Args:
        marca: Nombre de la marca (ej: 'Toyota', 'Volkswagen', 'Mazda').
    """
    vs = get_vector_store()

    results = vs.similarity_search(
        marca,
        k=10,
        filter={"marca": marca},
    )

    if not results:
        return f"No se encontró información para la marca '{marca}'."

    fragmentos = "\n---\n".join(
        f"[{d.metadata.get('modelo', '')} p.{d.metadata.get('page', '')}]\n{d.page_content}"
        for d in results
    )
    return f"Información de {marca}:\n\n{fragmentos}"


@tool
def comparar_modelos(modelo1: str, modelo2: str) -> str:
    """
    Extrae especificaciones de dos modelos y genera una tabla comparativa en markdown.
    Usa esta tool cuando el usuario quiera comparar dos vehículos entre sí.

    Args:
        modelo1: Nombre del primer modelo (ej: 'Hilux', 'Corolla Cross').
        modelo2: Nombre del segundo modelo (ej: 'Fortuner', 'Yaris Cross').
    """
    vs = get_vector_store()

    def _buscar(modelo: str):
        docs = vs.similarity_search(modelo, k=8)
        return "\n\n".join(d.page_content for d in docs)

    ctx1 = _buscar(modelo1)
    ctx2 = _buscar(modelo2)

    if not ctx1 and not ctx2:
        return f"No se encontró información para '{modelo1}' ni '{modelo2}'."

    prompt = (
        f"Eres un experto en fichas técnicas de vehículos. "
        f"Con base ÚNICAMENTE en la información proporcionada, genera una tabla comparativa "
        f"en markdown entre **{modelo1}** y **{modelo2}**.\n\n"
        f"Incluye filas para: Motor, Potencia, Torque, Transmisión, Tracción, "
        f"Dimensiones, Peso, Consumo, y cualquier otro dato relevante disponible.\n"
        f"Si un dato no está en la información, escribe N/D.\n\n"
        f"### Información de {modelo1}:\n{ctx1}\n\n"
        f"### Información de {modelo2}:\n{ctx2}"
    )

    response = _get_llm().invoke(prompt)
    return str(response.content)


@tool
def resumir_ficha(modelo: str) -> str:
    """
    Genera un resumen estructurado en markdown de la ficha técnica de un modelo.
    Usa esta tool cuando el usuario pida un resumen, overview o descripción general de un vehículo.

    Args:
        modelo: Nombre del modelo (ej: 'Prado', 'BZ4X', 'Mazda Cx 5 2026').
    """
    vs = get_vector_store()

    docs = vs.similarity_search(modelo, k=10)

    if not docs:
        return f"No se encontró información para el modelo '{modelo}'."

    ctx = "\n\n".join(d.page_content for d in docs)

    prompt = (
        f"Eres un experto en fichas técnicas de vehículos. "
        f"Con base ÚNICAMENTE en la siguiente información, genera un resumen estructurado "
        f"en markdown del **{modelo}**.\n\n"
        f"Organiza el resumen en estas secciones (omite las que no tengan datos):\n"
        f"- Motor y Transmisión\n"
        f"- Rendimiento y Consumo\n"
        f"- Dimensiones y Capacidades\n"
        f"- Equipamiento destacado\n"
        f"- Versiones disponibles\n\n"
        f"### Información disponible:\n{ctx}"
    )

    response = _get_llm().invoke(prompt)
    return str(response.content)
