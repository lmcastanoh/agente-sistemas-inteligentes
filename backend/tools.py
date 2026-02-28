# backend/tools.py
# ==============================================================================
# Tools de LangGraph para el sistema RAG de fichas tecnicas vehiculares.
#
# Estas tools son invocadas por el LLM en el nodo call_tools cuando el flujo
# pasa por Comparacion o Resumen. El ToolNode de LangGraph las ejecuta
# automaticamente a partir de los tool_calls generados por el LLM.
#
# Tools disponibles:
#   - listar_modelos_disponibles: catalogo de modelos indexados
#   - buscar_especificacion:      dato tecnico puntual de un modelo
#   - buscar_por_marca:           todos los modelos de una marca
#   - comparar_modelos:           tabla comparativa entre 2 modelos
#   - resumir_ficha:              resumen estructurado de un modelo
# ==============================================================================
from __future__ import annotations

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from rag_store import get_vector_store


def _get_llm():
    """Retorna instancia de LLM para generacion dentro de tools (comparar, resumir).

    Usa gpt-5-nano con temperature=0 para respuestas consistentes y deterministas.
    """
    return ChatOpenAI(model="gpt-5-nano", temperature=0)


@tool
def listar_modelos_disponibles(marca: str = "") -> str:
    """Retorna el catalogo de modelos indexados en la base de conocimiento.

    Consulta directamente la coleccion ChromaDB (sin similarity search)
    para listar todos los modelos unicos agrupados por marca.
    Si se indica una marca, filtra solo los modelos de esa marca.

    Usada cuando el usuario pregunta que modelos o vehiculos estan disponibles.

    Args:
        marca: Nombre de la marca a filtrar (ej: 'Toyota', 'Mazda'). Opcional.

    Returns:
        Lista formateada de modelos por marca, o mensaje si no hay resultados.
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
    """Busca un dato tecnico puntual para un modelo especifico.

    Realiza similarity search combinando la especificacion y el modelo
    para encontrar los chunks mas relevantes (k=6).

    Usada cuando el usuario pregunta por una caracteristica tecnica concreta
    como potencia, torque, autonomia, consumo o dimensiones.

    Args:
        especificacion: El dato tecnico buscado (ej: 'potencia', 'torque', 'autonomia').
        modelo:         El nombre del modelo del vehiculo (ej: 'Hilux', 'CX-5').

    Returns:
        Fragmentos de contexto con metadata [source, pagina], o mensaje si no hay datos.
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
    """Recupera informacion general de todos los modelos de una marca especifica.

    Realiza similarity search con filtro de metadata por marca (k=10).
    Util para preguntas sobre el catalogo completo de una marca.

    Usada cuando el usuario pregunta por una marca en general o quiere
    explorar modelos de una misma marca.

    Args:
        marca: Nombre de la marca (ej: 'Toyota', 'Volkswagen', 'Mazda').

    Returns:
        Fragmentos de contexto con metadata [modelo, pagina], o mensaje si no hay datos.
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
    """Genera una tabla comparativa en markdown entre dos modelos.

    Proceso:
    1. Busca chunks de cada modelo por separado (k=8 cada uno)
    2. Envia ambos contextos al LLM con instrucciones de formato
    3. El LLM genera tabla markdown solo con datos reales disponibles
    4. Si faltan demasiados datos, genera bullets explicativos en vez de tabla

    Usada cuando el usuario quiere comparar dos vehiculos entre si.

    Args:
        modelo1: Nombre del primer modelo (ej: 'Hilux', 'Corolla Cross').
        modelo2: Nombre del segundo modelo (ej: 'Fortuner', 'Yaris Cross').

    Returns:
        Tabla comparativa markdown o explicacion de datos faltantes.
    """
    vs = get_vector_store()

    def _buscar(modelo: str):
        """Busca chunks relevantes para un modelo especifico."""
        docs = vs.similarity_search(modelo, k=8)
        return "\n\n".join(d.page_content for d in docs)

    ctx1 = _buscar(modelo1)
    ctx2 = _buscar(modelo2)

    if not ctx1 and not ctx2:
        return f"No se encontró información para '{modelo1}' ni '{modelo2}'."
    prompt = f"""Eres un experto en fichas tecnicas de vehiculos.
Con base UNICAMENTE en la informacion proporcionada, genera una comparativa clara
en markdown entre **{modelo1}** y **{modelo2}**.

Reglas de formato:
- Si hay datos comparables, usa una tabla markdown limpia.
- Incluye solo filas con informacion real para al menos uno de los modelos.
- No llenes la tabla con N/D masivo.
- Si faltan demasiados datos para comparar, NO hagas tabla: responde en 2-4 bullets
  explicando que no hay informacion suficiente y que datos faltan.
- Cierra con una recomendacion corta (1-2 lineas) solo si hay sustento en los datos.

### Informacion de {modelo1}:
{ctx1}

### Informacion de {modelo2}:
{ctx2}
"""

    response = _get_llm().invoke(prompt)
    return str(response.content)


@tool
def resumir_ficha(modelo: str) -> str:
    """Genera un resumen estructurado en markdown de la ficha tecnica de un modelo.

    Proceso:
    1. Busca chunks del modelo (k=10 para cobertura amplia)
    2. Envia contexto al LLM con instrucciones de formato por secciones
    3. El LLM organiza en: Motor, Rendimiento, Dimensiones, Equipamiento, Versiones
    4. Si hay pocos datos, agrega seccion 'Datos faltantes' con bullets

    Usada cuando el usuario pide un resumen, overview o descripcion general.

    Args:
        modelo: Nombre del modelo (ej: 'Prado', 'BZ4X', 'Mazda Cx 5 2026').

    Returns:
        Resumen estructurado en markdown o mensaje si no hay datos.
    """
    vs = get_vector_store()

    docs = vs.similarity_search(modelo, k=10)

    if not docs:
        return f"No se encontró información para el modelo '{modelo}'."

    ctx = "\n\n".join(d.page_content for d in docs)
    prompt = f"""Eres un experto en fichas tecnicas de vehiculos.
Con base UNICAMENTE en la siguiente informacion, genera un resumen estructurado
en markdown del **{modelo}**.

Reglas de formato:
- Usa titulo y secciones cortas, faciles de leer.
- Organiza en estas secciones y omite las que no tengan datos:
  - Motor y transmision
  - Rendimiento y consumo
  - Dimensiones y capacidades
  - Equipamiento destacado
  - Versiones disponibles
- No repitas frases de disculpa.
- Si hay pocos datos, usa una seccion final 'Datos faltantes' con bullets.
- Evita tablas enormes con N/D.

### Informacion disponible:
{ctx}
"""

    response = _get_llm().invoke(prompt)
    return str(response.content)
