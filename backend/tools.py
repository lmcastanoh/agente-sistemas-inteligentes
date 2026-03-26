# backend/tools.py
# ==============================================================================
# Tools de LangGraph para el sistema RAG agéntico de fichas técnicas vehiculares.
#
# Estas tools son invocadas por el agente ReAct (nodo agent_reason) de forma
# autónoma. El agente decide cuándo y qué herramientas usar en su ciclo de
# razonamiento.
#
# Tools disponibles:
#   - listar_modelos_disponibles: catálogo de modelos indexados
#   - buscar_especificacion:      dato técnico puntual de un modelo
#   - buscar_por_marca:           todos los modelos de una marca
#   - comparar_modelos:           tabla comparativa entre 2 modelos
#   - resumir_ficha:              resumen estructurado de un modelo
#   - refinar_busqueda:           búsqueda adicional con query/filtros diferentes
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


@tool
def refinar_busqueda(query: str, k: int = 6, marca: str = "", modelo: str = "") -> str:
    """Realiza una búsqueda adicional en la base de conocimiento con query y filtros diferentes.

    Útil cuando el contexto inicial no responde la pregunta o falta información.
    Permite reformular la consulta o aplicar filtros distintos para obtener
    chunks más relevantes.

    Args:
        query:  Nueva consulta reformulada para búsqueda semántica.
        k:      Número de chunks a recuperar (default 6, max 15).
        marca:  Filtro opcional por marca (ej: 'Toyota', 'Mazda').
        modelo: Filtro opcional por modelo (ej: 'Hilux', 'CX-5').

    Returns:
        Fragmentos de contexto con metadata [doc_id, página], o mensaje si no hay resultados.
    """
    vs = get_vector_store()
    k = min(max(k, 1), 15)

    where_filter = None
    if modelo:
        no_hyphen = modelo.replace("-", " ")
        bases = {modelo, modelo.title(), no_hyphen, no_hyphen.title()}
        variants = list(bases)
        if marca:
            for b in list(bases):
                variants.append(f"{marca} {b}")
        where_filter = {"modelo": {"$in": variants}}
    elif marca:
        where_filter = {"marca": marca}

    if where_filter:
        results = vs.similarity_search(query, k=k, filter=where_filter)
        if not results:
            results = vs.similarity_search(query, k=k)
    else:
        results = vs.similarity_search(query, k=k)

    if not results:
        return f"No se encontraron resultados para: '{query}'."

    fragmentos = "\n---\n".join(
        f"[doc_id={d.metadata.get('doc_id', d.metadata.get('source', ''))}; "
        f"página={d.metadata.get('page', '')}]\n{d.page_content}"
        for d in results
    )
    return f"Resultados ({len(results)} chunks):\n\n{fragmentos}"


@tool
def corregir_respuesta(respuesta: str, instrucciones: str) -> str:
    """Corrige una respuesta aplicando instrucciones específicas.

    Útil para fixes menores: reformatear citas, quitar datos no respaldados,
    mejorar estructura markdown.

    Args:
        respuesta:     La respuesta original a corregir.
        instrucciones: Qué corregir (ej: 'mover citas al final como fuentes').

    Returns:
        La respuesta corregida.
    """
    llm = _get_llm()
    prompt = (
        "Corrige esta respuesta siguiendo las instrucciones.\n"
        "Devuelve SOLO la respuesta corregida, sin explicaciones.\n\n"
        f"Instrucciones: {instrucciones}\n\n"
        f"Respuesta original:\n{respuesta}"
    )
    result = llm.invoke(prompt)
    return str(result.content)


@tool
def regenerar_respuesta(pregunta: str, contexto: str) -> str:
    """Regenera una respuesta completa a partir de contexto nuevo o mejorado.

    Útil cuando la respuesta original tenía problemas graves (modelo equivocado,
    datos inventados) y se obtuvo mejor contexto con refinar_busqueda.

    Args:
        pregunta: La pregunta original del usuario.
        contexto: El contexto actualizado/mejorado para generar la respuesta.

    Returns:
        Nueva respuesta generada desde el contexto proporcionado.
    """
    llm = _get_llm()
    prompt = (
        "Genera una respuesta clara y estructurada en markdown basada\n"
        "ÚNICAMENTE en el contexto proporcionado. Incluye referencias numeradas [1], [2]\n"
        "y una sección **Fuentes:** al final con viñetas.\n\n"
        "Si el contexto no contiene información del modelo pedido, di explícitamente\n"
        "que no se encontró información.\n\n"
        f"Pregunta: {pregunta}\n\n"
        f"Contexto:\n{contexto}"
    )
    result = llm.invoke(prompt)
    return str(result.content)
