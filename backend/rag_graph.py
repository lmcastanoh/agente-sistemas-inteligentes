# backend/rag_graph.py
# ==============================================================================
# Grafo LangGraph principal del sistema RAG para fichas tecnicas vehiculares.
#
# Flujo: classify_intent → retrieve → decide_tools → [call_tools → tools] →
#        generate_grounded → evaluate_grounding → [retry | END]
#
# 4 rutas posibles:
#   A) GENERAL:     classify → answer_general → END (sin retrieval)
#   B) Busqueda:    classify → retrieve → decide_tools → generate → evaluate → END
#   C) Resumen:     classify → retrieve → decide_tools → call_tools → tools → generate → evaluate → END
#   D) Comparacion: classify → retrieve → decide_tools → call_tools → tools → generate → evaluate → END
#
# Features:
#   - Dynamic k: el clasificador sugiere cuantos chunks recuperar
#   - Regeneration loop: el critico puede rechazar y forzar un reintento (max 1)
#   - Memory: last_model/last_make persisten entre turnos con reducer _keep_latest
#   - Keyword fallback: regex corrige clasificaciones erroneas del LLM
#   - Trazabilidad: cada nodo registra su ruta, decisiones, chunks y prompt
# ==============================================================================
from __future__ import annotations

import json
import re
from typing import Annotated, Any, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFIER_USER_TEMPLATE,
    GROUNDED_GENERATION_SYSTEM_PROMPT,
    GROUNDED_GENERATION_USER_TEMPLATE,
    GROUNDING_CRITIC_SYSTEM_PROMPT,
    GROUNDING_CRITIC_USER_TEMPLATE,
)
from rag_store import get_vector_store
from schemas import GroundingEvaluation, IntentClassification, eval_to_dict, intent_to_dict
from tools import (
    buscar_especificacion,
    buscar_por_marca,
    comparar_modelos,
    listar_modelos_disponibles,
    resumir_ficha,
)

# ── Mapa fijo de k por intencion (fallback cuando suggested_k no esta disponible) ──
K_POR_INTENCION = {
    "comparación": 10,
    "resumen": 8,
    "búsqueda": 8,
    "general": 0,
}


def _k_desde_intent(intent: str) -> int:
    """Obtiene el numero de chunks (k) a recuperar segun la intencion clasificada.

    Mapea los nombres de intent (con mayuscula) a las claves del mapa fijo.
    Si el intent no se reconoce, usa 'busqueda' como fallback (k=4).

    Args:
        intent: Nombre del intent (ej: 'Busqueda', 'Resumen', 'Comparacion', 'GENERAL')

    Returns:
        Numero de chunks a recuperar de ChromaDB.
    """
    mapa = {
        "Búsqueda": "búsqueda",
        "Resumen": "resumen",
        "Comparación": "comparación",
        "GENERAL": "general",
    }
    return K_POR_INTENCION.get(mapa.get(intent, "búsqueda"), 4)


# ── Regex para keyword fallback ──────────────────────────────────────────────
# Detectan intents por palabras clave cuando el clasificador LLM falla.
# Se usan tanto en retrieve (para corregir k) como en decide_tools (para activar tools).
_RE_RESUMEN = re.compile(
    r"\b(resumen|resume[mn]?|resúm[ea]|overview|ficha\s+completa|descripción\s+general)\b",
    re.IGNORECASE,
)
_RE_COMPARACION = re.compile(
    r"\b(compar[aáeo]|versus|vs\.?|diferencias?\s+entre)\b",
    re.IGNORECASE,
)


def _keyword_intent_override(question: str) -> str | None:
    """Detecta intent por keywords cuando el clasificador LLM falla.

    Segunda red de seguridad: si el LLM clasifica mal (ej: "compara X vs Y"
    como Busqueda), esta funcion corrige usando regex.
    Comparacion tiene prioridad sobre Resumen.

    Args:
        question: Pregunta original del usuario.

    Returns:
        'Comparacion' o 'Resumen' si se detecta keyword, None si no.
    """
    if _RE_COMPARACION.search(question):
        return "Comparación"
    if _RE_RESUMEN.search(question):
        return "Resumen"
    return None


def _build_retrieval_filter(entities: dict[str, Any] | None) -> dict | None:
    """Construye filtro de metadata para ChromaDB a partir de las entidades del clasificador.

    Prioriza modelo sobre marca. Si hay modelo, genera variantes normalizadas
    y filtra con $in. Si solo hay marca, filtra por marca exacta.

    Args:
        entities: Dict con 'model', 'make', 'year', 'trim' del clasificador.

    Returns:
        Filtro compatible con ChromaDB (dict) o None si no hay entidades utiles.
    """
    if not entities:
        return None
    model = entities.get("model")
    make = entities.get("make")
    if model and len(model) >= 2:
        return {"modelo": {"$in": _model_variants(model, make)}}
    if make and len(make) >= 2:
        return {"marca": make}
    return None


def _model_variants(model: str, make: str | None = None) -> list[str]:
    """Genera variantes normalizadas de un nombre de modelo para matching flexible.

    Cubre variaciones comunes: con/sin guion, con/sin marca prefijada,
    Title case vs original. Ejemplo: 'cx-5' → {'cx-5', 'Cx-5', 'cx 5', 'Cx 5',
    'Mazda cx-5', 'Mazda Cx-5', ...}

    Args:
        model: Nombre del modelo (ej: 'cx-5', 'Hilux')
        make:  Marca opcional para generar variantes prefijadas (ej: 'Mazda')

    Returns:
        Lista de variantes del nombre del modelo.
    """
    no_hyphen = model.replace("-", " ")
    bases = {model, model.title(), no_hyphen, no_hyphen.title()}
    variants: set[str] = set(bases)
    if make:
        for b in bases:
            variants.add(f"{make} {b}")
    return list(variants)


# Regex para extraer los dos modelos de una query de comparacion.
# Ejemplo: "compara el Toyota Hilux vs Toyota Fortuner" → ['Toyota Hilux', 'Toyota Fortuner']
_RE_VS = re.compile(
    r"(?:compar[aáeo]\w*|diferencias?\s+entre)\s+(?:(?:el|la|los|las|del|al)\s+)?"
    r"(.+?)\s+(?:vs\.?|versus|contra|y|con)\s+(?:(?:el|la|los|las|del|al)\s+)?(.+)",
    re.IGNORECASE,
)


def _extract_comparison_models(question: str) -> list[str] | None:
    """Extrae los dos modelos de una query de comparacion usando regex.

    Busca patrones como "compara X vs Y", "diferencias entre X y Y".
    Usado para retrieval balanceado (k/2 por modelo) y para actualizar
    last_model con el ultimo modelo mencionado.

    Args:
        question: Pregunta original del usuario.

    Returns:
        Lista con 2 nombres de modelos, o None si no es comparacion.
    """
    m = _RE_VS.search(question)
    if not m:
        return None
    return [m.group(1).strip(), m.group(2).strip()]


def _extract_tool_results(messages: List[BaseMessage]) -> str:
    """Extrae contenido de ToolMessages del historial de mensajes.

    Despues de que ToolNode ejecuta las tools, los resultados quedan como
    ToolMessage en el estado. Esta funcion los concatena para pasarlos
    como contexto adicional a generate_grounded.

    Args:
        messages: Lista de mensajes del estado del grafo.

    Returns:
        Texto concatenado de todos los ToolMessages, separados por '---'.
    """
    parts: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.content:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if content.strip():
                parts.append(content.strip())
    return "\n\n---\n\n".join(parts)


# ── Configuracion del regeneration loop ──────────────────────────────────────
# Numero maximo de reintentos cuando el critico rechaza la respuesta (score < 0.5).
# En cada reintento, el feedback del critico se inyecta como instruccion de correccion.
MAX_RETRIES = 1


def _keep_latest(existing: Optional[str], new: Optional[str]) -> Optional[str]:
    """Reducer para last_model y last_make en RAGState.

    Conserva el ultimo valor no-None entre turnos conversacionales.
    Esto permite follow-ups como: "potencia del Hilux?" → "y cuanto pesa?"
    donde el segundo turno hereda 'Hilux' automaticamente.

    Args:
        existing: Valor actual en el estado (del turno anterior).
        new:      Valor nuevo del turno actual (puede ser None).

    Returns:
        El valor nuevo si no es None, sino el existente.
    """
    return new if new is not None else existing


# ── Estado del grafo ─────────────────────────────────────────────────────────
class RAGState(TypedDict):
    """Estado compartido entre todos los nodos del grafo LangGraph.

    Campos:
        question:        Pregunta actual del usuario.
        docs:            Documentos recuperados de ChromaDB.
        answer:          Respuesta final generada.
        messages:        Historial de mensajes (con reducer add_messages de LangGraph).
        intent:          Resultado del clasificador (dict de IntentClassification).
        eval_result:     Resultado del critico (dict de GroundingEvaluation).
        usar_tools:      Flag: True si el flujo pasa por call_tools/tools.
        trazabilidad:    Dict acumulativo con ruta, decisiones, chunks, prompts.
        last_model:      Ultimo modelo mencionado (persiste entre turnos con _keep_latest).
        last_make:       Ultima marca mencionada (persiste entre turnos con _keep_latest).
        retry_count:     Contador de reintentos del regeneration loop (0 = primer intento).
        critic_feedback: Lista de issues del critico para inyectar en el reintento.
    """

    question: str
    docs: List[Document]
    answer: str
    messages: Annotated[List[BaseMessage], add_messages]
    intent: Optional[dict[str, Any]]
    eval_result: Optional[dict[str, Any]]
    usar_tools: bool
    trazabilidad: dict[str, Any]
    last_model: Annotated[Optional[str], _keep_latest]
    last_make: Annotated[Optional[str], _keep_latest]
    retry_count: int
    critic_feedback: Optional[list[str]]


# ── Funciones auxiliares ─────────────────────────────────────────────────────

def _history_text(messages: List[BaseMessage], max_items: int = 8) -> str:
    """Construye texto de historial conversacional para el clasificador y el rewriter.

    Filtra ToolMessages y AIMessages con tool_calls (son datos crudos internos
    que no aportan contexto conversacional y desplazarian las preguntas reales).
    Solo conserva mensajes Human y AI con contenido textual real.

    Args:
        messages:  Lista completa de mensajes del estado.
        max_items: Maximo de mensajes recientes a incluir (default: 8).

    Returns:
        Texto formateado como "Usuario: ...\nAsistente: ..." o string vacio.
    """
    if not messages:
        return ""
    conversational: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            continue
        conversational.append(m)
    items = conversational[-max_items:]
    lines: list[str] = []
    for m in items:
        role = "Asistente" if isinstance(m, AIMessage) else "Usuario"
        content = m.content if isinstance(m.content, str) else str(m.content)
        if content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines)


def _retrieved_chunk_payload(docs: List[Document]) -> list[dict[str, Any]]:
    """Genera payload resumido de los chunks recuperados para trazabilidad y el critico.

    Extrae metadata clave de cada documento sin incluir el contenido completo.
    Usa source como fallback para doc_id si este no esta en la metadata.

    Args:
        docs: Lista de Documents recuperados de ChromaDB.

    Returns:
        Lista de dicts con doc_id, source, page y opcionalmente chunk_id.
    """
    chunks: list[dict[str, Any]] = []
    for d in docs:
        md = d.metadata or {}
        chunk = {
            "doc_id": md.get("doc_id") or md.get("source"),
            "source": md.get("source"),
            "page": md.get("page"),
        }
        if md.get("chunk_id"):
            chunk["chunk_id"] = md["chunk_id"]
        chunks.append(chunk)
    return chunks


def _fix_doubled_text(text: str) -> str:
    """Corrige texto con caracteres duplicados por extraccion corrupta de PDF.

    Algunos PDFs escaneados producen texto donde cada caracter aparece dos veces
    consecutivas (ej: 'TTOOYYOOTTAA' → 'TOYOTA'). Detecta este patron analizando
    los primeros 60 caracteres y, si mas del 70% son pares repetidos, deduplica
    tomando un caracter de cada dos.

    Args:
        text: Texto extraido del PDF (posiblemente corrupto).

    Returns:
        Texto corregido (deduplicado) o el original si no estaba corrupto.
    """
    if len(text) < 10:
        return text
    sample = text[:60]
    doubles = len(re.findall(r"([A-Za-z0-9])\1", sample))
    alphanums = len(re.findall(r"[A-Za-z0-9]", sample))
    if alphanums > 4 and (doubles * 2) / alphanums > 0.7:
        return text[::2]
    return text


def _retrieval_context(docs: List[Document]) -> str:
    """Formatea los documentos recuperados como contexto para el LLM generador.

    Cada bloque incluye una cabecera con identificadores de trazabilidad
    que el LLM debe copiar como cita en su respuesta.
    Formato: [doc_id=<archivo>; pagina=<N>] o con chunk_id si disponible.

    Aplica _fix_doubled_text para corregir texto corrupto de PDFs escaneados.

    Args:
        docs: Lista de Documents recuperados de ChromaDB.

    Returns:
        Texto formateado con bloques separados por '---', cada uno con cabecera + contenido.
    """
    blocks: list[str] = []
    for d in docs:
        md = d.metadata or {}
        content = _fix_doubled_text(d.page_content)
        doc_id = md.get("doc_id") or md.get("source", "desconocido")
        page = md.get("page", "N/A")
        chunk_id = md.get("chunk_id")
        if chunk_id:
            header = f"[doc_id={doc_id}; página={page}; chunk_id={chunk_id}]"
        else:
            header = f"[doc_id={doc_id}; página={page}]"
        blocks.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(blocks)


# ── Construccion del grafo ───────────────────────────────────────────────────

def build_rag_graph():
    """Construye y compila el grafo LangGraph completo del sistema RAG.

    Inicializa:
    - Vector store (ChromaDB con HuggingFace embeddings)
    - 4 instancias de LLM (router, answer, critic, rewrite) todas gpt-5-nano
    - 5 tools de LangGraph (listar, buscar, comparar, resumir, buscar_por_marca)
    - 8 nodos del grafo con sus edges y conditional edges
    - MemorySaver como checkpointer para persistir estado entre turnos

    Returns:
        Grafo LangGraph compilado listo para .invoke() o .ainvoke()
    """
    vs = get_vector_store()
    tools = [
        listar_modelos_disponibles,
        buscar_especificacion,
        buscar_por_marca,
        comparar_modelos,
        resumir_ficha,
    ]
    tool_node = ToolNode(tools)

    # 4 LLMs especializados (todos gpt-5-nano con diferentes temperatures)
    router_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)      # Clasificador: determinista
    answer_llm = ChatOpenAI(model="gpt-5-nano", temperature=0.2)     # Generador: leve creatividad
    answer_llm_with_tools = answer_llm.bind_tools(tools)             # Generador con tools bindeadas
    critic_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)       # Critico: determinista
    rewrite_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)      # Rewriter de queries

    # ── Nodo 1: classify_intent ──────────────────────────────────────────
    def classify_intent(state: RAGState) -> dict[str, Any]:
        """Clasifica la intencion del usuario y determina la ruta del grafo.

        Proceso:
        1. Construye input con historial conversacional + pregunta actual
        2. LLM clasifica en Busqueda|Resumen|Comparacion|GENERAL con structured output
        3. Actualiza last_model/last_make para follow-ups futuros
        4. Fallback de memory: si no hay modelo, usa last_model del turno anterior
        5. Registra clasificacion y decision en trazabilidad

        Retorna: intent, last_model, last_make, trazabilidad
        """
        question = state["question"]
        history = _history_text(state.get("messages") or [])
        classifier_input = (
            f"Historial reciente:\n{history}\n\nConsulta actual:\n{question}"
            if history
            else question
        )
        structured = router_llm.with_structured_output(IntentClassification)
        result: IntentClassification = structured.invoke(
            [
                SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
                HumanMessage(content=CLASSIFIER_USER_TEMPLATE.format(question=classifier_input)),
            ]
        )
        intent_data = intent_to_dict(result)

        # Actualizar last_model/last_make cuando el clasificador identifica un modelo
        entities = intent_data.get("entities") or {}
        updates: dict[str, Any] = {"intent": intent_data}
        current_model = entities.get("model")
        current_make = entities.get("make")

        # En comparaciones, el clasificador puede poner "Hilux vs Fortuner" como model.
        # Extraer el ultimo modelo mencionado en la query como referencia para follow-ups.
        comparison_models = _extract_comparison_models(question)
        if comparison_models:
            last_mentioned = comparison_models[-1]  # ej: "Toyota Fortuner"
            parts = last_mentioned.split()
            if len(parts) >= 2:
                updates["last_make"] = parts[0]
                updates["last_model"] = " ".join(parts[1:])
            else:
                updates["last_model"] = last_mentioned
        elif current_model:
            updates["last_model"] = current_model
            if current_make:
                updates["last_make"] = current_make

        # Fallback de memory: si no hay modelo en la pregunta, usar last_model del estado
        if not current_model and intent_data.get("needs_retrieval"):
            prev_model = state.get("last_model")
            prev_make = state.get("last_make")
            if prev_model:
                intent_data["entities"]["model"] = prev_model
                if prev_make and not current_make:
                    intent_data["entities"]["make"] = prev_make
                intent_data["_model_from_memory"] = True

        # Registrar en trazabilidad
        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = ["classify_intent"]
        traza["clasificacion"] = intent_data
        traza["decision"] = {
            "ruta_seleccionada": "rag" if intent_data["needs_retrieval"] else "general",
            "motivo": intent_data["reason"],
        }
        if intent_data.get("_model_from_memory"):
            traza["model_memory_fallback"] = {
                "model": state.get("last_model"),
                "make": state.get("last_make"),
            }
        updates["trazabilidad"] = traza
        return updates

    # Intents que siempre disparan tools (decision determinista en decide_tools)
    _TOOL_INTENTS = {"Comparación", "Resumen"}

    # ── Nodo 3: decide_tools ─────────────────────────────────────────────
    def decide_tools(state: RAGState) -> dict[str, Any]:
        """Decide si el flujo debe pasar por call_tools/tools o ir directo a generate.

        Decision determinista (no usa LLM):
        - Comparacion y Resumen SIEMPRE activan tools
        - Busqueda NO usa tools (genera directo desde contexto)
        - Keyword fallback como segunda red de seguridad

        Retorna: usar_tools (bool), trazabilidad actualizada
        """
        intent = state.get("intent") or {}
        intent_name = intent.get("intent", "")
        question = state.get("question", "")

        usar_tools = intent_name in _TOOL_INTENTS

        # Keyword fallback: si el clasificador erro, detectar por palabras clave
        kw_override = _keyword_intent_override(question)
        if not usar_tools and kw_override:
            usar_tools = True

        motivo = f"intent={intent_name}, en _TOOL_INTENTS={intent_name in _TOOL_INTENTS}"
        if kw_override and intent_name not in _TOOL_INTENTS:
            motivo += f", keyword_override={kw_override} (query matched fallback regex)"

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["decide_tools"]
        traza["tools_decision"] = {
            "usar_tools": usar_tools,
            "motivo": motivo,
        }
        return {"usar_tools": usar_tools, "trazabilidad": traza}

    # ── Nodo 2a: answer_general ──────────────────────────────────────────
    def answer_general(state: RAGState) -> dict[str, Any]:
        """Responde preguntas generales sin retrieval documental.

        Se activa cuando el clasificador determina GENERAL (needs_retrieval=false).
        Usa el LLM directamente sin consultar ChromaDB ni tools.
        Incluye historial conversacional para contexto.

        Retorna: answer, messages, trazabilidad
        """
        question = state["question"]
        history = _history_text(state.get("messages") or [])
        user_prompt = (
            f"Historial reciente:\n{history}\n\nConsulta actual:\n{question}"
            if history
            else question
        )

        response = answer_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Eres un asistente automotriz. "
                        "Responde de forma clara y concisa. "
                        "No uses tools ni recuperación documental para esta respuesta."
                    )
                ),
                HumanMessage(content=user_prompt),
            ]
        )
        answer = response.content if isinstance(response.content, str) else str(response.content)

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["answer_general"]
        traza["chunks_recuperados"] = []
        traza["prompt_repr"] = {
            "modo": "general_sin_retrieval",
            "system": "asistente automotriz respuesta directa",
        }
        return {
            "answer": answer,
            "messages": [response],
            "trazabilidad": traza,
            "origen_respuesta": "answer_general",
        }

    # ── Nodo 2b: retrieve ────────────────────────────────────────────────
    def retrieve(state: RAGState) -> dict[str, Any]:
        """Busqueda semantica en ChromaDB para recuperar chunks relevantes.

        Proceso:
        1. Determina k (dynamic k del LLM o mapa fijo, con keyword override)
        2. Reescribe la query si hay historial (resolver referencias como "ese modelo")
        3. Construye filtros de metadata (marca/modelo con variantes normalizadas)
        4. Comparaciones: retrieval balanceado (k/2 por cada modelo)
        5. Fallback: si el filtro no retorna nada, busca sin filtro

        Retorna: docs (List[Document]), trazabilidad actualizada
        """
        intent = state.get("intent") or {}
        intent_name = intent.get("intent", "Búsqueda")
        question = state["question"]

        # Keyword override: corregir k cuando el clasificador falla
        kw_override = _keyword_intent_override(question)
        effective_intent = kw_override if kw_override else intent_name

        # Dynamic k: usar suggested_k del LLM si esta disponible y valido
        suggested_k = intent.get("suggested_k")
        if suggested_k and isinstance(suggested_k, int) and 1 <= suggested_k <= 20:
            k = suggested_k
        else:
            k = _k_desde_intent(effective_intent)

        # Reescribir query si hay historial (resolver "ese", "el otro", "ese modelo")
        history = _history_text(state.get("messages") or [])

        if history:
            rewrite_prompt = (
                "Convierte la consulta actual en una consulta autocontenida para búsqueda semántica.\n"
                "Usa el historial solo para resolver referencias (ej: 'el otro', 'ese modelo').\n"
                "Devuelve SOLO la consulta final, sin explicación.\n\n"
                f"Historial:\n{history}\n\nConsulta actual:\n{question}"
            )
            rewritten = rewrite_llm.invoke([HumanMessage(content=rewrite_prompt)])
            retrieval_query = rewritten.content if isinstance(rewritten.content, str) else question
            retrieval_query = retrieval_query.strip() or question
        else:
            retrieval_query = question

        # Construir filtro de metadata desde entidades clasificadas
        entities = intent.get("entities") or {}
        where_filter = _build_retrieval_filter(entities)

        # Comparaciones: retrieval balanceado (k/2 por modelo)
        comparison_models = _extract_comparison_models(question)
        is_comparison = effective_intent == "Comparación" and comparison_models and len(comparison_models) == 2

        if k <= 0:
            docs: List[Document] = []
        elif is_comparison:
            # Retrieval balanceado: mitad de chunks para cada modelo
            k_per_model = k // 2
            docs = []
            filters_used = []
            for cm in comparison_models:
                # Extraer marca y modelo de cada parte (ej: "Toyota Hilux" → make=Toyota, model=Hilux)
                parts = cm.split()
                if len(parts) >= 2:
                    cm_make, cm_model = parts[0], " ".join(parts[1:])
                else:
                    cm_make, cm_model = None, cm
                cm_variants = _model_variants(cm_model, cm_make)
                cm_filter = {"modelo": {"$in": cm_variants}}
                cm_docs = vs.similarity_search(retrieval_query, k=k_per_model, filter=cm_filter)
                if not cm_docs:
                    # Fallback: buscar sin filtro especifico
                    cm_docs = vs.similarity_search(cm, k=k_per_model)
                docs.extend(cm_docs)
                filters_used.append({"model": cm, "variants": cm_variants, "found": len(cm_docs)})
            where_filter = {"comparison_balanced": filters_used}
        else:
            if where_filter:
                docs = vs.similarity_search(retrieval_query, k=k, filter=where_filter)
                # Fallback: si el filtro no devuelve resultados, buscar sin filtro
                if not docs:
                    docs = vs.similarity_search(retrieval_query, k=k)
                    where_filter = None
            else:
                docs = vs.similarity_search(retrieval_query, k=k)

        # Registrar en trazabilidad
        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["retrieve"]
        traza["k_utilizado"] = k
        traza["k_origen"] = "suggested_k" if (suggested_k and isinstance(suggested_k, int) and 1 <= suggested_k <= 20) else "mapa_fijo"
        if suggested_k:
            traza["suggested_k"] = suggested_k
        traza["query_retrieval"] = retrieval_query
        traza["chunks_recuperados"] = _retrieved_chunk_payload(docs)
        if where_filter:
            traza["metadata_filter"] = where_filter
        if kw_override:
            traza["keyword_override"] = {
                "intent_clasificador": intent_name,
                "intent_efectivo": effective_intent,
            }
        return {"docs": docs, "trazabilidad": traza}

    # ── Nodo 4: call_tools ───────────────────────────────────────────────
    def call_tools(state: RAGState) -> dict[str, Any]:
        """LLM genera tool_calls que luego ejecuta el ToolNode.

        El LLM con tools bindeadas decide que herramientas invocar
        (resumir_ficha, comparar_modelos, buscar_especificacion, etc.)
        basado en la pregunta del usuario. No ejecuta las tools directamente;
        genera AIMessage con tool_calls que el ToolNode procesa despues.

        Retorna: messages (con AIMessage que contiene tool_calls), trazabilidad
        """
        question = state["question"]
        docs = state.get("docs", [])
        context = _retrieval_context(docs) if docs else ""

        response = answer_llm_with_tools.invoke(
            [
                SystemMessage(
                    content=(
                        "Eres un asistente de fichas técnicas vehiculares con acceso a tools.\n"
                        "Usa las tools disponibles para responder la consulta.\n"
                        "DEBES invocar al menos una tool."
                    )
                ),
                HumanMessage(content=question),
            ]
        )
        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["call_tools"]
        return {"messages": [response], "trazabilidad": traza}

    # ── Nodo 6: generate_grounded ────────────────────────────────────────
    def generate_grounded(state: RAGState) -> dict[str, Any]:
        """Genera la respuesta final con grounding estricto.

        Combina contexto de retrieval (chunks de ChromaDB) con output de tools
        (si las hubo) y genera la respuesta usando el LLM con reglas estrictas
        anti-hallucination.

        Si es un reintento (critic_feedback presente), adjunta las instrucciones
        de correccion del critico al prompt para que el LLM corrija los problemas.

        Guarda en trazabilidad: prompt completo enviado, snippets de chunks con metadata.

        Retorna: answer, messages, trazabilidad con prompt_enviado y chunks_con_contenido
        """
        docs = state.get("docs", [])
        context = _retrieval_context(docs)
        question = state["question"]

        # Extraer resultados de tools (ToolMessages) si existen
        tool_output = ""
        if state.get("usar_tools", False):
            tool_output = _extract_tool_results(state.get("messages") or [])

        # Combinar contexto de retrieval con output de tools
        combined_context = context
        if tool_output:
            if combined_context:
                combined_context += "\n\n=== Resultado de tools ===\n\n" + tool_output
            else:
                combined_context = tool_output

        # Si no hay contexto ni tools, retornar mensaje de "no encontrado"
        if not docs and not tool_output:
            answer = "No encontrado en el contexto recuperado."
            traza = dict(state.get("trazabilidad") or {})
            traza["ruta"] = traza.get("ruta", []) + ["generate_grounded"]
            traza["prompt_repr"] = {
                "modo": "grounded_rag",
                "question": question,
                "retrieved_count": 0,
                "tool_output": bool(tool_output),
            }
            return {
                "answer": answer,
                "messages": [AIMessage(content=answer)],
                "trazabilidad": traza,
                "origen_respuesta": "generate_grounded",
            }

        # Construir instruccion de correccion si es un reintento del regeneration loop
        critic_feedback = state.get("critic_feedback")
        critic_instruction = ""
        if critic_feedback:
            critic_instruction = (
                "\n\n=== CORRECCIÓN REQUERIDA ===\n"
                "Tu respuesta anterior fue rechazada por el crítico. Corrige estos problemas:\n"
                + "\n".join(f"- {fb}" for fb in critic_feedback)
                + "\nGenera una nueva respuesta corregida."
            )

        prompt_repr = {
            "modo": "grounded_rag",
            "question": question,
            "retrieved_count": len(docs),
            "tool_output_included": bool(tool_output),
            "formato_cita": "[doc_id=...; página=...; chunk_id=...]",
            "retry_count": state.get("retry_count", 0),
            "has_critic_feedback": bool(critic_feedback),
        }

        user_content = GROUNDED_GENERATION_USER_TEMPLATE.format(
            question=question,
            context=combined_context,
        )
        if critic_instruction:
            user_content += critic_instruction

        # Guardar snippets de chunks para trazabilidad (primeros 200 chars de cada chunk)
        chunk_snippets = []
        for d in docs:
            md = d.metadata or {}
            snip = {
                "doc_id": md.get("doc_id") or md.get("source"),
                "source": md.get("source"),
                "page": md.get("page"),
                "snippet": d.page_content[:200],
            }
            if md.get("chunk_id"):
                snip["chunk_id"] = md["chunk_id"]
            chunk_snippets.append(snip)

        # NO pasar historial de conversacion para evitar contaminacion.
        # El query ya fue reescrito en retrieve() para ser autocontenido.
        response = answer_llm_with_tools.invoke(
            [
                SystemMessage(content=GROUNDED_GENERATION_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
        )
        answer = response.content if isinstance(response.content, str) else str(response.content)

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["generate_grounded"]
        traza["prompt_repr"] = prompt_repr
        traza["prompt_enviado"] = user_content
        traza["chunks_con_contenido"] = chunk_snippets
        return {
            "answer": answer,
            "messages": [response],
            "trazabilidad": traza,
            "origen_respuesta": "generate_grounded",
        }

    # ── Nodo 7: evaluate_grounding ───────────────────────────────────────
    def evaluate_grounding(state: RAGState) -> dict[str, Any]:
        """Critico de grounding: evalua la calidad de la respuesta generada.

        Evalua 5 criterios: soportada_por_contexto, tiene_citas, suficiente,
        aprobada y score (0.0-1.0).

        Tres caminos posibles:
        1. Aprobada (score >= 0.5 o approved=true) → flujo termina
        2. Rechazada + puede reintentar (score < 0.5, retry_count < MAX_RETRIES)
           → guarda feedback en critic_feedback, incrementa retry_count,
             el conditional edge redirige a generate_grounded
        3. Rechazada + sin reintentos → reemplaza respuesta con fallback seguro

        Retorna: eval_result, trazabilidad (con retry_number y will_retry),
                 opcionalmente retry_count, critic_feedback o answer de fallback
        """
        docs = state.get("docs", [])
        answer = state.get("answer", "")
        question = state["question"]
        retry_count = state.get("retry_count", 0)

        # Enviar al critico: pregunta + chunks + respuesta a evaluar
        retrieved_chunks_json = json.dumps(_retrieved_chunk_payload(docs), ensure_ascii=False)
        structured = critic_llm.with_structured_output(GroundingEvaluation)
        result: GroundingEvaluation = structured.invoke(
            [
                SystemMessage(content=GROUNDING_CRITIC_SYSTEM_PROMPT),
                HumanMessage(
                    content=GROUNDING_CRITIC_USER_TEMPLATE.format(
                        question=question,
                        retrieved_chunks=retrieved_chunks_json,
                        answer=answer,
                    )
                ),
            ]
        )
        eval_data = eval_to_dict(result)

        updates: dict[str, Any] = {}
        replaced = False
        rejected = not eval_data.get("approved", True) and eval_data.get("score", 1.0) < 0.5
        can_retry = rejected and retry_count < MAX_RETRIES

        if can_retry:
            # Regeneration loop: guardar feedback del critico e incrementar contador
            updates["retry_count"] = retry_count + 1
            updates["critic_feedback"] = eval_data.get("issues", [])
        elif rejected:
            # Max reintentos agotados — reemplazar con respuesta fallback segura
            issues = eval_data.get("issues", [])
            clarification = eval_data.get("clarification_question")
            fallback_parts = [
                "No fue posible generar una respuesta confiable basada en el contexto disponible."
            ]
            if issues:
                fallback_parts.append("Problemas detectados: " + "; ".join(issues) + ".")
            if clarification:
                fallback_parts.append(clarification)
            fallback_answer = "\n\n".join(fallback_parts)
            updates["answer"] = fallback_answer
            updates["messages"] = [AIMessage(content=fallback_answer)]
            replaced = True

        # Registrar verificacion en trazabilidad
        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["evaluate_grounding"]
        traza["verificacion"] = {
            "aprobada": eval_data.get("approved"),
            "puntuacion": eval_data.get("score"),
            "soportada_en_contexto": eval_data.get("supported_by_context"),
            "tiene_citas": eval_data.get("has_citations"),
            "suficiente": eval_data.get("complete_enough"),
            "issues": eval_data.get("issues", []),
            "pregunta_aclaracion": eval_data.get("clarification_question"),
            "respuesta_reemplazada": replaced,
            "retry_number": retry_count,
            "will_retry": can_retry,
        }
        updates["eval_result"] = eval_data
        updates["trazabilidad"] = traza
        return updates

    # ── Funciones de routing (conditional edges) ─────────────────────────

    def route_after_evaluate(state: RAGState) -> str:
        """Decide si reintentar generacion o terminar despues de la evaluacion.

        Si la respuesta fue rechazada (score < 0.5) y hay reintentos disponibles
        (retry_count <= MAX_RETRIES) y hay feedback del critico, redirige a
        generate_grounded para un nuevo intento con las correcciones.
        Si no, termina el grafo.
        """
        eval_data = state.get("eval_result") or {}
        retry_count = state.get("retry_count", 0)
        rejected = not eval_data.get("approved", True) and eval_data.get("score", 1.0) < 0.5
        if rejected and retry_count <= MAX_RETRIES and state.get("critic_feedback"):
            return "generate_grounded"
        return END

    def route_after_classify(state: RAGState) -> str:
        """Decide ruta despues de clasificar: retrieval o respuesta general directa."""
        intent = state.get("intent") or {}
        if intent.get("needs_retrieval", True):
            return "retrieve"
        return "answer_general"

    def route_after_decide_tools(state: RAGState) -> str:
        """Decide si pasar por tools o ir directo a generacion grounded."""
        if state.get("usar_tools", False):
            return "call_tools"
        return "generate_grounded"

    # ── Ensamblaje del grafo ─────────────────────────────────────────────

    graph = (
        StateGraph(RAGState)
        # Nodos
        .add_node("classify_intent", classify_intent)
        .add_node("answer_general", answer_general)
        .add_node("retrieve", retrieve)
        .add_node("decide_tools", decide_tools)
        .add_node("call_tools", call_tools)
        .add_node("tools", tool_node)
        .add_node("generate_grounded", generate_grounded)
        .add_node("evaluate_grounding", evaluate_grounding)
        # Edges
        .add_edge(START, "classify_intent")
        .add_conditional_edges(
            "classify_intent",
            route_after_classify,
            {"retrieve": "retrieve", "answer_general": "answer_general"},
        )
        .add_edge("answer_general", END)
        .add_edge("retrieve", "decide_tools")
        .add_conditional_edges(
            "decide_tools",
            route_after_decide_tools,
            {"call_tools": "call_tools", "generate_grounded": "generate_grounded"},
        )
        .add_edge("call_tools", "tools")
        .add_edge("tools", "generate_grounded")
        .add_edge("generate_grounded", "evaluate_grounding")
        .add_conditional_edges(
            "evaluate_grounding",
            route_after_evaluate,
            {"generate_grounded": "generate_grounded", END: END},
        )
        .compile(checkpointer=MemorySaver())
    )
    print(graph.get_graph().draw_ascii())
    return graph
