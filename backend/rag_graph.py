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

K_POR_INTENCION = {
    "comparación": 10,
    "resumen": 8,
    "búsqueda": 8,
    "general": 0,
}


def _k_desde_intent(intent: str) -> int:
    mapa = {
        "Búsqueda": "búsqueda",
        "Resumen": "resumen",
        "Comparación": "comparación",
        "GENERAL": "general",
    }
    return K_POR_INTENCION.get(mapa.get(intent, "búsqueda"), 4)


# Keyword fallback para detectar resúmenes y comparaciones que el clasificador no captó
_RE_RESUMEN = re.compile(
    r"\b(resumen|resume[mn]?|resúm[ea]|overview|ficha\s+completa|descripción\s+general)\b",
    re.IGNORECASE,
)
_RE_COMPARACION = re.compile(
    r"\b(compar[aáeo]|versus|vs\.?|diferencias?\s+entre)\b",
    re.IGNORECASE,
)


def _keyword_intent_override(question: str) -> str | None:
    """Detecta intent por keywords cuando el clasificador falla."""
    if _RE_COMPARACION.search(question):
        return "Comparación"
    if _RE_RESUMEN.search(question):
        return "Resumen"
    return None


def _build_retrieval_filter(entities: dict[str, Any] | None) -> dict | None:
    """Construye filtro de metadata para ChromaDB a partir de las entidades del clasificador."""
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
    """Genera variantes normalizadas de un nombre de modelo."""
    no_hyphen = model.replace("-", " ")
    bases = {model, model.title(), no_hyphen, no_hyphen.title()}
    variants: set[str] = set(bases)
    if make:
        for b in bases:
            variants.add(f"{make} {b}")
    return list(variants)


_RE_VS = re.compile(
    r"(?:compar[aáeo]\w*|diferencias?\s+entre)\s+(?:(?:el|la|los|las|del|al)\s+)?"
    r"(.+?)\s+(?:vs\.?|versus|contra|y|con)\s+(?:(?:el|la|los|las|del|al)\s+)?(.+)",
    re.IGNORECASE,
)


def _extract_comparison_models(question: str) -> list[str] | None:
    """Extrae los dos modelos de una query de comparación."""
    m = _RE_VS.search(question)
    if not m:
        return None
    return [m.group(1).strip(), m.group(2).strip()]


def _extract_tool_results(messages: List[BaseMessage]) -> str:
    """Extrae contenido de ToolMessages del historial."""
    parts: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.content:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if content.strip():
                parts.append(content.strip())
    return "\n\n---\n\n".join(parts)


MAX_RETRIES = 1


def _keep_latest(existing: Optional[str], new: Optional[str]) -> Optional[str]:
    """Reducer that keeps the latest non-None value."""
    return new if new is not None else existing


class RAGState(TypedDict):
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


def _history_text(messages: List[BaseMessage], max_items: int = 8) -> str:
    """Construye texto de historial conversacional para clasificador y rewriter.

    Filtra ToolMessages y AIMessages con tool_calls (son datos crudos internos,
    no aportan contexto conversacional y desplazan las preguntas reales).
    """
    if not messages:
        return ""
    # Solo mensajes conversacionales: Human + AI con contenido textual real
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
    """Corrige texto con caracteres duplicados por extracción corrupta de PDF.

    Detecta el patrón donde cada carácter aparece dos veces consecutivas
    (e.g., 'TTOOYYOOTTAA' → 'TOYOTA') y lo deduplica.
    Usa pares de alfanuméricos para detección robusta (ignora chars especiales corruptos).
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


def build_rag_graph():
    vs = get_vector_store()
    tools = [
        listar_modelos_disponibles,
        buscar_especificacion,
        buscar_por_marca,
        comparar_modelos,
        resumir_ficha,
    ]
    tool_node = ToolNode(tools)

    router_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    answer_llm = ChatOpenAI(model="gpt-5-nano", temperature=0.2)
    answer_llm_with_tools = answer_llm.bind_tools(tools)
    critic_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    rewrite_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    def classify_intent(state: RAGState) -> dict[str, Any]:
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
        # Extraer el último modelo mencionado en la query como referencia para follow-ups.
        comparison_models = _extract_comparison_models(question)
        if comparison_models:
            last_mentioned = comparison_models[-1]  # "Toyota Fortuner"
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

        # Fallback: si no hay modelo, usar last_model del estado anterior
        if not current_model and intent_data.get("needs_retrieval"):
            prev_model = state.get("last_model")
            prev_make = state.get("last_make")
            if prev_model:
                intent_data["entities"]["model"] = prev_model
                if prev_make and not current_make:
                    intent_data["entities"]["make"] = prev_make
                intent_data["_model_from_memory"] = True

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

    # Intents que siempre disparan tools
    _TOOL_INTENTS = {"Comparación", "Resumen"}

    def decide_tools(state: RAGState) -> dict[str, Any]:
        """Decisión determinista: Comparación y Resumen siempre usan tools.
        Incluye fallback por keywords en la pregunta original."""
        intent = state.get("intent") or {}
        intent_name = intent.get("intent", "")
        question = state.get("question", "")

        usar_tools = intent_name in _TOOL_INTENTS

        # Keyword fallback: si el clasificador erró, detectar por palabras clave
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

    def answer_general(state: RAGState) -> dict[str, Any]:
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

    def retrieve(state: RAGState) -> dict[str, Any]:
        intent = state.get("intent") or {}
        intent_name = intent.get("intent", "Búsqueda")
        question = state["question"]

        # Keyword override: corregir k cuando el clasificador falla
        kw_override = _keyword_intent_override(question)
        effective_intent = kw_override if kw_override else intent_name

        # Dynamic k: use LLM-suggested k if available, else fallback to fixed map
        suggested_k = intent.get("suggested_k")
        if suggested_k and isinstance(suggested_k, int) and 1 <= suggested_k <= 20:
            k = suggested_k
        else:
            k = _k_desde_intent(effective_intent)

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

        # Build metadata filter from classified entities
        entities = intent.get("entities") or {}
        where_filter = _build_retrieval_filter(entities)

        # Comparaciones: retrieval balanceado (k/2 por modelo)
        comparison_models = _extract_comparison_models(question)
        is_comparison = effective_intent == "Comparación" and comparison_models and len(comparison_models) == 2

        if k <= 0:
            docs: List[Document] = []
        elif is_comparison:
            k_per_model = k // 2
            docs = []
            filters_used = []
            for cm in comparison_models:
                # Extraer marca y modelo de cada parte (e.g., "Toyota Hilux" → make=Toyota, model=Hilux)
                parts = cm.split()
                if len(parts) >= 2:
                    cm_make, cm_model = parts[0], " ".join(parts[1:])
                else:
                    cm_make, cm_model = None, cm
                cm_variants = _model_variants(cm_model, cm_make)
                cm_filter = {"modelo": {"$in": cm_variants}}
                cm_docs = vs.similarity_search(retrieval_query, k=k_per_model, filter=cm_filter)
                if not cm_docs:
                    # Fallback: buscar sin filtro específico
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

    def call_tools(state: RAGState) -> dict[str, Any]:
        """LLM genera tool_calls que luego ejecuta el ToolNode."""
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

    def generate_grounded(state: RAGState) -> dict[str, Any]:
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

        # Build critic correction instruction if retrying
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

        # Save snippets for traceability
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

        # NO pasar historial de conversación para evitar contaminación.
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

    def evaluate_grounding(state: RAGState) -> dict[str, Any]:
        docs = state.get("docs", [])
        answer = state.get("answer", "")
        question = state["question"]
        retry_count = state.get("retry_count", 0)

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
            # Regeneration loop: save feedback and increment retry
            updates["retry_count"] = retry_count + 1
            updates["critic_feedback"] = eval_data.get("issues", [])
        elif rejected:
            # Max retries exhausted — fallback seguro
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

    def route_after_evaluate(state: RAGState) -> str:
        """Route after grounding evaluation: retry or end."""
        eval_data = state.get("eval_result") or {}
        retry_count = state.get("retry_count", 0)
        rejected = not eval_data.get("approved", True) and eval_data.get("score", 1.0) < 0.5
        if rejected and retry_count <= MAX_RETRIES and state.get("critic_feedback"):
            return "generate_grounded"
        return END

    # ── Routing functions ──────────────────────────────────────────

    def route_after_classify(state: RAGState) -> str:
        intent = state.get("intent") or {}
        if intent.get("needs_retrieval", True):
            return "retrieve"
        return "answer_general"

    def route_after_decide_tools(state: RAGState) -> str:
        if state.get("usar_tools", False):
            return "call_tools"
        return "generate_grounded"

    # ── Graph assembly ─────────────────────────────────────────────

    graph = (
        StateGraph(RAGState)
        .add_node("classify_intent", classify_intent)
        .add_node("answer_general", answer_general)
        .add_node("retrieve", retrieve)
        .add_node("decide_tools", decide_tools)
        .add_node("call_tools", call_tools)
        .add_node("tools", tool_node)
        .add_node("generate_grounded", generate_grounded)
        .add_node("evaluate_grounding", evaluate_grounding)
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
