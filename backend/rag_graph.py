from __future__ import annotations

import json
from typing import Annotated, Any, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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
    "búsqueda": 4,
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


class RAGState(TypedDict):
    question: str
    docs: List[Document]
    answer: str
    messages: Annotated[List[BaseMessage], add_messages]
    intent: Optional[dict[str, Any]]
    eval_result: Optional[dict[str, Any]]
    usar_tools: bool
    trazabilidad: dict[str, Any]


def _history_text(messages: List[BaseMessage], max_items: int = 8) -> str:
    if not messages:
        return ""
    items = messages[-max_items:]
    lines: list[str] = []
    for m in items:
        role = "Usuario"
        if isinstance(m, AIMessage):
            role = "Asistente"
        elif isinstance(m, SystemMessage):
            role = "Sistema"
        elif isinstance(m, ToolMessage):
            role = "Tool"
        content = m.content if isinstance(m.content, str) else str(m.content)
        if content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines)


def _retrieved_chunk_payload(docs: List[Document]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for d in docs:
        md = d.metadata or {}
        chunks.append(
            {
                "source": md.get("source"),
                "page": md.get("page"),
            }
        )
    return chunks


def _retrieval_context(docs: List[Document]) -> str:
    blocks: list[str] = []
    for d in docs:
        md = d.metadata or {}
        blocks.append(
            "\n".join(
                [
                    f"[Documento={md.get('source')}; página={md.get('page')}]",
                    d.page_content,
                ]
            )
        )
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

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = ["classify_intent"]
        traza["clasificacion"] = intent_data
        traza["decision"] = {
            "ruta_seleccionada": "rag" if intent_data["needs_retrieval"] else "general",
            "motivo": intent_data["reason"],
        }
        return {"intent": intent_data, "trazabilidad": traza}

    def decide_tools(state: RAGState) -> dict[str, Any]:
        """
        Decide si conviene usar tools en ruta no GENERAL.
        """
        question = state["question"]
        intent = state.get("intent") or {}
        history = _history_text(state.get("messages") or [])

        prompt = (
            "Decide si para responder la consulta conviene usar tools de catálogo.\n"
            "Responde SOLO JSON válido: {\"usar_tools\": true|false, \"motivo\": \"...\"}\n"
            "Usa true si: comparación entre modelos, resumen estructurado o datos específicos probablemente distribuidos.\n"
            "Usa false si el contexto recuperado probablemente será suficiente.\n\n"
            f"Intent: {intent.get('intent')}\n"
            f"Historial:\n{history}\n\n"
            f"Consulta:\n{question}"
        )
        raw = router_llm.invoke([HumanMessage(content=prompt)])
        try:
            data = json.loads(raw.content if isinstance(raw.content, str) else str(raw.content))
            usar_tools = bool(data.get("usar_tools", False))
            motivo = str(data.get("motivo", "")).strip()
        except Exception:
            usar_tools = False
            motivo = "fallback por parse inválido"

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["decide_tools"]
        traza["tools_decision"] = {"usar_tools": usar_tools, "motivo": motivo}
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
        k = _k_desde_intent(intent.get("intent", "Búsqueda"))
        history = _history_text(state.get("messages") or [])
        question = state["question"]

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

        if k <= 0:
            docs: List[Document] = []
        else:
            retriever = vs.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(retrieval_query)

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["retrieve"]
        traza["k_utilizado"] = k
        traza["query_retrieval"] = retrieval_query
        traza["chunks_recuperados"] = _retrieved_chunk_payload(docs)
        return {"docs": docs, "trazabilidad": traza}

    def generate_grounded(state: RAGState) -> dict[str, Any]:
        docs = state.get("docs", [])
        context = _retrieval_context(docs)
        question = state["question"]
        history = _history_text(state.get("messages") or [])

        if not docs:
            answer = "No encontrado en el contexto recuperado."
            traza = dict(state.get("trazabilidad") or {})
            traza["ruta"] = traza.get("ruta", []) + ["generate_grounded"]
            traza["prompt_repr"] = {
                "modo": "grounded_rag",
                "question": question,
                "retrieved_count": 0,
            }
            return {
                "answer": answer,
                "messages": [AIMessage(content=answer)],
                "trazabilidad": traza,
                "origen_respuesta": "generate_grounded",
            }

        prompt_repr = {
            "modo": "grounded_rag",
            "question": question,
            "retrieved_count": len(docs),
            "formato_cita": "[Documento=...; página=...]",
        }
        response = answer_llm_with_tools.invoke(
            [
                SystemMessage(content=GROUNDED_GENERATION_SYSTEM_PROMPT),
                HumanMessage(
                    content=GROUNDED_GENERATION_USER_TEMPLATE.format(
                        question=(
                            f"Historial reciente:\n{history}\n\nConsulta actual:\n{question}"
                            if history
                            else question
                        ),
                        context=context,
                    )
                ),
            ]
        )
        answer = response.content if isinstance(response.content, str) else str(response.content)

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["generate_grounded"]
        traza["prompt_repr"] = prompt_repr
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
        }
        return {"eval_result": eval_data, "trazabilidad": traza}

    # ── Routing functions ──────────────────────────────────────────

    def route_after_classify(state: RAGState) -> str:
        intent = state.get("intent") or {}
        if intent.get("needs_retrieval", True):
            return "retrieve"
        return "answer_general"

    def route_after_decide_tools(state: RAGState) -> str:
        if state.get("usar_tools", False):
            return "tools"
        return "generate_grounded"

    # ── Graph assembly ─────────────────────────────────────────────

    graph = (
        StateGraph(RAGState)
        .add_node("classify_intent", classify_intent)
        .add_node("answer_general", answer_general)
        .add_node("retrieve", retrieve)
        .add_node("decide_tools", decide_tools)
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
            {"tools": "tools", "generate_grounded": "generate_grounded"},
        )
        .add_edge("tools", "generate_grounded")
        .add_edge("generate_grounded", "evaluate_grounding")
        .add_edge("evaluate_grounding", END)
        .compile(checkpointer=MemorySaver())
    )
    print(graph.get_graph().draw_ascii())
    return graph
