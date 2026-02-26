# backend/rag_graph.py
from __future__ import annotations

from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from rag_store import get_vector_store
from tools import (
    buscar_especificacion,
    buscar_por_marca,
    comparar_modelos,
    listar_modelos_disponibles,
    resumir_ficha,
)

TOOLS = [
    listar_modelos_disponibles,
    buscar_especificacion,
    buscar_por_marca,
    comparar_modelos,
    resumir_ficha,
]


class RAGState(TypedDict):
    question: str
    docs: List[Document]
    answer: str
    messages: Annotated[List[BaseMessage], add_messages]


def _build_grounding_context(docs: List[Document]) -> tuple[str, str]:
    """
    Build retrieval context with stable source IDs for citation grounding.
    Returns:
      - context string consumed by the LLM
      - source catalog string shown as allowed citations
    """
    chunks: list[str] = []
    catalog: list[str] = []

    for idx, d in enumerate(docs, start=1):
        sid = f"S{idx}"
        source = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page", "")
        marca = d.metadata.get("marca", "")
        modelo = d.metadata.get("modelo", "")

        header = f"[{sid}] source={source} page={page} marca={marca} modelo={modelo}"
        chunks.append(f"{header}\n{d.page_content}")
        catalog.append(f"- [{sid}] {source} (p.{page})")

    return "\n\n".join(chunks), "\n".join(catalog)


def build_rag_graph():
    vs = get_vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.2)
    llm_with_tools = llm.bind_tools(TOOLS)

    def retrieve(state: RAGState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"docs": docs}

    def generate(state: RAGState) -> dict:
        context, source_catalog = _build_grounding_context(state["docs"])
        system_prompt = (
            "Eres un asistente experto en fichas tecnicas de vehiculos. "
            "Tienes acceso a herramientas para consultar el catalogo. "
            "Responde siempre en espanol usando el contexto o las herramientas disponibles.\n"
            "Prioriza legibilidad: usa encabezados cortos y bullets claros.\n"
            "No repitas disculpas ni texto redundante.\n"
            "Si faltan datos, indica brevemente que falta y que se necesita.\n"
            "Evita tablas grandes llenas de N/D.\n\n"
            "Reglas de grounding (obligatorias):\n"
            "1) Toda afirmacion factual debe estar respaldada por el contexto recuperado o por salida de tools.\n"
            "2) Cita evidencias del contexto con formato [S1], [S2], etc., inmediatamente despues de cada afirmacion.\n"
            "3) No inventes valores ni fuentes. Si no hay evidencia suficiente, dilo explicitamente.\n"
            "4) Si faltan datos para comparar o resumir, entrega una respuesta corta con 'Datos faltantes'.\n\n"
            f"Fuentes permitidas:\n{source_catalog}\n\n"
            f"Contexto recuperado:\n{context}"
        )

        history = state.get("messages") or []
        response = llm_with_tools.invoke([SystemMessage(content=system_prompt), *history])

        answer_content = response.content if isinstance(response.content, str) else ""
        if answer_content and state["docs"]:
            grounded_footer = (
                "\n\n---\n"
                "Fuentes recuperadas:\n"
                f"{source_catalog}"
            )
            answer_content = f"{answer_content}{grounded_footer}"
            response.content = answer_content

        return {"messages": [response], "answer": answer_content}

    def should_use_tool(state: RAGState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(TOOLS)
    memory = MemorySaver()

    graph = (
        StateGraph(RAGState)
        .add_node("retrieve", retrieve)
        .add_node("generate", generate)
        .add_node("tools", tool_node)
        .add_edge(START, "retrieve")
        .add_edge("retrieve", "generate")
        .add_conditional_edges("generate", should_use_tool, {"tools": "tools", END: END})
        .add_edge("tools", "generate")
        .compile(checkpointer=memory)
    )
    print(graph.get_graph().draw_ascii())
    return graph
