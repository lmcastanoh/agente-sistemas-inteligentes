# backend/rag_graph.py
from __future__ import annotations

from typing import Annotated, List, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages

from rag_store import get_vector_store
from tools import (
    listar_modelos_disponibles,
    buscar_especificacion,
    buscar_por_marca,
    comparar_modelos,
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
    docs:     List[Document]
    answer:   str
    messages: Annotated[List[BaseMessage], add_messages]


def build_rag_graph():
    vs = get_vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.2)
    llm_with_tools = llm.bind_tools(TOOLS)

    def retrieve(state: RAGState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"docs": docs}

    def generate(state: RAGState) -> dict:
        context = "\n\n".join(
            f"[marca={d.metadata.get('marca','')} modelo={d.metadata.get('modelo','')} p.{d.metadata.get('page','')}]\n{d.page_content}"
            for d in state["docs"]
        )
        system = (
            "Eres un asistente experto en fichas técnicas de vehículos. "
            "Tienes acceso a herramientas para consultar el catálogo. "
            "Responde siempre en español usando el contexto o las herramientas disponibles.\n\n"
            f"Contexto recuperado:\n{context}"
        )
        from langchain_core.messages import HumanMessage, SystemMessage
        msgs = state.get("messages") or []
        if not msgs:
            msgs = [SystemMessage(content=system), HumanMessage(content=state["question"])]

        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], "answer": response.content}

    def should_use_tool(state: RAGState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(TOOLS)

    graph = (
        StateGraph(RAGState)
        .add_node("retrieve", retrieve)
        .add_node("generate", generate)
        .add_node("tools", tool_node)
        .add_edge(START, "retrieve")
        .add_edge("retrieve", "generate")
        .add_conditional_edges("generate", should_use_tool, {"tools": "tools", END: END})
        .add_edge("tools", "generate")
        .compile()
    )
    print(graph.get_graph().draw_ascii())
    return graph
