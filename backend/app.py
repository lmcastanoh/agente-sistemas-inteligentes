# backend/app.py
# ==============================================================================
# API FastAPI para el sistema RAG de fichas tecnicas vehiculares.
#
# Endpoints:
#   POST /ingest       — Ingesta PDFs desde un directorio a ChromaDB
#   POST /chat/stream  — Chat con streaming SSE (Server-Sent Events)
#
# El grafo LangGraph se construye una sola vez al iniciar la aplicacion.
# Cada sesion de chat se identifica por session_id para mantener historial.
# ==============================================================================
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from rag_graph import build_rag_graph
from rag_store import ingest

# Cargar .env desde la raiz del proyecto (dos niveles arriba de backend/)
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="LangGraph RAG API")

# Construir el grafo LangGraph una sola vez al arrancar.
# Incluye: vector store, LLMs, tools, nodos y edges del grafo.
graph = build_rag_graph()


class ChatRequest(BaseModel):
    """Modelo de request para el endpoint de chat.

    Campos:
        question:   Pregunta del usuario en lenguaje natural.
        session_id: Identificador de sesion para mantener historial conversacional.
                    Permite follow-ups como "y cuanto pesa?" heredando el modelo previo.
    """

    question: str
    session_id: str = "default"


class IngestRequest(BaseModel):
    """Modelo de request para el endpoint de ingestion.

    Campos:
        data_dir: Ruta al directorio con PDFs organizados por marca.
                  Ejemplo: "./data" (relativo a backend/)
    """

    data_dir: str


@app.post("/ingest")
def ingest_route(req: IngestRequest):
    """Ingesta documentos PDF a ChromaDB.

    Proceso:
    1. Lee PDFs del directorio indicado (texto nativo o OCR)
    2. Divide en chunks (1000 chars, 150 overlap)
    3. Genera embeddings con all-MiniLM-L6-v2
    4. Almacena en ChromaDB con metadata (source, page, marca, modelo, doc_id, chunk_id)

    Retorna: cantidad de documentos, chunks e IDs agregados.
    """
    result = ingest(req.data_dir)
    return JSONResponse(result)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Chat con streaming SSE.

    Ejecuta el grafo RAG completo y emite 3 tipos de eventos SSE:
    - "token":         Respuesta final del RAG (texto)
    - "trazabilidad":  JSON con ruta completa, decisiones, chunks, evaluacion
    - "done":          Senial de fin del stream

    Nota: emite solo la respuesta final (no tokens intermedios) para evitar
    texto duplicado cuando el grafo reintenta la generacion internamente.
    """

    async def event_gen():
        # Estado inicial del grafo: pregunta, listas vacias
        inputs = {
            "question": req.question,
            "docs": [],
            "answer": "",
            "messages": [HumanMessage(content=req.question)],
            "agent_steps": [],
            "agent_context": "",
            "eval_steps": [],
        }
        # thread_id vincula esta invocacion a una sesion persistente (MemorySaver)
        config = {"configurable": {"thread_id": req.session_id}}

        # Streaming real: emitir tokens de generate_grounded en tiempo real
        # y eventos de progreso cuando cada nodo termina
        _TRACKED_NODES = {
            "classify_intent", "retrieve", "agent_reason",
            "generate_grounded", "eval_agent",
        }
        streamed_answer = False

        try:
            async for event in graph.astream_events(inputs, config=config, version="v2"):
                kind = event["event"]

                # Tokens del nodo generate_grounded en tiempo real
                if kind == "on_chat_model_stream":
                    node = event.get("metadata", {}).get("langgraph_node", "")
                    if node == "generate_grounded":
                        chunk = event["data"].get("chunk")
                        token = getattr(chunk, "content", "") if chunk else ""
                        if token:
                            # Escapar newlines para que SSE no los corte
                            yield {"event": "token", "data": token.replace("\n", "\\n")}
                            streamed_answer = True

                # Progreso: notificar cuando cada nodo termina
                elif kind == "on_chain_end":
                    name = event.get("name", "")
                    if name in _TRACKED_NODES:
                        yield {"event": "progress", "data": name}

        except Exception as exc:
            yield {"event": "token", "data": f"Error interno: {exc}"}

        # Fallback: si no se streamearon tokens (ej: answer_general o sin contexto)
        if not streamed_answer:
            try:
                final_state = await graph.aget_state(config)
                answer = final_state.values.get("answer", "")
                if isinstance(answer, str) and answer.strip():
                    yield {"event": "token", "data": answer.replace("\n", "\\n")}
            except Exception:
                pass

        # Emitir trazabilidad desde el estado final del grafo
        try:
            final_state = await graph.aget_state(config)
            traza = final_state.values.get("trazabilidad", {})
            if traza:
                yield {"event": "trazabilidad", "data": json.dumps(traza, ensure_ascii=False)}
        except Exception:
            pass

        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_gen())
