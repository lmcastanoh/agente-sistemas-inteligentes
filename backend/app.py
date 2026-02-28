# backend/app.py
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from rag_graph import build_rag_graph
from rag_store import ingest

# Load .env from project root (two levels up), not from inside backend/
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="LangGraph RAG API")

graph = build_rag_graph()


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class IngestRequest(BaseModel):
    data_dir: str


@app.post("/ingest")
def ingest_route(req: IngestRequest):
    result = ingest(req.data_dir)
    return JSONResponse(result)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streams only the final answer to avoid duplicated text when the graph retries
    grounded generation internally.
    """

    async def event_gen():
        inputs = {
            "question": req.question,
            "docs": [],
            "answer": "",
            "messages": [HumanMessage(content=req.question)],
            "usar_tools": False,
        }
        config = {"configurable": {"thread_id": req.session_id}}

        # Run graph to completion, then emit only final answer.
        try:
            final = await graph.ainvoke(inputs, config=config)
            answer = final.get("answer", "")
            if isinstance(answer, str) and answer.strip():
                yield {"event": "token", "data": answer}
        except Exception as exc:
            yield {"event": "token", "data": f"Error interno: {exc}"}

        # Emit trazabilidad from final graph state
        try:
            final_state = await graph.aget_state(config)
            traza = final_state.values.get("trazabilidad", {})
            if traza:
                yield {"event": "trazabilidad", "data": json.dumps(traza, ensure_ascii=False)}
        except Exception:
            pass

        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_gen())
