# backend/app.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from langchain_core.messages import ToolMessage

from rag_graph import build_rag_graph
from rag_store import ingest

from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root (two levels up), not from inside backend/
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="LangGraph RAG API")

graph = build_rag_graph()

class ChatRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    data_dir: str

@app.post("/ingest")
def ingest_route(req: IngestRequest):
    result = ingest(req.data_dir)
    return JSONResponse(result)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streams tokens using LangGraph stream_mode="messages" :contentReference[oaicite:12]{index=12}
    """
    async def event_gen():
        inputs = {"question": req.question, "docs": [], "answer": ""}

        async for chunk in graph.astream(inputs, stream_mode="messages"):
            try:
                token, meta = chunk
                if (hasattr(token, "content") and token.content
                        and not isinstance(token, ToolMessage)
                        and not getattr(token, "tool_calls", None)):
                    yield {"event": "token", "data": token.content}
            except Exception:
                pass

    return EventSourceResponse(event_gen())