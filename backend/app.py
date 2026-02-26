# backend/app.py
from __future__ import annotations

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
    Streams tokens using LangGraph stream_mode="messages".
    """

    async def event_gen():
        inputs = {
            "question": req.question,
            "docs": [],
            "answer": "",
            "messages": [HumanMessage(content=req.question)],
        }
        config = {"configurable": {"thread_id": req.session_id}}

        async for chunk in graph.astream(inputs, config=config, stream_mode="messages"):
            try:
                token, _meta = chunk
                if (
                    hasattr(token, "content")
                    and token.content
                    and not isinstance(token, ToolMessage)
                    and not getattr(token, "tool_calls", None)
                ):
                    yield {"event": "token", "data": token.content}
            except Exception:
                pass

    return EventSourceResponse(event_gen())
