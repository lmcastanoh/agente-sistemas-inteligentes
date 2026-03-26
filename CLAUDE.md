# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic RAG system for querying vehicle technical datasheets (fichas técnicas). Built with FastAPI + LangGraph + Streamlit. The system classifies user intent, retrieves relevant chunks from a ChromaDB vector store, reasons with ReAct agents, and generates grounded responses with citations.

## Commands

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run backend (from repo root)
cd backend && uvicorn app:app --reload --port 8001

# Run frontend
cd frontend && streamlit run streamlit_app.py

# Ingest PDF documents into vector store
curl -X POST http://localhost:8001/ingest -H "Content-Type: application/json" \
  -d '{"data_dir": "./data"}'

# Run intent classification tests
python backend/test_intent_routes.py
```

## Environment Variables

Requires a `.env` file at repo root:
- `OPENAI_API_KEY` — OpenAI API key (uses gpt-5-nano)
- `HF_TOKEN` — HuggingFace token for embeddings

## Architecture

**Two-route LangGraph state machine** defined in `backend/rag_graph.py`:

- **Route A (GENERAL):** `classify_intent` → `answer_general` → END — no retrieval needed
- **Route B (RAG):** `classify_intent` → `retrieve` → `agent_reason` → `generate_grounded` → `eval_agent` → END — full agentic pipeline with ReAct loops

**Key modules:**
- `backend/app.py` — FastAPI endpoints (`POST /chat/stream` SSE, `POST /ingest`)
- `backend/rag_graph.py` — LangGraph graph with 6 nodes, 2 ReAct agents, MemorySaver for conversation history
- `backend/rag_store.py` — ChromaDB ingestion pipeline (pdfplumber + EasyOCR for scanned pages, HuggingFace all-MiniLM-L6-v2 embeddings)
- `backend/tools.py` — 8 agentic tools (listar, buscar_spec, por_marca, comparar, resumir, refinar, corregir, regenerar)
- `backend/prompts.py` — System prompts for classifier, generator, agent, and evaluator
- `backend/schemas.py` — Pydantic models (IntentClassification with structured output)
- `frontend/streamlit_app.py` — Streamlit chat UI with session management and traceability panel
- `app.py` (root) — ASGI wrapper that imports `backend/app.py` for `uvicorn app:app` from repo root

**Data flow:** PDFs organized by brand folder → chunked (1000 chars, 150 overlap) → enriched metadata (marca, modelo, doc_id, chunk_id, ocr flag, page) → ChromaDB. Queries get dynamic k (4-12) based on intent type.

## Design Decisions

- **Anti-hallucination:** Mandatory citation format `[1], [2]` with `**Fuentes:**` section; eval agent verifies grounding; model verification checks chunks match requested vehicle
- **Comparison queries:** Balanced retrieval (k/2 per model) with variant normalization (e.g., cx-5 ↔ Cx 5)
- **Streaming:** SSE via `astream_events()` for token-by-token delivery with progress events
- **Memory:** MemorySaver checkpointer tracks `last_model`/`last_make` per session_id for follow-up questions
- **Intent classification:** Pydantic structured output (Búsqueda/Resumen/Comparación/GENERAL) with LLM-suggested k overridable by keyword regex fallback

## Language

The codebase, prompts, and UI are in **Spanish**. Variable names and code comments mix Spanish and English.
