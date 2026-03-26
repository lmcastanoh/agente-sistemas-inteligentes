import json
import re
import uuid

import requests
import streamlit as st

API_BASE = "http://localhost:8001"


def _clean_markdown(text: str) -> str:
    if not text:
        return ""

    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


st.set_page_config(page_title="RAG Agéntico - Fichas técnicas de vehículos", layout="wide")

st.markdown(
    """
    <style>
    /* ── Fondo general claro ── */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ecf5 100%);
        color: #1e293b;
    }
    .block-container {
        padding-top: 1.8rem;
        max-width: 980px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #162d4a 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background-color: #2563eb;
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #1d4ed8;
    }
    [data-testid="stSidebar"] input {
        background-color: #1e3a5f !important;
        border: 1px solid #3b6aa0 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }

    /* ── Título y caption ── */
    h1 {
        color: #1e3a5f !important;
        font-weight: 700 !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }

    /* ── Mensajes del chat ── */
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        font-size: 1rem !important;
        line-height: 1.55 !important;
        color: #1e293b !important;
    }

    /* ── Input del chat ── */
    [data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        color: #1e293b !important;
    }

    /* ── Expander trazabilidad ── */
    .streamlit-expanderHeader {
        background-color: #f0f4f8 !important;
        border-radius: 8px !important;
        color: #1e3a5f !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 0 0 8px 8px !important;
        color: #334155 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RAG Agéntico - Fichas técnicas de vehículos")
st.caption("Preguntas sobre fichas técnicas, comparaciones y recomendaciones.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Sesión")
    st.caption(f"session_id: {st.session_state.session_id}")
    if st.button("Nueva sesión", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.subheader("Ingesta de documentos")
    data_dir = st.text_input("Directorio de datos", value="./data")
    if st.button("Ingestar", use_container_width=True):
        try:
            r = requests.post(
                f"{API_BASE}/ingest",
                json={"data_dir": data_dir},
                timeout=300,
            )
            if r.headers.get("content-type", "").startswith("application/json"):
                st.success("Ingesta completada")
                st.json(r.json())
            else:
                st.error(f"Error backend ({r.status_code}): {r.text}")
        except Exception as exc:
            st.error(f"Fallo en ingesta: {exc}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Pregunta sobre autos...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        status_placeholder = st.empty()
        raw = ""
        trazabilidad_data: dict | None = None

        # Mapa de nombres de nodos a labels amigables para el usuario
        _NODE_LABELS = {
            "classify_intent": "Clasificando intención...",
            "retrieve": "Buscando documentos...",
            "agent_reason": "Razonando con herramientas...",
            "generate_grounded": "Generando respuesta...",
            "eval_agent": "Evaluando calidad...",
        }

        try:
            with requests.post(
                f"{API_BASE}/chat/stream",
                json={"question": prompt, "session_id": st.session_state.session_id},
                stream=True,
                timeout=300,
                headers={"Accept": "text/event-stream"},
            ) as r:
                if r.status_code != 200:
                    final_text = f"Error backend ({r.status_code}): {r.text}"
                    placeholder.error(final_text)
                else:
                    current_event = None
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            current_event = None
                            continue
                        if line.startswith("event: "):
                            current_event = line[len("event: ") :]
                        elif line.startswith("data: "):
                            data = line[len("data: ") :]
                            if current_event == "token":
                                status_placeholder.empty()
                                raw += data.replace("\\n", "\n")
                                placeholder.markdown(_clean_markdown(raw))
                            elif current_event == "progress":
                                label = _NODE_LABELS.get(data, data)
                                if not raw:
                                    status_placeholder.markdown(f"⏳ *{label}*")
                            elif current_event == "trazabilidad":
                                try:
                                    trazabilidad_data = json.loads(data)
                                except Exception:
                                    pass
                            elif current_event == "done":
                                status_placeholder.empty()
                                break

                    final_text = _clean_markdown(raw) if raw.strip() else "No se generó respuesta."
                    placeholder.markdown(final_text)
        except Exception as exc:
            final_text = f"Fallo de request: {exc}"
            placeholder.error(final_text)

        if trazabilidad_data:
            with st.expander("🔍 Trazabilidad de la respuesta"):
                ruta = trazabilidad_data.get("ruta") or trazabilidad_data.get("route") or []
                ruta_str = " -> ".join(ruta) if ruta else "Sin ruta disponible"
                st.markdown(f"**Ruta del grafo:** `{ruta_str}`")

                cls = trazabilidad_data.get("clasificacion") or trazabilidad_data.get("intent_json") or {}
                if cls:
                    st.markdown(
                        f"**Intención:** `{cls.get('intent', cls.get('intencion'))}` "
                        f"&nbsp;|&nbsp; **Requiere RAG:** `{cls.get('needs_retrieval', cls.get('requiere_rag'))}`"
                    )
                chunks = trazabilidad_data.get("chunks_recuperados") or trazabilidad_data.get("retrieved_chunks") or []
                if chunks:
                    k = trazabilidad_data.get("k_utilizado", trazabilidad_data.get("retrieval_k", "-"))
                    st.markdown(f"**Chunks recuperados:** {len(chunks)} (k={k})")
                    rows = [
                        f"- `{c.get('source', '')}` p.{c.get('page', '')} "
                        for c in chunks
                    ]
                    st.markdown("\n".join(rows))

                agent_steps = trazabilidad_data.get("agent_steps", [])
                if agent_steps:
                    st.markdown(f"**Pasos del agente ReAct:** {len(agent_steps)}")
                    tools_used = trazabilidad_data.get("tools_used", [])
                    if tools_used:
                        st.markdown(f"**Herramientas usadas:** {', '.join(tools_used)}")
                    for step in agent_steps:
                        if step.get("type") == "tool_call":
                            st.markdown(f"  - Paso {step['step']}: `{step['tool']}`({step.get('args', {})})")
                        elif step.get("type") == "final_reasoning":
                            st.markdown(f"  - Paso {step['step']}: Razonamiento final")

                eval_steps = trazabilidad_data.get("eval_steps", [])
                if eval_steps:
                    modified = trazabilidad_data.get("eval_modified", False)
                    st.markdown(f"**Evaluación:** {'Corregida' if modified else 'Aprobada sin cambios'}")
                    for step in eval_steps:
                        if step.get("type") == "correction":
                            st.markdown(f"  - Paso {step['step']}: `{step['tool']}`")
                        elif step.get("type") == "approved":
                            st.markdown(f"  - Paso {step['step']}: Aprobada")


    st.session_state.messages.append({"role": "assistant", "content": final_text})
