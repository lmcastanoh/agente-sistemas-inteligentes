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


st.set_page_config(page_title="RAG Autos", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px 700px at 30% -10%, #1b2c4a 0%, #0a1220 45%, #070b12 100%);
    }
    .block-container {
        padding-top: 1.8rem;
        max-width: 980px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2430 0%, #1a1f2a 100%);
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        font-size: 1rem !important;
        line-height: 1.55 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RAG Autos")
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
