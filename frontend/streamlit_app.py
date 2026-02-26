import re
import uuid

import requests
import streamlit as st

API_BASE = "http://localhost:8001"


def _clean_markdown(text: str) -> str:
    """Normalize imperfect streamed markdown into readable UI content."""
    if not text:
        return ""

    out = text.replace("\r\n", "\n").replace("\r", "\n")

    # Convert markdown headings to normal text to avoid giant rendering.
    out = re.sub(r"(?m)^\s*#{1,6}\s*(.+)$", r"\1", out)
    out = re.sub(r"(?m)^(\S+?)##\s*", r"\1\n", out)

    # Break common glued bullet patterns.
    out = out.replace(".-", ".\n- ")
    out = re.sub(r"(?<!\n)-\s+", "\n- ", out)

    # Add line breaks before common section labels if they are glued together.
    labels = [
        "Motor y transmision:",
        "Rendimiento y consumo:",
        "Dimensiones y capacidades:",
        "Equipamiento destacado:",
        "Versiones disponibles:",
        "Datos faltantes:",
    ]
    for label in labels:
        out = re.sub(
            rf"(?i)(?<!\n)({re.escape(label)})",
            r"\n**\1**",
            out,
        )

    # Convert section-like labels into bullets for compact reading.
    out = re.sub(
        r"(?im)^\s*(motor y transmision|rendimiento y consumo|dimensiones y capacidades|equipamiento destacado|versiones disponibles|datos faltantes)\s*:\s*",
        r"- **\1:** ",
        out,
    )

    # Normalize excessive spacing.
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


st.set_page_config(page_title="RAG Cars", layout="wide")

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
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4,
    [data-testid="stChatMessage"] h5,
    [data-testid="stChatMessage"] h6 {
        font-size: 1.1rem !important;
        line-height: 1.45 !important;
        margin: 0.25rem 0 !important;
        font-weight: 700 !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        font-size: 1rem !important;
        line-height: 1.55 !important;
    }
    [data-testid="stChatMessage"] ul {
        margin-top: 0.25rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RAG Cars Assistant")
st.caption("Preguntas sobre fichas tecnicas, comparaciones y recomendaciones.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Session")
    st.caption(f"session_id: {st.session_state.session_id}")
    if st.button("New session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.subheader("Ingest documents")
    data_dir = st.text_input("Data directory", value="./data")
    if st.button("Ingest", use_container_width=True):
        try:
            r = requests.post(
                f"{API_BASE}/ingest",
                json={"data_dir": data_dir},
                timeout=300,
            )
            if r.headers.get("content-type", "").startswith("application/json"):
                st.success("Ingest completed")
                st.json(r.json())
            else:
                st.error(f"Backend error ({r.status_code}): {r.text}")
        except Exception as exc:
            st.error(f"Ingest request failed: {exc}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about cars...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        raw = ""

        try:
            with requests.post(
                f"{API_BASE}/chat/stream",
                json={"question": prompt, "session_id": st.session_state.session_id},
                stream=True,
                timeout=300,
                headers={"Accept": "text/event-stream"},
            ) as r:
                if r.status_code != 200:
                    final_text = f"Backend error ({r.status_code}): {r.text}"
                    placeholder.error(final_text)
                else:
                    current_event = None
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            current_event = None
                            continue
                        if line.startswith("event: "):
                            current_event = line[len("event: ") :]
                        elif line.startswith("data: ") and current_event == "token":
                            raw += line[len("data: ") :]
                            placeholder.markdown(_clean_markdown(raw))

                    final_text = _clean_markdown(raw) if raw.strip() else "No response generated."
                    placeholder.markdown(final_text)
        except Exception as exc:
            final_text = f"Request failed: {exc}"
            placeholder.error(final_text)

    st.session_state.messages.append({"role": "assistant", "content": final_text})
