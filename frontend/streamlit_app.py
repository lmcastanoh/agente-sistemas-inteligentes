# frontend/streamlit_app.py
import requests
import streamlit as st

API_BASE = "http://localhost:8001"  # or 8000 if your backend is on 8000

st.title("RAG (LangChain + LangGraph)")

with st.sidebar:
    st.subheader("Ingest documents")
    data_dir = st.text_input("Data directory (txt files)", value="./data")
    if st.button("Ingest"):
        r = requests.post(
            f"{API_BASE}/ingest",
            json={"data_dir": data_dir},
            timeout=300
        )

        if r.headers.get("content-type") == "application/json":
            st.write(r.json())
        else:
            st.error(f"Backend error ({r.status_code}): {r.text}")

st.subheader("Ask a question")
q = st.text_input("Question")
go = st.button("Ask (stream)")

if go and q.strip():
    placeholder = st.empty()
    full = ""

    with requests.post(
        f"{API_BASE}/chat/stream",
        json={"question": q},
        stream=True,
        timeout=300,
        headers={"Accept": "text/event-stream"},
    ) as r:
        current_event = None
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                current_event = None
                continue
            if line.startswith("event: "):
                current_event = line[len("event: "):]
            elif line.startswith("data: ") and current_event == "token":
                full += line[len("data: "):]
                placeholder.markdown(full)