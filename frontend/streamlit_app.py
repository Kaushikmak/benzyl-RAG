import os
import uuid
from datetime import datetime

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Assistant", page_icon="RAG", layout="wide")

st.markdown(
    """
<style>
:root {
  --bg: #f3f3f3;
  --panel: #ffffff;
  --line: #cfcfcf;
  --text: #111111;
  --muted: #555555;
}
.stApp { background: var(--bg); color: var(--text); }
.block-container { max-width: 980px; }
[data-testid="stSidebar"] { background: #e9e9e9; }
.stChatMessage { border: 1px solid var(--line); border-radius: 8px; padding: 0.8rem; background: var(--panel); }
.source-chip { display:inline-block; border:1px solid #222; color:#111; padding:0.1rem 0.45rem; border-radius:999px; font-size:0.8rem; margin-right:0.35rem; }
.muted { color: var(--muted); font-size:0.8rem; }
</style>
""",
    unsafe_allow_html=True,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_source" not in st.session_state:
    st.session_state.selected_source = None


def query_rag(question: str, verbose: bool, use_web: bool):
    r = requests.post(
        f"{API_URL}/query",
        json={
            "query": question,
            "session_id": st.session_state.session_id,
            "verbose": verbose,
            "use_web": use_web,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def send_feedback(answer_id: str, thumb: str, reason_tags=None):
    payload = {
        "answer_id": answer_id,
        "session_id": st.session_state.session_id,
        "thumb": thumb,
        "reason_tags": reason_tags or [],
        "note": "",
    }
    requests.post(f"{API_URL}/feedback", json=payload, timeout=30)


def fetch_source(path: str):
    r = requests.get(f"{API_URL}/source", params={"path": path}, timeout=30)
    r.raise_for_status()
    return r.json()


def render_sources(idx: int, sources: list[str]):
    st.markdown("**Sources**")
    for src in sources:
        label = os.path.basename(src) if os.path.exists(src) else src
        if os.path.exists(src):
            if st.button(f"Open {label}", key=f"src_{idx}_{src}"):
                st.session_state.selected_source = src
        else:
            st.markdown(f"- {label}")


with st.sidebar:
    st.title("RAG")
    verbose = st.toggle("Verbose", value=False)
    use_web = st.toggle("Use Web", value=False)
    st.caption(f"Session `{st.session_state.session_id[:8]}`")

st.title("RAG Assistant")
st.caption("Local-first answers with optional trusted web augmentation")

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.markdown(f"<div class='muted'>{message['timestamp']}</div>", unsafe_allow_html=True)
            if message.get("sources"):
                render_sources(i, message["sources"])
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Helpful", key=f"up_{i}"):
                    send_feedback(message.get("answer_id", ""), "up")
            with c2:
                if st.button("Not Helpful", key=f"down_{i}"):
                    send_feedback(message.get("answer_id", ""), "down", ["irrelevant"])

if st.session_state.selected_source:
    with st.expander("Source Preview", expanded=True):
        try:
            doc = fetch_source(st.session_state.selected_source)
            st.markdown(f"**{doc['name']}**")
            st.code(doc["content"], language="markdown")
            if st.button("Close Preview"):
                st.session_state.selected_source = None
                st.rerun()
        except Exception as e:
            st.error(f"Unable to load source: {e}")

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()}
    )
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = query_rag(prompt, verbose=verbose, use_web=use_web)
                st.markdown(response["answer"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "timestamp": datetime.now().isoformat(),
                        "sources": response.get("sources", []),
                        "answer_id": response.get("answer_id"),
                    }
                )
            except Exception as e:
                st.error(f"Request failed: {e}")
