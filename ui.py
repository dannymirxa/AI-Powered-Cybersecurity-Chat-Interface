"""
ui.py
─────
Streamlit chat interface for the CyberSec AI assistant.

Features:
  • Connects to the FastAPI backend at API_URL (env var or sidebar input)
  • Streams AI responses token-by-token via SSE
  • Persists the session_id in st.session_state so conversation history
    survives page re-renders within the same browser tab
  • "New Chat" button clears the session on the backend and resets UI
  • Renders AI responses as Markdown (code blocks, lists, headings)
  • Shows a tool-use indicator when the agent is working
  • Displays source attribution badges when RAG context was used

Usage:
    streamlit run ui.py

Environment variables:
    API_URL   Base URL of the FastAPI server  (default: http://localhost:8000)
"""

import json
import os
import re
import uuid

import requests
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────────────────

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CyberSec AI",
    page_icon="🛡️",
    layout="wide",
)

# ── Session state defaults ──────────────────────────────────────────────────────

if "session_id"   not in st.session_state:
    st.session_state.session_id   = str(uuid.uuid4())
if "messages"     not in st.session_state:
    st.session_state.messages     = []      # list of {role, content}
if "api_url"      not in st.session_state:
    st.session_state.api_url      = DEFAULT_API_URL


# ── Helpers ─────────────────────────────────────────────────────────────────────

def api(path: str) -> str:
    return st.session_state.api_url.rstrip("/") + path


def _extract_sources(text: str) -> list[str]:
    """
    Pull source document names out of the agent's answer.
    The RAG agent returns lines like:  **Source**: some_doc.md
    """
    return re.findall(r"\*\*Source\*\*[:\s]+(.+?)(?:\n|$)", text)


def _clean_answer(text: str) -> str:
    """Remove the raw source lines from the answer — we render them as badges."""
    return re.sub(r"\*\*Source\*\*[:\s]+.+?(\n|$)", "", text).strip()


def new_chat():
    """Reset the UI and ask the backend to drop the session's checkpoints."""
    sid = st.session_state.session_id
    try:
        requests.delete(api(f"/session/{sid}"), timeout=5)
    except Exception:
        pass  # Backend might be down; still clear the UI
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages   = []
    st.rerun()


def restore_history():
    """
    Load conversation history from the backend for the current session_id.
    Useful when the user pastes a session ID to resume a previous conversation.
    """
    sid = st.session_state.session_id
    try:
        resp = requests.get(api(f"/history/{sid}"), timeout=10)
        if resp.ok:
            data = resp.json()
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in data.get("messages", [])
                if m["role"] in ("user", "assistant")
            ]
    except Exception:
        pass


# ── Custom CSS ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Sidebar */
  section[data-testid="stSidebar"] { background: #0f1117; }

  /* Chat bubbles */
  .user-bubble {
    background: #1e3a5f;
    border-radius: 12px 12px 2px 12px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    max-width: 80%;
    margin-left: auto;
    color: #e8eaf6;
  }
  .ai-bubble {
    background: #1a1a2e;
    border-radius: 12px 12px 12px 2px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    max-width: 90%;
    color: #e0e0e0;
    border-left: 3px solid #00b4d8;
  }

  /* Source badges */
  .source-badge {
    display: inline-block;
    background: #0d3b4f;
    color: #90e0ef;
    border: 1px solid #00b4d8;
    border-radius: 4px;
    font-size: 0.72rem;
    padding: 1px 7px;
    margin: 2px 3px;
  }

  /* Tool indicator */
  .tool-indicator {
    color: #ffd166;
    font-size: 0.82rem;
    font-style: italic;
  }

  /* Typing cursor blink */
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
  .cursor { animation: blink 1s step-start infinite; }

  /* Remove Streamlit default top padding */
  .block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ CyberSec AI")
    st.caption("Powered by LangGraph + Ollama")
    st.divider()

    # API URL setting
    new_url = st.text_input(
        "API URL",
        value=st.session_state.api_url,
        help="Base URL of the FastAPI backend",
    )
    if new_url != st.session_state.api_url:
        st.session_state.api_url = new_url

    st.divider()

    # Session management
    st.markdown("**Session**")
    st.code(st.session_state.session_id[:18] + "...", language=None)

    resume_id = st.text_input(
        "Resume session ID",
        placeholder="Paste a previous session ID",
        help="Enter a prior session ID to continue that conversation.",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Resume", use_container_width=True) and resume_id.strip():
            st.session_state.session_id = resume_id.strip()
            restore_history()
            st.rerun()
    with col2:
        if st.button("🗑 New Chat", use_container_width=True):
            new_chat()

    st.divider()
    st.markdown("""
    **Capabilities**
    - 📚 NIST / framework Q&A
    - 🔍 CVE lookup
    - 🌐 IP reputation check
    - 🔑 Password breach check
    - 🔒 Off-topic rejection
    """)
    st.divider()
    st.caption("Scope: cybersecurity topics only")


# ── Main header ─────────────────────────────────────────────────────────────────

st.markdown("# 🛡️ CyberSec AI Assistant")
st.caption("Ask me about vulnerabilities, frameworks, incident response, and more.")
st.divider()


# ── Render existing messages ────────────────────────────────────────────────────

for msg in st.session_state.messages:
    role    = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🛡️"):
            st.markdown(_clean_answer(content))
            if sources:
                badges = " ".join(
                    f'<span class="source-badge">📄 {s}</span>'
                    for s in sources
                )
                st.markdown(f"**Sources:** {badges}", unsafe_allow_html=True)


# ── Chat input ──────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a cybersecurity question ..."):
    prompt = prompt.strip()
    if not prompt:
        st.stop()

    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── Stream assistant response ────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🛡️"):
        # Loading indicator before first token
        status_placeholder = st.empty()
        status_placeholder.markdown(
            '<span class="tool-indicator">⚙️ Thinking ...</span>',
            unsafe_allow_html=True,
        )

        answer_placeholder = st.empty()
        full_answer        = ""
        received_session   = st.session_state.session_id
        sources: list[str] = []

        try:
            with requests.post(
                api("/chat/stream"),
                json={
                    "message":    prompt,
                    "session_id": st.session_state.session_id,
                },
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

                    # SSE event lines
                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip()
                        continue
                    if not line.startswith("data:"):
                        continue

                    data = line[len("data:"):].strip()

                    # Sentinel: done event
                    try:
                        parsed = json.loads(data)
                        if "session_id" in parsed:
                            received_session = parsed["session_id"]
                            break
                        if "error" in parsed:
                            full_answer += f"\n\n⚠️ {parsed['error']}"
                            break
                    except json.JSONDecodeError:
                        pass

                    # Regular text chunk — unescape \n back to newlines
                    chunk       = data.replace("\\n", "\n")
                    full_answer += chunk

                    # Clear loading indicator on first token
                    status_placeholder.empty()

                    # Show tool-use indicator if agent is still routing
                    if full_answer.strip() == "":
                        status_placeholder.markdown(
                            '<span class="tool-indicator">🔧 Using tools ...</span>',
                            unsafe_allow_html=True,
                        )

                    # Render incrementally
                    answer_placeholder.markdown(
                        _clean_answer(full_answer) + "<span class='cursor'>▌</span>",
                        unsafe_allow_html=True,
                    )

        except requests.exceptions.ConnectionError:
            full_answer = (
                "⚠️ Cannot reach the backend. "
                f"Make sure the FastAPI server is running at `{st.session_state.api_url}`."
            )
        except Exception as exc:
            full_answer = f"⚠️ Error: {exc}"

        # Final render — clean cursor, proper markdown
        status_placeholder.empty()
        sources = _extract_sources(full_answer)
        answer_placeholder.markdown(_clean_answer(full_answer))
        if sources:
            badges = " ".join(
                f'<span class="source-badge">📄 {s}</span>' for s in sources
            )
            st.markdown(f"**Sources:** {badges}", unsafe_allow_html=True)

        # Update session_id in case backend minted a new one
        st.session_state.session_id = received_session

    # Persist to session state
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_answer,
        "sources": sources,
    })
