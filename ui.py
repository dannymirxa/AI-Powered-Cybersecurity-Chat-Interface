"""
ui.py
─────
Streamlit chat interface for the CyberSec AI assistant.

Changes in this version:
  1. Agent routing shown inline (e.g. "📚 Using NIST / Framework Q&A")
  2. Session displayed as first-message summary, not raw UUID
  3. Sidebar shows all previous sessions from Milvus — click to switch
  4. Fresh session auto-created on every page load/refresh
  5. Fixed SSE parsing to handle 'event:' lines before 'data:' lines
  6. Added delete button (🗑) per previous session in sidebar
"""

import json
import os
import re
import uuid

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────────

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CyberSec AI",
    page_icon="🛡️",
    layout="wide",
)

# ── Session state defaults ───────────────────────────────────────────────────────
# Always start with a fresh session. If user wants to resume, they click
# a previous session from the sidebar list.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_summary" not in st.session_state:
    st.session_state.session_summary = "New conversation"
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL


# ── Helpers ────────────────────────────────────────────────────────────────────────

def api(path: str) -> str:
    return st.session_state.api_url.rstrip("/") + path


def _extract_sources(text: str) -> list[str]:
    return re.findall(r"\*\*Source\*\*[:\s]+(.+?)(?:\n|$)", text)


def _clean_answer(text: str) -> str:
    return re.sub(r"\*\*Source\*\*[:\s]+.+?(\n|$)", "", text).strip()


def _fetch_sessions() -> list[dict]:
    """Fetch all sessions from the backend. Returns [] on error."""
    try:
        resp = requests.get(api("/sessions"), timeout=5)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return []


def _switch_session(session_id: str, summary: str) -> None:
    """Switch to an existing session and load its history."""
    st.session_state.session_id      = session_id
    st.session_state.session_summary = summary or session_id[:20] + "..."
    st.session_state.messages = []
    try:
        resp = requests.get(api(f"/history/{session_id}"), timeout=10)
        if resp.ok:
            data = resp.json()
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in data.get("messages", [])
                if m["role"] in ("user", "assistant")
            ]
    except Exception:
        pass
    st.rerun()


def _delete_session(session_id: str) -> None:
    """
    Call DELETE /session/{id} on the backend, then — if the deleted session
    was the active one — start a fresh chat.  Always triggers a rerun so the
    sidebar list refreshes.
    """
    try:
        requests.delete(api(f"/session/{session_id}"), timeout=5)
    except Exception:
        pass  # best-effort: still clear locally even if backend unreachable

    if st.session_state.session_id == session_id:
        # Active session was deleted — start fresh
        st.session_state.session_id      = str(uuid.uuid4())
        st.session_state.session_summary = "New conversation"
        st.session_state.messages        = []

    st.rerun()


def new_chat() -> None:
    """Create a brand-new session (does NOT delete the old one from Milvus)."""
    st.session_state.session_id      = str(uuid.uuid4())
    st.session_state.session_summary = "New conversation"
    st.session_state.messages        = []
    st.rerun()


# ── Custom CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  section[data-testid="stSidebar"] { background: #0f1117; }

  .agent-indicator {
    background: #1a2a1a;
    border-left: 3px solid #4caf50;
    color: #a5d6a7;
    font-size: 0.8rem;
    padding: 4px 10px;
    margin: 4px 0;
    border-radius: 0 6px 6px 0;
  }
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
  .tool-indicator {
    color: #ffd166;
    font-size: 0.82rem;
    font-style: italic;
  }
  .session-item {
    background: #1a1a2e;
    border: 1px solid #2a2a3e;
    border-radius: 6px;
    padding: 6px 10px;
    margin: 4px 0;
    cursor: pointer;
    font-size: 0.82rem;
    color: #ccc;
  }
  .session-item:hover { border-color: #00b4d8; color: #fff; }
  .session-active {
    border-color: #00b4d8 !important;
    background: #0d2a3a !important;
    color: #90e0ef !important;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
  .cursor { animation: blink 1s step-start infinite; }
  .block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ CyberSec AI")
    st.caption("Powered by LangGraph + Ollama")
    st.divider()

    # API URL
    new_url = st.text_input("API URL", value=st.session_state.api_url,
                            help="Base URL of the FastAPI backend")
    if new_url != st.session_state.api_url:
        st.session_state.api_url = new_url

    st.divider()

    # Current session summary + New Chat button
    st.markdown("**Current Session**")
    st.info(f"💬 {st.session_state.session_summary}")

    col_new, col_del_cur = st.columns([3, 1])
    with col_new:
        if st.button("🗑 New Chat", use_container_width=True):
            new_chat()
    with col_del_cur:
        if st.button(
            "❌",
            key="del_current",
            help="Delete current session",
            use_container_width=True,
        ):
            _delete_session(st.session_state.session_id)

    st.divider()

    # Previous sessions list
    st.markdown("**Previous Sessions**")
    sessions = _fetch_sessions()
    # Filter out current session so it doesn't appear in the list
    prev_sessions = [s for s in sessions if s["session_id"] != st.session_state.session_id]

    if not prev_sessions:
        st.caption("No previous sessions found.")
    else:
        for s in prev_sessions:
            label   = s.get("summary") or s["session_id"][:24] + "..."
            display = label[:40] + "…" if len(label) > 40 else label
            sid     = s["session_id"]

            # Each session row: [chat button (wide)] [delete button (narrow)]
            col_chat, col_del = st.columns([5, 1])
            with col_chat:
                if st.button(
                    f"💬 {display}",
                    key=f"switch_{sid}",
                    use_container_width=True,
                ):
                    _switch_session(sid, label)
            with col_del:
                if st.button(
                    "🗑",
                    key=f"del_{sid}",
                    help=f"Delete: {label}",
                    use_container_width=True,
                ):
                    _delete_session(sid)

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


# ── Main header ──────────────────────────────────────────────────────────────────────

st.markdown("# 🛡️ CyberSec AI Assistant")
st.caption("Ask me about vulnerabilities, frameworks, incident response, and more.")
st.divider()


# ── Render existing messages ───────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    role    = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])
    agents_used = msg.get("agents_used", [])

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🛡️"):
            # Show which agents were called
            for label in agents_used:
                st.markdown(
                    f'<div class="agent-indicator">✨ Using {label}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(_clean_answer(content))
            if sources:
                badges = " ".join(
                    f'<span class="source-badge">📄 {s}</span>' for s in sources
                )
                st.markdown(f"**Sources:** {badges}", unsafe_allow_html=True)


# ── Chat input ──────────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a cybersecurity question ..."):
    prompt = prompt.strip()
    if not prompt:
        st.stop()

    # Update session summary on the very first message
    if not st.session_state.messages:
        st.session_state.session_summary = prompt[:80] + ("…" if len(prompt) > 80 else "")

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── Stream assistant response ────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🛡️"):
        status_placeholder = st.empty()
        agents_placeholder = st.empty()
        answer_placeholder = st.empty()

        status_placeholder.markdown(
            '<span class="tool-indicator">⚙️ Thinking ...</span>',
            unsafe_allow_html=True,
        )

        full_answer        = ""
        agents_used: list[str] = []
        received_session   = st.session_state.session_id
        sources: list[str] = []
        current_event      = ""   # tracks the most recent SSE event: label

        try:
            with requests.post(
                api("/chat/stream"),
                json={"message": prompt, "session_id": st.session_state.session_id},
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        current_event = ""  # blank line resets event type
                        continue

                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

                    if line.startswith("event:"):
                        current_event = line[len("event:"):].strip()
                        continue

                    if not line.startswith("data:"):
                        continue

                    data = line[len("data:"):].strip()

                    # ── Handle by event type ─────────────────────────────────
                    if current_event == "done":
                        try:
                            parsed = json.loads(data)
                            received_session = parsed.get("session_id", received_session)
                        except Exception:
                            pass
                        break

                    if current_event == "error":
                        try:
                            parsed = json.loads(data)
                            full_answer += f"\n\n⚠️ {parsed.get('error', data)}"
                        except Exception:
                            full_answer += f"\n\n⚠️ {data}"
                        break

                    if current_event == "agent":
                        try:
                            parsed = json.loads(data)
                            label  = parsed.get("label", "")
                            if label and label not in agents_used:
                                agents_used.append(label)
                                # Show agent indicators accumulated so far
                                indicators = "".join(
                                    f'<div class="agent-indicator">✨ Using {lbl}</div>'
                                    for lbl in agents_used
                                )
                                agents_placeholder.markdown(indicators, unsafe_allow_html=True)
                        except Exception:
                            pass
                        continue

                    # Default: text chunk
                    chunk        = data.replace("\\n", "\n")
                    full_answer += chunk
                    status_placeholder.empty()   # clear "Thinking..."
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

        # Final render
        status_placeholder.empty()
        sources = _extract_sources(full_answer)
        answer_placeholder.markdown(_clean_answer(full_answer))
        if sources:
            badges = " ".join(f'<span class="source-badge">📄 {s}</span>' for s in sources)
            st.markdown(f"**Sources:** {badges}", unsafe_allow_html=True)

        st.session_state.session_id = received_session

    st.session_state.messages.append({
        "role":        "assistant",
        "content":     full_answer,
        "sources":     sources,
        "agents_used": agents_used,
    })
