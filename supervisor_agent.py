"""
supervisor_agent.py
───────────────────
LangGraph multi-agent supervisor.

Refactored in this version
--------------------------
- Removed MilvusCheckpointer dependency entirely.
  The graph is compiled WITHOUT a checkpointer — LangGraph no longer owns state.
- Conversation history is fetched explicitly from agents.chat_memory
  (Milvus chat_memory collection) before each graph.astream() call.
- History is prepended to the message list so the supervisor LLM has full
  context without any checkpoint replay bugs.
- api.py is responsible for writing each turn (user + assistant) to Milvus
  via agents.chat_memory.append_message().
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph_supervisor import create_supervisor

from agents.chat_memory import get_recent_messages
from agents.rag_agent import create_rag_agent
from agents.threat_agent import create_threat_agent
from agents.audit_agent import create_audit_agent

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL  = os.getenv("OLLAMA_URL",       "http://localhost:11434")
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "qwen2.5:3b")

MCP_SERVER_PATH = str(Path(__file__).resolve().parent / "mcp_rag_server.py")

MCP_ENV = {
    k: os.environ[k]
    for k in ("ABUSEIPDB_API_KEY", "MILVUS_URI", "OLLAMA_URL")
    if k in os.environ
}

# ── Label tables ──────────────────────────────────────────────────────────────────────────────────────

TOOL_LABELS: dict[str, str] = {
    "search_knowledge_base": "\U0001f4da NIST / Framework Q&A",
    "lookup_cve":            "\U0001f50d CVE lookup",
    "check_ip":              "\U0001f310 IP reputation check",
    "check_breach":          "\U0001f511 Password breach check",
}

NODE_FALLBACK_LABELS: dict[str, str] = {
    "rag_agent": "\U0001f4da NIST / Framework Q&A",
}


SUPERVISOR_PROMPT = """You are the Cybersecurity AI Supervisor coordinating a team
of specialised security agents. Your role is to:

1. Understand the user's cybersecurity question or task
2. Route it to the most appropriate agent (one or more in sequence if needed)
3. Synthesise the agents' findings into a clear, actionable final answer

Routing rules — delegate to:
  • rag_agent      → questions about NIST CSF, frameworks, best practices,
                     incident response procedures, compliance, governance
  • threat_agent   → specific CVE IDs, product vulnerabilities, suspicious IP
                     addresses, active threat analysis
  • audit_agent    → password breach checks, credential hygiene, access policy

Multi-agent situations — use both in order when:
  • A CVE question also needs framework guidance (threat_agent → rag_agent)
  • A breach check needs remediation policy (audit_agent → rag_agent)

Context awareness:
  • Short conversation history is prepended as earlier messages in this thread
  • Use prior turns to resolve pronouns and follow-up questions
  • Do not re-explain things already covered unless the user explicitly asks

Scope enforcement:
  • If the question is completely unrelated to cybersecurity, politely decline
    and explain your focus area. Do not route to any agent.

Always present the final answer in a structured, readable format.
"""


async def build_supervisor_graph():
    mcp_config = {
        "cybersec_tools": {
            "command":   sys.executable,
            "args":      [MCP_SERVER_PATH],
            "transport": "stdio",
            "env":       MCP_ENV if MCP_ENV else None,
        }
    }

    client = MultiServerMCPClient(mcp_config)
    tools  = await client.get_tools()
    print(f"✅ Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")

    rag_agent    = create_rag_agent(tools)
    threat_agent = create_threat_agent(tools)
    audit_agent  = create_audit_agent(tools)

    supervisor_llm = ChatOllama(
        model=SUPERVISOR_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=4096,
    )
    print(f"✅ Supervisor model: {SUPERVISOR_MODEL}")

    workflow = create_supervisor(
        agents=[rag_agent, threat_agent, audit_agent],
        model=supervisor_llm,
        prompt=SUPERVISOR_PROMPT,
    )
    # Compile WITHOUT a checkpointer — history is managed via agents.chat_memory
    graph = workflow.compile()
    return graph, client


def _history_to_messages(session_id: str) -> list[dict]:
    """
    Fetch recent turns from Milvus and convert to the role/content dict
    format expected by LangGraph.
    """
    messages = []
    for row in get_recent_messages(session_id, limit=6):
        role    = row.get("role")
        content = row.get("content") or ""
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


async def stream(graph, messages: list[dict], session_id: str):
    """
    Async generator → yields ("agent", label) | ("text", chunk)

    History injection
    -----------------
    Recent turns are fetched from Milvus and prepended to `messages` before
    calling graph.astream(). The graph runs statelessly each time — no
    checkpoint replay, no shadowy-previous-answer bugs.
    """
    announced_agents: set[str] = set()
    supervisor_answer = ""

    # Inject history BEFORE the current user message
    history     = _history_to_messages(session_id)
    conversation = history + messages

    async for step in graph.astream(
        {"messages": conversation},
        stream_mode="updates",
    ):
        for node_name, node_state in step.items():
            msgs = node_state.get("messages", [])

            # ── Tool-call labels (threat_agent / audit_agent) ────────────────────
            for msg in msgs:
                if not isinstance(msg, AIMessage):
                    continue
                for tc in getattr(msg, "tool_calls", []) or []:
                    tool_name = (
                        tc.get("name", "") if isinstance(tc, dict)
                        else getattr(tc, "name", "")
                    )
                    if tool_name in TOOL_LABELS:
                        label = TOOL_LABELS[tool_name]
                        if label not in announced_agents:
                            announced_agents.add(label)
                            yield ("agent", label)

            # ── Node-level fallback (rag_agent only) ────────────────────────
            if node_name in NODE_FALLBACK_LABELS:
                fallback = NODE_FALLBACK_LABELS[node_name]
                if fallback not in announced_agents:
                    announced_agents.add(fallback)
                    yield ("agent", fallback)

            # ── Supervisor final answer ─────────────────────────────────────────
            if node_name != "supervisor":
                continue

            for msg in reversed(msgs):
                if not isinstance(msg, AIMessage):
                    continue
                sender = getattr(msg, "name", None) or ""
                if sender in {"rag_agent", "threat_agent", "audit_agent"}:
                    continue

                content = msg.content or ""
                if isinstance(content, list):
                    content = "".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                if not content:
                    break

                if content != supervisor_answer:
                    delta = content[len(supervisor_answer):]
                    if delta:
                        yield ("text", delta)
                    supervisor_answer = content
                break


async def get_history(session_id: str) -> list[dict]:
    """Return recent messages for a session (used by GET /history)."""
    return _history_to_messages(session_id)
