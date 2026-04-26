"""
supervisor_agent.py
───────────────────
LangGraph multi-agent supervisor.

Agent label fix
---------------
Previously only the first question showed the 'Using ...' indicator.
Root cause: in LangGraph supervisor with stream_mode="updates", the step
that contains a sub-agent's AIMessage with tool_calls can arrive BEFORE the
tool result step, and the sub-agent name is set on the message. Scanning
only for tool_calls in AIMessages missed turns where:
  1. The tool_call AIMessage was emitted under the supervisor node (routing),
     not the sub-agent node.
  2. The sub-agent node emitted its result as an AIMessage with name set to
     the agent (e.g. 'threat_agent') but no tool_calls key.

Fix strategy — three complementary label sources, checked in this order:
  A. Any node: AIMessage.tool_calls  → TOOL_LABELS (original, kept)
  B. Any node: ToolMessage presence  → NODE_TO_LABEL (new — a ToolMessage
     always follows a tool call, so its presence in the step confirms which
     agent/tool ran)
  C. Node name itself               → NODE_TO_LABEL (new — covers sub-agent
     nodes that have no tool_calls in their delta messages)

This ensures the indicator fires on question 1, 2, 3 … every time.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
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

# ── Label tables ───────────────────────────────────────────────────────────────────────────────────────

# Source A: MCP tool name  → UI label
TOOL_LABELS: dict[str, str] = {
    "search_knowledge_base": "\U0001f4da NIST / Framework Q&A",
    "lookup_cve":            "\U0001f50d CVE lookup",
    "check_ip":              "\U0001f310 IP reputation check",
    "check_breach":          "\U0001f511 Password breach check",
    "get_conversation_history": "\U0001f9e0 Conversation history",
}

# Source B/C: LangGraph node name  → UI label
# Used when:
#   B) A ToolMessage is present in the node's step (tool just ran)
#   C) The node itself activated (catches rag_agent which has one tool)
NODE_TO_LABEL: dict[str, str] = {
    "rag_agent":    "\U0001f4da NIST / Framework Q&A",
    "threat_agent": "\U0001f50d CVE / Threat Analysis",
    "audit_agent":  "\U0001f511 Security Audit",
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
    graph = workflow.compile()
    return graph, client


def _history_to_messages(session_id: str) -> list[dict]:
    messages = []
    for row in get_recent_messages(session_id, limit=6):
        role    = row.get("role")
        content = row.get("content") or ""
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


def _try_emit_label(
    node_name: str,
    msgs: list,
    announced: set[str],
) -> list[tuple[str, str]]:
    """
    Returns a list of ("agent", label) events to emit for this step.

    Three sources checked in priority order:
      A. AIMessage.tool_calls  → TOOL_LABELS   (most specific)
      B. ToolMessage present   → NODE_TO_LABEL  (tool just completed)
      C. Node name             → NODE_TO_LABEL  (node activated)
    """
    events: list[tuple[str, str]] = []

    def _emit(label: str) -> None:
        if label and label not in announced:
            announced.add(label)
            events.append(("agent", label))

    # A — scan every AIMessage in this step for tool_calls
    for msg in msgs:
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", []) or []:
            tool_name = (
                tc.get("name", "") if isinstance(tc, dict)
                else getattr(tc, "name", "")
            )
            if tool_name in TOOL_LABELS:
                _emit(TOOL_LABELS[tool_name])

    # B — if a ToolMessage is in the step, the tool ran; use node label
    has_tool_result = any(isinstance(m, ToolMessage) for m in msgs)
    if has_tool_result and node_name in NODE_TO_LABEL:
        _emit(NODE_TO_LABEL[node_name])

    # C — node itself activated; use node label as fallback
    if node_name in NODE_TO_LABEL:
        _emit(NODE_TO_LABEL[node_name])

    return events


async def stream(graph, messages: list[dict], session_id: str):
    """
    Async generator → yields ("agent", label) | ("text", chunk)

    Emits exactly one agent label per turn (the first one encountered),
    then streams the supervisor's final answer as text chunks.
    """
    # Reset per-turn tracking
    announced_agents: set[str] = set()
    supervisor_answer = ""

    history      = _history_to_messages(session_id)
    conversation = history + messages

    async for step in graph.astream(
        {"messages": conversation},
        stream_mode="updates",
    ):
        for node_name, node_state in step.items():
            msgs = node_state.get("messages", [])

            # ── Emit agent label ─────────────────────────────────────────────────
            for event in _try_emit_label(node_name, msgs, announced_agents):
                yield event

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
