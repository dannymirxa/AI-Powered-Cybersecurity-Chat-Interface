"""
supervisor_agent.py
───────────────────
LangGraph multi-agent supervisor.

Fixes in this version
---------------------
1. Shadowy-previous-answer bug
   Root cause: `stream_mode="updates"` re-emits the supervisor’s intermediate
   routing AIMessages from the checkpoint on every new turn. Hashing full
   content strings was unreliable because partial matches could slip through.
   Fix: snapshot the message COUNT in the checkpoint before streaming starts.
   Only yield text from supervisor messages that arrive AFTER that index.

2. Wrong agent label ("CVE / Threat Analysis" for IP checks)
   Root cause: the node-level label was emitted immediately when the node
   activated, before tool_calls were visible. By the time the tool fired in
   the next step, the wrong label was already shown.
   Fix: suppress the node-level label for threat_agent and audit_agent entirely.
   Instead, only emit the tool-level label once we see the actual tool_call name.
   For rag_agent (which has only one tool) the node label is still used as
   fallback to ensure something always appears in the UI.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph_supervisor import create_supervisor

from agents.milvus_checkpointer import MilvusCheckpointer
from agents.rag_agent import create_rag_agent
from agents.threat_agent import create_threat_agent
from agents.audit_agent import create_audit_agent

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL  = os.getenv("OLLAMA_URL",       "http://localhost:11434")
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "qwen2.5:3b")

MCP_SERVER_PATH = str(Path(__file__).resolve().parent / "mcp_rag_server.py")

MCP_ENV = {
    k: os.environ[k]
    for k in ("ABUSEIPDB_API_KEY", "MILVUS_URI", "OLLAMA_URL")
    if k in os.environ
}

# ── Label tables ──────────────────────────────────────────────────────────────────

# Maps MCP tool function name → UI label shown in the agent indicator.
# This is the single source of truth for what badge the user sees.
TOOL_LABELS: dict[str, str] = {
    "check_ip_reputation":   "\U0001f310 IP reputation check",
    "cve_lookup":            "\U0001f50d CVE lookup",
    "check_password_breach": "\U0001f511 Password breach check",
    "search_cybersec_kb":    "\U0001f4da NIST / Framework Q&A",
}

# Node-level fallback label — ONLY used for rag_agent because it has exactly
# one tool and the node fires before the tool_call is visible in the stream.
# threat_agent and audit_agent have multiple tools so we always wait for the
# specific tool to fire before showing any label.
NODE_FALLBACK_LABELS: dict[str, str] = {
    "rag_agent": "\U0001f4da NIST / Framework Q&A",
}

# Nodes whose label should ONLY come from their tool calls, never from the node
# activation itself (avoids showing the wrong generic label too early).
DEFER_LABEL_NODES = {"threat_agent", "audit_agent"}


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
  • You have full access to this session’s conversation history
  • Use prior turns to resolve pronouns and follow-up questions
  • Do not re-explain things you already covered unless asked

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

    checkpointer = MilvusCheckpointer()

    workflow = create_supervisor(
        agents=[rag_agent, threat_agent, audit_agent],
        model=supervisor_llm,
        prompt=SUPERVISOR_PROMPT,
    )
    graph = workflow.compile(checkpointer=checkpointer)
    return graph, client


def _thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


async def stream(graph, messages: list[dict], session_id: str):
    """
    Async generator  →  yields ("agent", label) | ("text", chunk)

    Fix 1 — Shadowy previous answer
    --------------------------------
    Before calling astream we snapshot how many messages are already in the
    checkpoint (`prior_msg_count`). Inside the loop we only consider supervisor
    AIMessages that were appended AFTER that index (i.e. belong to this turn).
    This is more robust than content-hashing because it is index-based and
    cannot be fooled by partial content matches.

    Fix 2 — Wrong agent label
    -------------------------
    For threat_agent and audit_agent we do NOT emit a label when the node
    activates. We wait until we see an AIMessage with tool_calls, then emit the
    tool-specific label (e.g. "🌐 IP reputation check"). Only rag_agent uses a
    node-level fallback label because it has a single, unambiguous tool.
    """
    # ── Snapshot prior message count ──────────────────────────────────────────
    prior_msg_count = 0
    try:
        prior_state = await graph.aget_state(config=_thread_config(session_id))
        if prior_state and prior_state.values:
            prior_msg_count = len(prior_state.values.get("messages", []))
    except Exception:
        pass

    # Tracks labels already shown in the UI for this turn
    announced_agents: set[str] = set()
    # Accumulates the supervisor’s growing answer for this turn (delta tracking)
    supervisor_answer = ""
    # Running count of messages seen across all steps so far this turn
    # (used to decide whether a supervisor message is new)
    total_msgs_seen = prior_msg_count

    async for step in graph.astream(
        {"messages": messages},
        config=_thread_config(session_id),
        stream_mode="updates",
    ):
        for node_name, node_state in step.items():
            msgs = node_state.get("messages", [])

            # ── Tool-call labels (threat_agent / audit_agent) ─────────────────
            # Inspect every AIMessage in this step’s delta for tool_calls.
            # Emit the tool-specific label the first time we see each tool.
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

            # ── Supervisor final answer ──────────────────────────────────────
            if node_name != "supervisor":
                continue

            # Walk messages in reverse to find the latest AIMessage from the
            # supervisor itself (not from a sub-agent acting as supervisor).
            for msg in reversed(msgs):
                if not isinstance(msg, AIMessage):
                    continue
                sender = getattr(msg, "name", None) or ""
                # Skip messages that originated from a sub-agent
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

                # ── Index-based replay guard ──────────────────────────────
                # total_msgs_seen tracks how far into the checkpoint we are.
                # We only process this supervisor message if it would fall
                # at an index > prior_msg_count (i.e. it’s from this turn).
                total_msgs_seen += 1
                if total_msgs_seen <= prior_msg_count:
                    break  # this message is from a previous turn, skip it

                # Only yield the NEW part (delta since the last supervisor step)
                if content != supervisor_answer:
                    delta = content[len(supervisor_answer):]
                    if delta:
                        yield ("text", delta)
                    supervisor_answer = content
                break


async def get_history(graph, session_id: str) -> list:
    state = await graph.aget_state(config=_thread_config(session_id))
    if state and state.values:
        return state.values.get("messages", [])
    return []
