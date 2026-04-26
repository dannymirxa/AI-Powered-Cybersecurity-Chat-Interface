"""
supervisor_agent.py
───────────────────
LangGraph multi-agent supervisor.

Fixes applied:
  1. AGENT_LABELS now show tool-accurate labels:
       threat_agent → splits into CVE label vs IP label based on tool called
       audit_agent  → splits into Password breach vs generic Credential Audit
     Since LangGraph node names are fixed, we use a two-level label system:
       - node-level label (shown immediately when node starts)
       - tool-level label (shown when a specific tool fires, overrides node label)
  2. stream() reply-replay bug fixed: we track a set of already-yielded content
     hashes so the supervisor’s repeated intermediate steps don’t re-emit
     content that belongs to the previous turn stored in the checkpoint.
"""

import hashlib
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

# ── Agent labels ─────────────────────────────────────────────────────────────────

# Node-level labels — shown as soon as the agent node activates.
# These are the fallback when we cannot yet tell which specific tool will run.
AGENT_NODE_LABELS: dict[str, str] = {
    "rag_agent":    "\U0001f4da NIST / Framework Q&A",
    "threat_agent": "\U0001f50d CVE / Threat Analysis",
    "audit_agent":  "\U0001f511 Credential Audit",
}

# Tool-level labels — override the node label once we see which MCP tool fires.
# Keys are the tool function names as registered in mcp_rag_server.py.
TOOL_LABELS: dict[str, str] = {
    "check_ip_reputation":  "\U0001f310 IP reputation check",
    "cve_lookup":           "\U0001f50d CVE lookup",
    "check_password_breach": "\U0001f511 Password breach check",
    "search_cybersec_kb":   "\U0001f4da NIST / Framework Q&A",
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


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


async def stream(graph, messages: list[dict], session_id: str):
    """
    Async generator that yields either:

      ("agent", label)   — emitted when a specialist agent node or tool activates
      ("text",  chunk)   — incremental supervisor answer text

    Uses stream_mode="updates" so each step delta is independent.

    Bug fixes vs previous version:
      • Tool-level labels: when a tool message fires, we emit the more specific
        tool label (e.g. "🌐 IP reputation check") instead of the node label
        ("🔍 CVE / Threat Analysis"). This fixes the wrong agent badge shown in UI.
      • Reply-replay guard: we collect hashes of all AIMessage contents seen in
        the checkpoint BEFORE this turn starts, then skip any supervisor output
        whose full content matches a previously-seen hash. This prevents the
        previous turn’s answer from being re-emitted at the top of the next turn.
    """
    # ── Collect hashes of messages already in checkpoint (previous turns) ────────
    seen_hashes: set[str] = set()
    try:
        prior_state = await graph.aget_state(config=_thread_config(session_id))
        if prior_state and prior_state.values:
            for msg in prior_state.values.get("messages", []):
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content
                    if isinstance(content, list):
                        content = "".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    seen_hashes.add(_content_hash(content))
    except Exception:
        pass  # non-fatal — worst case we might emit a duplicate once

    announced_agents: set[str] = set()
    supervisor_answer = ""

    async for step in graph.astream(
        {"messages": messages},
        config=_thread_config(session_id),
        stream_mode="updates",
    ):
        for node_name, node_state in step.items():

            # ── Tool-level agent label (most specific) ─────────────────────────
            # When a tool call fires inside an agent node, pick the tool-level
            # label so the UI shows e.g. "🌐 IP reputation check" not "🔍 CVE / ..."
            msgs = node_state.get("messages", [])
            for msg in msgs:
                if not isinstance(msg, AIMessage):
                    continue
                for tc in getattr(msg, "tool_calls", []) or []:
                    tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if tool_name in TOOL_LABELS:
                        label = TOOL_LABELS[tool_name]
                        if label not in announced_agents:
                            announced_agents.add(label)
                            yield ("agent", label)

            # ── Node-level fallback label ────────────────────────────────────
            # Only emit the node label if no tool label was emitted for this node.
            if node_name in AGENT_NODE_LABELS:
                node_label = AGENT_NODE_LABELS[node_name]
                if node_label not in announced_agents and not any(
                    lbl for lbl in announced_agents
                    if lbl != node_label
                    and node_name in ("threat_agent", "audit_agent")
                ):
                    # Check if we already emitted a tool-level label for this node's tools
                    node_tools_fired = any(
                        TOOL_LABELS[tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")]
                        in announced_agents
                        for msg in msgs
                        if isinstance(msg, AIMessage)
                        for tc in (getattr(msg, "tool_calls", []) or [])
                        if (tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")) in TOOL_LABELS
                    )
                    if not node_tools_fired:
                        announced_agents.add(node_label)
                        yield ("agent", node_label)

            # ── Supervisor final answer ─────────────────────────────────────────
            if node_name == "supervisor":
                for msg in reversed(msgs):
                    if not isinstance(msg, AIMessage):
                        continue
                    sender = getattr(msg, "name", None) or ""
                    if sender in AGENT_NODE_LABELS:
                        continue
                    content = msg.content or ""
                    if isinstance(content, list):
                        content = "".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    if not content:
                        break

                    # Skip content that belongs to a previous turn
                    if _content_hash(content) in seen_hashes:
                        break

                    # Only yield NEW delta since last supervisor step
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
