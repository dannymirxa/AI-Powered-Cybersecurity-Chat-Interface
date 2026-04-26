"""
supervisor_agent.py
───────────────────
LangGraph multi-agent supervisor.

stream() now uses stream_mode="updates" instead of "values" so each step
yields only the delta from that node — not the full accumulated state.
This fixes the bug where the previous turn’s answer re-appeared at the
start of every new response.

stream() also yields agent-routing SSE events so the UI can show which
specialist agent is being called.
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

# ── Config ──────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL  = os.getenv("OLLAMA_URL",       "http://localhost:11434")
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "qwen2.5:3b")

MCP_SERVER_PATH = str(Path(__file__).resolve().parent / "mcp_rag_server.py")

MCP_ENV = {
    k: os.environ[k]
    for k in ("ABUSEIPDB_API_KEY", "MILVUS_URI", "OLLAMA_URL")
    if k in os.environ
}

# Maps LangGraph node name → human-readable label emitted in SSE agent events
AGENT_LABELS = {
    "rag_agent":    "📚 NIST / Framework Q&A",
    "threat_agent": "🔍 CVE / Threat Analysis",
    "audit_agent":  "🔑 Credential Audit",
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


async def stream(graph, messages: list[dict], session_id: str):
    """
    Async generator that yields either:

      ("agent", label)   — emitted ONCE when a specialist agent node activates
      ("text",  chunk)   — incremental supervisor answer text

    Uses stream_mode="updates" so each step delta is independent.
    This fixes the bug where stream_mode="values" replayed the full accumulated
    state including the previous turn’s answer on every new message.
    """
    announced_agents: set[str] = set()
    supervisor_answer = ""

    async for step in graph.astream(
        {"messages": messages},
        config=_thread_config(session_id),
        stream_mode="updates",
    ):
        for node_name, node_state in step.items():
            # ── Agent routing indicator ─────────────────────────────────────
            if node_name in AGENT_LABELS and node_name not in announced_agents:
                announced_agents.add(node_name)
                yield ("agent", AGENT_LABELS[node_name])

            # ── Supervisor final answer ─────────────────────────────────────
            # The supervisor node name in langgraph_supervisor is "supervisor"
            if node_name == "supervisor":
                msgs = node_state.get("messages", [])
                for msg in reversed(msgs):
                    if not isinstance(msg, AIMessage):
                        continue
                    sender = getattr(msg, "name", None) or ""
                    if sender in AGENT_LABELS:
                        continue
                    content = msg.content or ""
                    if isinstance(content, list):
                        content = "".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    # Only yield NEW content (delta since last supervisor step)
                    if content and content != supervisor_answer:
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
