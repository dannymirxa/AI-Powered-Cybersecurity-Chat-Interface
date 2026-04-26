"""
supervisor_agent.py
───────────────────
LangGraph multi-agent supervisor for the AI Cybersecurity Chat Interface.

Architecture:
                        ┌─────────────────┐
                        │   SUPERVISOR    │  ← SUPERVISOR_MODEL (default: qwen2.5:3b)
                        │  (orchestrator) │
                        └────────┬────────┘
                 ┌───────────┼───────────────┐
                 │               │               │
          ┌────────────┐  ┌────────────┐  ┌────────────┐
          │ rag_agent  │  │threat_agent│  │audit_agent │
          │AGENT_MODEL │  │AGENT_MODEL │  │AGENT_MODEL │
          └────────────┘  └────────────┘  └────────────┘
                                │
                        MCP Tool Server
                    (mcp_rag_server.py via stdio)

Model env vars (all optional — defaults shown):
  SUPERVISOR_MODEL   qwen2.5:3b   orchestrator / router
  AGENT_MODEL        qwen2.5:3b   all three sub-agents

Key design decision — MCP client lifecycle:
  MultiServerMCPClient must be used as an async context manager
  (`async with client`) before get_tools() can be called.  We enter the
  context once at startup (inside build_supervisor_graph) and keep it open
  for the lifetime of the FastAPI process, closing it only in the lifespan
  shutdown hook.  The caller receives both the compiled graph and the
  live client so it can close it properly.
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

# ── Supervisor system prompt ─────────────────────────────────────────────────────

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

# ── Graph builder ────────────────────────────────────────────────────────────────

async def build_supervisor_graph():
    """
    Build and compile the LangGraph supervisor.

    Returns (graph, mcp_client) where mcp_client is the already-entered
    async context manager.  The caller MUST call
    `await mcp_client.__aexit__(None, None, None)` on shutdown.

    Root cause of the original startup hang:
      MultiServerMCPClient.get_tools() only works AFTER the client context
      has been entered (async with client).  Previously the code called
      get_tools() on an unentered client, which blocked indefinitely.
      We now enter the context first, then fetch tools.
    """
    mcp_config = {
        "cybersec_tools": {
            "command": sys.executable,
            "args":    [MCP_SERVER_PATH],
            "transport": "stdio",
            "env": MCP_ENV if MCP_ENV else None,
        }
    }

    # ── Enter the MCP client context FIRST, THEN get tools ────────────────────
    client = MultiServerMCPClient(mcp_config)
    await client.__aenter__()          # starts the stdio subprocess
    tools  = await client.get_tools()  # now safe to call
    print(f"✅ Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")

    # ── Specialist agents ─────────────────────────────────────────────────────────────
    rag_agent    = create_rag_agent(tools)
    threat_agent = create_threat_agent(tools)
    audit_agent  = create_audit_agent(tools)

    # ── Supervisor LLM ─────────────────────────────────────────────────────────────
    supervisor_llm = ChatOllama(
        model=SUPERVISOR_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=4096,
    )
    print(f"✅ Supervisor model: {SUPERVISOR_MODEL}")

    # ── Checkpointer ────────────────────────────────────────────────────────────────
    checkpointer = MilvusCheckpointer()

    # ── Compile ───────────────────────────────────────────────────────────────────
    workflow = create_supervisor(
        agents=[rag_agent, threat_agent, audit_agent],
        model=supervisor_llm,
        prompt=SUPERVISOR_PROMPT,
    )
    graph = workflow.compile(checkpointer=checkpointer)
    return graph, client


# ── Helpers ────────────────────────────────────────────────────────────────────

def _thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


async def invoke(graph, messages: list[dict], session_id: str) -> dict:
    return await graph.ainvoke(
        {"messages": messages},
        config=_thread_config(session_id),
    )


async def stream(graph, messages: list[dict], session_id: str):
    """
    Async generator — yields incremental text chunks from the supervisor's
    final AIMessage only.

    Only yields content from proper AIMessage instances that come from the
    supervisor node (not sub-agents).
    """
    SUB_AGENTS = {"rag_agent", "threat_agent", "audit_agent"}

    last_ai_content = ""
    seen_content    = ""

    async for state in graph.astream(
        {"messages": messages},
        config=_thread_config(session_id),
        stream_mode="values",
    ):
        msgs = state.get("messages", [])

        for msg in reversed(msgs):
            if not isinstance(msg, AIMessage):
                continue
            sender = getattr(msg, "name", None) or ""
            if sender in SUB_AGENTS:
                continue
            content = msg.content or ""
            if isinstance(content, list):
                content = "".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            last_ai_content = content
            break

        if last_ai_content and last_ai_content != seen_content:
            delta = last_ai_content[len(seen_content):]
            if delta:
                yield delta
            seen_content = last_ai_content


async def get_history(graph, session_id: str) -> list:
    state = await graph.aget_state(config=_thread_config(session_id))
    if state and state.values:
        return state.values.get("messages", [])
    return []
