"""
supervisor_agent.py
───────────────────
LangGraph multi-agent supervisor for the AI Cybersecurity Chat Interface.

Architecture:
                        ┌─────────────────┐
                        │   SUPERVISOR    │  ← qwen2.5:14b (routing brain)
                        │  (orchestrator) │
                        └────────┬────────┘
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
          ┌────────────┐  ┌────────────┐  ┌────────────┐
          │ rag_agent  │  │threat_agent│  │audit_agent │
          │(qwen2.5:7b)│  │(qwen2.5:7b)│  │(qwen2.5:7b)│
          └────────────┘  └────────────┘  └────────────┘
                                │
                        MCP Tool Server
                    (mcp_rag_server.py via stdio)

Session memory:
    MilvusCheckpointer persists full message history per thread_id to Milvus.
    Each chat session gets a unique thread_id — the graph replays the full
    history on every turn so the supervisor + agents see prior context.

NOTE: langchain-mcp-adapters >=0.1.0 removed context manager support from
      MultiServerMCPClient. Tools must be fetched with await client.get_tools().

Usage (single turn):
    graph, client = await build_supervisor_graph()
    result = await invoke(graph, messages, session_id="abc123")
    await client.close()   # or let it GC — stdio subprocess exits with process

Usage (streaming):
    graph, client = await build_supervisor_graph()
    async for chunk in stream(graph, messages, session_id="abc123"):
        print(chunk, end="", flush=True)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph_supervisor import create_supervisor

from agents.milvus_checkpointer import MilvusCheckpointer
from agents.rag_agent import create_rag_agent
from agents.threat_agent import create_threat_agent
from agents.audit_agent import create_audit_agent

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "qwen2.5:14b")

# mcp_rag_server.py lives at the project root (same level as this file)
MCP_SERVER_PATH = str(Path(__file__).resolve().parent / "mcp_rag_server.py")

# Forward relevant env vars to the MCP subprocess
MCP_ENV = {
    k: os.environ[k]
    for k in ("ABUSEIPDB_API_KEY", "MILVUS_URI", "OLLAMA_URL")
    if k in os.environ
}

# ── Supervisor system prompt ───────────────────────────────────────────────────

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
  • You have full access to this session's conversation history
  • Use prior turns to resolve pronouns and follow-up questions
    e.g. "tell me more about that" refers to the last topic discussed
  • Do not re-explain things you already covered unless asked

Scope enforcement:
  • If the question is completely unrelated to cybersecurity, politely decline
    and explain your focus area. Do not route to any agent.

Always present the final answer in a structured, readable format.
"""

# ── Graph builder ──────────────────────────────────────────────────────────────

async def build_supervisor_graph():
    """
    Build and return the compiled LangGraph supervisor.

    langchain-mcp-adapters >=0.1.0 removed context manager support —
    MultiServerMCPClient is now initialised directly and tools fetched
    with await client.get_tools().

    The MCP server (mcp_rag_server.py) is spawned as a stdio subprocess
    automatically by MultiServerMCPClient — no manual startup needed.

    Returns:
        tuple[CompiledGraph, MultiServerMCPClient]
        Keep a reference to `client` for the lifetime of the app.
        The subprocess exits automatically when the Python process ends.

    Example:
        graph, client = await build_supervisor_graph()

        # single turn
        result = await invoke(graph, msgs, session_id="u-123")

        # streaming
        async for chunk in stream(graph, msgs, session_id="u-123"):
            print(chunk, end="", flush=True)
    """
    # ── Step 1: start MCP client and load tools ────────────────────────────────
    mcp_config = {
        "cybersec_tools": {
            "command": sys.executable,
            "args": [MCP_SERVER_PATH],
            "transport": "stdio",
            "env": MCP_ENV if MCP_ENV else None,
        }
    }

    client = MultiServerMCPClient(mcp_config)
    tools = await client.get_tools()   # ← correct API for >=0.1.0

    print(f"✅ Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")

    # ── Step 2: build specialist agents ───────────────────────────────────────
    rag_agent    = create_rag_agent(tools)
    threat_agent = create_threat_agent(tools)
    audit_agent  = create_audit_agent(tools)

    # ── Step 3: supervisor LLM ────────────────────────────────────────────────
    supervisor_llm = ChatOllama(
        model=SUPERVISOR_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=8192,
    )

    # ── Step 4: Milvus-backed checkpointer for session memory ─────────────────
    # Persists full conversation history per thread_id to Milvus.
    # Survives server restarts — swap MemorySaver for this in production.
    checkpointer = MilvusCheckpointer()

    # ── Step 5: compile graph ─────────────────────────────────────────────────
    workflow = create_supervisor(
        agents=[rag_agent, threat_agent, audit_agent],
        model=supervisor_llm,
        prompt=SUPERVISOR_PROMPT,
    )

    graph = workflow.compile(checkpointer=checkpointer)

    return graph, client


# ── Session config helper ─────────────────────────────────────────────────────

def _thread_config(session_id: str) -> dict:
    """Build the LangGraph config dict for a given session."""
    return {"configurable": {"thread_id": session_id}}


# ── Session-aware call helpers ────────────────────────────────────────────────

async def invoke(graph, messages: list[dict], session_id: str) -> dict:
    """
    One-shot invocation with session memory.

    Pass only the LATEST user message — LangGraph automatically appends it
    to the thread's history and feeds the full history to the model.

    Args:
        graph:      Compiled graph from build_supervisor_graph()
        messages:   [{"role": "user", "content": "..."}]  ← latest turn only
        session_id: Unique session identifier (e.g. UUID per browser tab)

    Returns:
        Full LangGraph state dict — last message is the assistant reply.
    """
    return await graph.ainvoke(
        {"messages": messages},
        config=_thread_config(session_id),
    )


async def stream(graph, messages: list[dict], session_id: str):
    """
    Streaming invocation with session memory.

    Async generator — yields text chunks as they arrive from the final
    assistant message. Suitable for SSE or WebSocket streaming.

    Args:
        graph:      Compiled graph from build_supervisor_graph()
        messages:   [{"role": "user", "content": "..."}]  ← latest turn only
        session_id: Unique session identifier

    Yields:
        str — incremental text chunks
    """
    seen_content = ""

    async for chunk in graph.astream(
        {"messages": messages},
        config=_thread_config(session_id),
        stream_mode="values",
    ):
        last = chunk["messages"][-1]
        content = getattr(last, "content", None)

        # Only yield the NEW portion of the final message to avoid repeats
        if content and isinstance(content, str) and content != seen_content:
            delta = content[len(seen_content):]
            if delta:
                yield delta
            seen_content = content


async def get_history(graph, session_id: str) -> list:
    """
    Return the full message history for a session.
    Useful for restoring the UI after a page refresh.

    Args:
        graph:      Compiled graph from build_supervisor_graph()
        session_id: Session identifier

    Returns:
        List of LangChain message objects (HumanMessage, AIMessage, etc.)
    """
    state = await graph.aget_state(config=_thread_config(session_id))
    if state and state.values:
        return state.values.get("messages", [])
    return []
