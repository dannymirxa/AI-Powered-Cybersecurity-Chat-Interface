"""
supervisor.py
─────────────
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
    MemorySaver checkpointer persists full message history per thread_id.
    Each chat session gets a unique thread_id — the graph replays the full
    history on every turn so the supervisor + agents see prior context.

Usage (single turn):
    async with build_supervisor_graph() as graph:
        result = await invoke(graph, messages, session_id="abc123")

Usage (streaming):
    async with build_supervisor_graph() as graph:
        async for chunk in stream(graph, messages, session_id="abc123"):
            print(chunk, end="", flush=True)
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from agents.milvus_checkpointer import MilvusCheckpointer
from langgraph_supervisor import create_supervisor

from agents.rag_agent import create_rag_agent
from agents.threat_agent import create_threat_agent
from agents.audit_agent import create_audit_agent

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

MCP_SERVER_PATH = str(
    Path(__file__).resolve().parent.parent / "mcp_rag_server.py"
)

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

@asynccontextmanager
async def build_supervisor_graph() -> AsyncIterator:
    """
    Async context manager that:
      1. Spawns the MCP server as a subprocess (stdio transport)
      2. Loads all 4 tools from it
      3. Builds the three specialist agents
      4. Wraps them in a LangGraph supervisor compiled with MemorySaver
      5. Yields the compiled, session-aware graph
      6. Cleans up the MCP connection on exit

    The same graph instance should be reused across the entire app lifetime
    (e.g. created once at FastAPI startup), not recreated per request.
    MemorySaver holds all sessions in-memory — swap for SqliteSaver or
    PostgresSaver for persistence across server restarts.

    Example:
        async with build_supervisor_graph() as graph:
            result = await invoke(graph, msgs, session_id="u-123")
    """
    mcp_config = {
        "cybersec_tools": {
            "command": sys.executable,
            "args": [MCP_SERVER_PATH],
            "transport": "stdio",
            "env": MCP_ENV if MCP_ENV else None,
        }
    }

    async with MultiServerMCPClient(mcp_config) as mcp_client:
        tools = mcp_client.get_tools()

        # ── Specialist agents ─────────────────────────────────────────────────
        rag_agent    = create_rag_agent(tools)
        threat_agent = create_threat_agent(tools)
        audit_agent  = create_audit_agent(tools)

        # ── Supervisor LLM ────────────────────────────────────────────────────
        supervisor_llm = ChatOllama(
            model=os.getenv("SUPERVISOR_MODEL", "qwen2.5:14b"),
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            num_ctx=8192,
        )

        # ── Compile with MemorySaver ──────────────────────────────────────────
        # MemorySaver stores conversation history keyed by thread_id.
        # Each unique thread_id = one independent chat session.
        checkpointer = MilvusCheckpointer()  # persists to Milvus chat_history collection

        workflow = create_supervisor(
            agents=[rag_agent, threat_agent, audit_agent],
            model=supervisor_llm,
            prompt=SUPERVISOR_PROMPT,
        )

        graph = workflow.compile(checkpointer=checkpointer)
        yield graph


# ── Session-aware call helpers ────────────────────────────────────────────────

def _thread_config(session_id: str) -> dict:
    """Build the LangGraph config dict for a given session."""
    return {"configurable": {"thread_id": session_id}}


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