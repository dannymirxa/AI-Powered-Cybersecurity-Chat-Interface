"""
api.py
──────
FastAPI layer over the LangGraph supervisor agent.

Endpoints:
  POST /chat              — non-streaming, returns full response JSON
  POST /chat/stream       — streaming, returns text/event-stream (SSE)
  GET  /history/{session} — return conversation history for a session
  DELETE /session/{id}    — clear a session (New Chat)
  GET  /health            — liveness probe

Environment variables:
  OLLAMA_URL         http://localhost:11434
  SUPERVISOR_MODEL   qwen2.5:3b
  AGENT_MODEL        qwen2.5:3b
  MILVUS_URI         http://localhost:19530
  ABUSEIPDB_API_KEY  (optional — needed for IP reputation tool)
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from supervisor_agent import build_supervisor_graph, stream as agent_stream, get_history

# ── App state ──────────────────────────────────────────────────────────────────

APP_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the supervisor graph once on startup; tear down MCP client on shutdown."""
    print("⏳ Building supervisor graph ...")
    graph, mcp_client = await build_supervisor_graph()
    APP_STATE["graph"]      = graph
    APP_STATE["mcp_client"] = mcp_client
    print("✅ Supervisor graph ready")
    yield
    print("🔌 Shutting down MCP client ...")
    await mcp_client.__aexit__(None, None, None)


app = FastAPI(
    title="CyberSec AI Chat API",
    description="AI-powered cybersecurity assistant backed by a LangGraph supervisor.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str                = Field(..., min_length=1, max_length=4000)
    session_id: str | None      = Field(default=None)


class ChatResponse(BaseModel):
    session_id: str
    answer:     str


class HistoryMessage(BaseModel):
    role:    str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages:   list[HistoryMessage]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_session(session_id: str | None) -> str:
    """Return the given session_id or mint a new UUID."""
    return session_id or str(uuid.uuid4())


def _validate_message(message: str) -> str:
    """Strip whitespace and raise 422 if the result is empty."""
    msg = message.strip()
    if not msg:
        raise HTTPException(status_code=422, detail="Message must not be blank.")
    return msg


def _serialise_history(messages: list) -> list[HistoryMessage]:
    """Convert LangChain message objects to plain role/content dicts."""
    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, ToolMessage):
            role = "tool"
        else:
            role = "system"
        content = m.content
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        out.append(HistoryMessage(role=role, content=str(content)))
    return out


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Non-streaming chat endpoint.
    Returns the full answer after the graph finishes.
    """
    graph      = APP_STATE.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")

    message    = _validate_message(req.message)
    session_id = _resolve_session(req.session_id)

    full_answer = ""
    try:
        async for chunk in agent_stream(graph, [{"role": "user", "content": message}], session_id):
            full_answer += chunk
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return ChatResponse(session_id=session_id, answer=full_answer)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    SSE format:
        data: <chunk text>\n\n

    The client should listen for 'data:' lines and append them to the
    displayed message as they arrive.  A final sentinel event signals
    completion:
        event: done
        data: {"session_id": "..."}\n\n
    """
    graph = APP_STATE.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")

    message    = _validate_message(req.message)
    session_id = _resolve_session(req.session_id)

    async def event_generator():
        try:
            async for chunk in agent_stream(
                graph,
                [{"role": "user", "content": message}],
                session_id,
            ):
                # Escape newlines inside a chunk so each SSE 'data:' line
                # stays on a single line as per the SSE spec.
                safe = chunk.replace("\n", "\\n")
                yield f"data: {safe}\n\n"
        except Exception as exc:
            err = json.dumps({"error": str(exc)})
            yield f"event: error\ndata: {err}\n\n"
        finally:
            done = json.dumps({"session_id": session_id})
            yield f"event: done\ndata: {done}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def history(session_id: str) -> HistoryResponse:
    """Return the full conversation history for a session."""
    graph = APP_STATE.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    msgs = await get_history(graph, session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=_serialise_history(msgs),
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Clear all checkpoints for a session (New Chat)."""
    graph = APP_STATE.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    # Access the checkpointer directly from the compiled graph
    checkpointer = graph.checkpointer
    deleted = checkpointer.clear_session(session_id)
    return {"session_id": session_id, "deleted_checkpoints": deleted}
