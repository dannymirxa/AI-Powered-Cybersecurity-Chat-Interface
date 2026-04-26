"""
api.py
──────
FastAPI layer over the LangGraph supervisor agent.

Endpoints:
  POST /chat              — non-streaming, returns full response JSON
  POST /chat/stream       — streaming SSE  (text chunks + agent events)
  GET  /history/{session} — conversation history for a session
  GET  /sessions          — list all sessions with summaries
  DELETE /session/{id}    — clear a session (New Chat)
  GET  /health            — liveness probe

SSE event format for /chat/stream:
  Regular text chunks:       data: <text>\n\n
  Agent routing indicator:   event: agent\ndata: {"label": "..."}\n\n
  Done sentinel:             event: done\ndata: {"session_id": "..."}\n\n
  Error:                     event: error\ndata: {"error": "..."}\n\n
"""

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

# ── App state ──────────────────────────────────────────────────────────────────────

APP_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    print("⏳ Building supervisor graph ...")
    graph, mcp_client = await build_supervisor_graph()
    APP_STATE["graph"]      = graph
    APP_STATE["mcp_client"] = mcp_client
    print("✅ Supervisor graph ready")
    yield
    print("🔌 Shutting down ...")


app = FastAPI(
    title="CyberSec AI Chat API",
    description="AI-powered cybersecurity assistant backed by a LangGraph supervisor.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str          = Field(..., min_length=1, max_length=4000)
    session_id: str | None   = Field(default=None)


class ChatResponse(BaseModel):
    session_id: str
    answer:     str


class HistoryMessage(BaseModel):
    role:    str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages:   list[HistoryMessage]


class SessionInfo(BaseModel):
    session_id: str
    summary:    str
    created_at: int
    updated_at: int


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_session(session_id: str | None) -> str:
    return session_id or str(uuid.uuid4())


def _validate_message(message: str) -> str:
    msg = message.strip()
    if not msg:
        raise HTTPException(status_code=422, detail="Message must not be blank.")
    return msg


def _serialise_history(messages: list) -> list[HistoryMessage]:
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


def _register_session(graph, session_id: str, summary: str) -> None:
    """Persist session summary to Milvus via the checkpointer."""
    try:
        checkpointer = graph.checkpointer
        if hasattr(checkpointer, "register_session"):
            checkpointer.register_session(session_id, summary)
    except Exception as exc:
        # Non-fatal — session list is cosmetic
        print(f"⚠️  Session registry update failed: {exc}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/sessions", response_model=list[SessionInfo])
async def list_sessions() -> list[SessionInfo]:
    """Return all sessions in Milvus ordered by most-recently updated."""
    graph = APP_STATE.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    checkpointer = graph.checkpointer
    if not hasattr(checkpointer, "list_sessions"):
        return []
    rows = checkpointer.list_sessions()
    return [
        SessionInfo(
            session_id=r["session_id"],
            summary=r.get("summary", ""),
            created_at=r.get("created_at", 0),
            updated_at=r.get("updated_at", 0),
        )
        for r in rows
    ]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    graph      = APP_STATE.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    message    = _validate_message(req.message)
    session_id = _resolve_session(req.session_id)
    _register_session(graph, session_id, message[:100])
    full_answer = ""
    try:
        async for kind, value in agent_stream(graph, [{"role": "user", "content": message}], session_id):
            if kind == "text":
                full_answer += value
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
    return ChatResponse(session_id=session_id, answer=full_answer)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    Streaming SSE endpoint.

    Event types:
      (default)      data: <text chunk>              regular answer text
      event: agent   data: {"label": "..."}           agent routing indicator
      event: done    data: {"session_id": "..."}      completion sentinel
      event: error   data: {"error": "..."}           error
    """
    graph = APP_STATE.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    message    = _validate_message(req.message)
    session_id = _resolve_session(req.session_id)
    _register_session(graph, session_id, message[:100])

    async def event_generator():
        try:
            async for kind, value in agent_stream(
                graph,
                [{"role": "user", "content": message}],
                session_id,
            ):
                if kind == "agent":
                    payload = json.dumps({"label": value})
                    yield f"event: agent\ndata: {payload}\n\n"
                elif kind == "text":
                    safe = value.replace("\n", "\\n")
                    yield f"data: {safe}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield f"event: done\ndata: {json.dumps({'session_id': session_id})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def history(session_id: str) -> HistoryResponse:
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
    graph = APP_STATE.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")
    checkpointer = graph.checkpointer
    deleted = checkpointer.clear_session(session_id)
    return {"session_id": session_id, "deleted_checkpoints": deleted}
