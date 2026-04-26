"""
api.py
──────
FastAPI layer over the LangGraph supervisor agent.
All ports and service URLs are read from environment variables.

Endpoints:
  POST /chat              — non-streaming, returns full response JSON
  POST /chat/stream       — streaming SSE  (text chunks + agent events)
  GET  /history/{session} — conversation history for a session
  GET  /sessions          — list all sessions with summaries
  DELETE /session/{id}    — clear a session (New Chat)
  GET  /health            — liveness probe
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agents.chat_memory import append_message, clear_session, list_sessions
from supervisor_agent import build_supervisor_graph, stream as agent_stream, get_history

load_dotenv()

# CORS origins — override via CORS_ORIGINS (comma-separated list or "*")
_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ── App state ─────────────────────────────────────────────────────────────────

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
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str        = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None)


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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_session(session_id: str | None) -> str:
    return session_id or str(uuid.uuid4())


def _validate_message(message: str) -> str:
    msg = message.strip()
    if not msg:
        raise HTTPException(status_code=422, detail="Message must not be blank.")
    return msg


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/sessions", response_model=list[SessionInfo])
async def sessions() -> list[SessionInfo]:
    return [SessionInfo(**row) for row in list_sessions()]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    graph = APP_STATE.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")

    message    = _validate_message(req.message)
    session_id = _resolve_session(req.session_id)

    append_message(session_id, "user", message)

    full_answer = ""
    try:
        async for kind, value in agent_stream(
            graph, [{"role": "user", "content": message}], session_id
        ):
            if kind == "text":
                full_answer += value
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    append_message(session_id, "assistant", full_answer)
    return ChatResponse(session_id=session_id, answer=full_answer)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    graph = APP_STATE.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised.")

    message    = _validate_message(req.message)
    session_id = _resolve_session(req.session_id)

    append_message(session_id, "user", message)

    async def event_generator():
        full_answer = ""
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
                    full_answer += value
                    safe = value.replace("\n", "\\n")
                    yield f"data: {safe}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            if full_answer:
                append_message(session_id, "assistant", full_answer)
            yield f"event: done\ndata: {json.dumps({'session_id': session_id})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def history(session_id: str) -> HistoryResponse:
    msgs = await get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[HistoryMessage(role=m["role"], content=m["content"]) for m in msgs],
    )


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    deleted = clear_session(session_id)
    return {"session_id": session_id, "deleted_checkpoints": deleted}
