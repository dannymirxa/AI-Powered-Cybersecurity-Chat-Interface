"""
memory/chat_memory.py
─────────────────────
Programmatic chat memory backed by Milvus (Docker).

Two plain functions — no checkpointer, no LangGraph state magic:

  insert_conversation(session_id, agent, user_message, ai_response)
      Called after every LangGraph turn by the FastAPI endpoint.

  get_recent_history(session_id, limit=5) -> list[dict]
      Called by agents (as a tool) when they need prior context.
      Returns turns in chronological order (oldest first).

Milvus collection schema
────────────────────────
Collection : chat_memory
Fields:
  id            VARCHAR(64) PK
  session_id    VARCHAR(255)  — partitioned on this
  agent         VARCHAR(128)
  user_message  VARCHAR(4096)
  ai_response   VARCHAR(8192)
  created_at    VARCHAR(32)   — ISO-8601 UTC string, used for ORDER BY

We deliberately use VARCHAR / scalar fields only — no vectors.
Milvus supports scalar-only collections; this keeps the approach
consistent with the rest of the stack without adding a second database.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
    connections,
    utility,
)
from dotenv import load_dotenv

load_dotenv()

MILVUS_URI        = os.getenv("MILVUS_URI",        "http://localhost:19530")
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "chat_memory")


# ── Collection bootstrap ──────────────────────────────────────────────────────

def _ensure_collection(client: MilvusClient) -> None:
    """Create the chat_memory collection if it doesn't exist."""
    if client.has_collection(MEMORY_COLLECTION):
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id",           DataType.VARCHAR, max_length=64,   is_primary=True)
    schema.add_field("session_id",   DataType.VARCHAR, max_length=255)
    schema.add_field("agent",        DataType.VARCHAR, max_length=128)
    schema.add_field("user_message", DataType.VARCHAR, max_length=4096)
    schema.add_field("ai_response",  DataType.VARCHAR, max_length=8192)
    schema.add_field("created_at",   DataType.VARCHAR, max_length=32)

    # Index on session_id so queries are fast even with many rows
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="session_id",
        index_type="Trie",   # exact-match index for VARCHAR
    )

    client.create_collection(
        collection_name=MEMORY_COLLECTION,
        schema=schema,
        index_params=index_params,
    )


def _get_client() -> MilvusClient:
    client = MilvusClient(uri=MILVUS_URI)
    _ensure_collection(client)
    return client


# ── Public API ────────────────────────────────────────────────────────────────

def insert_conversation(
    session_id: str,
    agent: str,
    user_message: str,
    ai_response: str,
) -> None:
    """
    Persist one conversation turn to Milvus.
    Call this in the FastAPI /chat endpoint after streaming is complete.
    """
    client = _get_client()
    try:
        client.insert(
            collection_name=MEMORY_COLLECTION,
            data=[{
                "id":           str(uuid.uuid4()),
                "session_id":   session_id,
                "agent":        agent,
                "user_message": user_message[:4000],   # guard against huge inputs
                "ai_response":  ai_response[:8000],
                "created_at":   datetime.now(timezone.utc).isoformat(),
            }],
        )
    finally:
        client.close()


def get_recent_history(
    session_id: str,
    limit: int = 5,
) -> list[dict]:
    """
    Retrieve the most recent `limit` turns for a session.

    Returns a list of dicts in chronological order (oldest first):
        [
          {"agent": str, "user": str, "assistant": str, "at": str},
          ...
        ]

    Returns an empty list if no history exists yet.
    """
    client = _get_client()
    try:
        results = client.query(
            collection_name=MEMORY_COLLECTION,
            filter=f'session_id == "{session_id}"',
            output_fields=["agent", "user_message", "ai_response", "created_at"],
            limit=limit * 2,  # over-fetch so we can sort and trim
        )
    finally:
        client.close()

    if not results:
        return []

    # Sort by created_at ascending, take the latest `limit` turns
    sorted_results = sorted(results, key=lambda r: r["created_at"])
    recent = sorted_results[-limit:]

    return [
        {
            "agent":     r["agent"],
            "user":      r["user_message"],
            "assistant": r["ai_response"],
            "at":        r["created_at"],
        }
        for r in recent
    ]


# ── LangChain/LangGraph tool wrapper ─────────────────────────────────────────

def get_chat_history_tool(session_id: str, limit: int = 5) -> str:
    """
    Tool-friendly wrapper for agents to call.

    Returns a plain-text formatted summary of recent conversation turns
    so the agent can include it in its reasoning context.

    This function is registered as a LangGraph tool in each sub-agent.
    """
    history = get_recent_history(session_id, limit)
    if not history:
        return "No previous conversation found for this session."

    lines = [f"Last {len(history)} turn(s) of conversation:"]
    for i, turn in enumerate(history, 1):
        lines.append(f"\n[Turn {i} | {turn['at']} | agent: {turn['agent']}]")
        lines.append(f"  USER      : {turn['user']}")
        lines.append(f"  ASSISTANT : {turn['assistant']}")
    return "\n".join(lines)


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    TEST_SESSION = "test-session-001"

    print("Inserting two test turns...")
    insert_conversation(
        session_id=TEST_SESSION,
        agent="rag_agent",
        user_message="What is NIST CSF 2.0?",
        ai_response="NIST CSF 2.0 is a cybersecurity framework with six core functions: Govern, Identify, Protect, Detect, Respond, Recover.",
    )
    insert_conversation(
        session_id=TEST_SESSION,
        agent="threat_agent",
        user_message="Has the password 'password123' appeared in any data breaches?",
        ai_response="Yes — 'password123' has appeared in over 2.4 million breach records. Risk level: CRITICAL.",
    )

    print("\nFetching recent history...")
    history = get_recent_history(TEST_SESSION, limit=5)
    print(json.dumps(history, indent=2))

    print("\nFormatted tool output:")
    print(get_chat_history_tool(TEST_SESSION))
