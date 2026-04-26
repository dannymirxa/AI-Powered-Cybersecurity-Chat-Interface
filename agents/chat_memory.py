"""
agents/chat_memory.py
─────────────────────
Simple Milvus-backed chat memory.

Replaces MilvusCheckpointer with lightweight functions:
  - append_message(session_id, role, content)  → insert a row
  - get_recent_messages(session_id, limit)      → SELECT TOP N ... ORDER BY created_at
  - list_sessions(limit)                        → all sessions with summary
  - clear_session(session_id)                   → delete all rows for a session

Milvus collection schema (auto-created on first use):
  id          VARCHAR(64)    primary key
  session_id  VARCHAR(128)
  role        VARCHAR(32)    'user' | 'assistant'
  content     VARCHAR(12000)
  created_at  INT64          epoch-milliseconds
  _vec        FLOAT_VECTOR(2) dummy — Milvus requires at least one vector field;
                              always stored as [0.0, 0.0] and never queried.
"""

import os
import time
import uuid
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

MILVUS_URI             = os.getenv("MILVUS_URI", "http://localhost:19530")
CHAT_MEMORY_COLLECTION = os.getenv("CHAT_MEMORY_COLLECTION", "chat_memory")
CHAT_MEMORY_MAX_TEXT   = int(os.getenv("CHAT_MEMORY_MAX_TEXT", "12000"))

# Dummy vector stored with every row so Milvus schema validation passes.
# The field is never used for similarity search.
_DUMMY_VEC = [0.0, 0.0]


def _connect() -> None:
    connections.connect(alias="default", uri=MILVUS_URI)


def _build_collection() -> Collection:
    fields = [
        FieldSchema(name="id",         dtype=DataType.VARCHAR, is_primary=True,
                    auto_id=False, max_length=64),
        FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="role",       dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="content",    dtype=DataType.VARCHAR,
                    max_length=CHAT_MEMORY_MAX_TEXT),
        FieldSchema(name="created_at", dtype=DataType.INT64),
        # Milvus requires at least one vector field per collection.
        # We use a tiny 2-dim float vector that is always [0, 0] and
        # exists purely to satisfy the schema requirement.
        FieldSchema(name="_vec",       dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    schema     = CollectionSchema(
        fields=fields,
        description="Ordered chat memory — CyberSec AI",
    )
    collection = Collection(name=CHAT_MEMORY_COLLECTION, schema=schema)

    # Scalar index on created_at for ordered queries
    try:
        collection.create_index(
            field_name="created_at",
            index_params={"index_type": "STL_SORT"},
        )
    except Exception:
        pass

    # Required vector index (FLAT, never used for search)
    collection.create_index(
        field_name="_vec",
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )

    collection.load()
    return collection


def get_collection() -> Collection:
    _connect()
    if utility.has_collection(CHAT_MEMORY_COLLECTION):
        col = Collection(CHAT_MEMORY_COLLECTION)
        col.load()
        return col
    return _build_collection()


def append_message(session_id: str, role: str, content: str) -> dict[str, Any]:
    """Insert one conversation turn into Milvus chat_memory."""
    col = get_collection()
    payload = {
        "id":         str(uuid.uuid4()),
        "session_id": session_id,
        "role":       role,
        "content":    (content or "")[:CHAT_MEMORY_MAX_TEXT],
        "created_at": int(time.time() * 1000),
        "_vec":       _DUMMY_VEC,
    }
    col.insert([payload])
    col.flush()
    return payload


def get_recent_messages(session_id: str, limit: int = 6) -> list[dict[str, Any]]:
    """
    Fetch the last `limit` messages for a session, in chronological order.
    Equivalent to:
      SELECT role, content, created_at
      FROM chat_memory
      WHERE session_id = :session_id
      ORDER BY created_at DESC
      LIMIT :limit
    """
    col  = get_collection()
    safe = session_id.replace('"', '\\"')
    rows = col.query(
        expr=f'session_id == "{safe}"',
        output_fields=["session_id", "role", "content", "created_at"],
        limit=max(1, min(int(limit), 50)),
    )
    rows = sorted(rows, key=lambda r: r.get("created_at", 0))
    if len(rows) > limit:
        rows = rows[-limit:]
    return rows


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """Return one summary row per session, sorted by most recently updated."""
    col  = get_collection()
    rows = col.query(
        expr="created_at >= 0",
        output_fields=["session_id", "role", "content", "created_at"],
        limit=max(1, min(int(limit) * 20, 1000)),
    )
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid        = row.get("session_id")
        created_at = row.get("created_at", 0)
        content    = row.get("content", "")
        if sid not in grouped:
            grouped[sid] = {
                "session_id": sid,
                "summary":    content[:100],
                "created_at": created_at,
                "updated_at": created_at,
            }
        else:
            grouped[sid]["updated_at"] = max(grouped[sid]["updated_at"], created_at)
            grouped[sid]["created_at"] = min(grouped[sid]["created_at"], created_at)
            if row.get("role") == "user" and content:
                grouped[sid]["summary"] = content[:100]
    sessions = sorted(grouped.values(), key=lambda r: r["updated_at"], reverse=True)
    return sessions[:limit]


def clear_session(session_id: str) -> int:
    """Delete all messages for a session. Returns number of rows deleted."""
    col  = get_collection()
    safe = session_id.replace('"', '\\"')
    rows = col.query(
        expr=f'session_id == "{safe}"',
        output_fields=["id"],
        limit=1000,
    )
    ids = [r["id"] for r in rows if r.get("id")]
    if not ids:
        return 0
    id_list = ", ".join(f'"{i}"' for i in ids)
    col.delete(expr=f"id in [{id_list}]")
    col.flush()
    return len(ids)
