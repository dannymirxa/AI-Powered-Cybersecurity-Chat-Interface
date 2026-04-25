"""
milvus_checkpointer.py
──────────────────────
Milvus-backed conversation history store for LangGraph.

Two responsibilities:
  1. MilvusChatStore    — low-level CRUD for chat sessions in Milvus
  2. MilvusCheckpointer — LangGraph BaseCheckpointSaver implementation
                          that plugs directly into graph.compile()

Collection schema  (chat_history):
  ┌──────────────────┬─────────────────┬────────────────────────────────────┐
  │ Field            │ Type            │ Notes                              │
  ├──────────────────┼─────────────────┼────────────────────────────────────┤
  │ id               │ VARCHAR         │ "{session_id}_{checkpoint_id}"     │
  │ session_id       │ VARCHAR         │ one UUID per browser tab / session  │
  │ checkpoint_id    │ VARCHAR         │ LangGraph internal checkpoint uuid  │
  │ parent_id        │ VARCHAR         │ parent checkpoint (or "")           │
  │ checkpoint_data  │ VARCHAR         │ JSON-serialised checkpoint blob     │
  │ metadata         │ VARCHAR         │ JSON-serialised metadata            │
  │ created_at       │ INT64           │ unix timestamp (ms)                 │
  │ _vec             │ FLOAT_VECTOR(2) │ dummy — satisfies Milvus vector     │
  │                  │                 │ field requirement (dim must be ≥2)  │
  └──────────────────┴─────────────────┴────────────────────────────────────┘

Note on the dummy vector field:
  Milvus 2.x requires every collection to have at least one FLOAT_VECTOR field
  with dim in range [2, 32768]. The `_vec` field (dim=2, value always [0.0, 0.0])
  satisfies this constraint without affecting any query behaviour — all
  reads/writes use scalar fields only.

Usage:
    from agents.milvus_checkpointer import MilvusCheckpointer

    checkpointer = MilvusCheckpointer()          # uses MILVUS_URI env var
    graph = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "session-uuid-here"}}
    await graph.ainvoke({"messages": [...]}, config=config)
"""

import json
import os
import time
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from pymilvus import DataType, MilvusClient

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION_NAME = "chat_history"
MILVUS_URI      = os.getenv("MILVUS_URI", "http://localhost:19530")

_ID_LEN         = 256
_SESSION_LEN    = 128
_CHECKPOINT_LEN = 128
_DATA_LEN       = 65535
_META_LEN       = 4096
_DUMMY_VEC_DIM  = 2            # Milvus requires dim in range [2, 32768]
_DUMMY_VEC_VAL  = [0.0, 0.0]  # placeholder — this field is never searched


# ── Collection bootstrap ────────────────────────────────────────────────────

def ensure_chat_history_collection(uri: str = MILVUS_URI) -> None:
    """
    Create the chat_history collection in Milvus if it does not exist.
    Safe to call multiple times — idempotent.

    Milvus 2.x requires at least one FLOAT_VECTOR field with dim in [2, 32768]
    per collection. A dummy 2-dim vector field `_vec` satisfies this; it is
    never used for search.
    """
    client = MilvusClient(uri=uri)

    if client.has_collection(COLLECTION_NAME):
        print(f"✅ Collection '{COLLECTION_NAME}' already exists — skipping creation")
        client.close()
        return

    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description="LangGraph conversation checkpoint store",
    )

    # ─ Scalar fields ────────────────────────────────────────────────────────────
    schema.add_field(field_name="id",             datatype=DataType.VARCHAR, max_length=_ID_LEN,         is_primary=True)
    schema.add_field(field_name="session_id",     datatype=DataType.VARCHAR, max_length=_SESSION_LEN)
    schema.add_field(field_name="checkpoint_id",  datatype=DataType.VARCHAR, max_length=_CHECKPOINT_LEN)
    schema.add_field(field_name="parent_id",      datatype=DataType.VARCHAR, max_length=_CHECKPOINT_LEN)
    schema.add_field(field_name="checkpoint_data",datatype=DataType.VARCHAR, max_length=_DATA_LEN)
    schema.add_field(field_name="metadata",       datatype=DataType.VARCHAR, max_length=_META_LEN)
    schema.add_field(field_name="created_at",     datatype=DataType.INT64)

    # ─ Required dummy vector field (dim must be ≥ 2) ─────────────────────────
    schema.add_field(
        field_name="_vec",
        datatype=DataType.FLOAT_VECTOR,
        dim=_DUMMY_VEC_DIM,   # 2 — minimum allowed by Milvus
    )

    # ─ Indexes ────────────────────────────────────────────────────────────────
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="session_id", index_type="")           # scalar auto-index
    index_params.add_index(field_name="_vec",        index_type="FLAT", metric_type="L2")  # required for vector field

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"✅ Created Milvus collection '{COLLECTION_NAME}'")
    client.close()


# ── Low-level chat store ──────────────────────────────────────────────────────

class MilvusChatStore:
    """
    Thin wrapper around MilvusClient for reading/writing checkpoint records.
    Used internally by MilvusCheckpointer.
    """

    def __init__(self, uri: str = MILVUS_URI):
        ensure_chat_history_collection(uri)
        self.client = MilvusClient(uri=uri)

    def close(self):
        self.client.close()

    # ─ Write ────────────────────────────────────────────────────────────────

    def save(
        self,
        session_id: str,
        checkpoint_id: str,
        parent_id: str,
        checkpoint_data: dict,
        metadata: dict,
    ) -> None:
        """Upsert a checkpoint record."""
        record_id = f"{session_id}_{checkpoint_id}"
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            data=[{
                "id":              record_id,
                "session_id":      session_id,
                "checkpoint_id":   checkpoint_id,
                "parent_id":       parent_id or "",
                "checkpoint_data": json.dumps(checkpoint_data, default=str),
                "metadata":        json.dumps(metadata, default=str),
                "created_at":      int(time.time() * 1000),
                "_vec":            _DUMMY_VEC_VAL,  # always [0.0, 0.0]
            }],
        )

    # ─ Read ───────────────────────────────────────────────────────────────────

    def get_latest(self, session_id: str) -> Optional[dict]:
        """Return the most recent checkpoint for a session."""
        results = self.client.query(
            collection_name=COLLECTION_NAME,
            filter=f'session_id == "{session_id}"',
            output_fields=[
                "id", "session_id", "checkpoint_id",
                "parent_id", "checkpoint_data", "metadata", "created_at",
            ],
        )
        if not results:
            return None
        return sorted(results, key=lambda r: r["created_at"], reverse=True)[0]
        # note: _deserialise called by caller

    def get_latest(self, session_id: str) -> Optional[dict]:
        """Return the most recent checkpoint for a session."""
        results = self.client.query(
            collection_name=COLLECTION_NAME,
            filter=f'session_id == "{session_id}"',
            output_fields=[
                "id", "session_id", "checkpoint_id",
                "parent_id", "checkpoint_data", "metadata", "created_at",
            ],
        )
        if not results:
            return None
        latest = sorted(results, key=lambda r: r["created_at"], reverse=True)[0]
        return self._deserialise(latest)

    def get_by_checkpoint_id(
        self, session_id: str, checkpoint_id: str
    ) -> Optional[dict]:
        """Return a specific checkpoint by session + checkpoint ID."""
        record_id = f"{session_id}_{checkpoint_id}"
        results = self.client.get(
            collection_name=COLLECTION_NAME,
            ids=[record_id],
            output_fields=[
                "id", "session_id", "checkpoint_id",
                "parent_id", "checkpoint_data", "metadata", "created_at",
            ],
        )
        if not results:
            return None
        return self._deserialise(results[0])

    def list_checkpoints(self, session_id: str) -> list[dict]:
        """Return all checkpoints for a session, newest first."""
        results = self.client.query(
            collection_name=COLLECTION_NAME,
            filter=f'session_id == "{session_id}"',
            output_fields=[
                "id", "session_id", "checkpoint_id",
                "parent_id", "checkpoint_data", "metadata", "created_at",
            ],
        )
        records = [self._deserialise(r) for r in results]
        return sorted(records, key=lambda r: r["created_at"], reverse=True)

    def delete_session(self, session_id: str) -> int:
        """Delete all checkpoints for a session. Returns count deleted."""
        results = self.client.query(
            collection_name=COLLECTION_NAME,
            filter=f'session_id == "{session_id}"',
            output_fields=["id"],
        )
        ids = [r["id"] for r in results]
        if ids:
            self.client.delete(collection_name=COLLECTION_NAME, ids=ids)
        return len(ids)

    @staticmethod
    def _deserialise(record: dict) -> dict:
        record["checkpoint_data"] = json.loads(record["checkpoint_data"])
        record["metadata"]        = json.loads(record["metadata"])
        return record


# ── LangGraph checkpointer ───────────────────────────────────────────────────

class MilvusCheckpointer(BaseCheckpointSaver):
    """
    LangGraph-compatible checkpoint saver backed by Milvus.
    Plug directly into graph.compile(checkpointer=MilvusCheckpointer()).
    """

    def __init__(self, uri: str = MILVUS_URI):
        super().__init__()
        self.store = MilvusChatStore(uri=uri)

    # ─ Required by BaseCheckpointSaver ──────────────────────────────────

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        session_id    = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        record = (
            self.store.get_by_checkpoint_id(session_id, checkpoint_id)
            if checkpoint_id else
            self.store.get_latest(session_id)
        )
        if record is None:
            return None
        return CheckpointTuple(
            config={"configurable": {"thread_id": session_id, "checkpoint_id": record["checkpoint_id"]}},
            checkpoint=record["checkpoint_data"],
            metadata=record["metadata"],
            parent_config={"configurable": {"thread_id": session_id, "checkpoint_id": record["parent_id"]}} if record["parent_id"] else None,
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            return
        session_id = config["configurable"]["thread_id"]
        count = 0
        for record in self.store.list_checkpoints(session_id):
            if limit and count >= limit:
                break
            yield CheckpointTuple(
                config={"configurable": {"thread_id": session_id, "checkpoint_id": record["checkpoint_id"]}},
                checkpoint=record["checkpoint_data"],
                metadata=record["metadata"],
                parent_config={"configurable": {"thread_id": session_id, "checkpoint_id": record["parent_id"]}} if record["parent_id"] else None,
            )
            count += 1

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> RunnableConfig:
        session_id    = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]
        parent_id     = config["configurable"].get("checkpoint_id", "")
        self.store.save(
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            parent_id=parent_id,
            checkpoint_data=checkpoint,
            metadata=metadata,
        )
        return {"configurable": {"thread_id": session_id, "checkpoint_id": checkpoint_id}}

    def put_writes(self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str) -> None:
        pass

    # ─ Async variants ──────────────────────────────────────────────────────

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str) -> None:
        pass

    # ─ Utility ──────────────────────────────────────────────────────────────

    def clear_session(self, session_id: str) -> int:
        """Delete all history for a session (New Chat button). Returns count deleted."""
        deleted = self.store.delete_session(session_id)
        print(f"🗑️  Cleared {deleted} checkpoints for session {session_id!r}")
        return deleted

    def close(self):
        self.store.close()
