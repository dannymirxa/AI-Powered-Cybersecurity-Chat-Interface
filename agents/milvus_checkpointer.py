from __future__ import annotations
"""
milvus_checkpointer.py
──────────────────────
Milvus-backed conversation history + session registry for LangGraph.

Collections:
  chat_history   — LangGraph checkpoints (one row per checkpoint)
  chat_sessions  — Session registry: one row per session with summary + timestamps

chat_history schema:
  id, session_id, checkpoint_id, parent_id,
  checkpoint_data (gzip+b64, VARCHAR 65535),
  metadata, created_at, _vec (dummy)

chat_sessions schema:
  session_id    VARCHAR(128)  PK
  summary       VARCHAR(512)  first user message (truncated to 100 chars)
  created_at    INT64         ms timestamp of first message
  updated_at    INT64         ms timestamp of last activity
  _vec          FLOAT_VECTOR(2)  dummy

Checkpoint slimming:
  ToolMessage content is stripped before saving to keep checkpoint_data
  within the 65,535 char VARCHAR limit.
"""

import base64
import copy
import gzip
import json
import logging
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

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────────

CHECKPOINT_COLLECTION = "chat_history"
SESSION_COLLECTION    = "chat_sessions"
MILVUS_URI            = os.getenv("MILVUS_URI", "http://localhost:19530")

_ID_LEN          = 256
_SESSION_LEN     = 128
_CHECKPOINT_LEN  = 128
_DATA_LEN        = 65535
_META_LEN        = 4096
_SUMMARY_LEN     = 512
_SUMMARY_PREVIEW = 100   # chars to keep from first user message as summary
_DUMMY_VEC_DIM   = 2
_DUMMY_VEC_VAL   = [0.0, 0.0]
_KEEP_FULL       = {"human", "ai", "system"}


# ── Checkpoint slimmer ──────────────────────────────────────────────────────────

def _slim_checkpoint(checkpoint: dict) -> dict:
    """Strip ToolMessage content before saving to keep size within VARCHAR(65535)."""
    slimmed = copy.deepcopy(checkpoint)
    channel_values = slimmed.get("channel_values", {})
    messages = channel_values.get("messages", [])
    slim_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            slim_messages.append(msg)
            continue
        msg_type = msg.get("type", "").lower()
        if msg_type in _KEEP_FULL:
            slim_messages.append(msg)
        else:
            tool_name = msg.get("name") or msg.get("tool_call_id", "unknown_tool")
            slim_msg = {k: v for k, v in msg.items() if k not in ("content", "artifact")}
            slim_msg["content"] = f"[tool_result: {tool_name} (truncated for storage)]"
            slim_messages.append(slim_msg)
    channel_values["messages"] = slim_messages
    slimmed["channel_values"]  = channel_values
    return slimmed


# ── Compression helpers ──────────────────────────────────────────────────────────

def _compress(data: dict) -> str:
    raw  = json.dumps(data, default=str).encode("utf-8")
    comp = gzip.compress(raw, compresslevel=9)
    return base64.b64encode(comp).decode("ascii")


def _decompress(blob: str) -> dict:
    if blob.startswith("{") or blob.startswith("["):
        return json.loads(blob)
    raw = gzip.decompress(base64.b64decode(blob))
    return json.loads(raw.decode("utf-8"))


# ── Collection bootstrap ────────────────────────────────────────────────────────

def _ensure_collection(
    client: MilvusClient,
    name: str,
    build_schema,
    build_index,
    description: str = "",
) -> None:
    """Generic idempotent collection creator."""
    if client.has_collection(name):
        return
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description=description,
    )
    build_schema(schema)
    index_params = client.prepare_index_params()
    build_index(index_params)
    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )
    print(f"✅ Created Milvus collection '{name}'")


def _checkpoint_schema(schema):
    schema.add_field(field_name="id",              datatype=DataType.VARCHAR, max_length=_ID_LEN,          is_primary=True)
    schema.add_field(field_name="session_id",      datatype=DataType.VARCHAR, max_length=_SESSION_LEN)
    schema.add_field(field_name="checkpoint_id",   datatype=DataType.VARCHAR, max_length=_CHECKPOINT_LEN)
    schema.add_field(field_name="parent_id",       datatype=DataType.VARCHAR, max_length=_CHECKPOINT_LEN)
    schema.add_field(field_name="checkpoint_data", datatype=DataType.VARCHAR, max_length=_DATA_LEN)
    schema.add_field(field_name="metadata",        datatype=DataType.VARCHAR, max_length=_META_LEN)
    schema.add_field(field_name="created_at",      datatype=DataType.INT64)
    schema.add_field(field_name="_vec",             datatype=DataType.FLOAT_VECTOR, dim=_DUMMY_VEC_DIM)


def _checkpoint_index(index_params):
    index_params.add_index(field_name="session_id", index_type="")
    index_params.add_index(field_name="_vec", index_type="FLAT", metric_type="L2")


def _session_schema(schema):
    schema.add_field(field_name="session_id",  datatype=DataType.VARCHAR, max_length=_SESSION_LEN,  is_primary=True)
    schema.add_field(field_name="summary",     datatype=DataType.VARCHAR, max_length=_SUMMARY_LEN)
    schema.add_field(field_name="created_at",  datatype=DataType.INT64)
    schema.add_field(field_name="updated_at",  datatype=DataType.INT64)
    schema.add_field(field_name="_vec",        datatype=DataType.FLOAT_VECTOR, dim=_DUMMY_VEC_DIM)


def _session_index(index_params):
    index_params.add_index(field_name="_vec", index_type="FLAT", metric_type="L2")


def ensure_collections(uri: str = MILVUS_URI) -> None:
    """Idempotently create both collections."""
    client = MilvusClient(uri=uri)
    _ensure_collection(client, CHECKPOINT_COLLECTION, _checkpoint_schema, _checkpoint_index,
                       "LangGraph conversation checkpoint store")
    _ensure_collection(client, SESSION_COLLECTION, _session_schema, _session_index,
                       "Session registry with summary")
    client.close()


# ── Low-level chat store ────────────────────────────────────────────────────────

class MilvusChatStore:
    def __init__(self, uri: str = MILVUS_URI):
        ensure_collections(uri)
        self.client = MilvusClient(uri=uri)

    def close(self):
        self.client.close()

    # ── Checkpoint CRUD ────────────────────────────────────────────────────────────

    def save(
        self,
        session_id: str,
        checkpoint_id: str,
        parent_id: str,
        checkpoint_data: dict,
        metadata: dict,
    ) -> None:
        slimmed    = _slim_checkpoint(checkpoint_data)
        compressed = _compress(slimmed)
        if len(compressed) > _DATA_LEN:
            logger.warning(
                "Checkpoint %s too large even after slimming+compression "
                "(%d chars > %d). Skipping Milvus persistence for this turn.",
                checkpoint_id, len(compressed), _DATA_LEN,
            )
            return
        meta_str = json.dumps(metadata, default=str)
        if len(meta_str) > _META_LEN:
            meta_str = meta_str[:_META_LEN - 3] + "..."
        self.client.upsert(
            collection_name=CHECKPOINT_COLLECTION,
            data=[{
                "id":              f"{session_id}_{checkpoint_id}",
                "session_id":      session_id,
                "checkpoint_id":   checkpoint_id,
                "parent_id":       parent_id or "",
                "checkpoint_data": compressed,
                "metadata":        meta_str,
                "created_at":      int(time.time() * 1000),
                "_vec":            _DUMMY_VEC_VAL,
            }],
        )

    def get_latest(self, session_id: str) -> Optional[dict]:
        results = self.client.query(
            collection_name=CHECKPOINT_COLLECTION,
            filter=f'session_id == "{session_id}"',
            output_fields=["id", "session_id", "checkpoint_id", "parent_id",
                           "checkpoint_data", "metadata", "created_at"],
        )
        if not results:
            return None
        latest = sorted(results, key=lambda r: r["created_at"], reverse=True)[0]
        return self._deserialise(latest)

    def get_by_checkpoint_id(self, session_id: str, checkpoint_id: str) -> Optional[dict]:
        results = self.client.get(
            collection_name=CHECKPOINT_COLLECTION,
            ids=[f"{session_id}_{checkpoint_id}"],
            output_fields=["id", "session_id", "checkpoint_id", "parent_id",
                           "checkpoint_data", "metadata", "created_at"],
        )
        if not results:
            return None
        return self._deserialise(results[0])

    def list_checkpoints(self, session_id: str) -> list[dict]:
        results = self.client.query(
            collection_name=CHECKPOINT_COLLECTION,
            filter=f'session_id == "{session_id}"',
            output_fields=["id", "session_id", "checkpoint_id", "parent_id",
                           "checkpoint_data", "metadata", "created_at"],
        )
        records = [self._deserialise(r) for r in results]
        return sorted(records, key=lambda r: r["created_at"], reverse=True)

    def delete_session(self, session_id: str) -> int:
        # Delete checkpoints
        results = self.client.query(
            collection_name=CHECKPOINT_COLLECTION,
            filter=f'session_id == "{session_id}"',
            output_fields=["id"],
        )
        ids = [r["id"] for r in results]
        if ids:
            self.client.delete(collection_name=CHECKPOINT_COLLECTION, ids=ids)
        # Delete session registry entry
        try:
            self.client.delete(collection_name=SESSION_COLLECTION, ids=[session_id])
        except Exception:
            pass
        return len(ids)

    @staticmethod
    def _deserialise(record: dict) -> dict:
        record["checkpoint_data"] = _decompress(record["checkpoint_data"])
        record["metadata"]        = json.loads(record["metadata"])
        return record

    # ── Session registry CRUD ───────────────────────────────────────────────────────

    def upsert_session(self, session_id: str, summary: str) -> None:
        """Create or update a session registry entry."""
        now = int(time.time() * 1000)
        # Check if exists to preserve created_at
        existing = self.client.get(
            collection_name=SESSION_COLLECTION,
            ids=[session_id],
            output_fields=["session_id", "created_at"],
        )
        created_at = existing[0]["created_at"] if existing else now
        self.client.upsert(
            collection_name=SESSION_COLLECTION,
            data=[{
                "session_id": session_id,
                "summary":    summary[:_SUMMARY_LEN],
                "created_at": created_at,
                "updated_at": now,
                "_vec":       _DUMMY_VEC_VAL,
            }],
        )

    def list_sessions(self) -> list[dict]:
        """Return all sessions sorted newest-first."""
        try:
            results = self.client.query(
                collection_name=SESSION_COLLECTION,
                filter="session_id != ''",
                output_fields=["session_id", "summary", "created_at", "updated_at"],
            )
        except Exception:
            return []
        return sorted(results, key=lambda r: r.get("updated_at", 0), reverse=True)


# ── LangGraph checkpointer ────────────────────────────────────────────────────────

class MilvusCheckpointer(BaseCheckpointSaver):
    def __init__(self, uri: str = MILVUS_URI):
        super().__init__()
        self.store = MilvusChatStore(uri=uri)

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
            parent_config=(
                {"configurable": {"thread_id": session_id, "checkpoint_id": record["parent_id"]}}
                if record["parent_id"] else None
            ),
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
                parent_config=(
                    {"configurable": {"thread_id": session_id, "checkpoint_id": record["parent_id"]}}
                    if record["parent_id"] else None
                ),
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

    def put_writes(
        self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str
    ) -> None:
        pass

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

    async def aput_writes(
        self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str
    ) -> None:
        pass

    def clear_session(self, session_id: str) -> int:
        deleted = self.store.delete_session(session_id)
        print(f"🗑️  Cleared {deleted} checkpoints for session {session_id!r}")
        return deleted

    def register_session(self, session_id: str, summary: str) -> None:
        """Upsert a session registry entry (called on first user message)."""
        self.store.upsert_session(session_id, summary)

    def list_sessions(self) -> list[dict]:
        """Return all sessions from the registry."""
        return self.store.list_sessions()

    def close(self):
        self.store.close()
