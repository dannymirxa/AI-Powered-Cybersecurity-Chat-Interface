# CyberSec AI — Architecture

## Chat Memory: How It Works

### Old approach (removed)
The old code used `MilvusCheckpointer` passed directly into `StateGraph.compile(checkpointer=...)`.  
This caused a critical bug: on every new message, LangGraph rehydrated the **full prior state** — including all previous AI responses — into the context window. The model would then summarise what it already had instead of answering the new question.

### New approach
Chat memory is handled **programmatically** with two plain functions:

```
memory/chat_memory.py
  insert_conversation(session_id, agent, user_message, ai_response)
  get_recent_history(session_id, limit=5) -> list[dict]
  get_chat_history_tool(session_id, limit=5) -> str   ← agent-facing tool
```

The LangGraph `StateGraph` is compiled **with no checkpointer**:
```python
graph = supervisor_graph.compile()   # no checkpointer=
```

### Flow per request

```
User message
     │
     ▼
FastAPI /chat endpoint
     │  passes only: { session_id, user_message }
     ▼
LangGraph supervisor  ──── clean state, no prior history injected
     │
     ├── routes to sub-agent (rag_agent / threat_agent / audit_agent)
     │        │
     │        │  agent CAN call get_chat_history_tool(session_id)
     │        │  if it needs context from prior turns
     │        │  (e.g. "elaborate more" follow-up questions)
     │        ▼
     │   agent produces response
     │
     ▼
FastAPI streams response to frontend
     │
     ▼
insert_conversation(session_id, agent, user_message, ai_response)
  └─ writes one row to Milvus chat_memory collection
```

### Milvus collections

| Collection     | Purpose                         | Has vectors? |
|----------------|---------------------------------|--------------|
| `cybersec_kb`  | RAG knowledge base              | ✅ Yes        |
| `chat_memory`  | Conversation history per session| ❌ Scalar only |

Both collections live in the same Milvus Docker instance (`port 19530`).
The `chat_memory` collection is created automatically on first use by `_ensure_collection()`.

### Environment variables

| Variable            | Default                  | Description                        |
|---------------------|--------------------------|------------------------------------|
| `MILVUS_URI`        | `http://localhost:19530` | Milvus Docker endpoint             |
| `MEMORY_COLLECTION` | `chat_memory`            | Collection name for chat history   |
| `COLLECTION_NAME`   | `cybersec_kb`            | Collection name for RAG KB         |
| `OLLAMA_URL`        | `http://localhost:11434` | Ollama embedding endpoint          |
| `EMBED_MODEL`       | `nomic-embed-text`       | Embedding model name               |
| `ABUSEIPDB_API_KEY` | *(required for IP tool)* | AbuseIPDB API key                  |
