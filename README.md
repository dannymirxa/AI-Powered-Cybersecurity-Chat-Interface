# 🛡️ AI-Powered Cybersecurity Chat Interface

A full-stack AI chat application for the cybersecurity domain, built with
**LangGraph**, **Ollama**, **Milvus**, and **Streamlit** — running entirely
locally in Docker with no cloud dependencies.

> **Assessment ref:** EGS-AI-CHAT-2026 | EC-Council Global Services — EGS-AI Division

---

## ✨ What It Does

| Phase | Capability | Implementation |
|---|---|---|
| **01** Backend & LLM | Streaming SSE chat, domain scoping, session memory | FastAPI + LangGraph + Ollama (`ministral-3:8b`) |
| **02** Knowledge Retrieval | RAG over NIST CSF 2.0 docs, source attribution | Milvus vector store + `nomic-embed-text` embeddings |
| **03** Chat UI | Real-time streaming, agent indicators, session history | Streamlit with custom CSS |
| **04** Agentic Tools | 4 autonomous tools invoked without user prompting | MCP server via LangGraph supervisor |
| **05** Code Quality | Env vars throughout, input validation, modular layout | `.env.example`, Pydantic models, Docker Compose |

---

## 🧠 Agentic Tools (Phase 04)

The supervisor LLM autonomously decides which tool(s) to call based on the
conversation. There is no manual trigger. The UI shows a `✨ Using …` indicator
when a tool is invoked.

| Tool | Agent | Purpose |
|---|---|---|
| `search_knowledge_base` | `rag_agent` | Semantic search over NIST CSF 2.0 knowledge base |
| `lookup_cve` | `threat_agent` | Real-time CVE details from NIST NVD (no API key needed) |
| `check_ip` | `threat_agent` | IP abuse score from AbuseIPDB |
| `check_breach` | `audit_agent` | k-anonymity password breach check via Have I Been Pwned |

**Why these tools?** Each maps directly to a real analyst workflow:
- A SOC analyst investigating an alert needs CVE context + framework remediation guidance
- A security engineer auditing access policies needs breach data + NIST controls
- A threat hunter needs IP reputation data without leaving the chat interface

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Network                          │
│                                                             │
│  ┌──────────┐     ┌──────────────────────────────────────┐  │
│  │Streamlit │────▶│         FastAPI (api.py)             │  │
│  │  :8501    │     │   LangGraph Supervisor               │  │
│  └──────────┘     │  ┌────────────────────────────────┐  │  │
│                   │  │ rag_agent  threat_agent  audit  │  │  │
│                   │  └────────┬─────────────┬───────┘  │  │
│                   └────────┬──────────────┬───────┘  │
│                           │              │           │
│           ┌──────────────▼────┐    ┌────────▼────────┐  │
│           │ MCP Server               │    │ Milvus :19530  │  │
│           │ (mcp_rag_server.py)      │    │ RAG KB         │  │
│           │ 4 tools over stdio       │    │ Chat Memory    │  │
│           └────────────────────────┘    └────────────────┘  │
│                       │                                   │
│           ┌──────────────▼────────┐                        │
│           │ Ollama :11434            │                        │
│           │ ministral-3:8b (LLM)     │                        │
│           │ nomic-embed-text (embed) │                        │
│           └────────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **MCP (Model Context Protocol)** is used as the tool layer. The 4 tools run in a separate subprocess (`mcp_rag_server.py`) communicating over stdio. This decouples tool logic from the agent graph and makes tools independently testable.
- **LangGraph supervisor pattern** routes each message to the most appropriate specialist agent(s). The supervisor LLM sees the tool outputs and synthesises a final answer — agents never respond directly to the user.
- **Milvus** stores both the RAG knowledge base (`cybersec_kb`) and chat history (`chat_memory`) in separate collections. The graph is stateless; all memory is in Milvus.
- **Streaming** is implemented via SSE (Server-Sent Events). The `api.py` streams `text` chunks and `agent` events; `ui.py` parses them to render typing animation and tool indicators in real time.

---

## 🚀 Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with Compose v2
- 8 GB RAM minimum (16 GB recommended)
- NVIDIA GPU with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) *(optional but recommended for acceptable response speed)*

### 1. Clone the repository

```bash
git clone https://github.com/dannymirxa/AI-Powered-Cybersecurity-Chat-Interface.git
cd AI-Powered-Cybersecurity-Chat-Interface
git checkout portable
```

### 2. Create your `.env` file

```bash
cp .env.example .env
```

Set your AbuseIPDB key (the only required change):

```env
ABUSEIPDB_API_KEY=your_key_here
```

> Get a free key at https://www.abuseipdb.com/register (1,000 checks/day free)

### 3. Start the full stack

```bash
docker compose up -d
```

This will:
1. Start Ollama, etcd, MinIO, and Milvus
2. **Automatically pull `ministral-3:8b` and `nomic-embed-text`** via `ollama-init` (first run only — ~5–10 min depending on connection)
3. Build and start the FastAPI + Streamlit `app` container

| Service | URL | Port var |
|---|---|---|
| **Streamlit UI** | http://localhost:8501 | `UI_PORT` |
| **FastAPI backend** | http://localhost:8000 | `API_PORT` |
| **API docs (Swagger)** | http://localhost:8000/docs | `API_PORT` |
| **MinIO console** | http://localhost:9001 | `MINIO_CONSOLE_PORT` |

### 4. Ingest the knowledge base (first time only)

The RAG knowledge base must be populated before the `rag_agent` can answer
framework questions. Convert a PDF and ingest it:

```bash
# Convert your PDF to Markdown
docker compose exec app python RAG/pdf_to_markdown.py RAG/your_doc.pdf

# Embed and store in Milvus
docker compose exec app python RAG/ingest.py RAG/your_doc.md

# To wipe and re-ingest from scratch:
docker compose exec app python RAG/ingest.py RAG/your_doc.md --recreate
```

> The NIST CSF 2.0 document is freely available at https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf

---

## ⚙️ Configuration Reference

All config is via environment variables. Copy `.env.example` → `.env`.

### API Keys

| Variable | Required | Description |
|---|---|---|
| `ABUSEIPDB_API_KEY` | **Yes** | AbuseIPDB key for IP reputation checks |

### Models

| Variable | Default | Description |
|---|---|---|
| `SUPERVISOR_MODEL` | `ministral-3:8b` | LLM used by the supervisor node |
| `AGENT_MODEL` | `ministral-3:8b` | LLM used by the three specialist agents |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for RAG vector search |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL *(auto-set in Docker)* |
| `SUPERVISOR_TEMPERATURE` | `0` | Supervisor LLM temperature |
| `SUPERVISOR_CTX` | `4096` | Supervisor context window |
| `AGENT_TEMPERATURE` | `0.1` | Agent LLM temperature |
| `AGENT_CTX` | `4096` | Agent context window |

> **Low VRAM tip (8 GB):** Set `SUPERVISOR_CTX=2048` and `AGENT_CTX=2048`.

### Milvus

| Variable | Default | Description |
|---|---|---|
| `MILVUS_URI` | `http://localhost:19530` | Milvus URI *(auto-set in Docker)* |
| `COLLECTION_NAME` | `cybersec_kb` | RAG knowledge base collection |
| `CHAT_MEMORY_COLLECTION` | `chat_memory` | Conversation history collection |
| `CHAT_MEMORY_MAX_TEXT` | `12000` | Max chars stored per message |
| `HISTORY_LIMIT` | `6` | Prior turns injected per LLM call |

### RAG Tuning

| Variable | Default | Description |
|---|---|---|
| `RAG_TOP_K` | `5` | Knowledge base chunks returned per query |
| `RAG_SCORE_THRESHOLD` | `0.60` | Min cosine similarity to include a chunk |
| `INGEST_BATCH_SIZE` | `32` | Chunks per Milvus insert batch |

### Threat Analysis

| Variable | Default | Description |
|---|---|---|
| `IP_MALICIOUS_THRESHOLD` | `25` | AbuseIPDB confidence % to flag an IP |

### Timeouts & Ports

| Variable | Default | Description |
|---|---|---|
| `HTTP_TIMEOUT` | `15` | Timeout (s) for CVE/IP/HIBP API calls |
| `EMBED_TIMEOUT` | `60` | Timeout (s) for Ollama embedding calls |
| `OLLAMA_PORT` | `11434` | Host port for Ollama |
| `MILVUS_PORT` | `19530` | Host port for Milvus gRPC |
| `API_PORT` | `8000` | Host port for FastAPI |
| `UI_PORT` | `8501` | Host port for Streamlit |
| `MINIO_PORT` | `9000` | Host port for MinIO S3 API |
| `MINIO_CONSOLE_PORT` | `9001` | Host port for MinIO console |

---

## 📁 Project Structure

```
.
├── api.py                  # FastAPI — all HTTP endpoints, SSE streaming
├── supervisor_agent.py     # LangGraph supervisor + streaming + history injection
├── mcp_rag_server.py       # MCP tool server — 4 cybersec tools over stdio
├── ui.py                   # Streamlit chat UI with SSE streaming + session sidebar
├── Dockerfile
├── docker-compose.yaml
├── docker-entrypoint.sh    # Starts uvicorn + streamlit in a single container
├── requirements.txt
├── .env.example            # ← copy to .env and fill in ABUSEIPDB_API_KEY
├── agents/
│   ├── chat_memory.py      # Milvus-backed session memory
│   ├── rag_agent.py        # Knowledge base Q&A agent
│   ├── threat_agent.py     # CVE + IP threat intelligence agent
│   └── audit_agent.py      # Password breach & credential audit agent
├── tools/
│   └── rag_tool.py         # Milvus search + NVD + AbuseIPDB + HIBP API helpers
└── RAG/
    ├── ingest.py           # Markdown → embeddings → Milvus
    └── pdf_to_markdown.py  # PDF → Markdown + overlapping chunks
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Non-streaming chat, returns full JSON response |
| `POST` | `/chat/stream` | Streaming SSE — `text` chunks + `agent` events |
| `GET` | `/history/{session_id}` | Conversation history for a session |
| `GET` | `/sessions` | All sessions with summary and timestamps |
| `DELETE` | `/session/{session_id}` | Delete a session from Milvus |
| `GET` | `/health` | Liveness probe |

Full interactive docs: **http://localhost:8000/docs**

---

## 📝 Conversation Memory Design

The LangGraph graph is **stateless** — it holds no memory between calls.
All memory lives in Milvus (`chat_memory` collection).

```
User message
    ↓
api.py → append_message(session_id, "user", ...)
    ↓
supervisor_agent → get_recent_messages(session_id, limit=HISTORY_LIMIT)
  → prepends last N turns before the current message
  → graph.astream({"messages": [history...] + [current]})
    ↓
api.py → append_message(session_id, "assistant", ...)
```

This design means the app survives container restarts without losing session history, and multiple app instances could share the same memory store.

---

## 🔒 Security Practices

- **No hardcoded secrets** — all API keys and URLs are env vars; `.env` is in `.gitignore`
- **Input validation** — `ChatRequest` uses Pydantic with `min_length=1`, `max_length=4000`; blank messages are rejected with HTTP 422
- **k-anonymity for password checks** — only the first 5 characters of the SHA-1 hash of a password are ever sent to HIBP; the plaintext never leaves the container
- **CORS** — configurable via `CORS_ORIGINS`; defaults to `*` for local dev
- **Off-topic enforcement** — the supervisor system prompt explicitly instructs the LLM to reject non-cybersecurity questions

---

## 🛠️ Local Development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit: set ABUSEIPDB_API_KEY, OLLAMA_URL, MILVUS_URI

# Start only the data services
docker compose up -d etcd minio milvus

# Start Ollama locally
ollama serve &
ollama pull ministral-3:8b
ollama pull nomic-embed-text

# Start backend and UI in separate terminals
uvicorn api:app --reload --port 8000
streamlit run ui.py
```

---

## 🐛 Troubleshooting

**Ollama container unhealthy**
GPU init can take up to 2 minutes on first start. The healthcheck has `start_period: 120s`.
```bash
docker compose logs ollama
```

**App container keeps restarting**
Make sure you rebuilt after the latest code pull:
```bash
git pull && docker compose build --no-cache app && docker compose up -d
```

**Milvus not ready**
```bash
docker compose logs milvus
# Wait for: "Milvus Proxy successfully initialized and ready to serve"
```

**Knowledge base is empty / RAG returns no results**
```bash
docker compose exec app python RAG/ingest.py RAG/your_doc.md --recreate
```

**Port conflict** — change the relevant `*_PORT` variable in `.env`, then:
```bash
docker compose down && docker compose up -d
```

**Reset everything (wipes all data and volumes)**
```bash
docker compose down -v
```

---

## 📄 License

MIT
