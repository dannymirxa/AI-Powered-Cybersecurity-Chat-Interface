# 🛡️ CyberSec AI — AI-Powered Cybersecurity Chat Interface

An intelligent cybersecurity assistant built with **LangGraph**, **Ollama**, and **Milvus**.
It routes questions to specialised agents, checks live threat feeds, and maintains
full conversation memory — all running locally in Docker.

---

## ✨ Features

| Capability | Details |
|---|---|
| 📚 NIST / Framework Q&A | Semantic RAG search over NIST CSF 2.0 and related docs |
| 🔍 CVE Lookup | Real-time vulnerability data from NIST NVD |
| 🌐 IP Reputation | Abuse score via AbuseIPDB |
| 🔑 Password Breach Check | k-anonymity check via Have I Been Pwned |
| 🧠 Conversation Memory | Per-session chat history stored in Milvus |
| 🔒 Scope Enforcement | Off-topic questions politely rejected |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Network                          │
│                                                             │
│  ┌──────────┐     ┌──────────────────────────────────────┐  │
│  │Streamlit │────▶│         FastAPI (api.py)             │  │
│  │ :UI_PORT │     │   LangGraph Supervisor               │  │
│  └──────────┘     │  ┌────────────────────────────────┐  │  │
│                   │  │ rag_agent  threat_agent  audit  │  │  │
│                   │  └────────────────────────────────┘  │  │
│                   └───────────────┬──────────────┬───────┘  │
│                                   │              │           │
│              ┌────────────────────▼──┐  ┌────────▼────────┐  │
│              │  Ollama :OLLAMA_PORT  │  │ Milvus :MILVUS  │  │
│              │  SUPERVISOR_MODEL     │  │ (RAG + Memory)  │  │
│              │  EMBED_MODEL          │  └─────────────────┘  │
│              └──────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose v2)
- 8 GB RAM minimum (16 GB recommended)
- NVIDIA GPU with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) *(optional but recommended)*

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

Open `.env` and set your values. The only **required** change is:

```env
ABUSEIPDB_API_KEY=your_abuseipdb_api_key_here
```

> Get a free key at https://www.abuseipdb.com/register (1,000 checks/day on the free tier)

All other variables have safe defaults. See [Configuration Reference](#%EF%B8%8F-configuration-reference) for the full list.

### 3. Pull Ollama models *(first time only)*

```bash
docker compose run --rm ollama-init
```

This pulls `SUPERVISOR_MODEL` and `EMBED_MODEL` into a persistent Docker volume.
The Ollama container must be healthy before this command starts — if it times out,
wait a few seconds and re-run (GPU initialisation can take up to 2 minutes on first boot).

### 4. Start the full stack

```bash
docker compose up -d
```

| Service | Default URL | Port variable |
|---|---|---|
| **Streamlit UI** | http://localhost:8501 | `UI_PORT` |
| **FastAPI backend** | http://localhost:8000 | `API_PORT` |
| **API docs (Swagger)** | http://localhost:8000/docs | `API_PORT` |
| **MinIO console** | http://localhost:9001 | `MINIO_CONSOLE_PORT` |

### 5. Ingest knowledge base documents

Convert a PDF to Markdown, then ingest it:

```bash
# Step 1 — convert PDF to Markdown (outputs a .md file alongside the PDF)
docker compose exec app python RAG/pdf_to_markdown.py RAG/your_doc.pdf

# Step 2 — embed and store in Milvus
docker compose exec app python RAG/ingest.py RAG/your_doc.md

# To wipe and re-ingest from scratch:
docker compose exec app python RAG/ingest.py RAG/your_doc.md --recreate
```

---

## ⚙️ Configuration Reference

All configuration is via environment variables. Copy `.env.example` → `.env` and edit as needed.

### API Keys

| Variable | Required | Description |
|---|---|---|
| `ABUSEIPDB_API_KEY` | **Yes** | AbuseIPDB key for IP reputation checks. Free at [abuseipdb.com](https://www.abuseipdb.com/register) |

### Ollama / Models

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL *(auto-overridden in Docker)* |
| `OLLAMA_PORT` | `11434` | Host port Ollama is exposed on |
| `SUPERVISOR_MODEL` | `ministral:8b` | LLM used by the supervisor node |
| `AGENT_MODEL` | `ministral:8b` | LLM used by the three specialist agents |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for RAG vector search |
| `SUPERVISOR_TEMPERATURE` | `0` | Supervisor LLM temperature |
| `SUPERVISOR_CTX` | `4096` | Supervisor context window (tokens) |
| `AGENT_TEMPERATURE` | `0.1` | Agent LLM temperature |
| `AGENT_CTX` | `4096` | Agent context window (tokens) |

> **Tip for 8 GB VRAM:** Set `SUPERVISOR_CTX=2048` and `AGENT_CTX=2048` to reduce
> memory pressure. Consider using `AGENT_MODEL=qwen2.5:3b` for the specialist agents
> if you hit out-of-memory errors.

### Milvus

| Variable | Default | Description |
|---|---|---|
| `MILVUS_URI` | `http://localhost:19530` | Milvus connection URI *(auto-overridden in Docker)* |
| `MILVUS_PORT` | `19530` | Host port for Milvus gRPC |
| `MILVUS_METRICS_PORT` | `9091` | Host port for Milvus health endpoint |
| `COLLECTION_NAME` | `cybersec_kb` | Milvus collection for the RAG knowledge base |
| `CHAT_MEMORY_COLLECTION` | `chat_memory` | Milvus collection for conversation history |
| `CHAT_MEMORY_MAX_TEXT` | `12000` | Max characters stored per message |
| `HISTORY_LIMIT` | `6` | Prior turns injected per LLM call |

### MinIO

| Variable | Default | Description |
|---|---|---|
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key — **change in production** |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key — **change in production** |
| `MINIO_PORT` | `9000` | Host port for MinIO S3 API |
| `MINIO_CONSOLE_PORT` | `9001` | Host port for MinIO web console |

### Backend & UI

| Variable | Default | Description |
|---|---|---|
| `API_PORT` | `8000` | Host port for the FastAPI backend |
| `UI_PORT` | `8501` | Host port for the Streamlit UI |
| `API_URL` | `http://localhost:8000` | Backend URL used by the Streamlit UI |
| `CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |

### RAG Tuning

| Variable | Default | Description |
|---|---|---|
| `RAG_TOP_K` | `5` | Max knowledge-base chunks returned per query |
| `RAG_SCORE_THRESHOLD` | `0.60` | Min cosine similarity to include a chunk |
| `INGEST_BATCH_SIZE` | `32` | Chunks per Milvus insert batch during ingestion |

### Threat Analysis

| Variable | Default | Description |
|---|---|---|
| `IP_MALICIOUS_THRESHOLD` | `25` | AbuseIPDB confidence % above which IP is flagged |

### Timeouts

| Variable | Default | Description |
|---|---|---|
| `HTTP_TIMEOUT` | `15` | Timeout (s) for CVE / IP / HIBP API calls |
| `EMBED_TIMEOUT` | `60` | Timeout (s) for Ollama embedding calls |

### Advanced: External API URL Overrides

| Variable | Default |
|---|---|
| `NVD_BASE_URL` | `https://services.nvd.nist.gov/rest/json/cves/2.0` |
| `ABUSEIPDB_BASE_URL` | `https://api.abuseipdb.com/api/v2` |
| `HIBP_BASE_URL` | `https://api.pwnedpasswords.com/range` |

---

## 📁 Project Structure

```
.
├── api.py                  # FastAPI backend
├── supervisor_agent.py     # LangGraph supervisor + history injection
├── mcp_rag_server.py       # MCP tool server (5 tools, stdio transport)
├── ui.py                   # Streamlit chat interface
├── Dockerfile
├── docker-compose.yaml
├── docker-entrypoint.sh
├── requirements.txt
├── .env.example            # ← copy to .env and fill in values
├── agents/
│   ├── chat_memory.py      # Milvus-backed conversation memory
│   ├── rag_agent.py        # NIST / framework knowledge agent
│   ├── threat_agent.py     # CVE & IP threat analysis agent
│   └── audit_agent.py      # Password breach & credential audit agent
├── tools/
│   └── rag_tool.py         # Milvus vector search + external API helpers
└── RAG/
    ├── ingest.py           # Markdown → embeddings → Milvus
    └── pdf_to_markdown.py  # PDF → Markdown + chunking
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Non-streaming chat |
| `POST` | `/chat/stream` | Streaming SSE chat |
| `GET` | `/history/{session_id}` | Conversation history for a session |
| `GET` | `/sessions` | List all sessions |
| `DELETE` | `/session/{session_id}` | Delete a session |
| `GET` | `/health` | Liveness probe |

Full interactive docs: **http://localhost:8000/docs**

---

## 🛠️ Local Development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set ABUSEIPDB_API_KEY, confirm OLLAMA_URL and MILVUS_URI point to localhost

# Start Milvus only
docker compose up -d etcd minio milvus

# Start Ollama locally
ollama serve &
ollama pull ministral:8b
ollama pull nomic-embed-text

# Start backend and UI
uvicorn api:app --reload --port 8000
streamlit run ui.py   # in a separate terminal
```

---

## 📝 Memory Architecture

```
User message
    │
    ▼
api.py → append_message(session_id, "user", ...)
    │
    ▼
supervisor_agent → get_recent_messages(session_id, limit=HISTORY_LIMIT)
  → prepends last N turns to the current message
  → graph.astream({"messages": history + [current]})
    │
    ▼
api.py → append_message(session_id, "assistant", ...)
```

The LangGraph graph is **stateless** — all memory lives in Milvus (`chat_memory`
collection). This means the graph can be restarted without losing history.

---

## 🐛 Troubleshooting

**Ollama container is unhealthy**

GPU initialisation and CUDA library loading can take up to 2 minutes on first
container start. The healthcheck has a `start_period: 120s` to account for this.
If it still fails, check the logs:
```bash
docker compose logs ollama
```

**Milvus not ready**
```bash
docker compose logs milvus
# Wait for: "Milvus Proxy successfully initialized and ready to serve"
```

**Ollama model not found after `ollama-init`**
```bash
docker compose run --rm ollama-init
```

**Port conflict** — change the relevant `*_PORT` variable in `.env`:
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
