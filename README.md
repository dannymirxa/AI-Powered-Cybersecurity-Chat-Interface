# 🛡️ CyberSec AI — AI-Powered Cybersecurity Chat Interface

An intelligent cybersecurity assistant built with **LangGraph**, **Ollama**, and **Milvus**.
It routes your questions to specialised agents, checks live threat feeds, and maintains
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
┌─────────────────────────────────────────────────────┐
│                     Docker Network                  │
│                                                     │
│  ┌──────────┐     ┌───────────────────────────────┐ │
│  │ Streamlit│────▶│        FastAPI (api.py)        │ │
│  │  UI      │     │                               │ │
│  │ :8501    │     │   LangGraph Supervisor         │ │
│  └──────────┘     │   ┌──────────────────────┐    │ │
│                   │   │  rag_agent           │    │ │
│                   │   │  threat_agent        │    │ │
│                   │   │  audit_agent         │    │ │
│                   │   └──────────────────────┘    │ │
│                   │          │          │          │ │
│                   └──────────┼──────────┼──────────┘ │
│                              │          │             │
│              ┌───────────────▼──┐  ┌────▼──────────┐ │
│              │  Ollama :11434   │  │ Milvus :19530 │ │
│              │  ministral:8b    │  │ (RAG + Memory)│ │
│              │  nomic-embed-text│  └───────────────┘ │
│              └──────────────────┘                    │
└─────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **No LangGraph checkpointer** — conversation history is stored as plain rows in a
  dedicated Milvus `chat_memory` collection and injected before each graph run.
  This eliminates the "ghost previous answer" bug caused by checkpoint replay.
- **MCP tool server** (`mcp_rag_server.py`) exposes all tools over stdio so agents
  share one subprocess instead of each opening their own Milvus / HTTP connections.
- **Single Docker image** — the FastAPI backend and Streamlit UI run inside the same
  container via a shell entrypoint, keeping the compose file simple.

---

## 🚀 Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose v2)
- 8 GB RAM minimum (16 GB recommended for GPU-accelerated inference)
- **(Optional)** NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for faster inference

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

Edit `.env` and fill in your keys:

```env
# Required for IP reputation checks
ABUSEIPDB_API_KEY=your_abuseipdb_api_key_here

# Ollama & Milvus — overridden automatically by docker-compose
OLLAMA_URL=http://localhost:11434
MILVUS_URI=http://localhost:19530

# Supervisor LLM model
SUPERVISOR_MODEL=ministral:8b
```

> Get a free AbuseIPDB key at https://www.abuseipdb.com/register

### 3. Pull Ollama models (first time only)

```bash
docker compose run --rm ollama-init
```

This pulls `ministral:8b` (~4.7 GB) and `nomic-embed-text` (~274 MB) into a
persistent Docker volume so they survive container restarts.

### 4. Start the full stack

```bash
docker compose up -d
```

| Service | URL |
|---|---|
| **Streamlit UI** | http://localhost:8501 |
| **FastAPI backend** | http://localhost:8000 |
| **API docs (Swagger)** | http://localhost:8000/docs |
| **MinIO console** | http://localhost:9001 |

### 5. Ingest knowledge base documents

Before the first query, populate the Milvus RAG collection:

```bash
docker compose exec app python RAG/ingest.py
```

Place your PDF documents in the `RAG/` folder beforehand.

---

## 📁 Project Structure

```
.
├── api.py                  # FastAPI backend — chat, history, session endpoints
├── supervisor_agent.py     # LangGraph supervisor + history injection
├── mcp_rag_server.py       # MCP tool server (5 tools exposed via stdio)
├── ui.py                   # Streamlit chat interface
├── Dockerfile              # Application image
├── docker-compose.yaml     # Full stack (app + ollama + milvus)
├── docker-entrypoint.sh    # Starts FastAPI + Streamlit inside container
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── agents/
│   ├── chat_memory.py      # Milvus-backed conversation memory
│   ├── rag_agent.py        # NIST / framework knowledge agent
│   ├── threat_agent.py     # CVE & IP threat analysis agent
│   └── audit_agent.py      # Password breach & credential audit agent
├── tools/
│   └── rag_tool.py         # Milvus vector search + external API helpers
└── RAG/
    ├── ingest.py           # PDF → chunks → embeddings → Milvus
    └── pdf_to_markdown.py  # PDF text extraction utility
```

---

## 🔧 Configuration

All configuration is via environment variables (`.env` or `docker compose` `environment:`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `MILVUS_URI` | `http://localhost:19530` | Milvus server URI |
| `SUPERVISOR_MODEL` | `ministral:8b` | LLM used by the supervisor |
| `ABUSEIPDB_API_KEY` | — | AbuseIPDB API key (IP checks) |
| `CHAT_MEMORY_COLLECTION` | `chat_memory` | Milvus collection for conversation history |
| `CHAT_MEMORY_MAX_TEXT` | `12000` | Max characters stored per message |
| `API_URL` | `http://localhost:8000` | Backend URL used by the Streamlit UI |

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Non-streaming chat (returns full JSON) |
| `POST` | `/chat/stream` | Streaming SSE chat |
| `GET` | `/history/{session_id}` | Retrieve conversation history |
| `GET` | `/sessions` | List all sessions |
| `DELETE` | `/session/{session_id}` | Delete a session |
| `GET` | `/health` | Liveness probe |

Full interactive docs available at **http://localhost:8000/docs**.

---

## 🛠️ Local Development (without Docker)

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Milvus (Docker still needed for the DB)
docker compose up -d etcd minio milvus

# 4. Start Ollama locally
ollama serve &
ollama pull ministral:8b
ollama pull nomic-embed-text

# 5. Start the backend
uvicorn api:app --reload --port 8000

# 6. Start the UI (separate terminal)
streamlit run ui.py
```

---

## 📝 How Conversation Memory Works

Unlike typical LangGraph apps that use a checkpointer, this project stores memory
programmatically in a dedicated Milvus collection:

```
User sends message
       │
       ▼
api.py: append_message(session_id, "user", message)
       │
       ▼
supervisor_agent: get_recent_messages(session_id, limit=6)
  → prepends last 6 turns to the message list
  → calls graph.astream({"messages": history + [current]})
       │
       ▼
LangGraph graph runs statelessly — no checkpoint
       │
       ▼
api.py: append_message(session_id, "assistant", answer)
```

This approach is deterministic, debuggable, and never replays old state.

---

## 🐛 Troubleshooting

**Milvus not ready / connection refused**
```bash
docker compose logs milvus
# Wait for: "Milvus Proxy successfully initialized and ready to serve"
```

**Ollama model not found**
```bash
docker compose run --rm ollama-init
```

**Reset everything (wipe all data)**
```bash
docker compose down -v
```

---

## 📄 License

MIT
