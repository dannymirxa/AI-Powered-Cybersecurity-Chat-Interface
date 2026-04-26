# ─────────────────────────────────────────────────────────────────────────────
# CyberSec AI — Application Image
# Runs both the FastAPI backend (api.py) and the Streamlit UI (ui.py).
# Ollama and Milvus are separate services defined in docker-compose.yaml.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Ports:
#   8000 — FastAPI backend
#   8501 — Streamlit UI
EXPOSE 8000 8501

# Entrypoint script starts both processes
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
