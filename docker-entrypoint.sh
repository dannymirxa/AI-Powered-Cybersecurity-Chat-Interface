#!/bin/sh
set -e

echo "Starting FastAPI backend on :8000 ..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit UI on :8501 ..."
streamlit run ui.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false &

# Wait for any child to exit; if one dies the container restarts
wait -n
