#!/bin/sh
set -e

# Start FastAPI backend
echo "Starting FastAPI backend on :8000 ..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Start Streamlit UI
echo "Starting Streamlit UI on :8501 ..."
streamlit run ui.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false &
STREAMLIT_PID=$!

# Keep the container alive as long as both processes are running.
# If either exits (crash or clean exit), the container stops so Docker
# can restart it according to the restart policy.
wait $FASTAPI_PID $STREAMLIT_PID
