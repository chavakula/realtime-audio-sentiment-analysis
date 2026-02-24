#!/bin/bash
set -e

echo "Starting FastAPI server on port 8000..."
uvicorn server.main:app --host 0.0.0.0 --port 8000 --ws-max-size 16777216 &

echo "Starting Streamlit dashboard on port 8501..."
streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false

# Keep container alive
wait
