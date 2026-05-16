#!/bin/bash
# Run the InsightMeet backend
cd "$(dirname "$0")"

# Activate venv if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

echo "🚀 Starting InsightMeet backend on http://localhost:8000"
uvicorn server:app --host 0.0.0.0 --port 8000 --reload