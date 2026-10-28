#!/bin/bash

cd "$(dirname "$0")"
source .venv/bin/activate

cleanup() {
    echo ""
    echo "Stopping Bank Policy Assistant..."
    kill "$API_PID" "$WEB_PID" 2>/dev/null
    wait "$API_PID" "$WEB_PID" 2>/dev/null
    exit 0
}

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

python -m http.server 5500 --bind 127.0.0.1 --directory frontend &
WEB_PID=$!

trap cleanup INT TERM

echo ""
echo "Bank Policy Assistant is starting..."
echo "Frontend: http://127.0.0.1:5500"
echo "API docs: http://127.0.0.1:8000/docs"
echo ""
echo "Keep this terminal open. Press Ctrl+C to stop both servers."

wait
