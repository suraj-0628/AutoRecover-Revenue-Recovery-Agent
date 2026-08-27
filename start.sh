#!/bin/bash
# Start all recovery agent services
# Usage: ./start.sh

cd "$(dirname "$0")"

# Load environment variables from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded .env"
else
    echo "ERROR: .env file not found. Copy .env.example to .env and add your keys."
    exit 1
fi

source .venv/bin/activate

# --- Pre-flight: Download ChromaDB ONNX embedding model if not cached ---
echo "Running pre-flight checks..."
.venv/bin/python3 -m recovery_agent.scripts.download_models
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to download ChromaDB embedding models. Check internet connection."
    echo "The RAG engine requires the ONNX model to be cached locally."
    echo "You can also download it manually: python -m recovery_agent.scripts.download_models"
    exit 1
fi

echo "Starting Revenue Recovery Agent services..."

# Kill any existing instances
pkill -f "recovery_agent.dashboard" 2>/dev/null
pkill -f "recovery_agent.webhook" 2>/dev/null
pkill -f "recovery_agent.frontend" 2>/dev/null
sleep 1

# Start dashboard
setsid .venv/bin/python3 -m recovery_agent.dashboard < /dev/null > /tmp/dashboard.log 2>&1 &
DASHBOARD_PID=$!

# Start webhook listener
setsid .venv/bin/python3 -m recovery_agent.webhook < /dev/null > /tmp/webhook.log 2>&1 &
WEBHOOK_PID=$!

# Start frontend (customer + merchant)
setsid .venv/bin/python3 -m recovery_agent.frontend < /dev/null > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!

sleep 3

# Verify services
DASHBOARD_OK=$(curl -s http://localhost:${DASHBOARD_PORT:-5001}/api/metrics | grep -q "total_cases" && echo "YES" || echo "NO")
WEBHOOK_OK=$(curl -s http://localhost:${WEBHOOK_PORT:-5000}/health | grep -q "status" && echo "YES" || echo "NO")
FRONTEND_OK=$(curl -s http://localhost:5002/health | grep -q "status" && echo "YES" || echo "NO")

echo ""
echo "========================================="
echo "  ALL SERVICES RUNNING"
echo "========================================="
echo ""
echo "  --- Customer & Merchant ---"
echo "  Customer Checkout: http://localhost:5002/pay"
echo "  Merchant Dashboard: http://localhost:5002/merchant"
echo ""
echo "  --- Backend ---"
echo "  Recovery Dashboard: http://localhost:${DASHBOARD_PORT:-5001}"
echo "  Agent Flow:         http://localhost:${DASHBOARD_PORT:-5001}/graph"
echo "  Webhook Listener:   http://localhost:${WEBHOOK_PORT:-5000}/webhook"
echo ""
echo "  --- Status ---"
echo "  Dashboard:  [$DASHBOARD_OK]"
echo "  Webhook:    [$WEBHOOK_OK]"
echo "  Frontend:   [$FRONTEND_OK]"
echo ""
echo "  To stop: pkill -f recovery_agent"
echo "========================================="
