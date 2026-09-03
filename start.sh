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

# --- Pre-flight: Verify critical imports work in venv ---
echo "Verifying critical imports..."
.venv/bin/python3 -c "
import sys
fails = []
for mod in [
    'langchain.tools',
    'langchain_core.messages',
    'langgraph.graph',
    'langgraph.prebuilt',
    'langgraph.checkpoint.memory',
    'pydantic',
    'flask',
    'flask_socketio',
    'razorpay',
    'networkx',
]:
    try:
        __import__(mod)
    except ImportError as e:
        fails.append(f'  {mod}: {e}')
if fails:
    print('FATAL: Missing dependencies:')
    for f in fails:
        print(f)
    print('Run: .venv/bin/pip install -e .')
    sys.exit(1)
print('All critical imports OK')
" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Pre-flight import check failed. Install missing deps with: .venv/bin/pip install -e ."
    exit 1
fi

echo "Starting Revenue Recovery Agent services..."

# Kill any existing instances
pkill -f "recovery_agent.dashboard" 2>/dev/null
pkill -f "recovery_agent.webhook" 2>/dev/null
pkill -f "recovery_agent.frontend" 2>/dev/null
pkill -f "recovery_agent.daemon_worker" 2>/dev/null
pkill -f "phoenix.server" 2>/dev/null
sleep 1

# Start Phoenix observability server
export PHOENIX_PORT=6006
setsid .venv/bin/phoenix serve > /tmp/phoenix.log 2>&1 &
PHOENIX_PID=$!
sleep 2

# Start dashboard
setsid .venv/bin/python3 -m recovery_agent.dashboard < /dev/null > /tmp/dashboard.log 2>&1 &
DASHBOARD_PID=$!

# Start webhook listener
setsid .venv/bin/python3 -m recovery_agent.webhook < /dev/null > /tmp/webhook.log 2>&1 &
WEBHOOK_PID=$!

# Start frontend (customer + merchant)
setsid .venv/bin/python3 -m recovery_agent.frontend < /dev/null > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!

# Start daemon worker (executes scheduled retries)
setsid .venv/bin/python3 -m recovery_agent.daemon_worker < /dev/null > /tmp/daemon_worker.log 2>&1 &
DAEMON_PID=$!

sleep 3

# Verify services
PHOENIX_OK=$(curl -s http://localhost:6006/ | grep -q "" && echo "YES" || echo "NO")
DASHBOARD_OK=$(curl -s http://localhost:${DASHBOARD_PORT:-5001}/api/metrics | grep -q "total_cases" && echo "YES" || echo "NO")
WEBHOOK_OK=$(curl -s http://localhost:${WEBHOOK_PORT:-5000}/health | grep -q "status" && echo "YES" || echo "NO")
FRONTEND_OK=$(curl -s http://localhost:5002/health | grep -q "status" && echo "YES" || echo "NO")
DAEMON_OK=$(pgrep -f "recovery_agent.daemon_worker" > /dev/null && echo "YES" || echo "NO")

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
echo "  --- Observability ---"
echo "  Phoenix Tracing:    http://localhost:6006"
echo ""
echo "  --- Status ---"
echo "  Phoenix:    [$PHOENIX_OK]"
echo "  Dashboard:  [$DASHBOARD_OK]"
echo "  Webhook:    [$WEBHOOK_OK]"
echo "  Frontend:   [$FRONTEND_OK]"
echo "  Daemon:     [$DAEMON_OK]"
echo ""
echo "  To stop: pkill -f recovery_agent"
echo "========================================="
