#!/bin/bash
# Only a service may spend the account's 30-link lifetime quota.
# An ad-hoc script does not inherit this and therefore cannot.
export RAZORPAY_WRITES_OK=1
# Same control for the paid SuperU call allowance: only a service may spend it.
# A verification script run as `python -c` once placed a real call because the
# test-environment sniff failed open; this cannot fail open.
export SUPERU_CALLS_OK=1

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
# Non-fatal: this only powers the RAG knowledge lookup. If it can't download
# (offline, mirror down), the rest of the stack — checkout, agent, batch, HUD —
# still runs; only the knowledge-base tool degrades. Don't block the demo on it.
echo "Running pre-flight checks..."
.venv/bin/python3 -m recovery_agent.scripts.download_models
if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: could not download the RAG embedding model (offline?)."
    echo "Continuing — the knowledge-base lookup will be degraded, everything"
    echo "else works. To fix later: python -m recovery_agent.scripts.download_models"
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
pkill -f "recovery_agent.webhook" 2>/dev/null
pkill -f "recovery_agent.frontend" 2>/dev/null
pkill -f "recovery_agent.daemon_worker" 2>/dev/null
# Was "phoenix.server" — as a pkill -f regex that never matches the actual
# command line ("phoenix serve": a space, not a dot, and "serve" is one
# character short of "server"). The old process outlived every restart,
# including ones meant to pick up PHOENIX_WORKING_DIR — traces and the
# price table kept going to Phoenix's default location, silently orphaned
# by the next "restart".
pkill -f "\.venv/bin/phoenix serve" 2>/dev/null
sleep 1

# Start Phoenix observability server.
# PHOENIX_WORKING_DIR pins its SQLite to the repo's data dir — without it,
# traces (and the custom model price table the cost view depends on) live
# wherever the default lands and a restart can orphan them.
export PHOENIX_PORT=6006
export PHOENIX_WORKING_DIR="$(pwd)/data/phoenix"
mkdir -p "$PHOENIX_WORKING_DIR"
# `phoenix serve` exits once on a FRESH database, right after its first-boot
# migration; the retry then starts warm and serves indefinitely. The loop
# self-heals that one-time cold-start exit so tracing reliably comes up on a
# fresh clone too (matches docker-entrypoint.sh).
setsid bash -c 'while true; do .venv/bin/phoenix serve; echo "[start] phoenix exited, restarting in 2s"; sleep 2; done' < /dev/null > /tmp/phoenix.log 2>&1 &
PHOENIX_PID=$!

# Wait for Phoenix to actually answer instead of guessing at a sleep.
# It runs schema migrations on first boot against a fresh working dir, which
# takes far longer than any fixed pause — so a 2-second sleep meant the
# startup banner reported "Phoenix: [NO]" and the price seeder below died
# with "Connection refused", on a server that was perfectly healthy moments
# later. Bounded, so a genuinely dead Phoenix still fails fast-ish.
echo -n "Waiting for Phoenix"
_phx_up=""
for _ in $(seq 1 90); do
    if curl -s -o /dev/null --max-time 2 "http://localhost:${PHOENIX_PORT}/" 2>/dev/null; then
        echo " up."
        _phx_up=1
        break
    fi
    echo -n "."
    sleep 2
done
# On a fresh clone the first-boot migration plus the one-time cold-start restart
# can take a few minutes. Don't block the whole stack on it — say so and move on;
# the restart loop brings tracing up on its own shortly after.
[ -z "$_phx_up" ] && echo " still starting (first-boot migration); it will come up at :${PHOENIX_PORT} shortly."

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

# Seed Phoenix's model price table with the proxy fleet (idempotent, fast).
# Without prices, spans carry tokens but no cost — and the per-case cost
# rollup in the Sessions view stays empty.
.venv/bin/python3 -m recovery_agent.scripts.seed_phoenix_model_costs || true

# Verify services.
#
# Each check RETRIES for a few seconds instead of asking once. A service that
# is merely still binding its port is not a failed service, and a banner that
# reports one as [NO] while it serves traffic a second later teaches you to
# distrust the banner — the worst possible thing to hand someone on demo day.
wait_ok() {              # wait_ok <url> <expected-substring> [attempts]
    local url="$1" want="$2" tries="${3:-10}"
    for _ in $(seq 1 "$tries"); do
        if curl -s --max-time 2 "$url" 2>/dev/null | grep -q "$want"; then
            echo "YES"; return
        fi
        sleep 1
    done
    echo "NO"
}

# Phoenix serves HTML, so match on the status code rather than piping a body
# into `grep -q ""` — an empty body from a server that IS up reads as failure.
PHOENIX_OK=$(for _ in $(seq 1 10); do
    if curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://localhost:${PHOENIX_PORT}/" 2>/dev/null | grep -q "^2"; then
        echo "YES"; break
    fi
    sleep 1
done)
PHOENIX_OK=${PHOENIX_OK:-NO}
WEBHOOK_OK=$(wait_ok "http://localhost:${WEBHOOK_PORT:-5000}/health" "status")
FRONTEND_OK=$(wait_ok "http://localhost:5002/health" "status")
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
echo "  Webhook Listener:   http://localhost:${WEBHOOK_PORT:-5000}/webhook"
echo ""
echo "  --- Observability ---"
echo "  Phoenix Tracing:    http://localhost:6006"
echo ""
echo "  --- Status ---"
echo "  Phoenix:    [$PHOENIX_OK]"
echo "  Webhook:    [$WEBHOOK_OK]"
echo "  Frontend:   [$FRONTEND_OK]"
echo "  Daemon:     [$DAEMON_OK]"
echo ""
echo "  To stop: pkill -f recovery_agent"
echo "========================================="
