#!/usr/bin/env bash
# Start the whole AutoRecover stack in one container. The frontend runs in the
# foreground so the container lives with it and its logs stream to `docker logs`.
set -euo pipefail

# Best-effort: cache the RAG embedding model. Non-fatal — only the knowledge-base
# lookup degrades if it can't download; everything else runs.
python -m recovery_agent.scripts.download_models \
    || echo "WARN: RAG embedding model unavailable (offline?); knowledge base degraded."

# Background services (logs to files; the frontend below streams to stdout).
python -m recovery_agent.dashboard      > /tmp/dashboard.log     2>&1 &
python -m recovery_agent.webhook        > /tmp/webhook.log       2>&1 &
python -m recovery_agent.daemon_worker  > /tmp/daemon.log        2>&1 &
phoenix serve                           > /tmp/phoenix.log       2>&1 &

echo "AutoRecover up — checkout: http://localhost:5002/pay | HUD: http://localhost:5002/merchant"

# Frontend = checkout + merchant HUD + the live agent. Foreground = PID 1.
exec python -m recovery_agent.frontend
