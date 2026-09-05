#!/usr/bin/env bash
# Start the whole AutoRecover stack in one container. The frontend runs in the
# foreground so the container lives with it and its logs stream to `docker logs`.
set -euo pipefail

# Warm the RAG embedding model in the BACKGROUND so services start immediately.
# It's a ~80MB download; blocking on it would make the stack look hung for
# minutes on first run. Non-fatal — the knowledge-base lookup just degrades
# until it finishes; everything else is up right away.
( python -m recovery_agent.scripts.download_models \
    && echo "RAG embedding model ready." \
    || echo "WARN: RAG embedding model unavailable (offline?); knowledge base degraded." ) &

# Background services (logs to files; the frontend below streams to stdout).
python -m recovery_agent.webhook       > /tmp/webhook.log 2>&1 &
python -m recovery_agent.daemon_worker > /tmp/daemon.log  2>&1 &

# Phoenix runs its first-boot DB migration inside `phoenix serve`, and on a FRESH
# database that process comes up and then exits right after the migration
# finishes. The retry starts against the now-migrated DB and serves normally and
# indefinitely (verified: a warm start stays up 7+ min; a cold start exits once).
# The loop makes that one-time cold-start exit self-heal, so tracing comes up on
# its own a couple of minutes in rather than staying dark.
( while true; do phoenix serve; echo "[entrypoint] phoenix exited, restarting in 2s"; sleep 2; done ) > /tmp/phoenix.log 2>&1 &

echo "AutoRecover up — checkout: http://localhost:5002/pay | HUD: http://localhost:5002/merchant"

# Frontend = checkout + merchant HUD + the live agent. Foreground = PID 1.
exec python -m recovery_agent.frontend
