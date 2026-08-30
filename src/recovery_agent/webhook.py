"""Webhook listener — PURE SECURE INGESTOR.

Stripped of all agent logic. This module:
1. Verifies HMAC SHA256 signature (STRICT — no bypasses)
2. Deduplicates events by event_id (idempotency)
3. Parses the Razorpay payload
4. Forwards the event to frontend.py via HTTP POST to /api/webhook-forward

The frontend is the SINGLE SOURCE OF TRUTH for agent execution and UI broadcasting.

Usage:
    python -m recovery_agent.main webhook
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any

from flask import Flask, request, jsonify

logger = logging.getLogger(__name__)

app = Flask(__name__)

WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5002")

# --- Idempotency: in-memory event_id dedup with TTL expiry ---
_processed_events: dict[str, datetime] = {}
_event_lock = threading.Lock()
_IDEMPOTENCY_TTL_SECONDS = 86400  # 24 hours


def _is_duplicate_event(event_id: str) -> bool:
    """Check if event_id was already processed. Deduplicates within TTL window."""
    if not event_id:
        return False  # No event_id = can't deduplicate, allow through

    now = datetime.now(timezone.utc)
    with _event_lock:
        # Purge expired entries
        expired = [k for k, v in _processed_events.items()
                   if (now - v).total_seconds() > _IDEMPOTENCY_TTL_SECONDS]
        for k in expired:
            del _processed_events[k]

        # Check if already processed
        if event_id in _processed_events:
            return True

        # Record this event as processed
        _processed_events[event_id] = now
        return False


def verify_signature(payload: bytes, signature: str) -> bool:
    """Verify Razorpay webhook HMAC SHA256 signature.

    STRICT: If WEBHOOK_SECRET is not configured, REJECT all requests.
    No bypasses. No silent passes.
    """
    if not WEBHOOK_SECRET:
        print("[webhook] CRITICAL: RAZORPAY_WEBHOOK_SECRET not configured. Rejecting request.", file=sys.stderr)
        return False
    if not signature:
        print("[webhook] CRITICAL: No X-Razorpay-Signature header. Rejecting request.", file=sys.stderr)
        return False
    expected = hmac.new(WEBHOOK_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def forward_to_frontend(event_type: str, payload: dict) -> dict[str, Any]:
    """Forward parsed webhook event to frontend.py via HTTP POST."""
    import urllib.request
    import urllib.error
    import json

    forward_payload = {
        "event": event_type,
        "payload": payload,
        "source": "razorpay_webhook",
    }

    url = f"{FRONTEND_URL}/api/webhook-forward"
    data = json.dumps(forward_payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.URLError as e:
        print(f"[webhook] ERROR: Failed to forward to frontend: {e}", file=sys.stderr)
        return {"status": "error", "message": str(e)}
    except Exception as e:
        print(f"[webhook] ERROR: Unexpected error forwarding to frontend: {e}", file=sys.stderr)
        return {"status": "error", "message": str(e)}


def _forward_to_frontend_async(event_type: str, payload: dict) -> None:
    """Fire-and-forget background forwarding via thread pool.

    Non-blocking: spawns a daemon thread for the HTTP POST so the
    webhook handler returns 200 immediately. Under load, this prevents
    Flask worker thread starvation from slow frontend responses.
    """
    def _do_forward():
        try:
            forward_to_frontend(event_type, payload)
        except Exception as e:
            print(f"[webhook] ERROR: Background forward failed: {e}", file=sys.stderr)

    t = threading.Thread(target=_do_forward, daemon=True, name=f"fwd-{event_type[:32]}")
    t.start()


@app.route("/webhook", methods=["POST"])
def handle_webhook():
    """Receive Razorpay webhook, verify signature, deduplicate, forward to frontend.

    Returns 200 immediately after dedup check. Forwarding to frontend
    happens in a background thread to avoid blocking the webhook worker.
    """
    signature = request.headers.get("X-Razorpay-Signature", "")
    if not verify_signature(request.data, signature):
        return jsonify({"error": "Invalid signature"}), 401

    payload = request.json
    event = payload.get("event", "")

    # Idempotency: extract event_id from Razorpay payload
    event_id = payload.get("event_id", "")
    if not event_id:
        payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
        entity_id = payment_entity.get("id", "")
        event_id = f"{event}:{entity_id}" if entity_id else ""

    if _is_duplicate_event(event_id):
        print(f"[webhook] Duplicate event ignored: {event_id}", file=sys.stderr)
        return jsonify({"status": "duplicate", "event_id": event_id}), 200

    print(f"[webhook] Received event: {event}")

    # Offload forwarding to background thread — return 200 immediately
    _forward_to_frontend_async(event, payload)

    return jsonify({
        "status": "accepted",
        "event": event,
        "event_id": event_id,
        "forwarding": "async",
    }), 200


@app.route("/health", methods=["GET"])
def health():
    """Webhook ingestor health check."""
    return jsonify({
        "status": "ok",
        "service": "webhook_ingestor",
        "hmac_configured": bool(WEBHOOK_SECRET),
        "frontend_url": FRONTEND_URL,
    })


@app.route("/superu/call-complete", methods=["POST"])
def superu_call_complete():
    """Webhook callback from SuperU when an AI voice call completes.

    SuperU sends call outcome data (answered, voicemail, no-answer, etc.)
    which we feed back into the agent's observation loop.
    """
    data = request.json or {}
    logger.info("[SuperU Callback] Received: %s", data)

    call_outcome = data.get("outcome", data.get("status", "unknown"))
    payment_id = data.get("metadata", {}).get("payment_id", "")
    transcript = data.get("transcript", "")
    duration = data.get("duration_seconds", 0)

    # Log the outcome for the agent to observe
    if payment_id:
        from recovery_agent.state_store import StateStore
        store = StateStore()
        trail = store.get_trail(payment_id)
        trail.append({
            "step": "superu_call_complete",
            "msg": f"SuperU call completed: {call_outcome}",
            "detail": f"Duration: {duration}s | Transcript preview: {transcript[:200] if transcript else 'N/A'}",
            "outcome": call_outcome,
            "duration_seconds": duration,
        })
        store.set_trail(payment_id, trail)
        store.flush()

    return jsonify({"status": "received", "payment_id": payment_id}), 200


def process_webhook_payload(payload: dict) -> dict:
    """Process a raw webhook event dictionary programmatically.

    Backward-compatible wrapper for tests and direct API usage.
    Parses the payload and returns a result dict without forwarding to frontend.
    """
    event = payload.get("event", "")
    data = payload.get("payload", {})

    if event == "payment.failed":
        payment = data.get("payment", {}).get("entity", {})
        payment_id = payment.get("id", "")
        amount = payment.get("amount", 0) / 100
        error_code = payment.get("error_code", "")
        error_reason = payment.get("error_reason", "")

        return {
            "event": event,
            "status": "processed",
            "payment_id": payment_id,
            "amount": amount,
            "error_code": error_code,
            "error_reason": error_reason,
            "status_code": 200,
        }
    elif event == "payment.captured":
        payment = data.get("payment", {}).get("entity", {})
        payment_id = payment.get("id", "")
        amount = payment.get("amount", 0) / 100

        return {
            "event": event,
            "status": "processed",
            "payment_id": payment_id,
            "amount": amount,
            "status_code": 200,
        }
    elif "dispute" in event:
        return {"event": event, "status": "handled", "status_code": 200}

    return {"event": event, "status": "ignored", "status_code": 200}


def main():
    port = int(os.getenv("WEBHOOK_PORT", "5000"))
    if not WEBHOOK_SECRET:
        print("[webhook] WARNING: RAZORPAY_WEBHOOK_SECRET is empty. All requests will be REJECTED.", file=sys.stderr)
    print(f"Webhook ingestor: http://localhost:{port}/webhook")
    print(f"Health check: http://localhost:{port}/health")
    print(f"Forwarding to: {FRONTEND_URL}/api/webhook-forward")
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
