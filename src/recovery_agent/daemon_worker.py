"""Background Daemon Worker — executes scheduled retries autonomously.

Polls the StateStore every 60 seconds for due jobs. When datetime.now() >= target_time,
triggers the Razorpay SDK to execute the retry, completely independent of the frontend UI thread.

Reports results back to the frontend via HTTP POST to /api/daemon-retry-complete.

Usage:
    python -m recovery_agent.daemon_worker
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Any


POLL_INTERVAL = int(os.getenv("DAEMON_POLL_INTERVAL", "60"))  # seconds
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5002")


def register_retry_job(
    payment_id: str,
    amount: float,
    target_timestamp: str,
    action: str = "retry_payment",
    method: str = "card",
    customer: dict | None = None,
    reason: str = "",
    confidence: float = 0.5,
) -> dict:
    """Register a background retry job with the StateStore.

    Returns the job dict with all fields the frontend expects to
    spread into a socket emit and store as ``scheduled_job``.
    """
    import uuid
    from recovery_agent.state_store import StateStore

    store = StateStore()
    job_id = f"retry_{payment_id}_{uuid.uuid4().hex[:8]}"

    metadata = {
        "amount": amount,
        "method": method,
        "customer": customer or {},
        "reason": reason,
        "confidence": confidence,
    }

    store.schedule_job(
        job_id=job_id,
        payment_id=payment_id,
        target_time=target_timestamp,
        action=action,
        metadata=metadata,
    )
    store.flush()

    return {
        "job_id": job_id,
        "payment_id": payment_id,
        "target_timestamp": target_timestamp,
        "action": action,
        "method": method,
        "reason": reason,
        "confidence": confidence,
        "status": "scheduled",
    }


# --- Retry Execution ---

def execute_retry(job: dict[str, Any]) -> dict[str, Any]:
    """Execute a retry job via the Razorpay SDK."""
    from recovery_agent.razorpay_client import RazorpayClient

    payment_id = job.get("payment_id", "")
    amount = job.get("metadata", {}).get("amount", 0)
    action = job.get("action", "retry_payment")
    method = job.get("metadata", {}).get("method", "card")

    client = RazorpayClient()

    if not client.is_configured:
        return {
            "status": "error",
            "message": "Razorpay client not configured",
            "payment_id": payment_id,
        }

    try:
        if action == "retry_payment" or action == "update_payment_method":
            order = client.create_order(amount=amount, receipt=f"retry_{payment_id}")
            order_id = order.get("id", "")
            return {
                "status": "retry_created",
                "payment_id": payment_id,
                "order_id": order_id,
                "amount": amount,
                "method": method,
                "message": f"Retry order {order_id} created for INR {amount:,.2f}",
            }
        elif action == "send_notification":
            link = client.create_payment_link(
                amount=amount,
                customer=job.get("metadata", {}).get("customer", {}),
                notes={"recovery_agent": "daemon_retry", "original_payment": payment_id},
            )
            return {
                "status": "link_created",
                "payment_id": payment_id,
                "link_url": link.get("short_url", ""),
                "link_id": link.get("id", ""),
                "amount": amount,
                "message": f"Payment link created for INR {amount:,.2f}",
            }
        else:
            return {
                "status": "error",
                "message": f"Unknown action: {action}",
                "payment_id": payment_id,
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "payment_id": payment_id,
        }


def notify_frontend(job: dict[str, Any], result: dict[str, Any]) -> bool:
    """Notify frontend.py of retry execution via HTTP POST."""
    payload = {
        "job_id": job.get("job_id", ""),
        "payment_id": job.get("payment_id", ""),
        "action": job.get("action", ""),
        "result": result,
        "source": "daemon_worker",
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{FRONTEND_URL}/api/daemon-retry-complete",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"[daemon] Failed to notify frontend: {e}", file=sys.stderr)
        return False


# --- Daemon Loop ---

def _check_daemon_guardrails(job: dict) -> bool:
    """Check if a retry job should be deferred due to guardrail policies.

    Returns True if the job should be deferred (blocked), False if OK to execute.
    """
    from datetime import timedelta

    # Check quiet hours (9 PM – 8 AM IST)
    now_utc = datetime.now(timezone.utc)
    IST_OFFSET = timedelta(hours=5, minutes=30)
    now_ist = now_utc + IST_OFFSET
    hour = now_ist.hour
    if hour >= 21 or hour < 8:
        print(f"[daemon] DEFERRED: Quiet hours active (IST {now_ist.strftime('%H:%M')}). Job will retry next poll.")
        return True

    # Check frequency cap: max 2 communications per 24h per payment
    from recovery_agent.state_store import StateStore
    store = StateStore()
    payment_id = job.get("payment_id", "")
    trail = store.get_trail(payment_id)
    now_ts = now_utc.isoformat()
    recent_comms = [
        e for e in trail
        if e.get("step", "").startswith(("notification", "superu_call", "scheduled"))
        and e.get("ts", "") > (now_utc - timedelta(hours=24)).isoformat()
    ]
    if len(recent_comms) >= 2:
        print(f"[daemon] DEFERRED: Frequency cap reached for {payment_id} ({len(recent_comms)} in 24h).")
        return True

    return False


def _process_due_jobs() -> int:
    """Check StateStore for due jobs and execute them. Returns count executed."""
    from recovery_agent.state_store import StateStore

    store = StateStore()
    now = datetime.now(timezone.utc)
    due_jobs = store.get_due_jobs(now)
    executed_count = 0

    for job in due_jobs:
        job_id = job.get("job_id", "")
        payment_id = job.get("payment_id", "")

        # Check guardrails before executing
        if _check_daemon_guardrails(job):
            # Defer: move target_time to next poll interval
            new_target = (now + timedelta(seconds=POLL_INTERVAL * 2)).isoformat()
            with store._lock:
                store._jobs[job_id]["target_time"] = new_target
            store.flush()
            continue

        print(f"[daemon] Executing retry job {job_id} for payment {payment_id}")

        result = execute_retry(job)

        if result.get("status") in ("retry_created", "link_created"):
            # Notify frontend FIRST, then mark complete
            notified = notify_frontend(job, result)
            if notified:
                store.complete_job(job_id)
                store.flush()
                print(f"[daemon] Job {job_id} executed and frontend notified: {result.get('status')}")
            else:
                store.fail_job(job_id, "Frontend notification failed — will retry")
                store.flush()
                print(f"[daemon] Job {job_id} executed but frontend notification FAILED")
        else:
            store.fail_job(job_id, result.get("message", "Unknown error"))
            store.flush()
            print(f"[daemon] Job {job_id} failed: {result.get('message')}")

        executed_count += 1

    return executed_count


def daemon_loop():
    """Main daemon loop — polls every POLL_INTERVAL seconds."""
    print(f"[daemon] Starting daemon worker (poll interval: {POLL_INTERVAL}s)")
    print(f"[daemon] Frontend URL: {FRONTEND_URL}")

    while True:
        try:
            executed = _process_due_jobs()
            if executed > 0:
                print(f"[daemon] Processed {executed} retry job(s)")
        except Exception as e:
            print(f"[daemon] Error in daemon loop: {e}", file=sys.stderr)

        time.sleep(POLL_INTERVAL)


# --- CLI Entry Point ---

def main():
    """Run the daemon worker as a standalone process."""
    daemon_loop()


if __name__ == "__main__":
    main()
