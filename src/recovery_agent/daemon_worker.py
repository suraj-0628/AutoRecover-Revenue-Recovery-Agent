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
from datetime import datetime, timezone
from typing import Any


POLL_INTERVAL = int(os.getenv("DAEMON_POLL_INTERVAL", "60"))  # seconds
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5002")


# --- Retry Execution ---

def register_retry_job(
    payment_id: str,
    amount: float,
    target_timestamp: str,
    action: str = "retry_payment",
    method: str = "card",
    customer: dict[str, Any] | None = None,
    reason: str = "",
    confidence: float = 0.5,
    job_id: str = "",
) -> dict[str, Any]:
    """Queue a scheduled retry for this worker to execute.

    This function did not exist. `frontend.py` imported it whenever the agent
    scheduled a silent retry, so the whole case died with

        Agent Execution Error: cannot import name 'register_retry_job'

    right after the agent had done the correct thing. `StateStore.schedule_job`
    was already written and had zero callers — the two halves of the silent-retry
    path were never joined, so Tier 1 could not complete even once.
    """
    from recovery_agent.state_store import StateStore

    store = StateStore()
    # Adopt the job the tool already created rather than minting a second one.
    #
    # `retry_in_hours` schedules a job and returns its id; this then created
    # ANOTHER with its own id, so one retry left two pending jobs and the daemon
    # would execute it twice — two retry orders, two notifications, for a single
    # decision the agent made once.
    job_id = job_id or f"job_{payment_id}_{int(time.time())}"
    metadata = {
        "amount": float(amount or 0),
        "method": method,
        "customer": customer or {},
        "reason": reason,
        "confidence": float(confidence or 0),
    }
    store.schedule_job(
        job_id=job_id,
        payment_id=payment_id,
        target_time=target_timestamp,
        action=action,
        metadata=metadata,
    )
    store.flush()

    print(f"[daemon] registered {job_id} for {payment_id} at {target_timestamp}", flush=True)
    return {
        "status": "scheduled",
        "job_id": job_id,
        "payment_id": payment_id,
        "target_timestamp": target_timestamp,
        "action": action,
        "amount": float(amount or 0),
        "reason": reason,
        "confidence": float(confidence or 0),
    }


def execute_retry(job: dict[str, Any]) -> dict[str, Any]:
    """Execute a retry job via the Razorpay SDK."""
    from recovery_agent.razorpay_client import RazorpayClient

    payment_id = job.get("payment_id", "")
    amount = job.get("metadata", {}).get("amount", 0)
    action = job.get("action", "retry_payment")
    method = job.get("metadata", {}).get("method", "card")

    # A wake-up reaches nobody and moves no money, so it needs no gateway —
    # and must not be refused when Razorpay is unconfigured. It exists so the
    # agent's own `wait_for_customer` is a promise the system keeps.
    if action == "wake_agent":
        return {
            "status": "woken",
            "payment_id": payment_id,
            "reason": job.get("metadata", {}).get("reason", ""),
            "message": "the wait the agent asked for has elapsed",
        }

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

def _process_due_jobs() -> int:
    """Check StateStore for due jobs and execute them. Returns count executed."""
    from recovery_agent.state_store import StateStore

    store = StateStore()
    # The daemon loaded its snapshot at startup; jobs the frontend scheduled
    # since then are invisible without this. It also keeps this process's
    # flushes from writing that startup snapshot over the frontend's work.
    store.refresh()
    now = datetime.now(timezone.utc)
    due_jobs = store.get_due_jobs(now)
    executed_count = 0

    for job in due_jobs:
        job_id = job.get("job_id", "")
        payment_id = job.get("payment_id", "")
        print(f"[daemon] Executing retry job {job_id} for payment {payment_id}")

        result = execute_retry(job)

        if result.get("status") in ("retry_created", "link_created", "woken"):
            store.complete_job(job_id)
            store.flush()
            notify_frontend(job, result)
            print(f"[daemon] Job {job_id} executed: {result.get('status')}")
        else:
            store.fail_job(job_id, result.get("message", "Unknown error"))
            store.flush()
            print(f"[daemon] Job {job_id} failed: {result.get('message')}")

        executed_count += 1

    return executed_count


#: How often to pull SuperU's billed call costs into the ledger. Their
#: call-ended webhook carries only a uuid and needs a public URL, so polling
#: the log is the reliable path for a local deployment. The read places no
#: calls and spends no credits; reconciliation is keyed on each call's uuid,
#: so polling can never double-count.
VOICE_RECONCILE_INTERVAL = int(os.getenv("VOICE_RECONCILE_INTERVAL", "300"))


def _reconcile_voice_costs() -> None:
    """Fold SuperU's own per-call charges into the cost ledger. Never raises —
    accounting must not be able to stop scheduled retries from firing."""
    try:
        from recovery_agent.integrations import superu_reconcile
        out = superu_reconcile.reconcile()
        if out.get("recorded"):
            print(f"[daemon] voice costs: recorded {out['recorded']} billed "
                  f"call(s), INR {out['inr']:.2f}", flush=True)
    except Exception as e:
        print(f"[daemon] voice cost reconcile failed: {e}", file=sys.stderr)


def daemon_loop():
    """Main daemon loop — polls every POLL_INTERVAL seconds."""
    print(f"[daemon] Starting daemon worker (poll interval: {POLL_INTERVAL}s)")
    print(f"[daemon] Frontend URL: {FRONTEND_URL}")

    last_reconcile = 0.0
    while True:
        try:
            executed = _process_due_jobs()
            if executed > 0:
                print(f"[daemon] Processed {executed} retry job(s)")
        except Exception as e:
            print(f"[daemon] Error in daemon loop: {e}", file=sys.stderr)

        now = time.monotonic()
        if now - last_reconcile >= VOICE_RECONCILE_INTERVAL:
            last_reconcile = now
            _reconcile_voice_costs()

        time.sleep(POLL_INTERVAL)


# --- CLI Entry Point ---

def main():
    """Run the daemon worker as a standalone process."""
    from recovery_agent.observability import init_observability
    init_observability("daemon")
    daemon_loop()


if __name__ == "__main__":
    main()
