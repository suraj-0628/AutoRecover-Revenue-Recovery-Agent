"""Background Daemon Worker — executes scheduled retries autonomously.

When datetime.now() >= target_timestamp, the daemon triggers the Razorpay SDK
to execute the retry, completely independent of the frontend UI thread.

Polls the retry_jobs store every 30 seconds.
Executes retries via the Razorpay SDK.
Reports results back to the frontend via HTTP POST.

Usage:
    python -m recovery_agent.daemon_worker
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# --- Job Store ---

_JOBS_DIR = Path(os.getenv("RETRY_JOBS_DIR", "/tmp/recovery_agent_jobs"))
_JOBS_DIR.mkdir(parents=True, exist_ok=True)

POLL_INTERVAL = int(os.getenv("DAEMON_POLL_INTERVAL", "30"))  # seconds
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5002")

_jobs_lock = threading.Lock()


def _job_file(job_id: str) -> Path:
    return _JOBS_DIR / f"{job_id}.json"


def save_job(job: dict[str, Any]) -> None:
    """Persist a retry job to disk."""
    job_id = job["job_id"]
    with _jobs_lock:
        path = _job_file(job_id)
        path.write_text(json.dumps(job, indent=2, default=str))


def load_pending_jobs() -> list[dict[str, Any]]:
    """Load all pending jobs from disk."""
    jobs = []
    with _jobs_lock:
        for path in _JOBS_DIR.glob("*.json"):
            try:
                job = json.loads(path.read_text())
                if job.get("status") == "pending":
                    jobs.append(job)
            except Exception:
                continue
    return jobs


def mark_job_executed(job_id: str, result: dict[str, Any]) -> None:
    """Mark a job as executed and persist the result."""
    with _jobs_lock:
        path = _job_file(job_id)
        if path.exists():
            job = json.loads(path.read_text())
            job["status"] = "executed"
            job["executed_at"] = datetime.now(timezone.utc).isoformat()
            job["result"] = result
            path.write_text(json.dumps(job, indent=2, default=str))


def mark_job_failed(job_id: str, error: str) -> None:
    """Mark a job as failed."""
    with _jobs_lock:
        path = _job_file(job_id)
        if path.exists():
            job = json.loads(path.read_text())
            job["status"] = "failed"
            job["failed_at"] = datetime.now(timezone.utc).isoformat()
            job["error"] = error
            path.write_text(json.dumps(job, indent=2, default=str))


# --- Retry Execution ---

def execute_retry(job: dict[str, Any]) -> dict[str, Any]:
    """Execute a retry job via the Razorpay SDK."""
    from recovery_agent.razorpay_client import RazorpayClient

    payment_id = job.get("payment_id", "")
    amount = job.get("amount", 0)
    action = job.get("action", "retry_payment")
    method = job.get("method", "card")
    order_id = job.get("order_id", "")

    client = RazorpayClient()

    if not client.is_configured:
        return {
            "status": "error",
            "message": "Razorpay client not configured",
            "payment_id": payment_id,
        }

    try:
        if action == "retry_payment" or action == "update_payment_method":
            # Create a new order for retry
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
            # Create a payment link
            link = client.create_payment_link(
                amount=amount,
                customer=job.get("customer", {}),
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
    """Check all pending jobs and execute those that are due. Returns count executed."""
    now = datetime.now(timezone.utc)
    pending = load_pending_jobs()
    executed_count = 0

    for job in pending:
        target_ts = job.get("target_timestamp", "")
        if not target_ts:
            continue

        try:
            target_time = datetime.fromisoformat(target_ts)
            if target_time.tzinfo is None:
                target_time = target_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        if now >= target_time:
            print(f"[daemon] Executing retry job {job['job_id']} for payment {job.get('payment_id', '')}")
            result = execute_retry(job)

            if result.get("status") in ("retry_created", "link_created"):
                mark_job_executed(job["job_id"], result)
                notify_frontend(job, result)
                print(f"[daemon] Job {job['job_id']} executed: {result.get('status')}")
            else:
                mark_job_failed(job["job_id"], result.get("message", "Unknown error"))
                print(f"[daemon] Job {job['job_id']} failed: {result.get('message')}")

            executed_count += 1

    return executed_count


def daemon_loop():
    """Main daemon loop — polls every POLL_INTERVAL seconds."""
    print(f"[daemon] Starting daemon worker (poll interval: {POLL_INTERVAL}s)")
    print(f"[daemon] Jobs directory: {_JOBS_DIR}")
    print(f"[daemon] Frontend URL: {FRONTEND_URL}")

    while True:
        try:
            executed = _process_due_jobs()
            if executed > 0:
                print(f"[daemon] Processed {executed} retry job(s)")
        except Exception as e:
            print(f"[daemon] Error in daemon loop: {e}", file=sys.stderr)

        time.sleep(POLL_INTERVAL)


# --- Public API for registering jobs ---

def register_retry_job(
    payment_id: str,
    amount: float,
    target_timestamp: str,
    action: str = "retry_payment",
    method: str = "card",
    customer: dict | None = None,
    order_id: str = "",
    reason: str = "",
    confidence: float = 0.5,
) -> dict[str, Any]:
    """Register a retry job with the daemon. Returns the job dict."""
    import uuid

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job = {
        "job_id": job_id,
        "payment_id": payment_id,
        "amount": amount,
        "target_timestamp": target_timestamp,
        "action": action,
        "method": method,
        "customer": customer or {},
        "order_id": order_id,
        "reason": reason,
        "confidence": confidence,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    save_job(job)
    print(f"[daemon] Registered retry job {job_id} for {payment_id} at {target_timestamp}")
    return job


def get_job_status(job_id: str) -> dict[str, Any] | None:
    """Get the status of a specific job."""
    path = _job_file(job_id)
    if path.exists():
        return json.loads(path.read_text())
    return None


def list_all_jobs() -> list[dict[str, Any]]:
    """List all jobs (pending, executed, failed)."""
    jobs = []
    for path in _JOBS_DIR.glob("*.json"):
        try:
            jobs.append(json.loads(path.read_text()))
        except Exception:
            continue
    return jobs


# --- CLI Entry Point ---

def main():
    """Run the daemon worker as a standalone process."""
    daemon_loop()


if __name__ == "__main__":
    main()
