"""File-based state store for frontend payments, trails, pending actions, and scheduled jobs.

Persists state to JSON files in data/ so that server restarts do not lose
in-flight recovery sessions. Uses atomic writes (temp file + rename) and
a threading lock to prevent corruption under concurrent Flask/SocketIO access.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DATA_DIR = Path(os.getenv("STATE_DIR", "data"))


_INSTANCES: dict[Path, "StateStore"] = {}
_INSTANCES_LOCK = threading.Lock()


class StateStore:
    """Thread-safe, file-backed store for live payment state.

    ONE instance per directory, per process. This is not an optimisation — a
    second instance silently destroys the first one's writes.

    Every writer here holds the whole state in memory and `flush()` rewrites the
    files wholesale. The frontend keeps a long-lived store; tools were each
    building their own, writing, and flushing. The tool's write landed on disk,
    then the frontend's next flush — from an instance that had loaded before the
    tool ran — wrote its stale copy back over it.

    Live, that made the entire recovery ladder inert: `send_recovery_notification`
    returned `"rung": "offer"`, proving the rung had been recorded, while the
    case record read `ladder: []`. Escalation gating reads that record, so it
    could never see a rung as climbed.

        a = StateStore(); b = StateStore()
        b.update_payment("x", ladder={...}); b.flush()   # on disk
        a.flush()                                        # gone
    """

    def __new__(cls, data_dir: Path | None = None) -> "StateStore":
        key = Path(data_dir or _DATA_DIR).resolve()
        with _INSTANCES_LOCK:
            inst = _INSTANCES.get(key)
            if inst is None:
                inst = super().__new__(cls)
                inst._initialised = False
                _INSTANCES[key] = inst
            return inst

    def __init__(self, data_dir: Path | None = None) -> None:
        if getattr(self, "_initialised", False):
            return                      # already loaded; do not wipe live state
        self._dir = Path(data_dir or _DATA_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._payments: dict[str, dict] = {}
        self._trails: dict[str, list[dict]] = {}
        self._pending: dict[str, dict] = {}
        self._jobs: dict[str, dict] = {}

        self._load()
        self._initialised = True

    @classmethod
    def reset_instances(cls) -> None:
        """Drop the cache. For tests that need a store to reload from disk."""
        with _INSTANCES_LOCK:
            _INSTANCES.clear()

    # ── persistence ──────────────────────────────────────────────

    def _atomic_write(self, path: Path, data: Any) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, str(path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _load(self) -> None:
        for name, target in (
            ("live_payments.json", self._payments),
            ("live_trails.json", self._trails),
            ("live_pending.json", self._pending),
            ("live_jobs.json", self._jobs),
        ):
            path = self._dir / name
            if path.exists():
                try:
                    with open(path) as f:
                        target.update(json.load(f))
                except (json.JSONDecodeError, OSError):
                    pass

    def flush(self) -> None:
        """Write current state to disk. Call after critical mutations."""
        with self._lock:
            self._atomic_write(self._dir / "live_payments.json", self._payments)
            self._atomic_write(self._dir / "live_trails.json", self._trails)
            self._atomic_write(self._dir / "live_pending.json", self._pending)
            self._atomic_write(self._dir / "live_jobs.json", self._jobs)

    # ── payments ─────────────────────────────────────────────────

    def get_payment(self, payment_id: str) -> dict | None:
        return self._payments.get(payment_id)

    def has_payment(self, payment_id: str) -> bool:
        return payment_id in self._payments

    def save_payment(self, payment_id: str, data: dict) -> None:
        with self._lock:
            self._payments[payment_id] = data

    #: Money in the bank is a fact, not a status. Once a case is `recovered`
    #: nothing may write over it — not a failed closing run, not a stale
    #: in-flight write, not a tool-name fallback. A recovery that reads as a
    #: loss is worse than no record at all, and it has happened twice: an LLM
    #: 404 on a bookkeeping run rewrote a captured INR 2,374.05 to `failed`,
    #: and a run that only scheduled a retry wrote `failed` over a live case.
    _ABSORBING = "recovered"

    def update_payment(self, payment_id: str, force: bool = False,
                       **fields: Any) -> None:
        with self._lock:
            rec = self._payments.get(payment_id)
            if rec is None:
                return
            if (not force and rec.get("status") == self._ABSORBING
                    and "status" in fields
                    and fields["status"] != self._ABSORBING):
                fields = {k: v for k, v in fields.items() if k != "status"}
            rec.update(fields)

    def remove_payment(self, payment_id: str) -> None:
        with self._lock:
            self._payments.pop(payment_id, None)

    def all_payments(self) -> dict[str, dict]:
        return dict(self._payments)

    def payment_values(self) -> list[dict]:
        return list(self._payments.values())

    # ── trails ───────────────────────────────────────────────────

    def get_trail(self, payment_id: str) -> list[dict]:
        return self._trails.setdefault(payment_id, [])

    def set_trail(self, payment_id: str, trail: list[dict]) -> None:
        with self._lock:
            self._trails[payment_id] = trail

    # ── pending actions ──────────────────────────────────────────

    def get_pending(self, payment_id: str) -> dict | None:
        return self._pending.get(payment_id)

    def has_pending(self, payment_id: str) -> bool:
        return payment_id in self._pending

    def save_pending(self, payment_id: str, data: dict) -> None:
        with self._lock:
            self._pending[payment_id] = data

    def remove_pending(self, payment_id: str) -> None:
        with self._lock:
            self._pending.pop(payment_id, None)

    # ── scheduled jobs ───────────────────────────────────────────

    def schedule_job(
        self,
        job_id: str,
        payment_id: str,
        target_time: str,
        action: str,
        metadata: dict | None = None,
    ) -> None:
        """Persist a scheduled retry job to disk."""
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "payment_id": payment_id,
                "target_time": target_time,
                "action": action,
                "metadata": metadata or {},
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

    def get_due_jobs(self, now: datetime | None = None) -> list[dict]:
        """Return all pending jobs whose target_time <= now."""
        now = now or datetime.now(timezone.utc)
        due = []
        for job in self._jobs.values():
            if job.get("status") != "pending":
                continue
            try:
                target = datetime.fromisoformat(job["target_time"])
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                if target <= now:
                    due.append(job)
            except (ValueError, KeyError):
                continue
        return due

    def complete_job(self, job_id: str) -> None:
        """Mark a job as completed."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

    def fail_job(self, job_id: str, error: str = "") -> None:
        """Mark a job as failed."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = error
                self._jobs[job_id]["failed_at"] = datetime.now(timezone.utc).isoformat()
