"""File-based state store for frontend payments, trails, pending actions, and scheduled jobs.

Persists state to JSON files in data/ so that server restarts do not lose
in-flight recovery sessions. Uses atomic writes (temp file + rename) and
a threading lock to prevent corruption under concurrent Flask/SocketIO access.

Cross-process safety: start.sh runs the frontend, the webhook receiver and the
daemon as separate processes, and each holds its own in-memory copy of these
files. `flush()` therefore does not blindly rewrite the files from memory — it
takes a cross-process file lock, re-reads the disk, and writes a per-key merge
in which only the keys THIS process changed override what is on disk. Change
detection is a shadow snapshot of the disk state taken at load/flush, so even
call sites that mutate a record dict in place are seen. Without this, any
daemon or webhook flush wrote its startup snapshot over every case the
frontend had advanced since — silently, wholesale, and atomically.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout


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
        #: section name -> {key: canonical JSON of the entry AS LAST SEEN ON
        #: DISK}. What differs from this at flush time is what this process
        #: changed; everything else yields to whatever another process wrote.
        self._shadow: dict[str, dict[str, str]] = {}

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

    #: (file name, attribute) for each persisted section, in flush order.
    _SECTIONS = (
        ("live_payments.json", "_payments"),
        ("live_trails.json", "_trails"),
        ("live_pending.json", "_pending"),
        ("live_jobs.json", "_jobs"),
    )

    @staticmethod
    def _canon(value: Any) -> str:
        return json.dumps(value, sort_keys=True, default=str)

    def _read_disk(self, name: str) -> dict:
        path = self._dir / name
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _file_lock(self) -> FileLock:
        """One lock file guards all four JSON files across processes."""
        return FileLock(str(self._dir / ".store.lock"), timeout=5)

    def _load(self) -> None:
        for name, attr in self._SECTIONS:
            disk = self._read_disk(name)
            getattr(self, attr).update(disk)
            self._shadow[name] = {k: self._canon(v) for k, v in disk.items()}

    def _merged(self, mem: dict, disk: dict, shadow: dict[str, str],
                absorbing: bool = False) -> dict:
        """Per-key merge: keys this process changed win; the rest is disk's.

        A key is "ours" when its canonical form differs from the shadow — the
        disk state as this process last saw it — which also catches call sites
        that mutate a record dict in place. A key present in the shadow but
        missing from memory was deleted here, and stays deleted. Everything
        else takes the disk version, so another process's newer write is never
        overwritten by our stale copy of a record we did not touch. Where the
        two are canonically equal we keep OUR object, so references held by
        callers stay live.

        `absorbing` extends the _ABSORBING rule across processes: even when
        our changed record wins the key, a `recovered` already on disk keeps
        its status and capture fields. Money in the bank is a fact — another
        process having seen it does not make it less so.
        """
        merged = dict(disk)
        for key, val in mem.items():
            c = self._canon(val)
            if shadow.get(key) != c:
                merged[key] = val               # locally added or changed
                disk_val = disk.get(key)
                if (absorbing and isinstance(disk_val, dict)
                        and isinstance(val, dict)
                        and disk_val.get("status") == self._ABSORBING
                        and val.get("status") != self._ABSORBING):
                    keep = {f: disk_val[f]
                            for f in ("status", "recovered_amount",
                                      "recovered_payment_id") if f in disk_val}
                    merged[key] = {**val, **keep}
            elif key in disk and self._canon(disk[key]) == c:
                merged[key] = val               # equal — preserve identity
        for key in shadow:
            if key not in mem:
                merged.pop(key, None)           # locally deleted
        return merged

    def flush(self) -> None:
        """Merge this process's changes onto the disk state and write it.

        On file-lock timeout the write proceeds WITHOUT the merge — the old
        wholesale behaviour — because refusing to persist a recovery over
        bookkeeping is worse than the race. It is loud about it.
        """
        with self._lock:
            try:
                with self._file_lock():
                    for name, attr in self._SECTIONS:
                        mem = getattr(self, attr)
                        merged = self._merged(mem, self._read_disk(name),
                                              self._shadow.get(name, {}),
                                              absorbing=(attr == "_payments"))
                        mem.clear()
                        mem.update(merged)
                        self._atomic_write(self._dir / name, mem)
                        self._shadow[name] = {k: self._canon(v)
                                              for k, v in mem.items()}
            except Timeout:
                print(f"[state_store] WARNING: could not lock {self._dir} in 5s; "
                      f"writing WITHOUT cross-process merge", flush=True)
                for name, attr in self._SECTIONS:
                    mem = getattr(self, attr)
                    self._atomic_write(self._dir / name, mem)
                    self._shadow[name] = {k: self._canon(v) for k, v in mem.items()}

    def refresh(self) -> None:
        """Pull other processes' writes into memory without writing anything.

        The daemon loads once at startup and then polls its snapshot, so a job
        the frontend scheduled a minute later was invisible forever. After a
        refresh, locally-changed keys are still ours (and still count as dirty
        at the next flush); everything else is what is on disk right now.
        """
        with self._lock:
            try:
                with self._file_lock():
                    for name, attr in self._SECTIONS:
                        mem = getattr(self, attr)
                        disk = self._read_disk(name)
                        merged = self._merged(mem, disk,
                                              self._shadow.get(name, {}),
                                              absorbing=(attr == "_payments"))
                        mem.clear()
                        mem.update(merged)
                        # The shadow is the DISK state: keys where our local
                        # change won stay dirty for the next flush.
                        self._shadow[name] = {k: self._canon(v)
                                              for k, v in disk.items()}
            except Timeout:
                print(f"[state_store] WARNING: could not lock {self._dir} in 5s; "
                      f"refresh skipped", flush=True)

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

    def cancel_jobs_for(self, payment_id: str, reason: str = "") -> list[str]:
        """Cancel this case's pending jobs. Returns the ids cancelled.

        A case that ends still has its timers running. Live (pay_4tnzl57fu):
        the agent asked to be woken in 16 minutes, the customer paid 15 seconds
        later, and the case closed as recovered — leaving a `wake_agent` job
        that would still fire afterwards. The hand-off refuses it (the case is
        closed), so nothing is chased, but a settled case should not be leaving
        alarms set for itself; the ending is the moment to clear them.
        """
        cancelled = []
        with self._lock:
            for job_id, job in self._jobs.items():
                if job.get("payment_id") != payment_id:
                    continue
                if job.get("status") != "pending":
                    continue
                job["status"] = "cancelled"
                job["cancelled_at"] = datetime.now(timezone.utc).isoformat()
                if reason:
                    job["cancelled_reason"] = reason[:200]
                cancelled.append(job_id)
        return cancelled

    def fail_job(self, job_id: str, error: str = "") -> None:
        """Mark a job as failed."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = error
                self._jobs[job_id]["failed_at"] = datetime.now(timezone.utc).isoformat()
