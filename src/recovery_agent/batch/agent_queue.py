"""The exception queue: one worker, feeding the outliers to the real agent.

A wave's report saying "3 to the agent" is a promise. Until this module, it was
an unkept one — the cases were listed and nobody worked them. This is the desk
they land on.

One worker, deliberately. The LLM quota is a single shared resource; N threads
sleeping on the same gate cost N threads, wake in a burst against the limit,
make ordering unfair and aborts impossible. One queue makes the wait honest:
exceptions are worked in the order they arrived, the depth is a number the
dashboard can show, and the honest throughput on the free proxy — roughly a
case a minute — is printed rather than pretended away. `AGENT_WORKERS` raises
throughput when a paid key exists; that is configuration, not architecture.

Each case is stamped with the run that referred it before the session starts,
so whatever the agent recovers still lands on the batch's measured total.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from recovery_agent import audit

#: Referrals waiting beyond this are refused at enqueue rather than silently
#: growing a backlog nobody is reading.
MAX_DEPTH = int(os.getenv("AGENT_QUEUE_DEPTH", "50"))
WORKERS = max(1, int(os.getenv("AGENT_WORKERS", "1")))


@dataclass
class Referral:
    payment_id: str
    why: str
    batch_run_id: str = ""
    cycle_id: str = ""
    enqueued_at: str = field(default_factory=lambda: datetime.now(
        timezone.utc).isoformat())


class AgentQueue:
    """A bounded queue drained by daemon workers through one session runner."""

    def __init__(self, runner: Callable[[Referral], None],
                 workers: int = WORKERS, depth: int = MAX_DEPTH):
        self._runner = runner
        self._q: queue.Queue[Referral] = queue.Queue(maxsize=depth)
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        self.worked = 0
        self.failed = 0
        self._threads = [
            threading.Thread(target=self._drain, daemon=True,
                             name=f"agent-queue-{i}")
            for i in range(max(1, workers))]
        for t in self._threads:
            t.start()

    def submit(self, referral: Referral) -> str:
        """Queue one exception for a full agent session. Returns '' or why not.

        One referral per case at a time: a case already waiting does not need
        a second place in line, and the wave loop re-refers every wave.
        """
        with self._lock:
            if referral.payment_id in self._seen:
                return "already_queued"
        try:
            self._q.put_nowait(referral)
        except queue.Full:
            audit.record(audit.CASE_EXCEPTION, payment_id=referral.payment_id,
                         batch_run_id=referral.batch_run_id,
                         actor="agent_queue",
                         reason=f"queue full at {MAX_DEPTH}; not enqueued")
            return "queue_full"
        with self._lock:
            self._seen.add(referral.payment_id)
        audit.record(audit.CASE_SELECTED, payment_id=referral.payment_id,
                     batch_run_id=referral.batch_run_id, actor="agent_queue",
                     reason=f"queued for the agent: {referral.why}",
                     depth=self.depth())
        return ""

    def depth(self) -> int:
        return self._q.qsize()

    def stats(self) -> dict[str, Any]:
        return {"depth": self.depth(), "worked": self.worked,
                "failed": self.failed, "workers": len(self._threads)}

    def _drain(self) -> None:
        while True:
            referral = self._q.get()
            try:
                self._work_one(referral)
            finally:
                with self._lock:
                    self._seen.discard(referral.payment_id)
                self._q.task_done()

    def _work_one(self, referral: Referral) -> None:
        started = time.monotonic()
        try:
            if referral.batch_run_id:
                # Whatever the session recovers must land on the run that
                # referred it, or the batch under-counts its own outcome.
                from recovery_agent.batch.executor import _stamp
                _stamp(referral.payment_id, referral.batch_run_id)
            self._runner(referral)
            self.worked += 1
            audit.record(audit.ACTION_RESULT, payment_id=referral.payment_id,
                         batch_run_id=referral.batch_run_id,
                         actor="agent_queue", action="agent_session",
                         result="ok",
                         seconds=round(time.monotonic() - started, 1))
        except Exception as exc:
            self.failed += 1
            audit.record(audit.CASE_EXCEPTION, payment_id=referral.payment_id,
                         batch_run_id=referral.batch_run_id,
                         actor="agent_queue",
                         reason=f"session failed: {type(exc).__name__}: {exc}")


_default: AgentQueue | None = None
_default_lock = threading.Lock()


def get(runner: Callable[[Referral], None] | None = None) -> AgentQueue | None:
    """The process-wide queue. First caller with a runner creates it."""
    global _default
    with _default_lock:
        if _default is None and runner is not None:
            _default = AgentQueue(runner)
        return _default
