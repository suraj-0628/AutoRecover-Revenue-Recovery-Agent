"""One gate for every LLM call, that stays a gate under concurrency.

The old pacing was a module attribute and a sleep: each thread read the same
`_last_call_time`, computed the same delay, slept it, and woke *together* — N
threads turned an 8-second gap into a burst of N followed by silence, which is
the exact opposite of what a provider's rate limit wants. It also could not be
asked anything: how long the system has spent waiting was nobody's number.

This bucket reserves instead of sleeping blind: under one lock each caller
takes the next free slot and moves the pointer, then waits for its own slot
outside the lock. Grants are spaced by construction, arrival order is
preserved, and a champion session, a queued exception and a live hand-off can
all draw from the same allowance without ever synchronising into a burst.

What it deliberately is not: a queue with priorities. Every caller is a case
some customer is waiting on; fairness here is arrival order, full stop.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

#: Matches the 8-second gap the old gate enforced (~7.5 calls/minute against
#: a free-tier ~5 RPM quota with headroom). Env-tunable because a paid key
#: changes the number, not the architecture.
DEFAULT_CALLS_PER_MINUTE = 7.5


class TokenBucket:
    """Reservation-based pacing on a monotonic clock."""

    def __init__(self, calls_per_minute: float = DEFAULT_CALLS_PER_MINUTE):
        self.interval = (60.0 / calls_per_minute) if calls_per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next_free = 0.0
        self.calls = 0
        self.waited_seconds = 0.0

    def acquire(self) -> float:
        """Block until this caller's slot arrives. Returns seconds waited."""
        if self.interval <= 0:
            with self._lock:
                self.calls += 1
            return 0.0
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_free)
            self._next_free = slot + self.interval
            self.calls += 1
        wait = slot - now
        if wait > 0:
            time.sleep(wait)
            with self._lock:
                self.waited_seconds += wait
        return max(0.0, wait)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"calls": self.calls,
                    "waited_seconds": round(self.waited_seconds, 2),
                    "interval_seconds": round(self.interval, 2)}


_gate: TokenBucket | None = None
_gate_lock = threading.Lock()


def llm_gate() -> TokenBucket:
    """The process-wide gate for model calls.

    Zero-interval under pytest for the same reason the old gate was: the pause
    respects a real provider, and charging it to a scripted model is how a
    fast suite stops being run.
    """
    global _gate
    with _gate_lock:
        if _gate is None:
            if os.getenv("PYTEST_CURRENT_TEST"):
                rate = 0.0
            else:
                try:
                    rate = float(os.getenv("LLM_CALLS_PER_MINUTE",
                                           DEFAULT_CALLS_PER_MINUTE))
                except (TypeError, ValueError):
                    rate = DEFAULT_CALLS_PER_MINUTE
            _gate = TokenBucket(rate)
        return _gate
