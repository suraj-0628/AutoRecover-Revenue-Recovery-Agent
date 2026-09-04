"""Who is actually on the checkout page right now.

`deliver_page_push` used to `socketio.emit` globally and return
`{"status": "delivered"}` unconditionally — the page filters by payment id in
the browser, so the server never knew whether anyone was there. That made
`page_push` a phantom rung: `ladder.record_rung` fired for customers who had
closed the tab hours earlier, the ladder advanced past a contact that never
happened, and the escalation ticket carried it as a `customer_signal`.

A rung is a record of having reached someone. It has to be true.

This lives in its own module rather than in `frontend.py` because `ladder.py`
needs it, and a domain rule importing the web transport is a layering inversion
— one that fails as a circular import in some processes and silently succeeds
in others, which is worse than either.
"""
from __future__ import annotations

import threading

_watchers: dict[str, set[str]] = {}
_lock = threading.Lock()


def watch(sid: str, payment_id: str) -> None:
    """A checkout page has announced which payment it is showing."""
    if not sid or not payment_id:
        return
    with _lock:
        _watchers.setdefault(payment_id, set()).add(sid)


def forget(sid: str) -> None:
    """That page has gone."""
    with _lock:
        for pid in [p for p, sids in _watchers.items() if sid in sids]:
            _watchers[pid].discard(sid)
            if not _watchers[pid]:
                del _watchers[pid]


def is_live(payment_id: str) -> bool:
    with _lock:
        return bool(_watchers.get(payment_id))


def live_payments() -> list[str]:
    with _lock:
        return sorted(_watchers)


def reset() -> None:
    """For tests."""
    with _lock:
        _watchers.clear()
