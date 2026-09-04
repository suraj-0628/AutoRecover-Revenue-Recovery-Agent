"""The batch of cases a human has to work.

Stage 6 of RECOVERY-LADDER.md. When the ladder runs out — every channel tried,
attempts exhausted, or the agent judging that no automation can help — the case
goes here rather than quietly ending.

What a queue entry has to contain
---------------------------------
The previous escalation wrote `{ticket_id, payment_id, reason, status}` and
nothing else. Someone picking that up has no amount, no way to contact the
customer, no idea what was already tried, and no link to send. They would have to
reconstruct the case from logs before they could say a word to anybody — so in
practice the queue was a place cases went to be forgotten.

An entry here carries what a person needs to act in one glance: the money, the
customer, what the agent tried and what the customer did about it, what was
offered, and the live link.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_QUEUE_PATH = Path(os.getenv("ESCALATION_QUEUE_PATH", "data/escalations/queue.jsonl"))
_lock = threading.Lock()

OPEN = "open"
RESOLVED = "resolved"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_ticket_id(payment_id: str) -> str:
    """Stable, readable, and never malformed.

    The old scheme was `f"ESC-{payment_id[-8:]}-{stamp}"`, which produced `ESC--…`
    for an empty payment_id — several of those are sitting in data/escalations.
    """
    tail = "".join(c for c in str(payment_id or "") if c.isalnum())[-8:]
    if not tail:
        tail = uuid.uuid4().hex[:8]
    # The timestamp alone collides: escalating the same payment twice inside one
    # second produced identical ids, and since the file is folded by ticket_id the
    # second entry silently overwrote the first one's resolution.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"ESC-{tail}-{stamp}-{uuid.uuid4().hex[:6]}"


def enqueue(
    payment_id: str,
    reason: str,
    *,
    amount: float = 0.0,
    currency: str = "INR",
    customer: dict[str, Any] | None = None,
    attempts: list[dict[str, Any]] | None = None,
    customer_signals: list[str] | None = None,
    offer: dict[str, Any] | None = None,
    recovery_link: str = "",
    failure_code: str = "",
    source: str = "agent",
) -> dict[str, Any]:
    """Add a case to the human queue, with everything needed to act on it."""
    ticket = {
        "ticket_id": make_ticket_id(payment_id),
        "payment_id": payment_id,
        "status": OPEN,
        "created_at": _now(),
        "reason": reason,
        "amount": round(float(amount or 0), 2),
        "currency": currency,
        "failure_code": failure_code,
        "customer": {k: v for k, v in (customer or {}).items() if v},
        "attempts": attempts or [],
        # What the customer actually did — dismissing a notification in four
        # seconds is worth more to a human than any status field.
        "customer_signals": customer_signals or [],
        "offer": offer or {},
        "recovery_link": recovery_link,
        "source": source,
    }

    with _lock:
        _QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Idempotent per payment: re-escalating a case already waiting on a human
        # would bury the queue in duplicates of the same problem.
        #
        # This must read the *folded* state, not raw rows. The file is
        # append-only, so a resolved ticket still has its original `open` row on
        # disk — scanning raw rows matched that stale entry and refused to
        # re-escalate a case that had already been closed and failed again.
        for existing in _latest_by_ticket().values():
            if existing.get("payment_id") == payment_id and existing.get("status") == OPEN:
                return existing
        with _QUEUE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ticket) + "\n")

    # Compliant escalation has to be demonstrable, not asserted: the log records
    # who was handed to a human, on what grounds, and after which rungs — so a
    # batch report can show its escalations came at the end of the ladder rather
    # than instead of it.
    try:
        from recovery_agent import audit
        from recovery_agent.state_store import StateStore
        rec = StateStore().get_payment(payment_id) or {}
        audit.record(audit.ESCALATION_RAISED, payment_id=payment_id,
                     batch_run_id=rec.get("batch_run_id") or "",
                     actor=source, reason=reason, amount_rupees=amount,
                     ticket_id=ticket["ticket_id"],
                     climbed=sorted(rec.get("ladder") or {}),
                     customer_signals=customer_signals or [])
    except Exception:
        pass                        # a queue write must not fail on logging

    return ticket


def _read_all() -> list[dict[str, Any]]:
    if not _QUEUE_PATH.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in _QUEUE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue                      # a torn line must not hide the rest
    return out


def _latest_by_ticket() -> dict[str, dict[str, Any]]:
    """The file is append-only, so the last entry per ticket wins."""
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_all():
        tid = row.get("ticket_id")
        if tid:
            latest[tid] = row
    return latest


def list_tickets(status: str | None = OPEN, limit: int = 200) -> list[dict[str, Any]]:
    rows = list(_latest_by_ticket().values())
    if status:
        rows = [r for r in rows if r.get("status") == status]
    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return rows[:limit]


def resolve(ticket_id: str, outcome: str = "", by: str = "human") -> dict[str, Any] | None:
    """Close a ticket by appending its resolved state — history is never rewritten."""
    current = _latest_by_ticket().get(ticket_id)
    if current is None:
        return None
    closed = {**current, "status": RESOLVED, "resolved_at": _now(),
              "outcome": outcome, "resolved_by": by}
    with _lock:
        _QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _QUEUE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(closed) + "\n")
    return closed


def summary() -> dict[str, Any]:
    rows = list(_latest_by_ticket().values())
    open_rows = [r for r in rows if r.get("status") == OPEN]
    return {
        "open": len(open_rows),
        "resolved": len(rows) - len(open_rows),
        "open_value": round(sum(float(r.get("amount") or 0) for r in open_rows), 2),
        "currency": (open_rows[0].get("currency") if open_rows else "INR"),
    }
