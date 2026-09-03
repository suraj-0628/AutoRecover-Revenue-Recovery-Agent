"""The recovery ladder: which rungs a case has climbed, and what comes next.

Escalation used to be whatever the agent reached for when a tool refused it. On
`pay_rxfo0unaq` the approval gate blocked a discounted link at rung 2 and the
agent filed a human ticket for a INR 1,19,970 order that had been contacted
exactly once — a silent in-page push. A human queue is the *last* resort, not
the fallback for a blocked call.

Enforcing that in the system prompt would not hold; a prompt is advice. This
module is the record, and `escalate_to_human` consults it before it will file
anything. The rungs are deliberately coarse — the agent still decides what to
say, when, and to whom. It just cannot skip to the end.

The store record is the source of truth, not the Case object: a Case is rebuilt
from scratch on every hand-off run, so anything held only there is forgotten
between rungs — which is exactly how a ladder gets skipped.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

# Rung order is the recovery policy. Each entry is
#   (key, human label, what makes it possible)
RUNGS: list[tuple[str, str]] = [
    ("page_push", "silent in-page notification"),
    ("offer", "discounted offer by email, shown on the page at the same time"),
    ("voice_call", "voice call to ask why and negotiate within policy"),
    ("post_call_email", "email with the agreed pay link after the call"),
    ("alternate_path", "a different route than the ones already tried"),
]

#: Minimum gap between the offer and the call. The customer is given a real
#: chance to use the offer before the phone rings.
VOICE_DELAY_MINUTES = int(os.getenv("LADDER_VOICE_DELAY_MINUTES", "15"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def voice_available() -> bool:
    return os.getenv("VOICE_CALLS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def rung_possible(key: str, record: dict) -> tuple[bool, str]:
    """Can this rung ever be climbed for this case? Returns (possible, why not)."""
    customer = record.get("customer") or {}
    email = customer.get("email") or record.get("customer_email") or ""
    phone = customer.get("contact") or record.get("customer_phone") or ""

    if key in ("offer", "post_call_email", "alternate_path") and not (email or phone):
        return False, "no email or phone on file for this customer"
    if key == "voice_call":
        if not voice_available():
            return False, "voice calling is switched off for this deployment"
        if not phone:
            return False, "no phone number on file"
    if key == "post_call_email":
        if not climbed(record, "voice_call"):
            return False, "there was no call to follow up"
        if not email:
            return False, "no email on file"
    return True, ""


def climbed(record: dict, key: str) -> bool:
    return bool((record.get("ladder") or {}).get(key))


def record_rung(payment_id: str, key: str, detail: str = "") -> None:
    """Mark a rung as climbed. Safe to call more than once; the first wins."""
    if key not in dict(RUNGS):
        return
    try:
        from recovery_agent.state_store import StateStore
        store = StateStore()
        rec = store.get_payment(payment_id)
        if rec is None:
            return
        ladder = dict(rec.get("ladder") or {})
        if key in ladder:
            return
        ladder[key] = {"at": _now(), "detail": detail[:300]}
        store.update_payment(payment_id, ladder=ladder)
        store.flush()
    except Exception:
        pass                            # never let bookkeeping break a recovery


def actions_tried(record: dict) -> list[str]:
    return list(record.get("actions_tried") or [])


def record_action(payment_id: str, signature: str, is_rung: bool = False) -> None:
    """Note a recovery action that actually reached the customer.

    Rung 5 is deliberately not a named tool. Which route is worth trying after an
    offer has failed — a different rail, a silent retry timed to payday, a fresh
    offer shape — depends on the customer and the failure, and that judgement is
    the agent's. So the ledger does not prescribe one; it notices when the agent
    takes an approach it has not taken before, and counts that as the rung.

    The signature is what makes two attempts different. Emailing the same link
    twice is one action; emailing a link and then scheduling a retry is two.
    """
    if not signature:
        return
    try:
        from recovery_agent.state_store import StateStore
        store = StateStore()
        rec = store.get_payment(payment_id)
        if rec is None:
            return
        tried = list(rec.get("actions_tried") or [])
        if signature in tried:
            return                      # a repeat is not a different path
        tried.append(signature)
        store.update_payment(payment_id, actions_tried=tried)
        store.flush()

        # Anything genuinely new AFTER the offer has been made is the alternate
        # path. Before that, it is just the ladder being climbed normally.
        rec = store.get_payment(payment_id) or {}
        # An action that IS one of the named rungs cannot also be the
        # alternate path — the offer email is the offer, not a different route.
        if not is_rung and climbed(rec, "offer") and not climbed(rec, "alternate_path"):
            baseline = {a for a in tried[:-1]}
            if baseline:                # something came before it to differ from
                record_rung(payment_id, "alternate_path", signature)
    except Exception:
        pass


def state(record: dict) -> dict:
    """What has been climbed, what is left, and what is simply not possible."""
    done, remaining, impossible = [], [], []
    for key, label in RUNGS:
        if climbed(record, key):
            done.append(key)
            continue
        possible, why = rung_possible(key, record)
        (remaining if possible else impossible).append(
            {"rung": key, "what": label, **({} if possible else {"why_not": why})})
    return {"climbed": done, "remaining": remaining, "unavailable": impossible}


def next_rung(record: dict) -> dict | None:
    rest = state(record)["remaining"]
    return rest[0] if rest else None


def voice_wait_remaining_minutes(record: dict) -> float:
    """How much longer before a call is allowed. 0 once the gap has passed."""
    offer = (record.get("ladder") or {}).get("offer") or {}
    at = offer.get("at")
    if not at:
        return 0.0
    try:
        started = datetime.fromisoformat(at)
    except ValueError:
        return 0.0
    elapsed = (datetime.now(timezone.utc) - started).total_seconds() / 60
    return max(0.0, VOICE_DELAY_MINUTES - elapsed)


def exhausted(record: dict) -> bool:
    """True when nothing is left that could still be tried."""
    return not state(record)["remaining"]


#: Failure kinds where chasing the customer is the WRONG move, so the ladder does
#: not apply and a human should see the case at once. Deliberately narrow: a dead
#: card is not one of these — the recovery for a dead card is another rail.
DO_NOT_PURSUE = ("fraud", "risk", "dispute", "chargeback", "blocked_by_risk",
                 "suspected", "frozen")


def pursuit_barred(record: dict) -> str:
    """Why this case must not be worked at all, or '' if it may be."""
    code = str(record.get("failure_code") or "").lower()
    reason = str(record.get("failure_reason") or "").lower()
    for word in DO_NOT_PURSUE:
        if word in code or word in reason:
            return f"failure looks like {word}; contacting the customer is not appropriate"
    if record.get("opted_out"):
        return "customer has opted out of contact"
    return ""
