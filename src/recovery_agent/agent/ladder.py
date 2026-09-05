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

# Rung order is the recovery policy. Each entry is (key, human label).
#
# THERE IS NO SINGLE LADDER. What to try, and in what order, depends on what
# actually broke — one sequence for every failure is the "generic agent"
# problem in its purest form. The 2026-09-03 case matrix showed it costing
# real money: an insufficient-funds case (C1) was made to send an in-page
# push and an emailed offer before it was allowed to schedule the retry that
# was the only thing that could work, because `page_push -> offer` was rung 1
# and 2 for everyone. A bank decline was pushed toward a discount when the
# customer's problem was an instrument, not a price.
#
# The rungs stay coarse and the agent still decides what to say, to whom, and
# when. It just climbs the ladder that belongs to THIS failure.

#: They chose not to complete. Nothing is broken, so price is the only lever —
#: this is the one family where a discount is the honest next move.
_DROPOFF: list[tuple[str, str]] = [
    ("page_push", "silent in-page notification"),
    ("offer", "discounted offer by email, shown on the page at the same time"),
    ("voice_call", "voice call to ask why and negotiate within policy"),
    ("post_call_email", "email with the agreed pay link after the call"),
    ("alternate_path", "a different route than the ones already tried"),
]

#: The bank refused the instrument. The customer TRIED to pay, so the fix is a
#: different rail at the same price. A discount only after full price fails too.
_METHOD: list[tuple[str, str]] = [
    ("page_push", "silent in-page notification"),
    ("rail_switch", "the SAME amount on a method that has worked for this "
                    "customer before"),
    ("offer", "discounted offer, once a full-price rail has also failed"),
    ("voice_call", "voice call to ask why and negotiate within policy"),
    ("alternate_path", "a different route than the ones already tried"),
]

#: The account was short. That is a timing problem: they need a different DAY,
#: not a different message — so the quiet retry leads and contact comes after.
_FUNDS: list[tuple[str, str]] = [
    ("silent_retry", "a quiet retry timed for when they are likely to be paid"),
    ("page_push", "silent in-page notification"),
    ("offer", "discounted offer, once a retry has also failed"),
    ("alternate_path", "a different route than the ones already tried"),
]

#: Our plumbing failed, not the customer. Retry it. There is deliberately NO
#: offer rung here: discounting a gateway timeout pays the customer to forgive
#: our own outage.
_TRANSIENT: list[tuple[str, str]] = [
    ("silent_retry", "a quiet retry — the gateway dropped it, not the customer"),
    ("page_push", "silent in-page notification"),
    ("alternate_path", "a different route than the ones already tried"),
]

#: Unclassified: fail closed. Full price before any discount, same as method.
_UNKNOWN: list[tuple[str, str]] = [
    ("page_push", "silent in-page notification"),
    ("rail_switch", "the SAME amount on another payment method"),
    ("offer", "discounted offer, once a full-price attempt has also failed"),
    ("alternate_path", "a different route than the ones already tried"),
]

#: They hit a real failure, saw the alternatives, and stopped anyway.
#:
#: This is neither a plain bank decline nor a plain drop-off, and treating it
#: as either gets it wrong. A pure drop-off never tried: nothing is broken, so
#: price is the only lever. A plain method failure IS still trying: the rail is
#: the problem, and a discount does not fix a refused instrument. But someone
#: who was declined, was offered other rails by Razorpay's own screen, and
#: cancelled has demonstrated BOTH — the instrument failed them and their
#: appetite is gone. So the route and the reason are both worth offering, with
#: the reason first, because the alternatives have already been in front of
#: them once and did not move them.
_HESITANT: list[tuple[str, str]] = [
    ("page_push", "silent in-page notification"),
    ("offer", "the discount — their reluctance is already demonstrated, so "
              "price is a fair lever here"),
    ("rail_switch", "a working rail at the discounted price, if the offer "
                    "alone does not move them"),
    ("voice_call", "voice call to ask why and negotiate within policy"),
    ("alternate_path", "a different route than the ones already tried"),
]

RUNGS_BY_KIND: dict[str, list[tuple[str, str]]] = {
    "dropoff": _DROPOFF,
    "method": _METHOD,
    "funds": _FUNDS,
    "transient": _TRANSIENT,
    "risk": [],                 # never pursued — see pursuit_barred()
    "unknown": _UNKNOWN,
}

#: The drop-off ladder remains the default shape for anything that asks for
#: "the rungs" without a case in hand.
RUNGS: list[tuple[str, str]] = _DROPOFF

#: Every rung name that exists in any ladder, for validation.
ALL_RUNGS: dict[str, str] = {
    key: label for seq in RUNGS_BY_KIND.values() for key, label in seq
}


def kind_of(record: dict) -> str:
    """What KIND of failure this is — the thing that picks the ladder."""
    try:
        from recovery_agent.agent.classify import failure_kind
        return failure_kind(record) or "unknown"
    except Exception:
        return "unknown"


def abandoned_after_failure(record: dict) -> bool:
    """Did the customer hit a real failure and then give up on their own?

    Set at ingress when a cancellation arrives on a case that already carries
    a gateway failure. It does not change the DIAGNOSIS — the bank really did
    decline — it records that the customer has since seen the alternatives and
    walked, which is the evidence a method case otherwise earns by trying full
    price first.
    """
    return bool(record.get("abandoned_after_failure"))


def _entry_rung(record: dict) -> str:
    """The rung the customer's own answer starts us on, if it names one."""
    code = (record.get("drop_reason") or {}).get("code") or ""
    if not code:
        return ""
    try:
        from recovery_agent.drop_reasons import get as _reason
        return (_reason(code) or {}).get("entry_rung") or ""
    except Exception:
        return ""


def _offer_vetoed_by_testimony(record: dict) -> str:
    """Why the customer's own answer rules a discount out, or '' if it doesn't."""
    code = (record.get("drop_reason") or {}).get("code") or ""
    if not code:
        return ""
    try:
        from recovery_agent.drop_reasons import get as _reason
        spec = _reason(code) or {}
    except Exception:
        return ""
    if spec and not spec.get("offer_ok", True):
        return (f'the customer said "{spec.get("label", code)}" — '
                f"{spec.get('means', 'not a price objection')}, which no "
                f"discount answers")
    return ""


def rungs_for(record: dict) -> list[tuple[str, str]]:
    """The ladder that belongs to this failure, entered where the customer put us.

    The stated reason used to pick the ladder and then say nothing about where
    on it to start, so testimony could only ever VETO a discount (`offer_ok`)
    and never reach for one. A customer who answered "I found a better price
    elsewhere" still got rung one of the drop-off ladder — a page push reading
    "You left this payment incomplete. Can we help?" and a five-minute wait —
    while the briefing above it said, correctly, that a discount is the one
    thing that answers a price objection and to offer it promptly
    (pay_p9c6jv4x1). The advice changed; the plan did not, and the plan is
    what the agent follows.

    Rungs BEFORE the named one are dropped rather than marked climbed: they
    were not attempted, and recording attempts that never happened would be a
    lie the rest of the ladder reasons from.
    """
    kind = kind_of(record)
    # Someone who failed AND then walked is on neither the "still trying"
    # ladder nor the "nothing broke" one — see _HESITANT.
    if abandoned_after_failure(record) and kind in ("method", "unknown"):
        ladder = _HESITANT
    else:
        ladder = RUNGS_BY_KIND.get(kind, _UNKNOWN)

    entry = _entry_rung(record)
    if entry:
        for i, (key, _label) in enumerate(ladder):
            if key == entry:
                # Never climb BACK to a rung already used; only skip forward.
                return ladder[i:] if i else ladder
    return ladder


def has_rung(record: dict, key: str) -> bool:
    """Is this rung part of THIS case's ladder at all?"""
    return any(k == key for k, _ in rungs_for(record))

#: Minimum gap between the offer and the call. The customer is given a real
#: chance to use the offer before the phone rings.
VOICE_DELAY_MINUTES = int(os.getenv("LADDER_VOICE_DELAY_MINUTES", "15"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def voice_available() -> bool:
    # Operator setting first, env second. Reading only the env made the
    # dashboard's voice switch decorative: it wrote a value nothing consulted.
    try:
        from recovery_agent import guardrail_config
        return bool(guardrail_config.get("voice_enabled"))
    except Exception:
        return os.getenv("VOICE_CALLS_ENABLED", "").strip().lower() in (
            "1", "true", "yes", "on")


def rung_possible(key: str, record: dict) -> tuple[bool, str]:
    """Can this rung ever be climbed for this case? Returns (possible, why not)."""
    customer = record.get("customer") or {}
    email = customer.get("email") or record.get("customer_email") or ""
    phone = customer.get("contact") or record.get("customer_phone") or ""

    if key == "page_push":
        # The silent rung only exists while the customer is on the page. Claiming
        # it when nobody is listening advances the ladder past a contact that
        # never happened, and carries that into the escalation ticket.
        from recovery_agent import presence
        if not presence.is_live(str(record.get("payment_id") or "")):
            return False, "no checkout page is open for this customer"

    if key == "silent_retry":
        # Costs nothing and reaches nobody, so it is always available — that is
        # the entire point of putting it first for funds and transient cases.
        return True, ""

    if key == "rail_switch":
        if record.get("links_unavailable"):
            return False, ("this Razorpay test account has spent its 30 payment "
                           "links, so no alternate rail can be offered")
        if not (email or phone):
            return False, "no email or phone on file for this customer"
        return True, ""

    if key in ("offer", "post_call_email", "alternate_path") and not (email or phone):
        return False, "no email or phone on file for this customer"
    if key == "offer" and record.get("links_unavailable"):
        # The offer rung needs a payment link, and this account cannot create
        # one. Treating it as merely un-climbed would stall every case here
        # short of escalation forever.
        return False, ("this Razorpay test account has spent its 30 payment "
                       "links, so no offer can be made")
    if key == "offer":
        # Testimony that vetoes a discount (offer_ok: False) is not advice
        # about this rung — get_recovery_offer refuses it outright, so it can
        # never be climbed. Same rule as the spent link quota above: an
        # unclimbable rung left in "remaining" holds exhausted() false
        # forever, with escalation and closure both refused behind it.
        veto = _offer_vetoed_by_testimony(record)
        if veto:
            return False, veto
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
    if key not in ALL_RUNGS:
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
        # A rung is a contact the merchant made. An escalation ticket later
        # claims "already tried: page_push, offer" on the strength of this
        # dict, so the same fact goes into the append-only log where it cannot
        # be quietly rewritten.
        from recovery_agent import audit
        audit.record(audit.LADDER_RUNG, payment_id=payment_id,
                     batch_run_id=rec.get("batch_run_id") or "",
                     action=key, reason=detail[:300],
                     climbed=sorted(ladder))
    except Exception:
        pass                            # never let bookkeeping break a recovery


def actions_tried(record: dict) -> list[str]:
    return list(record.get("actions_tried") or [])


def _offer_resolved(record: dict) -> bool:
    """Nothing about the offer stage is still owed before rung 5 can count."""
    if climbed(record, "offer"):
        return True
    if not has_rung(record, "offer"):
        return True
    possible, _why = rung_possible("offer", record)
    return not possible


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

        # Anything genuinely new AFTER the offer stage is settled is the
        # alternate path. Before that, it is just the ladder being climbed
        # normally. "Settled" is not only "climbed": a ladder with no offer
        # rung (transient), or one whose offer can never be climbed (testimony
        # veto, spent link quota), must not hold rung 5 hostage to a stage
        # that cannot happen — live (no_balance cases, 2026-09-04), that left
        # exhausted() false forever, so neither escalation nor closure could
        # ever be permitted.
        rec = store.get_payment(payment_id) or {}
        # An action that IS one of the named rungs cannot also be the
        # alternate path — the offer email is the offer, not a different route.
        if not is_rung and _offer_resolved(rec) and not climbed(rec, "alternate_path"):
            baseline = {a for a in tried[:-1]}
            if baseline:                # something came before it to differ from
                record_rung(payment_id, "alternate_path", signature)
    except Exception:
        pass


def state(record: dict) -> dict:
    """What has been climbed, what is left, and what is simply not possible.

    Walks the ladder for THIS failure kind, not one global sequence.
    """
    done, remaining, impossible = [], [], []
    for key, label in rungs_for(record):
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


def retry_pending(record: dict) -> bool:
    """Is a scheduled retry still in the future and un-run?"""
    job = record.get("scheduled_job") or {}
    if not job:
        return False
    if str(job.get("status") or "").lower() in ("completed", "failed", "cancelled"):
        return False
    target = job.get("target_timestamp") or job.get("target_time") or ""
    try:
        when = datetime.fromisoformat(str(target))
    except (TypeError, ValueError):
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when > datetime.now(timezone.utc)


def exhausted(record: dict) -> bool:
    """True when nothing is left that could still be tried.

    A case with a retry still on the clock is NOT exhausted — it is waiting.
    Live (C1, 2026-09-03): an insufficient-funds case was escalated to a human
    while retries were scheduled for the next day AND three days out, and the
    agent's closing note claimed both had "failed" when neither had run. A
    pending retry is the most likely thing to recover a funds case; handing it
    to a person before it fires spends a human on a case that was working.
    """
    if retry_pending(record):
        return False
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
