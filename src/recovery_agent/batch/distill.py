"""One worked case becomes the group's plan — the champion mechanism.

This is where the agent's intelligence enters the batch path. A batch shares a
cause, so before a bin is worked, ONE representative case — the champion — is
given to the live agent as a full session: perception, ladder, memory, the
negotiation loop, everything. The agent does not know it is deciding for a
group; it simply works its case well. This module then reads what it *did* off
the durable record and turns those decisions into a `BatchPlan` for the rest of
the bin: the same inference, applied N times, at zero further LLM cost.

What is distilled and what deliberately is not:

**Distilled** — the shape of the decision: which rung the agent acted on, what
kind of action it took there (a link and a message, a scheduled retry), whether
it held full price, and how long it chose to wait. These generalise: they are
answers to "what does this *cause* need", which is the batch's defining shared
property.

**Not distilled** — anything personal to the champion. The message body comes
from the merchant's policy templates, never from the champion's prose, because
prose is where one customer's name, amount or history leaks into another
customer's inbox. And a *discount* is never generalised: if the champion's link
was below the owed amount, that rung distils to `exception`, because a discount
is a judgement about one customer's circumstances — the agent that granted it
saw signals the rest of the bin has not shown.

The distiller is deterministic. The intelligence already happened, in the
champion session; copying it must not add a second model call that could
mis-transcribe what the first one decided.
"""
from __future__ import annotations

import re
from typing import Any

from recovery_agent.agent import ladder
from recovery_agent.batch import planner
from recovery_agent.batch.plan import BatchPlan, PlanRejected, validate
from recovery_agent.batch.tiers import Tier


#: Used when the champion messaged but the bin's policy default has no
#: template to lend (a retry-only plan, say). Deliberately plain and generic:
#: the safe floor, never the champion's own prose.
_FALLBACK_SUBJECT = "Complete your pending payment"
_FALLBACK_BODY = ("Hi {name}, your payment of {amount} is still pending. "
                  "You can complete it securely here: {link}")


class DistillationFailed(Exception):
    """The champion's session produced nothing a group can reuse.

    Not an error in the champion — an escalation, a pursuit bar, or a session
    that only gathered information are all legitimate outcomes. They are just
    not decisions that generalise, so the bin falls back to the policy default.
    """


def snapshot(record: dict) -> dict[str, Any]:
    """What to remember about a case before its champion session runs.

    The distiller works on the *difference* — only what the session added is
    the champion's decision. Rungs climbed last week are history, not advice.
    """
    return {
        "rungs": set((record.get("ladder") or {}).keys()),
        "links": len(record.get("recovery_links") or []),
        "contacts": len(record.get("contacts") or []),
    }


def distill(record: dict, before: dict, *, batch_key: str, tier: str | Tier,
            base: BatchPlan | None = None) -> BatchPlan:
    """Turn one champion's recorded actions into the bin's plan.

    Raises `DistillationFailed` when the session left nothing reusable; the
    caller falls back to the policy default, which is always safe.
    """
    owed = float(record.get("amount") or 0)
    lad = record.get("ladder") or {}

    new_rungs = [(k, v) for k, v in lad.items()
                 if k not in before.get("rungs", set())]
    # In the order the agent climbed them, which is the order it decided them.
    new_rungs.sort(key=lambda kv: str((kv[1] or {}).get("at") or ""))

    if not new_rungs:
        raise DistillationFailed(
            "the champion session climbed no rung — nothing to generalise")
    if record.get("status") == "escalated":
        raise DistillationFailed(
            "the champion was escalated — a case the agent hands to a human "
            "is the opposite of a case whose treatment scales")
    barred = ladder.pursuit_barred(record)
    if barred:
        raise DistillationFailed(f"the champion is not pursuable: {barred}")

    new_links = (record.get("recovery_links") or [])[before.get("links", 0):]
    made_contact = len(record.get("contacts") or []) > before.get("contacts", 0)

    steps: dict[str, dict[str, Any]] = {}
    retry_hours: float | None = None
    observed: list[str] = []

    for rung, entry in new_rungs:
        detail = str((entry or {}).get("detail") or "")
        observed.append(rung)

        if rung == "silent_retry":
            steps[rung] = {"action": "retry", "full_price": True,
                           "why": f"the champion's agent chose a quiet retry "
                                  f"({detail or 'no detail'})"}
            hours = re.search(r"in\s+(\d+(?:\.\d+)?)h", detail)
            retry_hours = float(hours.group(1)) if hours else 24.0
            continue

        if rung == "page_push":
            # A batch works cases whose customer left long ago; the live rung
            # does not translate. The default plan's skip step covers it.
            continue

        if new_links:
            link_amount = float(new_links[-1].get("amount") or 0)
            if owed and link_amount < owed - 0.01:
                # The one decision that must NOT scale. The agent granted this
                # customer a discount for reasons visible in this session only.
                steps[rung] = {
                    "action": "exception",
                    "why": "the champion needed a discount — a per-customer "
                           "judgement, so each case earns its own session",
                }
            else:
                steps[rung] = {
                    "action": "link_and_notify", "full_price": True,
                    "why": f"the champion's agent sent a full-price link at "
                           f"{rung} ({detail or 'no detail'})",
                }
            continue

        if made_contact:
            steps[rung] = {"action": "notify_only", "full_price": True,
                           "why": f"the champion's agent messaged without a "
                                  f"link at {rung}"}
            continue

        # A rung climbed with neither a link nor a contact is bookkeeping the
        # distiller does not understand; route the rung to the agent rather
        # than guess.
        steps[rung] = {"action": "exception",
                       "why": f"the champion climbed {rung} in a way the "
                              f"distiller cannot classify"}

    if not steps:
        raise DistillationFailed("every climbed rung was live-only")

    # The champion answers "what does this cause need"; the policy default
    # answers everything else — rungs the champion never reached, the message
    # template (policy prose, never the champion's), the allowed rails (not
    # retained on the record). Champion steps win where both speak.
    base = base or planner.plan_for(batch_key, tier)
    merged = {k: v.as_dict() for k, v in base.steps.items()}
    merged.update(steps)

    raw: dict[str, Any] = {
        "cause": (f"{base.cause} | champion "
                  f"{record.get('payment_id', '?')}: {' -> '.join(observed)}"),
        "steps": merged,
        "rails": list(base.rails),
        "subject": base.subject or _FALLBACK_SUBJECT,
        "body": base.body or _FALLBACK_BODY,
        "expire_in_minutes": base.expire_in_minutes,
    }
    if retry_hours is not None:
        raw["retry_hours"] = retry_hours
    elif base.retry_hours is not None:
        raw["retry_hours"] = base.retry_hours
    if any(s.get("action") == "retry" for s in merged.values()) \
            and raw.get("retry_hours") is None:
        raw["retry_hours"] = 24.0

    tier_key = tier.key if isinstance(tier, Tier) else str(tier)
    try:
        plan = validate(raw, batch_key=batch_key, tier=tier_key,
                        source="champion")
    except PlanRejected as exc:
        raise DistillationFailed(f"the distilled plan failed validation: {exc}")

    return BatchPlan(**{
        **plan.as_dict(), "steps": plan.steps, "rails": plan.rails,
        "provenance": {**plan.provenance,
                       "champion": record.get("payment_id", ""),
                       "observed_rungs": observed},
    })
