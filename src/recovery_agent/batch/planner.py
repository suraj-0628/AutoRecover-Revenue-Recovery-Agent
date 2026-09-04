"""What to do with a band of a batch — decided once, from the merchant's policy.

This planner is deterministic. That is a choice, not a gap.

The batch's own description already states the strategy: `bank_declined` says
*"these need a different rail, at full price"*, `insufficient_funds` says *"a
different day, not a different message"*, `dropoff` says *"the only lever here is
a reason to come back"*. Those sentences are in `classify.py` because they are
the merchant's policy, and `merchant_dunning_rules.md` fills in the rest per
amount band. Asking a model to re-derive a decision that is already written down
adds a network call and a failure mode without adding an insight.

Where the model does belong is the message and the judgement calls — and the
seam for it is `plan_for()` returning a dict that `plan.validate()` then checks.
An LLM planner drops in by producing that same dict; every clamp, every refusal
and every audit hook stays exactly where it is. That is worth more than the call
itself: whatever writes the plan, the plan is validated the same way.

Three rules the defaults follow, each from something the codebase already knows:

**Free actions before paid ones.** A Razorpay test account allows 30 payment
links for its lifetime and cancelling does not return one. `retry_in_hours` costs
nothing and is unlimited, so the batches a retry actually helps — a short account,
a dropped gateway — get a retry, and the scarce links go to the batches where
only a link will do.

**Full price unless the failure is a lost intention.** A declined card is not a
pricing problem; discounting it gives away money to solve something a different
rail solves for free. Only `dropoff` — where nothing broke and the customer
simply left — has a discount as a defensible lever, and even there the run's
`max_discount_paise` still has to allow it.

**Anything past the offer goes to a person.** `ladder.py` deliberately leaves
`alternate_path` undefined: *which* route is worth trying after an offer has
failed is the agent's judgement. A plan that named one would be inventing policy.
"""
from __future__ import annotations

from typing import Any

from recovery_agent.batch.plan import BatchPlan, PlanRejected, validate
from recovery_agent.batch.tiers import Tier, amount_tier, load_tiers

#: Per batch: what to do on the rung a case is actually on, and why.
#:
#: The keys are rungs from `ladder.RUNGS_BY_KIND`, which is per failure kind —
#: a declined instrument climbs `page_push -> rail_switch -> offer`, a short
#: account climbs `silent_retry -> page_push -> offer`. That is the same policy
#: this file encodes, which is the point: the plan does not get to invent an
#: order, it fills in what to *do* on the rung the ladder has already chosen.
#:
#: `alternate_path` is absent everywhere on purpose. `ladder.py` leaves it
#: undefined because which route is worth trying after everything else has
#: failed is the agent's judgement, and a plan that named one would be inventing
#: policy for a case that has already resisted the policy.
_PAGE_PUSH_SKIP = {
    "action": "skip",
    "why": "someone is on the checkout page right now — the live agent has this "
           "case, and a batch must not talk over it",
}

#: A discount is a decision about money, and the run's discount budget is zero
#: unless a human raised it. Reaching this rung means full price has already
#: failed, which is exactly when the question stops being mechanical.
_OFFER_EXCEPTION = {
    "action": "exception",
    "why": "a discount after a full-price attempt has failed is a judgement "
           "call, not a shared decision",
}

_DEFAULTS: dict[str, dict[str, Any]] = {
    "bank_declined": {
        "cause": "the instrument was refused, so the same rail will refuse it again",
        "rails": ["upi", "netbanking"],
        "steps": {
            "page_push": _PAGE_PUSH_SKIP,
            "rail_switch": {"action": "link_and_notify", "full_price": True,
                            "why": "the same amount on another rail — the "
                                   "customer tried to pay, so this is an "
                                   "instrument problem, not a price one"},
            "offer": _OFFER_EXCEPTION,
        },
        "subject": "A different way to complete your payment",
        "body": ("Hi {name}, your card was declined by the bank, so we have sent "
                 "a link that also accepts UPI and net banking. {amount} is due: "
                 "{link}"),
    },
    "insufficient_funds": {
        "cause": "the account was short, which is a timing problem, not a pricing one",
        "steps": {
            "silent_retry": {"action": "retry", "full_price": True,
                             "why": "a different day, not a different message"},
            "page_push": _PAGE_PUSH_SKIP,
            "offer": _OFFER_EXCEPTION,
        },
        "retry_hours": 24,
    },
    "transient": {
        "cause": "the gateway or the network dropped it; nothing was wrong with the payment",
        "steps": {
            "silent_retry": {"action": "retry", "full_price": True,
                             "why": "the cheapest thing that works is trying again"},
            "page_push": _PAGE_PUSH_SKIP,
        },
        "retry_hours": 1,
    },
    "dropoff": {
        "cause": "nothing broke; they walked away, so the lever is a reason to come back",
        "rails": ["upi", "card", "netbanking"],
        "steps": {
            "page_push": _PAGE_PUSH_SKIP,
            "offer": {"action": "link_and_notify", "full_price": True,
                      "why": "a live link and a deadline. Full price until a "
                             "discount budget is authorised — the run's is zero "
                             "by default"},
            "voice_call": {"action": "exception",
                           "why": "a call is a person's judgement about one "
                                  "customer, never a batch decision"},
        },
        "subject": "Your order is still waiting",
        "body": ("Hi {name}, your order is still held for you. {amount} is due "
                 "and this link expires shortly: {link}"),
    },
}


def runnable_batches() -> list[str]:
    """Batches a shared plan may be written for.

    `unclassified` is excluded deliberately: a plan is a shared decision, and
    "we do not know why this failed" is precisely the case with no shared cause
    to stand on. Those go to the agent one at a time.
    """
    from recovery_agent.agent.classify import BATCHES
    return [b["key"] for b in BATCHES if b.get("runnable")]


def plan_for(batch_key: str, tier: str | Tier, *,
             source: str = "default") -> BatchPlan:
    """The plan for one (batch x band). Raises `PlanRejected` if there is none."""
    spec = _DEFAULTS.get(batch_key)
    if spec is None:
        raise PlanRejected(f"no plan is defined for the batch {batch_key!r}")

    band = tier if isinstance(tier, Tier) else _tier_by_key(tier)
    raw = dict(spec)
    raw["steps"] = {k: dict(v) for k, v in spec["steps"].items()}

    # The band's own row from the merchant's policy, carried onto the plan so
    # the audit trail says which rule authorised the action rather than leaving
    # it to be reconstructed later.
    raw["cause"] = f"{spec['cause']} | band: {band.title}"
    if band.retry_window and any(s["action"] == "retry"
                                 for s in raw["steps"].values()):
        # The batch knows the cause and the band knows the customer's value, so
        # each constrains the other: whichever says "sooner" wins. A gateway
        # blip does not need the medium band's eighteen-hour pacing, and a short
        # account does not need the blip's one hour.
        raw["retry_hours"] = min(float(raw.get("retry_hours", 24)),
                                 _first_gap(band.retry_window, 24.0))

    plan = validate(raw, batch_key=batch_key, tier=band.key, source=source)
    return BatchPlan(**{**plan.as_dict(),
                        "steps": plan.steps,
                        "rails": plan.rails,
                        "provenance": {**plan.provenance,
                                       "kb_tier": band.key,
                                       "kb_incentive": band.incentive,
                                       "kb_escalation": band.escalation}})


def plans_for(batch_key: str, records: list[dict], *,
              source: str = "default") -> tuple[dict[str, BatchPlan], list[dict]]:
    """One plan per band actually present. Returns (plans, rejections).

    Bands with no cases are not planned for: a plan is a decision about specific
    customers, and there is no decision to make about none of them.
    """
    plans: dict[str, BatchPlan] = {}
    rejected: list[dict] = []
    present = {amount_tier(r.get("amount")).key for r in records}
    for band in load_tiers():
        if band.key not in present:
            continue
        try:
            plans[band.key] = plan_for(batch_key, band, source=source)
        except PlanRejected as exc:
            rejected.append({"tier": band.key, "why": str(exc)})
    return plans, rejected


def _tier_by_key(key: str) -> Tier:
    for band in load_tiers():
        if band.key == key:
            return band
    raise PlanRejected(f"unknown amount band {key!r}")


def _first_gap(window: str, fallback: float) -> float:
    """How long before the *first* retry, from a policy line like
    `72 hours (4 attempts)`.

    The policy states a span and a number of attempts, not a gap. Reading the
    span as the gap would schedule a medium-band retry three days out and a
    high-value one a fortnight out — the opposite of what the policy is asking
    for, which is more attempts for a larger order, not fewer and later. Four
    attempts across seventy-two hours is one every eighteen.
    """
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*(minute|hour|day)", window, re.I)
    if not m:
        return float(fallback)
    span = float(m.group(1)) * {"minute": 1 / 60, "hour": 1.0,
                                "day": 24.0}[m.group(2).lower()]
    attempts = re.search(r"(\d+)\s*attempts?", window, re.I)
    return span / max(1, int(attempts.group(1)) if attempts else 1)
