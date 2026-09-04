"""What a whole band of a batch will be offered, decided once.

A plan is the decision; the executor is the application. Two rules make that
split safe, and both exist because a plan is applied many times:

**A plan never carries a rupee figure.** It carries an `offer_stage`. The
payable amount is re-derived per case by `offers.quote(record["amount"], stage)`.
A plan holding a number would apply one case's discount to another case's
amount — the exact class of error `offers.py` exists to prevent, multiplied by
the size of the batch.

**A plan never names a rung a case is not on.** The executor reads
`ladder.next_rung(record)` fresh, per case, and looks it up in `steps`. A miss is
an exception routed to the agent, never an improvisation. A case that has moved
on since planning is precisely the case a shared decision no longer fits.

Validation runs *after* the planner and never instead of it: whatever produced
the plan — a hand-written default or a model — the same clamps apply.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from recovery_agent.agent.offers import OfferStage, load_policy

#: What an executor is allowed to do for one rung. Anything outside this set has
#: to go to the agent — a plan cannot invent a new kind of action.
ACTIONS = ("link_and_notify", "notify_only", "retry", "skip", "exception")

#: Rails a payment link may be restricted to. A plan naming anything else is
#: rejected rather than passed to Razorpay to find out.
RAILS = ("upi", "card", "netbanking", "wallet", "emi", "paylater")

_STAGES = (OfferStage.SILENT_PUSH, OfferStage.EMAIL_OFFER,
           OfferStage.UI_OFFER, OfferStage.VOICE_NEGOTIATE)

#: A retry sooner than this is pestering; later than this the cart is cold.
MIN_RETRY_HOURS, MAX_RETRY_HOURS = 0.5, 168.0

MAX_SUBJECT, MAX_BODY = 120, 1200


class PlanRejected(ValueError):
    """A plan that failed validation. The band becomes exceptions.

    Deliberately not a "best effort" repair: a half-understood plan applied to
    twenty customers is worse than twenty cases going to the agent one at a time.
    """


@dataclass(frozen=True)
class Step:
    """What to do for one ladder rung."""
    action: str
    why: str = ""
    full_price: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "why": self.why,
                "full_price": self.full_price}


@dataclass(frozen=True)
class BatchBudget:
    """What one run may spend before it stops and says so.

    `BATCH_RUN_LIMIT` was a per-click cap, so clicking twice spent twice. A
    budget lives on the run, and every case must also pass the executor's
    `already_in_a_run` check — so a second click creates a run that acts on
    nothing rather than doubling the spend.
    """
    max_cases: int = 25
    #: The one that actually binds. A Razorpay test account allows 30 payment
    #: links for its lifetime and cancelling does not give one back.
    max_links: int = 5
    max_emails: int = 25
    max_llm_calls: int = 8
    #: Money given away without a human. Zero by default: a batch is the wrong
    #: place to discover a discount policy.
    max_discount_paise: int = 0
    max_wallclock_seconds: int = 900
    abort_after_consecutive_failures: int = 3

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_cases": self.max_cases, "max_links": self.max_links,
            "max_emails": self.max_emails, "max_llm_calls": self.max_llm_calls,
            "max_discount_paise": self.max_discount_paise,
            "max_wallclock_seconds": self.max_wallclock_seconds,
            "abort_after_consecutive_failures":
                self.abort_after_consecutive_failures,
        }

    @classmethod
    def from_request(cls, data: dict | None) -> "BatchBudget":
        """A caller may tighten the budget but never loosen it past the ceiling."""
        data = data or {}
        ceiling = cls()
        def pick(name: str) -> int:
            try:
                asked = int(data.get(name, getattr(ceiling, name)))
            except (TypeError, ValueError):
                return getattr(ceiling, name)
            return max(0, min(asked, getattr(ceiling, name)))
        return cls(**{k: pick(k) for k in ceiling.as_dict()})


@dataclass(frozen=True)
class BatchPlan:
    """One decision, for one (batch key x amount band)."""
    batch_key: str
    tier: str
    cause: str
    steps: dict[str, Step]
    rails: tuple[str, ...] = ()
    offer_stage: str | None = None
    retry_hours: float | None = None
    subject: str = ""
    body: str = ""
    expire_in_minutes: int = 16
    provenance: dict[str, Any] = field(default_factory=dict)

    def step_for(self, rung: str | None) -> Step | None:
        return self.steps.get(rung or "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_key": self.batch_key, "tier": self.tier, "cause": self.cause,
            "steps": {k: v.as_dict() for k, v in self.steps.items()},
            "rails": list(self.rails), "offer_stage": self.offer_stage,
            "retry_hours": self.retry_hours, "subject": self.subject,
            "body": self.body, "expire_in_minutes": self.expire_in_minutes,
            "provenance": self.provenance,
        }

    def digest(self) -> str:
        """A stable id for this exact plan, for the audit trail.

        Identifies the *decision*, not the moment it was serialised, so the
        creation timestamp is left out — two identical plans made a second apart
        are the same decision and should be one id in the log. The rest of
        provenance stays in: a plan written by the model and an identical one
        written by the stub were made on different authority.
        """
        body = self.as_dict()
        body["provenance"] = {k: v for k, v in body["provenance"].items()
                              if k != "created_at"}
        blob = json.dumps(body, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def validate(raw: dict[str, Any], *, batch_key: str, tier: str,
             source: str = "stub") -> BatchPlan:
    """Turn an untrusted dict into a plan, or refuse it.

    Every field is checked and clamped. There is no partial-parse path: a plan
    is either fully understood or the whole band goes to the agent.
    """
    if not isinstance(raw, dict):
        raise PlanRejected("plan is not an object")

    steps_raw = raw.get("steps") or {}
    if not isinstance(steps_raw, dict) or not steps_raw:
        raise PlanRejected("plan has no steps")

    steps: dict[str, Step] = {}
    for rung, spec in steps_raw.items():
        if not isinstance(spec, dict):
            raise PlanRejected(f"step for {rung!r} is not an object")
        action = str(spec.get("action") or "").strip()
        if action not in ACTIONS:
            raise PlanRejected(f"unknown action {action!r} for rung {rung!r}")
        steps[str(rung)] = Step(action=action,
                                why=str(spec.get("why") or "")[:300],
                                full_price=bool(spec.get("full_price", True)))

    rails = tuple(r for r in (str(x).strip().lower()
                              for x in (raw.get("rails") or [])) if r)
    unknown = [r for r in rails if r not in RAILS]
    if unknown:
        raise PlanRejected(f"unknown rails {unknown}")

    stage = raw.get("offer_stage")
    if stage is not None:
        stage = str(stage).strip()
        if stage not in _STAGES:
            raise PlanRejected(f"unknown offer stage {stage!r}")

    # A plan that gives money away where the policy allows none is not clamped
    # quietly — it is refused, because the discrepancy means the planner was
    # reasoning from something other than the policy.
    if stage:
        ceiling = load_policy().max_discount_pct
        if ceiling <= 0:
            raise PlanRejected("policy permits no discount at all")

    retry_hours = raw.get("retry_hours")
    if retry_hours is not None:
        try:
            retry_hours = float(retry_hours)
        except (TypeError, ValueError):
            raise PlanRejected(f"retry_hours {retry_hours!r} is not a number")
        retry_hours = max(MIN_RETRY_HOURS, min(retry_hours, MAX_RETRY_HOURS))

    needs_retry = any(s.action == "retry" for s in steps.values())
    if needs_retry and retry_hours is None:
        raise PlanRejected("a retry step needs retry_hours")

    needs_message = any(s.action in ("link_and_notify", "notify_only")
                        for s in steps.values())
    subject = str(raw.get("subject") or "")[:MAX_SUBJECT]
    body = str(raw.get("body") or "")[:MAX_BODY]
    if needs_message and not body.strip():
        raise PlanRejected("a notify step needs a message body")

    try:
        expire = int(raw.get("expire_in_minutes", 16))
    except (TypeError, ValueError):
        expire = 16
    # Razorpay enforces a 15-minute floor on expire_by; asking for less is a
    # request the gateway will refuse.
    expire = max(16, min(expire, 60 * 48))

    return BatchPlan(
        batch_key=batch_key, tier=tier,
        cause=str(raw.get("cause") or "")[:400],
        steps=steps, rails=rails, offer_stage=stage, retry_hours=retry_hours,
        subject=subject, body=body, expire_in_minutes=expire,
        provenance={
            "source": source,
            "policy_source": "merchant_dunning_rules.md",
            "model": os.getenv("LLM_MODEL", "") if source == "llm" else "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
