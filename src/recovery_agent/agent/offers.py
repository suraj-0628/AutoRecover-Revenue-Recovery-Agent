"""Offer policy — what the agent is allowed to give away, grounded in the KB.

The store's discount policy lives in `data/knowledge_base/merchant_dunning_rules.md`
("Discount Tier Strategy"). This module reads the caps from there and turns them
into a hard bound the agent cannot exceed.

Why the caps are enforced here and not in the prompt
---------------------------------------------------
The agent negotiates with real money. Asked to "stay within policy" it will
usually comply — but "usually" is not a control, and a customer who says *"I
found it cheaper elsewhere"* is precisely the input that talks a model past its
instructions. So the model chooses **whether** to offer; this module decides
**how much**, and the ceiling is not reachable by argument.

Same principle as the voice-call spend cap: anything that costs money is
deterministic, and the LLM operates inside the envelope it returns.

Stage, not elapsed days
-----------------------
The KB tiers discounts by days since failure (0-1d -> 0%, 7d+ -> 20%). A live
recovery ladder runs in minutes, so the *stage* of the ladder is mapped onto
those tiers rather than wall-clock days. The caps are untouched.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

_KB_FILE = (Path(__file__).resolve().parent.parent.parent.parent
            / "data" / "knowledge_base" / "merchant_dunning_rules.md")

# Fallbacks used only if the KB file is missing or unparseable. They match the
# committed policy; they are not a licence to invent more generous terms.
_FALLBACK_MAX_PCT = 20.0
_FALLBACK_MIN_TXN = 500.0
_FALLBACK_VALIDITY_HOURS = 48


class OfferStage:
    """Where we are in the recovery ladder."""
    SILENT_PUSH = "silent_push"     # in-page nudge, no money given away
    EMAIL_OFFER = "email_offer"     # first real incentive
    UI_OFFER = "ui_offer"           # same incentive, surfaced on the page
    VOICE_NEGOTIATE = "voice"       # agent may negotiate up to the policy ceiling


#: Ladder stage -> the KB discount tier that stage is allowed to reach.
_STAGE_CEILING_PCT: dict[str, float] = {
    OfferStage.SILENT_PUSH: 0.0,    # a silent retry must cost nothing
    OfferStage.EMAIL_OFFER: 5.0,
    OfferStage.UI_OFFER: 5.0,       # same offer, second surface — never richer
    OfferStage.VOICE_NEGOTIATE: 20.0,
}


@dataclass(frozen=True)
class OfferPolicy:
    """The caps, as read from the knowledge base."""
    max_discount_pct: float = _FALLBACK_MAX_PCT
    min_transaction_rupees: float = _FALLBACK_MIN_TXN
    validity_hours: int = _FALLBACK_VALIDITY_HOURS
    one_time_only: bool = True
    source: str = "fallback"


@dataclass(frozen=True)
class Offer:
    """A concrete, bounded offer the agent may make."""
    stage: str
    allowed: bool
    discount_pct: float = 0.0
    discount_rupees: float = 0.0
    payable_rupees: float = 0.0
    reason: str = ""
    policy_source: str = ""
    caps: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "allowed": self.allowed,
            "discount_pct": self.discount_pct,
            "discount_rupees": self.discount_rupees,
            "payable_rupees": self.payable_rupees,
            "reason": self.reason,
            "policy_source": self.policy_source,
            "caps": self.caps,
        }


_cached: OfferPolicy | None = None


def load_policy(refresh: bool = False) -> OfferPolicy:
    """Read the discount caps out of the knowledge base."""
    global _cached
    if _cached is not None and not refresh:
        return _cached
    try:
        text = _KB_FILE.read_text(encoding="utf-8")
    except OSError:
        _cached = OfferPolicy()
        return _cached

    def _num(pattern: str, default: float) -> float:
        m = re.search(pattern, text, re.I)
        if not m:
            return default
        return float(m.group(1).replace(",", ""))

    _cached = OfferPolicy(
        max_discount_pct=_num(r"Maximum total discount\*?\*?:?\s*(\d+(?:\.\d+)?)\s*%",
                              _FALLBACK_MAX_PCT),
        min_transaction_rupees=_num(
            r"Minimum transaction for discount\*?\*?:?\s*₹\s*([\d,]+)", _FALLBACK_MIN_TXN),
        validity_hours=int(_num(r"Discount validity\*?\*?:?\s*(\d+)\s*hours",
                                _FALLBACK_VALIDITY_HOURS)),
        one_time_only=bool(re.search(r"One-time only", text, re.I)),
        source=_KB_FILE.name,
    )
    return _cached


def _round_rupees(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def quote(
    amount_rupees: float,
    stage: str,
    requested_pct: float | None = None,
    already_offered_pct: float = 0.0,
) -> Offer:
    """The best offer permitted at this stage, for this amount.

    `requested_pct` is what the agent (or a negotiating customer) asked for. It
    is clamped, never honoured on trust. An offer can only ever go up to the
    stage ceiling and the global cap, whichever is lower.
    """
    policy = load_policy()
    caps = {
        "max_discount_pct": policy.max_discount_pct,
        "min_transaction_rupees": policy.min_transaction_rupees,
        "validity_hours": policy.validity_hours,
        "one_time_only": policy.one_time_only,
        "stage_ceiling_pct": _STAGE_CEILING_PCT.get(stage, 0.0),
    }

    if stage not in _STAGE_CEILING_PCT:
        return Offer(stage=stage, allowed=False,
                     reason=f"unknown ladder stage {stage!r}; no discount authorised",
                     policy_source=policy.source, caps=caps)

    if amount_rupees < policy.min_transaction_rupees:
        return Offer(stage=stage, allowed=False, payable_rupees=_round_rupees(amount_rupees),
                     reason=f"INR {amount_rupees:,.2f} is below the INR "
                            f"{policy.min_transaction_rupees:,.0f} minimum for any discount",
                     policy_source=policy.source, caps=caps)

    ceiling = min(_STAGE_CEILING_PCT[stage], policy.max_discount_pct)
    if ceiling <= 0:
        return Offer(stage=stage, allowed=False, payable_rupees=_round_rupees(amount_rupees),
                     reason="this stage is silent recovery — no incentive is authorised yet",
                     policy_source=policy.source, caps=caps)

    wanted = ceiling if requested_pct is None else float(requested_pct)
    pct = max(0.0, min(wanted, ceiling))

    # One-time-only means the total given away, not the amount per offer.
    if policy.one_time_only and already_offered_pct > 0:
        pct = max(0.0, min(pct, ceiling - already_offered_pct))
        if pct <= 0:
            return Offer(stage=stage, allowed=False,
                         payable_rupees=_round_rupees(amount_rupees),
                         reason=f"{already_offered_pct:.0f}% has already been offered on "
                                f"this transaction; the policy allows no more",
                         policy_source=policy.source, caps=caps)

    discount = _round_rupees(amount_rupees * pct / 100.0)
    clamped = requested_pct is not None and float(requested_pct) > ceiling
    return Offer(
        stage=stage, allowed=True, discount_pct=pct, discount_rupees=discount,
        payable_rupees=_round_rupees(amount_rupees - discount),
        reason=(f"{pct:.0f}% authorised at stage {stage}"
                + (f" (requested {float(requested_pct):.0f}% exceeds the "
                   f"{ceiling:.0f}% ceiling and was reduced)" if clamped else "")),
        policy_source=policy.source, caps=caps,
    )
