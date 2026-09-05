"""What a script would have done at the same decision points.

The project could report what it recovered but never whether the AGENT was the
reason. That makes every number a description rather than a claim: money
arrived after the agent acted, which is not the same as money arriving because
it did.

A live holdout arm is the gold standard and was judged not worth the effort.
This is the cheap honest alternative: every decision the agent has actually
made is recorded with the perception facts it was made under, so a scripted
policy can be asked the same question at the same moment and the two answers
scored side by side. No traffic is split and no customer is experimented on —
the comparison is entirely offline, over decisions that already happened.

It is a WEAKER claim than a holdout, deliberately: it shows the agent choosing
better than a script at the points where a choice existed. It cannot show that
either policy caused a recovery. Saying so is the point.
"""
from __future__ import annotations

from typing import Any, Callable

Decision = list[dict[str, Any]]


def _link(facts: dict, amount: float) -> Decision:
    return [{"name": "generate_recovery_payment_link",
             "args": {"payment_id": facts.get("payment_id", ""),
                      "amount": round(amount, 2)}}]


def fixed_ladder(facts: dict) -> Decision:
    """The naive ladder: nudge, then discount, then hand it to a person.

    One sequence for every failure, which is what this system looked like
    before the ladder was split by failure kind. It is the honest strawman —
    a competent engineer with no agent would write approximately this.
    """
    climbed = set(facts.get("climbed") or [])
    owed = float(facts.get("owed") or 0)
    if "page_push" not in climbed and "page_push" not in str(facts.get("unavailable")):
        return [{"name": "send_page_push", "args": {"payment_id": facts.get("payment_id", "")}}]
    if "offer" not in climbed:
        return _link(facts, owed * 0.95)          # always 5% at rung 2
    return [{"name": "escalate_to_human",
             "args": {"payment_id": facts.get("payment_id", "")}}]


def always_discount(facts: dict) -> Decision:
    """Lead with money. The policy a growth team reaches for by default."""
    return _link(facts, float(facts.get("owed") or 0) * 0.95)


def never_discount(facts: dict) -> Decision:
    """Never give anything away. Safe on margin, blind to a price objection."""
    return _link(facts, float(facts.get("owed") or 0))


def do_nothing(facts: dict) -> Decision:
    """The floor. Conformant by construction — it never acts, so it never
    breaks a rule. Included because a conformance score that a do-nothing
    policy can win is a score measuring the wrong thing, and the comparison
    should say so out loud."""
    return []


BASELINES: dict[str, Callable[[dict], Decision]] = {
    "fixed_ladder": fixed_ladder,
    "always_discount": always_discount,
    "never_discount": never_discount,
    "do_nothing": do_nothing,
}


def giveaway_rupees(facts: dict, chosen: Decision) -> float:
    """Margin this decision would hand over. The number conformance misses.

    A policy can be perfectly conformant and still expensive, so the
    comparison reports money alongside rule-following rather than collapsing
    both into one score.
    """
    owed = float(facts.get("owed") or 0)
    total = 0.0
    for call in chosen or []:
        args = call.get("args") or {}
        if call.get("name") == "generate_recovery_payment_link":
            amount = float(args.get("amount") or 0)
            if owed > 0 and 0 < amount < owed:
                total += owed - amount
        elif call.get("name") == "show_page_offer":
            payable = float(args.get("payable_amount") or 0)
            if owed > 0 and 0 < payable < owed:
                total += owed - payable
    return round(total, 2)


def contacts_spent(chosen: Decision) -> int:
    """Customer contacts a decision would consume — a finite budget per the
    merchant's own fatigue rules, and the thing an eager policy overspends."""
    reaching = {"send_recovery_notification", "initiate_voice_call",
                "send_page_push", "show_page_offer"}
    return sum(1 for c in (chosen or []) if c.get("name") in reaching)
