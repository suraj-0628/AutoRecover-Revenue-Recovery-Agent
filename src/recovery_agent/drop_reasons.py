"""Ask the customer why, instead of inferring it wrongly.

Two customers abandon a checkout after a bank decline and look IDENTICAL to
this system:

  - one has no balance in the wallet they just tried — a discount is useless
    to them, they need TIME, and a recovery link sent now is a link nobody can
    pay;
  - one found it cheaper elsewhere — a discount is exactly the lever, and
    waiting is a lost sale.

Nothing in a Razorpay error code separates those, and guessing costs money in
both directions: money given away to someone who could not have paid at any
price, and money lost by waiting out someone who was one nudge from buying.

So the checkout asks. One question, on the page the customer is already
looking at, before the retry notification. Their answer is testimony about
their own intent — the strongest evidence available about what to do next —
so it OUTRANKS every inference drawn from the failure code.

"Something else" takes free text and is flagged for a human, because the
honest answer to an answer we did not anticipate is a person reading it, not
a keyword match pretending to understand.
"""
from __future__ import annotations

from typing import Any

#: Each answer: what the customer sees, what it MEANS for recovery, and what
#: the agent is told to do. `kind` feeds the ladder; `offer_ok` is whether a
#: discount is even relevant.
REASONS: list[dict[str, Any]] = [
    {
        "code": "no_balance",
        "label": "I didn't have enough balance",
        "kind": "funds",
        "offer_ok": False,
        "means": "the account or wallet was short at the time",
        "do": ("Money off does not put money in their account. Schedule a "
               "quiet retry for when they are likely to have funds (24-72h, "
               "aim for a payday) and do NOT contact them again before it. "
               "Sending a link now is sending a link nobody can pay."),
    },
    {
        "code": "better_price",
        "label": "I found a better price elsewhere",
        "kind": "dropoff",
        "offer_ok": True,
        # Straight to the discount. The dropoff ladder opens with page_push,
        # so a customer who has SAID "I found it cheaper" was answered with
        # "You left this payment incomplete. Can we help?" and a five minute
        # wait, while they were comparing prices in another tab
        # (pay_p9c6jv4x1). The rung they need is rung two; nothing is learned
        # by climbing rung one first, and the delay is the whole cost.
        "entry_rung": "offer",
        "means": "a price objection, stated outright",
        "do": ("This is the one case a discount genuinely answers. Offer what "
               "policy allows, promptly — they are comparing right now, and "
               "the ceiling does not move because they say a rival is "
               "cheaper."),
    },
    {
        "code": "payment_kept_failing",
        "label": "The payment kept failing",
        "kind": "method",
        "offer_ok": False,
        # They have told us the instrument is the problem. Notifying them
        # about it again says nothing they do not already know; the rail is
        # the answer, and it is rung two.
        "entry_rung": "rail_switch",
        "means": "the instrument, not the price",
        "do": ("Give them a route, not a discount: a link at the FULL amount "
               "on a rail that has worked for this customer before. Check "
               "get_customer_payment_history first."),
    },
    {
        "code": "just_browsing",
        "label": "I changed my mind / just browsing",
        "kind": "dropoff",
        "offer_ok": True,
        "means": "no intent to buy right now",
        "do": ("Low expectation. One offer is worth making, but do not chase "
               "this customer through every channel — they have told you they "
               "were not really buying."),
    },
    {
        "code": "will_pay_later",
        "label": "I'll pay later",
        "kind": "dropoff",
        "offer_ok": False,
        "means": "intent is intact; only the timing is wrong",
        "do": ("Do not discount someone who has said they intend to pay — you "
               "would be paying them to do what they already said they would. "
               "Schedule a quiet reminder and leave them alone until then."),
    },
    {
        "code": "other",
        "label": "Something else",
        "kind": "",                 # decided by a human, not by us
        "offer_ok": False,
        "free_text": True,
        "means": "a reason we did not anticipate",
        "do": ("Their own words are in the case. Do NOT pattern-match them "
               "into one of the buckets above — this has been flagged for a "
               "person to read. Take a cheap, reversible step at most, and "
               "escalate rather than spend money on a guess."),
    },
]

BY_CODE = {r["code"]: r for r in REASONS}


def choices() -> list[dict[str, Any]]:
    """What the checkout renders. Only what the customer needs to see."""
    return [{"code": r["code"], "label": r["label"],
             "free_text": bool(r.get("free_text"))} for r in REASONS]


def get(code: str) -> dict[str, Any] | None:
    return BY_CODE.get(str(code or "").strip())


def briefing_line(reason: dict[str, Any]) -> str:
    """How a stated reason is put to the agent — as testimony, not a guess."""
    spec = get(reason.get("code", "")) or {}
    text = str(reason.get("text") or "").strip()
    said = f'They said: "{text}"' if text else f"They chose: {spec.get('label', '?')}"
    return (f"THE CUSTOMER TOLD YOU WHY. {said}\n"
            f"  This is testimony, not an inference from an error code — it "
            f"outranks anything you would have guessed.\n"
            f"  What it means: {spec.get('means', 'unclear')}.\n"
            f"  What to do: {spec.get('do', 'Proceed carefully.')}")
