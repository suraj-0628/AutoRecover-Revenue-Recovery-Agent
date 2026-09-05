"""What a human learned about a case, in words the system can act on.

A recovery attempt can send perfectly and still not land, and the reason often
arrives where no webhook can see it: a reply, a phone call, a support ticket.
This is the operator-side twin of `drop_reasons` — there the CUSTOMER testifies
about their own intent before recovery starts; here the OPERATOR testifies
about what actually happened after an attempt, and that label re-bins the case
so the next wave treats it for what it now is, not what an error code guessed
days ago.

One rule is deliberately not expressible in this vocabulary: **a label can
never mark money as recovered.** Only the gateway can — `paid_outside` closes
a case, in its own column, and adds nothing to the recovered total. A batch
report a human can inflate by clicking is not a measurement.

Precedence: an operator label outranks even the customer's own drop-reason
answer, because it is NEWER — it describes what happened to the latest attempt,
while the drop-reason describes intent before any attempt was made.
"""
from __future__ import annotations

from typing import Any

#: Each label: what the operator clicks, what it MEANS, which failure `kind`
#: the classifier should now use (empty = the kind does not change), and the
#: side effects the label carries. `to_agent` routes the case to a full agent
#: session instead of any shared plan — for the labels where the next move is
#: a judgement call, not a treatment.
LABELS: list[dict[str, Any]] = [
    {
        "code": "promised_later",
        "label": "No balance yet — promised to pay later",
        "kind": "funds",
        "means": "the money is coming; the instrument and intent are fine",
        "next": "a quiet retry timed for when funds are likely, no contact before it",
    },
    {
        "code": "instrument_broken",
        "label": "Card or bank problem persists",
        "kind": "method",
        "means": "the customer tried; the instrument keeps refusing",
        "next": "the same amount on a different rail, at full price",
    },
    {
        "code": "checkout_glitch",
        "label": "Technical error at checkout",
        "kind": "transient",
        "means": "nothing wrong with the customer or the instrument",
        "next": "a quick retry — the cheapest thing that works",
    },
    {
        "code": "too_expensive",
        "label": "Says it is too expensive",
        "kind": "dropoff",
        "to_agent": True,
        "means": "a price objection, stated to a person",
        "next": ("a full agent session — a discount after attempts have "
                 "failed is a judgement about one customer, never a shared plan"),
    },
    {
        "code": "bad_contact",
        "label": "Wrong contact or unreachable",
        "kind": "",
        "to_agent": True,
        "means": "the messages are going nowhere",
        "next": "a person fixes the contact details or escalates",
    },
    {
        "code": "do_not_contact",
        "label": "Asked us to stop contacting them",
        "kind": "",
        "opts_out": True,
        "means": "an explicit request to be left alone",
        "next": "never contacted again, by anything, immediately",
    },
    {
        "code": "dispute",
        "label": "Disputes the charge / looks like fraud",
        "kind": "risk",
        "means": "this is not a recovery problem",
        "next": "never chased — straight to a person",
    },
    {
        "code": "paid_outside",
        "label": "Paid outside this link",
        "kind": "",
        "settles": True,
        "means": "money reportedly arrived on a rail we do not watch",
        "next": ("the case closes as settled-outside and counts in its own "
                 "column — never in recovered money, which stays "
                 "gateway-verified only"),
    },
    {
        "code": "other",
        "label": "Something else",
        "kind": "unknown",
        "wants_note": True,
        "means": "an answer this list did not anticipate",
        "next": "the agent reads the note and works the case one at a time",
    },
]

_BY_CODE = {entry["code"]: entry for entry in LABELS}


def get(code: str) -> dict[str, Any] | None:
    return _BY_CODE.get(str(code or "").strip())


def choices() -> list[dict[str, Any]]:
    """What the verdict UI renders, in this order."""
    return [{k: v for k, v in entry.items()} for entry in LABELS]
