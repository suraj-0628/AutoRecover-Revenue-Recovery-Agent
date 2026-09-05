"""Sort revenue that is still at risk into batches a single agent run can work.

Real-time recovery handles one payment while the customer is still there. Most
failures are not like that: the bank said no, the account was empty, a retry is
waiting on a clock, a person has the case. Those pile up, and worked one at a
time they are all context-switch and no leverage.

They are also not the same problem as each other, which is the point of sorting
them first. A declined card needs a different rail. An empty account needs a
different *day*. Someone who chose not to pay needs a reason to come back. Doing
them in one undifferentiated queue means the agent re-derives the category for
every case; doing them in batches means it decides once and applies it many
times.

The categories come from what the case record already holds — failure code,
ladder position, scheduled jobs, contact details — so nothing here needs a model
to run, and a batch is the same shape whether it holds one case or two hundred.
"""
from __future__ import annotations

from typing import Any

#: What each known failure type means for what to DO about it.
#:
#: - method    the instrument was refused. Another rail, at full price.
#: - funds     the account was short. Another day, not another message.
#: - transient nothing was wrong with the customer or the card; the plumbing
#:             failed. Retry it. Do NOT apologise with money.
#: - dropoff   they chose not to complete. The only lever is a reason to return.
#: - risk      do not pursue; a person decides.
_KIND_BY_TYPE = {
    "card_expired": "method",
    "bank_declined": "method",
    "mandate_revoked": "method",
    "insufficient_funds": "funds",
    "network_timeout": "transient",
    "risk_block": "risk",
    "user_dropoff": "dropoff",
}

#: Fallback only, for a code the catalog has never seen. Deliberately does NOT
#: contain "timeout": that word appears in `gateway_timeout` and `network_timeout`,
#: and matching it as a drop-off is how a 504 came to be described to the agent as
#: "the customer chose not to complete. Nothing is broken."
_METHOD_WORDS = ("declin", "expired", "invalid", "issuer", "bank", "mandate",
                 "authentication", "cvv", "card_not_supported", "lost", "stolen")
_FUNDS_WORDS = ("insufficient", "funds", "limit_exceeded", "balance")
_TRANSIENT_WORDS = ("timeout", "timed out", "network", "gateway", "unavailable",
                    "5xx", "502", "503", "504", "connection")
_DROPOFF_WORDS = ("cancel", "dropoff", "drop_off", "abandon", "closed")
_RISK_WORDS = ("fraud", "risk", "dispute", "chargeback", "suspected", "frozen")

#: Values of `decline_strategy` that name a state rather than a cause. Treating
#: these as a diagnosis would be worse than treating the field as empty.
_NOT_A_CAUSE = {"", "no_response", "unknown", "none", "null"}


def failure_kind(record: dict) -> str:
    """What kind of problem this is — method, funds, transient, dropoff, risk.

    One definition, used by the batch view, the agent's briefing and the offer
    policy. Two copies would drift, and a case would then be a bank decline in
    one place and a drop-off in the other.

    The catalog is consulted FIRST. It was already there — 14 codes mapped to
    failure types in `diagnosis.FAILURE_CODE_MAP` — while this function matched
    word fragments and disagreed with it: `gateway_timeout` came out as a
    drop-off, `do_not_honor` and `mandate_revoked` as unknown. Substring
    matching is the fallback for codes the catalog has never seen, not the rule.
    """
    code = str(record.get("failure_code") or "").strip().lower()

    # `decline_strategy` is where the cause actually lives.
    #
    # `failure_code` is set on 14 of 80 live records; `decline_strategy` — which
    # `frontend.py` writes on every run — is set on 73. Reading only the first
    # put five cases worth INR 17,995 into the drop-off batch, four of them
    # `card_expired` and one `insufficient_funds`, in the batch whose whole
    # premise is "the only lever here is a reason to come back".
    #
    # Nothing leaked, because `get_recovery_offer` re-derives the kind from the
    # same empty field, gets `unknown`, and fails closed. The control held by
    # accident. A batch plan is decided once and applied many times, so bad
    # input here is multiplied by the size of the batch.
    strategy = str(record.get("decline_strategy") or "").strip().lower()
    if strategy in _NOT_A_CAUSE:
        strategy = ""

    # THE OPERATOR'S LABEL FIRST OF ALL.
    #
    # It outranks even the customer's own drop-reason answer, because it is
    # NEWER: the drop-reason describes intent before any attempt was made,
    # while the label describes what happened to the latest attempt. A
    # customer who said "no balance" last week and whose card the bank now
    # refuses is a method problem today, whatever they were then.
    labeled = (record.get("operator_label") or {}).get("code") or ""
    if labeled:
        try:
            from recovery_agent.labels import get as _label
            spec = _label(labeled)
            if spec and spec.get("kind"):
                return spec["kind"]
        except Exception:
            pass

    # THE CUSTOMER'S OWN ANSWER FIRST.
    #
    # Everything below infers intent from an error code. When the customer has
    # actually been asked and has told us, inference is not merely weaker — it
    # is answering a question that has already been answered. "I had no
    # balance" and "I found it cheaper elsewhere" produce the SAME bank
    # decline and demand opposite responses: one needs time and no discount,
    # the other needs a discount and no waiting.
    stated = (record.get("drop_reason") or {}).get("code") or ""
    if stated:
        try:
            from recovery_agent.drop_reasons import get as _reason
            spec = _reason(stated)
            if spec and spec.get("kind"):
                return spec["kind"]
        except Exception:
            pass

    reason = str(record.get("failure_reason") or "").lower()

    def _catalog(candidate: str) -> str:
        if not candidate:
            return ""
        try:
            from recovery_agent.agent.diagnosis import FAILURE_CODE_MAP
            known = FAILURE_CODE_MAP.get(candidate)
        except Exception:
            return ""
        if known is None:
            return ""
        return _KIND_BY_TYPE.get(getattr(known, "value", str(known)), "")

    def _words(blob: str) -> str:
        if any(w in blob for w in _RISK_WORDS):
            return "risk"
        if any(w in blob for w in _FUNDS_WORDS):
            return "funds"
        if any(w in blob for w in _TRANSIENT_WORDS):
            return "transient"
        if any(w in blob for w in _METHOD_WORDS):
            return "method"
        if any(w in blob for w in _DROPOFF_WORDS):
            return "dropoff"
        return ""

    # PRECEDENCE: evidence about THIS failure beats a field derived from an
    # earlier one.
    #
    # `decline_strategy` used to be consulted right after `failure_code`, ahead
    # of the failure text. Live (pay_qssihjc5z, INR 12,495): the customer
    # cancelled once, so the frontend wrote decline_strategy="customer_cancelled";
    # their next attempt was declined BY THE BANK, arriving as the generic
    # wrapper code BAD_REQUEST_ERROR with "declined by the bank" in the reason.
    # The wrapper missed the catalog, the stale strategy hit it, and a bank
    # decline was classified as a drop-off — which authorised 5% off (INR
    # 624.75) on a payment the customer had been trying to MAKE.
    #
    # The reason text describes the failure in hand; the strategy is a
    # left-over. So: exact code, then the live text, then the derived field.
    kind = _catalog(code)
    if kind:
        return kind

    kind = _words(f"{code} {reason}")
    if kind:
        return kind

    kind = _catalog(strategy)
    if kind:
        return kind

    blob = f"{code} {strategy} {reason}"
    if any(w in blob for w in _RISK_WORDS):
        return "risk"
    if any(w in blob for w in _FUNDS_WORDS):
        return "funds"
    if any(w in blob for w in _TRANSIENT_WORDS):
        return "transient"
    if any(w in blob for w in _METHOD_WORDS):
        return "method"
    if any(w in blob for w in _DROPOFF_WORDS):
        return "dropoff"
    return "unknown"


#: Ordered so the most actionable batch reads first. `key` is the API/URL id.
#: `runnable` says whether a single shared plan may act on the whole bucket.
#: It lives here rather than in the page, because "can this be worked in
#: bulk" is a property of the failure class, not of how it is displayed.
BATCHES: list[dict[str, str]] = [
    {"key": "bank_declined", "title": "Bank declined",
     "what": "The instrument was refused. These need a different rail, at full "
             "price — the price was never the problem.",
     "runnable": True},
    {"key": "insufficient_funds", "title": "No money at the time",
     "what": "The account was short. These need a different day, not a "
             "different message — retry timed to when they are likely paid.",
     "runnable": True},
    {"key": "transient", "title": "Failed in transit",
     "what": "The gateway or the network dropped it — nothing was wrong with "
             "the customer or the card. These want a retry, not a message and "
             "certainly not a discount.",
     "runnable": True},
    {"key": "dropoff", "title": "Chose not to complete",
     "what": "Nothing broke; they walked away. The only lever here is a reason "
             "to come back, which is what an offer is for.",
     "runnable": True},
    {"key": "awaiting_retry", "title": "Retry already scheduled",
     "what": "A silent retry is on the clock. Nothing to do until it fires — "
             "listed so the money is visible, not forgotten.",
     "runnable": False},
    {"key": "escalated", "title": "With a human",
     "what": "The ladder was exhausted and a person has the case.",
     "runnable": False},
    {"key": "unclassified", "title": "Cause not established",
     "what": "The failure code, the decline strategy and the reason text all "
             "came back empty or unrecognised. A batch plan needs a shared "
             "cause to stand on, so these are not planned — they go to the "
             "agent one at a time.",
     "runnable": False},
    {"key": "risk", "title": "Risk / dispute",
     "what": "Fraud, risk or a dispute. These are not chased — they go straight "
             "to a person.",
     "runnable": False},
]

BATCH_BY_KEY = {b["key"]: b for b in BATCHES}

#: Statuses that mean the money is no longer at risk.
#: `settled_outside` is a human's report of money on a rail we do not watch —
#: it leaves the batches exactly like a recovery, but its amount is never added
#: to recovered money, which stays gateway-verified only.
_SETTLED = ("recovered", "settled_outside")
#: A case the real-time agent is still working. Batching it would put two runs
#: on one payment, which the session model does not allow.
_IN_FLIGHT = ("recovering",)


def classify(record: dict) -> str | None:
    """Which batch this case belongs to, or None if it does not belong in one."""
    status = str(record.get("status") or "")
    if status in _SETTLED or float(record.get("recovered_amount") or 0) > 0:
        return None
    if record.get("closed"):
        closed = record.get("closed") or {}
        if closed.get("outcome") == "escalated":
            return "escalated"
        if closed.get("outcome") == "recovered":
            return None
    if status in _IN_FLIGHT:
        return None

    kind = failure_kind(record)
    if kind == "risk":
        return "risk"
    if status == "escalated":
        return "escalated"

    # A case with no contact details is not a batch, it is an artifact.
    #
    # The checkout will not start a payment until name, a valid email and a
    # ten-digit phone are all present (`validateCustomerForm`), so no real
    # customer journey can produce a record without them. The ones that existed
    # were leftovers from before the customer dict was persisted at ingress.
    #
    # Counting them inflated revenue-at-risk with money nobody ever tried to
    # pay: 138 of 186 cases, INR 9,72,527, none of it workable and none of it
    # real.
    customer = record.get("customer") or {}
    if not (customer.get("email") or record.get("customer_email")
            or customer.get("contact") or record.get("customer_phone")):
        return None

    job = record.get("scheduled_job") or {}
    # Every terminal state means "this will not fire": a completed or failed
    # job is no more pending than a cancelled one, and none of them may file
    # the case under "nothing to do until it fires".
    _job_over = str(job.get("status") or "").lower() in (
        "cancelled", "completed", "failed")
    if status == "scheduled" or (job and not _job_over):
        return "awaiting_retry"

    if kind == "funds":
        return "insufficient_funds"
    if kind == "transient":
        return "transient"
    if kind == "method":
        return "bank_declined"
    if kind == "dropoff":
        return "dropoff"

    # An unclassified failure is NOT a drop-off.
    #
    # This used to end `return "dropoff"  # an unclassified drop is still a
    # drop`, and that line is how a card_expired reached the batch whose fix is
    # a discount. "We do not know why this failed" is precisely the case where a
    # shared plan has no shared cause to stand on, so it gets its own bucket and
    # is worked one case at a time.
    return "unclassified"


def summarise(records: list[dict]) -> list[dict[str, Any]]:
    """Every batch, with what it holds. Empty batches are kept.

    A batch with nothing in it is information — it says that class of failure is
    not happening — and hiding it makes the page rearrange itself every refresh.
    """
    buckets: dict[str, list[dict]] = {b["key"]: [] for b in BATCHES}
    for rec in records:
        key = classify(rec)
        if key:
            buckets[key].append(rec)

    out = []
    for meta in BATCHES:
        items = buckets[meta["key"]]
        out.append({
            **meta,
            "count": len(items),
            "value": round(sum(float(r.get("amount") or 0) for r in items), 2),
            "payments": [{
                "payment_id": r.get("payment_id"),
                "amount": float(r.get("amount") or 0),
                "failure_code": r.get("failure_code") or "",
                "failure_reason": (r.get("failure_reason") or "")[:120],
                "customer": (r.get("customer") or {}).get("email")
                            or r.get("customer_email") or "",
                "status": r.get("status"),
                "climbed": list((r.get("ladder") or {}).keys()),
            } for r in items],
        })
    return out
