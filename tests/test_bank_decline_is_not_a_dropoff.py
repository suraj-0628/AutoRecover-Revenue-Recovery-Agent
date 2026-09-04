"""A stale derived field must not re-diagnose a new failure as the old one.

pay_qssihjc5z, INR 12,495, live 2026-09-04. The customer cancelled once, so
the frontend wrote `decline_strategy="customer_cancelled"`. Their next attempt
was declined BY THE BANK — arriving as Razorpay's generic wrapper code
`BAD_REQUEST_ERROR` with "declined by the bank" in the reason text.

`failure_kind` consulted the catalog for (code, strategy) in that order: the
wrapper missed, the STALE strategy hit, and a bank decline was classified as a
drop-off. That authorised 5% off — INR 624.75 — on a payment the customer had
been actively trying to make, on the one family where a discount fixes nothing.

Evidence about THIS failure now outranks a field derived from an earlier one.
"""
import pytest

from recovery_agent.agent import ladder
from recovery_agent.agent.classify import failure_kind

# The exact record shape the live case had at the moment 5% was authorised.
LIVE = {
    "payment_id": "pay_qssihjc5z", "amount": 12495.0,
    "failure_code": "BAD_REQUEST_ERROR",
    "decline_strategy": "customer_cancelled",           # stale, from the cancel
    "failure_reason": ("Your payment didn't go through as it was declined by "
                       "the bank. Try another payment method or contact your bank."),
    "customer": {"email": "a@b.com", "contact": "9000000000"}, "ladder": {},
}


def test_the_live_case_is_a_method_failure_not_a_dropoff():
    assert failure_kind(LIVE) == "method"


def test_it_therefore_climbs_the_rail_switch_ladder():
    rungs = [k for k, _ in ladder.rungs_for(LIVE)]
    assert "rail_switch" in rungs
    assert rungs.index("rail_switch") < rungs.index("offer")


def test_the_discount_is_refused_until_full_price_has_failed():
    """The money consequence, end to end through the offer policy."""
    import json
    import recovery_agent.state_store as state_store
    from recovery_agent.agent.tools import get_recovery_offer
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    old = state_store._DATA_DIR
    state_store._DATA_DIR = tmp
    state_store.StateStore.reset_instances()
    try:
        s = state_store.StateStore()
        s.save_payment("pay_qssihjc5z", dict(LIVE))
        s.flush()
        r = json.loads(get_recovery_offer.invoke({
            "amount": 12495.0, "stage": "ui_offer",
            "payment_id": "pay_qssihjc5z"}))
        assert r["allowed"] is False
        assert "refused by the bank" in r["reason"]
    finally:
        state_store._DATA_DIR = old
        state_store.StateStore.reset_instances()


# ── precedence, stated as rules ─────────────────────────────────────────────

def test_an_exact_failure_code_still_wins():
    assert failure_kind({"failure_code": "insufficient_funds",
                         "decline_strategy": "customer_cancelled"}) == "funds"


def test_a_genuine_dropoff_is_still_a_dropoff():
    assert failure_kind({"failure_code": "customer_cancelled",
                         "decline_strategy": "customer_cancelled",
                         "failure_reason": "Payment cancelled by customer"}) == "dropoff"


@pytest.mark.parametrize("strategy,kind", [
    ("insufficient_funds", "funds"),
    ("user_dropoff", "dropoff"),
    ("bank_declined", "method"),
])
def test_the_derived_field_still_carries_records_that_have_nothing_else(strategy, kind):
    """Most historical records have only `decline_strategy` — it stays a real
    signal, just the weakest one."""
    assert failure_kind({"decline_strategy": strategy}) == kind


def test_a_wrapper_code_with_no_reason_falls_back_to_the_strategy():
    assert failure_kind({"failure_code": "BAD_REQUEST_ERROR",
                         "decline_strategy": "insufficient_funds"}) == "funds"


# ── the checkout must stop manufacturing the stale signal ───────────────────

def test_a_close_after_a_failure_is_marked_as_such():
    """Neither a fresh drop-off nor nothing at all.

    The page must not silently re-diagnose a bank decline as a change of mind
    (that authorised 5% off a payment the customer was trying to make), but it
    must still TELL the server they walked — that is what distinguishes a
    hesitant customer from one who never hit a failure.

    It must NOT close the window for them: Razorpay's own screen offers other
    rails after a decline, and a customer reaching for UPI themselves is the
    cheapest recovery available. An earlier version called `rzp.close()` here
    and interrupted exactly that."""
    from pathlib import Path
    import recovery_agent.frontend as F
    text = Path(F.__file__).read_text()
    assert "_closedAfterFailure = true;" in text
    assert "rzp.close();" not in text, \
        "let the customer use Razorpay's own retry options"
    i = text.index("modal: {ondismiss")
    body = text[i:i + 2000]
    assert "if (_closedAfterFailure)" in body
    assert "Closed the checkout after a failed attempt" in body, \
        "the walk-away is still reported, so the server can record it"


def test_walking_away_after_a_failure_is_not_the_same_ladder():
    """Three journeys, three ladders — the point of the whole distinction."""
    from recovery_agent.agent import ladder
    base = dict(LIVE)
    still_trying = [k for k, _ in ladder.rungs_for(base)]
    walked = [k for k, _ in ladder.rungs_for({**base, "abandoned_after_failure": True})]
    pure_drop = [k for k, _ in ladder.rungs_for(
        {**base, "failure_code": "customer_cancelled",
         "failure_reason": "Payment cancelled by customer"})]

    # Still trying: the rail is the problem, so a working rail comes first.
    assert still_trying.index("rail_switch") < still_trying.index("offer")
    # Walked away: reluctance is proven, so the reason comes before the route.
    assert walked.index("offer") < walked.index("rail_switch")
    # Never failed: nothing to switch to — there is no rail_switch rung at all.
    assert "rail_switch" not in pure_drop


def test_walking_away_unlocks_the_discount_without_spending_a_link():
    """A method case normally earns the discount by trying full price first.
    Someone who saw the alternatives and closed has already shown us."""
    import json, tempfile
    from pathlib import Path as _P
    import recovery_agent.state_store as state_store
    from recovery_agent.agent.tools import get_recovery_offer

    tmp = _P(tempfile.mkdtemp())
    old_dir = state_store._DATA_DIR
    state_store._DATA_DIR = tmp
    state_store.StateStore.reset_instances()
    try:
        s = state_store.StateStore()
        s.save_payment("pay_qssihjc5z",
                       {**LIVE, "abandoned_after_failure": True})
        s.flush()
        r = json.loads(get_recovery_offer.invoke({
            "amount": 12495.0, "stage": "ui_offer",
            "payment_id": "pay_qssihjc5z"}))
        assert r["allowed"] is True, "walking away is the evidence"
        assert r["discount_pct"] > 0
    finally:
        state_store._DATA_DIR = old_dir
        state_store.StateStore.reset_instances()


def test_a_new_failure_refreshes_the_derived_field():
    from pathlib import Path
    import recovery_agent.frontend as F
    text = Path(F.__file__).read_text()
    i = text.index("def payment_failed")
    # To the end of the function, not a fixed byte count. The window used to
    # be `i + 4600`; adding a comment inside payment_failed pushed the line
    # this asserts on past the cutoff and failed a test about a behaviour that
    # had not changed. A slice that moves when prose moves tests the prose.
    body = text[i:text.index("\n@app.route", i)]
    assert "decline_strategy=failure_code" in body, \
        "the derived field must not outlive the failure it was derived from"
