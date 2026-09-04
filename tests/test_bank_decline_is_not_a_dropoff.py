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

def test_the_page_does_not_report_a_dropoff_after_a_failure():
    """The customer is stuck behind Razorpay's iframe with an invisible offer
    underneath it. Their only escape is closing the modal — which used to be
    filed as "changed their mind"."""
    from pathlib import Path
    import recovery_agent.frontend as F
    text = Path(F.__file__).read_text()
    assert "_closedAfterFailure = true;" in text
    assert "rzp.close();" in text, "get them out of the dead rail-selection screen"
    i = text.index("modal: {ondismiss")
    body = text[i:i + 900]
    assert "if (_closedAfterFailure)" in body and "return;" in body, \
        "a dismissal we caused must not be reported as a customer cancelling"


def test_a_new_failure_refreshes_the_derived_field():
    from pathlib import Path
    import recovery_agent.frontend as F
    text = Path(F.__file__).read_text()
    i = text.index("def payment_failed")
    body = text[i:i + 4600]
    assert "decline_strategy=failure_code" in body, \
        "the derived field must not outlive the failure it was derived from"
