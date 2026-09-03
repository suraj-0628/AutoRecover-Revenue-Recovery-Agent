"""Revenue that cannot be recovered in real time, sorted before it is worked.

A declined card needs a different rail; an empty account needs a different day;
someone who walked away needs a reason to come back. Worked in one queue the
agent re-derives that for every case — sorting first means it decides once and
applies many times.
"""
import json
import pathlib
import tempfile

import pytest

from recovery_agent.agent.classify import (BATCHES, BATCH_BY_KEY, classify,
                                           failure_kind, summarise)

FRONTEND = (pathlib.Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
            / "frontend.py").read_text()
INDEX = (pathlib.Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
         / "templates" / "index.html").read_text()


def rec(**over):
    r = {"payment_id": "p", "amount": 2499.0, "status": "awaiting_customer",
         "customer": {"email": "a@b.com"}, "failure_code": "", "failure_reason": ""}
    r.update(over)
    return r


# ── one definition of what kind of failure this is ──────────────────────

@pytest.mark.parametrize("code,expected", [
    ("bank_declined", "method"), ("card_expired", "method"),
    ("insufficient_funds", "funds"), ("customer_cancelled", "dropoff"),
    ("fraud_suspected", "risk"), ("something_new", "unknown"),
])
def test_failure_kind(code, expected):
    assert failure_kind(rec(failure_code=code)) == expected


def test_perception_uses_the_same_definition():
    """Two copies would drift, and a case would be a bank decline in one place
    and a drop-off in the other."""
    PERC = (pathlib.Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
            / "agent" / "perception.py").read_text()
    assert "from recovery_agent.agent.classify import failure_kind" in PERC


# ── which batch a case belongs in ───────────────────────────────────────

def test_a_recovered_case_is_in_no_batch():
    assert classify(rec(status="recovered")) is None
    assert classify(rec(recovered_amount=2499.0)) is None


def test_a_case_being_worked_right_now_is_in_no_batch():
    """Batching it would put two runs on one payment, which the session model
    does not allow."""
    assert classify(rec(status="recovering")) is None


def test_a_decline_goes_to_the_rail_batch():
    assert classify(rec(failure_code="bank_declined")) == "bank_declined"


def test_an_empty_account_goes_to_its_own_batch():
    assert classify(rec(failure_code="insufficient_funds")) == "insufficient_funds"


def test_a_scheduled_retry_is_not_re_worked():
    assert classify(rec(status="scheduled",
                        failure_code="bank_declined")) == "awaiting_retry"


def test_risk_outranks_everything():
    assert classify(rec(status="scheduled",
                        failure_code="fraud_suspected")) == "risk"


def test_a_case_with_no_contact_details_is_its_own_batch():
    assert classify(rec(customer={}, customer_email="",
                        customer_phone="")) == "unreachable"


def test_an_escalated_case_shows_as_with_a_human():
    assert classify(rec(status="escalated")) == "escalated"
    assert classify(rec(closed={"outcome": "escalated"})) == "escalated"


def test_an_unclassifiable_failure_is_still_counted():
    """Money at risk must never fall out of every batch and disappear."""
    assert classify(rec(failure_code="???")) is not None


# ── the summary ─────────────────────────────────────────────────────────

def test_empty_batches_are_kept():
    """An empty batch says that class of failure is not happening, and hiding
    it makes the page rearrange itself on every refresh."""
    out = summarise([rec(failure_code="bank_declined")])
    assert len(out) == len(BATCHES)
    assert {b["key"] for b in out} == set(BATCH_BY_KEY)


def test_value_at_risk_adds_up():
    out = summarise([rec(payment_id="a", amount=100.0, failure_code="bank_declined"),
                     rec(payment_id="b", amount=250.0, failure_code="bank_declined")])
    bank = [b for b in out if b["key"] == "bank_declined"][0]
    assert bank["count"] == 2 and bank["value"] == 350.0


def test_recovered_money_is_not_counted_as_at_risk():
    out = summarise([rec(status="recovered", amount=9999.0)])
    assert sum(b["value"] for b in out) == 0.0


# ── running a batch ─────────────────────────────────────────────────────

def test_a_run_never_starts_a_second_session_on_one_payment():
    i = FRONTEND.index("def api_run_batch")
    body = FRONTEND[i:i + 3000]
    assert "if pid in active_agent_payments:" in body
    assert "already being worked" in body


def test_a_run_is_capped_and_says_so():
    """A batch can hold hundreds; a run that quietly worked all of them would
    send hundreds of emails and burn the payment link quota in one click."""
    i = FRONTEND.index("def api_run_batch")
    body = FRONTEND[i:i + 3000]
    assert "BATCH_RUN_LIMIT" in FRONTEND
    assert '"limit": limit' in body and '"total_in_batch"' in body
    assert "limit " in INDEX and "per run" in INDEX, (
        "a partial run must not read as a complete one"
    )


def test_a_run_skips_cases_with_no_way_to_contact_them():
    i = FRONTEND.index("def api_run_batch")
    assert "no way to contact them" in FRONTEND[i:i + 3000]


def test_an_unknown_batch_is_rejected():
    i = FRONTEND.index("def api_run_batch")
    assert "unknown batch" in FRONTEND[i:i + 800]


# ── the view ────────────────────────────────────────────────────────────

def test_the_rail_does_not_restructure_the_live_view():
    """Fixed rather than a grid column, so the existing workspace is shifted
    rather than rebuilt — the live view stays exactly as it was."""
    assert "position: fixed" in INDEX[INDEX.index(".railnav {"):INDEX.index(".railnav {") + 300]
    assert 'showView(\'live\')' in INDEX and 'showView(\'batches\')' in INDEX


def test_the_batch_view_uses_the_existing_design_tokens():
    i = INDEX.index(".batch-card {")
    body = INDEX[i:i + 400]
    assert "var(--panel-bg)" in body and "var(--border-color)" in body
