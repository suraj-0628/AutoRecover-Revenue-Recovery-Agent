"""One ladder for every failure was the generic-agent problem itself.

Before this, `page_push -> offer -> voice -> post_call_email -> alternate_path`
was climbed by every case regardless of what broke, so:

  - an insufficient-funds case (C1 of the 2026-09-03 matrix) was made to push
    and then email a discount before it was allowed to schedule the retry that
    was the only thing capable of working — and was then escalated to a human
    while two retries were still on the clock;
  - a bank decline was steered toward a discount when the customer had TRIED
    to pay and the problem was the instrument, not the price;
  - a gateway timeout could reach a discount rung at all, which pays the
    customer to forgive our own outage.

The rungs stay coarse and the agent still decides what to say and when. It just
climbs the ladder that belongs to THIS failure.
"""
from datetime import datetime, timedelta, timezone

import pytest

from recovery_agent.agent import ladder


def rec(**over) -> dict:
    base = {"payment_id": "pay_k", "amount": 4999.0,
            "customer": {"email": "a@b.com", "contact": "9000000000"},
            "ladder": {}}
    base.update(over)
    return base


def climbed(*rungs) -> dict:
    return {r: {"at": datetime.now(timezone.utc).isoformat(), "detail": ""}
            for r in rungs}


def rungs(**over) -> list[str]:
    return [k for k, _ in ladder.rungs_for(rec(**over))]


# ── the ladder is chosen by what broke ──────────────────────────────────────

@pytest.mark.parametrize("code,first", [
    ("insufficient_funds", "silent_retry"),   # a different DAY, not a message
    ("network_timeout", "silent_retry"),      # our plumbing; just retry it
    ("gateway_timeout", "silent_retry"),
    ("bank_declined", "page_push"),           # cheapest nudge still leads
    ("customer_cancelled", "page_push"),
])
def test_the_first_rung_follows_the_failure(code, first):
    assert rungs(failure_code=code)[0] == first


def test_a_bank_decline_tries_another_rail_before_any_discount():
    seq = rungs(failure_code="bank_declined")
    assert seq.index("rail_switch") < seq.index("offer"), (
        "the customer tried to pay — a discount does not fix a declined card")


def test_a_transient_failure_has_no_discount_rung_at_all():
    assert "offer" not in rungs(failure_code="network_timeout")
    assert "offer" not in rungs(failure_code="gateway_timeout")


def test_a_dropoff_is_the_one_family_where_the_discount_comes_early():
    seq = rungs(failure_code="customer_cancelled")
    assert seq[:2] == ["page_push", "offer"]
    assert "rail_switch" not in seq, "nothing was refused; there is no rail to switch"


def test_an_unclassified_failure_fails_closed():
    seq = rungs(failure_code="something_new_and_odd")
    assert seq.index("rail_switch") < seq.index("offer")


def test_risk_has_no_ladder_at_all():
    assert rungs(failure_code="fraud_suspected") == []
    assert ladder.pursuit_barred(rec(failure_code="fraud_suspected"))


# ── the rung a case is actually on ──────────────────────────────────────────

def test_funds_leads_with_the_retry_not_a_message():
    n = ladder.next_rung(rec(failure_code="insufficient_funds"))
    assert n["rung"] == "silent_retry"


def test_a_declined_card_goes_to_the_rail_switch_after_the_push():
    n = ladder.next_rung(rec(failure_code="bank_declined",
                             ladder=climbed("page_push")))
    assert n["rung"] == "rail_switch"


def test_a_dropoff_goes_to_the_offer_after_the_push():
    n = ladder.next_rung(rec(failure_code="customer_cancelled",
                             ladder=climbed("page_push")))
    assert n["rung"] == "offer"


def test_the_discount_opens_up_once_full_price_has_also_failed():
    n = ladder.next_rung(rec(failure_code="bank_declined",
                             ladder=climbed("page_push", "rail_switch")))
    assert n["rung"] == "offer", "reluctance is now proven; the discount is honest"


# ── a scheduled retry means waiting, not exhaustion ─────────────────────────

def _with_job(status="scheduled", hours=24):
    when = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
    return rec(failure_code="insufficient_funds", customer={},
               ladder=climbed("silent_retry", "page_push"),
               scheduled_job={"status": status, "target_timestamp": when})


def test_a_pending_retry_blocks_exhaustion():
    assert ladder.retry_pending(_with_job())
    assert not ladder.exhausted(_with_job())


def test_a_finished_retry_does_not():
    assert not ladder.retry_pending(_with_job(status="completed"))
    assert ladder.exhausted(_with_job(status="completed"))


def test_a_retry_whose_time_has_passed_does_not_block_forever():
    assert not ladder.retry_pending(_with_job(hours=-2))


def test_escalation_is_refused_while_a_retry_is_still_coming(monkeypatch, tmp_path):
    """The C1 failure exactly: a human was handed a case that was working."""
    import recovery_agent.state_store as state_store
    from recovery_agent.agent.tools import escalate_to_human
    import json

    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    state_store.StateStore.reset_instances()
    try:
        s = state_store.StateStore()
        s.save_payment("pay_k", _with_job())
        s.flush()
        r = json.loads(escalate_to_human.invoke(
            {"payment_id": "pay_k", "reason": "nothing is working"}))
        assert r["status"] == "blocked"
    finally:
        state_store.StateStore.reset_instances()


# ── every rung a ladder names must be a real, recordable rung ───────────────

def test_every_rung_in_every_ladder_is_recordable():
    for kind, seq in ladder.RUNGS_BY_KIND.items():
        for key, label in seq:
            assert key in ladder.ALL_RUNGS, f"{kind}: {key} is not a known rung"
            assert label, f"{kind}: {key} has no human label"


def test_has_rung_answers_per_case():
    assert ladder.has_rung(rec(failure_code="bank_declined"), "rail_switch")
    assert not ladder.has_rung(rec(failure_code="customer_cancelled"), "rail_switch")
    assert not ladder.has_rung(rec(failure_code="network_timeout"), "offer")
