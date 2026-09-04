"""The champion mechanism: one worked case becomes the bin's plan.

The tests are shaped around the two promises in the module docstring: the
*shape* of the champion's decision generalises, and nothing *personal* to the
champion does. The second promise is the one that costs money when broken, so
it gets the sharpest tests: a discount never scales, and no message body is
ever taken from the champion's session.
"""
import pytest

from recovery_agent.batch import distill
from recovery_agent.batch.plan import BatchPlan


def champion(**kw) -> dict:
    rec = {"payment_id": "pay_champ", "amount": 7999.0, "status": "failed",
           "decline_strategy": "card_expired",
           "customer": {"email": "champ@x.com", "name": "Asha Champion"},
           "ladder": {}, "recovery_links": [], "contacts": []}
    rec.update(kw)
    return rec


def run_distill(rec, before=None, key="bank_declined", tier="medium"):
    return distill.distill(rec, before or {"rungs": set(), "links": 0,
                                           "contacts": 0},
                           batch_key=key, tier=tier)


# ── the shape of the decision generalises ────────────────────────────────

def test_a_full_price_link_becomes_the_bin_s_step():
    rec = champion(
        ladder={"rail_switch": {"at": "2026-09-04T10:00:00+00:00",
                                "detail": "UPI link sent at full price"}},
        recovery_links=[{"link_id": "plink_1", "amount": 7999.0}],
        contacts=[{"channel": "email", "at": "2026-09-04T10:00:01+00:00"}])
    plan = run_distill(rec)
    step = plan.steps["rail_switch"]
    assert step.action == "link_and_notify" and step.full_price
    assert plan.provenance["champion"] == "pay_champ"
    assert plan.provenance["observed_rungs"] == ["rail_switch"]


def test_a_scheduled_retry_becomes_the_bin_s_step_with_its_hours():
    rec = champion(
        decline_strategy="insufficient_funds",
        ladder={"silent_retry": {"at": "2026-09-04T10:00:00+00:00",
                                 "detail": "retry scheduled in 18.0h"}})
    plan = run_distill(rec, key="insufficient_funds")
    assert plan.steps["silent_retry"].action == "retry"
    assert plan.retry_hours == 18.0, "the agent's chosen wait, not a default"


def test_rungs_are_read_in_the_order_the_agent_climbed_them():
    rec = champion(
        decline_strategy="insufficient_funds",
        ladder={"offer": {"at": "2026-09-04T11:00:00+00:00", "detail": ""},
                "silent_retry": {"at": "2026-09-04T10:00:00+00:00",
                                 "detail": "retry scheduled in 12.0h"}},
        recovery_links=[{"link_id": "plink_1", "amount": 7999.0}],
        contacts=[{"channel": "email", "at": "x"}])
    plan = run_distill(rec, key="insufficient_funds")
    assert plan.provenance["observed_rungs"] == ["silent_retry", "offer"]


# ── nothing personal to the champion scales ──────────────────────────────

def test_a_discount_is_never_generalised():
    """The one decision that must not scale: the agent granted THIS customer
    a discount for reasons visible in this session only. The rung goes to the
    agent per case, not to forty inboxes at 10 percent off."""
    rec = champion(
        ladder={"rail_switch": {"at": "2026-09-04T10:00:00+00:00",
                                "detail": "5% off"}},
        recovery_links=[{"link_id": "plink_1", "amount": 7599.05}],
        contacts=[{"channel": "email", "at": "x"}])
    plan = run_distill(rec)
    assert plan.steps["rail_switch"].action == "exception"
    assert "discount" in plan.steps["rail_switch"].why


def test_the_message_body_is_policy_prose_not_the_champion_s():
    """Prose is where one customer's name and history leak into another's
    inbox, so the body always comes from the merchant's template."""
    rec = champion(
        ladder={"rail_switch": {"at": "x", "detail": "sent to Asha Champion"}},
        recovery_links=[{"link_id": "plink_1", "amount": 7999.0}],
        contacts=[{"channel": "email", "at": "x"}])
    plan = run_distill(rec)
    assert "Asha" not in plan.body and "champ@x.com" not in plan.body
    assert "{name}" in plan.body and "{link}" in plan.body


# ── sessions that leave nothing reusable ─────────────────────────────────

@pytest.mark.parametrize("rec,why", [
    (champion(), "climbed no rung"),
    (champion(status="escalated",
              ladder={"rail_switch": {"at": "x", "detail": ""}}), "escalated"),
    (champion(failure_reason="suspected fraud",
              ladder={"rail_switch": {"at": "x", "detail": ""}}),
     "not pursuable"),
])
def test_a_session_with_nothing_to_reuse_fails_loudly(rec, why):
    """The caller falls back to the policy default — but only because this
    raised, never because a half-understood session was half-distilled."""
    with pytest.raises(distill.DistillationFailed, match=why):
        run_distill(rec)


def test_only_what_the_session_added_is_distilled():
    """Rungs climbed last week are history, not advice."""
    rec = champion(
        ladder={"page_push": {"at": "2026-09-01T09:00:00+00:00", "detail": ""},
                "rail_switch": {"at": "2026-09-04T10:00:00+00:00",
                                "detail": ""}},
        recovery_links=[{"link_id": "old", "amount": 7999.0},
                        {"link_id": "new", "amount": 7999.0}],
        contacts=[{"channel": "email", "at": "x"}])
    before = {"rungs": {"page_push"}, "links": 1, "contacts": 0}
    plan = distill.distill(rec, before, batch_key="bank_declined",
                           tier="medium")
    assert plan.provenance["observed_rungs"] == ["rail_switch"]


def test_a_live_only_rung_does_not_translate_to_the_batch():
    rec = champion(ladder={"page_push": {"at": "x", "detail": "pushed"}})
    with pytest.raises(distill.DistillationFailed, match="live-only"):
        run_distill(rec)


# ── the merge with the policy default ────────────────────────────────────

def test_the_champion_wins_where_both_speak_and_the_default_fills_the_rest():
    rec = champion(
        ladder={"rail_switch": {"at": "x", "detail": ""}},
        recovery_links=[{"link_id": "plink_1", "amount": 7999.0}],
        contacts=[{"channel": "email", "at": "x"}])
    plan = run_distill(rec)
    # The default's page_push skip survives; the champion's rail_switch step
    # replaced the default's.
    assert plan.steps["page_push"].action == "skip"
    assert "champion" in plan.steps["rail_switch"].why
    assert plan.rails, "rails come from policy; the record does not keep them"


def test_the_distilled_plan_passes_the_same_validator_as_any_other():
    """Champion or model or hand-written — one gate for all of them."""
    rec = champion(
        ladder={"rail_switch": {"at": "x", "detail": ""}},
        recovery_links=[{"link_id": "plink_1", "amount": 7999.0}],
        contacts=[{"channel": "email", "at": "x"}])
    plan = run_distill(rec)
    assert isinstance(plan, BatchPlan)
    assert plan.provenance["source"] == "champion"
    assert plan.digest()          # stable id for the audit trail
