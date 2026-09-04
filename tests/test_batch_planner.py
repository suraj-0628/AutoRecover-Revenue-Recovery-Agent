"""The plans, and their one dangerous coupling.

A plan says what to do on a rung. The rungs come from `ladder.RUNGS_BY_KIND`,
which is per failure kind — a declined instrument climbs
`page_push -> rail_switch -> offer`, a short account climbs
`silent_retry -> page_push -> offer`. Nothing in the type system ties a plan's
keys to that list, so a rung renamed on one side turns every case in the batch
into an exception on the other, silently and at full batch size. That already
happened once. These tests are the tie.
"""
import pytest

from recovery_agent.agent import ladder
from recovery_agent.batch.plan import PlanRejected
from recovery_agent.batch.planner import (plan_for, plans_for,
                                          runnable_batches)
from recovery_agent.batch.tiers import load_tiers

#: A record that lands in each batch, with contact details and nothing climbed.
SAMPLES = {
    "bank_declined": "card_expired",
    "insufficient_funds": "insufficient_funds",
    "transient": "gateway_timeout",
    "dropoff": "user_dropoff",
}


def rec(strategy: str, amount: float = 7999.0) -> dict:
    return {"payment_id": "pay_1", "amount": amount, "status": "failed",
            "decline_strategy": strategy,
            "customer": {"email": "a@b.com", "name": "A"}}


# ── the coupling ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("batch,strategy", sorted(SAMPLES.items()))
def test_the_plan_covers_the_rung_the_case_is_actually_on(batch, strategy):
    """The test that would have caught it: the ladder's first available rung for
    this failure kind must be a rung the plan has an answer for. A miss is not a
    bug in one case — it is every case in the batch becoming an exception."""
    record = rec(strategy)
    nxt = ladder.next_rung(record)
    assert nxt is not None, f"{batch} has no reachable rung at all"
    step = plan_for(batch, "medium").step_for(nxt["rung"])
    assert step is not None, (
        f"{batch} starts on {nxt['rung']!r}, which no plan step covers")
    assert step.action != "skip", (
        f"{batch} would skip its own first rung")


@pytest.mark.parametrize("batch", sorted(SAMPLES))
def test_every_step_names_a_rung_that_exists_for_that_failure_kind(batch):
    """A plan cannot invent a rung. Every key has to appear in the ladder this
    batch's cases actually climb."""
    from recovery_agent.agent.classify import failure_kind
    kind = failure_kind(rec(SAMPLES[batch]))
    real = {r for r, _ in ladder.RUNGS_BY_KIND[kind]}
    assert set(plan_for(batch, "medium").steps) <= real, (
        f"{batch} plans for rungs its cases never reach")


@pytest.mark.parametrize("batch", sorted(SAMPLES))
def test_no_plan_reaches_for_the_rung_the_ladder_leaves_undefined(batch):
    """`ladder.py` leaves `alternate_path` undefined because which route is
    worth trying after everything else failed is the agent's judgement. A plan
    naming one would be inventing policy for the case that has already resisted
    the policy."""
    assert "alternate_path" not in plan_for(batch, "medium").steps


# ── coverage ─────────────────────────────────────────────────────────────

def test_every_runnable_batch_has_a_plan_in_every_band():
    for batch in runnable_batches():
        for band in load_tiers():
            plan = plan_for(batch, band)
            assert plan.steps and plan.tier == band.key


def test_a_batch_with_no_shared_cause_gets_no_plan():
    """`unclassified` is where the failure code, the decline strategy and the
    reason text all came back empty. A plan is a shared decision, and there is
    no shared cause here to base one on."""
    assert "unclassified" not in runnable_batches()
    with pytest.raises(PlanRejected, match="no plan is defined"):
        plan_for("unclassified", "medium")


def test_risk_is_never_planned_for():
    assert "risk" not in runnable_batches()
    with pytest.raises(PlanRejected):
        plan_for("risk", "medium")


def test_only_the_bands_present_are_planned_for():
    """A plan is a decision about specific customers; there is no decision to
    make about none of them."""
    plans, rejected = plans_for("dropoff", [rec("user_dropoff", 999),
                                            rec("user_dropoff", 7999)])
    assert sorted(plans) == ["medium", "small"]
    assert rejected == []


# ── what the defaults commit to ──────────────────────────────────────────

def test_a_declined_instrument_is_not_discounted():
    """The customer tried to pay. That is an instrument problem, and a discount
    solves nothing a different rail does not solve for free."""
    plan = plan_for("bank_declined", "medium")
    assert plan.offer_stage is None
    assert plan.steps["rail_switch"].full_price is True
    assert set(plan.rails) == {"upi", "netbanking"}


def test_a_short_account_gets_a_free_retry_not_a_paid_link():
    """A Razorpay test account has 30 payment links for its lifetime and
    `retry_in_hours` is free and unlimited. The batches a retry actually helps
    must not spend the scarce thing."""
    for batch in ("insufficient_funds", "transient"):
        plan = plan_for(batch, "medium")
        assert plan.steps[ladder.RUNGS_BY_KIND[
            "funds" if batch == "insufficient_funds" else "transient"
        ][0][0]].action == "retry"
        assert plan.retry_hours


@pytest.mark.parametrize("band,hours", [("micro", 12.0), ("small", 16.0),
                                        ("medium", 18.0), ("large", 24.0)])
def test_the_retry_gap_is_the_window_divided_by_the_attempts(band, hours):
    """The policy says `72 hours (4 attempts)` — a span and a count, not a gap.
    Reading the span as the gap would schedule a medium retry three days out and
    a high-value one a fortnight out, which is the opposite of what a policy
    asking for *more* attempts on a larger order means."""
    assert plan_for("insufficient_funds", band).retry_hours == hours


def test_a_blip_is_retried_sooner_than_the_band_would_pace_it():
    """The batch knows the cause and the band knows the customer's value;
    whichever says sooner wins. A dropped gateway does not need eighteen hours."""
    assert plan_for("transient", "large").retry_hours == 1.0


def test_a_discount_after_full_price_failed_goes_to_a_person():
    for batch in ("bank_declined", "insufficient_funds", "dropoff"):
        step = plan_for(batch, "medium").steps.get("offer")
        if batch == "dropoff":
            assert step.action == "link_and_notify" and step.full_price
        else:
            assert step.action == "exception"


def test_a_customer_still_on_the_page_is_left_to_the_live_agent():
    for batch in runnable_batches():
        step = plan_for(batch, "medium").steps.get("page_push")
        assert step is not None and step.action == "skip"


# ── provenance ───────────────────────────────────────────────────────────

def test_a_plan_says_which_policy_authorised_it():
    plan = plan_for("dropoff", "medium")
    assert plan.provenance["policy_source"] == "merchant_dunning_rules.md"
    assert plan.provenance["kb_tier"] == "medium"
    assert plan.provenance["source"] == "default"
    assert "Medium" in plan.cause, "the band is named in the stated cause"


def test_the_seam_for_a_model_written_plan_is_the_same_validator():
    """An LLM planner drops in by producing the same dict `validate()` checks.
    Every clamp and refusal stays where it is — which is worth more than the
    call itself."""
    import inspect
    from recovery_agent.batch import planner
    assert "validate(" in inspect.getsource(planner.plan_for)
