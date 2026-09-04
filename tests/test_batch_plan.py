"""A plan is a decision applied many times, so a bad plan is a bad decision
multiplied by the batch size.

There is deliberately no partial-parse path. A plan is either fully understood
or the whole band goes to the agent one case at a time — twenty cases handled
individually is a worse day than a shared plan, but a half-understood plan
applied to twenty customers is a worse outcome.
"""
import pytest

from recovery_agent.batch.plan import (BatchBudget, BatchPlan, PlanRejected,
                                       Step, validate)


def raw(**kw):
    base = {"steps": {"offer": {"action": "link_and_notify"}},
            "body": "Hi {name}, {amount} is due: {link}", "cause": "funds"}
    base.update(kw)
    return base


def v(**kw):
    return validate(raw(**kw), batch_key="insufficient_funds", tier="medium")


# ── what a valid plan looks like ─────────────────────────────────────────

def test_a_plain_plan_survives_validation():
    plan = v()
    assert plan.step_for("offer").action == "link_and_notify"
    assert plan.batch_key == "insufficient_funds" and plan.tier == "medium"
    assert plan.provenance["policy_source"] == "merchant_dunning_rules.md"


def test_a_rung_the_plan_does_not_cover_returns_nothing():
    """The executor turns this into an exception. There is no default step —
    a plan that guessed would be improvising on a customer's behalf."""
    assert v().step_for("voice_call") is None
    assert v().step_for(None) is None


def test_the_digest_is_stable_and_distinguishes_plans():
    """The audit trail records which plan authorised an action, so two different
    plans must not share an id."""
    assert v().digest() == v().digest()
    assert v().digest() != v(body="something else entirely").digest()


# ── refusals ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad,why", [
    ({"steps": {}}, "no steps"),
    ({"steps": {"offer": {"action": "wire_them_cash"}}}, "unknown action"),
    ({"steps": {"offer": "link_and_notify"}}, "not an object"),
    (raw(rails=["upi", "carrier_pigeon"]), "unknown rails"),
    (raw(offer_stage="whatever_feels_right"), "unknown offer stage"),
    (raw(steps={"offer": {"action": "retry"}}), "needs retry_hours"),
    (raw(body="   "), "needs a message body"),
])
def test_a_plan_that_is_not_fully_understood_is_refused(bad, why):
    with pytest.raises(PlanRejected, match=why):
        validate(bad, batch_key="k", tier="medium")


def test_a_plan_that_is_not_an_object_is_refused():
    with pytest.raises(PlanRejected):
        validate("send them all an email", batch_key="k", tier="medium")


# ── clamps, for the fields where a wrong value is merely wrong ───────────

@pytest.mark.parametrize("asked,got", [(0.1, 0.5), (5, 5.0), (10_000, 168.0)])
def test_retry_hours_is_clamped_into_a_defensible_window(asked, got):
    """Sooner than half an hour is pestering; later than a week the cart is
    cold and the customer has forgotten the purchase."""
    assert v(steps={"offer": {"action": "retry"}}, retry_hours=asked
             ).retry_hours == got


def test_an_unparseable_retry_window_is_refused_not_guessed():
    with pytest.raises(PlanRejected, match="not a number"):
        v(steps={"offer": {"action": "retry"}}, retry_hours="whenever")


def test_expiry_respects_razorpay_s_fifteen_minute_floor():
    """Asking for less is a request the gateway refuses, so the plan cannot
    make a promise the link will not keep."""
    assert v(expire_in_minutes=5).expire_in_minutes == 16
    assert v(expire_in_minutes=99999).expire_in_minutes == 60 * 48
    assert v(expire_in_minutes="soon").expire_in_minutes == 16


def test_message_text_is_capped():
    plan = v(subject="s" * 500, body="b" * 5000)
    assert len(plan.subject) == 120 and len(plan.body) == 1200


# ── the two rules that make a shared plan safe ───────────────────────────

def test_a_plan_carries_no_rupee_figure_at_all():
    """It carries an `offer_stage`; the payable amount is re-derived per case.
    A plan holding a number would apply one case's discount to another case's
    amount — the exact error `offers.py` exists to prevent, multiplied by the
    size of the batch."""
    fields = set(BatchPlan.__dataclass_fields__)
    assert not {f for f in fields if "amount" in f or "rupee" in f
                or "paise" in f or f in ("price", "discount")}


def test_the_plan_is_frozen():
    """One plan is read by every worker in the run. A mutable plan is one
    thread's edit becoming another customer's offer."""
    plan = v()
    with pytest.raises(Exception):
        plan.batch_key = "something_else"
    with pytest.raises(Exception):
        Step("skip").action = "link_and_notify"


# ── the budget ───────────────────────────────────────────────────────────

def test_a_request_may_tighten_the_budget_but_never_loosen_it():
    ceiling = BatchBudget()
    asked = BatchBudget.from_request(
        {"max_links": 2, "max_cases": 10_000, "max_discount_paise": 999_999})
    assert asked.max_links == 2
    assert asked.max_cases == ceiling.max_cases
    assert asked.max_discount_paise == ceiling.max_discount_paise


def test_a_nonsense_budget_falls_back_to_the_ceiling():
    assert BatchBudget.from_request({"max_links": "lots"}).max_links == \
        BatchBudget().max_links
    assert BatchBudget.from_request(None).max_cases == BatchBudget().max_cases
    assert BatchBudget.from_request({"max_links": -5}).max_links == 0


def test_the_default_budget_gives_no_money_away():
    """A batch is the wrong place to discover a discount policy: nobody is
    watching the first twenty-five of them."""
    assert BatchBudget().max_discount_paise == 0
