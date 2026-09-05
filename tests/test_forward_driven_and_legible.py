"""Three UX/correctness refinements after the recovery flow came back:

1. Tool calls are sequential — one per turn — so each result can shape the
   next call (parallel_tool_calls=False).
2. The checkout asks ONE question with THREE options: the instrument, the
   price, the money.
3. The monologue opens by naming the failure, not "Opened for pay_x".
"""
from __future__ import annotations


# ── 1. sequential tool calls ────────────────────────────────────────────

def test_the_model_is_bound_for_one_tool_at_a_time():
    from recovery_agent.agent.graph import _build_model
    bound = _build_model()
    # bind_tools returns a RunnableBinding carrying the request kwargs.
    kwargs = getattr(bound, "kwargs", {})
    assert kwargs.get("parallel_tool_calls") is False, (
        "the agent must call tools sequentially so each result shapes the next")


def test_the_turn_cap_has_headroom_for_sequential_work():
    from recovery_agent.agent.graph import MAX_TURNS_PER_RUN
    assert MAX_TURNS_PER_RUN >= 12, (
        "one-at-a-time tool calls take more turns; the cap was raised to match")


# ── 2. three-option checkout question ───────────────────────────────────

def test_the_checkout_asks_exactly_three():
    from recovery_agent import drop_reasons
    codes = [c["code"] for c in drop_reasons.choices()]
    assert codes == ["payment_kept_failing", "better_price", "no_balance"], (
        f"the checkout question should be instrument/price/money, got {codes}")


def test_the_dropped_options_still_exist_for_other_callers():
    """Only the CHECKOUT question shrank. will_pay_later / other are still
    valid drop-reason codes when set by any other path."""
    from recovery_agent import drop_reasons
    assert drop_reasons.get("will_pay_later") is not None
    assert drop_reasons.get("other") is not None


# ── 3. the monologue opens with the failure ─────────────────────────────

def test_a_bank_decline_reads_as_a_bank_decline():
    from recovery_agent.frontend import _failure_headline
    head = _failure_headline("BAD_REQUEST_ERROR", "declined by the bank", 2499.0)
    assert head.startswith("Bank declined the payment")
    assert "declined by the bank" in head
    assert "BAD_REQUEST_ERROR" in head


def test_insufficient_funds_reads_as_funds():
    from recovery_agent.frontend import _failure_headline
    head = _failure_headline("", "insufficient funds in the account", 999.0)
    assert head.startswith("Insufficient funds")


def test_a_bare_failure_still_gets_a_headline():
    from recovery_agent.frontend import _failure_headline
    head = _failure_headline("", "payment_failed", 100.0)
    assert head  # never empty
    assert "payment_failed" not in head  # the placeholder reason is not echoed


# ── 4. lead with the in-page top-bar when the customer is live ──────────

def test_a_live_customer_is_a_fact_in_the_briefing(tmp_path, monkeypatch):
    import recovery_agent.state_store as state_store
    from recovery_agent import presence
    from recovery_agent.state_store import StateStore
    from recovery_agent.agent.perception import ground_truth
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    StateStore.reset_instances()
    try:
        s = StateStore()
        s.save_payment("p", {"payment_id": "p", "amount": 2499.0,
                             "status": "recovering",
                             "customer": {"email": "a@b.com"},
                             "failure_reason": "declined by the bank"})
        s.flush()
        monkeypatch.setattr(presence, "is_live", lambda pid: True)
        assert ground_truth("p").get("customer_on_page") is True
        monkeypatch.setattr(presence, "is_live", lambda pid: False)
        assert ground_truth("p").get("customer_on_page") is False
    finally:
        StateStore.reset_instances()


def _open(**over):
    facts = {"known": True, "settled": False, "owed": 2499.0,
             "received": 0.0, "outstanding": 2499.0}
    facts.update(over)
    return facts


def test_a_live_customer_gets_steered_to_the_top_bar():
    from recovery_agent.agent.perception import as_briefing
    text = as_briefing(_open(customer_on_page=True))
    assert "ON THE CHECKOUT PAGE RIGHT NOW" in text
    assert "show_page_offer" in text
    assert "send_recovery_notification" in text  # email stays as the backup


def test_a_departed_customer_is_not_steered_to_the_page():
    from recovery_agent.agent.perception import as_briefing
    text = as_briefing(_open(customer_on_page=False))
    assert "ON THE CHECKOUT PAGE RIGHT NOW" not in text
