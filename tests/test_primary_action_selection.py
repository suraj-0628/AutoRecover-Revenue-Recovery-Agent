"""A refusal is not an action.

`escalate_to_human` returns `{"status": "blocked"}` when the ladder is not
exhausted — the gate doing its job. The action selector read that raw dict as
truthy and recorded the run as escalated: no ticket existed, the safety-net
enqueue was suppressed by `agent_already_escalated`, and `_handoff_to_agent`
refuses to reopen an "escalated" case. The case died silently, unticketed —
the one outcome the ladder architecture exists to prevent.

These tests call the real `_select_primary_action`, not a mirror of it.
"""
from recovery_agent.frontend import _select_primary_action


def _made(**tools):
    return {name: {"args": {}, "result": result} for name, result in tools.items()}


# ── the bug: blocked escalation must not count as an escalation ─────────────

def test_a_blocked_escalation_is_not_an_action():
    action, receipt, escalated = _select_primary_action(_made(
        escalate_to_human={"status": "blocked",
                           "reason": "the recovery ladder is not exhausted"},
    ))
    assert action == "none"
    assert receipt == {}
    assert escalated is False       # no ticket exists — the safety net must stay armed


def test_a_blocked_escalation_does_not_outrank_a_delivered_push():
    action, receipt, escalated = _select_primary_action(_made(
        escalate_to_human={"status": "blocked", "reason": "ladder not exhausted"},
        send_page_push={"status": "delivered"},
    ))
    assert action == "page_push"
    assert receipt == {"status": "delivered"}
    assert escalated is False


def test_a_blocked_escalation_does_not_outrank_a_scheduled_retry():
    action, receipt, escalated = _select_primary_action(_made(
        escalate_to_human={"status": "blocked", "reason": "ladder not exhausted"},
        retry_in_hours={"status": "scheduled", "job_id": "job_x", "delay_hours": 24},
    ))
    assert action == "wait_and_retry"
    assert receipt["job_id"] == "job_x"
    assert escalated is False


# ── a real escalation still wins, and still disarms the safety net ──────────

def test_a_real_escalation_is_the_action_and_disarms_the_net():
    action, receipt, escalated = _select_primary_action(_made(
        escalate_to_human={"status": "escalated", "ticket_id": "ESC-1"},
    ))
    assert action == "escalate_to_human"
    assert receipt["ticket_id"] == "ESC-1"
    assert escalated is True


def test_closing_outranks_everything_but_keeps_the_escalation_flag():
    # escalate, then close_case(outcome="escalated") — the prescribed ending.
    action, receipt, escalated = _select_primary_action(_made(
        escalate_to_human={"status": "escalated", "ticket_id": "ESC-2"},
        close_case={"status": "closed", "outcome": "escalated"},
    ))
    assert action == "close_case"
    assert receipt["outcome"] == "escalated"
    assert escalated is True        # a second ticket must not be filed


# ── the pre-existing selection rules still hold ─────────────────────────────

def test_the_link_is_the_receipt_even_when_the_notification_ran_last():
    action, receipt, _ = _select_primary_action(_made(
        generate_recovery_payment_link={"status": "ok", "link_url": "https://rzp.io/x",
                                        "link_id": "plink_1"},
        send_recovery_notification={"status": "ok", "channels": ["email"]},
    ))
    assert action == "send_notification"
    assert receipt["link_url"] == "https://rzp.io/x"


def test_a_push_alone_is_the_silent_first_rung():
    action, _, _ = _select_primary_action(_made(
        send_page_push={"status": "delivered"},
    ))
    assert action == "page_push"


def test_a_bookkeeping_run_is_no_action_not_the_most_severe_one():
    action, receipt, escalated = _select_primary_action(_made(
        manage_memory={"status": "ok"},
    ))
    assert (action, receipt, escalated) == ("none", {}, False)


def test_a_failed_tool_result_is_not_a_receipt():
    action, _, _ = _select_primary_action(_made(
        generate_recovery_payment_link={"status": "error", "message": "quota spent"},
    ))
    assert action == "none"
