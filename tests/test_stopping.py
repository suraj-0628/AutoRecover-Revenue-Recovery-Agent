"""Tests for stopping rules — verify tier transitions and stopping conditions.

These tests verify the core business logic: when should the agent stop,
when should it transition from Silent to Active tier, and when should it
escalate to a human.

If any of these tests break, the agent's safety boundaries are broken.
"""
from recovery_agent.agent.stopping import (
    check_stopping_rules,
    transition_to_active_tier,
)
from recovery_agent.models import (
    ActionType,
    Attempt,
    Case,
    CaseStatus,
    PaymentEvent,
    RecoveryTier,
)


def make_case(
    recovered: bool = False,
    failure_code: str = "51",
    tier: RecoveryTier = RecoveryTier.SILENT,
    silent_attempts: int = 0,
    max_silent_attempts: int = 3,
    attempt_count: int = 0,
    max_attempts: int = 5,
    attempts: list | None = None,
    hard_decline: bool = False,
) -> Case:
    case = Case(
        payment=PaymentEvent(
            payment_id="pay_test",
            customer_id="cust_test",
            amount=10000,
            failure_code=failure_code,
            metadata={"hard_decline_blocked": hard_decline},
        ),
        recovery_tier=tier,
        silent_attempts=silent_attempts,
        max_silent_attempts=max_silent_attempts,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        recovered=recovered,
    )
    if attempts:
        case.attempts = attempts
    return case


class TestRecoverySucceeded:
    def test_recovered_case_stops(self):
        case = make_case(recovered=True)
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is True
        assert reason == "Recovery succeeded"

    def test_non_recovered_case_does_not_stop_for_recovery(self):
        case = make_case(recovered=False)
        should_stop, reason = check_stopping_rules(case)
        assert reason != "Recovery succeeded"


class TestHardDecline:
    def test_hard_decline_stops_immediately(self):
        case = make_case(hard_decline=True, failure_code="41")
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is True
        assert "Hard decline" in reason
        assert "41" in reason

    def test_non_hard_decline_does_not_stop(self):
        case = make_case(hard_decline=False, failure_code="51")
        should_stop, reason = check_stopping_rules(case)
        assert "Hard decline" not in reason


class TestEscalation:
    def test_escalation_stops(self):
        case = make_case(attempts=[
            Attempt(action_type=ActionType.ESCALATE_TO_HUMAN, result="success"),
        ])
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is True
        assert "Escalated" in reason

    def test_abandon_stops(self):
        case = make_case(attempts=[
            Attempt(action_type=ActionType.ABANDON, result="success"),
        ])
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is True
        assert "abandoned" in reason.lower()


class TestSilentTierTransition:
    def test_silent_retry_failed_transitions_to_active(self):
        case = make_case(
            tier=RecoveryTier.SILENT,
            attempts=[Attempt(action_type=ActionType.RETRY_PAYMENT, result="failed")],
        )
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is False
        assert reason == "SILENT_RETRY_FAILED"

    def test_silent_wait_and_retry_failed_transitions(self):
        case = make_case(
            tier=RecoveryTier.SILENT,
            attempts=[Attempt(action_type=ActionType.WAIT_AND_RETRY, result="failed")],
        )
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is False
        assert reason == "SILENT_RETRY_FAILED"

    def test_silent_retry_success_does_not_transition(self):
        case = make_case(
            tier=RecoveryTier.SILENT,
            attempts=[Attempt(action_type=ActionType.RETRY_PAYMENT, result="success")],
        )
        should_stop, reason = check_stopping_rules(case)
        assert reason != "SILENT_RETRY_FAILED"

    def test_silent_tier_exhausted_transitions(self):
        case = make_case(
            tier=RecoveryTier.SILENT,
            silent_attempts=3,
            max_silent_attempts=3,
        )
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is False
        assert reason == "SILENT_TIER_EXHAUSTED"

    def test_silent_tier_not_exhausted(self):
        case = make_case(
            tier=RecoveryTier.SILENT,
            silent_attempts=1,
            max_silent_attempts=3,
        )
        should_stop, reason = check_stopping_rules(case)
        assert reason != "SILENT_TIER_EXHAUSTED"


class TestMaxAttempts:
    def test_max_attempts_stops(self):
        case = make_case(attempt_count=5, max_attempts=5)
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is True
        assert "Max attempts" in reason

    def test_below_max_attempts_does_not_stop(self):
        case = make_case(attempt_count=3, max_attempts=5)
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is False


class TestTransitionToActiveTier:
    def test_transition_sets_active_tier(self):
        case = make_case(tier=RecoveryTier.SILENT)
        case = transition_to_active_tier(case, reason="SILENT_RETRY_FAILED")
        assert case.recovery_tier == RecoveryTier.ACTIVE

    def test_transition_records_metadata(self):
        case = make_case(tier=RecoveryTier.SILENT)
        case = transition_to_active_tier(case, reason="SILENT_RETRY_FAILED")
        assert case.payment.metadata["tier_transition"] == "silent_to_active"

    def test_transition_retry_failed_sets_reason(self):
        case = make_case(tier=RecoveryTier.SILENT, attempt_count=2)
        case = transition_to_active_tier(case, reason="SILENT_RETRY_FAILED")
        assert "FAILED" in case.payment.metadata["tier_transition_reason"]

    def test_transition_exhausted_sets_reason(self):
        case = make_case(tier=RecoveryTier.SILENT, silent_attempts=3)
        case = transition_to_active_tier(case, reason="SILENT_TIER_EXHAUSTED")
        assert "exhausted" in case.payment.metadata["tier_transition_reason"].lower()
