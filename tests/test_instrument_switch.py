"""Test for Silent Tier Trap fix — verifies the exact payload from the issue."""
from __future__ import annotations

from recovery_agent.agent.decision import _assign_tier, _needs_instrument_switch, decide_intervention
from recovery_agent.agent.stopping import check_stopping_rules, run_stopping_check
from recovery_agent.models import (
    ActionType,
    Attempt,
    Case,
    CaseStatus,
    Diagnosis,
    FailureType,
    PaymentEvent,
    RecoveryTier,
)


def make_case(
    failure_reason: str = "",
    error_description: str = "",
    failure_code: str = "",
    failure_type: FailureType = FailureType.BANK_DECLINED,
) -> Case:
    event = PaymentEvent(
        event_type="payment_failed",
        payment_id="pay_inst_switch_001",
        customer_id="cust_inst_switch",
        amount=2999.0,
        currency="INR",
        status="failed",
        failure_reason=failure_reason,
        failure_code=failure_code,
        metadata={"error_description": error_description},
    )
    case = Case(payment=event, max_attempts=5)
    case.diagnosis = Diagnosis(
        root_cause=failure_type,
        confidence=0.85,
        reasoning=f"Diagnosed as {failure_type.value}",
    )
    return case


class TestInstrumentSwitchDetection:
    """Test that instrument-switching text forces ACTIVE tier."""

    def test_exact_payload_forces_active_tier(self):
        """The exact payload from the issue."""
        case = make_case(
            failure_reason="Your payment could not be completed due to a temporary technical issue. To complete the payment, use another payment instrument.",
            failure_code="bank_declined",
        )
        assert _needs_instrument_switch(case) is True
        tier = _assign_tier(case)
        assert tier == RecoveryTier.ACTIVE

    def test_use_another_payment_method(self):
        case = make_case(failure_reason="Transaction failed. Please use another payment method.")
        assert _needs_instrument_switch(case) is True
        assert _assign_tier(case) == RecoveryTier.ACTIVE

    def test_try_another_method(self):
        case = make_case(failure_reason="Card declined. Try another method.")
        assert _needs_instrument_switch(case) is True
        assert _assign_tier(case) == RecoveryTier.ACTIVE

    def test_expired_in_reason(self):
        case = make_case(failure_reason="Card has expired")
        assert _needs_instrument_switch(case) is True
        assert _assign_tier(case) == RecoveryTier.ACTIVE

    def test_invalid_card_in_reason(self):
        case = make_case(failure_reason="Invalid card number")
        assert _needs_instrument_switch(case) is True
        assert _assign_tier(case) == RecoveryTier.ACTIVE

    def test_mandate_revoked_forces_active(self):
        case = make_case(
            failure_reason="Mandate was revoked by customer",
            failure_type=FailureType.MANDATE_REVOKED,
        )
        assert _needs_instrument_switch(case) is True
        assert _assign_tier(case) == RecoveryTier.ACTIVE

    def test_error_description_triggers_active(self):
        """Keyword in error_description (not just failure_reason)."""
        case = make_case(
            failure_reason="Payment failed",
            error_description="To complete the payment, use another payment instrument.",
        )
        assert _needs_instrument_switch(case) is True
        assert _assign_tier(case) == RecoveryTier.ACTIVE

    def test_no_switch_text_stays_silent(self):
        """Normal bank decline without instrument-switch text stays SILENT."""
        case = make_case(
            failure_reason="Bank declined the transaction temporarily",
            failure_type=FailureType.BANK_DECLINED,
        )
        assert _needs_instrument_switch(case) is False
        assert _assign_tier(case) == RecoveryTier.SILENT

    def test_network_timeout_stays_silent(self):
        case = make_case(
            failure_reason="Gateway timeout",
            failure_type=FailureType.NETWORK_TIMEOUT,
        )
        assert _needs_instrument_switch(case) is False
        assert _assign_tier(case) == RecoveryTier.SILENT


class TestTierEscalationOnFailedRetry:
    """Test that a failed silent retry auto-escalates to ACTIVE tier."""

    def test_failed_retry_escalates_to_active(self):
        case = make_case(
            failure_reason="Bank declined the transaction temporarily",
            failure_type=FailureType.BANK_DECLINED,
        )
        case.recovery_tier = RecoveryTier.SILENT
        case.attempt_count = 1

        # Simulate a failed RETRY_PAYMENT attempt
        case.attempts.append(Attempt(
            action_type=ActionType.RETRY_PAYMENT,
            result="failed",
            tier=RecoveryTier.SILENT,
        ))

        should_stop, reason = check_stopping_rules(case)
        assert should_stop is False
        assert reason == "SILENT_RETRY_FAILED"

    def test_failed_wait_and_retry_escalates(self):
        case = make_case(
            failure_reason="Bank declined",
            failure_type=FailureType.BANK_DECLINED,
        )
        case.recovery_tier = RecoveryTier.SILENT
        case.attempt_count = 1

        case.attempts.append(Attempt(
            action_type=ActionType.WAIT_AND_RETRY,
            result="failed",
            tier=RecoveryTier.SILENT,
        ))

        should_stop, reason = check_stopping_rules(case)
        assert should_stop is False
        assert reason == "SILENT_RETRY_FAILED"

    def test_successful_retry_no_escalation(self):
        case = make_case(
            failure_reason="Bank declined",
            failure_type=FailureType.BANK_DECLINED,
        )
        case.recovery_tier = RecoveryTier.SILENT

        case.attempts.append(Attempt(
            action_type=ActionType.RETRY_PAYMENT,
            result="success",
            tier=RecoveryTier.SILENT,
        ))
        case.recovered = True

        should_stop, reason = check_stopping_rules(case)
        assert should_stop is True
        assert reason == "Recovery succeeded"

    def test_run_stopping_check_transitions_tier(self):
        case = make_case(
            failure_reason="Bank declined",
            failure_type=FailureType.BANK_DECLINED,
        )
        case.recovery_tier = RecoveryTier.SILENT
        case.attempt_count = 1

        case.attempts.append(Attempt(
            action_type=ActionType.RETRY_PAYMENT,
            result="failed",
            tier=RecoveryTier.SILENT,
        ))

        case = run_stopping_check(case)
        assert case.recovery_tier == RecoveryTier.ACTIVE
        assert case.payment.metadata["tier_transition"] == "silent_to_active"
        assert "FAILED" in case.payment.metadata["tier_transition_reason"]


class TestDecisionIntegration:
    """Test that the full decision pipeline handles instrument-switching correctly."""

    def test_instrument_switch_payload_gets_active_action(self):
        """Exact payload from issue should get UPDATE_PAYMENT_METHOD, not RETRY_PAYMENT."""
        case = make_case(
            failure_reason="Your payment could not be completed due to a temporary technical issue. To complete the payment, use another payment instrument.",
            failure_code="bank_declined",
        )

        action = decide_intervention(case)

        # Should be an ACTIVE-tier action, not a silent retry
        assert action in (ActionType.UPDATE_PAYMENT_METHOD, ActionType.SEND_NOTIFICATION,
                          ActionType.ESCALATE_TO_HUMAN)
        assert action != ActionType.RETRY_PAYMENT
        assert action != ActionType.WAIT_AND_RETRY
        assert case.recovery_tier == RecoveryTier.ACTIVE
