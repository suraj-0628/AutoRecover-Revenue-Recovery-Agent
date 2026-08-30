"""Unit tests for the recovery agent.

Covers:
- Diagnosis engine (rule-based, failure type classification)
- Decision matrix (cause × attempt → action)
- Stopping rules (max attempts, recovery, escalation, abandon)
- Execution (observable outcomes)
- Data models
"""
from __future__ import annotations

from recovery_agent.models import (
    ActionType,
    Case,
    CaseStatus,
    Diagnosis,
    FailureType,
    PaymentEvent,
)
from recovery_agent.agent.diagnosis import diagnose_payment_failure
from recovery_agent.agent.decision import run_decision
from recovery_agent.agent.stopping import check_stopping_rules, run_stopping_check
from recovery_agent.agent.execution import execute_action, observe_outcome
from recovery_agent.agent.test_generator import generate_payment_event, FAILURE_SCENARIOS


# ─── Helpers ──────────────────────────────────────────────────
def make_case(failure_reason: str = "Card has expired", failure_code: str = "card_expired", amount: float = 5000.0) -> Case:
    """Create a test case with given failure details."""
    event = PaymentEvent(
        event_type="payment_failed",
        payment_id="pay_test_001",
        customer_id="cust_test",
        amount=amount,
        currency="INR",
        status="failed",
        failure_reason=failure_reason,
        failure_code=failure_code,
    )
    return Case(payment=event, max_attempts=3)


# ─── Diagnosis Tests ──────────────────────────────────────────
class TestDiagnosis:
    """Test the rule-based diagnosis engine."""

    def test_card_expired_by_keyword(self):
        case = make_case(failure_reason="Card has expired", failure_code="card_expired")
        result = diagnose_payment_failure(case)
        assert result.root_cause == FailureType.CARD_EXPIRED
        assert result.confidence >= 0.7

    def test_insufficient_funds_by_keyword(self):
        case = make_case(failure_reason="Insufficient funds in account", failure_code="insufficient_funds")
        result = diagnose_payment_failure(case)
        assert result.root_cause == FailureType.INSUFFICIENT_FUNDS
        assert result.confidence >= 0.7

    def test_bank_declined_by_keyword(self):
        case = make_case(failure_reason="Bank declined the transaction", failure_code="bank_declined")
        result = diagnose_payment_failure(case)
        assert result.root_cause == FailureType.BANK_DECLINED
        assert result.confidence >= 0.7

    def test_network_timeout_by_keyword(self):
        case = make_case(failure_reason="Network timeout during processing", failure_code="network_timeout")
        result = diagnose_payment_failure(case)
        assert result.root_cause == FailureType.NETWORK_TIMEOUT
        assert result.confidence >= 0.7

    def test_risk_block_by_keyword(self):
        case = make_case(failure_reason="Risk check failed - suspicious activity", failure_code="risk_check_failed")
        result = diagnose_payment_failure(case)
        assert result.root_cause == FailureType.RISK_BLOCK
        assert result.confidence >= 0.7

    def test_mandate_revoked_by_keyword(self):
        case = make_case(failure_reason="Mandate has been revoked by customer", failure_code="mandate_inactive")
        result = diagnose_payment_failure(case)
        assert result.root_cause == FailureType.MANDATE_REVOKED
        assert result.confidence >= 0.7

    def test_unknown_failure_returns_unknown(self):
        case = make_case(failure_reason="Something weird happened", failure_code="unknown_error")
        result = diagnose_payment_failure(case)
        assert result.root_cause == FailureType.UNKNOWN

    def test_diagnosis_has_reasoning(self):
        case = make_case(failure_reason="Card expired", failure_code="card_expired")
        result = diagnose_payment_failure(case)
        assert result.reasoning != ""
        assert result.confidence > 0

    def test_diagnosis_returns_diagnosis_object(self):
        case = make_case()
        result = diagnose_payment_failure(case)
        assert isinstance(result, Diagnosis)
        assert isinstance(result.root_cause, FailureType)


# ─── Decision Tests ───────────────────────────────────────────
class TestDecision:
    """Test the decision matrix: cause × attempt → action."""

    def test_card_expired_attempt_0_uses_notification(self):
        case = make_case(failure_reason="Card expired", failure_code="card_expired")
        case.diagnosis = Diagnosis(root_cause=FailureType.CARD_EXPIRED, confidence=0.9, reasoning="test")
        case.attempt_count = 0
        result = run_decision(case)
        action = result.payment.metadata.get("decided_action")
        assert action in ("send_notification", "update_payment_method")

    def test_card_expired_attempt_1_uses_update(self):
        case = make_case(failure_reason="Card expired", failure_code="card_expired")
        case.diagnosis = Diagnosis(root_cause=FailureType.CARD_EXPIRED, confidence=0.9, reasoning="test")
        case.attempt_count = 1
        result = run_decision(case)
        action = result.payment.metadata.get("decided_action")
        assert action in ("send_notification", "update_payment_method", "escalate_to_human")

    def test_network_timeout_uses_retry(self):
        case = make_case(failure_reason="Network timeout", failure_code="network_timeout")
        case.diagnosis = Diagnosis(root_cause=FailureType.NETWORK_TIMEOUT, confidence=0.9, reasoning="test")
        case.attempt_count = 0
        result = run_decision(case)
        action = result.payment.metadata.get("decided_action")
        assert action in ("retry_payment", "wait_and_retry")

    def test_insufficient_funds_uses_wait(self):
        case = make_case(failure_reason="Insufficient funds", failure_code="insufficient_funds")
        case.diagnosis = Diagnosis(root_cause=FailureType.INSUFFICIENT_FUNDS, confidence=0.9, reasoning="test")
        case.attempt_count = 0
        result = run_decision(case)
        action = result.payment.metadata.get("decided_action")
        assert action in ("wait_and_retry", "send_notification")

    def test_risk_block_uses_escalation(self):
        case = make_case(failure_reason="Risk check failed", failure_code="risk_check_failed")
        case.diagnosis = Diagnosis(root_cause=FailureType.RISK_BLOCK, confidence=0.9, reasoning="test")
        case.attempt_count = 0
        result = run_decision(case)
        action = result.payment.metadata.get("decided_action")
        assert action in ("escalate_to_human", "send_notification")

    def test_mandate_revoked_uses_escalation(self):
        case = make_case(failure_reason="Mandate revoked", failure_code="mandate_inactive")
        case.diagnosis = Diagnosis(root_cause=FailureType.MANDATE_REVOKED, confidence=0.9, reasoning="test")
        case.attempt_count = 0
        result = run_decision(case)
        action = result.payment.metadata.get("decided_action")
        assert action in ("escalate_to_human", "send_notification", "update_payment_method")

    def test_high_attempt_count_escalates(self):
        case = make_case(failure_reason="Card expired", failure_code="card_expired")
        case.diagnosis = Diagnosis(root_cause=FailureType.CARD_EXPIRED, confidence=0.9, reasoning="test")
        case.attempt_count = 2
        result = run_decision(case)
        action = result.payment.metadata.get("decided_action")
        assert action == "escalate_to_human"

    def test_decision_sets_metadata(self):
        case = make_case()
        case.diagnosis = Diagnosis(root_cause=FailureType.NETWORK_TIMEOUT, confidence=0.9, reasoning="test")
        case.attempt_count = 0
        result = run_decision(case)
        assert "decided_action" in result.payment.metadata

    def test_all_failure_types_have_decisions(self):
        for ft in FailureType:
            if ft == FailureType.UNKNOWN:
                continue
            case = make_case()
            case.diagnosis = Diagnosis(root_cause=ft, confidence=0.9, reasoning="test")
            case.attempt_count = 0
            result = run_decision(case)
            assert "decided_action" in result.payment.metadata


# ─── Stopping Rules Tests ─────────────────────────────────────
class TestStoppingRules:
    """Test the stopping rules engine."""

    def test_recovered_stops(self):
        case = make_case()
        case.recovered = True
        case.recovered_amount = 5000.0
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is True
        assert "recovery" in reason.lower() or "recovered" in reason.lower()

    def test_max_attempts_stops(self):
        case = make_case()
        case.attempt_count = 3
        case.max_attempts = 3
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is True
        assert "max" in reason.lower() or "attempt" in reason.lower()

    def test_escalated_stops(self):
        from recovery_agent.models import Attempt
        case = make_case()
        case.attempts.append(Attempt(action_type=ActionType.ESCALATE_TO_HUMAN, result="success"))
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is True
        assert "escalat" in reason.lower()

    def test_not_stopped_when_acting(self):
        case = make_case()
        case.status = CaseStatus.ACTING
        case.attempt_count = 1
        case.max_attempts = 3
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is False

    def test_not_stopped_when_diagnosing(self):
        case = make_case()
        case.status = CaseStatus.DIAGNOSING
        case.attempt_count = 0
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is False

    def test_run_stopping_check_updates_status(self):
        case = make_case()
        case.recovered = True
        result = run_stopping_check(case)
        assert result.status in (CaseStatus.RECOVERED, CaseStatus.STOPPED)

    def test_max_attempts_5_allows_more_retries(self):
        case = make_case()
        case.attempt_count = 2
        case.max_attempts = 5
        should_stop, reason = check_stopping_rules(case)
        assert should_stop is False


# ─── Execution Tests ──────────────────────────────────────────
class TestExecution:
    """Test the execution layer — observable outcomes."""

    def test_send_notification_returns_observable(self):
        result = execute_action(ActionType.SEND_NOTIFICATION, "card_expired", 5000.0)
        assert result["action"] == "send_notification"
        assert result["observable"] == "customer_received_message"

    def test_retry_payment_returns_observable(self):
        result = execute_action(ActionType.RETRY_PAYMENT, "network_timeout", 5000.0)
        assert result["action"] == "retry_payment"
        assert result["observable"] == "order_exists"

    def test_update_payment_method_returns_observable(self):
        result = execute_action(ActionType.UPDATE_PAYMENT_METHOD, "card_expired", 5000.0)
        assert result["action"] == "update_payment_method"
        assert result["observable"] == "customer_received_link"

    def test_wait_and_retry_returns_observable(self):
        result = execute_action(ActionType.WAIT_AND_RETRY, "insufficient_funds", 5000.0)
        assert result["action"] == "wait_and_retry"
        assert result["observable"] == "retry_pending"

    def test_escalate_to_human_returns_observable(self):
        result = execute_action(ActionType.ESCALATE_TO_HUMAN, "risk_block", 5000.0)
        assert result["action"] == "escalate_to_human"
        assert result["observable"] == "human_notified"

    def test_abandon_returns_observable(self):
        result = execute_action(ActionType.ABANDON, "mandate_revoked", 5000.0)
        assert result["action"] == "abandon"
        assert result["observable"] == "none"

    def test_observe_customer_responded(self):
        execution = {"observable": "customer_received_message"}
        result = observe_outcome(ActionType.SEND_NOTIFICATION, execution, customer_responded=True)
        assert result["success"] is True
        assert result["should_continue"] is False

    def test_observe_customer_not_responded(self):
        execution = {"observable": "customer_received_message"}
        result = observe_outcome(ActionType.SEND_NOTIFICATION, execution, customer_responded=False)
        assert result["success"] is False
        assert result["should_continue"] is True

    def test_observe_order_completed(self):
        execution = {"observable": "order_exists"}
        result = observe_outcome(ActionType.RETRY_PAYMENT, execution, customer_responded=True)
        assert result["success"] is True

    def test_observe_order_not_completed(self):
        execution = {"observable": "order_exists"}
        result = observe_outcome(ActionType.RETRY_PAYMENT, execution, customer_responded=False)
        assert result["success"] is False
        assert result["should_continue"] is True

    def test_observe_wait_always_continues(self):
        execution = {"observable": "retry_pending"}
        result = observe_outcome(ActionType.WAIT_AND_RETRY, execution, customer_responded=False)
        assert result["success"] is False
        assert result["should_continue"] is True

    def test_observe_human_resolved(self):
        execution = {"observable": "human_notified"}
        result = observe_outcome(ActionType.ESCALATE_TO_HUMAN, execution, customer_responded=True)
        assert result["success"] is True

    def test_observe_human_not_resolved(self):
        execution = {"observable": "human_notified"}
        result = observe_outcome(ActionType.ESCALATE_TO_HUMAN, execution, customer_responded=False)
        assert result["success"] is False
        assert result["should_continue"] is True


# ─── Data Model Tests ─────────────────────────────────────────
class TestDataModels:
    """Test Pydantic data models."""

    def test_payment_event_creation(self):
        event = PaymentEvent(
            event_type="payment_failed",
            payment_id="pay_123",
            customer_id="cust_123",
            amount=1000.0,
        )
        assert event.payment_id == "pay_123"
        assert event.amount == 1000.0

    def test_case_creation(self):
        case = make_case()
        assert case.id != ""
        assert case.payment.payment_id == "pay_test_001"
        assert case.status == CaseStatus.OPEN
        assert case.attempt_count == 0

    def test_diagnosis_creation(self):
        d = Diagnosis(root_cause=FailureType.CARD_EXPIRED, confidence=0.9, reasoning="test")
        assert d.root_cause == FailureType.CARD_EXPIRED
        assert d.confidence == 0.9

    def test_attempt_creation(self):
        from recovery_agent.models import Attempt
        a = Attempt(action_type=ActionType.RETRY_PAYMENT, result="success")
        assert a.action_type == ActionType.RETRY_PAYMENT
        assert a.result == "success"

    def test_case_max_attempts_default(self):
        case = make_case()
        assert case.max_attempts == 3


# ─── Test Generator Tests ─────────────────────────────────────
class TestGenerator:
    """Test the synthetic test case generator."""

    def test_generate_payment_event(self):
        event = generate_payment_event(scenario=FAILURE_SCENARIOS[0])
        assert event.payment_id.startswith("pay_")
        assert event.amount > 0

    def test_all_scenarios_produce_events(self):
        for scenario in FAILURE_SCENARIOS:
            event = generate_payment_event(scenario=scenario)
            assert event.failure_reason != ""

    def test_scenarios_cover_all_types(self):
        types_covered = set()
        for scenario in FAILURE_SCENARIOS:
            types_covered.add(scenario.get("failure_type", "unknown"))
        assert len(types_covered) >= 5
