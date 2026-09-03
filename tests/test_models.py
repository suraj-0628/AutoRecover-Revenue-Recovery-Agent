"""Tests for data models — verify enum values, field defaults, and constraints.

These tests verify the data model is correct. If you change an enum value
or add a new field without updating these tests, the tests will fail.
"""
from recovery_agent.models import (
    ActionType,
    Case,
    CaseStatus,
    Diagnosis,
    FailureType,
    HARD_DECLINES,
    PaymentEvent,
    RecoveryTier,
)


class TestCaseDefaults:
    def test_new_case_has_open_status(self):
        case = Case(payment=PaymentEvent(payment_id="pay_1", customer_id="c1", amount=1000))
        assert case.status == CaseStatus.OPEN

    def test_new_case_has_zero_attempts(self):
        case = Case(payment=PaymentEvent(payment_id="pay_1", customer_id="c1", amount=1000))
        assert case.attempt_count == 0

    def test_new_case_has_no_diagnosis(self):
        case = Case(payment=PaymentEvent(payment_id="pay_1", customer_id="c1", amount=1000))
        assert case.diagnosis is None

    def test_new_case_starts_in_silent_tier(self):
        case = Case(payment=PaymentEvent(payment_id="pay_1", customer_id="c1", amount=1000))
        assert case.recovery_tier == RecoveryTier.SILENT

    def test_new_case_max_attempts_is_5(self):
        case = Case(payment=PaymentEvent(payment_id="pay_1", customer_id="c1", amount=1000))
        assert case.max_attempts == 5

    def test_new_case_max_silent_attempts_is_3(self):
        case = Case(payment=PaymentEvent(payment_id="pay_1", customer_id="c1", amount=1000))
        assert case.max_silent_attempts == 3


class TestPaymentEvent:
    def test_payment_event_has_required_fields(self):
        p = PaymentEvent(payment_id="pay_1", customer_id="c1", amount=5000)
        assert p.payment_id == "pay_1"
        assert p.customer_id == "c1"
        assert p.amount == 5000
        assert p.currency == "INR"

    def test_payment_event_failure_code_defaults_to_empty(self):
        p = PaymentEvent(payment_id="pay_1", customer_id="c1", amount=5000)
        assert p.failure_code == ""

    def test_payment_event_metadata_defaults_to_empty_dict(self):
        p = PaymentEvent(payment_id="pay_1", customer_id="c1", amount=5000)
        assert p.metadata == {}


class TestDiagnosis:
    def test_diagnosis_has_root_cause(self):
        d = Diagnosis(root_cause=FailureType.NETWORK_TIMEOUT, confidence=0.9)
        assert d.root_cause == FailureType.NETWORK_TIMEOUT

    def test_diagnosis_confidence_in_range(self):
        d = Diagnosis(root_cause=FailureType.NETWORK_TIMEOUT, confidence=0.85)
        assert 0.0 <= d.confidence <= 1.0

    def test_diagnosis_confidence_boundary_zero(self):
        d = Diagnosis(root_cause=FailureType.UNKNOWN, confidence=0.0)
        assert d.confidence == 0.0

    def test_diagnosis_confidence_boundary_one(self):
        d = Diagnosis(root_cause=FailureType.UNKNOWN, confidence=1.0)
        assert d.confidence == 1.0


class TestFailureType:
    def test_all_failure_types_exist(self):
        expected = {
            "CARD_EXPIRED", "INSUFFICIENT_FUNDS", "BANK_DECLINED",
            "NETWORK_TIMEOUT", "RISK_BLOCK", "MANDATE_REVOKED",
            "USER_DROPOFF", "UNKNOWN",
        }
        actual = {ft.name for ft in FailureType}
        assert expected == actual


class TestActionType:
    def test_all_action_types_exist(self):
        expected = {
            "RETRY_PAYMENT", "SEND_NOTIFICATION", "ESCALATE_TO_HUMAN",
            "UPDATE_PAYMENT_METHOD", "WAIT_AND_RETRY", "ABANDON", "VOICE_CALL",
        }
        actual = {at.name for at in ActionType}
        assert expected == actual


class TestRecoveryTier:
    def test_silent_tier_exists(self):
        assert RecoveryTier.SILENT.value == "silent"

    def test_active_tier_exists(self):
        assert RecoveryTier.ACTIVE.value == "active"


class TestHardDeclines:
    def test_hard_declines_contains_8_codes(self):
        assert len(HARD_DECLINES) == 8

    def test_hard_declines_contains_lost_card(self):
        assert "41" in HARD_DECLINES

    def test_hard_declines_contains_stolen_card(self):
        assert "43" in HARD_DECLINES

    def test_hard_declines_contains_expired_card(self):
        assert "54" in HARD_DECLINES

    def test_hard_declines_contains_invalid_card(self):
        assert "14" in HARD_DECLINES

    def test_hard_declines_contains_pick_up_card(self):
        assert "04" in HARD_DECLINES

    def test_hard_declines_contains_closed_account(self):
        assert "46" in HARD_DECLINES

    def test_hard_declines_contains_transaction_not_permitted(self):
        assert "57" in HARD_DECLINES

    def test_hard_declines_contains_cannot_complete(self):
        assert "93" in HARD_DECLINES
