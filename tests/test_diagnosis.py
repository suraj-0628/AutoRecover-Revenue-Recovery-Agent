"""Tests for diagnosis engine — verify failure code mapping.

These tests verify the FAILURE_CODE_MAP correctly maps Razorpay error codes
to FailureType enums. If you add a new error code or change the mapping,
these tests will catch it.
"""
from recovery_agent.agent.diagnosis import FAILURE_CODE_MAP, _build_diagnosis_prompt
from recovery_agent.models import Case, FailureType, PaymentEvent


class TestFailureCodeMap:
    def test_card_expired_maps_correctly(self):
        assert FAILURE_CODE_MAP["card_expired"] == FailureType.CARD_EXPIRED

    def test_insufficient_funds_maps_correctly(self):
        assert FAILURE_CODE_MAP["insufficient_funds"] == FailureType.INSUFFICIENT_FUNDS

    def test_do_not_honor_maps_to_bank_declined(self):
        assert FAILURE_CODE_MAP["do_not_honor"] == FailureType.BANK_DECLINED

    def test_generic_decline_maps_to_bank_declined(self):
        assert FAILURE_CODE_MAP["generic_decline"] == FailureType.BANK_DECLINED

    def test_network_error_maps_to_network_timeout(self):
        assert FAILURE_CODE_MAP["network_error"] == FailureType.NETWORK_TIMEOUT

    def test_network_timeout_maps_correctly(self):
        assert FAILURE_CODE_MAP["network_timeout"] == FailureType.NETWORK_TIMEOUT

    def test_gateway_timeout_maps_to_network_timeout(self):
        assert FAILURE_CODE_MAP["gateway_timeout"] == FailureType.NETWORK_TIMEOUT

    def test_timeout_maps_to_network_timeout(self):
        assert FAILURE_CODE_MAP["timeout"] == FailureType.NETWORK_TIMEOUT

    def test_risk_check_failed_maps_to_risk_block(self):
        assert FAILURE_CODE_MAP["risk_check_failed"] == FailureType.RISK_BLOCK

    def test_fraud_suspected_maps_to_risk_block(self):
        assert FAILURE_CODE_MAP["fraud_suspected"] == FailureType.RISK_BLOCK

    def test_mandate_inactive_maps_to_mandate_revoked(self):
        assert FAILURE_CODE_MAP["mandate_inactive"] == FailureType.MANDATE_REVOKED

    def test_mandate_revoked_maps_correctly(self):
        assert FAILURE_CODE_MAP["mandate_revoked"] == FailureType.MANDATE_REVOKED

    def test_customer_cancelled_maps_to_user_dropoff(self):
        assert FAILURE_CODE_MAP["customer_cancelled"] == FailureType.USER_DROPOFF

    def test_abandonment_maps_to_user_dropoff(self):
        assert FAILURE_CODE_MAP["abandonment"] == FailureType.USER_DROPOFF

    def test_all_network_codes_map_to_network_timeout(self):
        network_codes = ["network_error", "network_timeout", "gateway_timeout", "timeout"]
        for code in network_codes:
            assert FAILURE_CODE_MAP[code] == FailureType.NETWORK_TIMEOUT

    def test_all_risk_codes_map_to_risk_block(self):
        risk_codes = ["risk_check_failed", "fraud_suspected"]
        for code in risk_codes:
            assert FAILURE_CODE_MAP[code] == FailureType.RISK_BLOCK


class TestBuildDiagnosisPrompt:
    def test_prompt_contains_payment_id(self):
        case = Case(payment=PaymentEvent(
            payment_id="pay_xyz789", customer_id="c1", amount=1000,
        ))
        prompt = _build_diagnosis_prompt(case)
        assert "pay_xyz789" in prompt

    def test_prompt_contains_amount(self):
        case = Case(payment=PaymentEvent(
            payment_id="pay_1", customer_id="c1", amount=75000,
        ))
        prompt = _build_diagnosis_prompt(case)
        assert "75,000" in prompt

    def test_prompt_contains_failure_code(self):
        case = Case(payment=PaymentEvent(
            payment_id="pay_1", customer_id="c1", amount=1000,
            failure_code="card_expired",
        ))
        prompt = _build_diagnosis_prompt(case)
        assert "card_expired" in prompt

    def test_prompt_contains_failure_reason(self):
        case = Case(payment=PaymentEvent(
            payment_id="pay_1", customer_id="c1", amount=1000,
            failure_reason="Card has expired",
        ))
        prompt = _build_diagnosis_prompt(case)
        assert "Card has expired" in prompt

    def test_prompt_with_attempt_history(self):
        from recovery_agent.models import Attempt, ActionType
        case = Case(payment=PaymentEvent(
            payment_id="pay_1", customer_id="c1", amount=1000,
        ))
        case.attempts = [Attempt(action_type=ActionType.RETRY_PAYMENT, result="failed")]
        prompt = _build_diagnosis_prompt(case)
        assert "Previous attempt history" in prompt
        assert "retry_payment" in prompt

    def test_prompt_without_attempt_history(self):
        case = Case(payment=PaymentEvent(
            payment_id="pay_1", customer_id="c1", amount=1000,
        ))
        prompt = _build_diagnosis_prompt(case)
        assert "Previous attempt history" not in prompt
