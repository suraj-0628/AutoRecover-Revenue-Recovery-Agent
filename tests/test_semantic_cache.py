"""Tests for the Fast Path Cache (Semantic Routing).

Verifies that deterministic Razorpay failure codes are intercepted
before the ReAct loop, saving ~5-10s of LLM latency per case.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from recovery_agent.agent.semantic_cache import (
    FAST_PATH_CACHE,
    FastPathResult,
    fast_path_stats,
    is_deterministic,
    lookup_fast_path,
)
from recovery_agent.models import (
    ActionType,
    Case,
    CaseStatus,
    FailureType,
    PaymentEvent,
    RecoveryTier,
)


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests — lookup_fast_path
# ═══════════════════════════════════════════════════════════════════════

class TestLookupFastPath:
    """Test the lookup_fast_path function."""

    def test_card_expired_returns_update_payment_method(self):
        result = lookup_fast_path("card_expired")
        assert result is not None
        assert result.action == ActionType.UPDATE_PAYMENT_METHOD
        assert result.tier == RecoveryTier.ACTIVE
        assert result.diagnosis_root_cause == FailureType.CARD_EXPIRED

    def test_insufficient_funds_returns_wait_and_retry(self):
        result = lookup_fast_path("insufficient_funds")
        assert result is not None
        assert result.action == ActionType.WAIT_AND_RETRY
        assert result.tier == RecoveryTier.SILENT
        assert result.diagnosis_root_cause == FailureType.INSUFFICIENT_FUNDS

    def test_network_timeout_returns_retry_payment(self):
        result = lookup_fast_path("network_timeout")
        assert result is not None
        assert result.action == ActionType.RETRY_PAYMENT
        assert result.tier == RecoveryTier.SILENT
        assert result.diagnosis_root_cause == FailureType.NETWORK_TIMEOUT

    def test_gateway_timeout_returns_retry_payment(self):
        result = lookup_fast_path("gateway_timeout")
        assert result is not None
        assert result.action == ActionType.RETRY_PAYMENT
        assert result.tier == RecoveryTier.SILENT

    def test_bank_declined_returns_retry_payment(self):
        result = lookup_fast_path("bank_declined")
        assert result is not None
        assert result.action == ActionType.RETRY_PAYMENT
        assert result.tier == RecoveryTier.SILENT
        assert result.diagnosis_root_cause == FailureType.BANK_DECLINED

    def test_mandate_revoked_returns_send_notification(self):
        result = lookup_fast_path("mandate_revoked")
        assert result is not None
        assert result.action == ActionType.SEND_NOTIFICATION
        assert result.tier == RecoveryTier.ACTIVE
        assert result.diagnosis_root_cause == FailureType.MANDATE_REVOKED

    def test_abandonment_returns_send_notification(self):
        result = lookup_fast_path("abandonment")
        assert result is not None
        assert result.action == ActionType.SEND_NOTIFICATION
        assert result.tier == RecoveryTier.ACTIVE

    def test_risk_block_returns_escalate(self):
        result = lookup_fast_path("risk_block")
        assert result is not None
        assert result.action == ActionType.ESCALATE_TO_HUMAN
        assert result.tier == RecoveryTier.ACTIVE
        assert result.diagnosis_root_cause == FailureType.RISK_BLOCK

    def test_hard_decline_41_returns_escalate(self):
        result = lookup_fast_path("41")
        assert result is not None
        assert result.action == ActionType.ESCALATE_TO_HUMAN
        assert result.tier == RecoveryTier.ACTIVE

    def test_hard_decline_43_returns_escalate(self):
        result = lookup_fast_path("43")
        assert result is not None
        assert result.action == ActionType.ESCALATE_TO_HUMAN

    def test_hard_decline_54_returns_escalate(self):
        result = lookup_fast_path("54")
        assert result is not None
        assert result.action == ActionType.ESCALATE_TO_HUMAN

    def test_hard_decline_14_returns_escalate(self):
        result = lookup_fast_path("14")
        assert result is not None
        assert result.action == ActionType.ESCALATE_TO_HUMAN

    def test_hard_decline_04_returns_escalate(self):
        result = lookup_fast_path("04")
        assert result is not None
        assert result.action == ActionType.ESCALATE_TO_HUMAN

    def test_hard_decline_46_returns_escalate(self):
        result = lookup_fast_path("46")
        assert result is not None
        assert result.action == ActionType.ESCALATE_TO_HUMAN

    def test_hard_decline_57_returns_escalate(self):
        result = lookup_fast_path("57")
        assert result is not None
        assert result.action == ActionType.ESCALATE_TO_HUMAN

    def test_hard_decline_93_returns_escalate(self):
        result = lookup_fast_path("93")
        assert result is not None
        assert result.action == ActionType.ESCALATE_TO_HUMAN

    def test_unknown_code_returns_none(self):
        assert lookup_fast_path("some_weird_error") is None

    def test_empty_string_returns_none(self):
        assert lookup_fast_path("") is None

    def test_none_returns_none(self):
        assert lookup_fast_path(None) is None

    def test_case_insensitive(self):
        result = lookup_fast_path("CARD_EXPIRED")
        assert result is not None
        assert result.action == ActionType.UPDATE_PAYMENT_METHOD

    def test_whitespace_stripped(self):
        result = lookup_fast_path("  card_expired  ")
        assert result is not None
        assert result.action == ActionType.UPDATE_PAYMENT_METHOD


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests — is_deterministic
# ═══════════════════════════════════════════════════════════════════════

class TestIsDeterministic:
    def test_deterministic_code_returns_true(self):
        assert is_deterministic("card_expired") is True
        assert is_deterministic("insufficient_funds") is True
        assert is_deterministic("41") is True

    def test_unknown_code_returns_false(self):
        assert is_deterministic("some_random_error") is False
        assert is_deterministic("") is False
        assert is_deterministic(None) is False


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests — fast_path_stats
# ═══════════════════════════════════════════════════════════════════════

class TestFastPathStats:
    def test_total_count(self):
        stats = fast_path_stats()
        assert stats["total"] == len(FAST_PATH_CACHE)

    def test_has_silent_and_active(self):
        stats = fast_path_stats()
        assert stats["silent"] > 0
        assert stats["active"] > 0

    def test_action_counts_present(self):
        stats = fast_path_stats()
        assert "action_retry_payment" in stats
        assert "action_update_payment_method" in stats
        assert "action_escalate_to_human" in stats


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests — FastPathResult dataclass
# ═══════════════════════════════════════════════════════════════════════

class TestFastPathResult:
    def test_frozen(self):
        result = lookup_fast_path("card_expired")
        with pytest.raises(AttributeError):
            result.action = ActionType.ABANDON

    def test_metadata_default_empty(self):
        result = lookup_fast_path("card_expired")
        assert result.metadata == {}

    def test_reasoning_non_empty(self):
        for code, result in FAST_PATH_CACHE.items():
            assert result.reasoning, f"Missing reasoning for {code}"


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests — RecoveryAgent fast path interception
# ═══════════════════════════════════════════════════════════════════════

def _make_case(failure_code: str, failure_reason: str = "") -> Case:
    """Helper to create a test Case with the given failure_code."""
    return Case(
        payment=PaymentEvent(
            payment_id="pay_test123",
            customer_id="cust_001",
            amount=999.0,
            failure_code=failure_code,
            failure_reason=failure_reason or failure_code,
        ),
    )


class TestRecoveryAgentFastPath:
    """Integration tests: RecoveryAgent.run() uses fast path for deterministic failures."""

    def test_card_expired_bypasses_react_loop(self):
        """Card expired → UPDATE_PAYMENT_METHOD, no LLM call."""
        from recovery_agent.agent import RecoveryAgent

        case = _make_case("card_expired", "Card expiry date is in the past")
        agent = RecoveryAgent()
        result = agent.run(case)

        # Verify fast path was used
        assert result.payment.metadata.get("fast_path") is True
        assert result.payment.metadata["decided_action"] == "update_payment_method"
        assert result.recovery_tier == RecoveryTier.ACTIVE
        assert result.diagnosis is not None
        assert result.diagnosis.root_cause == FailureType.CARD_EXPIRED
        assert result.diagnosis.category == "fast_path"
        assert result.status in (CaseStatus.RECOVERED, CaseStatus.OPEN, CaseStatus.STOPPED, CaseStatus.ACTING)

    def test_insufficient_funds_bypasses_react_loop(self):
        """Insufficient funds → WAIT_AND_RETRY, starts in silent tier."""
        from recovery_agent.agent import RecoveryAgent

        case = _make_case("insufficient_funds", "Insufficient funds in account")
        agent = RecoveryAgent()
        result = agent.run(case)

        assert result.payment.metadata.get("fast_path") is True
        assert result.payment.metadata["decided_action"] == "wait_and_retry"
        # Tier starts as SILENT but may transition to ACTIVE if retry fails
        assert result.recovery_tier in (RecoveryTier.SILENT, RecoveryTier.ACTIVE)

    def test_hard_decline_41_bypasses_react_loop(self):
        """Hard decline 41 → ESCALATE_TO_HUMAN, penalty prevented."""
        from recovery_agent.agent import RecoveryAgent

        case = _make_case("41", "Lost card")
        agent = RecoveryAgent()
        result = agent.run(case)

        assert result.payment.metadata.get("fast_path") is True
        assert result.payment.metadata["decided_action"] == "escalate_to_human"
        assert result.recovery_tier == RecoveryTier.ACTIVE

    def test_unknown_code_falls_through_to_normal_flow(self):
        """Unknown failure code → normal ReAct loop (no fast_path metadata)."""
        from recovery_agent.agent import RecoveryAgent

        case = _make_case("some_obscure_error", "Something weird happened")
        agent = RecoveryAgent()
        result = agent.run(case)

        # Fast path should NOT have been used
        assert result.payment.metadata.get("fast_path") is not True

    def test_fast_path_audit_trail_has_all_steps(self):
        """Fast path should log detect, diagnose, decide, act, stop steps."""
        from recovery_agent.agent import RecoveryAgent

        case = _make_case("card_expired")
        agent = RecoveryAgent()
        result = agent.run(case)

        steps_logged = {entry.step.value for entry in result.audit_log}
        assert "detect" in steps_logged
        assert "diagnose" in steps_logged
        assert "decide" in steps_logged
        assert "act" in steps_logged
        assert "stop" in steps_logged

    def test_harness_mode_also_uses_fast_path(self):
        """Fast path should intercept in harness mode too."""
        from recovery_agent.agent import RecoveryAgent

        case = _make_case("network_timeout", "Gateway timeout")
        agent = RecoveryAgent(use_harness=True)
        result = agent.run(case)

        assert result.payment.metadata.get("fast_path") is True
        assert result.payment.metadata["decided_action"] == "retry_payment"
        assert result.recovery_tier in (RecoveryTier.SILENT, RecoveryTier.ACTIVE)

