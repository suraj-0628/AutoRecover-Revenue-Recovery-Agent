"""Tests for Empirical Evolution — Strategy Metrics, Thompson Bandit, A/B Testing.

Mandate 4 adds data-driven strategy selection:
- StrategyMetricsStore: SQLite conversion rate tracking
- ThompsonBandit: Thompson Sampling for exploration/exploitation
- ABTestFramework: Statistical A/B testing with chi-squared significance
"""
from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from recovery_agent.agent.strategy_metrics import (
    ABTestFramework,
    ABTestResult,
    ArmMetrics,
    StrategyMetricsStore,
    ThompsonBandit,
    _normal_cdf,
)
from recovery_agent.models import ActionType, FailureType


# ═══════════════════════════════════════════════════════════════
#  ARM METRICS
# ═══════════════════════════════════════════════════════════════

class TestArmMetrics:
    """Test ArmMetrics data class."""

    def test_empty_arm(self):
        arm = ArmMetrics()
        assert arm.conversion_rate == 0.0
        assert arm.alpha == 1.0  # prior
        assert arm.beta_param == 1.0  # prior

    def test_partial_arm(self):
        arm = ArmMetrics(successes=3, attempts=10)
        assert arm.conversion_rate == 0.3
        assert arm.alpha == 4.0  # 3 + 1 prior
        assert arm.beta_param == 8.0  # (10-3) + 1 prior

    def test_perfect_arm(self):
        arm = ArmMetrics(successes=5, attempts=5)
        assert arm.conversion_rate == 1.0
        assert arm.alpha == 6.0
        assert arm.beta_param == 1.0

    def test_zero_successes(self):
        arm = ArmMetrics(successes=0, attempts=10)
        assert arm.conversion_rate == 0.0
        assert arm.alpha == 1.0
        assert arm.beta_param == 11.0


# ═══════════════════════════════════════════════════════════════
#  STRATEGY METRICS STORE
# ═══════════════════════════════════════════════════════════════

class TestStrategyMetricsStore:
    """Test SQLite-backed strategy metrics store."""

    def test_in_memory_store(self):
        store = StrategyMetricsStore()
        stats = store.get_stats()
        assert stats["total_arms"] == 0
        assert stats["total_attempts"] == 0

    def test_persistent_store(self, tmp_path):
        db = tmp_path / "metrics.db"
        store = StrategyMetricsStore(db_path=db)
        store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        # Re-open to verify persistence
        store2 = StrategyMetricsStore(db_path=db)
        arm = store2.get_arm(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT)
        assert arm.attempts == 1
        assert arm.successes == 1

    def test_record_success(self):
        store = StrategyMetricsStore()
        store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        arm = store.get_arm(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT)
        assert arm.successes == 1
        assert arm.attempts == 1

    def test_record_failure(self):
        store = StrategyMetricsStore()
        store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, False)
        arm = store.get_arm(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT)
        assert arm.successes == 0
        assert arm.attempts == 1

    def test_multiple_records_accumulate(self):
        store = StrategyMetricsStore()
        for _ in range(3):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        for _ in range(2):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, False)
        arm = store.get_arm(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT)
        assert arm.successes == 3
        assert arm.attempts == 5
        assert arm.conversion_rate == 0.6

    def test_different_actions_tracked_separately(self):
        store = StrategyMetricsStore()
        store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.WAIT_AND_RETRY, False)
        arm1 = store.get_arm(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT)
        arm2 = store.get_arm(FailureType.NETWORK_TIMEOUT, ActionType.WAIT_AND_RETRY)
        assert arm1.successes == 1
        assert arm2.successes == 0

    def test_different_failure_types_tracked_separately(self):
        store = StrategyMetricsStore()
        store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        store.record_outcome(FailureType.CARD_EXPIRED, ActionType.RETRY_PAYMENT, False)
        arm1 = store.get_arm(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT)
        arm2 = store.get_arm(FailureType.CARD_EXPIRED, ActionType.RETRY_PAYMENT)
        assert arm1.successes == 1
        assert arm2.successes == 0

    def test_get_all_arms(self):
        store = StrategyMetricsStore()
        store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.WAIT_AND_RETRY, False)
        arms = store.get_all_arms(FailureType.NETWORK_TIMEOUT)
        assert len(arms) == 2
        assert ActionType.RETRY_PAYMENT in arms
        assert ActionType.WAIT_AND_RETRY in arms

    def test_get_top_actions(self):
        store = StrategyMetricsStore()
        # Record enough data to meet min_attempts
        for _ in range(6):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        for _ in range(10):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.WAIT_AND_RETRY, False)
        top = store.get_top_actions(FailureType.NETWORK_TIMEOUT, min_attempts=5)
        assert len(top) == 2
        assert top[0][0] == ActionType.RETRY_PAYMENT  # 100% > 0%
        assert top[0][1] == 1.0

    def test_get_top_actions_min_attempts_filter(self):
        store = StrategyMetricsStore()
        for _ in range(3):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        top = store.get_top_actions(FailureType.NETWORK_TIMEOUT, min_attempts=5)
        assert len(top) == 0  # Not enough attempts

    def test_get_total_attempts(self):
        store = StrategyMetricsStore()
        store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        store.record_outcome(FailureType.CARD_EXPIRED, ActionType.SEND_NOTIFICATION, False)
        assert store.get_total_attempts() == 2

    def test_get_failure_type_stats(self):
        store = StrategyMetricsStore()
        store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, False)
        stats = store.get_failure_type_stats()
        assert "network_timeout" in stats
        assert stats["network_timeout"]["successes"] == 1
        assert stats["network_timeout"]["attempts"] == 2
        assert stats["network_timeout"]["conversion_rate"] == 0.5

    def test_clear(self):
        store = StrategyMetricsStore()
        store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        store.clear()
        arm = store.get_arm(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT)
        assert arm.attempts == 0


# ═══════════════════════════════════════════════════════════════
#  THOMPSON SAMPLING BANDIT
# ═══════════════════════════════════════════════════════════════

class TestThompsonBandit:
    """Test Thompson Sampling bandit."""

    def test_returns_none_when_insufficient_data(self):
        store = StrategyMetricsStore()
        bandit = ThompsonBandit(store, min_samples=10)
        result = bandit.select_action(FailureType.NETWORK_TIMEOUT)
        assert result is None

    def test_returns_action_when_enough_data(self):
        store = StrategyMetricsStore()
        # Record enough data for one arm
        for _ in range(15):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        bandit = ThompsonBandit(store, min_samples=10)
        # Restrict to the arm that has data
        result = bandit.select_action(
            FailureType.NETWORK_TIMEOUT,
            eligible_actions=[ActionType.RETRY_PAYMENT],
        )
        assert result == ActionType.RETRY_PAYMENT

    def test_prefers_better_action(self):
        store = StrategyMetricsStore()
        # RETRY_PAYMENT: 90% success
        for _ in range(9):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        for _ in range(1):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, False)
        # WAIT_AND_RETRY: 20% success
        for _ in range(2):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.WAIT_AND_RETRY, True)
        for _ in range(8):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.WAIT_AND_RETRY, False)

        bandit = ThompsonBandit(store, min_samples=10)
        # Restrict to the two arms that have data
        choices = [
            bandit.select_action(
                FailureType.NETWORK_TIMEOUT,
                eligible_actions=[ActionType.RETRY_PAYMENT, ActionType.WAIT_AND_RETRY],
            )
            for _ in range(100)
        ]
        retry_count = choices.count(ActionType.RETRY_PAYMENT)
        assert retry_count > 60  # Should prefer the better arm

    def test_respects_eligible_actions(self):
        store = StrategyMetricsStore()
        for _ in range(15):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        bandit = ThompsonBandit(store, min_samples=10)
        result = bandit.select_action(
            FailureType.NETWORK_TIMEOUT,
            eligible_actions=[ActionType.WAIT_AND_RETRY],  # Not the best, but only eligible
        )
        assert result == ActionType.WAIT_AND_RETRY

    def test_confidence_increases_with_data(self):
        store = StrategyMetricsStore()
        bandit = ThompsonBandit(store, min_samples=10)

        # Two arms with close performance: RETRY 60%, WAIT 50%
        for i in range(10):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, i < 6)
        for i in range(10):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.WAIT_AND_RETRY, i < 5)
        conf_low = bandit.get_confidence(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT)

        # More data: RETRY 90%, WAIT 10% → clear winner emerges
        for i in range(50):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, i < 45)
        for i in range(50):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.WAIT_AND_RETRY, i < 5)
        conf_high = bandit.get_confidence(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT)

        assert conf_high > conf_low

    def test_empirical_context_empty_when_no_data(self):
        store = StrategyMetricsStore()
        bandit = ThompsonBandit(store)
        ctx = bandit.get_empirical_context(FailureType.NETWORK_TIMEOUT)
        assert ctx == ""

    def test_empirical_context_with_data(self):
        store = StrategyMetricsStore()
        for _ in range(8):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)
        for _ in range(2):
            store.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, False)
        bandit = ThompsonBandit(store)
        ctx = bandit.get_empirical_context(FailureType.NETWORK_TIMEOUT)
        assert "network_timeout" in ctx
        assert "retry_payment" in ctx
        assert "Bandit recommends" in ctx


# ═══════════════════════════════════════════════════════════════
#  A/B TEST FRAMEWORK
# ═══════════════════════════════════════════════════════════════

class TestABTestFramework:
    """Test A/B testing framework."""

    def test_assign_variant(self):
        store = StrategyMetricsStore()
        framework = ABTestFramework(store)
        variant = framework.assign_variant("case_001", "test_exp")
        assert variant in ("control", "treatment")

    def test_get_variant(self):
        store = StrategyMetricsStore()
        framework = ABTestFramework(store)
        framework.assign_variant("case_001", "test_exp")
        v = framework.get_variant("case_001", "test_exp")
        assert v is not None

    def test_get_variant_unknown_case(self):
        store = StrategyMetricsStore()
        framework = ABTestFramework(store)
        v = framework.get_variant("unknown", "test_exp")
        assert v is None

    def test_record_outcome(self):
        store = StrategyMetricsStore()
        framework = ABTestFramework(store)
        framework.assign_variant("case_001", "test_exp")
        framework.record_outcome(
            "case_001", "test_exp", ActionType.RETRY_PAYMENT, True, 5000.0
        )
        # Verify outcome recorded
        conn = store._get_conn()
        row = conn.execute(
            "SELECT * FROM ab_outcomes WHERE case_id = 'case_001'"
        ).fetchone()
        assert row is not None
        assert row["recovered"] == 1
        assert row["amount_recovered"] == 5000.0

    def test_record_outcome_skips_unknown_case(self):
        store = StrategyMetricsStore()
        framework = ABTestFramework(store)
        # Should not raise
        framework.record_outcome(
            "unknown", "test_exp", ActionType.RETRY_PAYMENT, True
        )

    def test_compute_result_no_data(self):
        store = StrategyMetricsStore()
        framework = ABTestFramework(store)
        result = framework.compute_result("test_exp")
        assert result.control_attempts == 0
        assert result.treatment_attempts == 0
        assert result.p_value == 1.0
        assert result.is_significant is False
        assert result.winner == "none"

    def test_compute_result_with_data(self):
        store = StrategyMetricsStore()
        framework = ABTestFramework(store)

        # Assign and record outcomes
        for i in range(20):
            v = framework.assign_variant(f"case_{i}", "test_exp")
            if v == "control":
                # Control gets 20% recovery
                framework.record_outcome(
                    f"case_{i}", "test_exp", ActionType.RETRY_PAYMENT, i % 5 == 0
                )
            else:
                # Treatment gets 80% recovery
                framework.record_outcome(
                    f"case_{i}", "test_exp", ActionType.SEND_NOTIFICATION, i % 5 != 0
                )

        result = framework.compute_result("test_exp")
        assert result.control_attempts > 0
        assert result.treatment_attempts > 0
        assert result.treatment_rate > result.control_rate


    def test_lift_calculation(self):
        result = ABTestResult(
            experiment="test",
            control_name="control",
            treatment_name="treatment",
            control_attempts=100,
            control_recoveries=30,
            treatment_attempts=100,
            treatment_recoveries=60,
        )
        assert result.lift == 1.0  # 100% lift

    def test_lift_zero_control(self):
        result = ABTestResult(
            experiment="test",
            control_name="control",
            treatment_name="treatment",
            control_attempts=100,
            control_recoveries=0,
            treatment_attempts=100,
            treatment_recoveries=30,
        )
        assert result.lift == float("inf")

    def test_p_value_significant(self):
        # Large difference should be significant
        result = ABTestResult(
            experiment="test",
            control_name="control",
            treatment_name="treatment",
            control_attempts=1000,
            control_recoveries=100,  # 10%
            treatment_attempts=1000,
            treatment_recoveries=500,  # 50%
        )
        assert result.p_value < 0.001
        assert result.is_significant is True
        assert result.winner == "treatment"

    def test_p_value_not_significant(self):
        # Small difference should not be significant
        result = ABTestResult(
            experiment="test",
            control_name="control",
            treatment_name="treatment",
            control_attempts=20,
            control_recoveries=10,  # 50%
            treatment_attempts=20,
            treatment_recoveries=11,  # 55%
        )
        assert result.p_value > 0.05
        assert result.is_significant is False
        assert result.winner == "none"

    def test_get_all_experiments(self):
        store = StrategyMetricsStore()
        framework = ABTestFramework(store)
        framework.assign_variant("c1", "exp1")
        framework.record_outcome("c1", "exp1", ActionType.RETRY_PAYMENT, True)
        framework.assign_variant("c2", "exp2")
        framework.record_outcome("c2", "exp2", ActionType.RETRY_PAYMENT, False)
        experiments = framework.get_all_experiments()
        assert "exp1" in experiments
        assert "exp2" in experiments


# ═══════════════════════════════════════════════════════════════
#  MATH UTILITIES
# ═══════════════════════════════════════════════════════════════

class TestMathUtilities:
    """Test math helper functions."""

    def test_normal_cdf_zero(self):
        assert abs(_normal_cdf(0.0) - 0.5) < 0.001

    def test_normal_cdf_positive(self):
        assert _normal_cdf(1.96) > 0.97

    def test_normal_cdf_negative(self):
        assert _normal_cdf(-1.96) < 0.03


# ═══════════════════════════════════════════════════════════════
#  HARNESS INTEGRATION
# ═══════════════════════════════════════════════════════════════

class TestHarnessStrategyMetricsIntegration:
    """Test strategy metrics integration with AgentHarness."""

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_harness_records_strategy_outcome(self, mock_llm, tmp_path):
        mock_llm.return_value = {
            "reasoning": "Recovered.",
            "tool_calls": [],
            "is_final": True,
            "status": "recovered",
        }
        from recovery_agent.agent.harness import AgentHarness

        metrics = StrategyMetricsStore()
        harness = AgentHarness(strategy_metrics=metrics)

        case = _make_case_with_attempt()
        harness.run_recovery_case(case)

        # Should have recorded the outcome
        assert metrics.get_total_attempts() >= 1

    @patch("recovery_agent.agent.harness.invoke_llm_json_async")
    def test_async_harness_records_strategy_outcome(self, mock_llm_async, tmp_path):
        import asyncio
        mock_llm_async.return_value = {
            "reasoning": "Done.",
            "tool_calls": [],
            "is_final": True,
            "status": "recovered",
        }
        from recovery_agent.agent.harness import AgentHarness

        metrics = StrategyMetricsStore()
        harness = AgentHarness(strategy_metrics=metrics)

        case = _make_case_with_attempt()
        asyncio.run(harness.run_recovery_case_async(case))

        assert metrics.get_total_attempts() >= 1


# ═══════════════════════════════════════════════════════════════
#  DECISION INTEGRATION
# ═══════════════════════════════════════════════════════════════

class TestDecisionBanditIntegration:
    """Test bandit integration with decision layer."""

    def test_bandit_recommendation_in_metadata(self):
        import recovery_agent.agent.decision as decision_mod
        from recovery_agent.agent.decision import decide_intervention
        from recovery_agent.models import Case, PaymentEvent, Diagnosis

        # Use fresh instances to avoid polluting global singletons
        metrics = StrategyMetricsStore()
        bandit = ThompsonBandit(metrics)

        # Seed with enough data for bandit to have a recommendation
        for _ in range(15):
            metrics.record_outcome(FailureType.NETWORK_TIMEOUT, ActionType.RETRY_PAYMENT, True)

        event = PaymentEvent(
            event_type="payment_failed",
            payment_id="pay_bandit_test",
            customer_id="cust_bandit",
            amount=5000.0,
            currency="INR",
            status="failed",
            failure_reason="Network timeout",
            failure_code="network_timeout",
        )
        case = Case(payment=event, max_attempts=5)
        case.diagnosis = Diagnosis(
            root_cause=FailureType.NETWORK_TIMEOUT,
            confidence=0.9,
            reasoning="Test",
        )
        case.recovery_tier = RecoveryTier.SILENT

        action = decide_intervention(
            case, strategy_metrics=metrics, bandit=bandit,
        )
        # Bandit should have made a recommendation
        assert "bandit_recommendation" in case.payment.metadata


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _make_case_with_attempt():
    """Create a case with an attempt for testing harness integration."""
    from recovery_agent.models import (
        Attempt,
        Case,
        PaymentEvent,
        FailureType,
    )

    event = PaymentEvent(
        event_type="payment_failed",
        payment_id="pay_metrics_test",
        customer_id="cust_metrics",
        amount=5000.0,
        currency="INR",
        status="failed",
        failure_reason="Test network timeout",
        failure_code="network_timeout",
    )
    case = Case(payment=event, max_attempts=5)
    case.diagnosis = Diagnosis(
        root_cause=FailureType.NETWORK_TIMEOUT,
        confidence=0.9,
        reasoning="Test",
    )
    case.attempts = [Attempt(action_type=ActionType.RETRY_PAYMENT)]
    case.recovered = True
    case.recovered_amount = 5000.0
    case.status = CaseStatus.RECOVERED
    return case


# Import needed for helpers
from recovery_agent.models import CaseStatus, Diagnosis, RecoveryTier
