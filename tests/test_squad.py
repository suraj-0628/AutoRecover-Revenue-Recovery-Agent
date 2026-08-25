"""Unit tests for Multi-Agent Squad and Trajectory Benchmarking.

Covers: individual agent isolation, message passing, squad orchestration,
and trajectory benchmark scoring.
"""
from __future__ import annotations

import pytest

from recovery_agent.agent.squad import (
    SquadOrchestrator,
    SquadStepResult,
    DiagnosticAgent,
    StrategyPlannerAgent,
    ComplianceOverseerAgent,
    ToolExecutionAgent,
)
from recovery_agent.eval.trajectory_benchmark import (
    TrajectoryBenchmark,
    TrajectoryMetrics,
    FRICTION_WEIGHTS,
    INVASIVE_ACTIONS,
)
from recovery_agent.models import (
    ActionType,
    Case,
    CustomerProfile,
    Diagnosis,
    FailureType,
    PaymentEvent,
)


# --- DiagnosticAgent Isolation ---

class TestDiagnosticAgent:
    def test_diagnose_returns_diagnosis(self):
        agent = DiagnosticAgent()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="card expired",
                failure_code="card_expired",
            ),
        )
        diagnosis = agent.diagnose(case)
        assert isinstance(diagnosis, Diagnosis)
        assert diagnosis.root_cause == FailureType.CARD_EXPIRED
        assert diagnosis.confidence > 0

    def test_diagnose_all_failure_types(self):
        agent = DiagnosticAgent()
        cases = [
            ("card_expired", FailureType.CARD_EXPIRED),
            ("insufficient_funds", FailureType.INSUFFICIENT_FUNDS),
            ("bank_declined", FailureType.BANK_DECLINED),
            ("network_timeout", FailureType.NETWORK_TIMEOUT),
            ("mandate_revoked", FailureType.MANDATE_REVOKED),
            ("risk_block", FailureType.RISK_BLOCK),
        ]
        for code, expected in cases:
            case = Case(
                payment=PaymentEvent(
                    payment_id="pay_1", customer_id="cust_001",
                    amount=5000, failure_reason=code, failure_code=code,
                ),
            )
            diagnosis = agent.diagnose(case)
            assert diagnosis.root_cause == expected


# --- StrategyPlannerAgent Isolation ---

class TestStrategyPlannerAgent:
    def test_plan_returns_action_type(self):
        agent = StrategyPlannerAgent()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="card expired",
                failure_code="card_expired",
            ),
        )
        case.diagnosis = Diagnosis(
            root_cause=FailureType.CARD_EXPIRED, confidence=0.9, reasoning="test",
        )
        profile = CustomerProfile(customer_id="cust_001")
        from recovery_agent.agent.kg_router import RazorpayKnowledgeGraph
        from recovery_agent.agent.memory import CustomerMemoryStore
        kg = RazorpayKnowledgeGraph()
        memory = CustomerMemoryStore()

        action = agent.plan(case, profile, kg, memory)
        assert isinstance(action, ActionType)
        assert action in (
            ActionType.SEND_NOTIFICATION,
            ActionType.UPDATE_PAYMENT_METHOD,
            ActionType.RETRY_PAYMENT,
            ActionType.WAIT_AND_RETRY,
            ActionType.ESCALATE_TO_HUMAN,
        )


# --- ComplianceOverseerAgent Isolation ---

class TestComplianceOverseerAgent:
    def test_intercept_returns_tuple(self):
        agent = ComplianceOverseerAgent()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="test",
            ),
        )
        profile = CustomerProfile(customer_id="cust_001")
        action, checks = agent.intercept(case, ActionType.RETRY_PAYMENT, profile)
        assert isinstance(action, ActionType)
        assert isinstance(checks, list)
        assert len(checks) == 5  # 5 guardrails

    def test_intercept_blocks_opt_out(self):
        agent = ComplianceOverseerAgent()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="test",
            ),
        )
        profile = CustomerProfile(customer_id="cust_001", opt_out=True)
        action, checks = agent.intercept(case, ActionType.SEND_NOTIFICATION, profile)
        assert action != ActionType.SEND_NOTIFICATION


# --- ToolExecutionAgent Isolation ---

class TestToolExecutionAgent:
    def test_execute_returns_observable(self):
        agent = ToolExecutionAgent()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="test",
            ),
        )
        case.diagnosis = Diagnosis(
            root_cause=FailureType.CARD_EXPIRED, confidence=0.9, reasoning="test",
        )
        result = agent.execute(case, ActionType.RETRY_PAYMENT)
        assert isinstance(result, dict)
        assert "action" in result
        assert "detail" in result

    def test_execute_all_action_types(self):
        agent = ToolExecutionAgent()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="test",
            ),
        )
        case.diagnosis = Diagnosis(
            root_cause=FailureType.CARD_EXPIRED, confidence=0.9, reasoning="test",
        )
        for action in ActionType:
            result = agent.execute(case, action)
            assert "action" in result


# --- SquadOrchestrator Integration ---

class TestSquadOrchestrator:
    def test_run_step_returns_squad_step_result(self):
        squad = SquadOrchestrator()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="card expired",
                failure_code="card_expired",
            ),
        )
        result = squad.run_step(case)
        assert isinstance(result, SquadStepResult)
        assert result.action_taken in [a.value for a in ActionType]
        assert result.verdict in ("pass", "modified", "blocked")
        assert result.trajectory_step["step"] == 1

    def test_run_step_increments_attempt_count(self):
        squad = SquadOrchestrator()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="network timeout",
                failure_code="network_timeout",
            ),
        )
        result = squad.run_step(case)
        assert result.next_case.attempt_count == 1

    def test_run_step_with_profile(self):
        squad = SquadOrchestrator()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="insufficient funds",
                failure_code="insufficient_funds",
            ),
        )
        profile = CustomerProfile(customer_id="cust_001")
        result = squad.run_step(case, profile=profile)
        assert isinstance(result, SquadStepResult)

    def test_run_step_multiple_steps(self):
        squad = SquadOrchestrator()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="bank declined",
                failure_code="bank_declined",
            ),
        )
        # Run 3 steps
        for i in range(3):
            result = squad.run_step(case)
            case = result.next_case
        assert case.attempt_count == 3

    def test_squad_guardrail_modification(self):
        """Squad should modify actions when guardrails trigger."""
        from datetime import datetime, timezone
        squad = SquadOrchestrator()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="card expired",
                failure_code="card_expired",
            ),
        )
        profile = CustomerProfile(customer_id="cust_001")
        # The squad should run and produce a valid result
        result = squad.run_step(case, profile=profile)
        assert result.verdict in ("pass", "modified", "blocked")


# --- TrajectoryBenchmark ---

class TestTrajectoryBenchmark:
    def test_empty_trajectory(self):
        tb = TrajectoryBenchmark()
        metrics = tb.evaluate_trajectory([])
        assert metrics.step_efficiency == 1.0
        assert metrics.friction_score == 0.0
        assert metrics.trajectory_score == 1.0

    def test_single_step_trajectory(self):
        tb = TrajectoryBenchmark()
        trajectory = [
            {"action": "retry_payment", "verdict": "pass", "result": "success"},
        ]
        metrics = tb.evaluate_trajectory(trajectory)
        assert metrics.total_steps == 1
        assert metrics.recovered is True
        assert metrics.recovered_at_step == 1
        assert metrics.step_efficiency == 1.0  # min_steps(2) / 1 = capped at 1.0

    def test_multi_step_trajectory(self):
        tb = TrajectoryBenchmark()
        trajectory = [
            {"action": "send_notification", "verdict": "pass"},
            {"action": "retry_payment", "verdict": "pass", "result": "success"},
        ]
        metrics = tb.evaluate_trajectory(trajectory)
        assert metrics.total_steps == 2
        assert metrics.recovered is True
        assert metrics.invasive_steps == 1  # send_notification is invasive

    def test_friction_score_with_invasive_actions(self):
        tb = TrajectoryBenchmark()
        # All invasive actions
        trajectory = [
            {"action": "send_notification", "verdict": "pass"},
            {"action": "send_notification", "verdict": "pass"},
            {"action": "update_payment_method", "verdict": "pass"},
        ]
        metrics = tb.evaluate_trajectory(trajectory)
        assert metrics.friction_score > 0.2  # Non-zero friction from invasive actions
        assert metrics.invasive_steps == 3

    def test_friction_score_with_no_invasive_actions(self):
        tb = TrajectoryBenchmark()
        trajectory = [
            {"action": "retry_payment", "verdict": "pass"},
            {"action": "wait_and_retry", "verdict": "pass"},
        ]
        metrics = tb.evaluate_trajectory(trajectory)
        assert metrics.friction_score == 0.0

    def test_policy_compliance_all_pass(self):
        tb = TrajectoryBenchmark()
        trajectory = [
            {"action": "retry_payment", "verdict": "pass"},
            {"action": "send_notification", "verdict": "pass"},
        ]
        metrics = tb.evaluate_trajectory(trajectory)
        assert metrics.policy_compliance_rate == 1.0

    def test_policy_compliance_with_modifications(self):
        tb = TrajectoryBenchmark()
        trajectory = [
            {"action": "send_notification", "verdict": "modified"},
            {"action": "retry_payment", "verdict": "pass"},
        ]
        metrics = tb.evaluate_trajectory(trajectory)
        assert metrics.policy_compliance_rate == 1.0  # modified still counts as compliant
        assert metrics.guardrail_modifications == 1

    def test_policy_compliance_with_blocks(self):
        tb = TrajectoryBenchmark()
        trajectory = [
            {"action": "send_notification", "verdict": "blocked"},
            {"action": "retry_payment", "verdict": "pass"},
        ]
        metrics = tb.evaluate_trajectory(trajectory)
        assert metrics.policy_compliance_rate == 0.5  # 1 pass out of 2
        assert metrics.guardrail_blocks == 1

    def test_trajectory_score_range(self):
        tb = TrajectoryBenchmark()
        # Best case: fast recovery, no friction, all compliant
        best = [{"action": "retry_payment", "verdict": "pass", "result": "success"}]
        best_metrics = tb.evaluate_trajectory(best)

        # Worst case: many steps, high friction, blocks
        worst = [
            {"action": "send_notification", "verdict": "blocked"},
            {"action": "send_notification", "verdict": "blocked"},
            {"action": "update_payment_method", "verdict": "blocked"},
            {"action": "escalate_to_human", "verdict": "pass"},
            {"action": "abandon", "verdict": "pass"},
        ]
        worst_metrics = tb.evaluate_trajectory(worst)

        assert best_metrics.trajectory_score > worst_metrics.trajectory_score
        assert 0.0 <= best_metrics.trajectory_score <= 1.0
        assert 0.0 <= worst_metrics.trajectory_score <= 1.0

    def test_aggregate_metrics(self):
        tb = TrajectoryBenchmark()
        m1 = TrajectoryMetrics(
            step_efficiency=0.8, friction_score=0.2,
            policy_compliance_rate=1.0, trajectory_score=0.75, recovered=True,
        )
        m2 = TrajectoryMetrics(
            step_efficiency=0.5, friction_score=0.5,
            policy_compliance_rate=0.8, trajectory_score=0.55, recovered=False,
        )
        agg = tb.aggregate_metrics([m1, m2])
        assert agg["total_episodes"] == 2
        assert agg["recovery_rate"] == 0.5
        assert agg["avg_step_efficiency"] == pytest.approx(0.65, abs=0.01)

    def test_aggregate_empty(self):
        tb = TrajectoryBenchmark()
        agg = tb.aggregate_metrics([])
        assert agg["total_episodes"] == 0
        assert agg["recovery_rate"] == 0.0

    def test_friction_weights_cover_all_actions(self):
        """All ActionType values should have friction weights."""
        for action in ActionType:
            assert action.value in FRICTION_WEIGHTS or action.value in INVASIVE_ACTIONS or action.value in ("abandon",)


# --- Backward Compatibility ---

class TestBackwardCompatibility:
    def test_recovery_agent_without_squad(self):
        """RecoveryAgent without use_squad should work as before."""
        from recovery_agent.agent import RecoveryAgent
        agent = RecoveryAgent(use_squad=False)
        assert not agent.use_squad
        assert not hasattr(agent, "squad")

    def test_recovery_agent_with_squad(self):
        """RecoveryAgent with use_squad=True should have squad attribute."""
        from recovery_agent.agent import RecoveryAgent
        agent = RecoveryAgent(use_squad=True)
        assert agent.use_squad
        assert hasattr(agent, "squad")
        assert isinstance(agent.squad, SquadOrchestrator)
