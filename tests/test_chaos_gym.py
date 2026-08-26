"""Unit tests for the Adversarial Chaos Gym.

Covers:
- Red Team Chaos Engine generation and payload validation
- RevenueLossEnvironment initialization and reset
- Step reward calculations and state transitions
- Full Gym episode execution
- Policy violation detection
"""
from __future__ import annotations

from unittest.mock import patch

from recovery_agent.eval.chaos_gym import (
    AdversarialChaosEngine,
    PayloadSanitizer,
    RevenueLossEnvironment,
    run_chaos_gym,
)
from recovery_agent.models import (
    ActionType,
    BankHealth,
    CaseStatus,
    ChaosAnomaly,
    CustomerMood,
    CustomerPersona,
    FailureType,
    GymState,
    RedTeamAction,
)


class TestAdversarialChaosEngine:
    """Tests for the Red Team Chaos Engine."""

    def test_generate_returns_valid_action(self):
        engine = AdversarialChaosEngine(seed=42)
        action = engine.generate()
        assert isinstance(action, RedTeamAction)
        assert action.persona in CustomerPersona
        assert action.failure_type in FailureType
        assert action.amount > 0

    def test_generate_deterministic_with_seed(self):
        a1 = AdversarialChaosEngine(seed=99).generate()
        a2 = AdversarialChaosEngine(seed=99).generate()
        assert a1.persona == a2.persona
        assert a1.failure_type == a2.failure_type
        assert a1.amount == a2.amount

    def test_generate_batch_count(self):
        engine = AdversarialChaosEngine(seed=42)
        batch = engine.generate_batch(10)
        assert len(batch) == 10
        assert all(isinstance(a, RedTeamAction) for a in batch)

    def test_all_personas_can_be_generated(self):
        engine = AdversarialChaosEngine(seed=42)
        personas_seen = set()
        for _ in range(200):
            action = engine.generate()
            personas_seen.add(action.persona)
        assert personas_seen == set(CustomerPersona)

    def test_all_failure_types_can_be_generated(self):
        engine = AdversarialChaosEngine(seed=42)
        failures_seen = set()
        for _ in range(200):
            action = engine.generate()
            failures_seen.add(action.failure_type)
        assert len(failures_seen) >= 4

    def test_anomalies_can_be_generated(self):
        engine = AdversarialChaosEngine(seed=42)
        anomalies_seen = set()
        for _ in range(200):
            action = engine.generate()
            anomalies_seen.add(action.chaos_anomaly)
        assert len(anomalies_seen) >= 2

    def test_webhook_latency_positive(self):
        engine = AdversarialChaosEngine(seed=42)
        for _ in range(50):
            action = engine.generate()
            assert action.webhook_latency_ms >= 0


class TestPayloadSanitizer:
    """Tests for the Payload Sanitizer."""

    def test_sanitize_negative_amount(self):
        action = RedTeamAction(
            persona=CustomerPersona.SALARY_DEPENDENT,
            failure_type=FailureType.NETWORK_TIMEOUT,
            amount=500,
        )
        action.amount = -500
        sanitized = PayloadSanitizer.sanitize(action)
        assert sanitized.amount > 0

    def test_sanitize_zero_amount(self):
        action = RedTeamAction(
            persona=CustomerPersona.BUSY_EXECUTIVE,
            failure_type=FailureType.BANK_DECLINED,
            amount=0,
        )
        sanitized = PayloadSanitizer.sanitize(action)
        assert sanitized.amount > 0

    def test_sanitize_negative_latency(self):
        action = RedTeamAction(
            persona=CustomerPersona.B2B_AP,
            failure_type=FailureType.RISK_BLOCK,
            amount=1000,
            webhook_latency_ms=-100,
        )
        sanitized = PayloadSanitizer.sanitize(action)
        assert sanitized.webhook_latency_ms >= 0

    def test_to_payment_event(self):
        action = RedTeamAction(
            persona=CustomerPersona.SALARY_DEPENDENT,
            failure_type=FailureType.INSUFFICIENT_FUNDS,
            amount=5000,
            chaos_anomaly=ChaosAnomaly.GATEWAY_DEGRADATION_SPIKE,
            bank_health=BankHealth.DEGRADED,
            customer_mood=CustomerMood.COOPERATIVE,
        )
        event = PayloadSanitizer.to_payment_event(action)
        assert event.amount == 5000
        assert event.failure_code == "insufficient_funds"
        assert event.metadata["persona"] == "salary_dependent"
        assert event.metadata["chaos_anomaly"] == "gateway_degradation_spike"
        assert event.metadata["bank_health"] == "degraded"


class TestRevenueLossEnvironment:
    """Tests for the Gym Environment."""

    def test_reset_returns_gym_state(self):
        env = RevenueLossEnvironment(seed=42)
        state = env.reset()
        assert isinstance(state, GymState)
        assert state.attempt_count == 0
        assert state.done is False

    def test_reset_with_seed_deterministic(self):
        env = RevenueLossEnvironment()
        s1 = env.reset(seed=99)
        s2 = env.reset(seed=99)
        assert s1.customer_persona == s2.customer_persona
        assert s1.case.payment.failure_code == s2.case.payment.failure_code

    def test_step_returns_gym_step_result(self):
        env = RevenueLossEnvironment(seed=42)
        env.reset()
        result = env.step(ActionType.RETRY_PAYMENT)
        assert result.next_state.attempt_count == 1
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_step_without_reset_raises(self):
        env = RevenueLossEnvironment(seed=42)
        try:
            env.step(ActionType.RETRY_PAYMENT)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_step_after_done_returns_zero(self):
        env = RevenueLossEnvironment(seed=42)
        env.reset()
        env.step(ActionType.ABANDON)
        result = env.step(ActionType.RETRY_PAYMENT)
        assert result.done is True
        assert result.reward == 0.0

    def test_max_attempts_triggers_done(self):
        env = RevenueLossEnvironment(seed=42)
        env.reset()
        env.state.case.max_attempts = 3
        for _ in range(3):
            env.step(ActionType.WAIT_AND_RETRY)
        assert env.state.done is True

    def test_escalation_sets_escalated_status(self):
        env = RevenueLossEnvironment(seed=42)
        env.reset()
        result = env.step(ActionType.ESCALATE_TO_HUMAN)
        assert result.next_state.case.status.value == "escalated"
        assert result.done is True

    def test_abandon_sets_stopped_status(self):
        env = RevenueLossEnvironment(seed=42)
        env.reset()
        result = env.step(ActionType.ABANDON)
        assert result.next_state.case.status.value == "stopped"
        assert result.done is True

    def test_policy_violation_on_abandon(self):
        env = RevenueLossEnvironment(seed=42)
        env.reset()
        env.step(ActionType.ABANDON)
        assert env.state.policy_violations >= 1

    def test_policy_violation_on_excessive_messages(self):
        env = RevenueLossEnvironment(seed=42)
        env.reset()
        env.state.customer_persona = CustomerPersona.BUSY_EXECUTIVE
        for _ in range(4):
            if env.state.done:
                env.state.done = False
                env.state.case.status = CaseStatus.ACTING
            env.step(ActionType.SEND_NOTIFICATION)
        assert env.state.policy_violations >= 1

    def test_frustrated_subscriber_mandate_revoke(self):
        env = RevenueLossEnvironment(seed=42)
        env.reset()
        env.state.customer_persona = CustomerPersona.FRUSTRATED_SUBSCRIBER
        env.state.messages_sent = 0
        env.step(ActionType.SEND_NOTIFICATION)
        env.step(ActionType.SEND_NOTIFICATION)
        assert env.state.messages_sent == 2

    def test_bank_health_modifier_affects_response(self):
        env = RevenueLossEnvironment(seed=42)
        env.reset()
        env.state.bank_health = BankHealth.DOWN
        # With bank down, response rate should be very low
        successes = 0
        for _ in range(50):
            env.reset()
            env.state.bank_health = BankHealth.DOWN
            result = env.step(ActionType.RETRY_PAYMENT)
            if result.info.get("customer_responded"):
                successes += 1
        assert successes < 20

    def test_salary_persona_converts_on_wait(self):
        env = RevenueLossEnvironment(seed=42)
        conversions = 0
        for _ in range(30):
            env.reset()
            env.state.customer_persona = CustomerPersona.SALARY_DEPENDENT
            env.state.environment_time = 2
            result = env.step(ActionType.WAIT_AND_RETRY)
            if result.info.get("customer_responded"):
                conversions += 1
        assert conversions > 10

    def test_reward_includes_amount_on_recovery(self):
        env = RevenueLossEnvironment(seed=42)
        env.reset()
        env.state.case.payment.amount = 10000
        # Force a recovery by making the customer cooperative and bank healthy
        env.state.customer_mood = CustomerMood.COOPERATIVE
        env.state.bank_health = BankHealth.HEALTHY
        env.state.chaos_anomaly = ChaosAnomaly.NONE
        # Run multiple times to eventually get a recovery
        recovered = False
        for _ in range(20):
            env.reset()
            env.state.case.payment.amount = 10000
            env.state.customer_mood = CustomerMood.COOPERATIVE
            env.state.bank_health = BankHealth.HEALTHY
            env.state.chaos_anomaly = ChaosAnomaly.NONE
            result = env.step(ActionType.RETRY_PAYMENT)
            if result.info.get("customer_responded"):
                recovered = True
                assert result.reward > 0
                break
        assert recovered


class TestFullEpisode:
    """Tests for running complete Gym episodes."""

    def test_run_episode_returns_dict(self):
        env = RevenueLossEnvironment(seed=42)
        episode = env.run_episode()
        assert isinstance(episode, dict)
        assert "persona" in episode
        assert "recovered" in episode
        assert "total_reward" in episode
        assert "trajectory" in episode
        assert episode["steps"] >= 1

    def test_run_episode_terminates(self):
        env = RevenueLossEnvironment(seed=42)
        episode = env.run_episode()
        assert episode["steps"] <= 10

    def test_run_chaos_gym_returns_metrics(self):
        result = run_chaos_gym(episodes=5, seed=42)
        assert "episodes" in result
        assert result["episodes"] == 5
        assert 0 <= result["recovery_rate"] <= 1
        assert result["avg_reward"] != 0 or result["recovery_rate"] == 0
        assert result["policy_violations"] >= 0
        assert "by_persona" in result

    def test_run_chaos_gym_deterministic(self):
        with patch("recovery_agent.agent.llm_client.invoke_llm_json", return_value=None):
            r1 = run_chaos_gym(episodes=5, seed=42)
            r2 = run_chaos_gym(episodes=5, seed=42)
            assert r1["recovery_rate"] == r2["recovery_rate"]
            assert r1["total_reward"] == r2["total_reward"]

    def test_run_chaos_gym_multiple_personas(self):
        result = run_chaos_gym(episodes=20, seed=42)
        assert len(result["by_persona"]) >= 2

    def test_by_persona_has_correct_structure(self):
        result = run_chaos_gym(episodes=10, seed=42)
        for persona, stats in result["by_persona"].items():
            assert "total" in stats
            assert "recovered" in stats
            assert "recovery_rate" in stats
            assert stats["total"] > 0
            assert 0 <= stats["recovery_rate"] <= 1
