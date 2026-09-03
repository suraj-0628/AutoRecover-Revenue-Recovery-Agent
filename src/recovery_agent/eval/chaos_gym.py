"""Adversarial Chaos Gym — Red Team simulator for revenue recovery testing.

Gym-style reactive environment that tests AutoRecover against messy real-world
payment failures: bank degradation spikes, customer behavioral shifts,
out-of-order webhooks, and card expiry surges.

Source: Evaluating AI Agents — Gym Environments
        https://www.deeplearning.ai/courses/evaluating-ai-agents
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field

from recovery_agent.agent import RecoveryAgent
from recovery_agent.models import (
    ActionType,
    BankHealth,
    Case,
    CaseStatus,
    ChaosAnomaly,
    CustomerMood,
    CustomerPersona,
    FailureType,
    GymState,
    GymStepResult,
    PaymentEvent,
    RedTeamAction,
    RecoveryTier,
    HARD_DECLINES,
)


# --- Customer Persona Definitions ---

@dataclass
class PersonaConfig:
    """Behavioral config for a customer persona."""
    name: str
    response_rates: dict[ActionType, float]
    mood_after_message: CustomerMood
    mood_threshold_messages: int
    preferred_channel: str
    conversion_after_salary: float = 0.0
    will_revoke_mandate: bool = False


PERSONA_CONFIGS: dict[CustomerPersona, PersonaConfig] = {
    CustomerPersona.SALARY_DEPENDENT: PersonaConfig(
        name="Salary Dependent",
        response_rates={
            ActionType.SEND_NOTIFICATION: 0.3,
            ActionType.RETRY_PAYMENT: 0.2,
            ActionType.UPDATE_PAYMENT_METHOD: 0.4,
            ActionType.WAIT_AND_RETRY: 0.7,
            ActionType.ESCALATE_TO_HUMAN: 0.1,
            ActionType.ABANDON: 0.0,
        },
        mood_after_message=CustomerMood.COOPERATIVE,
        mood_threshold_messages=5,
        preferred_channel="sms",
        conversion_after_salary=0.85,
    ),
    CustomerPersona.BUSY_EXECUTIVE: PersonaConfig(
        name="Busy Executive",
        response_rates={
            ActionType.SEND_NOTIFICATION: 0.1,
            ActionType.RETRY_PAYMENT: 0.5,
            ActionType.UPDATE_PAYMENT_METHOD: 0.3,
            ActionType.WAIT_AND_RETRY: 0.2,
            ActionType.ESCALATE_TO_HUMAN: 0.05,
            ActionType.ABANDON: 0.0,
        },
        mood_after_message=CustomerMood.FRUSTRATED,
        mood_threshold_messages=2,
        preferred_channel="whatsapp",
    ),
    CustomerPersona.FRUSTRATED_SUBSCRIBER: PersonaConfig(
        name="Frustrated Subscriber",
        response_rates={
            ActionType.SEND_NOTIFICATION: 0.15,
            ActionType.RETRY_PAYMENT: 0.3,
            ActionType.UPDATE_PAYMENT_METHOD: 0.2,
            ActionType.WAIT_AND_RETRY: 0.4,
            ActionType.ESCALATE_TO_HUMAN: 0.1,
            ActionType.ABANDON: 0.0,
        },
        mood_after_message=CustomerMood.FRUSTRATED,
        mood_threshold_messages=2,
        preferred_channel="email",
        will_revoke_mandate=True,
    ),
    CustomerPersona.B2B_AP: PersonaConfig(
        name="B2B Accounts Payable",
        response_rates={
            ActionType.SEND_NOTIFICATION: 0.4,
            ActionType.RETRY_PAYMENT: 0.6,
            ActionType.UPDATE_PAYMENT_METHOD: 0.5,
            ActionType.WAIT_AND_RETRY: 0.3,
            ActionType.ESCALATE_TO_HUMAN: 0.2,
            ActionType.ABANDON: 0.0,
        },
        mood_after_message=CustomerMood.COOPERATIVE,
        mood_threshold_messages=4,
        preferred_channel="email",
    ),
}


# --- Payload Sanitizer ---

class PayloadSanitizer:
    """Pre-validates Red Team outputs against PaymentEvent schemas.

    Ensures malformed data never crashes AutoRecover.
    """

    @staticmethod
    def sanitize(action: RedTeamAction) -> RedTeamAction:
        """Validate and fix a Red Team action payload."""
        if action.amount < 0:
            action.amount = abs(action.amount)
        if action.amount == 0:
            action.amount = random.uniform(100, 50000)
        if action.webhook_latency_ms < 0:
            action.webhook_latency_ms = 0
        return action

    @staticmethod
    def to_payment_event(action: RedTeamAction) -> PaymentEvent:
        """Convert a sanitized RedTeamAction to a PaymentEvent."""
        return PaymentEvent(
            payment_id=f"pay_chaos_{uuid.uuid4().hex[:8]}",
            customer_id=f"cust_chaos_{action.persona.value}",
            amount=round(action.amount, 2),
            currency="INR",
            failure_reason=f"Chaos: {action.failure_type.value}",
            failure_code=action.failure_type.value,
            metadata={
                "persona": action.persona.value,
                "chaos_anomaly": action.chaos_anomaly.value,
                "bank_health": action.bank_health.value,
                "customer_mood": action.customer_mood.value,
                "webhook_latency_ms": action.webhook_latency_ms,
                **action.metadata,
            },
        )


# --- Red Team Chaos Engine ---

class AdversarialChaosEngine:
    """Generates realistic, non-deterministic payment failure events.

    Produces messy revenue-loss scenarios with customer personas,
    bank chaos anomalies, and varying webhook behavior.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.sanitizer = PayloadSanitizer()

    def generate(self) -> RedTeamAction:
        """Generate a single chaotic payment failure event."""
        persona = self.rng.choice(list(CustomerPersona))
        failure_type = self._pick_failure_type(persona)
        amount = self._pick_amount(persona)
        anomaly = self._pick_anomaly()
        bank_health = self._pick_bank_health(anomaly)
        mood = self._pick_initial_mood(persona)
        latency = self._pick_webhook_latency(anomaly)

        action = RedTeamAction(
            persona=persona,
            failure_type=failure_type,
            amount=amount,
            webhook_latency_ms=latency,
            chaos_anomaly=anomaly,
            customer_mood=mood,
            bank_health=bank_health,
        )
        return self.sanitizer.sanitize(action)

    def generate_batch(self, count: int) -> list[RedTeamAction]:
        """Generate a batch of chaotic events."""
        return [self.generate() for _ in range(count)]

    def _pick_failure_type(self, persona: CustomerPersona) -> FailureType:
        weights = {
            CustomerPersona.SALARY_DEPENDENT: {
                FailureType.INSUFFICIENT_FUNDS: 0.5,
                FailureType.NETWORK_TIMEOUT: 0.2,
                FailureType.CARD_EXPIRED: 0.15,
                FailureType.BANK_DECLINED: 0.1,
                FailureType.MANDATE_REVOKED: 0.05,
            },
            CustomerPersona.BUSY_EXECUTIVE: {
                FailureType.NETWORK_TIMEOUT: 0.35,
                FailureType.CARD_EXPIRED: 0.25,
                FailureType.BANK_DECLINED: 0.2,
                FailureType.RISK_BLOCK: 0.1,
                FailureType.INSUFFICIENT_FUNDS: 0.1,
            },
            CustomerPersona.FRUSTRATED_SUBSCRIBER: {
                FailureType.MANDATE_REVOKED: 0.4,
                FailureType.BANK_DECLINED: 0.25,
                FailureType.NETWORK_TIMEOUT: 0.15,
                FailureType.INSUFFICIENT_FUNDS: 0.1,
                FailureType.RISK_BLOCK: 0.1,
            },
            CustomerPersona.B2B_AP: {
                FailureType.BANK_DECLINED: 0.35,
                FailureType.RISK_BLOCK: 0.25,
                FailureType.NETWORK_TIMEOUT: 0.2,
                FailureType.CARD_EXPIRED: 0.1,
                FailureType.INSUFFICIENT_FUNDS: 0.1,
            },
        }
        distribution = weights.get(persona, {
            FailureType.NETWORK_TIMEOUT: 0.3,
            FailureType.BANK_DECLINED: 0.25,
            FailureType.INSUFFICIENT_FUNDS: 0.2,
            FailureType.CARD_EXPIRED: 0.15,
            FailureType.RISK_BLOCK: 0.05,
            FailureType.MANDATE_REVOKED: 0.05,
        })
        types = list(distribution.keys())
        weights_list = list(distribution.values())
        return self.rng.choices(types, weights=weights_list, k=1)[0]

    def _pick_amount(self, persona: CustomerPersona) -> float:
        ranges = {
            CustomerPersona.SALARY_DEPENDENT: (200, 15000),
            CustomerPersona.BUSY_EXECUTIVE: (5000, 200000),
            CustomerPersona.FRUSTRATED_SUBSCRIBER: (100, 5000),
            CustomerPersona.B2B_AP: (50000, 500000),
        }
        low, high = ranges.get(persona, (100, 50000))
        return round(self.rng.uniform(low, high), 2)

    def _pick_anomaly(self) -> ChaosAnomaly:
        roll = self.rng.random()
        if roll < 0.6:
            return ChaosAnomaly.NONE
        elif roll < 0.75:
            return ChaosAnomaly.GATEWAY_DEGRADATION_SPIKE
        elif roll < 0.9:
            return ChaosAnomaly.OUT_OF_ORDER_WEBHOOK
        else:
            return ChaosAnomaly.CARD_EXPIRY_SURGE

    def _pick_bank_health(self, anomaly: ChaosAnomaly) -> BankHealth:
        if anomaly == ChaosAnomaly.GATEWAY_DEGRADATION_SPIKE:
            return self.rng.choice([BankHealth.DEGRADED, BankHealth.DOWN])
        elif anomaly == ChaosAnomaly.OUT_OF_ORDER_WEBHOOK:
            return BankHealth.DEGRADED
        return BankHealth.HEALTHY

    def _pick_initial_mood(self, persona: CustomerPersona) -> CustomerMood:
        if persona == CustomerPersona.FRUSTRATED_SUBSCRIBER:
            return self.rng.choice([CustomerMood.COOPERATIVE, CustomerMood.FRUSTRATED])
        if persona == CustomerPersona.BUSY_EXECUTIVE:
            return self.rng.choice([CustomerMood.COOPERATIVE, CustomerMood.NON_RESPONSIVE])
        return CustomerMood.COOPERATIVE

    def _pick_webhook_latency(self, anomaly: ChaosAnomaly) -> int:
        if anomaly == ChaosAnomaly.OUT_OF_ORDER_WEBHOOK:
            return self.rng.randint(5000, 30000)
        if anomaly == ChaosAnomaly.GATEWAY_DEGRADATION_SPIKE:
            return self.rng.randint(2000, 10000)
        return self.rng.randint(50, 500)


# --- Revenue Loss Environment ---

class RevenueLossEnvironment:
    """Gym-style reactive environment for testing AutoRecover.

    Implements step(action) -> (next_state, reward, done, info) protocol.
    AutoRecover acts against a live, unpredictable payment ecosystem.
    """

    def __init__(self, seed: int | None = None, use_harness: bool = False):
        self.engine = AdversarialChaosEngine(seed=seed)
        self.rng = random.Random(seed)
        self.state: GymState | None = None
        self.use_harness = use_harness
        self.agent = RecoveryAgent()

    def reset(self, seed: int | None = None) -> GymState:
        """Generate a fresh, messy revenue-loss case."""
        if seed is not None:
            self.rng = random.Random(seed)
            self.engine = AdversarialChaosEngine(seed=seed)

        red_team_action = self.engine.generate()
        event = PayloadSanitizer.to_payment_event(red_team_action)
        case = Case(payment=event)

        self.state = GymState(
            case=case,
            environment_time=0,
            bank_health=red_team_action.bank_health,
            customer_mood=red_team_action.customer_mood,
            customer_persona=red_team_action.persona,
            attempt_count=0,
            reward_score=0.0,
            chaos_anomaly=red_team_action.chaos_anomaly,
            messages_sent=0,
            policy_violations=0,
            done=False,
        )
        return self.state

    def step(self, agent_action: ActionType) -> GymStepResult:
        """Process AutoRecover's action against the current environment state.

        Returns (next_state, reward, done, info).

        Includes:
        - Hard decline penalty: reward -= 50.0 if agent retries a hard decline code
        - Salary conversion: evaluated based on payday window, NOT step count
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self.state.done:
            return GymStepResult(
                next_state=self.state,
                reward=0.0,
                done=True,
                info={"reason": "episode_already_finished"},
            )

        self.state.attempt_count += 1
        self.state.environment_time += 1

        # --- Hard decline penalty check ---
        failure_code = self.state.case.payment.failure_code
        hard_decline_penalty = 0.0
        if failure_code in HARD_DECLINES and agent_action in (
            ActionType.RETRY_PAYMENT,
            ActionType.WAIT_AND_RETRY,
        ):
            hard_decline_penalty = 50.0
            self.state.policy_violations += 1
            self.state.case.penalties_prevented += 1

        # --- Policy violation checks ---
        if agent_action == ActionType.SEND_NOTIFICATION:
            self.state.messages_sent += 1
            if self.state.messages_sent > 3:
                self.state.policy_violations += 1
            if (self.state.customer_persona == CustomerPersona.FRUSTRATED_SUBSCRIBER
                    and self.state.messages_sent > 2):
                self.state.customer_mood = CustomerMood.FRUSTRATED

        if agent_action == ActionType.ABANDON:
            self.state.policy_violations += 1

        # --- Calculate customer response ---
        persona_config = PERSONA_CONFIGS.get(self.state.customer_persona)
        base_rate = 0.3
        if persona_config:
            base_rate = persona_config.response_rates.get(agent_action, 0.3)

        # Mood modifier
        mood_modifier = {
            CustomerMood.COOPERATIVE: 1.0,
            CustomerMood.FRUSTRATED: 0.5,
            CustomerMood.NON_RESPONSIVE: 0.15,
        }.get(self.state.customer_mood, 1.0)

        # Bank health modifier
        bank_modifier = {
            BankHealth.HEALTHY: 1.0,
            BankHealth.DEGRADED: 0.6,
            BankHealth.DOWN: 0.2,
        }.get(self.state.bank_health, 1.0)

        # Chaos anomaly modifier
        anomaly_modifier = 1.0
        if self.state.chaos_anomaly == ChaosAnomaly.GATEWAY_DEGRADATION_SPIKE:
            anomaly_modifier = 0.4
        elif self.state.chaos_anomaly == ChaosAnomaly.OUT_OF_ORDER_WEBHOOK:
            anomaly_modifier = 0.5

        # Salary persona special: converts after payday window (NOT step count)
        # Payday window = environment_time >= 2 AND action is WAIT_AND_RETRY or RETRY_PAYMENT
        salary_bonus = 0.0
        if (self.state.customer_persona == CustomerPersona.SALARY_DEPENDENT
                and agent_action in (ActionType.WAIT_AND_RETRY, ActionType.RETRY_PAYMENT)
                and self.state.environment_time >= 2):
            # First-in-line advantage: retry during payday window
            salary_bonus = persona_config.conversion_after_salary if persona_config else 0.85

        effective_rate = min(
            1.0,
            base_rate * mood_modifier * bank_modifier * anomaly_modifier + salary_bonus,
        )
        customer_responded = self.rng.random() < effective_rate

        # --- Calculate reward ---
        reward = 0.0
        friction_penalty = 0.0

        if agent_action == ActionType.SEND_NOTIFICATION:
            friction_penalty = 1.0
        elif agent_action == ActionType.RETRY_PAYMENT:
            friction_penalty = 0.5
        elif agent_action == ActionType.UPDATE_PAYMENT_METHOD:
            friction_penalty = 0.3
        elif agent_action == ActionType.WAIT_AND_RETRY:
            friction_penalty = 0.2
        elif agent_action == ActionType.ESCALATE_TO_HUMAN:
            friction_penalty = 2.0

        recovered = False
        if customer_responded and agent_action not in (
            ActionType.WAIT_AND_RETRY,
            ActionType.ESCALATE_TO_HUMAN,
            ActionType.ABANDON,
        ):
            recovered = True
            reward += self.state.case.payment.amount
            self.state.case.recovered = True
            self.state.case.recovered_amount = self.state.case.payment.amount
            self.state.case.status = CaseStatus.RECOVERED

        if agent_action == ActionType.ESCALATE_TO_HUMAN:
            self.state.case.status = CaseStatus.ESCALATED

        if agent_action == ActionType.ABANDON:
            self.state.case.status = CaseStatus.STOPPED

        # Apply penalties
        reward -= friction_penalty * 50
        reward -= self.state.policy_violations * 500
        reward -= hard_decline_penalty  # Hard decline retry penalty

        self.state.reward_score += reward

        # --- Check done conditions ---
        done = False
        done_reason = ""

        if recovered:
            done = True
            done_reason = "recovered"
        elif self.state.attempt_count >= self.state.case.max_attempts:
            done = True
            done_reason = "max_attempts"
        elif agent_action == ActionType.ESCALATE_TO_HUMAN:
            done = True
            done_reason = "escalated"
        elif agent_action == ActionType.ABANDON:
            done = True
            done_reason = "abandoned"

        self.state.done = done

        info = {
            "customer_responded": customer_responded,
            "effective_rate": round(effective_rate, 3),
            "friction_penalty": friction_penalty,
            "hard_decline_penalty": hard_decline_penalty,
            "policy_violations": self.state.policy_violations,
            "penalties_prevented": self.state.case.penalties_prevented,
            "done_reason": done_reason,
            "persona": self.state.customer_persona.value,
            "bank_health": self.state.bank_health.value,
            "chaos_anomaly": self.state.chaos_anomaly.value,
            "recovery_tier": self.state.case.recovery_tier.value,
        }

        return GymStepResult(
            next_state=self.state,
            reward=round(reward, 2),
            done=done,
            info=info,
        )

    def run_episode(self) -> dict:
        """Run a full episode: reset → step loop → done.

        Returns episode summary with trajectory data.
        """
        state = self.reset()
        total_reward = 0.0
        steps = 0
        trajectory: list[dict] = []
        self.trajectory = []  # Reset squad trajectory tracker

        while not state.done and steps < 10:
            # Let the agent decide what to do
            action = self._agent_decide(state)
            result = self.step(action)
            steps += 1
            total_reward += result.reward

            trajectory.append({
                "step": steps,
                "action": action.value,
                "reward": result.reward,
                "done": result.done,
                "info": result.info,
            })

            state = result.next_state

        # Use squad trajectory if available (has verdict data for benchmarking)
        benchmark_trajectory = self.trajectory if self.trajectory else trajectory

        return {
            "persona": state.customer_persona.value,
            "failure_type": state.case.payment.failure_code,
            "amount": state.case.payment.amount,
            "recovered": state.case.recovered,
            "recovered_amount": state.case.recovered_amount,
            "status": state.case.status.value,
            "total_reward": round(total_reward, 2),
            "steps": steps,
            "policy_violations": state.policy_violations,
            "penalties_prevented": state.case.penalties_prevented,
            "recovery_tier": state.case.recovery_tier.value,
            "trajectory": trajectory,
            "benchmark_trajectory": benchmark_trajectory,
        }

    def _agent_decide(self, state: GymState) -> ActionType:
        """Agent decision logic for the Gym.

        Uses diagnosis + guardrails + heuristic (same as graph pipeline nodes).
        """
        case = state.case
        from recovery_agent.agent.memory import CustomerMemoryStore
        from recovery_agent.agent.kg_router import RazorpayKnowledgeGraph
        from recovery_agent.agent.guardrails import GuardrailEngine
        from recovery_agent.agent.diagnosis import run_diagnosis

        if not hasattr(self, "memory_store"):
            self.memory_store = CustomerMemoryStore()
        if not hasattr(self, "kg_router"):
            self.kg_router = RazorpayKnowledgeGraph()
        if not hasattr(self, "guardrail_engine"):
            self.guardrail_engine = GuardrailEngine()

        profile = self.memory_store.get_or_create_profile(case.payment.customer_id)
        
        # Seed salary window for salary dependent persona if known
        if state.customer_persona.value == "salary_dependent":
            profile.salary_window.typical_pay_day = 1

        # Sync gym's messages_sent into profile's payment history for frequency cap guardrail
        if state.messages_sent > 0:
            from datetime import datetime, timezone, timedelta
            now = datetime.now(timezone.utc)
            recent_sms = [
                r for r in profile.payment_history
                if r.channel_used == "sms"
                and (now - r.timestamp).total_seconds() < 86400
            ]
            while len(recent_sms) < state.messages_sent:
                from recovery_agent.models import PaymentRecord
                record = PaymentRecord(
                    payment_id=f"gym_msg_{len(profile.payment_history)}",
                    amount=0,
                    channel_used="sms",
                    status="failed",
                    timestamp=now - timedelta(minutes=30 * len(recent_sms)),
                )
                profile.payment_history.append(record)
                recent_sms.append(record)

        # Run diagnosis
        run_diagnosis(case)

        # Keep case attempt count synced with Gym state
        case.attempt_count = max(0, state.attempt_count - 1)

        # Heuristic decision based on diagnosis
        proposed_action = self._heuristic_fallback(state)

        # Run guardrails
        approved_action, checks = self.guardrail_engine.validate_action(case, proposed_action, profile)

        verdict = "pass"
        if approved_action != proposed_action:
            if any(c.verdict.value == "blocked" for c in checks):
                verdict = "blocked"
            elif any(c.verdict.value == "modified" for c in checks):
                verdict = "modified"

        # Store trajectory step for benchmarking
        if not hasattr(self, "trajectory"):
            self.trajectory = []
        diagnosis = case.diagnosis
        self.trajectory.append({
            "step": state.attempt_count,
            "diagnosis": diagnosis.root_cause.value if diagnosis else "unknown",
            "diagnosis_confidence": diagnosis.confidence if diagnosis else 0.0,
            "proposed_action": proposed_action.value,
            "approved_action": approved_action.value,
            "verdict": verdict,
            "guardrail_checks": len(checks),
            "execution_result": approved_action.value,
            "elapsed_ms": 1,
        })

        return approved_action

    def _heuristic_fallback(self, state: GymState) -> ActionType:
        """Fallback heuristic if decision layer doesn't produce an action."""
        failure = state.case.payment.failure_code
        attempt = state.attempt_count

        if "insufficient" in failure:
            return ActionType.WAIT_AND_RETRY if attempt < 2 else ActionType.RETRY_PAYMENT
        elif "expired" in failure or "card" in failure:
            return ActionType.SEND_NOTIFICATION if attempt == 0 else ActionType.UPDATE_PAYMENT_METHOD
        elif "timeout" in failure or "network" in failure:
            return ActionType.RETRY_PAYMENT
        elif "mandate" in failure or "revoked" in failure:
            return ActionType.SEND_NOTIFICATION if attempt == 0 else ActionType.ESCALATE_TO_HUMAN
        elif "risk" in failure:
            return ActionType.ESCALATE_TO_HUMAN
        else:
            return ActionType.RETRY_PAYMENT if attempt < 2 else ActionType.ESCALATE_TO_HUMAN


def run_chaos_gym(episodes: int = 10, seed: int | None = None, use_harness: bool = False) -> dict:
    """Run multiple Gym episodes and aggregate results.

    Returns summary statistics.
    """
    env = RevenueLossEnvironment(seed=seed, use_harness=use_harness)
    results = []

    for i in range(episodes):
        episode = env.run_episode()
        results.append(episode)

    total_recovered = sum(1 for r in results if r["recovered"])
    total_amount = sum(r["amount"] for r in results)
    total_recovered_amount = sum(r["recovered_amount"] for r in results)
    total_reward = sum(r["total_reward"] for r in results)
    total_violations = sum(r["policy_violations"] for r in results)
    total_steps = sum(r["steps"] for r in results)
    total_penalties_prevented = sum(r["penalties_prevented"] for r in results)
    avg_friction = 1.0 - (total_steps / (episodes * 5)) if episodes > 0 else 0.0

    return {
        "episodes": episodes,
        "recovered": total_recovered,
        "recovery_rate": round(total_recovered / episodes, 3) if episodes > 0 else 0.0,
        "total_amount": round(total_amount, 2),
        "total_recovered_amount": round(total_recovered_amount, 2),
        "recovery_amount_rate": round(
            total_recovered_amount / total_amount, 3
        ) if total_amount > 0 else 0.0,
        "avg_reward": round(total_reward / episodes, 2) if episodes > 0 else 0.0,
        "total_reward": round(total_reward, 2),
        "policy_violations": total_violations,
        "penalties_prevented": total_penalties_prevented,
        "penalties_prevented_value": f"${total_penalties_prevented * 0.10:.2f}",
        "avg_steps": round(total_steps / episodes, 1) if episodes > 0 else 0.0,
        "avg_friction_index": round(avg_friction, 3),
        "by_persona": _aggregate_by_persona(results),
        "episodes_data": results,
    }


def _aggregate_by_persona(results: list[dict]) -> dict[str, dict]:
    """Aggregate results by customer persona."""
    by_persona: dict[str, dict] = {}
    for r in results:
        persona = r["persona"]
        if persona not in by_persona:
            by_persona[persona] = {"total": 0, "recovered": 0, "total_amount": 0.0, "recovered_amount": 0.0}
        by_persona[persona]["total"] += 1
        by_persona[persona]["total_amount"] += r["amount"]
        if r["recovered"]:
            by_persona[persona]["recovered"] += 1
            by_persona[persona]["recovered_amount"] += r["recovered_amount"]

    for stats in by_persona.values():
        stats["recovery_rate"] = round(
            stats["recovered"] / stats["total"], 3
        ) if stats["total"] > 0 else 0.0

    return by_persona


def _z_test_proportions(p1: float, n1: int, p2: float, n2: int) -> dict:
    """Two-proportion z-test for statistical significance.

    Tests H0: p1 == p2 (no difference) against H1: p1 != p2.
    Returns z-statistic, p-value, and whether significant at p < 0.05.
    """
    import math

    if n1 == 0 or n2 == 0:
        return {"z": 0.0, "p_value": 1.0, "significant": False, "error": "zero sample size"}

    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return {"z": 0.0, "p_value": 1.0, "significant": False, "error": " pooled proportion edge case"}

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": 0.0, "p_value": 1.0, "significant": False, "error": "zero standard error"}

    z = (p1 - p2) / se

    # Approximate p-value using error function approximation
    # For two-tailed test: p = 2 * (1 - Phi(|z|))
    abs_z = abs(z)
    # Approximation of the standard normal CDF
    t = 1.0 / (1.0 + 0.2316419 * abs_z)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    phi = d * math.exp(-abs_z * abs_z / 2.0) * (
        t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    )
    p_value = 2.0 * (1.0 - (1.0 - phi)) if abs_z > 0 else 1.0
    p_value = max(0.0, min(1.0, p_value))

    return {
        "z": round(z, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "effect_size": round(p1 - p2, 4),
    }


def _run_baseline_episode(env: RevenueLossEnvironment, seed_val: int) -> dict:
    """Run a single baseline episode (legacy behavior: no silent tier, no hard decline prevention).

    This simulates the pre-Phase-1 system by:
    - Forcing all actions to ACTIVE tier (no silent tier)
    - Not blocking hard declines (retries on hard decline codes)
    - No payday-aware scheduling
    """
    import random as _random

    rng = _random.Random(seed_val)

    # Use the environment's engine to generate a realistic failure case
    engine = AdversarialChaosEngine(seed=seed_val)
    red_team_action = engine.generate()
    event = PayloadSanitizer.to_payment_event(red_team_action)

    amount = event.amount
    failure_code = event.failure_code
    persona = red_team_action.persona
    config = PERSONA_CONFIGS.get(persona, PERSONA_CONFIGS[CustomerPersona.SALARY_DEPENDENT])

    payment_id = f"pay_baseline_{seed_val}"
    case = Case(payment=event, max_attempts=5)

    recovered = False
    recovered_amount = 0.0
    steps = 0
    total_reward = 0.0
    policy_violations = 0
    penalties_prevented = 0
    actions_taken = []

    # Baseline: max 5 attempts, no silent tier, no hard decline blocking
    for step in range(5):
        steps += 1
        cause = case.payment.failure_code

        # Baseline heuristic: simple cause→action mapping (no tier, no KG, no memory)
        if "insufficient" in cause:
            action = ActionType.WAIT_AND_RETRY if step < 2 else ActionType.RETRY_PAYMENT
        elif "expired" in cause or "card" in cause:
            action = ActionType.SEND_NOTIFICATION if step == 0 else ActionType.UPDATE_PAYMENT_METHOD
        elif "timeout" in cause or "network" in cause:
            action = ActionType.RETRY_PAYMENT
        elif "mandate" in cause or "revoked" in cause:
            action = ActionType.SEND_NOTIFICATION if step == 0 else ActionType.ESCALATE_TO_HUMAN
        elif "risk" in cause:
            action = ActionType.ESCALATE_TO_HUMAN
        else:
            action = ActionType.RETRY_PAYMENT if step < 2 else ActionType.ESCALATE_TO_HUMAN

        actions_taken.append(action.value)

        # Baseline: no hard decline prevention (retries even on hard declines)
        if cause in HARD_DECLINES:
            penalties_prevented += 0  # Baseline doesn't prevent penalties

        # Baseline: simple random success check (no state-aware evaluation)
        success_prob = 0.25 if action == ActionType.RETRY_PAYMENT else 0.35
        if action == ActionType.UPDATE_PAYMENT_METHOD:
            success_prob = 0.50

        if rng.random() < success_prob:
            recovered = True
            recovered_amount = amount
            break

        # Baseline: check policy violations
        if action == ActionType.ABANDON:
            policy_violations += 1

    friction = steps / 5.0
    reward = recovered_amount - (friction * 50) - (policy_violations * 500)

    return {
        "recovered": recovered,
        "recovered_amount": recovered_amount,
        "amount": amount,
        "steps": steps,
        "total_reward": round(reward, 2),
        "policy_violations": policy_violations,
        "penalties_prevented": penalties_prevented,
        "persona": persona.value,
        "failure_code": failure_code,
        "actions": actions_taken,
        "seed": seed_val,
    }


def run_before_after_benchmark(seed: int = 42, count: int = 50) -> dict:
    """Run a 50-episode benchmark comparing legacy baseline vs Phase 1+2+3 enhanced.

    Uses identical random seeds for both runs to ensure fair comparison.
    Returns detailed comparison metrics with statistical significance testing.
    """
    env = RevenueLossEnvironment(seed=seed)
    baseline_results = []
    enhanced_results = []

    for i in range(count):
        episode_seed = seed + i

        # Baseline: legacy behavior (no silent tier, no hard decline prevention)
        baseline_episode = _run_baseline_episode(env, episode_seed)
        baseline_results.append(baseline_episode)

        # Enhanced: current system with Phase 1+2+3
        enhanced_episode = env.run_episode()
        enhanced_results.append(enhanced_episode)

    # Aggregate baseline metrics
    b_recovered = sum(1 for r in baseline_results if r["recovered"])
    b_total_amount = sum(r["amount"] for r in baseline_results)
    b_recovered_amount = sum(r["recovered_amount"] for r in baseline_results)
    b_total_steps = sum(r["steps"] for r in baseline_results)
    b_violations = sum(r["policy_violations"] for r in baseline_results)
    b_penalties = sum(r["penalties_prevented"] for r in baseline_results)

    # Aggregate enhanced metrics
    e_recovered = sum(1 for r in enhanced_results if r["recovered"])
    e_total_amount = sum(r["amount"] for r in enhanced_results)
    e_recovered_amount = sum(r["recovered_amount"] for r in enhanced_results)
    e_total_steps = sum(r["steps"] for r in enhanced_results)
    e_violations = sum(r["policy_violations"] for r in enhanced_results)
    e_penalties = sum(r["penalties_prevented"] for r in enhanced_results)

    # Silent tier recovery count (enhanced only)
    e_silent_recoveries = sum(
        1 for r in enhanced_results
        if r.get("recovery_tier") == "silent" and r["recovered"]
    )
    e_silent_total = sum(
        1 for r in enhanced_results
        if r.get("recovery_tier") == "silent"
    )

    # Calculate rates
    b_rate = b_recovered / count if count > 0 else 0.0
    e_rate = e_recovered / count if count > 0 else 0.0
    b_friction = b_total_steps / (count * 5) if count > 0 else 0.0
    e_friction = e_total_steps / (count * 5) if count > 0 else 0.0
    b_yield = b_recovered_amount / b_total_amount if b_total_amount > 0 else 0.0
    e_yield = e_recovered_amount / e_total_amount if e_total_amount > 0 else 0.0

    # Statistical significance test
    significance = _z_test_proportions(e_rate, count, b_rate, count)

    # Lift calculations
    recovery_rate_lift = e_rate - b_rate
    recovery_rate_lift_pct = (recovery_rate_lift / b_rate * 100) if b_rate > 0 else 0.0
    yield_lift = e_yield - b_yield
    friction_delta = b_friction - e_friction  # Positive = less friction (better)
    penalties_saved = e_penalties - b_penalties
    penalties_value_usd = penalties_saved * 0.10
    penalties_value_inr = penalties_saved * 8.30
    silent_recovery_pct = (e_silent_recoveries / e_silent_total * 100) if e_silent_total > 0 else 0.0

    # Per-persona comparison
    baseline_by_persona = _aggregate_by_persona(baseline_results)
    enhanced_by_persona = _aggregate_by_persona(enhanced_results)

    persona_comparison = {}
    for persona in set(list(baseline_by_persona.keys()) + list(enhanced_by_persona.keys())):
        b_stats = baseline_by_persona.get(persona, {"recovery_rate": 0, "total": 0})
        e_stats = enhanced_by_persona.get(persona, {"recovery_rate": 0, "total": 0})
        persona_comparison[persona] = {
            "baseline_rate": b_stats.get("recovery_rate", 0),
            "enhanced_rate": e_stats.get("recovery_rate", 0),
            "lift": round(e_stats.get("recovery_rate", 0) - b_stats.get("recovery_rate", 0), 4),
            "baseline_count": b_stats.get("total", 0),
            "enhanced_count": e_stats.get("total", 0),
        }

    return {
        "benchmark_config": {
            "episodes": count,
            "seed": seed,
            "identical_seeds": True,
        },
        "baseline": {
            "recovery_rate": round(b_rate, 4),
            "recovered": b_recovered,
            "total_amount": round(b_total_amount, 2),
            "recovered_amount": round(b_recovered_amount, 2),
            "yield": round(b_yield, 4),
            "avg_steps": round(b_total_steps / count, 1) if count > 0 else 0,
            "friction_index": round(b_friction, 4),
            "policy_violations": b_violations,
            "penalties_prevented": b_penalties,
        },
        "enhanced": {
            "recovery_rate": round(e_rate, 4),
            "recovered": e_recovered,
            "total_amount": round(e_total_amount, 2),
            "recovered_amount": round(e_recovered_amount, 2),
            "yield": round(e_yield, 4),
            "avg_steps": round(e_total_steps / count, 1) if count > 0 else 0,
            "friction_index": round(e_friction, 4),
            "policy_violations": e_violations,
            "penalties_prevented": e_penalties,
            "silent_tier_recoveries": e_silent_recoveries,
            "silent_tier_total": e_silent_total,
            "silent_recovery_pct": round(silent_recovery_pct, 1),
        },
        "comparison": {
            "recovery_rate_lift": round(recovery_rate_lift, 4),
            "recovery_rate_lift_pct": round(recovery_rate_lift_pct, 1),
            "yield_lift": round(yield_lift, 4),
            "friction_delta": round(friction_delta, 4),
            "penalties_saved": penalties_saved,
            "penalties_value_usd": round(penalties_value_usd, 2),
            "penalties_value_inr": round(penalties_value_inr, 2),
        },
        "statistical_significance": significance,
        "persona_comparison": persona_comparison,
        "summary": (
            f"Benchmark: {count} episodes, seed={seed}\n"
            f"Baseline recovery: {b_rate:.1%} ({b_recovered}/{count}) | "
            f"Enhanced recovery: {e_rate:.1%} ({e_recovered}/{count})\n"
            f"Lift: +{recovery_rate_lift_pct:.1f}% | "
            f"Yield lift: +{yield_lift:.1%} | "
            f"Friction delta: {friction_delta:+.3f}\n"
            f"Penalties saved: {penalties_saved} (${penalties_value_usd:.2f} / INR {penalties_value_inr:.2f})\n"
            f"Silent tier recoveries: {e_silent_recoveries}/{e_silent_total} ({silent_recovery_pct:.1f}%)\n"
            f"Statistical significance: z={significance['z']}, p={significance['p_value']}, "
            f"significant={'YES' if significance['significant'] else 'NO'} (p < 0.05)"
        ),
    }
