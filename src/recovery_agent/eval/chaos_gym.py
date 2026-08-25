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

    def __init__(self, seed: int | None = None):
        self.engine = AdversarialChaosEngine(seed=seed)
        self.rng = random.Random(seed)
        self.state: GymState | None = None
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

        # Salary persona special: converts after salary credit time (environment_time >= 2)
        salary_bonus = 0.0
        if (self.state.customer_persona == CustomerPersona.SALARY_DEPENDENT
                and self.state.environment_time >= 2):
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
            "policy_violations": self.state.policy_violations,
            "done_reason": done_reason,
            "persona": self.state.customer_persona.value,
            "bank_health": self.state.bank_health.value,
            "chaos_anomaly": self.state.chaos_anomaly.value,
        }

        return GymStepResult(
            next_state=self.state,
            reward=round(reward, 2),
            done=done,
            info=info,
        )

    def run_episode(self) -> dict:
        """Run a full episode: reset → step loop → done.

        Returns episode summary.
        """
        state = self.reset()
        total_reward = 0.0
        steps = 0
        trajectory: list[dict] = []

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
            "trajectory": trajectory,
        }

    def _agent_decide(self, state: GymState) -> ActionType:
        """Memory-aware agent decision logic for the Gym.

        Uses diagnosis, memory store, and decision layer.
        """
        case = state.case
        from recovery_agent.agent.diagnosis import run_diagnosis
        from recovery_agent.agent.decision import run_decision
        from recovery_agent.agent.memory import CustomerMemoryStore

        if case.diagnosis is None:
            case = run_diagnosis(case)

        if not hasattr(self, "memory_store"):
            self.memory_store = CustomerMemoryStore()

        profile = self.memory_store.get_or_create_profile(case.payment.customer_id)
        
        # Seed salary window for salary dependent persona if known
        if state.customer_persona.value == "salary_dependent":
            profile.salary_window.typical_pay_day = 1

        case = run_decision(case, profile=profile, memory=self.memory_store)

        action_value = case.payment.metadata.get("decided_action")
        if action_value:
            try:
                return ActionType(action_value)
            except ValueError:
                pass

        # Heuristic fallback based on failure type
        failure = case.payment.failure_code
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


def run_chaos_gym(episodes: int = 10, seed: int | None = None) -> dict:
    """Run multiple Gym episodes and aggregate results.

    Returns summary statistics.
    """
    env = RevenueLossEnvironment(seed=seed)
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
        "avg_steps": round(total_steps / episodes, 1) if episodes > 0 else 0.0,
        "avg_friction_index": round(avg_friction, 3),
        "by_persona": _aggregate_by_persona(results),
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
