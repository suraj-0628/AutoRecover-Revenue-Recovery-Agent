"""Trajectory Benchmarking Engine — computes step efficiency, friction, and quality scores.

Evaluates recovery trajectories across episodes to measure agent performance
beyond simple recovery rate.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrajectoryMetrics(BaseModel):
    """Metrics computed from a single trajectory."""
    step_efficiency: float = Field(ge=0.0, le=1.0, description="Ratio of minimal to actual steps")
    friction_score: float = Field(ge=0.0, le=1.0, description="Weighted invasive communication penalty")
    policy_compliance_rate: float = Field(ge=0.0, le=1.0, description="% of steps passing guardrails")
    trajectory_score: float = Field(ge=0.0, le=1.0, description="Aggregate quality score")
    total_steps: int = 0
    guardrail_modifications: int = 0
    guardrail_blocks: int = 0
    invasive_steps: int = 0
    recovered: bool = False
    recovered_at_step: int = 0


# Action weights for friction calculation
FRICTION_WEIGHTS: dict[str, float] = {
    "retry_payment": 0.0,       # No friction — standard retry
    "wait_and_retry": 0.0,      # No friction — passive wait
    "send_notification": 0.3,   # Moderate friction — customer contact
    "update_payment_method": 0.5,  # High friction — requires customer action
    "escalate_to_human": 0.8,   # Very high friction — human intervention
    "abandon": 1.0,             # Maximum friction — case dropped
}

# Actions considered "invasive" (contacting the customer)
INVASIVE_ACTIONS = {"send_notification", "update_payment_method"}

# Minimum steps needed for a successful recovery (best case)
MIN_STEPS_FOR_RECOVERY = 2  # diagnose + execute successfully


class TrajectoryBenchmark:
    """Computes trajectory quality metrics across episodes."""

    def __init__(self) -> None:
        self.trajectories: list[dict] = []

    def evaluate_trajectory(self, trajectory: list[dict]) -> TrajectoryMetrics:
        """Evaluate a single trajectory and return metrics.

        Args:
            trajectory: List of step dicts with keys: action, verdict, guardrail_checks, etc.
        """
        if not trajectory:
            return TrajectoryMetrics(
                step_efficiency=1.0,
                friction_score=0.0,
                policy_compliance_rate=1.0,
                trajectory_score=1.0,
                total_steps=0,
            )

        total_steps = len(trajectory)

        # --- Step Efficiency ---
        # Find if recovery happened and at which step
        recovered = False
        recovered_at = 0
        for i, step in enumerate(trajectory):
            action = step.get("action", "")
            # Recovery happens on retry_payment or when result indicates success
            if action == "retry_payment" and step.get("result") == "success":
                recovered = True
                recovered_at = i + 1
                break

        if recovered:
            # Efficiency = min_steps / actual_steps (capped at 1.0)
            step_efficiency = min(1.0, MIN_STEPS_FOR_RECOVERY / recovered_at)
        else:
            # If not recovered, efficiency is inverse of steps (fewer wasted steps = better)
            step_efficiency = max(0.0, 1.0 - (total_steps / 10.0))

        # --- Friction Score ---
        friction_sum = 0.0
        invasive_count = 0
        for step in trajectory:
            action = step.get("action", "unknown")
            weight = FRICTION_WEIGHTS.get(action, 0.5)
            friction_sum += weight
            if action in INVASIVE_ACTIONS:
                invasive_count += 1

        # Normalize friction to 0-1 (0 = no friction, 1 = max friction)
        friction_score = min(1.0, friction_sum / max(total_steps, 1))

        # --- Policy Compliance Rate ---
        compliant_steps = 0
        modifications = 0
        blocks = 0
        for step in trajectory:
            verdict = step.get("verdict", "pass")
            if verdict == "pass":
                compliant_steps += 1
            elif verdict == "modified":
                compliant_steps += 1  # Modified is still compliant
                modifications += 1
            elif verdict == "blocked":
                blocks += 1

        policy_compliance_rate = compliant_steps / max(total_steps, 1)

        # --- Aggregate Trajectory Score ---
        # Weighted combination of metrics
        trajectory_score = (
            0.35 * step_efficiency
            + 0.25 * (1.0 - friction_score)  # Lower friction = better
            + 0.25 * policy_compliance_rate
            + 0.15 * (1.0 if recovered else 0.0)
        )

        return TrajectoryMetrics(
            step_efficiency=round(step_efficiency, 3),
            friction_score=round(friction_score, 3),
            policy_compliance_rate=round(policy_compliance_rate, 3),
            trajectory_score=round(trajectory_score, 3),
            total_steps=total_steps,
            guardrail_modifications=modifications,
            guardrail_blocks=blocks,
            invasive_steps=invasive_count,
            recovered=recovered,
            recovered_at_step=recovered_at,
        )

    def evaluate_episode(self, episode_result: dict) -> TrajectoryMetrics:
        """Evaluate a full episode result (from chaos gym)."""
        trajectory = episode_result.get("trajectory", [])
        # Extract relevant fields from gym trajectory
        cleaned = []
        for step in trajectory:
            cleaned.append({
                "action": step.get("action", "unknown"),
                "verdict": step.get("info", {}).get("verdict", "pass"),
                "result": "success" if episode_result.get("recovered") and step == trajectory[-1] else "pending",
                "guardrail_checks": step.get("info", {}).get("guardrail_checks", 0),
            })
        return self.evaluate_trajectory(cleaned)

    def aggregate_metrics(self, metrics_list: list[TrajectoryMetrics]) -> dict:
        """Aggregate metrics across multiple trajectories."""
        if not metrics_list:
            return {
                "avg_step_efficiency": 0.0,
                "avg_friction_score": 0.0,
                "avg_policy_compliance": 0.0,
                "avg_trajectory_score": 0.0,
                "recovery_rate": 0.0,
                "total_episodes": 0,
            }

        n = len(metrics_list)
        return {
            "avg_step_efficiency": round(sum(m.step_efficiency for m in metrics_list) / n, 3),
            "avg_friction_score": round(sum(m.friction_score for m in metrics_list) / n, 3),
            "avg_policy_compliance": round(sum(m.policy_compliance_rate for m in metrics_list) / n, 3),
            "avg_trajectory_score": round(sum(m.trajectory_score for m in metrics_list) / n, 3),
            "recovery_rate": round(sum(1 for m in metrics_list if m.recovered) / n, 3),
            "total_episodes": n,
        }
