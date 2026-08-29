"""Empirical Evolution — Strategy Metrics, Thompson Sampling Bandit, A/B Testing.

Mandate 4 adds data-driven strategy selection:
1. StrategyMetricsStore: SQLite-backed conversion rate tracking per (failure_type, action)
2. ThompsonBandit: Thompson Sampling for exploration/exploitation balance
3. ABTestFramework: Statistical A/B testing with chi-squared significance

Design inspired by:
- Stripe: strategy metrics inform routing decisions
- Redux: empirical conversion rates override LLM defaults
- Churnkey: A/B testing proves intervention effectiveness
"""
from __future__ import annotations

import math
import random
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from recovery_agent.models import ActionType, FailureType


# ═══════════════════════════════════════════════════════════════
#  STRATEGY METRICS STORE
# ═══════════════════════════════════════════════════════════════

@dataclass
class ArmMetrics:
    """Metrics for a single (failure_type, action) arm."""
    successes: int = 0
    attempts: int = 0
    last_updated: float = 0.0

    @property
    def conversion_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    @property
    def alpha(self) -> float:
        """Beta distribution alpha parameter (successes + 1 prior)."""
        return self.successes + 1.0

    @property
    def beta_param(self) -> float:
        """Beta distribution beta parameter (failures + 1 prior)."""
        return (self.attempts - self.successes) + 1.0


class StrategyMetricsStore:
    """SQLite-backed store for strategy conversion rates.

    Tracks (failure_type, action) → {successes, attempts} for empirical
    strategy selection. Thread-safe with connection-per-thread pattern.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path) if db_path else ":memory:"
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_metrics (
                failure_type TEXT NOT NULL,
                action TEXT NOT NULL,
                successes INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_updated REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (failure_type, action)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ab_assignments (
                case_id TEXT PRIMARY KEY,
                experiment TEXT NOT NULL,
                variant TEXT NOT NULL,
                assigned_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ab_outcomes (
                case_id TEXT PRIMARY KEY,
                experiment TEXT NOT NULL,
                variant TEXT NOT NULL,
                action_taken TEXT NOT NULL,
                recovered INTEGER NOT NULL DEFAULT 0,
                amount_recovered REAL NOT NULL DEFAULT 0.0,
                recorded_at REAL NOT NULL
            )
        """)
        conn.commit()

    def record_outcome(
        self,
        failure_type: FailureType,
        action: ActionType,
        recovered: bool,
    ) -> None:
        """Record a strategy outcome.

        Args:
            failure_type: The root cause of the payment failure.
            action: The action that was taken.
            recovered: Whether the payment was recovered.
        """
        conn = self._get_conn()
        ft = failure_type.value
        act = action.value
        now = time.time()

        if recovered:
            conn.execute(
                """INSERT INTO strategy_metrics (failure_type, action, successes, attempts, last_updated)
                   VALUES (?, ?, 1, 1, ?)
                   ON CONFLICT(failure_type, action) DO UPDATE SET
                     successes = successes + 1,
                     attempts = attempts + 1,
                     last_updated = ?""",
                (ft, act, now, now),
            )
        else:
            conn.execute(
                """INSERT INTO strategy_metrics (failure_type, action, successes, attempts, last_updated)
                   VALUES (?, ?, 0, 1, ?)
                   ON CONFLICT(failure_type, action) DO UPDATE SET
                     attempts = attempts + 1,
                     last_updated = ?""",
                (ft, act, now, now),
            )
        conn.commit()

    def get_arm(self, failure_type: FailureType, action: ActionType) -> ArmMetrics:
        """Get metrics for a specific (failure_type, action) arm."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT successes, attempts, last_updated FROM strategy_metrics WHERE failure_type = ? AND action = ?",
            (failure_type.value, action.value),
        ).fetchone()
        if row is None:
            return ArmMetrics()
        return ArmMetrics(
            successes=row["successes"],
            attempts=row["attempts"],
            last_updated=row["last_updated"],
        )

    def get_all_arms(self, failure_type: FailureType) -> dict[ActionType, ArmMetrics]:
        """Get all arms for a given failure type."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT action, successes, attempts, last_updated FROM strategy_metrics WHERE failure_type = ?",
            (failure_type.value,),
        ).fetchall()
        result = {}
        for row in rows:
            try:
                action = ActionType(row["action"])
            except ValueError:
                continue
            result[action] = ArmMetrics(
                successes=row["successes"],
                attempts=row["attempts"],
                last_updated=row["last_updated"],
            )
        return result

    def get_top_actions(
        self,
        failure_type: FailureType,
        min_attempts: int = 5,
    ) -> list[tuple[ActionType, float]]:
        """Get top actions by conversion rate for a failure type.

        Returns list of (action, conversion_rate) sorted by rate descending.
        Only includes arms with at least min_attempts.
        """
        arms = self.get_all_arms(failure_type)
        qualified = [
            (action, arm.conversion_rate)
            for action, arm in arms.items()
            if arm.attempts >= min_attempts
        ]
        qualified.sort(key=lambda x: x[1], reverse=True)
        return qualified

    def get_total_attempts(self) -> int:
        """Get total number of recorded attempts across all arms."""
        conn = self._get_conn()
        row = conn.execute("SELECT COALESCE(SUM(attempts), 0) as total FROM strategy_metrics").fetchone()
        return row["total"]

    def get_failure_type_stats(self) -> dict[str, dict]:
        """Get aggregate stats per failure type."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT failure_type,
                      SUM(successes) as total_successes,
                      SUM(attempts) as total_attempts
               FROM strategy_metrics GROUP BY failure_type"""
        ).fetchall()
        result = {}
        for row in rows:
            result[row["failure_type"]] = {
                "successes": row["total_successes"],
                "attempts": row["total_attempts"],
                "conversion_rate": (
                    row["total_successes"] / row["total_attempts"]
                    if row["total_attempts"] > 0
                    else 0.0
                ),
            }
        return result

    def clear(self) -> None:
        """Clear all metrics."""
        conn = self._get_conn()
        conn.execute("DELETE FROM strategy_metrics")
        conn.commit()

    def get_stats(self) -> dict:
        """Get store statistics."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as arms, COALESCE(SUM(attempts), 0) as total_attempts FROM strategy_metrics"
        ).fetchone()
        return {
            "db_path": self._db_path,
            "total_arms": row["arms"],
            "total_attempts": row["total_attempts"],
        }


# ═══════════════════════════════════════════════════════════════
#  THOMPSON SAMPLING BANDIT
# ═══════════════════════════════════════════════════════════════

class ThompsonBandit:
    """Thompson Sampling bandit for strategy selection.

    Uses Beta distribution sampling to balance exploration (trying new actions)
    vs exploitation (using known good actions).

    For each (failure_type, action) pair:
    - alpha = successes + 1 (prior: assume 1 success)
    - beta = failures + 1 (prior: assume 1 failure)
    - Sample from Beta(alpha, beta) for each arm
    - Select arm with highest sample

    Reference: Thompson, 1933 — "On the Likelihood that one Unknown Probability
    Exceeds Another in View of the Evidence of Two Samples"
    """

    def __init__(self, metrics: StrategyMetricsStore, min_samples: int = 10):
        """Initialize the bandit.

        Args:
            metrics: Strategy metrics store for historical data.
            min_samples: Minimum total samples before bandit overrides LLM.
                         Below this threshold, bandit defers to LLM/heuristic.
        """
        self._metrics = metrics
        self._min_samples = min_samples

    def select_action(
        self,
        failure_type: FailureType,
        eligible_actions: list[ActionType] | None = None,
    ) -> ActionType | None:
        """Select the best action using Thompson Sampling.

        Args:
            failure_type: The failure type to select an action for.
            eligible_actions: Actions to consider. If None, uses all 6 actions.

        Returns:
            Selected action, or None if insufficient data.
        """
        if eligible_actions is None:
            eligible_actions = list(ActionType)

        arms = self._metrics.get_all_arms(failure_type)

        # Check if we have enough total samples
        total_samples = sum(arm.attempts for arm in arms.values())
        if total_samples < self._min_samples:
            return None  # Not enough data — defer to LLM

        # Sample from Beta distribution for each eligible arm
        samples: dict[ActionType, float] = {}
        for action in eligible_actions:
            arm = arms.get(action, ArmMetrics())
            # Thompson Sampling: sample from Beta(alpha, beta)
            sample = random.betavariate(arm.alpha, arm.beta_param)
            samples[action] = sample

        # Select arm with highest sample
        best_action = max(samples, key=samples.get)  # type: ignore
        return best_action

    def get_confidence(
        self,
        failure_type: FailureType,
        action: ActionType,
    ) -> float:
        """Get confidence score for an action on a failure type.

        Returns the probability that this action is the best,
        estimated via Monte Carlo sampling.
        """
        arm = self._metrics.get_arm(failure_type, action)
        all_arms = self._metrics.get_all_arms(failure_type)

        if not all_arms:
            return 0.0

        total_samples = sum(a.attempts for a in all_arms.values())
        if total_samples < self._min_samples:
            return 0.0

        # Monte Carlo: sample 1000 times, count how often this action wins
        n_samples = 1000
        wins = 0
        for _ in range(n_samples):
            this_sample = random.betavariate(arm.alpha, arm.beta_param)
            best = this_sample
            for other_action, other_arm in all_arms.items():
                if other_action == action:
                    continue
                other_sample = random.betavariate(other_arm.alpha, other_arm.beta_param)
                if other_sample > best:
                    best = other_sample
                    break
            if this_sample >= best:
                wins += 1

        return wins / n_samples

    def get_empirical_context(
        self,
        failure_type: FailureType,
    ) -> str:
        """Generate empirical context string for LLM prompt injection.

        Returns a formatted string showing conversion rates for each action,
        helping the LLM make data-informed decisions.
        """
        arms = self._metrics.get_all_arms(failure_type)
        if not arms:
            return ""

        total = sum(arm.attempts for arm in arms.values())
        if total < 5:
            return ""  # Not enough data to be informative

        lines = [f"Historical performance for {failure_type.value} (n={total}):"]
        sorted_arms = sorted(arms.items(), key=lambda x: x[1].conversion_rate, reverse=True)

        for action, arm in sorted_arms:
            if arm.attempts == 0:
                continue
            rate = arm.conversion_rate
            bandit_conf = self.get_confidence(failure_type, action)
            lines.append(
                f"  - {action.value}: {rate:.0%} conversion "
                f"({arm.successes}/{arm.attempts} attempts, "
                f"bandit confidence: {bandit_conf:.0%})"
            )

        # Add bandit recommendation
        recommended = self.select_action(failure_type)
        if recommended:
            lines.append(f"  → Bandit recommends: {recommended.value}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  A/B TEST FRAMEWORK
# ═══════════════════════════════════════════════════════════════

@dataclass
class ABTestResult:
    """Result of an A/B test comparison."""
    experiment: str
    control_name: str
    treatment_name: str
    control_attempts: int = 0
    control_recoveries: int = 0
    treatment_attempts: int = 0
    treatment_recoveries: int = 0

    @property
    def control_rate(self) -> float:
        if self.control_attempts == 0:
            return 0.0
        return self.control_recoveries / self.control_attempts

    @property
    def treatment_rate(self) -> float:
        if self.treatment_attempts == 0:
            return 0.0
        return self.treatment_recoveries / self.treatment_attempts

    @property
    def lift(self) -> float:
        """Relative improvement of treatment over control."""
        if self.control_rate == 0:
            return float("inf") if self.treatment_rate > 0 else 0.0
        return (self.treatment_rate - self.control_rate) / self.control_rate

    @property
    def p_value(self) -> float:
        """Two-proportion z-test p-value."""
        n1 = self.control_attempts
        n2 = self.treatment_attempts
        if n1 == 0 or n2 == 0:
            return 1.0

        p1 = self.control_recoveries / n1
        p2 = self.treatment_recoveries / n2

        # Pooled proportion
        p_pool = (self.control_recoveries + self.treatment_recoveries) / (n1 + n2)
        if p_pool == 0 or p_pool == 1:
            return 1.0

        se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        if se == 0:
            return 1.0

        z = (p2 - p1) / se
        return 2 * (1 - _normal_cdf(abs(z)))

    @property
    def is_significant(self) -> bool:
        """Is the result statistically significant at p < 0.05?"""
        return self.p_value < 0.05

    @property
    def winner(self) -> str:
        """Which variant is winning (or 'none' if not significant)?"""
        if not self.is_significant:
            return "none"
        if self.treatment_rate > self.control_rate:
            return self.treatment_name
        return self.control_name


class ABTestFramework:
    """A/B testing framework for strategy comparison.

    Tracks which strategy variant was assigned to each case,
    records outcomes, and computes statistical significance.
    """

    def __init__(self, metrics: StrategyMetricsStore):
        self._metrics = metrics

    def assign_variant(
        self,
        case_id: str,
        experiment: str,
        variants: list[str] | None = None,
    ) -> str:
        """Randomly assign a case to a variant.

        Args:
            case_id: Unique case identifier.
            experiment: Experiment name (e.g., 'bandit_vs_llm').
            variants: List of variant names. Default: ['control', 'treatment'].

        Returns:
            Assigned variant name.
        """
        if variants is None:
            variants = ["control", "treatment"]

        variant = random.choice(variants)
        conn = self._metrics._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO ab_assignments (case_id, experiment, variant, assigned_at)
               VALUES (?, ?, ?, ?)""",
            (case_id, experiment, variant, time.time()),
        )
        conn.commit()
        return variant

    def get_variant(self, case_id: str, experiment: str) -> str | None:
        """Get the assigned variant for a case."""
        conn = self._metrics._get_conn()
        row = conn.execute(
            "SELECT variant FROM ab_assignments WHERE case_id = ? AND experiment = ?",
            (case_id, experiment),
        ).fetchone()
        return row["variant"] if row else None

    def record_outcome(
        self,
        case_id: str,
        experiment: str,
        action_taken: ActionType,
        recovered: bool,
        amount_recovered: float = 0.0,
    ) -> None:
        """Record the outcome for an A/B test case."""
        variant = self.get_variant(case_id, experiment)
        if variant is None:
            return  # Case not part of this experiment

        conn = self._metrics._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO ab_outcomes
               (case_id, experiment, variant, action_taken, recovered, amount_recovered, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (case_id, experiment, variant, action_taken.value, int(recovered), amount_recovered, time.time()),
        )
        conn.commit()

    def compute_result(self, experiment: str) -> ABTestResult:
        """Compute A/B test result for an experiment.

        Assumes control = 'control' variant, treatment = 'treatment' variant.
        """
        conn = self._metrics._get_conn()

        control = conn.execute(
            """SELECT COUNT(*) as attempts, COALESCE(SUM(recovered), 0) as recoveries
               FROM ab_outcomes WHERE experiment = ? AND variant = 'control'""",
            (experiment,),
        ).fetchone()

        treatment = conn.execute(
            """SELECT COUNT(*) as attempts, COALESCE(SUM(recovered), 0) as recoveries
               FROM ab_outcomes WHERE experiment = ? AND variant = 'treatment'""",
            (experiment,),
        ).fetchone()

        return ABTestResult(
            experiment=experiment,
            control_name="control",
            treatment_name="treatment",
            control_attempts=control["attempts"],
            control_recoveries=control["recoveries"],
            treatment_attempts=treatment["attempts"],
            treatment_recoveries=treatment["recoveries"],
        )

    def get_all_experiments(self) -> list[str]:
        """Get all experiment names with recorded outcomes."""
        conn = self._metrics._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT experiment FROM ab_outcomes"
        ).fetchall()
        return [row["experiment"] for row in rows]


# ═══════════════════════════════════════════════════════════════
#  MATH UTILITIES
# ═══════════════════════════════════════════════════════════════

def _normal_cdf(x: float) -> float:
    """Approximation of the standard normal CDF.

    Uses the error function approximation (Abramowitz & Stegun).
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
