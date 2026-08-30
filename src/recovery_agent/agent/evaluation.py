"""Batch evaluation — runs multiple cases and measures recovery.

Source: Evaluation framework from Evaluating AI Agents (Arize AI)
        Gold-standard datasets from NeMo Agent Toolkit

FLAW-40: Paired comparison testing (baseline vs current)
FLAW-41: A/B test framework integration
FLAW-46: Regression detection
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from recovery_agent.agent import RecoveryAgent
from recovery_agent.models import Case, CaseStatus, PaymentEvent

BATCH_CSV_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "batch_transactions.csv"
EVAL_HISTORY_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "eval_history.jsonl"

# failure_code -> recoverable mapping
_RECOVERABLE_MAP: dict[str, bool] = {
    "insufficient_funds": True,
    "card_expired": True,
    "risk_block": False,
}


@dataclass
class BatchResult:
    """Results from a batch evaluation run."""
    total_cases: int = 0
    recovered: int = 0
    stopped: int = 0
    escalated: int = 0
    total_amount: float = 0.0
    recovered_amount: float = 0.0
    avg_attempts: float = 0.0
    total_attempts: int = 0
    cases: list[dict] = field(default_factory=list)
    by_failure_type: dict[str, dict] = field(default_factory=dict)

    @property
    def recovery_rate(self) -> float:
        return self.recovered / self.total_cases if self.total_cases > 0 else 0.0

    @property
    def recovery_amount_rate(self) -> float:
        return self.recovered_amount / self.total_amount if self.total_amount > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "BATCH EVALUATION RESULTS",
            "=" * 60,
            f"Total cases:     {self.total_cases}",
            f"Recovered:       {self.recovered} ({self.recovery_rate:.1%})",
            f"Stopped:         {self.stopped}",
            f"Escalated:       {self.escalated}",
            f"Total amount:    INR {self.total_amount:,.2f}",
            f"Recovered amount: INR {self.recovered_amount:,.2f} ({self.recovery_amount_rate:.1%})",
            f"Avg attempts:    {self.avg_attempts:.1f}",
            "-" * 60,
            "BY FAILURE TYPE:",
        ]
        for ftype, stats in self.by_failure_type.items():
            lines.append(
                f"  {ftype}: {stats['recovered']}/{stats['total']} recovered "
                f"({stats['rate']:.0%})"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def _load_batch_from_csv(csv_path: Path | None = None) -> list[PaymentEvent]:
    """Load payment events from a static CSV file."""
    path = csv_path or BATCH_CSV_PATH
    events: list[PaymentEvent] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(PaymentEvent(
                payment_id=row["payment_id"],
                customer_id=row["customer_id"],
                amount=float(row["amount"]),
                currency=row["currency"],
                failure_code=row["failure_code"],
                failure_reason=row["failure_reason"],
            ))
    return events


def _get_known_outcomes(events: list[PaymentEvent]) -> dict[str, dict]:
    """Derive known properties for each event from failure_code."""
    outcomes: dict[str, dict] = {}
    for event in events:
        outcomes[event.payment_id] = {
            "failure_type": event.failure_code,
            "recoverable": _RECOVERABLE_MAP.get(event.failure_code, False),
            "amount": event.amount,
        }
    return outcomes


def run_batch_evaluation(
    num_cases: int = 30,
    seed: int | None = None,
    csv_path: Path | None = None,
) -> BatchResult:
    """Run a batch of cases through the recovery agent and measure results.

    Reads from data/batch_transactions.csv by default. The seed parameter is
    kept for backward compatibility but is no longer used (static dataset).
    """
    events = _load_batch_from_csv(csv_path)
    known_outcomes = _get_known_outcomes(events)

    agent = RecoveryAgent()
    result = BatchResult()

    for event in events:
        case = Case(payment=event)
        final_case = agent.run(case)

        case_outcome = {
            "case_id": final_case.id,
            "payment_id": final_case.payment.payment_id,
            "failure_type": known_outcomes[event.payment_id]["failure_type"],
            "recoverable": known_outcomes[event.payment_id]["recoverable"],
            "status": final_case.status.value,
            "recovered": final_case.recovered,
            "recovered_amount": final_case.recovered_amount,
            "attempts": final_case.attempt_count,
            "amount": final_case.payment.amount,
        }
        result.cases.append(case_outcome)

        result.total_cases += 1
        result.total_amount += final_case.payment.amount
        result.total_attempts += final_case.attempt_count

        if final_case.recovered:
            result.recovered += 1
            result.recovered_amount += final_case.recovered_amount
        elif final_case.status == CaseStatus.STOPPED:
            result.stopped += 1
        elif final_case.status == CaseStatus.ESCALATED:
            result.escalated += 1

        ftype = known_outcomes[event.payment_id]["failure_type"]
        if ftype not in result.by_failure_type:
            result.by_failure_type[ftype] = {"total": 0, "recovered": 0, "rate": 0.0}
        result.by_failure_type[ftype]["total"] += 1
        if final_case.recovered:
            result.by_failure_type[ftype]["recovered"] += 1

    result.avg_attempts = result.total_attempts / result.total_cases if result.total_cases > 0 else 0
    for stats in result.by_failure_type.values():
        stats["rate"] = stats["recovered"] / stats["total"] if stats["total"] > 0 else 0.0

    return result


# --- FLAW-40: Paired Comparison Testing ---

@dataclass
class PairedComparisonResult:
    """Results from comparing baseline vs current agent."""
    baseline_recovery_rate: float = 0.0
    current_recovery_rate: float = 0.0
    improvement: float = 0.0
    improvement_pct: float = 0.0
    baseline_recovered_amount: float = 0.0
    current_recovered_amount: float = 0.0
    total_cases: int = 0
    cases_improved: int = 0
    cases_regressed: int = 0
    cases_unchanged: int = 0
    paired_details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "PAIRED COMPARISON: Baseline vs Current",
            "=" * 60,
            f"Total cases:     {self.total_cases}",
            f"Baseline rate:   {self.baseline_recovery_rate:.1%}",
            f"Current rate:    {self.current_recovery_rate:.1%}",
            f"Improvement:     {self.improvement_pct:+.1%} ({self.improvement:+.1%})",
            f"Amount delta:    INR {self.current_recovered_amount - self.baseline_recovered_amount:+,.2f}",
            f"Cases improved:  {self.cases_improved}",
            f"Cases regressed: {self.cases_regressed}",
            f"Cases unchanged: {self.cases_unchanged}",
            "=" * 60,
        ]
        return "\n".join(lines)


def run_paired_comparison(
    csv_path: Path | None = None,
    num_cases: int = 30,
) -> PairedComparisonResult:
    """Run identical cases through current agent and compare against historical baseline.

    FLAW-40: Enables empirical proof of improvement. "Before: 73%, After: 85%".
    """
    events = _load_batch_from_csv(csv_path)[:num_cases]
    known_outcomes = _get_known_outcomes(events)

    # Load or generate baseline
    baseline = _load_baseline()
    if not baseline:
        # No baseline exists — run current agent as baseline
        baseline = _run_agent_on_events(events, known_outcomes, label="baseline")
        _save_baseline(baseline)

    # Run current agent
    current = _run_agent_on_events(events, known_outcomes, label="current")

    # Compare
    result = PairedComparisonResult()
    result.total_cases = len(events)
    result.baseline_recovery_rate = baseline["recovery_rate"]
    result.current_recovery_rate = current["recovery_rate"]
    result.improvement = result.current_recovery_rate - result.baseline_recovery_rate
    result.improvement_pct = result.improvement / max(result.baseline_recovery_rate, 0.001)
    result.baseline_recovered_amount = baseline["recovered_amount"]
    result.current_recovered_amount = current["recovered_amount"]

    # Case-level comparison
    baseline_by_pid = {c["payment_id"]: c for c in baseline["cases"]}
    current_by_pid = {c["payment_id"]: c for c in current["cases"]}
    for pid in baseline_by_pid:
        if pid in current_by_pid:
            b = baseline_by_pid[pid]
            c = current_by_pid[pid]
            if c["recovered"] and not b["recovered"]:
                result.cases_improved += 1
            elif not c["recovered"] and b["recovered"]:
                result.cases_regressed += 1
            else:
                result.cases_unchanged += 1
            result.paired_details.append({
                "payment_id": pid,
                "baseline_recovered": b["recovered"],
                "current_recovered": c["recovered"],
                "baseline_attempts": b["attempts"],
                "current_attempts": c["attempts"],
            })

    return result


def _run_agent_on_events(
    events: list[PaymentEvent],
    known_outcomes: dict[str, dict],
    label: str = "run",
) -> dict:
    """Run agent on a list of events and return summary dict."""
    agent = RecoveryAgent()
    cases = []
    total_amount = 0.0
    recovered_amount = 0.0
    recovered_count = 0

    for event in events:
        case = Case(payment=event)
        final_case = agent.run(case)
        case_outcome = {
            "payment_id": final_case.payment.payment_id,
            "recovered": final_case.recovered,
            "recovered_amount": final_case.recovered_amount,
            "attempts": final_case.attempt_count,
            "amount": final_case.payment.amount,
        }
        cases.append(case_outcome)
        total_amount += final_case.payment.amount
        if final_case.recovered:
            recovered_count += 1
            recovered_amount += final_case.recovered_amount

    return {
        "label": label,
        "total_cases": len(events),
        "recovered": recovered_count,
        "recovery_rate": recovered_count / len(events) if events else 0,
        "total_amount": total_amount,
        "recovered_amount": recovered_amount,
        "cases": cases,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _load_baseline() -> dict | None:
    """Load baseline evaluation results from disk."""
    baseline_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "eval_baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            return json.load(f)
    return None


def _save_baseline(baseline: dict) -> None:
    """Save baseline evaluation results to disk."""
    baseline_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "eval_baseline.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)


def _save_eval_run(result: BatchResult) -> None:
    """Append evaluation run to history for regression detection."""
    EVAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recovery_rate": result.recovery_rate,
        "recovered_amount": result.recovered_amount,
        "total_cases": result.total_cases,
        "recovered": result.recovered,
        "avg_attempts": result.avg_attempts,
    }
    with open(EVAL_HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# --- FLAW-43: Statistical Significance Testing ---

def two_proportion_z_test(
    recovered_a: int, total_a: int,
    recovered_b: int, total_b: int,
) -> dict:
    """Two-proportion z-test for comparing recovery rates.

    FLAW-43: Enables "improvement is statistically significant (p < 0.05)" claims.

    Args:
        recovered_a, total_a: Baseline results
        recovered_b, total_b: Current results

    Returns:
        dict with z_stat, p_value, significant (at 0.05 level)
    """
    if total_a == 0 or total_b == 0:
        return {"z_stat": 0.0, "p_value": 1.0, "significant": False, "error": "zero samples"}

    p_a = recovered_a / total_a
    p_b = recovered_b / total_b
    p_pool = (recovered_a + recovered_b) / (total_a + total_b)

    se = math.sqrt(p_pool * (1 - p_pool) * (1/total_a + 1/total_b)) if p_pool > 0 and p_pool < 1 else 0.001
    z = (p_b - p_a) / se if se > 0 else 0.0

    # Approximate two-tailed p-value from z-score
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    return {
        "z_stat": round(z, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "baseline_rate": round(p_a, 4),
        "current_rate": round(p_b, 4),
        "difference": round(p_b - p_a, 4),
    }


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def compute_confidence_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> dict:
    """Compute Wilson score confidence interval for a proportion.

    FLAW-43: Provides confidence intervals for recovery rate claims.
    """
    if total == 0:
        return {"lower": 0.0, "upper": 0.0, "point_estimate": 0.0}

    z = 1.96 if confidence == 0.95 else 1.645  # 95% or 90%
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    return {
        "lower": round(max(0, center - spread), 4),
        "upper": round(min(1, center + spread), 4),
        "point_estimate": round(p, 4),
        "confidence": confidence,
    }


# --- FLAW-46: Regression Detection ---

@dataclass
class RegressionResult:
    """Result of comparing current run against historical baseline."""
    is_regression: bool = False
    metric: str = "recovery_rate"
    baseline_value: float = 0.0
    current_value: float = 0.0
    change: float = 0.0
    threshold: float = 0.05  # 5% drop triggers regression alert
    historical_runs: int = 0
    reason: str = ""

    def summary(self) -> str:
        status = "REGRESSION DETECTED" if self.is_regression else "No regression"
        return (
            f"[{status}] {self.metric}: {self.baseline_value:.1%} → {self.current_value:.1%} "
            f"(change: {self.change:+.1%}, threshold: -{self.threshold:.1%}) "
            f"based on {self.historical_runs} historical runs"
        )


def detect_regression(
    current_result: BatchResult,
    threshold: float = 0.05,
) -> RegressionResult:
    """Compare current batch result against historical runs to detect regressions.

    FLAW-46: Catches regressions before they reach production.
    """
    result = RegressionResult()
    result.metric = "recovery_rate"
    result.current_value = current_result.recovery_rate

    # Load historical runs
    history = _load_eval_history()
    if len(history) < 2:
        result.reason = f"Insufficient history ({len(history)} runs). Need at least 2."
        return result

    # Use median of last 5 runs as baseline
    recent_rates = [h["recovery_rate"] for h in history[-5:]]
    baseline_rate = sorted(recent_rates)[len(recent_rates) // 2]  # median
    result.baseline_value = baseline_rate
    result.change = result.current_value - baseline_rate
    result.historical_runs = len(history)
    result.threshold = threshold

    # Check for regression
    if result.change < -threshold:
        result.is_regression = True
        result.reason = (
            f"Recovery rate dropped by {abs(result.change):.1%} "
            f"(from {baseline_rate:.1%} to {result.current_value:.1%}). "
            f"This exceeds the {threshold:.1%} threshold."
        )
    else:
        result.reason = f"Recovery rate within acceptable range ({result.change:+.1%})."

    # Save current run to history
    _save_eval_run(current_result)

    return result


def _load_eval_history() -> list[dict]:
    """Load evaluation history from disk."""
    if not EVAL_HISTORY_PATH.exists():
        return []
    history = []
    with open(EVAL_HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return history
