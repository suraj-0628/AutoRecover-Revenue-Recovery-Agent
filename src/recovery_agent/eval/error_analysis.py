"""Error Analysis & Prioritization — using arize-phoenix-evals (real SDK).

Source: arize-phoenix-evals evaluate_dataframe pattern
        Error analysis from Evaluating AI Agents (Arize AI)

Categorizes errors by type and prioritizes fixes based on impact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from phoenix.evals import LLM, create_evaluator, evaluate_dataframe

from recovery_agent.models import ActionType, Case, CaseStatus, FailureType


# ═══════════════════════════════════════════════════════════════
# ERROR CATEGORIES
# ═══════════════════════════════════════════════════════════════

class ErrorCategory:
    """Error categories for systematic classification."""
    DIAGNOSIS_WRONG = "diagnosis_wrong"
    TOOL_WRONG = "tool_wrong"
    TIMING_WRONG = "timing_wrong"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    CUSTOMER_NO_RESPONSE = "customer_no_response"
    HARD_DECLINE_RETRY = "hard_decline_retry"
    MAX_ATTEMPTS = "max_attempts"
    ESCALATION_NEEDED = "escalation_needed"
    UNKNOWN = "unknown"


@dataclass
class ErrorEntry:
    """A single error occurrence."""
    error_id: str
    category: str
    timestamp: datetime
    payment_id: str
    failure_type: str
    action_taken: str
    expected_action: Optional[str]
    root_cause: str
    confidence: float
    attempt_count: int
    recovery_tier: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "error_id": self.error_id,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "payment_id": self.payment_id,
            "failure_type": self.failure_type,
            "action_taken": self.action_taken,
            "expected_action": self.expected_action,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "attempt_count": self.attempt_count,
            "recovery_tier": self.recovery_tier,
        }


# ═══════════════════════════════════════════════════════════════
# PHOENIX EVALUATORS — real arize-phoenix-evals
# ═══════════════════════════════════════════════════════════════

def _get_llm() -> LLM:
    """Get LLM for evaluation (uses same config as agent)."""
    import os
    from dotenv import load_dotenv
    load_dotenv()

    return LLM(
        provider="openai",
        model=os.getenv("LLM_MODEL", "antigravity/gemini-2.5-flash"),
        api_key=os.getenv("LLM_API_KEY", "dummy"),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:20128/v1"),
    )


@create_evaluator(name="error_category")
def error_category_evaluator(input: dict, output: dict) -> str:
    """Classify the error category based on input/output."""
    failure_type = input.get("failure_type", "unknown")
    action_taken = output.get("action_taken", "unknown")
    expected_action = output.get("expected_action", "")
    confidence = output.get("confidence", 0.0)
    attempt_count = output.get("attempt_count", 0)
    max_attempts = output.get("max_attempts", 3)

    # Hard decline retry detection
    from recovery_agent.models import HARD_DECLINES
    failure_code = input.get("failure_code", "")
    if failure_code in HARD_DECLINES:
        if action_taken in ("retry_payment", "update_payment_method"):
            return ErrorCategory.HARD_DECLINE_RETRY

    # Max attempts reached
    if attempt_count >= max_attempts:
        return ErrorCategory.MAX_ATTEMPTS

    # Customer didn't respond
    if output.get("customer_responded") is False:
        return ErrorCategory.CUSTOMER_NO_RESPONSE

    # Guardrail blocked
    if output.get("guardrail_blocked"):
        return ErrorCategory.GUARDRAIL_BLOCKED

    # Low confidence diagnosis
    if confidence < 0.5:
        return ErrorCategory.DIAGNOSIS_WRONG

    # Wrong tool selected
    if expected_action and action_taken != expected_action:
        return ErrorCategory.TOOL_WRONG

    return ErrorCategory.UNKNOWN


@create_evaluator(name="error_impact")
def error_impact_evaluator(input: dict, output: dict) -> float:
    """Score error impact (0-1, higher = more impactful)."""
    amount = input.get("amount", 0)
    attempt_count = output.get("attempt_count", 0)
    category = output.get("error_category", ErrorCategory.UNKNOWN)

    # Base impact from amount
    amount_impact = min(amount / 10000, 1.0)

    # Category multiplier
    category_multipliers = {
        ErrorCategory.HARD_DECLINE_RETRY: 1.5,  # Network penalties
        ErrorCategory.TOOL_WRONG: 1.2,
        ErrorCategory.DIAGNOSIS_WRONG: 1.1,
        ErrorCategory.GUARDRAIL_BLOCKED: 0.8,
        ErrorCategory.CUSTOMER_NO_RESPONSE: 0.6,
        ErrorCategory.MAX_ATTEMPTS: 1.0,
        ErrorCategory.ESCALATION_NEEDED: 0.9,
    }
    multiplier = category_multipliers.get(category, 1.0)

    return min(amount_impact * multiplier, 1.0)


# ═══════════════════════════════════════════════════════════════
# ERROR CLASSIFIER — deterministic (no LLM needed)
# ═══════════════════════════════════════════════════════════════

def classify_error(case: Case, action_taken: ActionType) -> str:
    """Classify the error category for a failed recovery attempt."""
    from recovery_agent.models import HARD_DECLINES

    if case.status == CaseStatus.STOPPED and case.attempt_count >= case.max_attempts:
        if case.payment.metadata.get("customer_responded") is False:
            return ErrorCategory.CUSTOMER_NO_RESPONSE
        return ErrorCategory.MAX_ATTEMPTS

    if case.status == CaseStatus.ESCALATED:
        return ErrorCategory.ESCALATION_NEEDED

    if case.payment.failure_code in HARD_DECLINES:
        if action_taken in (ActionType.RETRY_PAYMENT, ActionType.UPDATE_PAYMENT_METHOD):
            return ErrorCategory.HARD_DECLINE_RETRY

    if case.payment.metadata.get("guardrail_blocked"):
        return ErrorCategory.GUARDRAIL_BLOCKED

    if case.diagnosis and case.diagnosis.confidence < 0.5:
        return ErrorCategory.DIAGNOSIS_WRONG

    expected_action = case.payment.metadata.get("expected_action")
    if expected_action and action_taken.value != expected_action:
        return ErrorCategory.TOOL_WRONG

    if case.payment.metadata.get("timing_issue"):
        return ErrorCategory.TIMING_WRONG

    return ErrorCategory.UNKNOWN


# ═══════════════════════════════════════════════════════════════
# BATCH ANALYSIS — uses phoenix evaluate_dataframe
# ═══════════════════════════════════════════════════════════════

def build_error_dataframe(cases: list[Case]) -> pd.DataFrame:
    """Build a DataFrame from failed cases for evaluation."""
    rows = []
    for case in cases:
        if case.recovered:
            continue

        action_taken = case.payment.metadata.get("action_taken", "unknown")
        rows.append({
            "payment_id": case.payment.payment_id,
            "failure_type": case.payment.failure_type.value if case.payment.failure_type else "unknown",
            "failure_code": case.payment.failure_code or "",
            "amount": case.payment.amount,
            "action_taken": action_taken,
            "expected_action": case.payment.metadata.get("expected_action", ""),
            "confidence": case.diagnosis.confidence if case.diagnosis else 0.0,
            "attempt_count": case.attempt_count,
            "max_attempts": case.max_attempts,
            "recovery_tier": case.recovery_tier.value,
            "customer_responded": case.payment.metadata.get("customer_responded"),
            "guardrail_blocked": case.payment.metadata.get("guardrail_blocked", False),
        })
    return pd.DataFrame(rows)


def analyze_errors_with_phoenix(cases: list[Case]) -> pd.DataFrame:
    """Analyze errors using arize-phoenix-evals.

    Runs LLM-based error categorization and impact scoring on failed cases.
    """
    df = build_error_dataframe(cases)

    if df.empty:
        return df

    # Run phoenix evaluators
    results_df = evaluate_dataframe(
        dataframe=df,
        evaluators=[error_category_evaluator, error_impact_evaluator],
    )

    return results_df


def get_error_summary(results_df: pd.DataFrame) -> str:
    """Generate a summary from phoenix evaluation results."""
    if results_df.empty:
        return "No errors to analyze."

    lines = [
        "=" * 60,
        "ERROR ANALYSIS REPORT (arize-phoenix-evals)",
        "=" * 60,
        f"Total failed cases: {len(results_df)}",
        "",
    ]

    # Category distribution
    if "error_category_score" in results_df.columns:
        lines.append("ERRORS BY CATEGORY:")
        category_counts = results_df["error_category_score"].value_counts()
        for category, count in category_counts.items():
            pct = count / len(results_df) * 100
            lines.append(f"  {category}: {count} ({pct:.1f}%)")

    # Impact distribution
    if "error_impact_score" in results_df.columns:
        avg_impact = results_df["error_impact_score"].mean()
        lines.append(f"\nAverage Impact Score: {avg_impact:.2f}")

        high_impact = results_df[results_df["error_impact_score"] > 0.7]
        if not high_impact.empty:
            lines.append(f"High Impact Cases (>0.7): {len(high_impact)}")

    # By failure type
    if "failure_type" in results_df.columns:
        lines.append("\nERRORS BY FAILURE TYPE:")
        for ftype, count in results_df["failure_type"].value_counts().items():
            lines.append(f"  {ftype}: {count}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════

ERROR_LOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "error_analysis.jsonl"


def save_error_analysis(results_df: pd.DataFrame, path: Path | None = None):
    """Save error analysis to JSONL file."""
    filepath = path or ERROR_LOG_PATH
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if results_df.empty:
        return

    results_df.to_json(filepath, orient="records", lines=True)


def load_error_analysis(path: Path | None = None) -> pd.DataFrame:
    """Load historical error entries."""
    filepath = path or ERROR_LOG_PATH
    if not filepath.exists():
        return pd.DataFrame()

    return pd.read_json(filepath, orient="records", lines=True)


if __name__ == "__main__":
    print("Error Analysis module (arize-phoenix-evals).")
    print("Use analyze_errors_with_phoenix() to analyze cases.")
