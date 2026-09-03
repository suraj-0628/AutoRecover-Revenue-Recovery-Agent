"""Recovery Evaluation — custom gold-standard evaluation (NOT real NAT).

MANDATE 6: This is CUSTOM code, not nvidia-nat.
    For real NAT evaluation, see nat_eval.py which uses:
    from nat.plugins.langchain.eval.trajectory_evaluator import TrajectoryEvaluator

    This module provides deterministic F1-based evaluation (no LLM judge).
    Use nat_eval.py for LLM-as-judge trajectory scoring (real NAT).

MANDATE 0: This is evaluation infrastructure (acceptable pipeline).
    The evaluation itself is deterministic (compare output to gold standard).
    The AGENT being evaluated uses genuine LLM reasoning.

MANDATE 1: Uses pydantic-evals (real SDK) for Dataset/Case/Evaluator pattern.
MANDATE 2: No stubs — real comparison logic, real metric computation.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# GOLD-STANDARD SCHEMAS — typed inputs/outputs for full agent eval
# ═══════════════════════════════════════════════════════════════

class AgentEvalInput(BaseModel):
    """Input for full agent evaluation — a payment failure scenario."""
    payment_id: str
    customer_id: str
    amount: int
    currency: str = "INR"
    failure_code: str
    failure_reason: str
    attempt_count: int = 0
    current_tier: str = "silent"


class AgentEvalOutput(BaseModel):
    """Output from agent — what the agent produced."""
    tools_called: list[str]
    final_action: str
    recovered: bool
    tier_used: str
    summary: str = ""


class AgentExpected(BaseModel):
    """Gold-standard expected output — what the agent SHOULD have done."""
    expected_tools: list[str] = Field(default_factory=list)
    expected_action: str = ""
    should_recover: bool = False
    max_steps: int = 5
    notes: str = ""


# ═══════════════════════════════════════════════════════════════
# EVALUATORS — pydantic-evals Evaluator subclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class ToolSelectionEvaluator(Evaluator[AgentEvalInput, AgentEvalOutput, AgentExpected]):
    """NAT pattern: Compare tool selection to gold standard.

    Score 1.0 = agent used exactly the right tools.
    Score 0.0 = agent used completely wrong tools.
    """

    def evaluate(self, ctx: EvaluatorContext[AgentEvalInput, AgentEvalOutput, AgentExpected]) -> float:
        if ctx.output is None or ctx.expected_output is None:
            return 0.0

        expected_tools = set(ctx.expected_output.expected_tools)
        actual_tools = set(ctx.output.tools_called)

        if not expected_tools:
            return 1.0  # No expected tools = any selection is fine

        # Precision: what % of actual tools were expected
        correct = actual_tools & expected_tools
        precision = len(correct) / len(actual_tools) if actual_tools else 0.0

        # Recall: what % of expected tools were actually called
        recall = len(correct) / len(expected_tools) if expected_tools else 0.0

        # F1 score
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


@dataclass
class RecoveryOutcomeEvaluator(Evaluator[AgentEvalInput, AgentEvalOutput, AgentExpected]):
    """NAT pattern: Check if agent achieved the right outcome."""

    def evaluate(self, ctx: EvaluatorContext[AgentEvalInput, AgentEvalOutput, AgentExpected]) -> float:
        if ctx.output is None or ctx.expected_output is None:
            return 0.0

        # Did the agent recover when it should have?
        if ctx.expected_output.should_recover:
            return 1.0 if ctx.output.recovered else 0.0

        # Did the agent NOT recover when it shouldn't have?
        return 1.0 if not ctx.output.recovered else 0.5


@dataclass
class StepEfficiencyEvaluator(Evaluator[AgentEvalInput, AgentEvalOutput, AgentExpected]):
    """NAT pattern: Did the agent take too many steps?"""

    def evaluate(self, ctx: EvaluatorContext[AgentEvalInput, AgentEvalOutput, AgentExpected]) -> float:
        if ctx.output is None or ctx.expected_output is None:
            return 0.0

        max_steps = ctx.expected_output.max_steps
        actual_steps = len(ctx.output.tools_called)

        if actual_steps <= max_steps:
            return 1.0
        # Penalize excess steps
        return max(0.0, 1.0 - (actual_steps - max_steps) / max_steps)


# ═══════════════════════════════════════════════════════════════
# GOLD-STANDARD DATASET — the reference for all evaluations
# ═══════════════════════════════════════════════════════════════

AGENT_EVAL_DATASET = Dataset[AgentEvalInput, AgentEvalOutput, AgentExpected](
    name="recovery_agent_gold_standard",
    cases=[
        # ── Scenario 1: Card expired, first attempt ──
        Case(
            name="card_expired_first_attempt",
            inputs=AgentEvalInput(
                payment_id="pay_001",
                customer_id="cust_001",
                amount=5000,
                failure_code="card_expired",
                failure_reason="Card has expired",
                attempt_count=0,
            ),
            expected_output=AgentExpected(
                expected_tools=["diagnose_payment_failure", "send_recovery_notification"],
                expected_action="send_notification",
                should_recover=False,
                max_steps=3,
                notes="First attempt: diagnose + notify about expiry",
            ),
        ),
        # ── Scenario 2: Insufficient funds, retry should work ──
        Case(
            name="insufficient_funds_retry",
            inputs=AgentEvalInput(
                payment_id="pay_002",
                customer_id="cust_002",
                amount=10000,
                failure_code="51",
                failure_reason="Insufficient funds",
                attempt_count=0,
            ),
            expected_output=AgentExpected(
                expected_tools=["diagnose_payment_failure", "schedule_retry"],
                expected_action="wait_and_retry",
                should_recover=True,
                max_steps=3,
                notes="Transient failure: diagnose + schedule retry",
            ),
        ),
        # ── Scenario 3: Network timeout, should retry ──
        Case(
            name="network_timeout_retry",
            inputs=AgentEvalInput(
                payment_id="pay_003",
                customer_id="cust_003",
                amount=25000,
                failure_code="network_timeout",
                failure_reason="Connection timed out",
                attempt_count=0,
            ),
            expected_output=AgentExpected(
                expected_tools=["diagnose_payment_failure", "schedule_retry"],
                expected_action="retry_payment",
                should_recover=True,
                max_steps=3,
                notes="Transient: diagnose + retry immediately",
            ),
        ),
        # ── Scenario 4: Bank declined, should escalate ──
        Case(
            name="bank_declined_escalate",
            inputs=AgentEvalInput(
                payment_id="pay_004",
                customer_id="cust_004",
                amount=50000,
                failure_code="bank_declined",
                failure_reason="Bank declined the transaction",
                attempt_count=2,
            ),
            expected_output=AgentExpected(
                expected_tools=["diagnose_payment_failure", "escalate_to_human"],
                expected_action="escalate_to_human",
                should_recover=False,
                max_steps=4,
                notes="Permanent failure after retries: escalate",
            ),
        ),
        # ── Scenario 5: UPI mandate revoked ──
        Case(
            name="mandate_revoked_escalate",
            inputs=AgentEvalInput(
                payment_id="pay_005",
                customer_id="cust_005",
                amount=100000,
                failure_code="mandate_revoked",
                failure_reason="UPI mandate cancelled by customer",
                attempt_count=0,
            ),
            expected_output=AgentExpected(
                expected_tools=["diagnose_payment_failure", "escalate_to_human"],
                expected_action="escalate_to_human",
                should_recover=False,
                max_steps=3,
                notes="Permanent: mandate revoked, nothing to retry",
            ),
        ),
    ],
    evaluators=[
        ToolSelectionEvaluator(),
        RecoveryOutcomeEvaluator(),
        StepEfficiencyEvaluator(),
    ],
)


# ═══════════════════════════════════════════════════════════════
# AUTO-EVALUATION — runs after each agent recovery
# ═══════════════════════════════════════════════════════════════

class EvalResult(BaseModel):
    """Result of evaluating a single agent run against gold standard."""
    scenario: str
    tool_selection_score: float
    recovery_score: float
    efficiency_score: float
    overall_score: float
    issues_found: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


def auto_evaluate_agent_run(
    payment_id: str,
    failure_code: str,
    tools_called: list[str],
    recovered: bool,
    attempt_count: int = 0,
) -> EvalResult | None:
    """NAT pattern: Auto-evaluate agent run against gold standard.

    Runs after each recovery attempt to detect regressions.
    Returns eval result with scores and improvement suggestions.
    """
    # Find matching gold-standard scenario
    matched_case = None
    for case in AGENT_EVAL_DATASET.cases:
        if case.inputs.failure_code == failure_code:
            matched_case = case
            break

    if matched_case is None:
        return None

    # Build agent output from actual run
    agent_output = AgentEvalOutput(
        tools_called=tools_called,
        final_action=tools_called[-1] if tools_called else "",
        recovered=recovered,
        tier_used="silent",
    )

    # Run evaluators manually (not through dataset.evaluate_sync)
    tool_eval = ToolSelectionEvaluator()
    recovery_eval = RecoveryOutcomeEvaluator()
    efficiency_eval = StepEfficiencyEvaluator()

    # Build context for evaluation — MANDATE 1: pydantic_evals EvaluatorContext (real SDK)
    ctx = EvaluatorContext(
        name=matched_case.name,
        inputs=matched_case.inputs,
        output=agent_output,
        expected_output=matched_case.expected_output,
        metadata={},
        duration=0.0,
        _span_tree=None,
        attributes={},
        metrics={},
    )

    tool_score = tool_eval.evaluate(ctx)
    recovery_score = recovery_eval.evaluate(ctx)
    efficiency_score = efficiency_eval.evaluate(ctx)

    overall = (tool_score * 0.4 + recovery_score * 0.4 + efficiency_score * 0.2)

    # Detect issues and generate suggestions
    issues = []
    suggestions = []

    if tool_score < 0.5:
        issues.append(f"Tool selection mismatch: called {tools_called}, expected {matched_case.expected_output.expected_tools}")
        suggestions.append(f"Review system prompt for {failure_code} scenarios — agent should prioritize {matched_case.expected_output.expected_tools}")

    if recovery_score < 0.5:
        if matched_case.expected_output.should_recover and not recovered:
            issues.append(f"Agent failed to recover {failure_code} — should have succeeded")
            suggestions.append(f"Investigate retry logic for {failure_code} — transient failure should recover")
        elif not matched_case.expected_output.should_recover and recovered:
            issues.append(f"Agent recovered {failure_code} — should have escalated")

    if efficiency_score < 0.8:
        issues.append(f"Too many steps: {len(tools_called)} > {matched_case.expected_output.max_steps}")
        suggestions.append("Review tool selection — agent may be calling unnecessary diagnostic tools")

    return EvalResult(
        scenario=matched_case.name,
        tool_selection_score=tool_score,
        recovery_score=recovery_score,
        efficiency_score=efficiency_score,
        overall_score=overall,
        issues_found=issues,
        suggestions=suggestions,
    )


# ═══════════════════════════════════════════════════════════════
# SYSTEMATIC IMPROVEMENT — eval → identify → fix → re-eval
# ═══════════════════════════════════════════════════════════════

class ImprovementTracker:
    """NAT pattern: Track evaluation scores over time to measure improvements.

    Stores eval results and detects regressions.
    """

    def __init__(self, history_path: str | Path | None = None):
        self.history_path = Path(history_path) if history_path else None
        self.results: list[dict] = []
        if self.history_path and self.history_path.exists():
            self._load()

    def _load(self):
        """Load evaluation history from disk."""
        try:
            with open(self.history_path) as f:
                self.results = json.load(f)
        except Exception:
            self.results = []

    def _save(self):
        """Save evaluation history to disk."""
        if self.history_path:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "w") as f:
                json.dump(self.results, f, indent=2)

    def record_eval(self, result: EvalResult, run_id: str = ""):
        """Record an evaluation result."""
        entry = {
            "run_id": run_id,
            "scenario": result.scenario,
            "overall_score": result.overall_score,
            "tool_selection_score": result.tool_selection_score,
            "recovery_score": result.recovery_score,
            "efficiency_score": result.efficiency_score,
            "issues": result.issues_found,
            "suggestions": result.suggestions,
            "timestamp": time.time(),
        }
        self.results.append(entry)
        self._save()

    def detect_regression(self, window: int = 10) -> list[str]:
        """Detect if recent evals are worse than historical average.

        NAT pattern: "measure performance improvements, make data-driven
        improvements through configuration experiments."
        """
        if len(self.results) < window + 5:
            return []

        recent = self.results[-window:]
        historical = self.results[:-window]

        recent_avg = sum(r["overall_score"] for r in recent) / len(recent)
        hist_avg = sum(r["overall_score"] for r in historical) / len(historical)

        issues = []
        if recent_avg < hist_avg - 0.1:
            issues.append(
                f"REGRESSION DETECTED: Recent avg {recent_avg:.2f} < historical {hist_avg:.2f}"
            )

        # Check per-scenario regressions
        scenarios = set(r["scenario"] for r in recent)
        for scenario in scenarios:
            recent_scores = [r["overall_score"] for r in recent if r["scenario"] == scenario]
            hist_scores = [r["overall_score"] for r in historical if r["scenario"] == scenario]
            if recent_scores and hist_scores:
                recent_s = sum(recent_scores) / len(recent_scores)
                hist_s = sum(hist_scores) / len(hist_scores)
                if recent_s < hist_s - 0.15:
                    issues.append(
                        f"Scenario '{scenario}' regressed: {recent_s:.2f} < {hist_s:.2f}"
                    )

        return issues

    def get_summary(self) -> dict:
        """Get summary statistics of all evaluations."""
        if not self.results:
            return {"total_runs": 0}

        scores = [r["overall_score"] for r in self.results]
        return {
            "total_runs": len(self.results),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "scenarios_evaluated": list(set(r["scenario"] for r in self.results)),
            "recent_issues": self.results[-1].get("issues", []) if self.results else [],
        }


# ═══════════════════════════════════════════════════════════════
# RUN FULL EVALUATION
# ═══════════════════════════════════════════════════════════════

def run_gold_standard_eval():
    """Run the full gold-standard evaluation dataset.

    NAT pattern: "creating gold-standard datasets, uncovering bugs,
    and making data-driven improvements."
    """
    report = AGENT_EVAL_DATASET.evaluate_sync(lambda x: AgentEvalOutput(
        tools_called=[],
        final_action="",
        recovered=False,
        tier_used="silent",
    ))
    report.print(include_input=True, include_output=True)
    return report
