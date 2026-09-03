"""Component-Level Evaluations — using pydantic-evals (real SDK).

Source: pydantic-evals Dataset/Case/Evaluator pattern
        Component-level evaluations from Evaluating AI Agents (Arize AI)

Tests each agent component independently with labeled data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from recovery_agent.models import ActionType, FailureType


# ═══════════════════════════════════════════════════════════════
# INPUT/OUTPUT SCHEMAS — typed for pydantic-evals
# ═══════════════════════════════════════════════════════════════

class DiagnosisInput(BaseModel):
    """Input for diagnosis evaluation."""
    failure_code: str
    failure_reason: str


class DiagnosisOutput(BaseModel):
    """Output from diagnosis."""
    root_cause: str
    confidence: float
    category: str


class ToolSelectionInput(BaseModel):
    """Input for tool selection evaluation."""
    failure_type: str
    attempt_count: int
    amount: float = 1000.0


class ToolSelectionOutput(BaseModel):
    """Output from tool selection."""
    action: str
    reasoning: str = ""


class ReasoningEfficiencyInput(BaseModel):
    """Input for reasoning efficiency evaluation."""
    tool_calls: list[str]  # List of tool names called in sequence
    final_action: str  # The final action taken


class ReasoningEfficiencyOutput(BaseModel):
    """Output from reasoning efficiency."""
    steps_used: int  # Number of tool calls
    redundant_calls: int  # Number of redundant/repeated calls
    efficient: bool  # Whether the path was efficient


# ═══════════════════════════════════════════════════════════════
# EVALUATORS — pydantic-evals Evaluator subclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class DiagnosisAccuracy(Evaluator[DiagnosisInput, DiagnosisOutput]):
    """Check if diagnosis matches expected root cause."""

    def evaluate(self, ctx: EvaluatorContext[DiagnosisInput, DiagnosisOutput]) -> float:
        if ctx.output is None:
            return 0.0
        if ctx.output.root_cause == ctx.expected_output.root_cause:
            return 1.0
        return 0.0


@dataclass
class DiagnosisConfidence(Evaluator[DiagnosisInput, DiagnosisOutput]):
    """Check if diagnosis confidence is above threshold."""

    def evaluate(self, ctx: EvaluatorContext[DiagnosisInput, DiagnosisOutput]) -> bool:
        if ctx.output is None:
            return False
        return ctx.output.confidence >= 0.5


@dataclass
class ToolSelectionAccuracy(Evaluator[ToolSelectionInput, ToolSelectionOutput]):
    """Check if tool selection matches expected action."""

    def evaluate(self, ctx: EvaluatorContext[ToolSelectionInput, ToolSelectionOutput]) -> float:
        if ctx.output is None:
            return 0.0
        if ctx.output.action == ctx.expected_output.action:
            return 1.0
        return 0.0


@dataclass
class ReasoningEfficiency(Evaluator[ReasoningEfficiencyInput, ReasoningEfficiencyOutput]):
    """Check if agent reasoning was efficient — minimal redundant tool calls.

    CrewAI: "ReasoningEfficiencyEvaluator" — measures if agent took too many steps.
    A score of 1.0 means perfectly efficient (no redundant calls).
    """

    def evaluate(self, ctx: EvaluatorContext[ReasoningEfficiencyInput, ReasoningEfficiencyOutput]) -> float:
        if ctx.output is None:
            return 0.0
        if ctx.output.steps_used == 0:
            return 1.0
        # Score = 1 - (redundant_calls / total_calls), min 0
        if ctx.output.steps_used == 0:
            return 1.0
        efficiency = 1.0 - (ctx.output.redundant_calls / ctx.output.steps_used)
        return max(0.0, efficiency)


@dataclass
class ToolSelectionRelevance(Evaluator[ToolSelectionInput, ToolSelectionOutput]):
    """Check if the selected tool is semantically relevant to the failure type.

    Unlike ToolSelectionAccuracy (exact match), this checks relevance categories.
    """
    # Tool-to-failure-type relevance mapping
    _RELEVANCE: dict[str, set[str]] = {
        "send_notification": {"card_expired", "insufficient_funds", "network_timeout", "mandate_revoked"},
        "update_payment_method": {"card_expired", "incorrect_details"},
        "retry_payment": {"network_timeout", "insufficient_funds", "bank_declined"},
        "wait_and_retry": {"insufficient_funds", "bank_declined"},
        "escalate_to_human": {"bank_declined", "fraud_suspected", "risk_block"},
        "check_payment_status": set(),  # Universal
        "send_email": set(),  # Universal
        "send_sms": set(),  # Universal
    }

    def evaluate(self, ctx: EvaluatorContext[ToolSelectionInput, ToolSelectionOutput]) -> float:
        if ctx.output is None:
            return 0.0
        tool = ctx.output.action
        failure = ctx.inputs.failure_type
        relevant_failures = self._RELEVANCE.get(tool, set())
        if not relevant_failures:
            return 0.5  # Universal tools get neutral score
        if failure in relevant_failures:
            return 1.0
        return 0.0


# ═══════════════════════════════════════════════════════════════
# TARGET FUNCTIONS — actual implementations to evaluate
# ═══════════════════════════════════════════════════════════════

def run_diagnosis_target(inputs: DiagnosisInput) -> DiagnosisOutput:
    """Run diagnosis on failure code."""
    from recovery_agent.agent.diagnosis import run_diagnosis
    from recovery_agent.models import Case, PaymentEvent

    case = Case(
        payment=PaymentEvent(
            event_type="payment_failed",
            payment_id="eval_test",
            amount=1000,
            currency="INR",
            failure_code=inputs.failure_code,
            failure_reason=inputs.failure_reason,
            customer_id="eval_customer",
        ),
    )
    case = run_diagnosis(case)

    if case.diagnosis:
        return DiagnosisOutput(
            root_cause=case.diagnosis.root_cause.value,
            confidence=case.diagnosis.confidence,
            category=case.diagnosis.category,
        )
    return DiagnosisOutput(root_cause="unknown", confidence=0.0, category="unknown")


def run_tool_selection_target(inputs: ToolSelectionInput) -> ToolSelectionOutput:
    """Run tool selection on failure type."""
    from recovery_agent.agent.decision import decide_intervention
    from recovery_agent.agent.diagnosis import run_diagnosis
    from recovery_agent.models import Case, PaymentEvent, FailureType

    try:
        ft = FailureType(inputs.failure_type)
    except ValueError:
        ft = FailureType.UNKNOWN

    case = Case(
        payment=PaymentEvent(
            event_type="payment_failed",
            payment_id="eval_test",
            amount=inputs.amount,
            currency="INR",
            failure_code=inputs.failure_type,
            failure_reason=inputs.failure_type,
            customer_id="eval_customer",
        ),
        attempt_count=inputs.attempt_count,
    )
    case = run_diagnosis(case)
    action = decide_intervention(case)

    return ToolSelectionOutput(
        action=action.value,
        reasoning=case.payment.metadata.get("strategy_reasoning", ""),
    )


# ═══════════════════════════════════════════════════════════════
# DATASETS — labeled test cases
# ═══════════════════════════════════════════════════════════════

DIAGNOSIS_DATASET = Dataset[DiagnosisInput, DiagnosisOutput](
    name="diagnosis_accuracy",
    cases=[
        Case(
            name="insufficient_funds",
            inputs=DiagnosisInput(failure_code="51", failure_reason="Insufficient funds"),
            expected_output=DiagnosisOutput(root_cause="insufficient_funds", confidence=0.9, category="transient"),
        ),
        Case(
            name="card_expired",
            inputs=DiagnosisInput(failure_code="card_expired", failure_reason="Card has expired"),
            expected_output=DiagnosisOutput(root_cause="card_expired", confidence=0.95, category="permanent"),
        ),
        Case(
            name="network_timeout",
            inputs=DiagnosisInput(failure_code="network_timeout", failure_reason="Connection timed out"),
            expected_output=DiagnosisOutput(root_cause="network_timeout", confidence=0.85, category="transient"),
        ),
        Case(
            name="risk_block",
            inputs=DiagnosisInput(failure_code="risk_block", failure_reason="Risk check failed"),
            expected_output=DiagnosisOutput(root_cause="risk_block", confidence=0.9, category="permanent"),
        ),
        Case(
            name="mandate_revoked",
            inputs=DiagnosisInput(failure_code="mandate_revoked", failure_reason="UPI mandate cancelled"),
            expected_output=DiagnosisOutput(root_cause="mandate_revoked", confidence=0.9, category="permanent"),
        ),
    ],
    evaluators=[DiagnosisAccuracy(), DiagnosisConfidence()],
)

TOOL_SELECTION_DATASET = Dataset[ToolSelectionInput, ToolSelectionOutput](
    name="tool_selection_accuracy",
    cases=[
        Case(
            name="card_expired_first",
            inputs=ToolSelectionInput(failure_type="card_expired", attempt_count=0),
            expected_output=ToolSelectionOutput(action="send_notification"),
        ),
        Case(
            name="card_expired_second",
            inputs=ToolSelectionInput(failure_type="card_expired", attempt_count=1),
            expected_output=ToolSelectionOutput(action="update_payment_method"),
        ),
        Case(
            name="insufficient_funds",
            inputs=ToolSelectionInput(failure_type="insufficient_funds", attempt_count=0),
            expected_output=ToolSelectionOutput(action="wait_and_retry"),
        ),
        Case(
            name="network_timeout",
            inputs=ToolSelectionInput(failure_type="network_timeout", attempt_count=0),
            expected_output=ToolSelectionOutput(action="retry_payment"),
        ),
        Case(
            name="bank_declined",
            inputs=ToolSelectionInput(failure_type="bank_declined", attempt_count=0),
            expected_output=ToolSelectionOutput(action="escalate_to_human"),
        ),
    ],
    evaluators=[ToolSelectionAccuracy()],
)


REASONING_EFFICIENCY_DATASET = Dataset[ReasoningEfficiencyInput, ReasoningEfficiencyOutput](
    name="reasoning_efficiency",
    cases=[
        Case(
            name="efficient_single_tool",
            inputs=ReasoningEfficiencyInput(
                tool_calls=["check_payment_status", "send_notification"],
                final_action="send_notification",
            ),
            expected_output=ReasoningEfficiencyOutput(steps_used=2, redundant_calls=0, efficient=True),
        ),
        Case(
            name="redundant_calls",
            inputs=ReasoningEfficiencyInput(
                tool_calls=["check_payment_status", "check_payment_status", "send_notification"],
                final_action="send_notification",
            ),
            expected_output=ReasoningEfficiencyOutput(steps_used=3, redundant_calls=1, efficient=False),
        ),
        Case(
            name="no_tools_needed",
            inputs=ReasoningEfficiencyInput(
                tool_calls=[],
                final_action="escalate_to_human",
            ),
            expected_output=ReasoningEfficiencyOutput(steps_used=0, redundant_calls=0, efficient=True),
        ),
    ],
    evaluators=[ReasoningEfficiency()],
)


# ═══════════════════════════════════════════════════════════════
# RUN EVALUATIONS
# ═══════════════════════════════════════════════════════════════

def run_diagnosis_eval():
    """Run diagnosis evaluation."""
    report = DIAGNOSIS_DATASET.evaluate_sync(run_diagnosis_target)
    report.print(include_input=True, include_output=True)
    return report


def run_tool_selection_eval():
    """Run tool selection evaluation."""
    report = TOOL_SELECTION_DATASET.evaluate_sync(run_tool_selection_target)
    report.print(include_input=True, include_output=True)
    return report


def run_reasoning_efficiency_eval():
    """Run reasoning efficiency evaluation."""
    def _target(inputs: ReasoningEfficiencyInput) -> ReasoningEfficiencyOutput:
        # Count redundant calls (same tool called consecutively)
        redundant = 0
        for i in range(1, len(inputs.tool_calls)):
            if inputs.tool_calls[i] == inputs.tool_calls[i - 1]:
                redundant += 1
        return ReasoningEfficiencyOutput(
            steps_used=len(inputs.tool_calls),
            redundant_calls=redundant,
            efficient=redundant == 0,
        )

    report = REASONING_EFFICIENCY_DATASET.evaluate_sync(_target)
    report.print(include_input=True, include_output=True)
    return report


def run_all_component_evals():
    """Run all component evaluations."""
    print("\n" + "=" * 60)
    print("DIAGNOSIS EVALUATION")
    print("=" * 60)
    run_diagnosis_eval()

    print("\n" + "=" * 60)
    print("TOOL SELECTION EVALUATION")
    print("=" * 60)
    run_tool_selection_eval()

    print("\n" + "=" * 60)
    print("REASONING EFFICIENCY EVALUATION")
    print("=" * 60)
    run_reasoning_efficiency_eval()


if __name__ == "__main__":
    run_all_component_evals()
