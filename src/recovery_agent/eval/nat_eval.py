"""NAT Evaluation — using REAL nvidia-nat SDK (MANDATE 6 compliant).

MANDATE 6: This file imports and uses the ACTUAL nvidia-nat package.
    from nat.plugins.langchain.eval.trajectory_evaluator import TrajectoryEvaluator
    from nat.data_models.evaluator import EvalInputItem
    from nat.data_models.intermediate_step import IntermediateStep, IntermediateStepPayload

NVIDIA NeMo Agent Toolkit course pattern:
    - LLM-as-judge trajectory scoring
    - Gold-standard datasets for systematic evaluation
    - Data-driven improvements through evaluation experiments
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

# MANDATE 6: Real NAT imports — not custom code pretending to be NAT
from nat.plugins.langchain.eval.trajectory_evaluator import TrajectoryEvaluator
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.intermediate_step import IntermediateStep, IntermediateStepPayload, IntermediateStepType, StreamEventData
from nat.data_models.invocation_node import InvocationNode
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.plugins.eval.data_models.evaluator_io import EvalOutput, EvalOutputItem

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# TRAJECTORY CONVERSION — LangGraph messages → NAT IntermediateStep
# ═══════════════════════════════════════════════════════════════

def _convert_langgraph_trajectory(
    messages: list,
) -> list[IntermediateStep]:
    """Convert LangGraph message history to NAT IntermediateStep format.

    MANDATE 1: Uses nat.data_models.intermediate_step (real NAT SDK).

    NAT event_type mapping:
        LangGraph AIMessage (tool_calls) → TOOL_START (with data.input = tool args)
        LangGraph ToolMessage (result)    → TOOL_END   (with data.output = tool result)
        LangGraph AIMessage (final)       → LLM_END

    NAT _DEFAULT_EVENT_FILTER only includes TOOL_END and LLM_END.
    TOOL_START is filtered out. So we must carry tool_input into TOOL_END
    so the evaluator can read it from data.input.
    """
    steps = []
    parent_id = "recovery_agent"
    msg_counter = 0

    # Track tool calls to carry input from TOOL_START to TOOL_END
    pending_tool_calls: dict[str, dict] = {}  # tool_call_id → {name, args}

    for msg in messages:
        msg_class = type(msg).__name__
        msg_counter += 1
        step_uuid = f"step-{msg_counter}"

        if msg_class == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                # Tool call — NAT TOOL_START event
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    call_id = tc.get("id", f"call_{msg_counter}")
                    tool_args = tc.get("args", {})

                    # Track for TOOL_END
                    pending_tool_calls[call_id] = {
                        "name": tool_name,
                        "args": tool_args,
                    }

                    ancestry = InvocationNode(
                        function_id=call_id,
                        function_name=tool_name,
                    )

                    step_data = StreamEventData(
                        input=tool_args,
                        output=None,
                    )

                    step_payload = IntermediateStepPayload(
                        event_type=IntermediateStepType.TOOL_START,
                        event_timestamp=time.time(),
                        framework=LLMFrameworkEnum.LANGCHAIN,
                        name=tool_name,
                        tags=["tool_call"],
                        metadata=None,
                        data=step_data,
                        usage_info=None,
                        UUID=step_uuid,
                    )
                    steps.append(IntermediateStep(
                        parent_id=parent_id,
                        function_ancestry=ancestry,
                        payload=step_payload,
                    ))
            else:
                # LLM response (final)
                ancestry = InvocationNode(
                    function_id=f"llm_{msg_counter}",
                    function_name="agent_response",
                )

                step_payload = IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_END,
                    event_timestamp=time.time(),
                    framework=LLMFrameworkEnum.LANGCHAIN,
                    name="agent_response",
                    tags=["response"],
                    metadata=None,
                    data=None,
                    usage_info=None,
                    UUID=step_uuid,
                )
                steps.append(IntermediateStep(
                    parent_id=parent_id,
                    function_ancestry=ancestry,
                    payload=step_payload,
                ))

        elif msg_class == "ToolMessage":
            # Tool result — NAT TOOL_END event
            tool_name = getattr(msg, "name", "tool_result")
            tool_call_id = getattr(msg, "tool_call_id", f"call_{msg_counter}")

            ancestry = InvocationNode(
                function_id=tool_call_id,
                function_name=tool_name,
            )

            # Carry tool_input from the corresponding TOOL_START
            tool_input = ""
            if tool_call_id in pending_tool_calls:
                tool_input = pending_tool_calls[tool_call_id]["args"]
                del pending_tool_calls[tool_call_id]

            # NAT evaluator reads data.input for tool_input, data.output for tool_output
            step_data = StreamEventData(
                input=tool_input,
                output=str(msg.content),
            )

            step_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.TOOL_END,
                event_timestamp=time.time(),
                framework=LLMFrameworkEnum.LANGCHAIN,
                name=tool_name,
                tags=["tool_result"],
                metadata=None,
                data=step_data,
                usage_info=None,
                UUID=step_uuid,
            )
            steps.append(IntermediateStep(
                parent_id=parent_id,
                function_ancestry=ancestry,
                payload=step_payload,
            ))

    return steps


# ═══════════════════════════════════════════════════════════════
# NAT EVALUATOR — wraps TrajectoryEvaluator (real NAT SDK)
# ═══════════════════════════════════════════════════════════════

class RecoveryTrajectoryEvaluator:
    """NAT-based trajectory evaluator for recovery agent.

    MANDATE 6: Uses ACTUAL nat.plugins.langchain.eval.TrajectoryEvaluator.
    MANDATE 1: Uses nat.data_models.evaluator.EvalInputItem (real NAT SDK).

    NVIDIA NeMo course: "LLM-as-judge trajectory scoring"
    """

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool] | None = None):
        """Initialize with the same LLM used by the agent.

        MANDATE 6: Real TrajectoryEvaluator from nvidia-nat.
        """
        self.evaluator = TrajectoryEvaluator(llm=llm, tools=tools)

    async def evaluate_run(
        self,
        payment_id: str,
        question: str,
        answer: str,
        messages: list,
    ) -> EvalOutputItem:
        """Evaluate a single agent run using NAT trajectory scoring.

        MANDATE 6: Uses real EvalInputItem and TrajectoryEvaluator.
        """
        # Convert LangGraph trajectory to NAT format
        trajectory = _convert_langgraph_trajectory(messages)

        # Build NAT EvalInputItem
        item = EvalInputItem(
            id=payment_id,
            input_obj=question,
            output_obj=answer,
            expected_output_obj=None,
            expected_trajectory=[],
            trajectory=trajectory,
            full_dataset_entry=None,
        )

        # MANDATE 6: Call real NAT TrajectoryEvaluator
        result = await self.evaluator.evaluate_item(item)
        return result

    def evaluate_run_sync(
        self,
        payment_id: str,
        question: str,
        answer: str,
        messages: list,
    ) -> EvalOutputItem:
        """Synchronous wrapper for evaluate_run."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.evaluate_run(payment_id, question, answer, messages)
        )


# ═══════════════════════════════════════════════════════════════
# GOLD-STANDARD DATASET — for systematic evaluation
# ═══════════════════════════════════════════════════════════════

GOLD_STANDARD_SCENARIOS = [
    {
        "name": "card_expired_first_attempt",
        "payment_id": "pay_001",
        "customer_id": "cust_001",
        "amount": 5000,
        "failure_code": "card_expired",
        "failure_reason": "Card has expired",
        "expected_action": "send_notification",
        "should_recover": False,
    },
    {
        "name": "insufficient_funds_retry",
        "payment_id": "pay_002",
        "customer_id": "cust_002",
        "amount": 10000,
        "failure_code": "51",
        "failure_reason": "Insufficient funds",
        "expected_action": "wait_and_retry",
        "should_recover": True,
    },
    {
        "name": "network_timeout_retry",
        "payment_id": "pay_003",
        "customer_id": "cust_003",
        "amount": 25000,
        "failure_code": "network_timeout",
        "failure_reason": "Connection timed out",
        "expected_action": "retry_payment",
        "should_recover": True,
    },
    {
        "name": "bank_declined_escalate",
        "payment_id": "pay_004",
        "customer_id": "cust_004",
        "amount": 50000,
        "failure_code": "bank_declined",
        "failure_reason": "Bank declined the transaction",
        "expected_action": "escalate_to_human",
        "should_recover": False,
    },
    {
        "name": "mandate_revoked_escalate",
        "payment_id": "pay_005",
        "customer_id": "cust_005",
        "amount": 100000,
        "failure_code": "mandate_revoked",
        "failure_reason": "UPI mandate cancelled by customer",
        "expected_action": "escalate_to_human",
        "should_recover": False,
    },
]


# ═══════════════════════════════════════════════════════════════
# IMPROVEMENT TRACKER — tracks scores over time
# ═══════════════════════════════════════════════════════════════

class ImprovementTracker:
    """Track NAT evaluation scores over time to detect regressions.

    NVIDIA NeMo: "measure performance improvements, make data-driven
    improvements through configuration experiments."
    """

    def __init__(self, history_path: str | Path | None = None):
        self.history_path = Path(history_path) if history_path else None
        self.results: list[dict] = []
        if self.history_path and self.history_path.exists():
            self._load()

    def _load(self):
        try:
            with open(self.history_path) as f:
                self.results = json.load(f)
        except Exception:
            self.results = []

    def _save(self):
        if self.history_path:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "w") as f:
                json.dump(self.results, f, indent=2)

    def record_eval(self, scenario: str, score: float, reasoning: str = ""):
        entry = {
            "scenario": scenario,
            "score": score,
            "reasoning": reasoning[:500],
            "timestamp": time.time(),
        }
        self.results.append(entry)
        self._save()

    def detect_regression(self, window: int = 10) -> list[str]:
        if len(self.results) < window + 5:
            return []

        recent = self.results[-window:]
        historical = self.results[:-window]

        recent_avg = sum(r["score"] for r in recent) / len(recent)
        hist_avg = sum(r["score"] for r in historical) / len(historical)

        issues = []
        if recent_avg < hist_avg - 0.1:
            issues.append(
                f"REGRESSION: Recent avg {recent_avg:.2f} < historical {hist_avg:.2f}"
            )

        scenarios = set(r["scenario"] for r in recent)
        for scenario in scenarios:
            recent_scores = [r["score"] for r in recent if r["scenario"] == scenario]
            hist_scores = [r["score"] for r in historical if r["scenario"] == scenario]
            if recent_scores and hist_scores:
                recent_s = sum(recent_scores) / len(recent_scores)
                hist_s = sum(hist_scores) / len(hist_scores)
                if recent_s < hist_s - 0.15:
                    issues.append(
                        f"Scenario '{scenario}' regressed: {recent_s:.2f} < {hist_s:.2f}"
                    )

        return issues

    def get_summary(self) -> dict:
        if not self.results:
            return {"total_runs": 0}
        scores = [r["score"] for r in self.results]
        return {
            "total_runs": len(self.results),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "scenarios": list(set(r["scenario"] for r in self.results)),
        }
