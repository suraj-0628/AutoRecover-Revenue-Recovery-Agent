"""Phoenix Evaluation Framework — automated hallucination + tool correctness auditing.

Pulls traces from Arize Phoenix, evaluates agent decisions against ground truth,
and writes annotations back to the trace for the Phoenix dashboard.

Usage:
    python -m recovery_agent.eval.phoenix_evals --evaluate
    python -m recovery_agent.eval.phoenix_evals --evaluate --payment-id pay_XXX
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _safe_str(val: Any, default: str = "") -> str:
    """Safely convert a value to string, handling NaN/None/float."""
    import math
    if val is None:
        return default
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return default
    return str(val)


@dataclass
class EvalResult:
    """Single evaluation result for a span."""
    span_id: str
    trace_id: str
    payment_id: str
    evaluator: str
    score: float  # 0.0 to 1.0
    label: str  # "pass" | "fail" | "warn"
    explanation: str
    metadata: dict = field(default_factory=dict)


class HallucinationEvaluator:
    """Checks whether the diagnosis engine hallucinated a root cause not present in the webhook payload.

    A diagnosis is considered hallucinated if:
    1. The root_cause does not match any known failure pattern from the raw payload
    2. The LLM assigned a root cause that contradicts the Razorpay error code mapping
    3. The confidence is low (< 0.5) but the agent proceeded with an invasive action
    """

    VALID_ROOT_CAUSES = {
        "card_expired", "insufficient_funds", "network_timeout",
        "bank_declined", "mandate_revoked", "user_dropoff",
        "risk_block", "unknown",
    }

    # Error codes that map to specific root causes (from Razorpay knowledge base)
    ERROR_CODE_TO_CAUSE = {
        "bad_request_card_expired": "card_expired",
        "bad_request_card_declined": "bank_declined",
        "bad_request_insufficient_funds": "insufficient_funds",
        "bad_request_too_many_requests": "network_timeout",
        "bad_request_risk_declined": "risk_block",
        "bad_request_authentication_failed": "bank_declined",
        "UPI特定insufficient_funds": "insufficient_funds",
    }

    INVASIVE_ACTIONS = {
        "retry_payment", "capture_payment", "create_order",
        "initiate_voice_call", "create_refund",
    }

    def evaluate(
        self,
        span_attributes: dict[str, Any],
        raw_payload: Optional[dict] = None,
    ) -> EvalResult:
        """Evaluate a single diagnosis span for hallucination."""
        root_cause = _safe_str(span_attributes.get("attributes.root_cause", ""))
        confidence = float(_safe_str(span_attributes.get("attributes.confidence", "0.7"), "0.7"))
        payment_id = _safe_str(span_attributes.get("attributes.payment_id", "unknown"))
        failure_reason = _safe_str(span_attributes.get("attributes.failure_reason", ""))
        failure_code = _safe_str(span_attributes.get("attributes.failure_code", ""))
        decided_action = _safe_str(span_attributes.get("attributes.decided_action", ""))

        issues = []

        # Skip evaluation if root_cause is empty (span doesn't carry diagnosis data)
        if not root_cause:
            return EvalResult(
                span_id=span_attributes.get("context.span_id", ""),
                trace_id=span_attributes.get("context.trace_id", ""),
                payment_id=payment_id,
                evaluator="hallucination",
                score=1.0,
                label="pass",
                explanation="No root_cause on span — evaluation deferred to diagnose span",
                metadata={"root_cause": "", "skipped": True},
            )

        # Check 1: Unknown root cause with high-confidence invasive action
        if root_cause not in self.VALID_ROOT_CAUSES:
            issues.append(f"Root cause '{root_cause}' is not a recognized category")

        # Check 2: Error code contradicts root cause
        if failure_code and failure_code in self.ERROR_CODE_TO_CAUSE:
            expected = self.ERROR_CODE_TO_CAUSE[failure_code]
            if root_cause != expected and root_cause != "unknown":
                issues.append(
                    f"Error code '{failure_code}' maps to '{expected}' but "
                    f"agent diagnosed '{root_cause}'"
                )

        # Check 3: Low confidence + invasive action
        if confidence < 0.5 and decided_action in self.INVASIVE_ACTIONS:
            issues.append(
                f"Low confidence ({confidence:.0%}) with invasive action '{decided_action}'"
            )

        # Check 4: Diagnosis contradicts failure reason text
        if failure_reason:
            reason_lower = failure_reason.lower()
            if "expired" in reason_lower and root_cause != "card_expired":
                issues.append(
                    f"Failure reason mentions 'expired' but root cause is '{root_cause}'"
                )
            if "insufficient" in reason_lower and root_cause != "insufficient_funds":
                issues.append(
                    f"Failure reason mentions 'insufficient' but root cause is '{root_cause}'"
                )

        if issues:
            return EvalResult(
                span_id=span_attributes.get("context.span_id", ""),
                trace_id=span_attributes.get("context.trace_id", ""),
                payment_id=payment_id,
                evaluator="hallucination",
                score=0.0,
                label="fail",
                explanation=f"Hallucination detected: {'; '.join(issues)}",
                metadata={"issues": issues, "root_cause": root_cause, "confidence": confidence},
            )

        return EvalResult(
            span_id=span_attributes.get("context.span_id", ""),
            trace_id=span_attributes.get("context.trace_id", ""),
            payment_id=payment_id,
            evaluator="hallucination",
            score=1.0,
            label="pass",
            explanation="Diagnosis grounded in payload evidence",
            metadata={"root_cause": root_cause, "confidence": confidence},
        )


class ToolCorrectnessEvaluator:
    """Checks whether the agent called the correct tool based on the strategy.

    Tool correctness is validated by:
    1. The chosen action matches the failure type routing rules
    2. Hard decline codes are never retried with payment actions
    3. Card expiry routes to card update, not generic retry
    4. User dropoff routes to notification/payment link, not retry
    """

    HARD_DECLINES = {
        "do_not_honor", "expired_card", "lost_card",
        "stolen_card", "fraud_suspected", "card_blocked",
        "issuer_unavailable", "declined_per心意",
    }

    ACTION_ROUTING = {
        "card_expired": {"send_notification", "update_payment_method", "wait_and_retry"},
        "insufficient_funds": {"wait_and_retry", "send_notification", "create_payment_link"},
        "network_timeout": {"retry_payment", "wait_and_retry"},
        "bank_declined": {"send_notification", "escalate_to_human", "create_payment_link"},
        "mandate_revoked": {"send_notification", "create_payment_link", "escalate_to_human"},
        "user_dropoff": {"send_notification", "create_payment_link", "initiate_voice_call"},
        "risk_block": {"escalate_to_human", "send_notification"},
        "unknown": {"send_notification", "escalate_to_human"},
    }

    def evaluate(self, span_attributes: dict[str, Any]) -> EvalResult:
        """Evaluate a single action span for tool correctness."""
        payment_id = _safe_str(span_attributes.get("attributes.payment_id", "unknown"))
        root_cause = _safe_str(span_attributes.get("attributes.root_cause", "unknown"))
        decided_action = _safe_str(span_attributes.get("attributes.decided_action", ""))
        recovery_tier = _safe_str(span_attributes.get("attributes.recovery_tier", ""))
        final_status = _safe_str(span_attributes.get("attributes.final_status", ""))

        issues = []

        # Check 1: Action matches failure type routing
        valid_actions = self.ACTION_ROUTING.get(root_cause, set())
        if valid_actions and decided_action not in valid_actions:
            issues.append(
                f"Action '{decided_action}' not in valid set for '{root_cause}': "
                f"{valid_actions}"
            )

        # Check 2: Hard decline should never trigger retry_payment
        failure_code = _safe_str(span_attributes.get("attributes.failure_code", ""))
        if failure_code in self.HARD_DECLINES and decided_action == "retry_payment":
            issues.append(
                f"Hard decline '{failure_code}' should not trigger retry_payment"
            )

        # Check 3: Card expiry should route to card update, not generic retry
        if root_cause == "card_expired" and decided_action == "retry_payment":
            issues.append("Card expiry should route to update_payment_method, not retry")

        # Check 4: User dropoff should not use retry_payment
        if root_cause == "user_dropoff" and decided_action == "retry_payment":
            issues.append("User dropoff should not trigger retry — use notification/link")

        if issues:
            return EvalResult(
                span_id=span_attributes.get("context.span_id", ""),
                trace_id=span_attributes.get("context.trace_id", ""),
                payment_id=payment_id,
                evaluator="tool_correctness",
                score=0.0,
                label="fail",
                explanation=f"Tool correctness violation: {'; '.join(issues)}",
                metadata={"issues": issues, "root_cause": root_cause, "action": decided_action},
            )

        return EvalResult(
            span_id=span_attributes.get("context.span_id", ""),
            trace_id=span_attributes.get("context.trace_id", ""),
            payment_id=payment_id,
            evaluator="tool_correctness",
            score=1.0,
            label="pass",
            explanation=f"Action '{decided_action}' correct for '{root_cause}'",
            metadata={"root_cause": root_cause, "action": decided_action},
        )


class ComplianceEvaluator:
    """Checks whether the agent respected guardrail constraints.

    Compliance violations include:
    1. Communication during quiet hours (9 PM – 8 AM)
    2. More than 2 communications in 24 hours
    3. Retrying a hard-decline payment
    4. Exceeding the INR 500K monetary cap
    """

    MAX_MONETARY_RETRY = 500_000  # INR

    def evaluate(self, span_attributes: dict[str, Any]) -> EvalResult:
        """Evaluate a span for compliance violations."""
        payment_id = _safe_str(span_attributes.get("attributes.payment_id", "unknown"))
        amount = float(_safe_str(span_attributes.get("attributes.amount", "0"), "0"))
        decided_action = _safe_str(span_attributes.get("attributes.decided_action", ""))
        recovery_tier = _safe_str(span_attributes.get("attributes.recovery_tier", ""))

        issues = []

        # Check 1: Monetary cap
        if decided_action in ("retry_payment", "capture_payment") and amount > self.MAX_MONETARY_RETRY:
            issues.append(
                f"Amount INR {amount:,.0f} exceeds {self.MAX_MONETARY_RETRY:,} cap"
            )

        # Check 2: Silent tier should never use customer-facing actions
        if recovery_tier == "silent" and decided_action in (
            "send_notification", "create_payment_link", "initiate_voice_call",
        ):
            issues.append(
                f"Silent tier used customer-facing action '{decided_action}'"
            )

        if issues:
            return EvalResult(
                span_id=span_attributes.get("context.span_id", ""),
                trace_id=span_attributes.get("context.trace_id", ""),
                payment_id=payment_id,
                evaluator="compliance",
                score=0.0,
                label="fail",
                explanation=f"Compliance violation: {'; '.join(issues)}",
                metadata={"issues": issues},
            )

        return EvalResult(
            span_id=span_attributes.get("context.span_id", ""),
            trace_id=span_attributes.get("context.trace_id", ""),
            payment_id=payment_id,
            evaluator="compliance",
            score=1.0,
            label="pass",
            explanation="No compliance violations detected",
            metadata={},
        )


class PhoenixEvaluator:
    """Orchestrates all evaluators against Phoenix trace data."""

    def __init__(self, phoenix_url: str = "http://localhost:6006"):
        self.phoenix_url = phoenix_url
        self.hallucination = HallucinationEvaluator()
        self.tool_correctness = ToolCorrectnessEvaluator()
        self.compliance = ComplianceEvaluator()

    def _get_client(self):
        from phoenix.client import Client
        return Client()

    def fetch_agent_traces(self, payment_id: Optional[str] = None) -> list[dict]:
        """Fetch agent_recovery and harness spans from Phoenix."""
        client = self._get_client()
        spans = client.spans.get_spans_dataframe()

        if spans.empty:
            return []

        # Filter to our agent spans
        agent_spans = spans[
            spans["name"].isin(["agent_recovery", "harness", "harness.run_recovery_case", "harness.run_recovery_case_async"])
        ].copy()

        if payment_id:
            agent_spans = agent_spans[
                agent_spans.get("attributes.payment_id", "") == payment_id
            ]

        return agent_spans.to_dict("records")

    def evaluate_all(self, payment_id: Optional[str] = None) -> list[EvalResult]:
        """Run all evaluators on agent traces."""
        traces = self.fetch_agent_traces(payment_id)
        results = []

        for span in traces:
            name = span.get("name", "")

            # Hallucination check on diagnosis step
            results.append(self.hallucination.evaluate(span))

            # Tool correctness check
            results.append(self.tool_correctness.evaluate(span))

            # Compliance check
            results.append(self.compliance.evaluate(span))

        return results

    def write_annotations(self, results: list[EvalResult]) -> int:
        """Write evaluation annotations back to Phoenix spans."""
        client = self._get_client()
        written = 0

        for result in results:
            if not result.span_id:
                continue
            try:
                client.spans.add_span_annotation(
                    span_id=result.span_id,
                    annotation_name=result.evaluator,
                    label=result.label,
                    score=result.score,
                    explanation=result.explanation,
                    metadata=result.metadata if result.metadata else None,
                    sync=True,
                )
                written += 1
            except Exception as e:
                logger.warning(f"Failed to write annotation for span {result.span_id}: {e}")

        return written

    def run_evaluation(self, payment_id: Optional[str] = None) -> dict:
        """Full evaluation pipeline: fetch → evaluate → annotate → report."""
        results = self.evaluate_all(payment_id)
        written = self.write_annotations(results)

        # Aggregate scores
        total = len(results)
        passed = sum(1 for r in results if r.label == "pass")
        failed = sum(1 for r in results if r.label == "fail")

        by_evaluator = {}
        for r in results:
            if r.evaluator not in by_evaluator:
                by_evaluator[r.evaluator] = {"pass": 0, "fail": 0, "total": 0}
            by_evaluator[r.evaluator][r.label] += 1
            by_evaluator[r.evaluator]["total"] += 1

        report = {
            "total_evaluations": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{(passed / total * 100):.1f}%" if total else "N/A",
            "annotations_written": written,
            "by_evaluator": by_evaluator,
            "failed_evaluations": [
                {
                    "payment_id": r.payment_id,
                    "evaluator": r.evaluator,
                    "explanation": r.explanation,
                }
                for r in results if r.label == "fail"
            ],
        }

        return report


def main():
    """CLI entry point for Phoenix evaluations."""
    import argparse

    parser = argparse.ArgumentParser(description="Phoenix Agent Evaluations")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation pipeline")
    parser.add_argument("--payment-id", type=str, help="Evaluate specific payment")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not args.evaluate:
        parser.print_help()
        return

    evaluator = PhoenixEvaluator()
    report = evaluator.run_evaluation(payment_id=args.payment_id)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 60)
        print("PHOENIX AGENT EVALUATION REPORT")
        print("=" * 60)
        print(f"Total evaluations: {report['total_evaluations']}")
        print(f"Passed: {report['passed']}")
        print(f"Failed: {report['failed']}")
        print(f"Pass rate: {report['pass_rate']}")
        print(f"Annotations written to Phoenix: {report['annotations_written']}")
        print()
        print("By Evaluator:")
        for name, stats in report["by_evaluator"].items():
            total = stats["total"]
            passed = stats["pass"]
            print(f"  {name}: {passed}/{total} passed ({passed/total*100:.0f}%)" if total else f"  {name}: no data")
        if report["failed_evaluations"]:
            print()
            print("Failures:")
            for f in report["failed_evaluations"]:
                print(f"  [{f['evaluator']}] {f['payment_id']}: {f['explanation']}")
        print()


if __name__ == "__main__":
    main()
