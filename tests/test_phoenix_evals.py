"""Tests for Phoenix evaluation framework — hallucination, tool correctness, compliance evaluators."""
import pytest
from recovery_agent.eval.phoenix_evals import (
    HallucinationEvaluator,
    ToolCorrectnessEvaluator,
    ComplianceEvaluator,
    EvalResult,
)


class TestHallucinationEvaluator:
    def setup_method(self):
        self.evaluator = HallucinationEvaluator()

    def _make_span(self, **overrides):
        base = {
            "context.span_id": "span_123",
            "context.trace_id": "trace_456",
            "attributes.payment_id": "pay_test_001",
            "attributes.root_cause": "card_expired",
            "attributes.confidence": "0.85",
            "attributes.failure_reason": "Card expired",
            "attributes.failure_code": "",
            "attributes.decided_action": "send_notification",
        }
        base.update(overrides)
        return base

    def test_valid_root_cause_passes(self):
        span = self._make_span()
        result = self.evaluator.evaluate(span)
        assert result.label == "pass"
        assert result.score == 1.0

    def test_unknown_root_cause_fails(self):
        span = self._make_span(**{"attributes.root_cause": "nonexistent_cause"})
        result = self.evaluator.evaluate(span)
        assert result.label == "fail"
        assert result.score == 0.0
        assert "nonexistent_cause" in result.explanation

    def test_low_confidence_invasive_action_fails(self):
        span = self._make_span(**{
            "attributes.confidence": "0.3",
            "attributes.decided_action": "retry_payment",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "fail"
        assert "Low confidence" in result.explanation

    def test_low_confidence_non_invasive_passes(self):
        span = self._make_span(**{
            "attributes.confidence": "0.3",
            "attributes.decided_action": "send_notification",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "pass"

    def test_error_code_contradicts_root_cause(self):
        span = self._make_span(**{
            "attributes.failure_code": "bad_request_card_expired",
            "attributes.root_cause": "insufficient_funds",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "fail"
        assert "maps to" in result.explanation

    def test_expired_in_reason_but_wrong_cause(self):
        span = self._make_span(**{
            "attributes.failure_reason": "Card has expired",
            "attributes.root_cause": "network_timeout",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "fail"
        assert "expired" in result.explanation.lower()

    def test_insufficient_in_reason_but_wrong_cause(self):
        span = self._make_span(**{
            "attributes.failure_reason": "Insufficient funds in account",
            "attributes.root_cause": "card_expired",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "fail"
        assert "insufficient" in result.explanation.lower()

    def test_evaluator_metadata_populated(self):
        span = self._make_span()
        result = self.evaluator.evaluate(span)
        assert "root_cause" in result.metadata
        assert "confidence" in result.metadata

    def test_span_ids_propagated(self):
        span = self._make_span()
        result = self.evaluator.evaluate(span)
        assert result.span_id == "span_123"
        assert result.trace_id == "trace_456"
        assert result.payment_id == "pay_test_001"


class TestToolCorrectnessEvaluator:
    def setup_method(self):
        self.evaluator = ToolCorrectnessEvaluator()

    def _make_span(self, **overrides):
        base = {
            "context.span_id": "span_789",
            "context.trace_id": "trace_012",
            "attributes.payment_id": "pay_test_002",
            "attributes.root_cause": "card_expired",
            "attributes.decided_action": "send_notification",
            "attributes.recovery_tier": "active",
            "attributes.final_status": "action_dispatched",
            "attributes.failure_code": "",
        }
        base.update(overrides)
        return base

    def test_valid_action_for_cause(self):
        span = self._make_span(**{
            "attributes.root_cause": "card_expired",
            "attributes.decided_action": "send_notification",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "pass"

    def test_invalid_action_for_cause(self):
        span = self._make_span(**{
            "attributes.root_cause": "card_expired",
            "attributes.decided_action": "retry_payment",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "fail"
        assert "not in valid set" in result.explanation

    def test_hard_decline_retry_blocked(self):
        span = self._make_span(**{
            "attributes.failure_code": "expired_card",
            "attributes.decided_action": "retry_payment",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "fail"
        assert "Hard decline" in result.explanation

    def test_card_expiry_routes_to_update(self):
        span = self._make_span(**{
            "attributes.root_cause": "card_expired",
            "attributes.decided_action": "retry_payment",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "fail"
        assert "update_payment_method" in result.explanation

    def test_user_dropoff_no_retry(self):
        span = self._make_span(**{
            "attributes.root_cause": "user_dropoff",
            "attributes.decided_action": "retry_payment",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "fail"
        assert "User dropoff" in result.explanation

    def test_user_dropoff_with_link_passes(self):
        span = self._make_span(**{
            "attributes.root_cause": "user_dropoff",
            "attributes.decided_action": "create_payment_link",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "pass"

    def test_insufficient_funds_wait_passes(self):
        span = self._make_span(**{
            "attributes.root_cause": "insufficient_funds",
            "attributes.decided_action": "wait_and_retry",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "pass"

    def test_metadata_populated(self):
        span = self._make_span()
        result = self.evaluator.evaluate(span)
        assert "root_cause" in result.metadata
        assert "action" in result.metadata


class TestComplianceEvaluator:
    def setup_method(self):
        self.evaluator = ComplianceEvaluator()

    def _make_span(self, **overrides):
        base = {
            "context.span_id": "span_comp",
            "context.trace_id": "trace_comp",
            "attributes.payment_id": "pay_test_003",
            "attributes.amount": "5000",
            "attributes.decided_action": "send_notification",
            "attributes.recovery_tier": "active",
        }
        base.update(overrides)
        return base

    def test_normal_amount_passes(self):
        span = self._make_span(**{"attributes.amount": "5000"})
        result = self.evaluator.evaluate(span)
        assert result.label == "pass"

    def test_over_cap_fails(self):
        span = self._make_span(**{
            "attributes.amount": "600000",
            "attributes.decided_action": "retry_payment",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "fail"
        assert "exceeds" in result.explanation

    def test_silent_tier_customer_action_fails(self):
        span = self._make_span(**{
            "attributes.recovery_tier": "silent",
            "attributes.decided_action": "send_notification",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "fail"
        assert "Silent tier" in result.explanation

    def test_silent_tier_internal_action_passes(self):
        span = self._make_span(**{
            "attributes.recovery_tier": "silent",
            "attributes.decided_action": "retry_payment",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "pass"

    def test_active_tier_notification_passes(self):
        span = self._make_span(**{
            "attributes.recovery_tier": "active",
            "attributes.decided_action": "send_notification",
        })
        result = self.evaluator.evaluate(span)
        assert result.label == "pass"

    def test_span_ids_propagated(self):
        span = self._make_span()
        result = self.evaluator.evaluate(span)
        assert result.span_id == "span_comp"
        assert result.payment_id == "pay_test_003"


class TestEvalResult:
    def test_eval_result_fields(self):
        r = EvalResult(
            span_id="s1", trace_id="t1", payment_id="p1",
            evaluator="test", score=0.5, label="warn",
            explanation="test explanation", metadata={"key": "val"},
        )
        assert r.span_id == "s1"
        assert r.score == 0.5
        assert r.metadata["key"] == "val"

    def test_eval_result_defaults(self):
        r = EvalResult(
            span_id="", trace_id="", payment_id="",
            evaluator="test", score=1.0, label="pass",
            explanation="ok",
        )
        assert r.metadata == {}
