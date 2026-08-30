"""Unit tests for Guardrail Engine.

Covers: quiet hours deferral, frequency capping, double-debit lock,
opt-out compliance, monetary cap, and pre-execution interception.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recovery_agent.agent.guardrails import (
    GuardrailEngine,
    GuardrailVerdict,
    QuietHourGuardrail,
    FrequencyCapGuardrail,
    DoubleDebitLockGuardrail,
    OptOutGuardrail,
    MonetaryCapGuardrail,
)
from recovery_agent.models import (
    ActionType,
    Attempt,
    Case,
    CustomerProfile,
    PaymentEvent,
    PaymentRecord,
)


# --- Quiet Hours ---

class TestQuietHourGuardrail:
    def test_daytime_send_notification_passes(self):
        gh = QuietHourGuardrail()
        now = datetime(2026, 8, 25, 14, 0, tzinfo=timezone.utc)  # 2 PM
        result = gh.check(ActionType.SEND_NOTIFICATION, now=now)
        assert result.verdict == GuardrailVerdict.PASS

    def test_nighttime_send_notification_deferred(self):
        gh = QuietHourGuardrail()
        now = datetime(2026, 8, 25, 22, 0, tzinfo=timezone.utc)  # 10 PM
        result = gh.check(ActionType.SEND_NOTIFICATION, now=now)
        assert result.verdict == GuardrailVerdict.MODIFIED
        assert result.modified_action == ActionType.WAIT_AND_RETRY.value

    def test_early_morning_deferred(self):
        gh = QuietHourGuardrail()
        now = datetime(2026, 8, 25, 6, 0, tzinfo=timezone.utc)  # 6 AM
        result = gh.check(ActionType.SEND_NOTIFICATION, now=now)
        assert result.verdict == GuardrailVerdict.MODIFIED

    def test_8am边界_passes(self):
        gh = QuietHourGuardrail()
        now = datetime(2026, 8, 25, 8, 0, tzinfo=timezone.utc)  # 8 AM
        result = gh.check(ActionType.SEND_NOTIFICATION, now=now)
        assert result.verdict == GuardrailVerdict.PASS

    def test_9pm边界_deferred(self):
        gh = QuietHourGuardrail()
        now = datetime(2026, 8, 25, 21, 0, tzinfo=timezone.utc)  # 9 PM
        result = gh.check(ActionType.SEND_NOTIFICATION, now=now)
        assert result.verdict == GuardrailVerdict.MODIFIED

    def test_retry_payment_not_affected_by_quiet_hours(self):
        gh = QuietHourGuardrail()
        now = datetime(2026, 8, 25, 22, 0, tzinfo=timezone.utc)  # 10 PM
        result = gh.check(ActionType.RETRY_PAYMENT, now=now)
        assert result.verdict == GuardrailVerdict.PASS

    def test_escalate_not_affected_by_quiet_hours(self):
        gh = QuietHourGuardrail()
        now = datetime(2026, 8, 25, 22, 0, tzinfo=timezone.utc)
        result = gh.check(ActionType.ESCALATE_TO_HUMAN, now=now)
        assert result.verdict == GuardrailVerdict.PASS

    def test_update_payment_method_deferred(self):
        gh = QuietHourGuardrail()
        now = datetime(2026, 8, 25, 23, 30, tzinfo=timezone.utc)
        result = gh.check(ActionType.UPDATE_PAYMENT_METHOD, now=now)
        assert result.verdict == GuardrailVerdict.MODIFIED
        assert result.modified_action == ActionType.WAIT_AND_RETRY.value


# --- Frequency Cap ---

class TestFrequencyCapGuardrail:
    def test_no_profile_allows_action(self):
        fc = FrequencyCapGuardrail(max_contacts_per_24h=2)
        result = fc.check(ActionType.SEND_NOTIFICATION, profile=None)
        assert result.verdict == GuardrailVerdict.PASS

    def test_under_cap_allows(self):
        fc = FrequencyCapGuardrail(max_contacts_per_24h=2)
        profile = CustomerProfile(
            customer_id="cust_001",
            payment_history=[
                PaymentRecord(
                    payment_id="p1", amount=100, channel_used="sms",
                    status="failed",
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=12),
                ),
            ],
        )
        result = fc.check(ActionType.SEND_NOTIFICATION, profile=profile)
        assert result.verdict == GuardrailVerdict.PASS

    def test_at_cap_blocks(self):
        fc = FrequencyCapGuardrail(max_contacts_per_24h=2)
        now = datetime.now(timezone.utc)
        profile = CustomerProfile(
            customer_id="cust_001",
            payment_history=[
                PaymentRecord(
                    payment_id="p1", amount=100, channel_used="sms",
                    status="failed", timestamp=now - timedelta(hours=1),
                ),
                PaymentRecord(
                    payment_id="p2", amount=100, channel_used="email",
                    status="failed", timestamp=now - timedelta(hours=2),
                ),
            ],
        )
        result = fc.check(ActionType.SEND_NOTIFICATION, profile=profile)
        assert result.verdict == GuardrailVerdict.BLOCKED

    def test_non_communication_not_affected(self):
        fc = FrequencyCapGuardrail(max_contacts_per_24h=1)
        now = datetime.now(timezone.utc)
        profile = CustomerProfile(
            customer_id="cust_001",
            payment_history=[
                PaymentRecord(
                    payment_id="p1", amount=100, channel_used="sms",
                    status="failed", timestamp=now - timedelta(hours=1),
                ),
            ],
        )
        result = fc.check(ActionType.RETRY_PAYMENT, profile=profile)
        assert result.verdict == GuardrailVerdict.PASS

    def test_old_contacts_not_counted(self):
        fc = FrequencyCapGuardrail(max_contacts_per_24h=2)
        now = datetime.now(timezone.utc)
        profile = CustomerProfile(
            customer_id="cust_001",
            payment_history=[
                PaymentRecord(
                    payment_id="p1", amount=100, channel_used="sms",
                    status="failed", timestamp=now - timedelta(hours=25),
                ),
                PaymentRecord(
                    payment_id="p2", amount=100, channel_used="email",
                    status="failed", timestamp=now - timedelta(hours=26),
                ),
            ],
        )
        result = fc.check(ActionType.SEND_NOTIFICATION, profile=profile)
        assert result.verdict == GuardrailVerdict.PASS


# --- Double-Debit Lock ---

class TestDoubleDebitLockGuardrail:
    def test_non_retry_action_passes(self):
        ddl = DoubleDebitLockGuardrail()
        result = ddl.check(ActionType.SEND_NOTIFICATION)
        assert result.verdict == GuardrailVerdict.PASS

    def test_no_case_allows_retry(self):
        ddl = DoubleDebitLockGuardrail()
        result = ddl.check(ActionType.RETRY_PAYMENT, case=None)
        assert result.verdict == GuardrailVerdict.PASS

    def test_previous_success_blocks_retry(self):
        ddl = DoubleDebitLockGuardrail()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="test",
            ),
            attempts=[
                Attempt(
                    action_type=ActionType.RETRY_PAYMENT,
                    result="success",
                ),
            ],
            attempt_count=1,
        )
        result = ddl.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.BLOCKED
        assert "already succeeded" in result.reason

    def test_pending_payment_blocks_retry(self):
        ddl = DoubleDebitLockGuardrail()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="test",
            ),
            attempts=[
                Attempt(
                    action_type=ActionType.RETRY_PAYMENT,
                    result="pending",
                ),
            ],
            attempt_count=1,
        )
        result = ddl.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.BLOCKED
        assert "pending" in result.reason

    def test_failed_attempt_allows_retry(self):
        ddl = DoubleDebitLockGuardrail()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=5000, failure_reason="test",
            ),
            attempts=[
                Attempt(
                    action_type=ActionType.RETRY_PAYMENT,
                    result="failed",
                ),
            ],
            attempt_count=1,
        )
        result = ddl.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.PASS


# --- Opt-Out ---

class TestOptOutGuardrail:
    def test_not_opted_out_allows(self):
        og = OptOutGuardrail()
        profile = CustomerProfile(customer_id="cust_001", opt_out=False)
        result = og.check(ActionType.SEND_NOTIFICATION, profile=profile)
        assert result.verdict == GuardrailVerdict.PASS

    def test_opted_out_blocks_notification(self):
        og = OptOutGuardrail()
        profile = CustomerProfile(customer_id="cust_001", opt_out=True)
        result = og.check(ActionType.SEND_NOTIFICATION, profile=profile)
        assert result.verdict == GuardrailVerdict.BLOCKED

    def test_opted_out_blocks_update_payment(self):
        og = OptOutGuardrail()
        profile = CustomerProfile(customer_id="cust_001", opt_out=True)
        result = og.check(ActionType.UPDATE_PAYMENT_METHOD, profile=profile)
        assert result.verdict == GuardrailVerdict.BLOCKED

    def test_opted_out_allows_retry(self):
        og = OptOutGuardrail()
        profile = CustomerProfile(customer_id="cust_001", opt_out=True)
        result = og.check(ActionType.RETRY_PAYMENT, profile=profile)
        assert result.verdict == GuardrailVerdict.PASS

    def test_opted_out_allows_escalate(self):
        og = OptOutGuardrail()
        profile = CustomerProfile(customer_id="cust_001", opt_out=True)
        result = og.check(ActionType.ESCALATE_TO_HUMAN, profile=profile)
        assert result.verdict == GuardrailVerdict.PASS

    def test_no_profile_allows(self):
        og = OptOutGuardrail()
        result = og.check(ActionType.SEND_NOTIFICATION, profile=None)
        assert result.verdict == GuardrailVerdict.PASS


# --- Monetary Cap ---

class TestMonetaryCapGuardrail:
    def test_non_retry_action_passes(self):
        mc = MonetaryCapGuardrail(max_single_retry=500_000)
        result = mc.check(ActionType.SEND_NOTIFICATION)
        assert result.verdict == GuardrailVerdict.PASS

    def test_amount_within_cap(self):
        mc = MonetaryCapGuardrail(max_single_retry=500_000)
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=100_000, failure_reason="test",
            ),
        )
        result = mc.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.PASS

    def test_amount_exceeds_cap(self):
        mc = MonetaryCapGuardrail(max_single_retry=500_000)
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=600_000, failure_reason="test",
            ),
        )
        result = mc.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.BLOCKED
        assert "exceeds" in result.reason

    def test_exact_cap_amount_passes(self):
        mc = MonetaryCapGuardrail(max_single_retry=500_000)
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=500_000, failure_reason="test",
            ),
        )
        result = mc.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.PASS


# --- GuardrailEngine Integration ---

class TestGuardrailEngine:
    def test_all_pass_returns_original_action(self):
        engine = GuardrailEngine()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=1000, failure_reason="test",
            ),
        )
        now = datetime(2026, 8, 25, 14, 0, tzinfo=timezone.utc)  # 2 PM
        action, checks = engine.validate_action(case, ActionType.RETRY_PAYMENT, now=now)
        assert action == ActionType.RETRY_PAYMENT
        assert all(c.verdict == GuardrailVerdict.PASS for c in checks)

    def test_quiet_hours_modifies_action(self):
        engine = GuardrailEngine()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=1000, failure_reason="test",
            ),
        )
        now = datetime(2026, 8, 25, 22, 0, tzinfo=timezone.utc)  # 10 PM
        action, checks = engine.validate_action(
            case, ActionType.SEND_NOTIFICATION, now=now,
        )
        # Should be modified to WAIT_AND_RETRY (quiet hours) then possibly further
        assert action != ActionType.SEND_NOTIFICATION

    def test_opt_out_blocks_and_falls_to_wait(self):
        engine = GuardrailEngine()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=1000, failure_reason="test",
            ),
        )
        profile = CustomerProfile(customer_id="cust_001", opt_out=True)
        action, checks = engine.validate_action(
            case, ActionType.SEND_NOTIFICATION, profile=profile,
        )
        assert action == ActionType.WAIT_AND_RETRY

    def test_monetary_cap_blocks_and_falls_to_escalate(self):
        engine = GuardrailEngine()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=600_000, failure_reason="test",
            ),
        )
        action, checks = engine.validate_action(
            case, ActionType.RETRY_PAYMENT,
        )
        assert action == ActionType.ESCALATE_TO_HUMAN

    def test_guardrail_results_stored_in_metadata(self):
        engine = GuardrailEngine()
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=1000, failure_reason="test",
            ),
        )
        now = datetime(2026, 8, 25, 14, 0, tzinfo=timezone.utc)
        engine.validate_action(case, ActionType.RETRY_PAYMENT, now=now)
        assert "guardrail_checks" in case.payment.metadata
        assert "guardrail_final_action" in case.payment.metadata

    def test_backward_compatible_without_guardrails(self):
        """run_execution without guardrail_engine should work unchanged."""
        from recovery_agent.agent.execution import run_execution
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1", customer_id="cust_001",
                amount=1000, failure_reason="network timeout",
                failure_code="network_timeout",
            ),
        )
        from recovery_agent.agent.diagnosis import run_diagnosis
        from recovery_agent.agent.decision import run_decision
        case = run_diagnosis(case)
        case = run_decision(case)
        result = run_execution(case)
        assert result.attempt_count == 1
