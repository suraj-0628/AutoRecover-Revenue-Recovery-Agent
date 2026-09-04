"""Tests for guardrail engine — verify each safety policy.

These tests verify the 6 guardrails that protect against dangerous actions.
If any guardrail breaks, the agent could: send messages at 3 AM, spam customers,
double-debit payments, contact opted-out customers, or retry hard declines.

Each test verifies a specific safety boundary.
"""
from datetime import datetime, timedelta, timezone

from recovery_agent.agent.guardrails import (
    DoubleDebitLockGuardrail,
    FrequencyCapGuardrail,
    GuardrailEngine,
    GuardrailVerdict,
    HardDeclineGuardrail,
    MonetaryCapGuardrail,
    OptOutGuardrail,
    QuietHourGuardrail,
)
from recovery_agent.models import (
    ActionType,
    Attempt,
    Case,
    CustomerProfile,
    PaymentEvent,
    PaymentRecord,
    RecoveryTier,
)


def make_case(
    failure_code: str = "51",
    amount: float = 10000,
    tier: RecoveryTier = RecoveryTier.SILENT,
    attempts: list | None = None,
) -> Case:
    case = Case(
        payment=PaymentEvent(
            payment_id="pay_test",
            customer_id="cust_test",
            amount=amount,
            failure_code=failure_code,
        ),
        recovery_tier=tier,
    )
    if attempts:
        case.attempts = attempts
    return case


def make_profile(
    opt_out: bool = False,
    recent_contacts: int = 0,
) -> CustomerProfile:
    profile = CustomerProfile(customer_id="cust_test", opt_out=opt_out)
    now = datetime.now(timezone.utc)
    for i in range(recent_contacts):
        profile.payment_history.append(
            PaymentRecord(
                payment_id=f"pay_{i}",
                amount=1000,
                channel_used="sms",
                status="failed",
                timestamp=now - timedelta(hours=i),
            )
        )
    return profile


# --- Quiet Hours ---

class TestQuietHours:
    def test白天_passes(self):
        guardrail = QuietHourGuardrail()
        now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)  # 10 AM
        result = guardrail.check(ActionType.SEND_NOTIFICATION, now=now)
        assert result.verdict == GuardrailVerdict.PASS

    def test_night_deferred(self):
        """A ringing phone is what quiet hours are for. Email is not deferred —
        it waits in an inbox — so the deferred action here is the voice call."""
        guardrail = QuietHourGuardrail()
        now = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)  # 10 PM
        result = guardrail.check(ActionType.VOICE_CALL, now=now)
        assert result.verdict == GuardrailVerdict.MODIFIED
        assert result.modified_action == ActionType.WAIT_AND_RETRY.value

    def test_an_email_is_not_deferred_at_night(self):
        """It wakes nobody. Blocking it stalled a live 06:30 recovery."""
        guardrail = QuietHourGuardrail()
        now = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)
        result = guardrail.check(ActionType.SEND_NOTIFICATION, now=now)
        assert result.verdict == GuardrailVerdict.PASS

    def test_early_morning_deferred(self):
        guardrail = QuietHourGuardrail()
        now = datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)  # 3 AM
        result = guardrail.check(ActionType.VOICE_CALL, now=now)
        assert result.verdict == GuardrailVerdict.MODIFIED

    def test_8am_passes(self):
        guardrail = QuietHourGuardrail()
        now = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)  # 8 AM
        result = guardrail.check(ActionType.SEND_NOTIFICATION, now=now)
        assert result.verdict == GuardrailVerdict.PASS

    def test_9pm_deferred(self):
        guardrail = QuietHourGuardrail()
        now = datetime(2026, 1, 1, 21, 0, tzinfo=timezone.utc)  # 9 PM
        result = guardrail.check(ActionType.VOICE_CALL, now=now)
        assert result.verdict == GuardrailVerdict.MODIFIED

    def test_retry_not_deferred_at_night(self):
        guardrail = QuietHourGuardrail()
        now = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)  # 10 PM
        result = guardrail.check(ActionType.RETRY_PAYMENT, now=now)
        assert result.verdict == GuardrailVerdict.PASS


# --- Frequency Cap ---

class TestFrequencyCap:
    def test_no_profile_allows(self):
        guardrail = FrequencyCapGuardrail()
        result = guardrail.check(ActionType.SEND_NOTIFICATION, profile=None)
        assert result.verdict == GuardrailVerdict.PASS

    def test_under_cap_passes(self):
        guardrail = FrequencyCapGuardrail()
        profile = make_profile(recent_contacts=1)
        result = guardrail.check(ActionType.SEND_NOTIFICATION, profile=profile)
        assert result.verdict == GuardrailVerdict.PASS

    def test_at_cap_blocks(self):
        guardrail = FrequencyCapGuardrail()
        profile = make_profile(recent_contacts=2)
        result = guardrail.check(ActionType.SEND_NOTIFICATION, profile=profile)
        assert result.verdict == GuardrailVerdict.BLOCKED

    def test_non_communication_skips_cap(self):
        guardrail = FrequencyCapGuardrail()
        profile = make_profile(recent_contacts=10)
        result = guardrail.check(ActionType.RETRY_PAYMENT, profile=profile)
        assert result.verdict == GuardrailVerdict.PASS


# --- Double Debit Lock ---

class TestDoubleDebitLock:
    def test_non_retry_passes(self):
        guardrail = DoubleDebitLockGuardrail()
        result = guardrail.check(ActionType.SEND_NOTIFICATION)
        assert result.verdict == GuardrailVerdict.PASS

    def test_no_case_passes(self):
        guardrail = DoubleDebitLockGuardrail()
        result = guardrail.check(ActionType.RETRY_PAYMENT, case=None)
        assert result.verdict == GuardrailVerdict.PASS

    def test_already_succeeded_blocks(self):
        guardrail = DoubleDebitLockGuardrail()
        case = make_case(attempts=[
            Attempt(action_type=ActionType.RETRY_PAYMENT, result="success"),
        ])
        result = guardrail.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.BLOCKED

    def test_pending_retry_blocks(self):
        guardrail = DoubleDebitLockGuardrail()
        case = make_case(attempts=[
            Attempt(action_type=ActionType.RETRY_PAYMENT, result="pending"),
        ])
        result = guardrail.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.BLOCKED

    def test_no_prior_attempts_passes(self):
        guardrail = DoubleDebitLockGuardrail()
        case = make_case(attempts=[])
        result = guardrail.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.PASS


# --- Opt-Out ---

class TestOptOut:
    def test_not_opted_out_passes(self):
        guardrail = OptOutGuardrail()
        profile = make_profile(opt_out=False)
        result = guardrail.check(ActionType.SEND_NOTIFICATION, profile=profile)
        assert result.verdict == GuardrailVerdict.PASS

    def test_opted_out_blocks_notification(self):
        guardrail = OptOutGuardrail()
        profile = make_profile(opt_out=True)
        result = guardrail.check(ActionType.SEND_NOTIFICATION, profile=profile)
        assert result.verdict == GuardrailVerdict.BLOCKED

    def test_opted_out_blocks_update_payment(self):
        guardrail = OptOutGuardrail()
        profile = make_profile(opt_out=True)
        result = guardrail.check(ActionType.UPDATE_PAYMENT_METHOD, profile=profile)
        assert result.verdict == GuardrailVerdict.BLOCKED

    def test_opted_out_allows_retry(self):
        guardrail = OptOutGuardrail()
        profile = make_profile(opt_out=True)
        result = guardrail.check(ActionType.RETRY_PAYMENT, profile=profile)
        assert result.verdict == GuardrailVerdict.PASS

    def test_no_profile_passes(self):
        guardrail = OptOutGuardrail()
        result = guardrail.check(ActionType.SEND_NOTIFICATION, profile=None)
        assert result.verdict == GuardrailVerdict.PASS


# --- Monetary Cap ---

class TestMonetaryCap:
    def test_under_cap_passes(self):
        guardrail = MonetaryCapGuardrail()
        case = make_case(amount=100000)
        result = guardrail.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.PASS

    def test_over_cap_blocks(self):
        guardrail = MonetaryCapGuardrail()
        case = make_case(amount=600000)
        result = guardrail.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.BLOCKED

    def test_non_retry_skips_cap(self):
        guardrail = MonetaryCapGuardrail()
        case = make_case(amount=9999999)
        result = guardrail.check(ActionType.SEND_NOTIFICATION, case=case)
        assert result.verdict == GuardrailVerdict.PASS

    def test_exactly_at_cap_passes(self):
        guardrail = MonetaryCapGuardrail(max_single_retry=500000)
        case = make_case(amount=500000)
        result = guardrail.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.PASS


# --- Hard Decline ---

class TestHardDecline:
    def test_hard_decline_code_blocks_retry(self):
        guardrail = HardDeclineGuardrail()
        case = make_case(failure_code="41")
        result = guardrail.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.BLOCKED
        assert result.modified_action == ActionType.ESCALATE_TO_HUMAN.value

    def test_hard_decline_increments_penalties_prevented(self):
        guardrail = HardDeclineGuardrail()
        case = make_case(failure_code="43")
        guardrail.check(ActionType.RETRY_PAYMENT, case=case)
        assert case.penalties_prevented == 1

    def test_non_hard_decline_passes(self):
        guardrail = HardDeclineGuardrail()
        case = make_case(failure_code="51")
        result = guardrail.check(ActionType.RETRY_PAYMENT, case=case)
        assert result.verdict == GuardrailVerdict.PASS

    def test_non_retry_action_skips_check(self):
        guardrail = HardDeclineGuardrail()
        case = make_case(failure_code="41")
        result = guardrail.check(ActionType.SEND_NOTIFICATION, case=case)
        assert result.verdict == GuardrailVerdict.PASS

    def test_wait_and_retry_on_hard_decline_blocks(self):
        guardrail = HardDeclineGuardrail()
        case = make_case(failure_code="57")
        result = guardrail.check(ActionType.WAIT_AND_RETRY, case=case)
        assert result.verdict == GuardrailVerdict.BLOCKED


# --- GuardrailEngine Integration ---

class TestGuardrailEngine:
    def test_hard_decline_overrides_all(self):
        """Hard decline check runs first and blocks retry_payment on code 41."""
        engine = GuardrailEngine()
        case = make_case(failure_code="41")
        action, checks = engine.validate_action(case, ActionType.RETRY_PAYMENT)
        # Should be escalated, not retried
        assert action == ActionType.ESCALATE_TO_HUMAN

    def test_quiet_hours_modifies_a_voice_call(self):
        """Quiet hours defer what INTERRUPTS. A call becomes WAIT_AND_RETRY."""
        engine = GuardrailEngine()
        case = make_case()
        now = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)  # 10 PM
        action, checks = engine.validate_action(case, ActionType.VOICE_CALL, now=now)
        assert action == ActionType.WAIT_AND_RETRY

    def test_quiet_hours_leave_an_email_alone(self):
        """An email waits to be opened, so overnight it still goes. The SMS leg
        is suppressed inside the dispatcher, not by refusing the whole message."""
        engine = GuardrailEngine()
        case = make_case()
        now = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)
        action, checks = engine.validate_action(case, ActionType.SEND_NOTIFICATION, now=now)
        assert action == ActionType.SEND_NOTIFICATION

    def test_safe_action_passes_all_guardrails(self):
        """RETRY_PAYMENT on a normal case passes all guardrails."""
        engine = GuardrailEngine()
        case = make_case(failure_code="51", amount=10000)
        action, checks = engine.validate_action(case, ActionType.RETRY_PAYMENT)
        assert action == ActionType.RETRY_PAYMENT
        assert all(c.verdict != GuardrailVerdict.BLOCKED for c in checks)
