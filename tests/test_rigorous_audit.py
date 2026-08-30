"""Rigorous Lifecycle & Flaw Audit Tests.

Stress-tests AutoRecover against the 5 real-world financial failure scenarios
documented in RIGOROUS-SYSTEM-FLAW-AUDIT.md:
1. Unmapped third-party PSP errors (NPCI_UPI_Z9, HDFC_NETBANK_503).
2. Reflection & re-diagnosis on intervention failure.
3. Out-of-order webhook delivery & high latency spikes.
4. Quiet hours deferral & guardrail action modification.
5. Dispute events (payment.dispute.created) and partial payment handling.
"""
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from recovery_agent.agent.diagnosis import run_diagnosis
from recovery_agent.agent.guardrails import GuardrailEngine, GuardrailVerdict
from recovery_agent.agent.kg_router import RazorpayKnowledgeGraph
from recovery_agent.agent.memory import CustomerMemoryStore
from recovery_agent.agent.squad import SquadOrchestrator
from recovery_agent.models import (
    ActionType,
    Case,
    CaseStatus,
    CustomerProfile,
    FailureType,
    PaymentEvent,
)
from recovery_agent.webhook import process_webhook_payload


# --- Test 1: Unmapped Third-Party PSP Errors ---

class TestUnmappedPSPErrors:
    def test_npci_upi_z9_unmapped_error(self):
        """Test raw NPCI UPI error code not in standard dictionary."""
        event = PaymentEvent(
            payment_id="pay_npci_z9",
            order_id="order_npci_1",
            customer_id="cust_npci_1",
            amount=1500.0,
            status="failed",
            failure_code="NPCI_UPI_Z9_TIMED_OUT",
            error_description="NPCI PSP timed out during UPI PIN verification",
        )
        case = Case(payment=event)
        case = run_diagnosis(case)

        assert case.diagnosis is not None
        # Should gracefully map via keyword fallback or Layer 3 to NETWORK_TIMEOUT or UNKNOWN
        assert case.diagnosis.root_cause in (FailureType.NETWORK_TIMEOUT, FailureType.UNKNOWN)
        assert case.diagnosis.confidence > 0.0

    def test_hdfc_netbank_503_raw_log(self):
        """Test raw HDFC netbanking HTTP 503 gateway log."""
        event = PaymentEvent(
            payment_id="pay_hdfc_503",
            order_id="order_hdfc_1",
            customer_id="cust_hdfc_1",
            amount=5000.0,
            status="failed",
            failure_code="HDFC_NETBANK_HTTP_503_SERVICE_UNAVAILABLE",
            error_description="HDFC Netbanking gateway returned HTTP 503 Service Unavailable",
        )
        case = Case(payment=event)
        case = run_diagnosis(case)

        assert case.diagnosis is not None
        assert case.diagnosis.root_cause in (FailureType.BANK_DECLINED, FailureType.NETWORK_TIMEOUT, FailureType.UNKNOWN)


# --- Test 2: Reflection & Re-Diagnosis on Intervention Failure ---

class TestReflectionAndRediagnosis:
    def test_rediagnosis_after_intervention_failure(self):
        """Test that after an intervention fails, updated attempt history is passed to squad."""
        memory = CustomerMemoryStore()
        kg = RazorpayKnowledgeGraph()
        guardrails = GuardrailEngine()
        squad = SquadOrchestrator(memory_store=memory, kg_router=kg, guardrail_engine=guardrails)

        event = PaymentEvent(
            payment_id="pay_reflect_1",
            order_id="order_reflect_1",
            customer_id="cust_reflect_1",
            amount=2500.0,
            status="failed",
            failure_code="BAD_REQUEST_PAYMENT_FAILED",
            error_description="Generic payment failure",
        )
        case = Case(payment=event)
        profile = memory.get_or_create_profile(case.payment.customer_id)

        # Step 1: Initial squad step
        result1 = squad.run_step(case, profile=profile)
        assert result1.next_case.attempt_count == 1

        # Simulate attempt 1 failure in memory
        memory.update_profile_after_attempt(
            customer_id=profile.customer_id,
            attempt={"payment_id": "pay_reflect_1", "amount": 2500.0, "failure_type": "bank_declined"},
            success=False,
            channel="sms",
        )

        # Step 2: Second squad step with memory context
        result2 = squad.run_step(result1.next_case, profile=profile)
        assert result2.next_case.attempt_count == 2
        # Verify trajectory step recorded diagnostic reasoning
        assert "diagnosis" in result2.trajectory_step


# --- Test 3: Out-of-Order Webhooks & Latency ---

class TestOutOfOrderWebhooks:
    def test_out_of_order_order_paid_before_failed(self):
        """Test order.paid arriving before payment.failed."""
        # 1. Order paid arrives first
        paid_payload = {
            "event": "order.paid",
            "payload": {
                "order": {
                    "entity": {
                        "id": "order_ooo_100",
                        "amount": 350000,
                        "status": "paid",
                    }
                }
            }
        }
        res1 = process_webhook_payload(paid_payload)
        assert res1["status"] in ("ignored", "handled", "success")

        # 2. Late payment.failed arrives 10 seconds later
        failed_payload = {
            "event": "payment.failed",
            "payload": {
                "payment": {
                    "entity": {
                        "id": "pay_ooo_100",
                        "order_id": "order_ooo_100",
                        "amount": 350000,
                        "status": "failed",
                        "error_code": "BAD_REQUEST_PAYMENT_TIMED_OUT",
                        "error_description": "Network timeout",
                        "notes": {"customer_id": "cust_ooo_100"},
                    }
                }
            }
        }
        res2 = process_webhook_payload(failed_payload)
        assert res2 is not None


# --- Test 4: Quiet Hours Deferral & Guardrail Actions ---

class TestQuietHoursGuardrail:
    def test_quiet_hours_defers_notification_to_wait(self):
        """Test that at 11 PM UTC (quiet hours), SEND_NOTIFICATION is deferred to WAIT_AND_RETRY."""
        guardrails = GuardrailEngine()
        event = PaymentEvent(
            payment_id="pay_quiet_1",
            order_id="order_quiet_1",
            customer_id="cust_quiet_1",
            amount=1200.0,
            status="failed",
            failure_code="BAD_REQUEST_PAYMENT_CARD_EXPIRED",
        )
        case = Case(payment=event)
        profile = CustomerProfile(customer_id="cust_quiet_1")

        night_time = datetime(2026, 8, 25, 23, 0, 0, tzinfo=timezone.utc)
        approved_action, checks = guardrails.validate_action(
            case=case,
            action=ActionType.SEND_NOTIFICATION,
            profile=profile,
            now=night_time,
        )

        quiet_check = next(c for c in checks if c.guardrail == "quiet_hours")
        assert quiet_check.verdict == GuardrailVerdict.MODIFIED
        assert quiet_check.modified_action == ActionType.WAIT_AND_RETRY.value
        # Final action may differ due to subsequent guardrails (e.g. semantic LLM),
        # but quiet hours check itself must have modified to WAIT_AND_RETRY


# --- Test 5: Dispute Webhooks & Partial Payments ---

class TestDisputeAndPartialPayments:
    def test_payment_dispute_created_webhook(self):
        """Test payment.dispute.created webhook event."""
        dispute_payload = {
            "event": "payment.dispute.created",
            "payload": {
                "dispute": {
                    "entity": {
                        "id": "disp_100",
                        "payment_id": "pay_disputed_1",
                        "amount": 500000,
                        "status": "open",
                        "reason_code": "fraudulent",
                    }
                }
            }
        }
        result = process_webhook_payload(dispute_payload)
        assert result["event"] == "payment.dispute.created"
        assert result["status"] == "handled"

    def test_double_debit_lock_blocks_duplicate_retry(self):
        """Test that double debit lock blocks RETRY_PAYMENT if a payment succeeded."""
        guardrails = GuardrailEngine()
        event = PaymentEvent(
            payment_id="pay_dd_1",
            order_id="order_dd_1",
            customer_id="cust_dd_1",
            amount=4000.0,
            status="failed",
            failure_code="BAD_REQUEST_PAYMENT_TIMED_OUT",
        )
        case = Case(payment=event)
        # Add a successful attempt to case
        from recovery_agent.models import Attempt
        case.attempts.append(Attempt(action_type=ActionType.RETRY_PAYMENT, result="success"))

        approved_action, checks = guardrails.validate_action(
            case=case,
            action=ActionType.RETRY_PAYMENT,
            profile=None,
        )

        assert approved_action == ActionType.ESCALATE_TO_HUMAN
        dd_check = next(c for c in checks if c.guardrail == "double_debit_lock")
        assert dd_check.verdict == GuardrailVerdict.BLOCKED


# --- Test 6: Webhook Idempotency ---

class TestWebhookIdempotency:
    def test_duplicate_event_id_returns_duplicate(self):
        """Test that duplicate event_id returns 200 with status=duplicate."""
        from recovery_agent.webhook import _is_duplicate_event, _processed_events, _event_lock
        from datetime import datetime, timezone

        # Reset state
        with _event_lock:
            _processed_events.clear()

        event_id = "evt_test_dup_001"

        # First call — not duplicate
        assert _is_duplicate_event(event_id) is False

        # Second call — duplicate
        assert _is_duplicate_event(event_id) is True

        # Third call — still duplicate
        assert _is_duplicate_event(event_id) is True

    def test_different_event_ids_not_duplicate(self):
        """Test that different event_ids are not considered duplicates."""
        from recovery_agent.webhook import _is_duplicate_event, _processed_events, _event_lock

        with _event_lock:
            _processed_events.clear()

        assert _is_duplicate_event("evt_001") is False
        assert _is_duplicate_event("evt_002") is False
        assert _is_duplicate_event("evt_001") is True  # duplicate of first
        assert _is_duplicate_event("evt_002") is True  # duplicate of second

    def test_empty_event_id_not_deduplicated(self):
        """Test that empty event_id is allowed through (can't deduplicate)."""
        from recovery_agent.webhook import _is_duplicate_event, _processed_events, _event_lock

        with _event_lock:
            _processed_events.clear()

        # Empty event_id should always return False (not duplicate)
        assert _is_duplicate_event("") is False
        assert _is_duplicate_event("") is False
        assert _is_duplicate_event("") is False

    def test_idempotency_expiry_after_ttl(self):
        """Test that old event_ids expire after TTL window."""
        from recovery_agent.webhook import _is_duplicate_event, _processed_events, _event_lock, _IDEMPOTENCY_TTL_SECONDS

        with _event_lock:
            _processed_events.clear()

        event_id = "evt_ttl_test"

        # Record event with old timestamp (beyond TTL)
        from datetime import datetime, timezone, timedelta
        with _event_lock:
            _processed_events[event_id] = datetime.now(timezone.utc) - timedelta(seconds=_IDEMPOTENCY_TTL_SECONDS + 1)

        # Should NOT be considered duplicate (expired)
        assert _is_duplicate_event(event_id) is False
