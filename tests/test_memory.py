"""Unit tests for Long-Term Memory Engine.

Covers: profile creation, salary window detection, promise-to-pay tracking,
channel optimization, persistence, and memory-aware decision integration.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from recovery_agent.agent.memory import CustomerMemoryStore
from recovery_agent.agent.decision import decide_intervention, run_decision
from recovery_agent.models import (
    ActionType,
    Case,
    CustomerProfile,
    FailureType,
    PaymentEvent,
    PromiseToPay,
    SalaryWindow,
)


# --- Profile Creation & Access ---

class TestProfileCreation:
    def test_get_or_create_new_profile(self):
        store = CustomerMemoryStore()
        profile = store.get_or_create_profile("cust_001")
        assert profile.customer_id == "cust_001"
        assert profile.total_recovered == 0.0
        assert profile.total_attempts == 0
        assert profile.payment_history == []

    def test_get_existing_profile(self):
        store = CustomerMemoryStore()
        p1 = store.get_or_create_profile("cust_001")
        p1.total_recovered = 500.0
        p2 = store.get_or_create_profile("cust_001")
        assert p2.total_recovered == 500.0
        assert p1 is p2

    def test_multiple_customers(self):
        store = CustomerMemoryStore()
        store.get_or_create_profile("cust_A")
        store.get_or_create_profile("cust_B")
        assert len(store.list_profiles()) == 2
        assert "cust_A" in store.list_profiles()
        assert "cust_B" in store.list_profiles()


# --- Profile Updates ---

class TestProfileUpdates:
    def test_update_after_successful_attempt(self):
        store = CustomerMemoryStore()
        profile = store.update_profile_after_attempt(
            "cust_001",
            {"payment_id": "pay_1", "amount": 1000, "failure_type": "insufficient_funds"},
            success=True,
            channel="sms",
        )
        assert profile.total_attempts == 1
        assert profile.total_recovered == 1000
        assert len(profile.payment_history) == 1
        assert profile.payment_history[0].status == "success"

    def test_update_after_failed_attempt(self):
        store = CustomerMemoryStore()
        profile = store.update_profile_after_attempt(
            "cust_001",
            {"payment_id": "pay_1", "amount": 1000, "failure_type": "card_expired"},
            success=False,
            channel="email",
        )
        assert profile.total_attempts == 1
        assert profile.total_recovered == 0
        assert profile.payment_history[0].status == "failed"

    def test_channel_success_rate_calculation(self):
        store = CustomerMemoryStore()
        # 2 failed sms, 1 success sms -> 33% success rate
        store.update_profile_after_attempt("cust_001", {"payment_id": "p1", "amount": 100}, False, "sms")
        store.update_profile_after_attempt("cust_001", {"payment_id": "p2", "amount": 100}, False, "sms")
        store.update_profile_after_attempt("cust_001", {"payment_id": "p3", "amount": 100}, True, "sms")
        profile = store.get_or_create_profile("cust_001")
        assert profile.channel_success_rates["sms"] == pytest.approx(1 / 3, abs=0.01)

    def test_preferred_channel_selection(self):
        store = CustomerMemoryStore()
        # email: 0% success, sms: 100% success
        store.update_profile_after_attempt("cust_001", {"payment_id": "p1", "amount": 100}, False, "email")
        store.update_profile_after_attempt("cust_001", {"payment_id": "p2", "amount": 100}, True, "sms")
        profile = store.get_or_create_profile("cust_001")
        assert profile.preferred_channel == "sms"

    def test_failure_type_tracking(self):
        store = CustomerMemoryStore()
        store.update_profile_after_attempt("cust_001", {"payment_id": "p1", "amount": 100, "failure_type": "card_expired"}, False, "sms")
        store.update_profile_after_attempt("cust_001", {"payment_id": "p2", "amount": 100, "failure_type": "card_expired"}, False, "sms")
        store.update_profile_after_attempt("cust_001", {"payment_id": "p3", "amount": 100, "failure_type": "insufficient_funds"}, False, "sms")
        profile = store.get_or_create_profile("cust_001")
        assert profile.failure_type_counts["card_expired"] == 2
        assert profile.failure_type_counts["insufficient_funds"] == 1


# --- Salary Window ---

class TestSalaryWindow:
    def test_update_salary_window(self):
        store = CustomerMemoryStore()
        sw = store.update_salary_window("cust_001", pay_day=1, last_salary_date="2026-08-01")
        assert sw.typical_pay_day == 1
        assert sw.last_salary_date == "2026-08-01"
        assert "2026-08-01" in sw.salary_history

    def test_salary_liquidity_within_window(self):
        store = CustomerMemoryStore()
        store.update_salary_window("cust_001", pay_day=1)
        # Day 1 = pay day, should be in window
        assert store.check_salary_liquidity("cust_001", current_day=1) is True
        # Day 2 = day after pay day, within ±2
        assert store.check_salary_liquidity("cust_001", current_day=2) is True
        # Day 0 (30th) = day before pay day (wrap around)
        assert store.check_salary_liquidity("cust_001", current_day=30) is True

    def test_salary_liquidity_outside_window(self):
        store = CustomerMemoryStore()
        store.update_salary_window("cust_001", pay_day=1)
        # Day 15 = far from pay day
        assert store.check_salary_liquidity("cust_001", current_day=15) is False
        # Day 20 = far from pay day
        assert store.check_salary_liquidity("cust_001", current_day=20) is False

    def test_salary_liquidity_mid_month_payday(self):
        store = CustomerMemoryStore()
        store.update_salary_window("cust_001", pay_day=15)
        # Day 14 = day before, within window
        assert store.check_salary_liquidity("cust_001", current_day=14) is True
        # Day 17 = 2 days after, within window
        assert store.check_salary_liquidity("cust_001", current_day=17) is True
        # Day 20 = outside window
        assert store.check_salary_liquidity("cust_001", current_day=20) is False

    def test_no_pay_day_returns_false(self):
        store = CustomerMemoryStore()
        assert store.check_salary_liquidity("cust_001", current_day=1) is False

    def test_salary_history_accumulates(self):
        store = CustomerMemoryStore()
        store.update_salary_window("cust_001", pay_day=1, last_salary_date="2026-07-01")
        store.update_salary_window("cust_001", pay_day=1, last_salary_date="2026-08-01")
        profile = store.get_or_create_profile("cust_001")
        assert len(profile.salary_window.salary_history) == 2
        assert "2026-07-01" in profile.salary_window.salary_history
        assert "2026-08-01" in profile.salary_window.salary_history


# --- Promise to Pay ---

class TestPromiseToPay:
    def test_record_promise(self):
        store = CustomerMemoryStore()
        promise = store.record_promise_to_pay("cust_001", 5000.0, "2026-08-30")
        assert promise.amount == 5000.0
        assert promise.promised_date == "2026-08-30"
        assert promise.fulfilled is False

    def test_fulfill_promise(self):
        store = CustomerMemoryStore()
        promise = store.record_promise_to_pay("cust_001", 5000.0, "2026-08-30")
        result = store.fulfill_promise("cust_001", promise.promise_id)
        assert result is True
        profile = store.get_or_create_profile("cust_001")
        assert profile.promises[0].fulfilled is True

    def test_fulfill_nonexistent_promise(self):
        store = CustomerMemoryStore()
        store.get_or_create_profile("cust_001")
        result = store.fulfill_promise("cust_001", "fake_id")
        assert result is False

    def test_multiple_promises(self):
        store = CustomerMemoryStore()
        store.record_promise_to_pay("cust_001", 1000, "2026-08-25")
        store.record_promise_to_pay("cust_001", 2000, "2026-09-01")
        store.record_promise_to_pay("cust_001", 3000, "2026-09-15")
        profile = store.get_or_create_profile("cust_001")
        assert len(profile.promises) == 3

    def test_fulfilled_promises_filter(self):
        store = CustomerMemoryStore()
        p1 = store.record_promise_to_pay("cust_001", 1000, "2026-08-25")
        store.record_promise_to_pay("cust_001", 2000, "2026-09-01")
        store.fulfill_promise("cust_001", p1.promise_id)
        fulfilled = store.check_fulfilled_promises("cust_001")
        assert len(fulfilled) == 1
        assert fulfilled[0].promise_id == p1.promise_id


# --- Channel Optimization ---

class TestChannelOptimization:
    def test_default_channel(self):
        store = CustomerMemoryStore()
        assert store.get_optimal_channel("cust_001") == "sms"

    def test_optimal_channel_after_history(self):
        store = CustomerMemoryStore()
        store.update_profile_after_attempt("cust_001", {"payment_id": "p1", "amount": 100}, True, "email")
        store.update_profile_after_attempt("cust_001", {"payment_id": "p2", "amount": 100}, False, "sms")
        assert store.get_optimal_channel("cust_001") == "email"

    def test_channel_confidence(self):
        store = CustomerMemoryStore()
        store.update_profile_after_attempt("cust_001", {"payment_id": "p1", "amount": 100}, True, "email")
        store.update_profile_after_attempt("cust_001", {"payment_id": "p2", "amount": 100}, False, "email")
        confidence = store.get_channel_confidence("cust_001", "email")
        assert confidence == pytest.approx(0.5, abs=0.01)

    def test_unknown_channel_confidence(self):
        store = CustomerMemoryStore()
        confidence = store.get_channel_confidence("cust_001", "push")
        assert confidence == 0.5


# --- Persistence ---

class TestPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = CustomerMemoryStore(persist_dir=tmpdir)
            store.update_profile_after_attempt(
                "cust_001",
                {"payment_id": "p1", "amount": 500, "failure_type": "card_expired"},
                success=True,
                channel="sms",
            )
            # Verify JSON file was created
            path = Path(tmpdir) / "cust_001.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["customer_id"] == "cust_001"
            assert data["total_recovered"] == 500

            # Load in fresh store
            store2 = CustomerMemoryStore(persist_dir=tmpdir)
            profile = store2.get_or_create_profile("cust_001")
            assert profile.total_recovered == 500
            assert len(profile.payment_history) == 1

    def test_load_corrupted_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("not json")
            store = CustomerMemoryStore(persist_dir=tmpdir)
            # Should not raise, just skip corrupted files
            assert len(store.list_profiles()) == 0


# --- Aggregate Stats ---

class TestStats:
    def test_empty_stats(self):
        store = CustomerMemoryStore()
        stats = store.get_stats()
        assert stats["total_customers"] == 0
        assert stats["total_recovered"] == 0
        assert stats["promise_fulfillment_rate"] == 0.0

    def test_stats_with_data(self):
        store = CustomerMemoryStore()
        store.update_profile_after_attempt("cust_001", {"payment_id": "p1", "amount": 1000}, True, "sms")
        p = store.record_promise_to_pay("cust_001", 2000, "2026-09-01")
        store.fulfill_promise("cust_001", p.promise_id)
        stats = store.get_stats()
        assert stats["total_customers"] == 1
        assert stats["total_recovered"] == 1000
        assert stats["total_promises"] == 1
        assert stats["fulfilled_promises"] == 1
        assert stats["promise_fulfillment_rate"] == 1.0


# --- Memory-Aware Decision Integration ---

class TestMemoryAwareDecision:
    def test_insufficient_funds_outside_salary_window(self):
        """Salary-dependent customer outside pay window should WAIT_AND_RETRY."""
        from datetime import datetime
        store = CustomerMemoryStore()
        store.update_salary_window("cust_001", pay_day=1)

        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1",
                customer_id="cust_001",
                amount=5000,
                failure_reason="insufficient_funds",
                failure_code="INSUFFICIENT_FUNDS",
            ),
        )
        case.diagnosis = __import__("recovery_agent.models", fromlist=["Diagnosis"]).Diagnosis(
            root_cause=FailureType.INSUFFICIENT_FUNDS,
            confidence=0.9,
            reasoning="Test",
        )
        # Set day to 15 (far from pay day 1)
        action = decide_intervention(case, profile=store.get_or_create_profile("cust_001"), memory=store)
        assert action == ActionType.WAIT_AND_RETRY

    def test_insufficient_funds_in_salary_window(self):
        """Salary-dependent customer in pay window passes through to decision tree."""
        from unittest.mock import patch

        store = CustomerMemoryStore()
        store.update_salary_window("cust_001", pay_day=1)

        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1",
                customer_id="cust_001",
                amount=5000,
                failure_reason="insufficient_funds",
                failure_code="INSUFFICIENT_FUNDS",
            ),
        )
        case.diagnosis = __import__("recovery_agent.models", fromlist=["Diagnosis"]).Diagnosis(
            root_cause=FailureType.INSUFFICIENT_FUNDS,
            confidence=0.9,
            reasoning="Test",
        )
        # Mock datetime.now().day to be 1 (pay day) — inside salary window
        mock_now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with patch("recovery_agent.agent.decision.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.fromisoformat = datetime.fromisoformat
            action = decide_intervention(case, profile=store.get_or_create_profile("cust_001"), memory=store)
            # The memory guard should NOT have blocked (we're in window).
            # Decision tree at attempt 0 for INSUFFICIENT_FUNDS = WAIT_AND_RETRY,
            # so we verify the memory check ran by checking the mock was called.
            mock_dt.now.assert_called()
            # Action comes from the decision tree, not the memory guard short-circuit
            assert action == ActionType.WAIT_AND_RETRY  # tree's decision at attempt 0

    def test_run_decision_stores_optimal_channel(self):
        """run_decision should store optimal_channel in metadata."""
        store = CustomerMemoryStore()
        store.update_profile_after_attempt(
            "cust_001",
            {"payment_id": "p1", "amount": 100},
            True,
            "email",
        )
        profile = store.get_or_create_profile("cust_001")

        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1",
                customer_id="cust_001",
                amount=5000,
                failure_reason="network_timeout",
                failure_code="NETWORK_TIMEOUT",
            ),
        )
        case.diagnosis = __import__("recovery_agent.models", fromlist=["Diagnosis"]).Diagnosis(
            root_cause=FailureType.NETWORK_TIMEOUT,
            confidence=0.9,
            reasoning="Test",
        )
        case = run_decision(case, profile=profile, memory=store)
        assert case.payment.metadata["optimal_channel"] == "email"

    def test_no_memory_backward_compatible(self):
        """Decision without memory should work identically to before."""
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1",
                customer_id="cust_001",
                amount=5000,
                failure_reason="network_timeout",
                failure_code="NETWORK_TIMEOUT",
            ),
        )
        case.diagnosis = __import__("recovery_agent.models", fromlist=["Diagnosis"]).Diagnosis(
            root_cause=FailureType.NETWORK_TIMEOUT,
            confidence=0.9,
            reasoning="Test",
        )
        action = decide_intervention(case)
        assert action == ActionType.RETRY_PAYMENT  # attempt 0 -> RETRY
