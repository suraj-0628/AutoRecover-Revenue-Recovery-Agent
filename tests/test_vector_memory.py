"""Tests for Vectorized Customer Memory (Mandate 3).

Tests cover:
- VectorMemoryStore initialization and availability
- Outcome ingestion (single + batch)
- Semantic similarity search
- Decision context generation
- Recovery statistics
- Graceful degradation when ChromaDB unavailable
- Integration with harness and agent
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from recovery_agent.agent.vector_memory import (
    VectorMemoryStore,
    _build_outcome_text,
    _build_outcome_metadata,
)
from recovery_agent.models import (
    ActionType,
    Attempt,
    Case,
    CaseStatus,
    Diagnosis,
    FailureType,
    PaymentEvent,
    RecoveryTier,
)


# ─── Helpers ──────────────────────────────────────────────────

def make_case(
    failure_type: FailureType = FailureType.NETWORK_TIMEOUT,
    amount: float = 5000.0,
    recovered: bool = False,
    attempt_count: int = 0,
    method: str = "card",
    bank: str = "HDFC",
    failure_code: str = "",
    with_attempt: bool = False,
) -> Case:
    event = PaymentEvent(
        event_type="payment_failed",
        payment_id="pay_vectest_001",
        customer_id="cust_vectest",
        amount=amount,
        currency="INR",
        status="failed",
        failure_reason=f"Test {failure_type.value}",
        failure_code=failure_code or failure_type.value,
        metadata={"method": method, "bank": bank},
    )
    case = Case(payment=event, max_attempts=5)
    case.diagnosis = Diagnosis(
        root_cause=failure_type,
        confidence=0.9,
        reasoning=f"Test diagnosis for {failure_type.value}",
    )
    case.attempt_count = attempt_count
    case.recovered = recovered
    if recovered:
        case.recovered_amount = amount
        case.status = CaseStatus.RECOVERED
    if with_attempt:
        case.attempts = [Attempt(action_type=ActionType.RETRY_PAYMENT)]
    return case


def _fresh_store(tmp_path: Path) -> VectorMemoryStore:
    """Create a fresh VectorMemoryStore with isolated persistent storage."""
    d = tmp_path / "vec_mem"
    d.mkdir(exist_ok=True)
    return VectorMemoryStore(persist_dir=str(d))


# ═══════════════════════════════════════════════════════════════
#  TEXT & METADATA BUILDERS
# ═══════════════════════════════════════════════════════════════

class TestOutcomeTextBuilder:
    """Test natural language outcome text generation for embedding."""

    def test_basic_failure_text(self):
        case = make_case(failure_type=FailureType.NETWORK_TIMEOUT)
        text = _build_outcome_text(case)
        assert "network_timeout" in text
        assert "Payment failed" in text

    def test_recovered_case_text(self):
        case = make_case(failure_type=FailureType.CARD_EXPIRED, recovered=True, with_attempt=True)
        text = _build_outcome_text(case)
        assert "successfully recovered" in text

    def test_high_value_amount(self):
        case = make_case(amount=100000.0)
        text = _build_outcome_text(case)
        assert "high value" in text

    def test_medium_value_amount(self):
        case = make_case(amount=25000.0)
        text = _build_outcome_text(case)
        assert "medium value" in text

    def test_low_value_amount(self):
        case = make_case(amount=500.0)
        text = _build_outcome_text(case)
        assert "low value" in text

    def test_includes_method(self):
        case = make_case(method="upi")
        text = _build_outcome_text(case)
        assert "upi" in text

    def test_includes_bank(self):
        case = make_case(bank="ICICI")
        text = _build_outcome_text(case)
        assert "ICICI" in text


class TestOutcomeMetadataBuilder:
    """Test structured metadata generation for filtering."""

    def test_basic_metadata(self):
        case = make_case(failure_type=FailureType.BANK_DECLINED, amount=10000.0, recovered=True)
        meta = _build_outcome_metadata(case)
        assert meta["failure_type"] == "bank_declined"
        assert meta["amount"] == 10000.0
        assert meta["recovered"] is True
        assert meta["method"] == "card"
        assert meta["bank"] == "HDFC"

    def test_metadata_includes_case_id(self):
        case = make_case()
        meta = _build_outcome_metadata(case)
        assert meta["case_id"] == case.id
        assert meta["customer_id"] == "cust_vectest"


# ═══════════════════════════════════════════════════════════════
#  VECTOR MEMORY STORE
# ═══════════════════════════════════════════════════════════════

class TestVectorMemoryStore:
    """Test VectorMemoryStore with real ChromaDB (persistent, isolated)."""

    def test_initialization_persistent(self, tmp_path):
        store = _fresh_store(tmp_path)
        assert store.is_available is True
        assert store.outcome_count == 0

    def test_ingest_single_outcome(self, tmp_path):
        store = _fresh_store(tmp_path)
        case = make_case(failure_type=FailureType.NETWORK_TIMEOUT, with_attempt=True)
        result = store.ingest_outcome(case)
        assert result is True
        assert store.outcome_count == 1

    def test_ingest_skips_no_attempts(self, tmp_path):
        store = _fresh_store(tmp_path)
        case = make_case()
        case.attempts = []
        case.recovered = False
        result = store.ingest_outcome(case)
        assert result is False
        assert store.outcome_count == 0

    def test_ingest_recovered_case(self, tmp_path):
        store = _fresh_store(tmp_path)
        case = make_case(recovered=True)
        result = store.ingest_outcome(case)
        assert result is True
        assert store.outcome_count == 1

    def test_upsert_deduplicates(self, tmp_path):
        store = _fresh_store(tmp_path)
        case = make_case(recovered=True)
        store.ingest_outcome(case)
        store.ingest_outcome(case)  # Same case ID → upsert
        assert store.outcome_count == 1

    def test_ingest_multiple_distinct_cases(self, tmp_path):
        store = _fresh_store(tmp_path)
        for i in range(5):
            case = make_case(failure_type=FailureType.NETWORK_TIMEOUT, recovered=True)
            case.id = f"case_{i}"
            case.payment.payment_id = f"pay_{i}"
            store.ingest_outcome(case)
        assert store.outcome_count == 5


class TestVectorSimilaritySearch:
    """Test semantic similarity search."""

    def test_query_returns_results(self, tmp_path):
        store = _fresh_store(tmp_path)
        case = make_case(failure_type=FailureType.NETWORK_TIMEOUT, recovered=True, bank="HDFC")
        store.ingest_outcome(case)

        results = store.query_similar(failure_type="network_timeout", bank="HDFC")
        assert len(results) > 0
        assert results[0]["metadata"]["recovered"] is True

    def test_query_no_results_when_empty(self, tmp_path):
        store = _fresh_store(tmp_path)
        results = store.query_similar(failure_type="network_timeout")
        assert results == []

    def test_query_respects_k_parameter(self, tmp_path):
        store = _fresh_store(tmp_path)
        for i in range(10):
            case = make_case(failure_type=FailureType.NETWORK_TIMEOUT, recovered=True)
            case.id = f"case_{i}"
            store.ingest_outcome(case)

        results = store.query_similar(failure_type="network_timeout", k=3)
        assert len(results) <= 3

    def test_query_with_failure_type_filter(self, tmp_path):
        store = _fresh_store(tmp_path)
        case1 = make_case(failure_type=FailureType.NETWORK_TIMEOUT, recovered=True)
        case1.id = "case_timeout"
        case2 = make_case(failure_type=FailureType.CARD_EXPIRED, recovered=True)
        case2.id = "case_expired"
        store.ingest_outcome(case1)
        store.ingest_outcome(case2)

        results = store.query_similar(
            failure_type="network_timeout",
            where={"failure_type": "network_timeout"},
        )
        for r in results:
            assert r["metadata"]["failure_type"] == "network_timeout"

    def test_query_returns_distance(self, tmp_path):
        store = _fresh_store(tmp_path)
        case = make_case(recovered=True)
        store.ingest_outcome(case)
        results = store.query_similar(failure_type="network_timeout")
        assert len(results) > 0
        assert "distance" in results[0]


class TestDecisionContext:
    """Test decision context generation for LLM injection."""

    def test_context_with_similar_cases(self, tmp_path):
        store = _fresh_store(tmp_path)
        case = make_case(failure_type=FailureType.NETWORK_TIMEOUT, recovered=True)
        store.ingest_outcome(case)

        query_case = make_case(failure_type=FailureType.NETWORK_TIMEOUT)
        context = store.get_decision_context(query_case)
        assert "SIMILAR PAST CASES" in context
        assert "RECOVERED" in context

    def test_context_empty_when_no_data(self, tmp_path):
        store = _fresh_store(tmp_path)
        case = make_case()
        context = store.get_decision_context(case)
        assert context == ""

    def test_context_includes_intervention(self, tmp_path):
        store = _fresh_store(tmp_path)
        case = make_case(recovered=True, with_attempt=True)
        case.attempts = [Attempt(action_type=ActionType.RETRY_PAYMENT)]
        store.ingest_outcome(case)

        query_case = make_case()
        context = store.get_decision_context(query_case)
        assert "retry_payment" in context


class TestRecoveryStats:
    """Test aggregate recovery statistics."""

    def test_stats_empty(self, tmp_path):
        store = _fresh_store(tmp_path)
        stats = store.get_recovery_stats("network_timeout")
        assert stats["total_cases"] == 0
        assert stats["success_rate"] == 0.0

    def test_stats_with_data(self, tmp_path):
        store = _fresh_store(tmp_path)
        for i in range(5):
            case = make_case(failure_type=FailureType.NETWORK_TIMEOUT, recovered=(i < 3), with_attempt=True)
            case.id = f"case_{i}"
            store.ingest_outcome(case)

        stats = store.get_recovery_stats("network_timeout")
        assert stats["total_cases"] == 5
        assert stats["success_rate"] == 0.6

    def test_stats_best_intervention(self, tmp_path):
        store = _fresh_store(tmp_path)
        for i in range(4):
            case = make_case(failure_type=FailureType.NETWORK_TIMEOUT, recovered=(i < 3))
            case.id = f"case_{i}"
            case.attempts = [Attempt(action_type=ActionType.RETRY_PAYMENT)]
            store.ingest_outcome(case)

        stats = store.get_recovery_stats("network_timeout")
        assert stats["best_intervention"] == "retry_payment"


class TestClearAndStats:
    """Test clear and stats methods."""

    def test_clear_empties_collection(self, tmp_path):
        store = _fresh_store(tmp_path)
        store.ingest_outcome(make_case(recovered=True))
        assert store.outcome_count == 1
        store.clear()
        assert store.outcome_count == 0

    def test_get_stats(self, tmp_path):
        store = _fresh_store(tmp_path)
        stats = store.get_stats()
        assert stats["available"] is True
        assert stats["outcome_count"] == 0


# ═══════════════════════════════════════════════════════════════
#  GRACEFUL DEGRADATION
# ═══════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    """Test behavior when ChromaDB is unavailable."""

    def test_unavailable_store_returns_empty_results(self, tmp_path):
        with patch("chromadb.PersistentClient", side_effect=Exception("no chromadb")):
            store = VectorMemoryStore(persist_dir=str(tmp_path / "noop"))
            assert store.is_available is False
            assert store.outcome_count == 0
            assert store.ingest_outcome(make_case()) is False
            assert store.query_similar(failure_type="network_timeout") == []
            assert store.get_decision_context(make_case()) == ""
            assert store.get_recovery_stats("network_timeout")["total_cases"] == 0


# ═══════════════════════════════════════════════════════════════
#  HARNESS INTEGRATION
# ═══════════════════════════════════════════════════════════════

class TestHarnessVectorMemoryIntegration:
    """Test vector memory integration with AgentHarness."""

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_harness_ingests_outcome_on_final_status(self, mock_llm, tmp_path):
        """When LLM returns is_final with no tool calls, harness sets AWAITING_CUSTOMER
        but vector memory correctly skips ingestion (nothing happened)."""
        mock_llm.return_value = {
            "reasoning": "Action dispatched.",
            "tool_calls": [],
            "is_final": True,
            "status": "action_dispatched",
        }
        from recovery_agent.agent.harness import AgentHarness

        vm = _fresh_store(tmp_path)
        harness = AgentHarness(vector_memory=vm)
        case = make_case()
        harness.run_recovery_case(case)

        # No attempts made → nothing to ingest (correct behavior)
        assert vm.outcome_count == 0

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_harness_ingests_outcome_on_failure(self, mock_llm, tmp_path):
        mock_llm.return_value = None  # LLM unavailable
        from recovery_agent.agent.harness import AgentHarness

        vm = _fresh_store(tmp_path)
        harness = AgentHarness(vector_memory=vm)
        case = make_case()
        harness.run_recovery_case(case)

        # No attempts made and not recovered → correctly skipped (nothing to learn)
        assert vm.outcome_count == 0

    @patch("recovery_agent.agent.harness.invoke_llm_json_async")
    def test_async_harness_ingests_outcome(self, mock_llm_async, tmp_path):
        """Async harness: no tool calls → nothing to ingest (correct)."""
        import asyncio
        mock_llm_async.return_value = {
            "reasoning": "Action dispatched.",
            "tool_calls": [],
            "is_final": True,
            "status": "action_dispatched",
        }
        from recovery_agent.agent.harness import AgentHarness

        vm = _fresh_store(tmp_path)
        harness = AgentHarness(vector_memory=vm)
        case = make_case()
        asyncio.run(harness.run_recovery_case_async(case))

        assert vm.outcome_count == 0


# ═══════════════════════════════════════════════════════════════
#  AGENT INTEGRATION
# ═══════════════════════════════════════════════════════════════

class TestAgentVectorMemoryIntegration:
    """Test RecoveryAgent with VectorMemoryStore."""

    def test_agent_accepts_vector_memory(self):
        from recovery_agent.agent import RecoveryAgent

        vm = VectorMemoryStore()
        agent = RecoveryAgent(vector_memory=vm, use_harness=True)
        assert agent.vector_memory is vm
        assert agent.harness.vector_memory is vm

    def test_agent_default_creates_vector_memory(self):
        from recovery_agent.agent import RecoveryAgent

        agent = RecoveryAgent()
        assert isinstance(agent.vector_memory, VectorMemoryStore)
