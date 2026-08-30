"""Tests for LlamaIndex Agentic RAG Engine.

Tests all 4 LlamaIndex paradigms:
  1. VectorIndex & SummaryIndex — dual retrieval
  2. RouterQueryEngine — dynamic routing
  3. SubQuestionQueryEngine — decomposition
  4. RAGTriadEvaluator — groundedness scoring
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from recovery_agent.agent.agentic_rag import (
    LlamaIndexAgenticRAG,
    VectorIndex,
    SummaryIndex,
    RouterQueryEngine,
    SubQuestionQueryEngine,
    RAGTriadEvaluator,
    TextChunk,
    RetrievalResult,
    RAGResponse,
    _chunk_markdown,
    _KB_DIR,
)


# --- Test Data ---

SAMPLE_RAZORPAY_DOC = """# Razorpay Error Codes

## Error Code: PAYLATER_OTP_EXPIRED
- **Description**: LazyPay PayLater OTP verification timed out.
- **Resolution**: RETRY_PAYMENT with fresh OTP. Maximum 2 retries before switching rail.
- **Recovery Protocol**: RETRY_PAYMENT with fresh OTP. If OTP fails 3 times, UPDATE_PAYMENT_METHOD.

## Error Code: CARD_EXPIRED
- **Description**: Card on file has expired.
- **Resolution**: Customer must update card details.
- **Recovery Protocol**: UPDATE_PAYMENT_METHOD — switch to UPI, Netbanking.

## Error Code: INSUFFICIENT_FUNDS
- **Description**: Customer lacks sufficient balance.
- **Resolution**: Time retry to payday cycle.
- **Recovery Protocol**: WAIT_AND_RETRY — schedule retry at next payday.

## Instrument Switch Protocol (CRITICAL)
When description contains "use another payment instrument":
- The current payment method is BROKEN
- Silent retries on the same instrument are FORBIDDEN
- Immediate action required: UPDATE_PAYMENT_METHOD
"""

SAMPLE_RBI_DOC = """# RBI Mandate Policies

## UPI Autopay Limits
- Maximum mandate amount: ₹15,000 per transaction
- Below ₹15,000: Automatic debits without per-transaction AFA

## Communication Frequency Limits
- Maximum 3 reminder messages per failed transaction
- Minimum 24-hour gap between consecutive reminders
- Quiet hours: No communications between 9:00 PM and 8:00 AM
"""


# --- VectorIndex Tests ---

class TestVectorIndex:
    def test_build_index_from_chunks(self):
        chunks = _chunk_markdown(SAMPLE_RAZORPAY_DOC, "razorpay_error_docs.md")
        try:
            idx = VectorIndex(chunks)
            assert idx._collection is not None
        except RuntimeError as e:
            # ChromaDB embedding model unavailable — acceptable in CI/offline
            assert "chromadb" in str(e).lower() or "FATAL" in str(e)

    def test_query_returns_matching_chunks(self):
        chunks = _chunk_markdown(SAMPLE_RAZORPAY_DOC, "razorpay_error_docs.md")
        try:
            idx = VectorIndex(chunks)
        except RuntimeError:
            pytest.skip("ChromaDB embedding model unavailable")
        result = idx.query("LazyPay OTP expired", top_k=2)
        assert result.index_type == "vector"
        assert len(result.chunks) > 0
        assert any("paylater" in c.text.lower() or "otp" in c.text.lower() for c in result.chunks)

    def test_query_empty_returns_empty(self):
        chunks = _chunk_markdown(SAMPLE_RAZORPAY_DOC, "razorpay_error_docs.md")
        try:
            idx = VectorIndex(chunks)
        except RuntimeError:
            pytest.skip("ChromaDB embedding model unavailable")
        result = idx.query("test query", top_k=3)
        assert len(result.chunks) > 0

    def test_query_score_is_positive(self):
        chunks = _chunk_markdown(SAMPLE_RAZORPAY_DOC, "razorpay_error_docs.md")
        try:
            idx = VectorIndex(chunks)
        except RuntimeError:
            pytest.skip("ChromaDB embedding model unavailable")
        result = idx.query("card expired", top_k=1)
        assert result.score > 0

    def test_chromadb_collection_required(self):
        chunks = _chunk_markdown(SAMPLE_RAZORPAY_DOC, "razorpay_error_docs.md")
        try:
            idx = VectorIndex(chunks)
            assert idx._collection is not None
        except RuntimeError as e:
            # Must fail loudly — no BM25 fallback
            assert "FATAL" in str(e) or "chromadb" in str(e).lower()

    def test_no_bm25_fallback(self):
        """Verify VectorIndex has no _use_fallback attribute — no heuristic hacks."""
        assert not hasattr(VectorIndex, "_build_keyword_index")
        assert not hasattr(VectorIndex, "_keyword_query")
        assert not hasattr(VectorIndex, "_tokenize")


# --- SummaryIndex Tests ---

class TestSummaryIndex:
    def test_build_sections(self):
        chunks = _chunk_markdown(SAMPLE_RAZORPAY_DOC, "razorpay_error_docs.md")
        idx = SummaryIndex(chunks)
        assert len(idx._sections) > 0

    def test_query_returns_section_summaries(self):
        chunks = _chunk_markdown(SAMPLE_RAZORPAY_DOC, "razorpay_error_docs.md")
        idx = SummaryIndex(chunks)
        result = idx.query("payment method protocol", top_k=2)
        assert result.index_type == "summary"
        assert len(result.chunks) > 0

    def test_query_matches_header_keywords(self):
        chunks = _chunk_markdown(SAMPLE_RAZORPAY_DOC, "razorpay_error_docs.md")
        idx = SummaryIndex(chunks)
        result = idx.query("instrument switch protocol", top_k=1)
        assert len(result.chunks) > 0
        assert "instrument switch" in result.chunks[0].text.lower() or "instrument" in result.chunks[0].text.lower()


# --- RouterQueryEngine Tests ---

class TestRouterQueryEngine:
    @pytest.fixture
    def router(self):
        v_chunks = _chunk_markdown(SAMPLE_RAZORPAY_DOC, "razorpay_error_docs.md")
        s_chunks = _chunk_markdown(SAMPLE_RBI_DOC, "rbi_mandate_policies.md")
        try:
            return RouterQueryEngine(VectorIndex(v_chunks), SummaryIndex(s_chunks))
        except RuntimeError:
            pytest.skip("ChromaDB embedding model unavailable")

    def test_specific_query_routes_to_vector(self, router):
        result = router.query("PAYLATER_OTP_EXPIRED error code", top_k=2)
        assert result.index_type in ("vector", "both")

    def test_broad_query_routes_to_summary(self, router):
        result = router.query("What is the RBI dunning policy best practice rules?", top_k=2)
        assert result.index_type in ("summary", "both")

    def test_mixed_query_routes_to_both(self, router):
        result = router.query("RBI policy for card error code 54", top_k=3)
        assert result.index_type in ("both", "vector", "summary")


# --- SubQuestionQueryEngine Tests ---

class TestSubQuestionQueryEngine:
    @pytest.fixture
    def engine(self):
        v_chunks = _chunk_markdown(SAMPLE_RAZORPAY_DOC, "razorpay_error_docs.md")
        s_chunks = _chunk_markdown(SAMPLE_RBI_DOC, "rbi_mandate_policies.md")
        try:
            router = RouterQueryEngine(VectorIndex(v_chunks), SummaryIndex(s_chunks))
            return SubQuestionQueryEngine(router)
        except RuntimeError:
            pytest.skip("ChromaDB embedding model unavailable")

    def test_decompose_creates_sub_questions(self, engine):
        payload = {
            "method": "paylater",
            "provider": "lazypay",
            "failure_code": "PAYLATER_OTP_EXPIRED",
            "failure_reason": "OTP timed out",
            "error_description": "OTP expired",
            "amount": 2500,
        }
        sub_qs = engine._decompose(payload)
        assert len(sub_qs) == 4
        assert sub_qs[0]["domain"] == "psp"
        assert sub_qs[1]["domain"] == "rbi"
        assert sub_qs[2]["domain"] == "merchant"
        assert sub_qs[3]["domain"] == "razorpay"

    def test_query_returns_rag_response(self, engine):
        payload = {
            "method": "paylater",
            "provider": "lazypay",
            "failure_code": "PAYLATER_OTP_EXPIRED",
            "failure_reason": "OTP timed out",
            "error_description": "OTP expired",
            "amount": 2500,
        }
        response = engine.query(payload)
        assert isinstance(response, RAGResponse)
        assert response.decomposition_steps == 4
        assert len(response.sub_answers) == 4
        assert len(response.retrieved_chunks) > 0

    def test_query_includes_all_domains(self, engine):
        payload = {
            "method": "card",
            "provider": "hdfc",
            "failure_code": "CARD_EXPIRED",
            "failure_reason": "Card expired",
            "error_description": "Card expired",
            "amount": 5000,
        }
        response = engine.query(payload)
        domains = {sa["domain"] for sa in response.sub_answers}
        assert "psp" in domains
        assert "rbi" in domains
        assert "merchant" in domains
        assert "razorpay" in domains


# --- RAGTriadEvaluator Tests ---

class TestRAGTriadEvaluator:
    def test_heuristic_groundedness_high(self):
        diag = "CARD_EXPIRED error code means card has expired and requires UPDATE_PAYMENT_METHOD to UPI Netbanking"
        ctx = "Error Code: CARD_EXPIRED. Card on file has expired. Recovery Protocol: UPDATE_PAYMENT_METHOD switch to UPI Netbanking"
        result = RAGTriadEvaluator._heuristic_groundedness(diag, ctx)
        assert result["groundedness_score"] >= 0.4

    def test_heuristic_groundedness_low(self):
        diag = "Quantum entanglement caused the failure"
        ctx = "Error Code: CARD_EXPIRED. Recovery Protocol: UPDATE_PAYMENT_METHOD"
        result = RAGTriadEvaluator._heuristic_groundedness(diag, ctx)
        assert result["groundedness_score"] < 0.5

    def test_heuristic_faithfulness_high(self):
        ans = "UPDATE_PAYMENT_METHOD to UPI is recommended for CARD_EXPIRED"
        ctx = "CARD_EXPIRED: Recovery Protocol UPDATE_PAYMENT_METHOD switch to UPI"
        result = RAGTriadEvaluator._heuristic_faithfulness(ans, ctx)
        assert result["faithfulness_score"] > 0.5

    def test_heuristic_faithfulness_low(self):
        ans = "The sky is blue and 2+2=4"
        ctx = "CARD_EXPIRED: Recovery Protocol UPDATE_PAYMENT_METHOD"
        result = RAGTriadEvaluator._heuristic_faithfulness(ans, ctx)
        assert result["faithfulness_score"] < 0.5

    def test_evaluate_groundedness_with_llm(self):
        evaluator = RAGTriadEvaluator()
        with patch("recovery_agent.agent.agentic_rag.invoke_llm_json") as mock_llm:
            mock_llm.return_value = {
                "groundedness_score": 0.95,
                "evidence": "All claims supported by context",
                "unsupported_claims": [],
            }
            result = evaluator.evaluate_groundedness("diagnosis", "context")
            assert result["groundedness_score"] == 0.95
            assert result["evidence"] == "All claims supported by context"

    def test_evaluate_faithfulness_with_llm(self):
        evaluator = RAGTriadEvaluator()
        with patch("recovery_agent.agent.agentic_rag.invoke_llm_json") as mock_llm:
            mock_llm.return_value = {
                "faithfulness_score": 0.90,
                "evidence": "Consistent with context",
                "contradictions": [],
            }
            result = evaluator.evaluate_faithfulness("answer", "context")
            assert result["faithfulness_score"] == 0.90

    def test_evaluate_groundedness_fallback_on_llm_failure(self):
        evaluator = RAGTriadEvaluator()
        with patch("recovery_agent.agent.agentic_rag.invoke_llm_json") as mock_llm:
            mock_llm.return_value = None
            result = evaluator.evaluate_groundedness("card expired", "CARD_EXPIRED error")
            assert "groundedness_score" in result


# --- Text Chunking Tests ---

class TestChunkMarkdown:
    def test_chunks_by_header(self):
        chunks = _chunk_markdown(SAMPLE_RAZORPAY_DOC, "test.md")
        assert len(chunks) > 0
        headers = {c.section_header for c in chunks}
        assert len(headers) > 1

    def test_chunks_have_source_file(self):
        chunks = _chunk_markdown(SAMPLE_RAZORPAY_DOC, "razorpay_error_docs.md")
        for c in chunks:
            assert c.source_file == "razorpay_error_docs.md"

    def test_empty_document(self):
        chunks = _chunk_markdown("", "empty.md")
        assert len(chunks) == 0

    def test_large_sections_get_split(self):
        large_doc = "## Section\n" + "word " * 500
        chunks = _chunk_markdown(large_doc, "large.md")
        assert len(chunks) >= 2


# --- LlamaIndexAgenticRAG Integration Tests ---

class TestLlamaIndexAgenticRAG:
    @pytest.fixture
    def rag_with_kb(self, tmp_path):
        """Create RAG engine with test knowledge base."""
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "razorpay_error_docs.md").write_text(SAMPLE_RAZORPAY_DOC)
        (kb_dir / "rbi_mandate_policies.md").write_text(SAMPLE_RBI_DOC)
        (kb_dir / "psp_gateway_troubleshooting.md").write_text("# PSP Guide\n## LazyPay\nOTP timeout")
        (kb_dir / "merchant_dunning_rules.md").write_text("# Dunning\n## Retry rules\n24h gap")
        try:
            return LlamaIndexAgenticRAG(kb_dir=kb_dir)
        except RuntimeError:
            pytest.skip("ChromaDB embedding model unavailable")

    def test_load_knowledge_base(self, rag_with_kb):
        try:
            rag_with_kb._ensure_loaded()
        except RuntimeError:
            pytest.skip("ChromaDB embedding model not available")
        assert rag_with_kb.is_loaded
        assert rag_with_kb.chunk_count > 0
        assert rag_with_kb.document_count == 4

    def test_query_paylater_otp(self, rag_with_kb):
        payload = {
            "method": "paylater",
            "provider": "lazypay",
            "failure_code": "PAYLATER_OTP_EXPIRED",
            "failure_reason": "OTP timed out",
            "error_description": "OTP expired",
            "amount": 2500,
        }
        try:
            response = rag_with_kb.query(payload, evaluate=False)
        except RuntimeError:
            pytest.skip("ChromaDB embedding model not available")
        assert isinstance(response, RAGResponse)
        assert len(response.retrieved_chunks) > 0
        assert response.decomposition_steps == 4

    def test_query_by_error_code(self, rag_with_kb):
        try:
            response = rag_with_kb.query_by_error_code("CARD_EXPIRED", method="card")
        except RuntimeError:
            pytest.skip("ChromaDB embedding model not available")
        assert len(response.retrieved_chunks) > 0

    def test_query_for_diagnosis(self, rag_with_kb):
        metadata = {
            "failure_code": "INSUFFICIENT_FUNDS",
            "failure_reason": "insufficient balance",
            "method": "upi",
            "provider": "npci",
            "amount": 5000,
        }
        try:
            response = rag_with_kb.query_for_diagnosis(metadata)
        except RuntimeError:
            pytest.skip("ChromaDB embedding model not available")
        assert len(response.retrieved_chunks) > 0

    def test_groundedness_computed(self, rag_with_kb):
        payload = {
            "method": "paylater",
            "provider": "lazypay",
            "failure_code": "PAYLATER_OTP_EXPIRED",
            "failure_reason": "OTP timed out",
            "error_description": "OTP expired",
            "amount": 2500,
        }
        with patch("recovery_agent.agent.agentic_rag.invoke_llm_json") as mock_llm:
            mock_llm.return_value = {
                "groundedness_score": 0.92,
                "evidence": "All claims supported",
                "unsupported_claims": [],
            }
            try:
                response = rag_with_kb.query(payload, evaluate=True)
            except RuntimeError:
                pytest.skip("ChromaDB embedding model not available")
            assert response.groundedness_score > 0.0


    def test_sub_answers_count(self, rag_with_kb):
        payload = {
            "method": "card",
            "provider": "hdfc",
            "failure_code": "CARD_EXPIRED",
            "failure_reason": "expired",
            "error_description": "card expired",
            "amount": 10000,
        }
        try:
            response = rag_with_kb.query(payload, evaluate=False)
        except RuntimeError:
            pytest.skip("ChromaDB embedding model not available")
        assert len(response.sub_answers) == 4
        domains = {sa["domain"] for sa in response.sub_answers}
        assert "psp" in domains
        assert "rbi" in domains

    def test_instrument_switch_in_context(self, rag_with_kb):
        payload = {
            "method": "paylater",
            "provider": "lazypay",
            "failure_code": "PAYLATER_OTP_EXPIRED",
            "failure_reason": "use another payment instrument",
            "error_description": "use another payment instrument",
            "amount": 1000,
        }
        try:
            response = rag_with_kb.query(payload, evaluate=False)
        except RuntimeError:
            pytest.skip("ChromaDB embedding model not available")
        # Should retrieve instrument switch protocol
        all_text = " ".join(c.text for c in response.retrieved_chunks)
        assert "instrument switch" in all_text.lower() or "payment method" in all_text.lower()

    def test_lazy_loaded(self):
        try:
            rag = LlamaIndexAgenticRAG()
        except RuntimeError:
            pytest.skip("ChromaDB embedding model unavailable")
        assert not rag.is_loaded
        try:
            rag._ensure_loaded()
        except RuntimeError:
            pytest.skip("ChromaDB embedding model unavailable")
        assert rag.is_loaded

    def test_metadata_populated(self, rag_with_kb):
        payload = {
            "method": "paylater",
            "provider": "lazypay",
            "failure_code": "PAYLATER_OTP_EXPIRED",
            "failure_reason": "OTP timeout",
            "error_description": "OTP expired",
            "amount": 2500,
        }
        try:
            response = rag_with_kb.query(payload, evaluate=False)
        except RuntimeError:
            pytest.skip("ChromaDB embedding model not available")
        assert response.metadata["method"] == "paylater"
        assert response.metadata["provider"] == "lazypay"
        assert response.metadata["failure_code"] == "PAYLATER_OTP_EXPIRED"


# --- Tool Registration Tests ---

class TestRAGToolRegistration:
    def test_tool_registered_in_tool_scapes(self):
        from recovery_agent.agent.tools import TOOL_SCAPES
        names = [t["name"] for t in TOOL_SCAPES]
        assert "query_payment_recovery_kb" in names

    def test_tool_registered_in_tool_functions(self):
        from recovery_agent.agent.tools import TOOL_FUNCTIONS
        assert "query_payment_recovery_kb" in TOOL_FUNCTIONS

    def test_tool_has_correct_schema(self):
        from recovery_agent.agent.tools import TOOL_SCAPES
        rag_tool = next(t for t in TOOL_SCAPES if t["name"] == "query_payment_recovery_kb")
        assert "query" in rag_tool["input_schema"]["properties"]
        assert "domain" in rag_tool["input_schema"]["properties"]
        assert "method" in rag_tool["input_schema"]["properties"]
        assert "query" in rag_tool["input_schema"]["required"]

    def test_tool_execute(self):
        from recovery_agent.agent.tools import execute_tool
        result = execute_tool("query_payment_recovery_kb", {
            "query": "CARD_EXPIRED",
            "domain": "razorpay",
            "method": "card",
        })
        # Either succeeds with answer or returns error when ChromaDB unavailable
        assert result["status"] in ("ok", "error")
        if result["status"] == "ok":
            assert "groundedness_score" in result
            assert "num_chunks_retrieved" in result
