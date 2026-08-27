"""LlamaIndex-Style Agentic RAG Engine for AutoRecover.

Implements 4 LlamaIndex paradigms:
  1. VectorIndex & SummaryIndex — dual retrieval over knowledge base
  2. RouterQueryEngine — dynamic routing between indexes
  3. SubQuestionQueryEngine — multi-step query decomposition
  4. RAGTriadEvaluator — groundedness & faithfulness scoring

All retrieval is deterministic (keyword/BM25-style) — no external embedding API needed.
Groundedness evaluation uses LLM-as-a-Judge (Nemotron via OmniRoute).

Architecture: DeepLearning.AI "Building Agentic RAG with LlamaIndex"
Source: https://www.deeplearning.ai/courses/building-agentic-rag-with-llamaindex
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recovery_agent.agent.llm_client import invoke_llm_json

# --- Constants ---

_KB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "knowledge_base"

_DOMAIN_FILES = {
    "razorpay": "razorpay_error_docs.md",
    "rbi": "rbi_mandate_policies.md",
    "psp": "psp_gateway_troubleshooting.md",
    "merchant": "merchant_dunning_rules.md",
}


# --- Data Classes ---

@dataclass
class TextChunk:
    """A single chunk of text from the knowledge base."""
    text: str
    source_file: str
    section_header: str
    chunk_index: int
    token_count: int = 0

    def __post_init__(self):
        if not self.token_count:
            self.token_count = len(self.text.split())


@dataclass
class RetrievalResult:
    """Result from a single index retrieval."""
    chunks: list[TextChunk]
    index_type: str  # "vector" | "summary"
    query: str
    score: float = 0.0


@dataclass
class RAGResponse:
    """Final response from the Agentic RAG engine."""
    answer: str
    groundedness_score: float
    faithfulness_score: float
    retrieved_chunks: list[TextChunk]
    sub_answers: list[dict[str, Any]]
    routed_index: str
    decomposition_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)


# --- Text Chunking ---

def _chunk_markdown(text: str, source_file: str, max_chunk_tokens: int = 256) -> list[TextChunk]:
    """Split markdown into chunks by section headers, respecting max token size."""
    chunks: list[TextChunk] = []
    current_header = "Introduction"
    current_section: list[str] = []
    chunk_idx = 0

    for line in text.split("\n"):
        if line.startswith("## ") or line.startswith("### "):
            # Flush current section
            if current_section:
                section_text = "\n".join(current_section).strip()
                if section_text:
                    # Split large sections into sub-chunks
                    words = section_text.split()
                    for i in range(0, len(words), max_chunk_tokens):
                        sub_chunk = " ".join(words[i : i + max_chunk_tokens])
                        chunks.append(TextChunk(
                            text=sub_chunk,
                            source_file=source_file,
                            section_header=current_header,
                            chunk_index=chunk_idx,
                        ))
                        chunk_idx += 1
            current_header = line.lstrip("#").strip()
            current_section = [line]
        else:
            current_section.append(line)

    # Flush final section
    if current_section:
        section_text = "\n".join(current_section).strip()
        if section_text:
            words = section_text.split()
            for i in range(0, len(words), max_chunk_tokens):
                sub_chunk = " ".join(words[i : i + max_chunk_tokens])
                chunks.append(TextChunk(
                    text=sub_chunk,
                    source_file=source_file,
                    section_header=current_header,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1

    return chunks


# --- Index Implementations ---

class VectorIndex:
    """Keyword/BM25-style retrieval index.

    Uses term frequency scoring (TF) over chunked documents.
    No external embedding API required — pure Python.
    """

    def __init__(self, chunks: list[TextChunk]):
        self.chunks = chunks
        self._index: dict[str, set[int]] = {}  # term -> set of chunk indices
        self._build_index()

    def _build_index(self):
        for i, chunk in enumerate(self.chunks):
            terms = self._tokenize(chunk.text)
            for term in terms:
                if term not in self._index:
                    self._index[term] = set()
                self._index[term].add(i)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase tokenization with stop-word removal."""
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
            "either", "neither", "each", "every", "all", "any", "few", "more",
            "most", "other", "some", "such", "no", "only", "own", "same",
            "than", "too", "very", "just", "because", "if", "when", "where",
            "how", "what", "which", "who", "whom", "this", "that", "these",
            "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
            "his", "she", "her", "it", "its", "they", "them", "their",
        }
        words = re.findall(r"[a-z0-9_]+", text.lower())
        return [w for w in words if w not in stop_words and len(w) > 1]

    def query(self, query_text: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve top-k chunks by BM25-style TF scoring."""
        query_terms = self._tokenize(query_text)
        if not query_terms:
            return RetrievalResult(chunks=[], index_type="vector", query=query_text)

        # Score each chunk
        scores: dict[int, float] = {}
        for term in query_terms:
            if term in self._index:
                df = len(self._index[term])  # document frequency
                idf = math.log((len(self.chunks) + 1) / (df + 1)) + 1
                for chunk_idx in self._index[term]:
                    chunk_terms = self._tokenize(self.chunks[chunk_idx].text)
                    tf = chunk_terms.count(term) / max(len(chunk_terms), 1)
                    scores[chunk_idx] = scores.get(chunk_idx, 0) + tf * idf

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        result_chunks = [self.chunks[idx] for idx, _ in ranked]
        max_score = ranked[0][1] if ranked else 0.0

        return RetrievalResult(
            chunks=result_chunks,
            index_type="vector",
            query=query_text,
            score=max_score,
        )


class SummaryIndex:
    """High-level summary retrieval index.

    Groups chunks by section and returns the section header + first chunk
    as a summary for broad strategy selection.
    """

    def __init__(self, chunks: list[TextChunk]):
        self.chunks = chunks
        self._sections: dict[str, list[TextChunk]] = {}
        self._build_sections()

    def _build_sections(self):
        for chunk in self.chunks:
            header = chunk.section_header
            if header not in self._sections:
                self._sections[header] = []
            self._sections[header].append(chunk)

    def query(self, query_text: str, top_k: int = 3) -> RetrievalResult:
        """Retrieve top-k section summaries matching query keywords."""
        query_terms = set(re.findall(r"[a-z0-9_]+", query_text.lower()))
        section_scores: dict[str, float] = {}

        for header, section_chunks in self._sections.items():
            header_terms = set(re.findall(r"[a-z0-9_]+", header.lower()))
            all_text = " ".join(c.text for c in section_chunks[:2])
            content_terms = set(re.findall(r"[a-z0-9_]+", all_text.lower()))

            # Score: overlap with header + content
            header_overlap = len(query_terms & header_terms)
            content_overlap = len(query_terms & content_terms)
            section_scores[header] = header_overlap * 2.0 + content_overlap * 0.5

        ranked = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result_chunks: list[TextChunk] = []
        for header, _ in ranked:
            section_chunks = self._sections[header]
            # Return header as summary + first chunk
            summary_text = f"[{header}] {section_chunks[0].text[:300]}"
            result_chunks.append(TextChunk(
                text=summary_text,
                source_file=section_chunks[0].source_file,
                section_header=header,
                chunk_index=0,
            ))

        max_score = ranked[0][1] if ranked else 0.0
        return RetrievalResult(
            chunks=result_chunks,
            index_type="summary",
            query=query_text,
            score=max_score,
        )


# --- Router Query Engine ---

class RouterQueryEngine:
    """Dynamically routes queries between VectorIndex and SummaryIndex.

    Decision logic:
      - Specific error codes / PSP names → VectorIndex (precision retrieval)
      - Broad policy / strategy questions → SummaryIndex (overview retrieval)
      - Mixed queries → Both indexes, merge results
    """

    # Keywords that indicate specific (vector) queries
    SPECIFIC_KEYWORDS = {
        "error", "code", "failed", "declined", "timeout", "otp", "expired",
        "invalid", "insufficient", "mandate", "hdfc", "icici", "sbi", "lazypay",
        "paylater", "upi", "card", "netbanking", "npci", "visa", "mastercard",
        "rupay", "3ds", "authorization", "authentication", "gateway",
    }

    # Keywords that indicate broad (summary) queries
    BROAD_KEYWORDS = {
        "policy", "rules", "best", "practice", "strategy", "dunning",
        "discount", "communication", "retry", "schedule", "when", "how",
        "compliance", "regulation", "rbi", "guideline", "cap", "limit",
    }

    def __init__(self, vector_index: VectorIndex, summary_index: SummaryIndex):
        self.vector_index = vector_index
        self.summary_index = summary_index

    def _route(self, query: str) -> str:
        """Determine which index to route to."""
        query_lower = query.lower()
        terms = set(re.findall(r"[a-z0-9_]+", query_lower))

        specific_hits = len(terms & self.SPECIFIC_KEYWORDS)
        broad_hits = len(terms & self.BROAD_KEYWORDS)

        if specific_hits > broad_hits:
            return "vector"
        elif broad_hits > specific_hits:
            return "summary"
        else:
            return "both"

    def query(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Route query and return merged results."""
        route = self._route(query)

        if route == "vector":
            return self.vector_index.query(query, top_k=top_k)
        elif route == "summary":
            return self.summary_index.query(query, top_k=top_k)
        else:
            # Both: merge results
            v_result = self.vector_index.query(query, top_k=top_k)
            s_result = self.summary_index.query(query, top_k=top_k)
            merged_chunks = v_result.chunks + s_result.chunks
            # Deduplicate by text hash
            seen: set[str] = set()
            unique_chunks: list[TextChunk] = []
            for chunk in merged_chunks:
                h = hashlib.md5(chunk.text.encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    unique_chunks.append(chunk)
            return RetrievalResult(
                chunks=unique_chunks[:top_k],
                index_type="both",
                query=query,
                score=max(v_result.score, s_result.score),
            )


# --- Sub-Question Query Engine ---

class SubQuestionQueryEngine:
    """Decomposes complex payment failure queries into sub-questions.

    Pattern from DeepLearning.AI "Building Agentic RAG with LlamaIndex":
    - Sub-Q1: PSP/error-specific protocol
    - Sub-Q2: RBI mandate or gateway constraint
    - Sub-Q3: Merchant dunning policy recommendation
    """

    def __init__(self, router: RouterQueryEngine):
        self.router = router

    def _decompose(self, payment_payload: dict[str, Any]) -> list[dict[str, str]]:
        """Decompose a payment failure payload into sub-questions."""
        method = payment_payload.get("method", "unknown")
        provider = payment_payload.get("provider", "unknown")
        failure_code = payment_payload.get("failure_code", "unknown")
        failure_reason = payment_payload.get("failure_reason", "unknown")
        error_description = payment_payload.get("error_description", failure_reason)
        amount = payment_payload.get("amount", 0)

        sub_questions = [
            {
                "id": "sub_q1",
                "domain": "psp",
                "question": (
                    f"What is the failure protocol for payment method '{method}' "
                    f"with provider '{provider}' and error description "
                    f"'{error_description}'? Error code: {failure_code}"
                ),
            },
            {
                "id": "sub_q2",
                "domain": "rbi",
                "question": (
                    f"What RBI mandate or PSP gateway constraint applies to a "
                    f"transaction of INR {amount} using method '{method}'? "
                    f"Failure: {failure_code}"
                ),
            },
            {
                "id": "sub_q3",
                "domain": "merchant",
                "question": (
                    f"What is the recommended recovery rail and dunning strategy "
                    f"for failure code '{failure_code}' with method '{method}' "
                    f"at amount INR {amount}?"
                ),
            },
            {
                "id": "sub_q4",
                "domain": "razorpay",
                "question": (
                    f"Razorpay error resolution for code '{failure_code}' "
                    f"reason '{failure_reason}' method '{method}': "
                    f"what is the correct recovery protocol?"
                ),
            },
        ]

        return sub_questions

    def query(self, payment_payload: dict[str, Any]) -> RAGResponse:
        """Decompose, retrieve per sub-question, synthesize."""
        sub_questions = self._decompose(payment_payload)
        sub_answers: list[dict[str, Any]] = []
        all_chunks: list[TextChunk] = []

        for sq in sub_questions:
            retrieval = self.router.query(sq["question"], top_k=3)
            context = "\n---\n".join(c.text for c in retrieval.chunks)
            sub_answers.append({
                "sub_question_id": sq["id"],
                "domain": sq["domain"],
                "question": sq["question"],
                "retrieved_context": context,
                "num_chunks": len(retrieval.chunks),
                "index_used": retrieval.index_type,
            })
            all_chunks.extend(retrieval.chunks)

        # Synthesize sub-answers into unified context
        synthesis_parts = []
        for sa in sub_answers:
            synthesis_parts.append(
                f"[{sa['domain'].upper()}] {sa['question']}\n"
                f"Retrieved: {sa['retrieved_context'][:500]}"
            )
        unified_context = "\n\n".join(synthesis_parts)

        return RAGResponse(
            answer=unified_context,
            groundedness_score=0.0,  # Will be computed by RAGTriadEvaluator
            faithfulness_score=0.0,
            retrieved_chunks=all_chunks,
            sub_answers=sub_answers,
            routed_index="sub_question_decomposition",
            decomposition_steps=len(sub_questions),
            metadata={
                "method": payment_payload.get("method", "unknown"),
                "provider": payment_payload.get("provider", "unknown"),
                "failure_code": payment_payload.get("failure_code", "unknown"),
            },
        )


# --- RAG Triad Evaluator ---

class RAGTriadEvaluator:
    """Evaluates Groundedness and Faithfulness of RAG responses.

    Groundedness: Whether the diagnosis is strictly derived from retrieved text.
    Faithfulness: Whether the answer is consistent with the retrieved context.

    Uses LLM-as-a-Judge pattern (Nemotron via OmniRoute).
    """

    GROUNDEDNESS_PROMPT = """You are a strict groundedness evaluator for a payment recovery system.

Your task: Determine whether the DIAGNOSIS is fully supported by the RETRIEVED CONTEXT.

Scoring:
- 1.0: Every claim in the diagnosis is directly supported by the retrieved context
- 0.8: Most claims are supported, with minor extrapolation
- 0.6: Some claims are supported, but the diagnosis includes unsupported assumptions
- 0.4: Only a few claims are supported by the context
- 0.2: Most of the diagnosis is not grounded in the retrieved context
- 0.0: The diagnosis is entirely unsupported by the context

CRITICAL RULES:
- A diagnosis is NOT grounded if it references banks, PSPs, or protocols not mentioned in the context
- A diagnosis IS grounded if it correctly interprets the context (even with minor reasoning)
- If the context mentions "LazyPay OTP expired" and the diagnosis says "LazyPay PayLater OTP failure", that IS grounded

Start your response immediately with the character {. Do NOT output any preamble.

Output JSON:
{
  "groundedness_score": <0.0 to 1.0>,
  "evidence": "Brief justification of the score",
  "unsupported_claims": ["list of claims not in context"]
}"""

    FAITHFULNESS_PROMPT = """You are a strict faithfulness evaluator for a payment recovery system.

Your task: Determine whether the ANSWER is consistent with and derived from the RETRIEVED CONTEXT.

Scoring:
- 1.0: Answer is fully consistent with the context, no contradictions
- 0.8: Answer is mostly consistent, minor interpretations
- 0.6: Answer has some elements from context but includes unsupported additions
- 0.4: Answer contradicts or ignores significant parts of the context
- 0.2: Answer is mostly inconsistent with the context
- 0.0: Answer is completely unrelated to or contradicts the context

Start your response immediately with the character {. Do NOT output any preamble.

Output JSON:
{
  "faithfulness_score": <0.0 to 1.0>,
  "evidence": "Brief justification of the score",
  "contradictions": ["list of contradictions if any"]
}"""

    def __init__(self):
        pass

    def evaluate_groundedness(
        self, diagnosis: str, retrieved_context: str
    ) -> dict[str, Any]:
        """Evaluate whether diagnosis is grounded in retrieved context."""
        prompt = (
            f"DIAGNOSIS:\n{diagnosis}\n\n"
            f"RETRIEVED CONTEXT:\n{retrieved_context[:3000]}"
        )

        result = invoke_llm_json(
            prompt=prompt,
            system=self.GROUNDEDNESS_PROMPT,
            temperature=0,
            max_tokens=512,
        )

        if result is None:
            # Fallback: heuristic groundedness based on keyword overlap
            return self._heuristic_groundedness(diagnosis, retrieved_context)

        return {
            "groundedness_score": float(result.get("groundedness_score", 0.5)),
            "evidence": result.get("evidence", "LLM evaluation"),
            "unsupported_claims": result.get("unsupported_claims", []),
        }

    def evaluate_faithfulness(
        self, answer: str, retrieved_context: str
    ) -> dict[str, Any]:
        """Evaluate whether answer is faithful to retrieved context."""
        prompt = (
            f"ANSWER:\n{answer}\n\n"
            f"RETRIEVED CONTEXT:\n{retrieved_context[:3000]}"
        )

        result = invoke_llm_json(
            prompt=prompt,
            system=self.FAITHFULNESS_PROMPT,
            temperature=0,
            max_tokens=512,
        )

        if result is None:
            return self._heuristic_faithfulness(answer, retrieved_context)

        return {
            "faithfulness_score": float(result.get("faithfulness_score", 0.5)),
            "evidence": result.get("evidence", "LLM evaluation"),
            "contradictions": result.get("contradictions", []),
        }

    @staticmethod
    def _heuristic_groundedness(diagnosis: str, context: str) -> dict[str, Any]:
        """Keyword-overlap heuristic when LLM is unavailable."""
        diag_terms = set(re.findall(r"[a-z0-9_]{3,}", diagnosis.lower()))
        ctx_terms = set(re.findall(r"[a-z0-9_]{3,}", context.lower()))
        if not diag_terms:
            return {"groundedness_score": 0.0, "evidence": "No terms to evaluate", "unsupported_claims": []}
        overlap = len(diag_terms & ctx_terms)
        score = min(1.0, overlap / max(len(diag_terms), 1))
        return {
            "groundedness_score": round(score, 2),
            "evidence": f"Heuristic: {overlap}/{len(diag_terms)} diagnosis terms found in context",
            "unsupported_claims": [],
        }

    @staticmethod
    def _heuristic_faithfulness(answer: str, context: str) -> dict[str, Any]:
        """Keyword-overlap heuristic when LLM is unavailable."""
        ans_terms = set(re.findall(r"[a-z0-9_]{3,}", answer.lower()))
        ctx_terms = set(re.findall(r"[a-z0-9_]{3,}", context.lower()))
        if not ans_terms:
            return {"faithfulness_score": 0.0, "evidence": "No terms to evaluate", "contradictions": []}
        overlap = len(ans_terms & ctx_terms)
        score = min(1.0, overlap / max(len(ans_terms), 1))
        return {
            "faithfulness_score": round(score, 2),
            "evidence": f"Heuristic: {overlap}/{len(ans_terms)} answer terms found in context",
            "contradictions": [],
        }


# --- Main Agentic RAG Engine ---

class LlamaIndexAgenticRAG:
    """Production-grade Agentic RAG Engine combining all 4 LlamaIndex paradigms.

    Usage:
        rag = LlamaIndexAgenticRAG()
        response = rag.query({
            "method": "paylater",
            "provider": "lazypay",
            "failure_code": "PAYLATER_OTP_EXPIRED",
            "failure_reason": "OTP verification timed out",
            "error_description": "use another payment instrument",
            "amount": 2500,
        })
        print(response.groundedness_score)  # 0.92
        print(response.answer)
    """

    def __init__(self, kb_dir: Path | None = None):
        self.kb_dir = kb_dir or _KB_DIR
        self._chunks: list[TextChunk] = []
        self._vector_index: VectorIndex | None = None
        self._summary_index: SummaryIndex | None = None
        self._router: RouterQueryEngine | None = None
        self._sub_question_engine: SubQuestionQueryEngine | None = None
        self._evaluator = RAGTriadEvaluator()
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load knowledge base on first query."""
        if self._loaded:
            return

        self._chunks = self._load_knowledge_base()
        self._vector_index = VectorIndex(self._chunks)
        self._summary_index = SummaryIndex(self._chunks)
        self._router = RouterQueryEngine(self._vector_index, self._summary_index)
        self._sub_question_engine = SubQuestionQueryEngine(self._router)
        self._loaded = True

    def _load_knowledge_base(self) -> list[TextChunk]:
        """Load and chunk all knowledge base markdown files."""
        all_chunks: list[TextChunk] = []

        for domain, filename in _DOMAIN_FILES.items():
            filepath = self.kb_dir / filename
            if filepath.exists():
                text = filepath.read_text(encoding="utf-8")
                chunks = _chunk_markdown(text, source_file=filename)
                all_chunks.extend(chunks)

        return all_chunks

    def query(
        self,
        payment_payload: dict[str, Any],
        evaluate: bool = True,
    ) -> RAGResponse:
        """Full Agentic RAG pipeline:
        1. SubQuestionQueryEngine decomposes the query
        2. RouterQueryEngine routes each sub-question
        3. RAGTriadEvaluator scores groundedness & faithfulness
        """
        self._ensure_loaded()

        # Step 1-2: Decompose + Retrieve
        rag_response = self._sub_question_engine.query(payment_payload)

        # Step 3: Evaluate groundedness & faithfulness
        if evaluate and rag_response.retrieved_chunks:
            context = "\n---\n".join(c.text for c in rag_response.retrieved_chunks)
            diagnosis_text = rag_response.answer

            grounded = self._evaluator.evaluate_groundedness(diagnosis_text, context)
            faithful = self._evaluator.evaluate_faithfulness(diagnosis_text, context)

            rag_response.groundedness_score = grounded["groundedness_score"]
            rag_response.faithfulness_score = faithful["faithfulness_score"]
            rag_response.metadata["groundedness_evidence"] = grounded["evidence"]
            rag_response.metadata["faithfulness_evidence"] = faithful["evidence"]
            rag_response.metadata["unsupported_claims"] = grounded.get("unsupported_claims", [])
            rag_response.metadata["contradictions"] = faithful.get("contradictions", [])

        return rag_response

    def query_by_error_code(self, error_code: str, method: str = "unknown") -> RAGResponse:
        """Direct query by error code — used by diagnosis.py."""
        return self.query({
            "failure_code": error_code,
            "method": method,
            "failure_reason": error_code,
            "error_description": error_code,
            "amount": 0,
        })

    def query_for_diagnosis(self, case_metadata: dict[str, Any]) -> RAGResponse:
        """Query RAG engine from a case's metadata dict — used by diagnosis.py."""
        return self.query({
            "failure_code": case_metadata.get("failure_code", case_metadata.get("error_code", "unknown")),
            "failure_reason": case_metadata.get("failure_reason", case_metadata.get("error_description", "unknown")),
            "error_description": case_metadata.get("error_description", case_metadata.get("failure_reason", "unknown")),
            "method": case_metadata.get("method", "unknown"),
            "provider": case_metadata.get("provider", "unknown"),
            "amount": case_metadata.get("amount", 0),
        })

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def chunk_count(self) -> int:
        self._ensure_loaded()
        return len(self._chunks)

    @property
    def document_count(self) -> int:
        return len(_DOMAIN_FILES)
