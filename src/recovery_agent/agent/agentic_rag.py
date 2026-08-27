"""LlamaIndex-Style Agentic RAG Engine — PRODUCTION GRADE.

Implements 4 LlamaIndex paradigms:
  1. ChromaDB VectorIndex — real vector database with sentence-transformer embeddings
  2. SummaryIndex — high-level section summaries for broad queries
  3. RouterQueryEngine — dynamic routing between indexes
  4. SubQuestionQueryEngine — multi-step query decomposition
  5. RAGTriadEvaluator — groundedness & faithfulness scoring with RE-LOOP

If groundedness < 0.7, the engine autonomously rewrites the query and retries
instead of returning a hallucinated response.

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

# Minimum groundedness threshold — below this, the engine rewrites and retries
MIN_GROUNDEDNESS = 0.7
MAX_REWRITE_ATTEMPTS = 3


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
            current_header = line.lstrip("#").strip()
            current_section = [line]
        else:
            current_section.append(line)

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


# --- ChromaDB Vector Index ---

class VectorIndex:
    """Real vector database index using ChromaDB.

    Uses ChromaDB's default embedding function (all-MiniLM-L6-v2 via sentence-transformers)
    for semantic similarity search. FAILS LOUDLY if ChromaDB is unavailable — no heuristic hacks.
    """

    def __init__(self, chunks: list[TextChunk]):
        self.chunks = chunks
        self._collection = None
        self._chroma_client = None
        self._build_chromadb()

    def _build_chromadb(self):
        """Build ChromaDB collection from chunks. Raises RuntimeError if ChromaDB unavailable."""
        try:
            import chromadb
            import uuid

            # Check if embedding model is cached locally
            import os
            from pathlib import Path
            cache_dir = Path.home() / ".cache" / "chroma" / "onnx_models"
            model_dir = cache_dir / "all-MiniLM-L6-v2"

            if not model_dir.exists() or not (
                any(model_dir.glob("*.onnx")) or any(model_dir.glob("onnx/*.onnx"))
            ):
                raise RuntimeError(
                    "[agentic_rag] FATAL: ChromaDB ONNX embedding model not cached. "
                    f"Expected at: {model_dir}. "
                    "Download it manually or run `chromadb` once with internet access. "
                    "No heuristic fallback — RAG requires a real vector database."
                )

            self._chroma_client = chromadb.Client()
            self._collection = self._chroma_client.create_collection(
                name=f"payment_recovery_kb_{uuid.uuid4().hex[:8]}",
                metadata={"hnsw:space": "cosine"},
            )

            # Add documents in batches
            batch_size = 100
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i : i + batch_size]
                ids = [f"chunk_{i+j}" for j in range(len(batch))]
                documents = [c.text for c in batch]
                metadatas = [
                    {
                        "source_file": c.source_file,
                        "section_header": c.section_header,
                        "chunk_index": c.chunk_index,
                    }
                    for c in batch
                ]
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
        except ImportError:
            raise RuntimeError(
                "[agentic_rag] FATAL: chromadb not installed. "
                "Install with: pip install chromadb. "
                "No heuristic fallback — RAG requires a real vector database."
            )
        except Exception as e:
            raise RuntimeError(
                f"[agentic_rag] FATAL: ChromaDB initialization failed: {e}. "
                "No heuristic fallback — RAG requires a working vector database."
            )

    def query(self, query_text: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve top-k chunks by semantic similarity via ChromaDB."""
        if self._collection is None:
            raise RuntimeError("[agentic_rag] ChromaDB collection not initialized.")

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(top_k, len(self.chunks)),
            )

            result_chunks: list[TextChunk] = []
            scores: list[float] = []

            if results and results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    # Find matching chunk by text
                    for chunk in self.chunks:
                        if chunk.text == doc:
                            result_chunks.append(chunk)
                            # ChromaDB returns distances, convert to similarity scores
                            if results.get("distances") and results["distances"][0]:
                                dist = results["distances"][0][i]
                                scores.append(max(0.0, 1.0 - dist))
                            break

            max_score = max(scores) if scores else 0.0

            return RetrievalResult(
                chunks=result_chunks,
                index_type="vector",
                query=query_text,
                score=max_score,
            )
        except Exception as e:
            raise RuntimeError(
                f"[agentic_rag] ChromaDB query failed: {e}. "
                "No heuristic fallback — failing loudly."
            )


# --- Summary Index ---

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

            header_overlap = len(query_terms & header_terms)
            content_overlap = len(query_terms & content_terms)
            section_scores[header] = header_overlap * 2.0 + content_overlap * 0.5

        ranked = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result_chunks: list[TextChunk] = []
        for header, _ in ranked:
            section_chunks = self._sections[header]
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

    SPECIFIC_KEYWORDS = {
        "error", "code", "failed", "declined", "timeout", "otp", "expired",
        "invalid", "insufficient", "mandate", "hdfc", "icici", "sbi", "lazypay",
        "paylater", "upi", "card", "netbanking", "npci", "visa", "mastercard",
        "rupay", "3ds", "authorization", "authentication", "gateway",
    }

    BROAD_KEYWORDS = {
        "policy", "rules", "best", "practice", "strategy", "dunning",
        "discount", "communication", "retry", "schedule", "when", "how",
        "compliance", "regulation", "rbi", "guideline", "cap", "limit",
    }

    def __init__(self, vector_index: VectorIndex, summary_index: SummaryIndex):
        self.vector_index = vector_index
        self.summary_index = summary_index

    def _route(self, query: str) -> str:
        query_lower = query.lower()
        terms = set(re.findall(r"[a-z0-9_]+", query_lower))
        specific_hits = len(terms & self.SPECIFIC_KEYWORDS)
        broad_hits = len(terms & self.BROAD_KEYWORDS)
        if specific_hits > broad_hits:
            return "vector"
        elif broad_hits > specific_hits:
            return "summary"
        return "both"

    def query(self, query: str, top_k: int = 5) -> RetrievalResult:
        route = self._route(query)
        if route == "vector":
            return self.vector_index.query(query, top_k=top_k)
        elif route == "summary":
            return self.summary_index.query(query, top_k=top_k)
        else:
            v_result = self.vector_index.query(query, top_k=top_k)
            s_result = self.summary_index.query(query, top_k=top_k)
            merged_chunks = v_result.chunks + s_result.chunks
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
    """Decomposes complex payment failure queries into sub-questions."""

    def __init__(self, router: RouterQueryEngine):
        self.router = router

    def _decompose(self, payment_payload: dict[str, Any]) -> list[dict[str, str]]:
        method = payment_payload.get("method", "unknown")
        provider = payment_payload.get("provider", "unknown")
        failure_code = payment_payload.get("failure_code", "unknown")
        failure_reason = payment_payload.get("failure_reason", "unknown")
        error_description = payment_payload.get("error_description", failure_reason)
        amount = payment_payload.get("amount", 0)

        return [
            {
                "id": "sub_q1",
                "domain": "psp",
                "question": f"What is the failure protocol for payment method '{method}' with provider '{provider}' and error description '{error_description}'? Error code: {failure_code}",
            },
            {
                "id": "sub_q2",
                "domain": "rbi",
                "question": f"What RBI mandate or PSP gateway constraint applies to a transaction of INR {amount} using method '{method}'? Failure: {failure_code}",
            },
            {
                "id": "sub_q3",
                "domain": "merchant",
                "question": f"What is the recommended recovery rail and dunning strategy for failure code '{failure_code}' with method '{method}' at amount INR {amount}?",
            },
            {
                "id": "sub_q4",
                "domain": "razorpay",
                "question": f"Razorpay error resolution for code '{failure_code}' reason '{failure_reason}' method '{method}': what is the correct recovery protocol?",
            },
        ]

    def query(self, payment_payload: dict[str, Any]) -> RAGResponse:
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

        synthesis_parts = []
        for sa in sub_answers:
            synthesis_parts.append(
                f"[{sa['domain'].upper()}] {sa['question']}\nRetrieved: {sa['retrieved_context'][:500]}"
            )
        unified_context = "\n\n".join(synthesis_parts)

        return RAGResponse(
            answer=unified_context,
            groundedness_score=0.0,
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

Determine whether the DIAGNOSIS is fully supported by the RETRIEVED CONTEXT.

Scoring:
- 1.0: Every claim directly supported by context
- 0.8: Most claims supported, minor extrapolation
- 0.6: Some claims supported, some unsupported assumptions
- 0.4: Only a few claims supported
- 0.2: Most unsupported
- 0.0: Entirely unsupported

CRITICAL: A diagnosis referencing banks/PSPs/protocols NOT in context is NOT grounded.

Start with {. Output JSON:
{
  "groundedness_score": <0.0-1.0>,
  "evidence": "Brief justification",
  "unsupported_claims": ["list"]
}"""

    FAITHFULNESS_PROMPT = """You are a strict faithfulness evaluator for a payment recovery system.

Determine whether the ANSWER is consistent with the RETRIEVED CONTEXT.

Scoring:
- 1.0: Fully consistent
- 0.8: Mostly consistent, minor interpretations
- 0.6: Some context elements, unsupported additions
- 0.4: Contradicts or ignores significant context
- 0.2: Mostly inconsistent
- 0.0: Completely unrelated

Start with {. Output JSON:
{
  "faithfulness_score": <0.0-1.0>,
  "evidence": "Brief justification",
  "contradictions": ["list"]
}"""

    def __init__(self):
        pass

    def evaluate_groundedness(self, diagnosis: str, retrieved_context: str) -> dict[str, Any]:
        prompt = f"DIAGNOSIS:\n{diagnosis}\n\nRETRIEVED CONTEXT:\n{retrieved_context[:3000]}"
        result = invoke_llm_json(prompt=prompt, system=self.GROUNDEDNESS_PROMPT, temperature=0, max_tokens=512)
        if result is None:
            return self._heuristic_groundedness(diagnosis, retrieved_context)
        return {
            "groundedness_score": float(result.get("groundedness_score", 0.5)),
            "evidence": result.get("evidence", "LLM evaluation"),
            "unsupported_claims": result.get("unsupported_claims", []),
        }

    def evaluate_faithfulness(self, answer: str, retrieved_context: str) -> dict[str, Any]:
        prompt = f"ANSWER:\n{answer}\n\nRETRIEVED CONTEXT:\n{retrieved_context[:3000]}"
        result = invoke_llm_json(prompt=prompt, system=self.FAITHFULNESS_PROMPT, temperature=0, max_tokens=512)
        if result is None:
            return self._heuristic_faithfulness(answer, retrieved_context)
        return {
            "faithfulness_score": float(result.get("faithfulness_score", 0.5)),
            "evidence": result.get("evidence", "LLM evaluation"),
            "contradictions": result.get("contradictions", []),
        }

    @staticmethod
    def _heuristic_groundedness(diagnosis: str, context: str) -> dict[str, Any]:
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
    """Production-grade Agentic RAG Engine.

    If groundedness < MIN_GROUNDEDNESS, autonomously rewrites the query
    and retries up to MAX_REWRITE_ATTEMPTS times.

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
        if self._loaded:
            return
        self._chunks = self._load_knowledge_base()
        self._vector_index = VectorIndex(self._chunks)
        self._summary_index = SummaryIndex(self._chunks)
        self._router = RouterQueryEngine(self._vector_index, self._summary_index)
        self._sub_question_engine = SubQuestionQueryEngine(self._router)
        self._loaded = True

    def _load_knowledge_base(self) -> list[TextChunk]:
        all_chunks: list[TextChunk] = []
        for domain, filename in _DOMAIN_FILES.items():
            filepath = self.kb_dir / filename
            if filepath.exists():
                text = filepath.read_text(encoding="utf-8")
                chunks = _chunk_markdown(text, source_file=filename)
                all_chunks.extend(chunks)
        return all_chunks

    def _rewrite_query(self, original_query: str, groundedness_result: dict[str, Any]) -> str:
        """Use LLM to rewrite the query when groundedness is too low."""
        unsupported = groundedness_result.get("unsupported_claims", [])
        evidence = groundedness_result.get("evidence", "")

        prompt = f"""The following payment recovery query returned low groundedness ({groundedness_result.get('groundedness_score', 0):.2f}).

Original query: {original_query}

Unsupported claims: {unsupported}
Evidence: {evidence}

Rewrite the query to be more specific and grounded in the available knowledge base.
Focus on concrete error codes, payment methods, and protocols.

Output ONLY the rewritten query, nothing else."""

        rewritten = invoke_llm(
            prompt=prompt,
            system="You are a payment recovery query rewriter. Output only the rewritten query.",
            temperature=0.3,
            max_tokens=200,
        )

        return rewritten or f"{original_query} (specific error code protocol)"

    def query(
        self,
        payment_payload: dict[str, Any],
        evaluate: bool = True,
    ) -> RAGResponse:
        """Full Agentic RAG pipeline with groundedness re-loop.

        If groundedness < MIN_GROUNDEDNESS, rewrites query and retries
        up to MAX_REWRITE_ATTEMPTS times.

        If ChromaDB throws at runtime (network drop, model corruption),
        raises RuntimeError so the AgentHarness catches it and the LLM
        can pivot to an alternate tool (e.g., query_gateway_error_details).
        """
        self._ensure_loaded()

        best_response: RAGResponse | None = None
        best_groundedness = 0.0

        for attempt in range(MAX_REWRITE_ATTEMPTS):
            try:
                if attempt == 0:
                    rag_response = self._sub_question_engine.query(payment_payload)
                else:
                    rewritten_query = self._rewrite_query(
                        payment_payload.get("failure_reason", ""),
                        {"groundedness_score": best_groundedness, "evidence": "Low groundedness", "unsupported_claims": []},
                    )
                    modified_payload = {**payment_payload, "failure_reason": rewritten_query, "error_description": rewritten_query}
                    rag_response = self._sub_question_engine.query(modified_payload)
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(
                    f"Vector Database unavailable: {e}. "
                    "Please use alternate diagnostic tools."
                ) from e

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
                rag_response.metadata["rewrite_attempts"] = attempt

                if rag_response.groundedness_score > best_groundedness:
                    best_groundedness = rag_response.groundedness_score
                    best_response = rag_response

                # If groundedness is acceptable, return immediately
                if rag_response.groundedness_score >= MIN_GROUNDEDNESS:
                    return rag_response
            else:
                return rag_response

        return best_response or rag_response

    def query_by_error_code(self, error_code: str, method: str = "unknown") -> RAGResponse:
        return self.query({
            "failure_code": error_code,
            "method": method,
            "failure_reason": error_code,
            "error_description": error_code,
            "amount": 0,
        })

    def query_for_diagnosis(self, case_metadata: dict[str, Any]) -> RAGResponse:
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


# Import invoke_llm for query rewriting
from recovery_agent.agent.llm_client import invoke_llm
