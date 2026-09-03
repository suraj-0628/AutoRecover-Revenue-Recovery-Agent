"""Vectorized Customer Memory — ChromaDB-backed semantic search over payment outcomes.

Replaces flat key-value memory lookups with vector similarity search.
When a new failure comes in, queries for SIMILAR past cases (same failure
type, similar amount, same bank) and returns what interventions worked.

Architecture:
  - ChromaDB collection stores payment outcome documents
  - Each document = natural language description of a case
  - Metadata = structured fields for filtering (failure_type, recovered, etc.)
  - Semantic search finds similar past cases to inform new decisions

Graceful degradation:
  - If ChromaDB is unavailable, all queries return empty results
  - Falls back silently — no crashes, no hallucinated data
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from recovery_agent.models import Case


def _build_outcome_text(case: Case) -> str:
    """Build a natural language description of a payment outcome for embedding.

    This text is what ChromaDB embeds for semantic search. It must capture
    the essential features of the case so similar cases are found.
    """
    parts = []

    # Failure type
    if case.diagnosis:
        parts.append(f"Payment failed due to {case.diagnosis.root_cause.value}")
    elif case.payment.failure_code:
        parts.append(f"Payment failed with code {case.payment.failure_code}")
    else:
        parts.append(f"Payment failed: {case.payment.failure_reason}")

    # Amount tier
    if case.payment.amount >= 50000:
        parts.append("high value transaction")
    elif case.payment.amount >= 10000:
        parts.append("medium value transaction")
    else:
        parts.append("low value transaction")

    # Payment method
    method = case.payment.metadata.get("method", "")
    if method:
        parts.append(f"paid via {method}")

    # Bank
    bank = case.payment.metadata.get("bank", "") or case.payment.metadata.get("bank_name", "")
    if bank:
        parts.append(f"bank {bank}")

    # Provider
    provider = case.payment.metadata.get("provider", "")
    if provider:
        parts.append(f"provider {provider}")

    # Recovery outcome
    if case.recovered:
        parts.append("successfully recovered")
        # What action worked?
        if case.attempts:
            last = case.attempts[-1]
            parts.append(f"recovered using {last.action_type.value}")
    else:
        parts.append("recovery failed")

    # Attempt count
    if case.attempt_count > 0:
        parts.append(f"required {case.attempt_count} attempts")

    return ". ".join(parts)


def _build_outcome_metadata(case: Case) -> dict[str, Any]:
    """Build structured metadata for filtering in ChromaDB."""
    failure_type = case.diagnosis.root_cause.value if case.diagnosis else "unknown"
    method = case.payment.metadata.get("method", "unknown")
    bank = case.payment.metadata.get("bank", "") or case.payment.metadata.get("bank_name", "")

    # Determine which intervention was used (last attempt action)
    intervention = "none"
    if case.attempts:
        intervention = case.attempts[-1].action_type.value

    return {
        "case_id": case.id,
        "customer_id": case.payment.customer_id,
        "failure_type": failure_type,
        "failure_code": case.payment.failure_code,
        "amount": case.payment.amount,
        "method": method,
        "bank": bank,
        "recovered": case.recovered,
        "recovered_amount": case.recovered_amount,
        "attempt_count": case.attempt_count,
        "intervention": intervention,
        "recovery_tier": case.recovery_tier.value,
    }


class VectorMemoryStore:
    """ChromaDB-backed vector memory for semantic search over payment outcomes.

    Collections:
      - payment_outcomes: completed case outcomes (success/failure + intervention used)

    Usage:
      store = VectorMemoryStore(persist_dir="./data/vector_memory")
      store.ingest_outcome(case)                    # After recovery completes
      similar = store.query_similar(failure_type, amount, bank, k=5)
      context = store.get_decision_context(case)    # Ready-to-inject prompt text
    """

    def __init__(self, persist_dir: str | None = None):
        self._available = False
        self._client = None
        self._collection = None
        self._lock = threading.Lock()

        try:
            import chromadb

            if persist_dir:
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=persist_dir)
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name="payment_outcomes",
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
        except Exception as e:
            print(f"[vector_memory] WARNING: ChromaDB unavailable: {e}. Semantic search disabled.")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def outcome_count(self) -> int:
        if not self._available:
            return 0
        return self._collection.count()

    def ingest_outcome(self, case: Case) -> bool:
        """Ingest a completed case outcome into vector memory.

        Called after recovery completes (success, failure, or escalation).
        Returns True if ingested, False if skipped.
        """
        if not self._available:
            return False

        # Don't ingest cases with no attempts — nothing happened
        if not case.attempts and not case.recovered:
            return False

        doc_id = f"case_{case.id}"
        text = _build_outcome_text(case)
        metadata = _build_outcome_metadata(case)

        try:
            with self._lock:
                self._collection.upsert(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[metadata],
                )
            return True
        except Exception as e:
            print(f"[vector_memory] ERROR: Failed to ingest outcome: {e}")
            return False

    def query_similar(
        self,
        failure_type: str = "",
        amount: float = 0.0,
        bank: str = "",
        method: str = "",
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query for similar past payment outcomes.

        Returns a list of dicts with keys: text, metadata, distance.
        Lower distance = more similar.
        """
        if not self._available:
            return []

        # Build query text from available features
        query_parts = []
        if failure_type:
            query_parts.append(f"Payment failed due to {failure_type}")
        if amount >= 50000:
            query_parts.append("high value transaction")
        elif amount >= 10000:
            query_parts.append("medium value transaction")
        elif amount > 0:
            query_parts.append("low value transaction")
        if bank:
            query_parts.append(f"bank {bank}")
        if method:
            query_parts.append(f"paid via {method}")

        if not query_parts:
            return []

        query_text = ". ".join(query_parts)

        # Build ChromaDB where filter
        chroma_where = where or {}
        if failure_type and failure_type != "unknown":
            chroma_where["failure_type"] = failure_type

        try:
            with self._lock:
                results = self._collection.query(
                    query_texts=[query_text],
                    n_results=min(k, self._collection.count() or 1),
                    where=chroma_where if chroma_where else None,
                )

            similar = []
            if results and results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    dist = results["distances"][0][i] if results["distances"] else 0.0
                    similar.append({
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                    })

            return similar

        except Exception as e:
            print(f"[vector_memory] ERROR: Query failed: {e}")
            return []

    def get_decision_context(self, case: Case, k: int = 3) -> str:
        """Get a ready-to-inject context string for the LLM decision prompt.

        Queries for similar past outcomes and formats them as structured
        context that the strategy planner can reason about.
        """
        similar = self.query_similar(
            failure_type=case.diagnosis.root_cause.value if case.diagnosis else "",
            amount=case.payment.amount,
            bank=case.payment.metadata.get("bank", "") or case.payment.metadata.get("bank_name", ""),
            method=case.payment.metadata.get("method", ""),
            k=k,
        )

        if not similar:
            return ""

        lines = ["SIMILAR PAST CASES (what worked):"]
        for i, s in enumerate(similar, 1):
            meta = s.get("metadata", {})
            recovered = "RECOVERED" if meta.get("recovered") else "FAILED"
            intervention = meta.get("intervention", "unknown")
            amount = meta.get("amount", 0)
            failure = meta.get("failure_type", "unknown")
            lines.append(
                f"  {i}. {failure} / INR {amount:,.0f} → {recovered} via {intervention}"
            )

        return "\n".join(lines)

    def clear(self) -> bool:
        """Clear all stored outcomes. Used for testing."""
        if not self._available:
            return False
        try:
            with self._lock:
                self._client.delete_collection("payment_outcomes")
                self._collection = self._client.get_or_create_collection(
                    name="payment_outcomes",
                    metadata={"hnsw:space": "cosine"},
                )
            return True
        except Exception as e:
            print(f"[vector_memory] ERROR: Clear failed: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Return vector memory statistics."""
        return {
            "available": self._available,
            "outcome_count": self.outcome_count,
            "persist_dir": "in-memory" if self._client and not hasattr(self._client, '_path') else "persistent",
        }
