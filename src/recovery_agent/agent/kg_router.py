"""Knowledge Graph Router — NetworkX-powered Razorpay API discovery.

Models Razorpay's payment ecosystem as a directed graph with two edge types:
1. FAILURE EDGES: "When X fails, try Y" (existing)
2. PROCESS EDGES: "After using X, the next step is Y" (new — pr:hasNext semantics)

When a payment rail fails, the agent uses two-phase discovery:
1. SEMANTIC RETRIEVAL: Embed rail metadata → retrieve top-k relevant rails
2. PROCESS ENRICHMENT: Add rails connected via process edges (fixes false negatives)

This follows the pattern from DeepLearning.AI's "Knowledge Graphs for AI Agent API Discovery":
- Don't let the LLM pick from all APIs
- Discover a small relevant subset first
- Let the agent choose from that subset with process context
"""
from __future__ import annotations

from datetime import datetime, timezone

import networkx as nx

from recovery_agent.models import FailureType


# ═══════════════════════════════════════════════════════════════
# RAIL METADATA — API endpoint, parameters, conversion cost, channels
# ═══════════════════════════════════════════════════════════════

RAIL_DETAILS: dict[str, dict] = {
    "card_gateway": {
        "label": "Card Gateway",
        "api": "/payments",
        "method": "POST",
        "params": {"payment_method": "card", "token": "string"},
        "conversion_cost": 0.0,
        "channels": ["web", "app"],
        "description": "Standard card payment via Razorpay gateway",
        "failure_modes": ["card_expired", "insufficient_funds", "bank_declined", "network_timeout"],
        "prerequisites": [],
        "follow_ups": ["upi_autopay", "payment_link"],
    },
    "upi_autopay": {
        "label": "UPI Autopay",
        "api": "/subscriptions",
        "method": "POST",
        "params": {"customer_id": "string", "plan_id": "string", "auth_type": "upi_autopay"},
        "conversion_cost": 0.05,
        "channels": ["sms", "email", "whatsapp"],
        "description": "Recurring UPI mandate for automatic payments",
        "failure_modes": ["mandate_revoked", "insufficient_funds", "network_timeout"],
        "prerequisites": ["card_gateway", "payment_link"],
        "follow_ups": ["payment_link", "magic_checkout"],
    },
    "payment_link": {
        "label": "Payment Link",
        "api": "/payment_links",
        "method": "POST",
        "params": {"amount": "int", "currency": "INR", "description": "string"},
        "conversion_cost": 0.03,
        "channels": ["sms", "email", "whatsapp", "push"],
        "description": "Shareable payment link for one-time checkout",
        "failure_modes": ["card_expired", "insufficient_funds", "bank_declined"],
        "prerequisites": ["card_gateway", "upi_autopay", "magic_checkout"],
        "follow_ups": ["upi_autopay", "magic_checkout"],
    },
    "magic_checkout": {
        "label": "Magic Checkout",
        "api": "/magic_checkout",
        "method": "POST",
        "params": {"order_id": "string", "customer_id": "string", "method": "card|upi|netbanking"},
        "conversion_cost": 0.02,
        "channels": ["web", "app", "whatsapp"],
        "description": "1-click checkout with auto-filled payment details",
        "failure_modes": ["card_expired", "bank_declined", "network_timeout"],
        "prerequisites": ["card_gateway", "upi_autopay"],
        "follow_ups": ["optimizer_smart_router", "escrow_route"],
    },
    "optimizer_smart_router": {
        "label": "Smart Router (Optimizer)",
        "api": "/payments",
        "method": "POST",
        "params": {"amount": "int", "currency": "INR", "payment_method": "auto"},
        "conversion_cost": 0.01,
        "channels": ["web", "app"],
        "description": "Auto-routes to highest success-rate payment method",
        "failure_modes": ["bank_declined", "network_timeout", "insufficient_funds"],
        "prerequisites": ["card_gateway", "magic_checkout"],
        "follow_ups": ["payment_link", "upi_autopay"],
    },
    "escrow_route": {
        "label": "Escrow Settlement Route",
        "api": "/settlements",
        "method": "POST",
        "params": {"account_id": "string", "amount": "int"},
        "conversion_cost": 0.08,
        "channels": ["web"],
        "description": "Escrow-based settlement for high-value transactions",
        "failure_modes": ["risk_block"],
        "prerequisites": ["magic_checkout"],
        "follow_ups": ["payment_link"],
    },
}


# ═══════════════════════════════════════════════════════════════
# FAILURE → STARTING RAILS (deterministic fallback)
# ═══════════════════════════════════════════════════════════════

FAILURE_TO_STARTING_RAILS: dict[FailureType, list[str]] = {
    # An expired card is exactly the rail that CANNOT work — routing back to
    # card_gateway told the agent to retry the dead instrument. The customer has
    # to pay another way, so start with rails that let them choose one.
    FailureType.CARD_EXPIRED: ["payment_link", "upi_autopay", "magic_checkout"],
    FailureType.INSUFFICIENT_FUNDS: ["card_gateway", "upi_autopay"],
    FailureType.BANK_DECLINED: ["card_gateway", "optimizer_smart_router"],
    FailureType.NETWORK_TIMEOUT: ["card_gateway", "optimizer_smart_router"],
    FailureType.MANDATE_REVOKED: ["upi_autopay", "payment_link"],
    FailureType.RISK_BLOCK: ["payment_link", "magic_checkout"],
    FailureType.USER_DROPOFF: ["payment_link", "magic_checkout"],
    FailureType.UNKNOWN: ["card_gateway", "optimizer_smart_router"],
}


# ═══════════════════════════════════════════════════════════════
# EMBEDDING MODEL — sentence-transformers.util (MANDATE 1: use real SDK)
# ═══════════════════════════════════════════════════════════════

_embedding_model = None
_embedding_util = None


def _get_embedding_model():
    """Lazy-load sentence-transformers model + util via SDK."""
    global _embedding_model, _embedding_util
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer, util
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            _embedding_util = util
        except Exception:
            return None, None
    return _embedding_model, _embedding_util


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity via sentence_transformers.util.cos_sim — NOT raw math.

    Returns 0.0 when the optional embedding stack is absent rather than raising,
    so callers degrade instead of failing.
    """
    try:
        import torch
    except ImportError:
        return 0.0
    model, util = _get_embedding_model()
    if util is None:
        return 0.0
    return float(util.cos_sim(torch.tensor(a), torch.tensor(b))[0][0])


# ═══════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH — failure edges + process edges
# ═══════════════════════════════════════════════════════════════

class RazorpayKnowledgeGraph:
    """NetworkX Directed Graph modeling Razorpay's API ecosystem.

    Two edge types:
    - FAILURE edges: "When X fails, try Y" (weighted by conversion success)
    - PROCESS edges: "After using X, next step is Y" (pr:hasNext semantics)

    Two-phase discovery:
    1. Semantic retrieval: embed failure context → find top-k rails
    2. Process enrichment: add rails connected via process edges
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._rail_embeddings: dict[str, list[float]] = {}
        self._build_graph()
        self._build_rail_embeddings()

    def _build_graph(self) -> None:
        """Construct the full Razorpay API graph with failure + process edges."""
        # Add all rail nodes with metadata
        for rail_id, details in RAIL_DETAILS.items():
            self.graph.add_node(rail_id, **details)

        # --- FAILURE EDGES (existing) ---
        # Card gateway fails → try alternatives
        self.graph.add_edge(
            "card_gateway", "upi_autopay",
            weight=0.7, reason="card_failed_switch_to_upi",
            failure_types=["card_expired", "insufficient_funds"],
            edge_type="failure",
        )
        self.graph.add_edge(
            "card_gateway", "payment_link",
            weight=0.6, reason="card_failed_send_link",
            failure_types=["card_expired", "bank_declined"],
            edge_type="failure",
        )
        self.graph.add_edge(
            "card_gateway", "magic_checkout",
            weight=0.8, reason="card_failed_magic_checkout",
            failure_types=["card_expired"],
            edge_type="failure",
        )
        self.graph.add_edge(
            "card_gateway", "optimizer_smart_router",
            weight=0.75, reason="card_failed_smart_route",
            failure_types=["bank_declined", "network_timeout"],
            edge_type="failure",
        )

        # UPI Autopay fails → try alternatives
        self.graph.add_edge(
            "upi_autopay", "payment_link",
            weight=0.65, reason="upi_failed_send_link",
            failure_types=["mandate_revoked", "insufficient_funds"],
            edge_type="failure",
        )
        self.graph.add_edge(
            "upi_autopay", "card_gateway",
            weight=0.5, reason="upi_failed_try_card",
            failure_types=["network_timeout"],
            edge_type="failure",
        )
        self.graph.add_edge(
            "upi_autopay", "magic_checkout",
            weight=0.7, reason="upi_failed_magic_checkout",
            failure_types=["mandate_revoked"],
            edge_type="failure",
        )

        # Payment link → try other channels or smart routing
        self.graph.add_edge(
            "payment_link", "magic_checkout",
            weight=0.7, reason="link_low_conversion_magic",
            failure_types=["card_expired", "insufficient_funds"],
            edge_type="failure",
        )
        self.graph.add_edge(
            "payment_link", "optimizer_smart_router",
            weight=0.6, reason="link_failed_smart_route",
            failure_types=["bank_declined"],
            edge_type="failure",
        )

        # Magic checkout → smart router or escrow
        self.graph.add_edge(
            "magic_checkout", "optimizer_smart_router",
            weight=0.65, reason="magic_failed_smart_route",
            failure_types=["bank_declined", "network_timeout"],
            edge_type="failure",
        )
        self.graph.add_edge(
            "magic_checkout", "escrow_route",
            weight=0.4, reason="magic_failed_escrow_high_value",
            failure_types=["risk_block"],
            edge_type="failure",
        )

        # Smart router → fallback rails
        self.graph.add_edge(
            "optimizer_smart_router", "payment_link",
            weight=0.6, reason="smart_router_send_link",
            failure_types=["bank_declined", "network_timeout"],
            edge_type="failure",
        )
        self.graph.add_edge(
            "optimizer_smart_router", "upi_autopay",
            weight=0.55, reason="smart_router_try_upi",
            failure_types=["insufficient_funds"],
            edge_type="failure",
        )

        # Escrow → notification fallback
        self.graph.add_edge(
            "escrow_route", "payment_link",
            weight=0.5, reason="escrow_failed_send_link",
            failure_types=["risk_block"],
            edge_type="failure",
        )

        # --- PROCESS EDGES (new — pr:hasNext semantics) ---
        # These represent the natural business flow:
        # "After using rail X, the next step in the recovery process is Y"
        # Process edges have weight=1.0 (always valid) and edge_type="process"

        # After card gateway, offer UPI mandate for future payments
        self.graph.add_edge(
            "card_gateway", "upi_autopay",
            weight=1.0, reason="process:card_to_upi_mandate",
            edge_type="process",
            process_type="future_payment_setup",
        )
        # After payment link, offer UPI mandate for recurring
        self.graph.add_edge(
            "payment_link", "upi_autopay",
            weight=1.0, reason="process:link_to_upi_mandate",
            edge_type="process",
            process_type="future_payment_setup",
        )
        # After UPI mandate, send payment link for immediate retry
        self.graph.add_edge(
            "upi_autopay", "payment_link",
            weight=1.0, reason="process:upi_to_link_retry",
            edge_type="process",
            process_type="immediate_retry",
        )
        # After magic checkout, try smart router
        self.graph.add_edge(
            "magic_checkout", "optimizer_smart_router",
            weight=1.0, reason="process:magic_to_smart_router",
            edge_type="process",
            process_type="fallback_routing",
        )
        # After smart router, send payment link
        self.graph.add_edge(
            "optimizer_smart_router", "payment_link",
            weight=1.0, reason="process:smart_to_link",
            edge_type="process",
            process_type="customer_action_required",
        )
        # After escrow, send payment link
        self.graph.add_edge(
            "escrow_route", "payment_link",
            weight=1.0, reason="process:escrow_to_link",
            edge_type="process",
            process_type="customer_action_required",
        )

    def _build_rail_embeddings(self) -> None:
        """Pre-compute embeddings for each rail's metadata.

        This enables semantic retrieval: given a failure context,
        we embed it and find the most similar rails.
        """
        for rail_id, details in RAIL_DETAILS.items():
            # Build embedding text from rail metadata
            text_parts = [
                details["label"],
                details["description"],
                f"Failure modes: {', '.join(details.get('failure_modes', []))}",
                f"Channels: {', '.join(details.get('channels', []))}",
                f"Prerequisites: {', '.join(details.get('prerequisites', []))}",
                f"Follow-ups: {', '.join(details.get('follow_ups', []))}",
            ]
            embedding_text = " | ".join(text_parts)
            # Embed via SDK directly (MANDATE 1: real SDK, not raw Python)
            model, _ = _get_embedding_model()
            if model is not None:
                embedding = model.encode(embedding_text).tolist()
                self._rail_embeddings[rail_id] = embedding

    # ═══════════════════════════════════════════════════════════════
    # SEMANTIC RETRIEVAL — embed failure context, find similar rails
    # ═══════════════════════════════════════════════════════════════

    def semantic_discover_rails(
        self,
        failure_code: str,
        failure_reason: str = "",
        amount: float = 0.0,
        top_k: int = 3,
    ) -> list[dict]:
        """Phase 1: Semantic retrieval — find rails similar to failure context.

        Uses sentence_transformers.util.cos_sim when it is installed, and a
        deterministic failure-code mapping when it is not.

        The fallback below already existed but was unreachable: `import torch`
        ran *above* the `model is None` check, so on a machine without torch the
        whole call raised ImportError. `discover_recovery_rail` therefore failed
        100% of the time with "Discovery failed: No module named 'torch'" —
        burning an agent turn and injecting an error on every single run.
        Neither package is installed here, so the deterministic path is the live
        one; the imports now sit behind the guard.
        """
        model, _ = _get_embedding_model()
        if model is None:
            failure_type = self._resolve_failure_type(failure_code)
            starting_rails = FAILURE_TO_STARTING_RAILS.get(failure_type, ["card_gateway"])
            return [
                {"rail": rail, "similarity": 1.0, "source": "deterministic_fallback"}
                for rail in starting_rails
            ]

        import torch
        from sentence_transformers import util

        # Build failure context text
        context_parts = [failure_code]
        if failure_reason:
            context_parts.append(failure_reason)
        if amount > 0:
            context_parts.append(f"amount {amount}")
        context_text = " ".join(context_parts)

        # Embed failure context via SDK
        context_embedding = model.encode(context_text, convert_to_tensor=True)

        # Compute similarity against each pre-computed rail embedding via SDK
        similarities = []
        for rail_id, rail_embedding in self._rail_embeddings.items():
            rail_tensor = torch.tensor(rail_embedding)
            sim = float(util.cos_sim(context_embedding, rail_tensor)[0][0])
            similarities.append({"rail": rail_id, "similarity": round(sim, 4)})

        # Sort by similarity, return top-k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

    # ═══════════════════════════════════════════════════════════════
    # PROCESS ENRICHMENT — add rails connected via process edges
    # ═══════════════════════════════════════════════════════════════

    def enrich_with_process_edges(self, rails: list[dict]) -> list[dict]:
        """Phase 2: Add rails connected via business process edges.

        For each rail in the semantic results, find rails connected
        via process edges (pr:hasNext). This fixes false negatives —
        rails that are semantically distant but process-critical.
        """
        enriched = {r["rail"]: r for r in rails}

        for rail_result in rails:
            rail_id = rail_result["rail"]
            if rail_id not in self.graph:
                continue

            # Find outgoing process edges
            for _, target, data in self.graph.out_edges(rail_id, data=True):
                if data.get("edge_type") == "process" and target not in enriched:
                    enriched[target] = {
                        "rail": target,
                        "similarity": 0.0,
                        "source": "process_edge",
                        "process_type": data.get("process_type", ""),
                        "from_rail": rail_id,
                    }

            # Find incoming process edges (prerequisites)
            for source, _, data in self.graph.in_edges(rail_id, data=True):
                if data.get("edge_type") == "process" and source not in enriched:
                    enriched[source] = {
                        "rail": source,
                        "similarity": 0.0,
                        "source": "process_edge",
                        "process_type": data.get("process_type", ""),
                        "to_rail": rail_id,
                    }

        return list(enriched.values())

    # ═══════════════════════════════════════════════════════════════
    # COMBINED DISCOVERY — semantic + process enrichment
    # ═══════════════════════════════════════════════════════════════

    def discover_recovery_rails(
        self,
        failure_code: str,
        failure_reason: str = "",
        amount: float = 0.0,
        customer_id: str = "",
        preferred_channel: str = "",
    ) -> dict:
        """Two-phase discovery: semantic retrieval + process enrichment.

        Returns:
            {
                "rails": [...],           # Discovered rails with metadata
                "recommended": "rail_name",  # Best rail to try
                "process_order": [...],   # Recommended execution order
                "discovery_method": "semantic+process",
            }
        """
        # Phase 1: Semantic retrieval
        semantic_results = self.semantic_discover_rails(
            failure_code=failure_code,
            failure_reason=failure_reason,
            amount=amount,
            top_k=3,
        )

        # Phase 2: Process enrichment
        enriched_results = self.enrich_with_process_edges(semantic_results)

        # Add full rail details to each result
        for result in enriched_results:
            rail_id = result["rail"]
            details = RAIL_DETAILS.get(rail_id, {})
            result["label"] = details.get("label", rail_id)
            result["api"] = details.get("api", "")
            result["description"] = details.get("description", "")
            result["channels"] = details.get("channels", [])
            result["conversion_cost"] = details.get("conversion_cost", 0.0)
            result["failure_modes"] = details.get("failure_modes", [])
            result["prerequisites"] = details.get("prerequisites", [])
            result["follow_ups"] = details.get("follow_ups", [])

        # Build process order (topological sort of discovered rails)
        process_order = self._build_process_order(enriched_results)

        # Recommend best rail
        recommended = self._recommend_best_rail(
            enriched_results, preferred_channel, failure_code
        )

        return {
            "rails": enriched_results,
            "recommended": recommended,
            "process_order": process_order,
            "discovery_method": "semantic+process",
            "failure_code": failure_code,
        }

    def _build_process_order(self, rails: list[dict]) -> list[str]:
        """Build recommended execution order from process edges.

        Uses topological sort on the subgraph of discovered rails
        to determine the correct execution sequence.
        """
        rail_ids = [r["rail"] for r in rails]

        # Build subgraph of discovered rails with process edges
        subgraph = nx.DiGraph()
        for rail_id in rail_ids:
            subgraph.add_node(rail_id)

        for rail_id in rail_ids:
            if rail_id not in self.graph:
                continue
            for _, target, data in self.graph.out_edges(rail_id, data=True):
                if data.get("edge_type") == "process" and target in rail_ids:
                    subgraph.add_edge(rail_id, target)

        # Topological sort for execution order
        try:
            order = list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            # Cycle detected — fall back to discovery order
            order = rail_ids

        return order

    def _recommend_best_rail(
        self,
        rails: list[dict],
        preferred_channel: str,
        failure_code: str,
    ) -> str:
        """Recommend the best rail from discovered results.

        Considers: semantic similarity, channel compatibility, conversion cost.
        """
        if not rails:
            return "payment_link"

        failure_type = self._resolve_failure_type(failure_code)

        scored = []
        for rail in rails:
            rail_id = rail["rail"]
            score = rail.get("similarity", 0.0)

            # Boost if rail handles this failure type
            failure_modes = RAIL_DETAILS.get(rail_id, {}).get("failure_modes", [])
            if failure_code in failure_modes:
                score += 0.2

            # Boost if channel compatible
            if preferred_channel:
                channels = RAIL_DETAILS.get(rail_id, {}).get("channels", [])
                if preferred_channel in channels:
                    score += 0.1

            # Penalize high conversion cost
            conversion_cost = RAIL_DETAILS.get(rail_id, {}).get("conversion_cost", 0.0)
            score -= conversion_cost * 0.5

            scored.append({"rail": rail_id, "score": round(score, 4)})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[0]["rail"]

    # ═══════════════════════════════════════════════════════════════
    # EXISTING METHODS (preserved from original)
    # ═══════════════════════════════════════════════════════════════

    def discover_recovery_path(
        self,
        failure_code: str,
        customer_id: str = "",
        preferred_channel: str = "",
    ) -> list[str]:
        """Find best recovery rail path via graph traversal.

        Uses Dijkstra's shortest path weighted by conversion cost.
        Falls back to BFS if no weighted path exists.
        """
        failure_type = self._resolve_failure_type(failure_code)
        starting_rails = FAILURE_TO_STARTING_RAILS.get(failure_type, ["card_gateway"])

        best_path: list[str] = []
        best_cost = float("inf")

        for start in starting_rails:
            if start not in self.graph:
                continue
            try:
                for target in self.graph.nodes():
                    if target == start:
                        continue
                    try:
                        path = nx.shortest_path(self.graph, start, target)
                        cost = self._path_cost(path, preferred_channel)
                        if cost < best_cost:
                            best_cost = cost
                            best_path = path
                    except nx.NetworkXNoPath:
                        continue
            except nx.NodeNotFound:
                continue

        if not best_path and starting_rails:
            best_path = [starting_rails[0]]

        return best_path

    def _path_cost(self, path: list[str], preferred_channel: str = "") -> float:
        """Calculate total cost of a path (lower = better)."""
        if len(path) < 2:
            return 0.0

        total_cost = 0.0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                weight = edge_data.get("weight", 0.5)
                total_cost += 1.0 - weight
            else:
                total_cost += 1.0

            if preferred_channel:
                rail_data = self.graph.nodes.get(path[i + 1], {})
                channels = rail_data.get("channels", [])
                if preferred_channel not in channels:
                    total_cost += 0.2

        return total_cost

    def get_rail_details(self, rail_name: str) -> dict:
        """Return API endpoint schemas, parameters, and conversion costs."""
        return RAIL_DETAILS.get(rail_name, {})

    def recommend_optimal_rail(
        self,
        failure_code: str,
        preferred_channel: str = "",
    ) -> str:
        """Return the single best recommended rail for a failure type."""
        path = self.discover_recovery_path(
            failure_code=failure_code,
            preferred_channel=preferred_channel,
        )
        if path:
            return path[-1]
        return "payment_link"

    def get_available_rails(self, failure_code: str) -> list[str]:
        """List all reachable rails from the failure's starting point."""
        failure_type = self._resolve_failure_type(failure_code)
        starting_rails = FAILURE_TO_STARTING_RAILS.get(failure_type, ["card_gateway"])

        reachable: set[str] = set()
        for start in starting_rails:
            if start in self.graph:
                reachable.update(nx.descendants(self.graph, start))
                reachable.add(start)

        return sorted(reachable)

    def is_rail_viable(self, rail_name: str, failure_code: str) -> bool:
        """Check if a specific rail is reachable for a given failure type."""
        available = self.get_available_rails(failure_code)
        return rail_name in available

    def update_edge_weight(
        self,
        source_rail: str,
        target_rail: str,
        success: bool,
        learning_rate: float = 0.1,
    ) -> None:
        """Update edge weight based on historical conversion outcome."""
        if not self.graph.has_edge(source_rail, target_rail):
            return

        current_weight = self.graph[source_rail][target_rail].get("weight", 0.5)
        outcome = 1.0 if success else 0.0
        new_weight = current_weight + learning_rate * (outcome - current_weight)
        new_weight = max(0.1, min(0.95, new_weight))

        self.graph[source_rail][target_rail]["weight"] = new_weight
        self.graph[source_rail][target_rail]["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.graph[source_rail][target_rail]["conversion_count"] = (
            self.graph[source_rail][target_rail].get("conversion_count", 0) + 1
        )

    def get_edge_stats(self, source_rail: str, target_rail: str) -> dict:
        """Get statistics for a specific edge."""
        if not self.graph.has_edge(source_rail, target_rail):
            return {"error": "edge_not_found"}
        edge_data = self.graph[source_rail][target_rail]
        return {
            "weight": edge_data.get("weight", 0.5),
            "reason": edge_data.get("reason", ""),
            "edge_type": edge_data.get("edge_type", "failure"),
            "last_updated": edge_data.get("last_updated", ""),
            "conversion_count": edge_data.get("conversion_count", 0),
        }

    @staticmethod
    def _resolve_failure_type(failure_code: str) -> FailureType:
        """Map a Razorpay failure code to a FailureType.

        This used to carry its own six-entry table, which missed the codes
        Razorpay actually sends — `risk_check_failed`, `gateway_timeout`,
        `do_not_honor` and the rest all fell through to UNKNOWN, so the router
        gave generic advice for most real failures. Share the diagnosis engine's
        table instead, so the two can never disagree about what a code means.
        """
        from recovery_agent.agent.diagnosis import FAILURE_CODE_MAP

        code = (failure_code or "").strip().lower()
        if code in FAILURE_CODE_MAP:
            return FAILURE_CODE_MAP[code]
        # Razorpay codes vary by method; fall back to substring matching.
        for known, ftype in FAILURE_CODE_MAP.items():
            if known in code:
                return ftype
        return FailureType.UNKNOWN
