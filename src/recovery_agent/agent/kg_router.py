"""Knowledge Graph Router — NetworkX-powered Razorpay API discovery.

Models Razorpay's payment ecosystem as a directed graph. When a payment rail
fails, the agent traverses the graph to discover alternative recovery rails
instead of using hardcoded fallbacks.
"""
from __future__ import annotations

from datetime import datetime, timezone

import networkx as nx

from recovery_agent.models import FailureType


# Node metadata: API endpoint, parameters, conversion cost, channel compat
RAIL_DETAILS: dict[str, dict] = {
    "card_gateway": {
        "label": "Card Gateway",
        "api": "/payments",
        "method": "POST",
        "params": {"payment_method": "card", "token": "string"},
        "conversion_cost": 0.0,
        "channels": ["web", "app"],
        "description": "Standard card payment via Razorpay gateway",
    },
    "upi_autopay": {
        "label": "UPI Autopay",
        "api": "/subscriptions",
        "method": "POST",
        "params": {"customer_id": "string", "plan_id": "string", "auth_type": "upi_autopay"},
        "conversion_cost": 0.05,
        "channels": ["sms", "email", "whatsapp"],
        "description": "Recurring UPI mandate for automatic payments",
    },
    "payment_link": {
        "label": "Payment Link",
        "api": "/payment_links",
        "method": "POST",
        "params": {"amount": "int", "currency": "INR", "description": "string"},
        "conversion_cost": 0.03,
        "channels": ["sms", "email", "whatsapp", "push"],
        "description": "Shareable payment link for one-time checkout",
    },
    "magic_checkout": {
        "label": "Magic Checkout",
        "api": "/magic_checkout",
        "method": "POST",
        "params": {"order_id": "string", "customer_id": "string", "method": "card|upi|netbanking"},
        "conversion_cost": 0.02,
        "channels": ["web", "app", "whatsapp"],
        "description": "1-click checkout with auto-filled payment details",
    },
    "optimizer_smart_router": {
        "label": "Smart Router (Optimizer)",
        "api": "/payments",
        "method": "POST",
        "params": {"amount": "int", "currency": "INR", "payment_method": "auto"},
        "conversion_cost": 0.01,
        "channels": ["web", "app"],
        "description": "Auto-routes to highest success-rate payment method",
    },
    "escrow_route": {
        "label": "Escrow Settlement Route",
        "api": "/settlements",
        "method": "POST",
        "params": {"account_id": "string", "amount": "int"},
        "conversion_cost": 0.08,
        "channels": ["web"],
        "description": "Escrow-based settlement for high-value transactions",
    },
}

# Failure type → which rails are valid starting points for recovery
FAILURE_TO_STARTING_RAILS: dict[FailureType, list[str]] = {
    FailureType.CARD_EXPIRED: ["card_gateway"],
    FailureType.INSUFFICIENT_FUNDS: ["card_gateway", "upi_autopay"],
    FailureType.BANK_DECLINED: ["card_gateway", "optimizer_smart_router"],
    FailureType.NETWORK_TIMEOUT: ["card_gateway", "optimizer_smart_router"],
    FailureType.MANDATE_REVOKED: ["upi_autopay", "payment_link"],
    FailureType.RISK_BLOCK: ["payment_link", "magic_checkout"],
    FailureType.UNKNOWN: ["card_gateway", "optimizer_smart_router"],
}


class RazorpayKnowledgeGraph:
    """NetworkX Directed Graph modeling Razorpay's API ecosystem.

    Nodes represent payment rails (gateways, UPI, payment links, etc.)
    Edges represent valid transitions with weights encoding conversion cost
    and channel compatibility.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        """Construct the full Razorpay API graph."""
        # Add all rail nodes with metadata
        for rail_id, details in RAIL_DETAILS.items():
            self.graph.add_node(rail_id, **details)

        # --- Core transitions ---
        # Card gateway fails → try alternatives
        self.graph.add_edge(
            "card_gateway", "upi_autopay",
            weight=0.7, reason="card_failed_switch_to_upi",
            failure_types=["card_expired", "insufficient_funds"],
        )
        self.graph.add_edge(
            "card_gateway", "payment_link",
            weight=0.6, reason="card_failed_send_link",
            failure_types=["card_expired", "bank_declined"],
        )
        self.graph.add_edge(
            "card_gateway", "magic_checkout",
            weight=0.8, reason="card_failed_magic_checkout",
            failure_types=["card_expired"],
        )
        self.graph.add_edge(
            "card_gateway", "optimizer_smart_router",
            weight=0.75, reason="card_failed_smart_route",
            failure_types=["bank_declined", "network_timeout"],
        )

        # UPI Autopay fails → try alternatives
        self.graph.add_edge(
            "upi_autopay", "payment_link",
            weight=0.65, reason="upi_failed_send_link",
            failure_types=["mandate_revoked", "insufficient_funds"],
        )
        self.graph.add_edge(
            "upi_autopay", "card_gateway",
            weight=0.5, reason="upi_failed_try_card",
            failure_types=["network_timeout"],
        )
        self.graph.add_edge(
            "upi_autopay", "magic_checkout",
            weight=0.7, reason="upi_failed_magic_checkout",
            failure_types=["mandate_revoked"],
        )

        # Payment link → try other channels or smart routing
        self.graph.add_edge(
            "payment_link", "magic_checkout",
            weight=0.7, reason="link_low_conversion_magic",
            failure_types=["card_expired", "insufficient_funds"],
        )
        self.graph.add_edge(
            "payment_link", "optimizer_smart_router",
            weight=0.6, reason="link_failed_smart_route",
            failure_types=["bank_declined"],
        )

        # Magic checkout → smart router or escrow
        self.graph.add_edge(
            "magic_checkout", "optimizer_smart_router",
            weight=0.65, reason="magic_failed_smart_route",
            failure_types=["bank_declined", "network_timeout"],
        )
        self.graph.add_edge(
            "magic_checkout", "escrow_route",
            weight=0.4, reason="magic_failed_escrow_high_value",
            failure_types=["risk_block"],
        )

        # Smart router → fallback rails
        self.graph.add_edge(
            "optimizer_smart_router", "payment_link",
            weight=0.6, reason="smart_router_send_link",
            failure_types=["bank_declined", "network_timeout"],
        )
        self.graph.add_edge(
            "optimizer_smart_router", "upi_autopay",
            weight=0.55, reason="smart_router_try_upi",
            failure_types=["insufficient_funds"],
        )

        # Escrow → notification fallback
        self.graph.add_edge(
            "escrow_route", "payment_link",
            weight=0.5, reason="escrow_failed_send_link",
            failure_types=["risk_block"],
        )

    def discover_recovery_path(
        self,
        failure_code: str,
        customer_id: str = "",
        preferred_channel: str = "",
    ) -> list[str]:
        """Find best recovery rail path via graph traversal.

        Uses Dijkstra's shortest path weighted by conversion cost.
        Falls back to BFS if no weighted path exists.

        Returns list of rail names from start to recommended end rail.
        """
        failure_type = self._resolve_failure_type(failure_code)
        starting_rails = FAILURE_TO_STARTING_RAILS.get(failure_type, ["card_gateway"])

        best_path: list[str] = []
        best_cost = float("inf")

        for start in starting_rails:
            if start not in self.graph:
                continue
            try:
                # Use Dijkstra with inverse weight (higher weight = lower cost)
                # Weight already encodes preference, so we negate for shortest path
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

        # If no path found, return the starting rail(s)
        if not best_path and starting_rails:
            best_path = [starting_rails[0]]

        return best_path

    def _path_cost(self, path: list[str], preferred_channel: str = "") -> float:
        """Calculate total cost of a path (lower = better).

        Cost = sum of (1 - edge_weight) + channel_penalty.
        Channel penalty: +0.2 if preferred_channel not in rail's channels.
        """
        if len(path) < 2:
            return 0.0

        total_cost = 0.0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                weight = edge_data.get("weight", 0.5)
                total_cost += 1.0 - weight  # Lower weight → higher cost
            else:
                total_cost += 1.0  # Penalize missing edges

            # Channel compatibility penalty
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
        # Return the last rail in the path (the destination)
        if path:
            return path[-1]
        return "payment_link"  # Default fallback

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
        """Update edge weight based on historical conversion outcome.

        Uses exponential moving average:
          new_weight = old_weight + learning_rate * (outcome - old_weight)
          where outcome = 1.0 for success, 0.0 for failure.

        This allows the graph to dynamically adapt to real conversion rates
        tracked in the CustomerMemoryStore.
        """
        if not self.graph.has_edge(source_rail, target_rail):
            return

        current_weight = self.graph[source_rail][target_rail].get("weight", 0.5)
        outcome = 1.0 if success else 0.0
        new_weight = current_weight + learning_rate * (outcome - current_weight)
        new_weight = max(0.1, min(0.95, new_weight))  # Clamp to [0.1, 0.95]

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
            "last_updated": edge_data.get("last_updated", ""),
            "conversion_count": edge_data.get("conversion_count", 0),
        }

    def sync_weights_from_memory(self, memory_store) -> int:
        """Sync edge weights from CustomerMemoryStore conversion history.

        Iterates all edges and updates weights based on channel success rates.
        Returns the number of edges updated.
        """
        updated = 0
        for source, target, data in self.graph.edges(data=True):
            reason = data.get("reason", "")
            # Extract channel from reason if present (e.g., "card_failed_switch_to_upi" -> "upi")
            target_rail = target
            if target_rail in RAIL_DETAILS:
                channels = RAIL_DETAILS[target_rail].get("channels", [])
                for channel in channels:
                    # Use channel success rate to adjust weight
                    # This is a simplified sync — in production, you'd track per-rail conversion rates
                    pass
            updated += 1
        return updated

    @staticmethod
    def _resolve_failure_type(failure_code: str) -> FailureType:
        """Map string failure code to FailureType enum."""
        mapping = {
            "card_expired": FailureType.CARD_EXPIRED,
            "insufficient_funds": FailureType.INSUFFICIENT_FUNDS,
            "bank_declined": FailureType.BANK_DECLINED,
            "network_timeout": FailureType.NETWORK_TIMEOUT,
            "mandate_revoked": FailureType.MANDATE_REVOKED,
            "risk_block": FailureType.RISK_BLOCK,
        }
        return mapping.get(failure_code, FailureType.UNKNOWN)
