"""Unit tests for Knowledge Graph Router.

Covers: graph initialization, node/edge schema, traversal for all failure types,
multi-rail path discovery, channel preferences, fallback traversal, and
integration with the decision layer.
"""
from __future__ import annotations

import pytest

from recovery_agent.agent.kg_router import RazorpayKnowledgeGraph, RAIL_DETAILS
from recovery_agent.agent.decision import decide_intervention, run_decision
from recovery_agent.models import (
    ActionType,
    Case,
    Diagnosis,
    FailureType,
    PaymentEvent,
)


# --- Graph Initialization ---

class TestGraphInitialization:
    def test_graph_has_all_nodes(self):
        kg = RazorpayKnowledgeGraph()
        expected_nodes = set(RAIL_DETAILS.keys())
        assert expected_nodes.issubset(set(kg.graph.nodes()))

    def test_graph_is_directed(self):
        kg = RazorpayKnowledgeGraph()
        import networkx as nx
        assert isinstance(kg.graph, nx.DiGraph)

    def test_graph_has_edges(self):
        kg = RazorpayKnowledgeGraph()
        assert kg.graph.number_of_edges() > 0

    def test_node_metadata_populated(self):
        kg = RazorpayKnowledgeGraph()
        for node in RAIL_DETAILS:
            data = kg.graph.nodes[node]
            assert "label" in data
            assert "api" in data
            assert "channels" in data


# --- Edge Schema ---

class TestEdgeSchema:
    def test_edges_have_weights(self):
        kg = RazorpayKnowledgeGraph()
        for u, v, data in kg.graph.edges(data=True):
            assert "weight" in data, f"Edge {u} -> {v} missing weight"

    def test_edges_have_failure_types(self):
        kg = RazorpayKnowledgeGraph()
        for u, v, data in kg.graph.edges(data=True):
            assert "failure_types" in data, f"Edge {u} -> {v} missing failure_types"

    def test_card_gateway_has_outgoing_edges(self):
        kg = RazorpayKnowledgeGraph()
        outgoing = list(kg.graph.successors("card_gateway"))
        assert len(outgoing) >= 3


# --- Recovery Path Discovery ---

class TestRecoveryPathDiscovery:
    def test_card_expired_discovery(self):
        kg = RazorpayKnowledgeGraph()
        path = kg.discover_recovery_path("card_expired")
        assert len(path) > 0
        assert path[0] in ("card_gateway",)

    def test_insufficient_funds_discovery(self):
        kg = RazorpayKnowledgeGraph()
        path = kg.discover_recovery_path("insufficient_funds")
        assert len(path) > 0

    def test_bank_declined_discovery(self):
        kg = RazorpayKnowledgeGraph()
        path = kg.discover_recovery_path("bank_declined")
        assert len(path) > 0

    def test_network_timeout_discovery(self):
        kg = RazorpayKnowledgeGraph()
        path = kg.discover_recovery_path("network_timeout")
        assert len(path) > 0

    def test_mandate_revoked_discovery(self):
        kg = RazorpayKnowledgeGraph()
        path = kg.discover_recovery_path("mandate_revoked")
        assert len(path) > 0
        # Should start from upi_autopay or payment_link
        assert path[0] in ("upi_autopay", "payment_link")

    def test_risk_block_discovery(self):
        kg = RazorpayKnowledgeGraph()
        path = kg.discover_recovery_path("risk_block")
        assert len(path) > 0
        # Should start from payment_link or magic_checkout
        assert path[0] in ("payment_link", "magic_checkout")

    def test_unknown_failure_discovery(self):
        kg = RazorpayKnowledgeGraph()
        path = kg.discover_recovery_path("unknown_failure")
        assert len(path) > 0

    def test_path_is_valid_graph_path(self):
        kg = RazorpayKnowledgeGraph()
        for failure_code in ["card_expired", "bank_declined", "network_timeout"]:
            path = kg.discover_recovery_path(failure_code)
            # Each consecutive pair should be connected
            for i in range(len(path) - 1):
                assert kg.graph.has_edge(path[i], path[i + 1]) or path[i] == path[i + 1]


# --- Channel Preference ---

class TestChannelPreference:
    def test_sms_preference_favors_payment_link(self):
        kg = RazorpayKnowledgeGraph()
        path = kg.discover_recovery_path(
            "card_expired", preferred_channel="sms"
        )
        # payment_link supports sms, so it should appear
        assert "payment_link" in path or "upi_autopay" in path

    def test_whatsapp_preference(self):
        kg = RazorpayKnowledgeGraph()
        path = kg.discover_recovery_path(
            "card_expired", preferred_channel="whatsapp"
        )
        assert len(path) > 0

    def test_channel_affects_cost(self):
        kg = RazorpayKnowledgeGraph()
        # payment_link supports sms, escrow does not
        cost_with = kg._path_cost(["card_gateway", "payment_link"], "sms")
        cost_without = kg._path_cost(["card_gateway", "escrow_route"], "sms")
        # payment_link should have lower cost (sms is in its channels)
        assert cost_with < cost_without


# --- Rail Details ---

class TestRailDetails:
    def test_get_rail_details_valid(self):
        kg = RazorpayKnowledgeGraph()
        details = kg.get_rail_details("card_gateway")
        assert details["label"] == "Card Gateway"
        assert details["api"] == "/payments"
        assert "channels" in details

    def test_get_rail_details_invalid(self):
        kg = RazorpayKnowledgeGraph()
        details = kg.get_rail_details("nonexistent_rail")
        assert details == {}

    def test_all_rails_have_required_fields(self):
        kg = RazorpayKnowledgeGraph()
        for rail_id in RAIL_DETAILS:
            details = kg.get_rail_details(rail_id)
            assert "label" in details
            assert "api" in details
            assert "conversion_cost" in details
            assert "channels" in details


# --- Recommend Optimal Rail ---

class TestRecommendOptimalRail:
    def test_card_expired_recommends_valid_rail(self):
        kg = RazorpayKnowledgeGraph()
        rail = kg.recommend_optimal_rail("card_expired")
        assert rail in RAIL_DETAILS

    def test_bank_declined_recommends_valid_rail(self):
        kg = RazorpayKnowledgeGraph()
        rail = kg.recommend_optimal_rail("bank_declined")
        assert rail in RAIL_DETAILS

    def test_recommendation_with_channel(self):
        kg = RazorpayKnowledgeGraph()
        rail = kg.recommend_optimal_rail("card_expired", preferred_channel="sms")
        assert rail in RAIL_DETAILS

    def test_recommendation_always_returns_string(self):
        kg = RazorpayKnowledgeGraph()
        for code in ["card_expired", "bank_declined", "network_timeout",
                      "mandate_revoked", "risk_block", "insufficient_funds"]:
            rail = kg.recommend_optimal_rail(code)
            assert isinstance(rail, str)
            assert len(rail) > 0


# --- Available Rails ---

class TestAvailableRails:
    def test_card_expired_available_rails(self):
        kg = RazorpayKnowledgeGraph()
        rails = kg.get_available_rails("card_expired")
        assert "card_gateway" in rails
        assert len(rails) >= 3

    def test_mandate_revoked_available_rails(self):
        kg = RazorpayKnowledgeGraph()
        rails = kg.get_available_rails("mandate_revoked")
        assert "upi_autopay" in rails or "payment_link" in rails

    def test_rail_viability_check(self):
        kg = RazorpayKnowledgeGraph()
        assert kg.is_rail_viable("payment_link", "card_expired")
        assert kg.is_rail_viable("upi_autopay", "card_expired")


# --- Fallback Traversal ---

class TestFallbackTraversal:
    def test_single_rail_failure_still_finds_path(self):
        """If primary rail has no outgoing edges for a failure type, should fallback."""
        kg = RazorpayKnowledgeGraph()
        # All failure types should produce a non-empty path
        for code in ["card_expired", "bank_declined", "network_timeout",
                      "mandate_revoked", "risk_block", "insufficient_funds"]:
            path = kg.discover_recovery_path(code)
            assert len(path) >= 1

    def test_nonexistent_starting_rail_fallback(self):
        """If failure code maps to unknown starting rail, should still work."""
        kg = RazorpayKnowledgeGraph()
        path = kg.discover_recovery_path("totally_unknown_error")
        assert len(path) >= 1


# --- Decision Integration ---

class TestDecisionIntegration:
    def test_decision_populates_kg_metadata(self):
        """decide_intervention should populate rail metadata in case."""
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1",
                customer_id="cust_001",
                amount=5000,
                failure_reason="card expired",
                failure_code="card_expired",
            ),
        )
        case.diagnosis = Diagnosis(
            root_cause=FailureType.CARD_EXPIRED,
            confidence=0.9,
            reasoning="Test",
        )
        kg = RazorpayKnowledgeGraph()
        decide_intervention(case, kg_router=kg)
        assert "discovered_rail_path" in case.payment.metadata
        assert "recommended_api_rail" in case.payment.metadata
        assert len(case.payment.metadata["discovered_rail_path"]) > 0

    def test_run_decision_populates_kg_metadata(self):
        """run_decision should populate rail metadata in case."""
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1",
                customer_id="cust_001",
                amount=5000,
                failure_reason="bank declined",
                failure_code="bank_declined",
            ),
        )
        case.diagnosis = Diagnosis(
            root_cause=FailureType.BANK_DECLINED,
            confidence=0.9,
            reasoning="Test",
        )
        kg = RazorpayKnowledgeGraph()
        case = run_decision(case, kg_router=kg)
        assert case.payment.metadata.get("recommended_api_rail") in RAIL_DETAILS

    def test_no_kg_router_backward_compatible(self):
        """Decision without KG router should work identically to before."""
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1",
                customer_id="cust_001",
                amount=5000,
                failure_reason="network timeout",
                failure_code="network_timeout",
            ),
        )
        case.diagnosis = Diagnosis(
            root_cause=FailureType.NETWORK_TIMEOUT,
            confidence=0.9,
            reasoning="Test",
        )
        action = decide_intervention(case)
        assert action == ActionType.RETRY_PAYMENT

    def test_risk_block_uses_kg_path(self):
        """Risk block should find a path through magic_checkout."""
        case = Case(
            payment=PaymentEvent(
                payment_id="pay_1",
                customer_id="cust_001",
                amount=5000,
                failure_reason="risk block",
                failure_code="risk_block",
            ),
        )
        case.diagnosis = Diagnosis(
            root_cause=FailureType.RISK_BLOCK,
            confidence=0.9,
            reasoning="Test",
        )
        kg = RazorpayKnowledgeGraph()
        decide_intervention(case, kg_router=kg)
        path = case.payment.metadata["discovered_rail_path"]
        # Risk block should go through payment_link or magic_checkout
        assert any(r in path for r in ["payment_link", "magic_checkout", "escrow_route"])
