"""Tests for graph.py — verify ReAct loop structure.

These tests verify the graph has the correct nodes and edges.
If you add a node or change an edge, these tests will catch it.
"""
from recovery_agent.agent.graph import build_graph, build_initial_state
from recovery_agent.models import Case, PaymentEvent


class TestGraphStructure:
    def test_graph_builds_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        # Get the compiled graph's node names
        nodes = set(graph.get_graph().nodes)
        expected = {
            "__start__", "__end__",
            "agent", "tools", "stopping_check", "self_critique",
            "tool_repetition_guard", "human_approval_gate", "mask_outputs",
        }
        assert expected.issubset(nodes), f"Missing nodes: {expected - nodes}"

    def test_graph_has_agent_node(self):
        graph = build_graph()
        nodes = set(graph.get_graph().nodes)
        assert "agent" in nodes

    def test_graph_has_tools_node(self):
        graph = build_graph()
        nodes = set(graph.get_graph().nodes)
        assert "tools" in nodes

    def test_graph_has_stopping_check(self):
        graph = build_graph()
        nodes = set(graph.get_graph().nodes)
        assert "stopping_check" in nodes

    def test_graph_has_self_critique(self):
        graph = build_graph()
        nodes = set(graph.get_graph().nodes)
        assert "self_critique" in nodes

    def test_graph_has_guard_nodes(self):
        graph = build_graph()
        nodes = set(graph.get_graph().nodes)
        assert "tool_repetition_guard" in nodes
        assert "human_approval_gate" in nodes
        assert "mask_outputs" in nodes


class TestBuildInitialState:
    def test_initial_state_has_messages(self):
        case = Case(payment=PaymentEvent(
            payment_id="pay_1", customer_id="c1", amount=1000,
            failure_code="51", failure_reason="Insufficient funds",
        ))
        state = build_initial_state(case)
        assert "messages" in state
        assert len(state["messages"]) > 0

    def test_initial_state_first_message_is_human(self):
        case = Case(payment=PaymentEvent(
            payment_id="pay_1", customer_id="c1", amount=1000,
            failure_code="51", failure_reason="Insufficient funds",
        ))
        state = build_initial_state(case)
        from langchain_core.messages import HumanMessage
        assert isinstance(state["messages"][0], HumanMessage)

    def test_initial_state_contains_payment_id(self):
        case = Case(payment=PaymentEvent(
            payment_id="pay_abc123", customer_id="c1", amount=1000,
        ))
        state = build_initial_state(case)
        content = state["messages"][0].content
        assert "pay_abc123" in content

    def test_initial_state_contains_amount(self):
        case = Case(payment=PaymentEvent(
            payment_id="pay_1", customer_id="c1", amount=50000,
        ))
        state = build_initial_state(case)
        content = state["messages"][0].content
        assert "50,000" in content

    def test_initial_state_contains_failure_code(self):
        case = Case(payment=PaymentEvent(
            payment_id="pay_1", customer_id="c1", amount=1000,
            failure_code="51",
        ))
        state = build_initial_state(case)
        content = state["messages"][0].content
        assert "51" in content

    def test_initial_state_has_tool_call_history(self):
        case = Case(payment=PaymentEvent(
            payment_id="pay_1", customer_id="c1", amount=1000,
        ))
        state = build_initial_state(case)
        assert "tool_call_history" in state
        assert state["tool_call_history"] == []
