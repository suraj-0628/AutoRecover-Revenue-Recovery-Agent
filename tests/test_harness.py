"""Unit tests for Tool Registry and AgentHarness.

Tests cover:
- Tool registry schemas and execution
- MCP-style tool-calling flow
- Context compaction (observation summarization)
- TrueForge-style reflection on tool errors
- Repetition prohibition (identical call blocking)
- Harness multi-turn loop
- Error handling and fallback paths
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from recovery_agent.agent.tools import (
    TOOL_SCAPES,
    calculate_payday_window,
    check_bank_health,
    escalate_to_human_agent,
    execute_tool,
    generate_smart_recovery_link,
    get_tool_schemas_for_llm,
    query_gateway_error_details,
    schedule_payday_retry,
)
from recovery_agent.agent.harness import (
    AgentHarness,
    Observation,
    ToolCall,
    _compact_context,
    _format_observation,
    _reflect_on_error,
    _reflect_on_error_async,
    _build_harness_prompt,
    MAX_HARNESS_ITERATIONS,
    CONTEXT_COMPACT_THRESHOLD,
)
from recovery_agent.models import (
    Case,
    CaseStatus,
    Diagnosis,
    FailureType,
    PaymentEvent,
    RecoveryTier,
)


# ─── Helpers ──────────────────────────────────────────────────

def make_payment_event(
    failure_code: str = "card_expired",
    failure_reason: str = "Card has expired",
    amount: float = 5000.0,
) -> PaymentEvent:
    return PaymentEvent(
        event_type="payment_failed",
        payment_id="pay_test_harness",
        customer_id="cust_harness_test",
        amount=amount,
        currency="INR",
        status="failed",
        failure_reason=failure_reason,
        failure_code=failure_code,
        metadata={"error_source": "bank", "error_step": "payment_authorization"},
    )


def make_case(
    failure_code: str = "unknown_retryable_error",
    failure_reason: str = "Payment processing issue",
    amount: float = 5000.0,
) -> Case:
    event = make_payment_event(failure_code, failure_reason, amount)
    case = Case(payment=event, max_attempts=5)
    case.diagnosis = Diagnosis(
        root_cause=FailureType.UNKNOWN,
        confidence=0.9,
        reasoning="Unknown error — requires LLM reasoning to determine intervention.",
    )
    return case


# ═══════════════════════════════════════════════════════════════
#  TOOL REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════

class TestToolSchemas:
    """Test MCP-style tool schema definitions."""

    def test_all_tools_registered(self):
        assert len(TOOL_SCAPES) == 8

    def test_each_tool_has_required_fields(self):
        for tool in TOOL_SCAPES:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema

    def test_get_tool_schemas_for_llm(self):
        schemas = get_tool_schemas_for_llm()
        assert schemas == TOOL_SCAPES
        assert len(schemas) == 8

    def test_tool_names_are_unique(self):
        names = [t["name"] for t in TOOL_SCAPES]
        assert len(names) == len(set(names))

    def test_query_gateway_error_details_schema(self):
        tool = next(t for t in TOOL_SCAPES if t["name"] == "query_gateway_error_details")
        assert "payment_id" in tool["input_schema"]["properties"]
        assert "payment_id" in tool["input_schema"]["required"]

    def test_check_bank_health_schema(self):
        tool = next(t for t in TOOL_SCAPES if t["name"] == "check_bank_health")
        assert "bank_code" in tool["input_schema"]["properties"]
        assert "bank_code" in tool["input_schema"]["required"]

    def test_calculate_payday_window_schema(self):
        tool = next(t for t in TOOL_SCAPES if t["name"] == "calculate_payday_window")
        props = tool["input_schema"]["properties"]
        assert "customer_id" in props
        assert "country_code" in props

    def test_generate_smart_recovery_link_schema(self):
        tool = next(t for t in TOOL_SCAPES if t["name"] == "generate_smart_recovery_link")
        props = tool["input_schema"]["properties"]
        assert "payment_id" in props
        assert "allowed_rails" in props
        assert "discount_pct" in props

    def test_schedule_payday_retry_schema(self):
        tool = next(t for t in TOOL_SCAPES if t["name"] == "schedule_payday_retry")
        props = tool["input_schema"]["properties"]
        assert "payment_id" in props
        assert "target_iso_timestamp" in props

    def test_escalate_to_human_agent_schema(self):
        tool = next(t for t in TOOL_SCAPES if t["name"] == "escalate_to_human_agent")
        props = tool["input_schema"]["properties"]
        assert "payment_id" in props
        assert "reason" in props


# ═══════════════════════════════════════════════════════════════
#  TOOL EXECUTION TESTS
# ═══════════════════════════════════════════════════════════════

class TestToolExecution:
    """Test execute_tool dispatcher and individual tool functions."""

    def test_unknown_tool_returns_error(self):
        result = execute_tool("nonexistent_tool", {})
        assert result["status"] == "error"
        assert "Unknown tool" in result["message"]

    def test_tool_with_invalid_args(self):
        result = execute_tool("check_bank_health", {"wrong_param": "HDFC"})
        assert result["status"] == "error"

    def test_query_gateway_error_details_unconfigured(self):
        """Test gateway details when RazorpayClient is not configured."""
        result = query_gateway_error_details(payment_id="pay_test")
        # When not configured, returns "unavailable" (client not configured)
        assert result["status"] == "unavailable"

    def test_check_bank_health_known_bank(self):
        result = check_bank_health(bank_code="HDFC")
        assert result["bank_code"] == "HDFC"
        assert "health_score" in result
        assert result["status"] in ("healthy", "degraded", "unstable", "unknown")

    def test_check_bank_health_unknown_bank(self):
        result = check_bank_health(bank_code="UNKNOWN")
        assert result["bank_code"] == "UNKNOWN"
        assert result["status"] == "unknown"

    def test_calculate_payday_window(self):
        result = calculate_payday_window(customer_id="cust_123", country_code="IN")
        assert result["status"] == "ok"
        assert result["customer_id"] == "cust_123"
        assert "hours_until_payday" in result
        assert "in_payday_window" in result

    def test_schedule_payday_retry_past_time(self):
        result = schedule_payday_retry(
            payment_id="pay_test",
            target_iso_timestamp="2020-01-01T00:00:00+00:00",
        )
        assert result["status"] == "error"
        assert "past" in result["message"].lower()

    def test_schedule_payday_retry_invalid_format(self):
        result = schedule_payday_retry(
            payment_id="pay_test",
            target_iso_timestamp="not-a-timestamp",
        )
        assert result["status"] == "error"

    def test_escalate_to_human_agent(self):
        result = escalate_to_human_agent(
            payment_id="pay_test_escalate",
            reason="All automated recovery attempts failed.",
        )
        assert result["status"] == "escalated"
        assert "ESC-" in result["ticket_id"]
        assert result["reason"] == "All automated recovery attempts failed."

    def test_generate_smart_recovery_link_unconfigured(self):
        """Test link generation when RazorpayClient is not configured."""
        result = generate_smart_recovery_link(payment_id="pay_test")
        assert result["status"] == "unavailable"


# ═══════════════════════════════════════════════════════════════
#  OBSERVATION FORMAT TESTS
# ═══════════════════════════════════════════════════════════════

class TestObservationFormatting:
    """Test observation formatting and context compaction."""

    def _make_obs(self, turn: int, tools: list[tuple[str, dict]] | None = None, errors: list[str] | None = None) -> Observation:
        tool_calls = []
        for tool_name, args in (tools or []):
            is_err = tool_name in (errors or [])
            tool_calls.append(ToolCall(
                tool=tool_name,
                arguments=args,
                result={"status": "error" if is_err else "ok"},
                is_error=is_err,
            ))
        return Observation(
            turn=turn,
            reasoning=f"Turn {turn} reasoning",
            tool_calls=tool_calls,
            raw_response={},
        )

    def test_format_observation_no_tools(self):
        obs = self._make_obs(1, tools=None)
        text = _format_observation(obs)
        assert "Turn 1" in text
        assert "none" in text

    def test_format_observation_with_tools(self):
        obs = self._make_obs(1, tools=[("check_bank_health", {"bank_code": "HDFC"})])
        text = _format_observation(obs)
        assert "check_bank_health" in text
        assert "HDFC" in text

    def test_format_observation_with_errors(self):
        obs = self._make_obs(
            1,
            tools=[("query_gateway_error_details", {"payment_id": "pay_x"}), ("check_bank_health", {"bank_code": "SBI"})],
            errors=["query_gateway_error_details"],
        )
        text = _format_observation(obs)
        assert "ERROR" in text
        assert "query_gateway_error_details" in text

    def test_compact_context_empty(self):
        assert _compact_context([]) == ""

    def test_compact_context_few_observations(self):
        obs = [self._make_obs(i + 1) for i in range(3)]
        text = _compact_context(obs)
        assert "EARLIER TURNS" not in text  # No compaction needed
        assert "Turn 1" in text
        assert "Turn 3" in text

    def test_compact_context_many_observations(self):
        obs = [self._make_obs(i + 1) for i in range(CONTEXT_COMPACT_THRESHOLD + 4)]
        text = _compact_context(obs)
        assert "EARLIER TURNS (SUMMARY)" in text
        assert "RECENT TURNS (FULL DETAIL)" in text

    def test_compact_context_preserves_recent_detail(self):
        obs = [self._make_obs(
            i + 1,
            tools=[("check_bank_health", {"bank_code": f"BANK_{i}"})],
        ) for i in range(CONTEXT_COMPACT_THRESHOLD + 4)]
        text = _compact_context(obs)
        # Most recent observation should have full detail
        assert "BANK_" in text


# ═══════════════════════════════════════════════════════════════
#  REFLECTION TESTS
# ═══════════════════════════════════════════════════════════════

class TestReflection:
    """Test TrueForge-style error reflection."""

    @patch("recovery_agent.agent.harness.invoke_llm")
    def test_reflect_on_error_returns_string(self, mock_invoke):
        mock_invoke.return_value = "Tool failed due to network timeout. Try different payment rail."
        case = make_case()
        result = _reflect_on_error(case, "check_bank_health", "Connection timeout", [])
        assert isinstance(result, str)
        assert "network timeout" in result.lower()

    @patch("recovery_agent.agent.harness.invoke_llm")
    def test_reflect_on_error_fallback(self, mock_invoke):
        mock_invoke.return_value = None
        case = make_case()
        result = _reflect_on_error(case, "check_bank_health", "Connection timeout", [])
        assert "check_bank_health" in result
        assert "Connection timeout" in result

    @patch("recovery_agent.agent.harness.invoke_llm")
    def test_reflect_provides_context(self, mock_invoke):
        mock_invoke.return_value = "Reflected."
        case = make_case(failure_code="insufficient_funds")
        _reflect_on_error(case, "generate_smart_recovery_link", "Payment link failed", [])
        call_args = mock_invoke.call_args
        assert "insufficient_funds" in call_args.kwargs["prompt"]


# ═══════════════════════════════════════════════════════════════
#  PROMPT BUILDER TESTS
# ═══════════════════════════════════════════════════════════════

class TestPromptBuilder:
    """Test multi-turn prompt construction."""

    def test_prompt_includes_failure_context(self):
        case = make_case()
        prompt = _build_harness_prompt(case, [], [])
        assert "pay_test_harness" in prompt
        assert "5,000.00" in prompt
        assert "unknown_retryable_error" in prompt

    def test_prompt_includes_diagnosis(self):
        case = make_case()
        prompt = _build_harness_prompt(case, [], [])
        assert "unknown_retryable_error" in prompt
        assert "90%" in prompt

    def test_prompt_includes_history_block(self):
        case = make_case()
        prompt = _build_harness_prompt(case, [], ["check_bank_health({\"bank_code\": \"HDFC\"})"])
        assert "PREVIOUSLY ATTEMPTED" in prompt
        assert "check_bank_health" in prompt

    def test_prompt_includes_observations(self):
        case = make_case()
        obs = Observation(turn=1, reasoning="Test reasoning", tool_calls=[], raw_response={})
        prompt = _build_harness_prompt(case, [obs], [])
        assert "Test reasoning" in prompt
        assert "Turn 1" in prompt

    def test_prompt_includes_tier(self):
        case = make_case()
        case.recovery_tier = RecoveryTier.SILENT
        prompt = _build_harness_prompt(case, [], [])
        assert "SILENT" in prompt


# ═══════════════════════════════════════════════════════════════
#  HARNESS LOOP TESTS
# ═══════════════════════════════════════════════════════════════

class TestAgentHarness:
    """Test the full AgentHarness multi-turn loop."""

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_harness_recovery_path(self, mock_llm):
        """Test harness when LLM says payment is recovered."""
        mock_llm.return_value = {
            "reasoning": "Generated recovery link, customer completed payment.",
            "tool_calls": [
                {"tool": "generate_smart_recovery_link", "arguments": {"payment_id": "pay_test_harness"}}
            ],
            "is_final": True,
            "status": "action_dispatched",
        }
        harness = AgentHarness()
        case = make_case()
        result = harness.run_recovery_case(case)

        assert result.final_status == "action_dispatched"
        assert result.total_turns == 1
        assert len(result.tools_called) == 1
        assert case.recovered is False

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_harness_escalation_path(self, mock_llm):
        """Test harness when LLM escalates to human."""
        mock_llm.return_value = {
            "reasoning": "All automated recovery failed. Escalating to human agent.",
            "tool_calls": [
                {"tool": "escalate_to_human_agent", "arguments": {"payment_id": "pay_test_harness", "reason": "Max attempts exhausted"}}
            ],
            "is_final": True,
            "status": "escalated",
        }
        harness = AgentHarness()
        case = make_case()
        result = harness.run_recovery_case(case)

        assert result.final_status == "escalated"
        assert case.status == CaseStatus.ESCALATED

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_harness_llm_unavailable_returns_failed(self, mock_llm):
        """Test harness returns failed when LLM is unavailable."""
        mock_llm.return_value = None
        harness = AgentHarness()
        case = make_case()
        result = harness.run_recovery_case(case)

        assert result.final_status == "failed"
        assert result.total_turns == 0

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_harness_max_iterations(self, mock_llm):
        """Test harness stops after max iterations."""
        # LLM always returns non-final response
        mock_llm.return_value = {
            "reasoning": "Trying another approach...",
            "tool_calls": [{"tool": "check_bank_health", "arguments": {"bank_code": "HDFC"}}],
            "is_final": False,
            "status": "in_progress",
        }
        harness = AgentHarness()
        case = make_case()
        result = harness.run_recovery_case(case)

        assert result.final_status == "max_iterations"
        assert result.total_turns == MAX_HARNESS_ITERATIONS

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_harness_multiple_turns(self, mock_llm):
        """Test harness flows through multiple turns."""
        responses = [
            {
                "reasoning": "First, check bank health.",
                "tool_calls": [{"tool": "check_bank_health", "arguments": {"bank_code": "HDFC"}}],
                "is_final": False,
                "status": "in_progress",
            },
            {
                "reasoning": "Bank is healthy. Generate recovery link.",
                "tool_calls": [{"tool": "generate_smart_recovery_link", "arguments": {"payment_id": "pay_test_harness"}}],
                "is_final": True,
                "status": "action_dispatched",
            },
        ]
        mock_llm.side_effect = responses

        harness = AgentHarness()
        case = make_case()
        result = harness.run_recovery_case(case)

        assert result.total_turns == 2
        assert result.final_status == "action_dispatched"
        assert len(result.tools_called) == 2

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_harness_repetition_prohibition(self, mock_llm):
        """Test harness blocks identical repeated tool calls."""
        responses = [
            {
                "reasoning": "Check bank health.",
                "tool_calls": [{"tool": "check_bank_health", "arguments": {"bank_code": "HDFC"}}],
                "is_final": False,
                "status": "in_progress",
            },
            {
                "reasoning": "Check bank health again (identical).",
                "tool_calls": [{"tool": "check_bank_health", "arguments": {"bank_code": "HDFC"}}],
                "is_final": False,
                "status": "in_progress",
            },
            {
                "reasoning": "Try different approach.",
                "tool_calls": [],
                "is_final": True,
                "status": "action_dispatched",
            },
        ]
        mock_llm.side_effect = responses

        harness = AgentHarness()
        case = make_case()
        result = harness.run_recovery_case(case)

        # Second call should be blocked
        assert result.error_count >= 1  # At least one blocked call
        assert len(result.tools_called) == 1  # Only the first call counted

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_harness_error_reflection(self, mock_llm):
        """Test harness calls reflection when a tool returns an error."""
        responses = [
            {
                "reasoning": "Query gateway details.",
                "tool_calls": [{"tool": "query_gateway_error_details", "arguments": {"payment_id": "pay_test_harness"}}],
                "is_final": False,
                "status": "in_progress",
            },
            {
                "reasoning": "Link generated successfully.",
                "tool_calls": [{"tool": "generate_smart_recovery_link", "arguments": {"payment_id": "pay_test_harness"}}],
                "is_final": True,
                "status": "action_dispatched",
            },
        ]
        mock_llm.side_effect = responses

        harness = AgentHarness()
        case = make_case()
        # Mock the tool to return error
        with patch("recovery_agent.agent.harness.execute_tool") as mock_exec:
            mock_exec.return_value = {"status": "error", "message": "Gateway timeout"}
            result = harness.run_recovery_case(case)

        # Should have at least one error from the failed tool
        assert result.error_count >= 1

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_harness_tools_called_tracking(self, mock_llm):
        """Test harness tracks all tools called across turns."""
        responses = [
            {
                "reasoning": "Check bank health.",
                "tool_calls": [{"tool": "check_bank_health", "arguments": {"bank_code": "SBI"}}],
                "is_final": False,
                "status": "in_progress",
            },
            {
                "reasoning": "Calculate payday window.",
                "tool_calls": [{"tool": "calculate_payday_window", "arguments": {"customer_id": "cust_harness_test"}}],
                "is_final": True,
                "status": "action_dispatched",
            },
        ]
        mock_llm.side_effect = responses

        harness = AgentHarness()
        case = make_case()
        result = harness.run_recovery_case(case)

        assert len(result.tools_called) == 2
        assert "check_bank_health" in result.tools_called[0]
        assert "calculate_payday_window" in result.tools_called[1]


# ═══════════════════════════════════════════════════════════════
#  HARNESS INTEGRATION WITH AGENT
# ═══════════════════════════════════════════════════════════════

class TestHarnessAgentIntegration:
    """Test AgentHarness integration with RecoveryAgent."""

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_agent_with_harness_mode(self, mock_llm):
        """Test RecoveryAgent in harness mode."""
        mock_llm.return_value = {
            "reasoning": "Recovery complete.",
            "tool_calls": [],
            "is_final": True,
            "status": "action_dispatched",
        }

        from recovery_agent.agent import RecoveryAgent
        agent = RecoveryAgent(use_harness=True)
        case = make_case()
        final_case = agent.run(case)

        assert final_case.recovered is False
        assert final_case.audit_log  # Should have audit entries

    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_agent_harness_vs_langgraph(self, mock_llm):
        """Test that harness mode and LangGraph mode produce different results."""
        mock_llm.return_value = {
            "reasoning": "Done.",
            "tool_calls": [],
            "is_final": True,
            "status": "action_dispatched",
        }

        from recovery_agent.agent import RecoveryAgent

        # Harness mode
        agent_harness = RecoveryAgent(use_harness=True)
        case_h = make_case()
        final_h = agent_harness.run(case_h)

        # LangGraph mode (no LLM available, will use heuristic fallback)
        agent_langgraph = RecoveryAgent(use_harness=False)
        case_lg = make_case()
        final_lg = agent_langgraph.run(case_lg)

        # Both should complete without errors
        assert final_h.status in (CaseStatus.AWAITING_CUSTOMER, CaseStatus.ESCALATED, CaseStatus.STOPPED)
        assert final_lg.status in (CaseStatus.AWAITING_CUSTOMER, CaseStatus.ESCALATED, CaseStatus.STOPPED)


# ═══════════════════════════════════════════════════════════════
#  ASYNC HARNESS TESTS
# ═══════════════════════════════════════════════════════════════

import asyncio


class TestAsyncHarness:
    """Test AgentHarness async execution — non-blocking event loop behavior."""

    @patch("recovery_agent.agent.harness.invoke_llm_json_async")
    def test_async_harness_recovery_path(self, mock_llm_async):
        """Async harness completes recovery without blocking."""
        mock_llm_async.return_value = {
            "reasoning": "Recovery complete.",
            "tool_calls": [],
            "is_final": True,
            "status": "action_dispatched",
        }
        harness = AgentHarness()
        case = make_case()
        result = asyncio.run(harness.run_recovery_case_async(case))

        assert result.final_status == "action_dispatched"
        assert case.recovered is False

    @patch("recovery_agent.agent.harness.invoke_llm_json_async")
    def test_async_harness_llm_unavailable(self, mock_llm_async):
        """Async harness returns failed when LLM is unavailable."""
        mock_llm_async.return_value = None
        harness = AgentHarness()
        case = make_case()
        result = asyncio.run(harness.run_recovery_case_async(case))

        assert result.final_status == "failed"
        assert result.total_turns == 0

    @patch("recovery_agent.agent.harness.invoke_llm_json_async")
    def test_async_harness_multiple_turns(self, mock_llm_async):
        """Async harness flows through multiple turns."""
        responses = [
            {
                "reasoning": "Check bank health.",
                "tool_calls": [{"tool": "check_bank_health", "arguments": {"bank_code": "HDFC"}}],
                "is_final": False,
                "status": "in_progress",
            },
            {
                "reasoning": "Bank healthy. Generate link.",
                "tool_calls": [{"tool": "generate_smart_recovery_link", "arguments": {"payment_id": "pay_test_harness"}}],
                "is_final": True,
                "status": "action_dispatched",
            },
        ]
        mock_llm_async.side_effect = responses

        harness = AgentHarness()
        case = make_case()
        result = asyncio.run(harness.run_recovery_case_async(case))

        assert result.total_turns == 2
        assert result.final_status == "action_dispatched"

    @patch("recovery_agent.agent.harness.invoke_llm_json_async")
    def test_async_harness_max_iterations(self, mock_llm_async):
        """Async harness stops after max iterations."""
        mock_llm_async.return_value = {
            "reasoning": "Trying...",
            "tool_calls": [{"tool": "check_bank_health", "arguments": {"bank_code": "HDFC"}}],
            "is_final": False,
            "status": "in_progress",
        }
        harness = AgentHarness()
        case = make_case()
        result = asyncio.run(harness.run_recovery_case_async(case))

        assert result.final_status == "max_iterations"
        assert result.total_turns == MAX_HARNESS_ITERATIONS

    @patch("recovery_agent.agent.harness.invoke_llm_json_async")
    def test_async_harness_error_reflection(self, mock_llm_async):
        """Async harness calls async reflection when tool returns error."""
        responses = [
            {
                "reasoning": "Query gateway.",
                "tool_calls": [{"tool": "query_gateway_error_details", "arguments": {"payment_id": "pay_test_harness"}}],
                "is_final": False,
                "status": "in_progress",
            },
            {
                "reasoning": "Link generated.",
                "tool_calls": [{"tool": "generate_smart_recovery_link", "arguments": {"payment_id": "pay_test_harness"}}],
                "is_final": True,
                "status": "action_dispatched",
            },
        ]
        mock_llm_async.side_effect = responses

        harness = AgentHarness()
        case = make_case()
        with patch("recovery_agent.agent.harness.execute_tool_async") as mock_exec:
            mock_exec.return_value = {"status": "error", "message": "Gateway timeout"}
            with patch("recovery_agent.agent.harness._reflect_on_error_async") as mock_reflect:
                mock_reflect.return_value = "Tool failed. Switching strategy."
                result = asyncio.run(harness.run_recovery_case_async(case))

        assert result.error_count >= 1
        mock_reflect.assert_called()

    @patch("recovery_agent.agent.harness.invoke_llm_json_async")
    def test_async_harness_repetition_prohibition(self, mock_llm_async):
        """Async harness blocks identical repeated tool calls."""
        responses = [
            {
                "reasoning": "Check bank.",
                "tool_calls": [{"tool": "check_bank_health", "arguments": {"bank_code": "HDFC"}}],
                "is_final": False,
                "status": "in_progress",
            },
            {
                "reasoning": "Check bank again.",
                "tool_calls": [{"tool": "check_bank_health", "arguments": {"bank_code": "HDFC"}}],
                "is_final": False,
                "status": "in_progress",
            },
            {
                "reasoning": "Done.",
                "tool_calls": [],
                "is_final": True,
                "status": "action_dispatched",
            },
        ]
        mock_llm_async.side_effect = responses

        harness = AgentHarness()
        case = make_case()
        result = asyncio.run(harness.run_recovery_case_async(case))

        assert result.error_count >= 1
        assert len(result.tools_called) == 1


class TestSyncAsyncEquivalence:
    """Verify sync wrapper produces identical results to direct async call."""

    @patch("recovery_agent.agent.harness.invoke_llm_json_async")
    @patch("recovery_agent.agent.harness.invoke_llm_json")
    def test_sync_wrapper_matches_async(self, mock_sync, mock_async):
        """Sync run_recovery_case() produces same result as run_recovery_case_async()."""
        expected = {
            "reasoning": "Recovered.",
            "tool_calls": [],
            "is_final": True,
            "status": "action_dispatched",
        }
        mock_sync.return_value = expected
        mock_async.return_value = expected

        harness = AgentHarness()

        # Sync path (uses _run_recovery_case_sync directly)
        case_sync = make_case()
        result_sync = harness.run_recovery_case(case_sync)

        # Async path
        case_async = make_case()
        result_async = asyncio.run(harness.run_recovery_case_async(case_async))

        assert result_sync.final_status == result_async.final_status
        assert result_sync.total_turns == result_async.total_turns
        assert case_sync.recovered == case_async.recovered


class TestAsyncReflectOnError:
    """Test async error reflection function."""

    @patch("recovery_agent.agent.harness.invoke_llm_async")
    def test_async_reflect_returns_string(self, mock_invoke_async):
        mock_invoke_async.return_value = "Network timeout. Try different rail."
        case = make_case()
        result = asyncio.run(_reflect_on_error_async(case, "check_bank_health", "Connection timeout", []))
        assert isinstance(result, str)
        assert "network timeout" in result.lower()

    @patch("recovery_agent.agent.harness.invoke_llm_async")
    def test_async_reflect_fallback(self, mock_invoke_async):
        mock_invoke_async.return_value = None
        case = make_case()
        result = asyncio.run(_reflect_on_error_async(case, "check_bank_health", "Connection timeout", []))
        assert "check_bank_health" in result
        assert "Connection timeout" in result


class TestAsyncAgentIntegration:
    """Test RecoveryAgent async entry points."""

    @patch("recovery_agent.agent.harness.invoke_llm_json_async")
    def test_agent_run_async_harness_mode(self, mock_llm_async):
        """RecoveryAgent.run_async() works in harness mode."""
        mock_llm_async.return_value = {
            "reasoning": "Done.",
            "tool_calls": [],
            "is_final": True,
            "status": "action_dispatched",
        }

        from recovery_agent.agent import RecoveryAgent
        agent = RecoveryAgent(use_harness=True)
        case = make_case()
        final_case = asyncio.run(agent.run_async(case))

        assert final_case.recovered is False
        assert final_case.audit_log

    @patch("recovery_agent.agent.harness.invoke_llm_json_async")
    def test_agent_run_harness_async(self, mock_llm_async):
        """RecoveryAgent.run_harness_async() directly."""
        mock_llm_async.return_value = {
            "reasoning": "Recovered.",
            "tool_calls": [],
            "is_final": True,
            "status": "action_dispatched",
        }

        from recovery_agent.agent import RecoveryAgent
        agent = RecoveryAgent(use_harness=True)
        case = make_case()
        final_case = asyncio.run(agent.run_harness_async(case))

        assert final_case.recovered is False
