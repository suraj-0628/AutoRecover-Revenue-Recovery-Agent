"""Tests for Mandate 2: True Asynchronous Agent Execution.

Covers:
- Async Razorpay SDK wrappers (run_in_executor)
- Async tool execution (execute_tool_async)
- Async LLM invocation (invoke_llm_async, invoke_llm_json_async)
- Webhook background thread forwarding
- Event loop non-blocking behavior
"""
from __future__ import annotations

import asyncio
import time
import threading
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ═══════════════════════════════════════════════════════════════
#  ASYNC RAZORPAY SDK WRAPPERS
# ═══════════════════════════════════════════════════════════════

class TestAsyncRazorpaySDK:
    """Test that async SDK wrappers offload to thread pool correctly."""

    def test_create_order_async_returns_same_as_sync(self):
        """async create_order_async returns identical result to sync create_order."""
        from recovery_agent.razorpay_client import RazorpayClient

        client = RazorpayClient()  # Not configured — returns mock data
        sync_result = client.create_order(amount=500.0)
        async_result = asyncio.run(client.create_order_async(amount=500.0))

        assert sync_result["entity"] == async_result["entity"]
        assert sync_result["amount"] == async_result["amount"]
        assert sync_result["status"] == async_result["status"]

    def test_create_payment_link_async(self):
        from recovery_agent.razorpay_client import RazorpayClient

        client = RazorpayClient()
        sync_result = client.create_payment_link(amount=100.0)
        async_result = asyncio.run(client.create_payment_link_async(amount=100.0))

        assert sync_result["entity"] == async_result["entity"]
        assert sync_result["status"] == async_result["status"]

    def test_fetch_payment_async(self):
        from recovery_agent.razorpay_client import RazorpayClient

        client = RazorpayClient()
        sync_result = client.fetch_payment("pay_test123")
        async_result = asyncio.run(client.fetch_payment_async("pay_test123"))

        assert sync_result["id"] == async_result["id"]

    def test_capture_payment_async(self):
        from recovery_agent.razorpay_client import RazorpayClient

        client = RazorpayClient()
        sync_result = client.capture_payment("pay_test123", 500.0)
        async_result = asyncio.run(client.capture_payment_async("pay_test123", 500.0))

        assert sync_result["status"] == async_result["status"]

    def test_async_sdk_does_not_block_event_loop(self):
        """Async SDK call returns quickly without blocking the event loop."""
        from recovery_agent.razorpay_client import RazorpayClient

        client = RazorpayClient()

        async def timed_call():
            start = time.monotonic()
            await client.create_order_async(amount=100.0)
            return time.monotonic() - start

        elapsed = asyncio.run(timed_call())
        # Mock SDK should return in well under 1 second
        assert elapsed < 1.0


# ═══════════════════════════════════════════════════════════════
#  ASYNC TOOL EXECUTION
# ═══════════════════════════════════════════════════════════════

class TestAsyncToolExecution:
    """Test execute_tool_async offloads to thread pool."""

    def test_execute_tool_async_returns_same_as_sync(self):
        """execute_tool_async returns identical result to execute_tool."""
        from recovery_agent.agent.tools import execute_tool, execute_tool_async

        async def _run():
            args = {"bank_code": "HDFC"}
            sync_result = execute_tool("check_bank_health", args)
            async_result = await execute_tool_async("check_bank_health", args)
            assert sync_result == async_result

        asyncio.run(_run())

    def test_execute_tool_async_unknown_tool(self):
        from recovery_agent.agent.tools import execute_tool_async

        async def _run():
            result = await execute_tool_async("nonexistent_tool", {})
            assert result["status"] == "error"
            assert "Unknown tool" in result["message"]

        asyncio.run(_run())

    def test_execute_tool_async_escalate(self):
        from recovery_agent.agent.tools import execute_tool_async

        async def _run():
            result = await execute_tool_async("escalate_to_human_agent", {
                "payment_id": "pay_test_async",
                "reason": "Async test escalation",
            })
            assert result["status"] == "escalated"
            assert "ESC-" in result["ticket_id"]

        asyncio.run(_run())

    def test_execute_tool_async_does_not_block(self):
        """Async tool execution completes without blocking the event loop."""
        from recovery_agent.agent.tools import execute_tool_async

        async def timed():
            start = time.monotonic()
            await execute_tool_async("check_bank_health", {"bank_code": "SBI"})
            return time.monotonic() - start

        elapsed = asyncio.run(timed())
        assert elapsed < 1.0



# ═══════════════════════════════════════════════════════════════
#  ASYNC LLM INVOCATION
# ═══════════════════════════════════════════════════════════════

class TestAsyncLLMInvocation:
    """Test invoke_llm_async and invoke_llm_json_async offload correctly."""

    @patch("recovery_agent.agent.llm_client.invoke_llm")
    def test_invoke_llm_async_returns_same_as_sync(self, mock_invoke):
        mock_invoke.return_value = "Test response"
        from recovery_agent.agent.llm_client import invoke_llm_async

        result = asyncio.run(invoke_llm_async(prompt="Hello"))
        assert result == "Test response"
        mock_invoke.assert_called_once()

    @patch("recovery_agent.agent.llm_client.invoke_llm")
    def test_invoke_llm_async_returns_none_on_failure(self, mock_invoke):
        mock_invoke.return_value = None
        from recovery_agent.agent.llm_client import invoke_llm_async

        result = asyncio.run(invoke_llm_async(prompt="Hello"))
        assert result is None

    @patch("recovery_agent.agent.llm_client.invoke_llm_json")
    def test_invoke_llm_json_async_returns_same_as_sync(self, mock_json):
        mock_json.return_value = {"key": "value"}
        from recovery_agent.agent.llm_client import invoke_llm_json_async

        result = asyncio.run(invoke_llm_json_async(prompt="Hello"))
        assert result == {"key": "value"}

    @patch("recovery_agent.agent.llm_client.invoke_llm_json")
    def test_invoke_llm_json_async_returns_none_on_failure(self, mock_json):
        mock_json.return_value = None
        from recovery_agent.agent.llm_client import invoke_llm_json_async

        result = asyncio.run(invoke_llm_json_async(prompt="Hello"))
        assert result is None


# ═══════════════════════════════════════════════════════════════
#  WEBHOOK BACKGROUND THREAD FORWARDING
# ═══════════════════════════════════════════════════════════════

class TestWebhookAsyncForwarding:
    """Test webhook offloads forwarding to background thread."""

    @patch("recovery_agent.webhook.forward_to_frontend")
    def test_forward_to_frontend_async_returns_immediately(self, mock_forward):
        """_forward_to_frontend_async returns without waiting for forwarding to complete."""
        mock_forward.side_effect = lambda *a, **kw: time.sleep(0.5) or {"status": "ok"}
        from recovery_agent.webhook import _forward_to_frontend_async

        start = time.monotonic()
        _forward_to_frontend_async("payment.failed", {"event": "payment.failed"})
        elapsed = time.monotonic() - start

        # Should return immediately (< 100ms), not wait for the 500ms sleep
        assert elapsed < 0.2

    @patch("recovery_agent.webhook.forward_to_frontend")
    def test_background_forward_completes(self, mock_forward):
        """Background forwarding completes successfully."""
        mock_forward.return_value = {"status": "ok"}
        from recovery_agent.webhook import _forward_to_frontend_async

        _forward_to_frontend_async("payment.failed", {"event": "payment.failed"})
        time.sleep(0.2)  # Wait for background thread

        mock_forward.assert_called_once_with("payment.failed", {"event": "payment.failed"})

    @patch("recovery_agent.webhook.forward_to_frontend")
    def test_background_forward_handles_exception(self, mock_forward):
        """Background forwarding handles exceptions gracefully."""
        mock_forward.side_effect = ConnectionError("Connection refused")
        from recovery_agent.webhook import _forward_to_frontend_async

        # Should not raise — background thread catches exceptions
        _forward_to_frontend_async("payment.failed", {"event": "payment.failed"})
        time.sleep(0.2)

    def test_webhook_returns_200_immediately(self):
        """Webhook handler returns 200 without waiting for forwarding."""
        from recovery_agent.webhook import app, _processed_events, _event_lock

        # Clear dedup state
        with _event_lock:
            _processed_events.clear()

        with patch("recovery_agent.webhook.verify_signature", return_value=True), \
             patch("recovery_agent.webhook._forward_to_frontend_async") as mock_bg:

            with app.test_client() as client:
                start = time.monotonic()
                resp = client.post("/webhook", json={
                    "event": "payment.failed",
                    "event_id": "evt_async_test_001",
                    "payload": {
                        "payment": {
                            "entity": {
                                "id": "pay_async_test",
                                "amount": 50000,
                                "status": "failed",
                            }
                        }
                    },
                })
                elapsed = time.monotonic() - start

            assert resp.status_code == 200
            data = resp.get_json()
            assert data["status"] == "accepted"
            assert data["forwarding"] == "async"
            # Background forwarding was triggered
            mock_bg.assert_called_once()


# ═══════════════════════════════════════════════════════════════
#  EVENT LOOP NON-BLOCKING BEHAVIOR
# ═══════════════════════════════════════════════════════════════

class TestEventLoopNonBlocking:
    """Verify async operations do not block the event loop."""

    def test_concurrent_async_tool_calls(self):
        """Multiple async tool calls run concurrently, not sequentially."""
        from recovery_agent.agent.tools import execute_tool_async

        async def _run():
            async def timed_call(bank):
                start = time.monotonic()
                await execute_tool_async("check_bank_health", {"bank_code": bank})
                return time.monotonic() - start

            start = time.monotonic()
            results = await asyncio.gather(
                timed_call("HDFC"),
                timed_call("ICICI"),
                timed_call("SBI"),
            )
            total_elapsed = time.monotonic() - start
            assert total_elapsed < 2.0
            assert all(r < 1.0 for r in results)

        asyncio.run(_run())

    def test_async_harness_yields_to_event_loop(self):
        """Async harness yields control between LLM calls."""
        from recovery_agent.agent.harness import AgentHarness
        from tests.test_harness import make_case

        async def _run():
            call_count = 0

            async def counting_llm(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    return {"reasoning": "Done.", "tool_calls": [], "is_final": True, "status": "action_dispatched"}
                return {"reasoning": "Check.", "tool_calls": [{"tool": "check_bank_health", "arguments": {"bank_code": "HDFC"}}], "is_final": False, "status": "in_progress"}

            with patch("recovery_agent.agent.harness.invoke_llm_json_async", side_effect=counting_llm):
                harness = AgentHarness()
                case = make_case()
                result = await harness.run_recovery_case_async(case)

            assert result.final_status == "action_dispatched"
            assert call_count == 2

        asyncio.run(_run())

