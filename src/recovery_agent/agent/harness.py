"""AgentHarness — TrueForge-style multi-turn reasoning loop for revenue recovery.

Implements:
  Context Engineering → LLM Tool Plan → Guardrail Interception →
  Tool Execution → Observation Feed → Re-plan / Stop

Key features:
  - Full session trajectory history across multiple turns
  - Automatic context compaction (summarize older audit entries)
  - TrueForge-style reflection: error observations feed back into context
  - Prohibition on repeating identical tool calls
  - Structured tool-calling via MCP-style tool registry

Source: TrueForge agent harness pattern
        Context Engineering from Agentic AI (Andrew Ng)
        Tool Use pattern from Module 3
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from recovery_agent.agent.guardrails import GuardrailEngine
from recovery_agent.agent.llm_client import invoke_llm, invoke_llm_json, invoke_llm_json_async, invoke_llm_async
from recovery_agent.agent.memory import CustomerMemoryStore
from recovery_agent.agent.strategy_metrics import StrategyMetricsStore
from recovery_agent.agent.tools import (
    TOOL_SCAPES,
    execute_tool,
    execute_tool_async,
    get_tool_schemas_for_llm,
)
from recovery_agent.agent.vector_memory import VectorMemoryStore
from recovery_agent.models import (
    ActionType,
    AuditStep,
    Case,
    CaseStatus,
    RecoveryTier,
)


# --- Constants ---

# Map harness tool names to ActionTypes for guardrail interception.
# Only ACTION tools (tools that trigger customer-facing or state-changing behavior)
# are intercepted. Diagnostic tools (query_*, check_*) pass through unchecked.
_TOOL_ACTION_MAP: dict[str, ActionType] = {
    "generate_smart_recovery_link": ActionType.UPDATE_PAYMENT_METHOD,
    "schedule_payday_retry": ActionType.WAIT_AND_RETRY,
    "escalate_to_human_agent": ActionType.ESCALATE_TO_HUMAN,
    "initiate_voice_call": ActionType.VOICE_CALL,
}

MAX_HARNESS_ITERATIONS = 8
CONTEXT_COMPACT_THRESHOLD = 6  # Summarize older entries after this many turns
HARNESS_SYSTEM_PROMPT = """You are a Razorpay revenue recovery agent operating in a multi-turn reasoning loop.

You have access to tools that can query gateway errors, check bank health, calculate payday windows,
generate recovery links, schedule retries, and escalate to humans.

YOUR GOAL: Recover the failed payment by choosing the right sequence of tool calls.

═══ REASONING RULES ═══

1. ANALYZE the failure context before choosing any tool.
2. NEVER repeat the same tool call with identical parameters — if it failed, try a different approach.
3. When a tool returns an error, REFLECT on why it failed and adjust your strategy.
4. Prefer non-invasive tools first (query, check) before action tools (generate link, schedule retry).
5. If a tool returns "Payment could not be completed, use another payment instrument", you MUST
   switch to a different recovery strategy (e.g., generate_smart_recovery_link with different rails).
6. STOP when: payment recovered, max attempts reached, or human escalation is warranted.

═══ ANTI-HALLUCINATION RULE ═══

NEVER invent payment method names, bank names, or providers that are NOT in the PAYMENT FAILURE CONTEXT.
- If method is "paylater" → state "LazyPay PayLater", NOT "HDFC Netbanking" or "card".
- If method is "upi" → state the UPI VPA or provider shown, NOT a card network.
- If method is "card" → use the card_network and bank shown in context.
- If method is "netbanking" → use the bank shown in context.
- If provider is "lazypay" → always reference "LazyPay" in your reasoning.
- Only reference entities that appear in the PAYMENT FAILURE CONTEXT below.

═══ INSTRUMENT SWITCH RULE ═══

If the failure message contains "use another payment instrument", "use another payment method",
"try another method", "expired", or "invalid card" — the current payment method is BROKEN.
Retrying the same method WILL fail again. You MUST:
  - Use generate_smart_recovery_link with DIFFERENT rails (UPI, Netbanking, Wallet)
  - NEVER use schedule_payday_retry on the same broken instrument
  - The customer MUST switch to a working payment method

═══ TOOL SELECTION GUIDE ═══

- query_payment_recovery_kb: **PRIMARY DIAGNOSTIC TOOL** — Query Razorpay error docs, RBI mandates, PSP guides, and merchant policies. ALWAYS call this FIRST for any failure to get grounded knowledge base context. Decomposes queries and evaluates groundedness.
- query_gateway_error_details: When you need more error context beyond the initial failure code.
- check_bank_health: When failure might be bank-side (network_timeout, bank_declined).
- calculate_payday_window: When failure is insufficient_funds and you need timing info.
- generate_smart_recovery_link: When customer needs a fresh payment link to retry.
- schedule_payday_retry: When timing the retry to payday would help (insufficient_funds).
- escalate_to_human_agent: When automated recovery has failed or case is complex.

CRITICAL INSTRUCTION: If the STRATEGY PLANNER DECISION is 'send_notification' or 'update_payment_method', the system will automatically handle the dispatch after you finish. You DO NOT need to call a tool for these. Simply emit is_final=true and status=action_dispatched on Turn 1 to hand control back to the system execution layer. NEVER emit status=recovered — recovery is only confirmed by a real Razorpay webhook or customer response, never by the agent.

═══ OUTPUT FORMAT ═══

Respond with a JSON object containing:
{
  "reasoning": "Brief explanation of your thought process",
  "tool_calls": [
    {"tool": "tool_name", "arguments": {"param": "value"}}
  ],
  "is_final": false,
  "status": "in_progress"
}

When recovery is achieved or you should stop:
{
  "reasoning": "Why recovery is complete or should stop",
  "tool_calls": [],
  "is_final": true,
  "status": "recovered" | "escalated" | "failed" | "max_iterations"
}
"""


# --- Data Classes ---

@dataclass
class ToolCall:
    """Record of a single tool invocation."""
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%H:%M:%S"))
    is_error: bool = False


@dataclass
class Observation:
    """A single observation entry in the harness trajectory."""
    turn: int
    reasoning: str
    tool_calls: list[ToolCall]
    raw_response: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%H:%M:%S"))


@dataclass
class HarnessResult:
    """Final result of a harness run."""
    case: Case
    observations: list[Observation]
    final_status: str  # "recovered", "escalated", "failed", "max_iterations"
    total_turns: int
    tools_called: list[str]
    error_count: int


# --- Context Engineering ---

def _compact_context(observations: list[Observation], max_entries: int = CONTEXT_COMPACT_THRESHOLD) -> str:
    """Summarize older observations to keep the prompt context tight.

    Recent observations are kept in full detail.
    Older observations are summarized into a single context block.
    """
    if not observations:
        return ""

    if len(observations) <= max_entries:
        # All observations fit — return full detail
        return "\n".join(_format_observation(o) for o in observations)

    # Split: older (summarized) + recent (full detail)
    older = observations[:len(observations) - max_entries + 2]
    recent = observations[len(observations) - max_entries + 2:]

    summary_parts = []
    for o in older:
        tools_str = ", ".join(f"{tc.tool}({json.dumps(tc.arguments)})" for tc in o.tool_calls) if o.tool_calls else "none"
        error_tools = [tc.tool for tc in o.tool_calls if tc.is_error]
        summary_parts.append(
            f"Turn {o.turn}: {tools_str}"
            + (f" [ERRORS: {', '.join(error_tools)}]" if error_tools else " [OK]")
        )

    compact = (
        "═══ EARLIER TURNS (SUMMARY) ═══\n"
        + "\n".join(summary_parts)
        + "\n\n═══ RECENT TURNS (FULL DETAIL) ═══\n"
        + "\n".join(_format_observation(o) for o in recent)
    )
    return compact


def _format_observation(obs: Observation) -> str:
    """Format a single observation for the LLM prompt."""
    tools_str = ""
    if obs.tool_calls:
        calls = []
        for tc in obs.tool_calls:
            result_status = tc.result.get("status", "unknown")
            calls.append(f"  - {tc.tool}({json.dumps(tc.arguments)}) → {result_status}")
            if tc.is_error:
                msg = tc.result.get("message", "unknown error")
                calls.append(f"    ERROR: {msg}")
        tools_str = "\n".join(calls)

    return (
        f"Turn {obs.turn} [{obs.timestamp}]:\n"
        f"Reasoning: {obs.reasoning}\n"
        f"Tools called:\n{tools_str if tools_str else '  none'}"
    )


def _build_harness_prompt(case: Case, observations: list[Observation], history: list[str], similar_context: str = "") -> str:
    """Build the multi-turn reasoning prompt with compacted context and similar cases."""
    # Build failure context
    failure_ctx = (
        f"PAYMENT FAILURE CONTEXT:\n"
        f"  Payment ID: {case.payment.payment_id}\n"
        f"  Amount: {case.payment.currency} {case.payment.amount:,.2f}\n"
        f"  Failure Code: {case.payment.failure_code}\n"
        f"  Failure Reason: {case.payment.failure_reason}\n"
        f"  Payment Method: {case.payment.metadata.get('method', 'unknown')}\n"
        f"  Provider: {case.payment.metadata.get('provider', 'unknown')}\n"
        f"  Error Source: {case.payment.metadata.get('error_source', 'unknown')}\n"
        f"  Error Step: {case.payment.metadata.get('error_step', 'unknown')}\n"
        f"  Error Description: {case.payment.metadata.get('error_description', case.payment.failure_reason)}\n"
    )

    # Add diagnosis context if available
    if case.diagnosis:
        failure_ctx += (
            f"\nDIAGNOSIS:\n"
            f"  Root Cause: {case.diagnosis.root_cause.value}\n"
            f"  Confidence: {case.diagnosis.confidence:.0%}\n"
            f"  Reasoning: {case.diagnosis.reasoning}\n"
        )

    # Add recovery tier
    failure_ctx += (
        f"\nRECOVERY TIER: {case.recovery_tier.value.upper()}\n"
        f"ATTEMPT: {case.attempt_count + 1}/{case.max_attempts}\n"
        f"PENALTIES PREVENTED: {case.penalties_prevented}\n"
    )

    # Add strategy planner decision
    decided_action = case.payment.metadata.get("decided_action")
    if decided_action:
        failure_ctx += (
            f"\nSTRATEGY PLANNER DECISION:\n"
            f"  The guardrail-approved strategy is: {decided_action}\n"
            f"  Your goal is to use your tools to support this strategy.\n"
            f"  If no further tools are needed to execute this strategy, output is_final=true and status=recovered immediately.\n"
        )

    # Add previously blocked tools (to prevent repetition)
    if history:
        failure_ctx += f"\nPREVIOUSLY ATTEMPTED (DO NOT REPEAT IDENTICAL CALLS):\n"
        for h in history:
            failure_ctx += f"  - {h}\n"

    # Add compacted observations
    obs_text = _compact_context(observations) if observations else "No previous observations — this is the first turn."

    return f"""{failure_ctx}

═══ SIMILAR PAST CASES ═══
{similar_context if similar_context else "No similar cases found in memory."}

═══ OBSERVATION HISTORY ═══
{obs_text}

═══ YOUR TASK ═══
Choose the next tool call(s) based on the failure context, observation history, and similar past cases.
If a previous tool call returned an error, REFLECT on why and try a DIFFERENT approach.
Do NOT repeat the same tool call with identical parameters.

Respond with a JSON object:"""


# --- TrueForge Reflection ---

def _reflect_on_error(
    case: Case,
    failed_tool: str,
    error_message: str,
    observations: list[Observation],
) -> str:
    """Prompt the LLM to reflect on why a tool call failed.

    Returns the reflection reasoning string.
    """
    prompt = (
        f"A tool call just failed during recovery of payment {case.payment.payment_id}.\n\n"
        f"Failed tool: {failed_tool}\n"
        f"Error message: {error_message}\n"
        f"Payment failure code: {case.payment.failure_code}\n"
        f"Diagnosis: {case.diagnosis.root_cause.value if case.diagnosis else 'unknown'}\n\n"
        f"Why might this tool have failed? What should we try instead?\n"
        f"Respond with a brief reflection (1-2 sentences)."
    )

    reflection = invoke_llm(
        prompt=prompt,
        system="You are a payment recovery strategist. Reflect briefly on tool failures.",
        temperature=0.3,
        max_tokens=200,
    )

    return reflection or f"Tool {failed_tool} returned error: {error_message}. Switching strategy."


async def _reflect_on_error_async(
    case: Case,
    failed_tool: str,
    error_message: str,
    observations: list[Observation],
) -> str:
    """Non-blocking LLM reflection on tool failure via thread pool."""
    prompt = (
        f"A tool call just failed during recovery of payment {case.payment.payment_id}.\n\n"
        f"Failed tool: {failed_tool}\n"
        f"Error message: {error_message}\n"
        f"Payment failure code: {case.payment.failure_code}\n"
        f"Diagnosis: {case.diagnosis.root_cause.value if case.diagnosis else 'unknown'}\n\n"
        f"Why might this tool have failed? What should we try instead?\n"
        f"Respond with a brief reflection (1-2 sentences)."
    )

    reflection = await invoke_llm_async(
        prompt=prompt,
        system="You are a payment recovery strategist. Reflect briefly on tool failures.",
        temperature=0.3,
        max_tokens=200,
    )

    return reflection or f"Tool {failed_tool} returned error: {error_message}. Switching strategy."


# --- Main Harness Loop ---

class AgentHarness:
    """TrueForge-style agent harness for multi-turn recovery reasoning.

    Loop: Context Engineering → LLM Tool Plan → Guardrail Interception →
          Tool Execution → Observation Feed → Re-plan / Stop

    Maintains full session trajectory history across multiple turns.
    Implements automatic context compaction and error reflection.
    """

    def __init__(
        self,
        memory_store: CustomerMemoryStore | None = None,
        guardrail_engine: GuardrailEngine | None = None,
        vector_memory: VectorMemoryStore | None = None,
        strategy_metrics: StrategyMetricsStore | None = None,
        bandit: "ThompsonBandit | None" = None,
    ):
        self.memory = memory_store or CustomerMemoryStore()
        self.guardrails = guardrail_engine or GuardrailEngine()
        self.vector_memory = vector_memory or VectorMemoryStore()
        self.strategy_metrics = strategy_metrics or StrategyMetricsStore()
        self.bandit = bandit

    def _guardrail_intercept(
        self, case: Case, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Check if a tool call is blocked or modified by the guardrail engine.

        Returns exec_result dict if the guardrail BLOCKS the action (caller should
        skip normal execution and use this result directly).
        Returns None if the action is APPROVED (caller should proceed with execute_tool).

        Maps tool names to ActionTypes:
          generate_smart_recovery_link → UPDATE_PAYMENT_METHOD
          schedule_payday_retry        → WAIT_AND_RETRY
          escalate_to_human_agent      → ESCALATE_TO_HUMAN
        """
        mapped_action = _TOOL_ACTION_MAP.get(tool_name)
        if mapped_action is None:
            return None  # Not an action tool — no interception

        profile = self.memory.get_or_create_profile(case.payment.customer_id)
        approved_action, checks = self.guardrails.validate_action(case, mapped_action, profile)

        if approved_action == mapped_action:
            return None  # Approved — proceed with normal execution

        # Blocked or modified — build error message from check results
        blocked_reasons = []
        for check in checks:
            if check.verdict.value in ("blocked", "modified"):
                blocked_reasons.append(f"{check.guardrail}: {check.reason}")

        if not blocked_reasons:
            blocked_reasons.append(
                f"Guardrail redirected {mapped_action.value} to {approved_action.value}"
            )

        error_message = (
            f"Guardrail blocked action {mapped_action.value}. "
            f"Reasons: {'; '.join(blocked_reasons)}. "
            f"Please pivot strategy to {approved_action.value}."
        )

        return {"status": "blocked", "message": error_message}

    def run_recovery_case(self, case: Case) -> HarnessResult:
        """Run the full multi-turn harness loop on a case (sync, backward-compatible).

        When called from an async context (event loop already running),
        delegates to run_recovery_case_async via a thread pool to avoid
        blocking the event loop. When called synchronously, runs the
        original blocking loop directly.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.run_recovery_case_async(case))
                return future.result()
        else:
            return self._run_recovery_case_sync(case)

    def _run_recovery_case_sync(self, case: Case) -> HarnessResult:
        """Original synchronous harness loop — kept for backward compatibility.

        Direct callers (tests, CLI) use this path. Async callers use
        run_recovery_case_async which offloads LLM/SDK to thread pool.
        """
        from recovery_agent.agent.stopping import check_stopping_rules, transition_to_active_tier

        try:
            from opentelemetry import trace
            tracer = trace.get_tracer("recovery-agent-harness")
        except Exception:
            tracer = None

        observations: list[Observation] = []
        tools_called_history: list[str] = []
        error_count = 0

        attrs = {
            "payment_id": case.payment.payment_id,
            "amount": case.payment.amount,
            "failure_code": case.payment.failure_code or "",
            "failure_reason": case.payment.failure_reason or "",
            "customer_id": case.payment.customer_id or "",
            "max_attempts": case.max_attempts,
            "recovery_tier": case.recovery_tier.value if case.recovery_tier else "",
        }
        ctx = tracer.start_span("harness.run_recovery_case", attributes=attrs) if tracer else None
        try:
            for turn in range(MAX_HARNESS_ITERATIONS):
                turn_ctx = tracer.start_span(
                    f"harness.turn_{turn}",
                    attributes={"turn": turn, "payment_id": case.payment.payment_id},
                ) if tracer else None
                try:
                    # Check stopping rules before each turn
                    should_stop, stop_reason = check_stopping_rules(case)
                    if should_stop:
                        case.status = CaseStatus.STOPPED
                        self._ingest_outcome(case)
                        if turn_ctx:
                            turn_ctx.set_attribute("stop_reason", stop_reason)
                            turn_ctx.set_attribute("final_status", "stopped")
                        return HarnessResult(
                            case=case, observations=observations,
                            final_status="stopped", total_turns=turn,
                            tools_called=tools_called_history, error_count=error_count,
                        )
                    if stop_reason in ("SILENT_RETRY_FAILED", "SILENT_TIER_EXHAUSTED"):
                        case = transition_to_active_tier(case, stop_reason)
                        observations.append(Observation(
                            turn=turn + 1,
                            reasoning=f"Tier transition: {stop_reason}",
                            tool_calls=[], raw_response={"tier_transition": stop_reason},
                        ))

                    similar_ctx = self.vector_memory.get_decision_context(case)
                    prompt = _build_harness_prompt(case, observations, tools_called_history, similar_ctx)

                    result = invoke_llm_json(
                        prompt=prompt,
                        system=HARNESS_SYSTEM_PROMPT,
                        temperature=0,
                        max_tokens=1024,
                    )

                    if result is None:
                        self._ingest_outcome(case)
                        if turn_ctx:
                            turn_ctx.set_attribute("final_status", "llm_failed")
                        return HarnessResult(
                            case=case,
                            observations=observations,
                            final_status="failed",
                            total_turns=turn,
                            tools_called=tools_called_history,
                            error_count=error_count,
                        )

                    reasoning = result.get("reasoning", "")
                    tool_calls_raw = result.get("tool_calls", [])
                    is_final = result.get("is_final", False)
                    final_status = result.get("status", "in_progress")

                    if turn_ctx:
                        turn_ctx.set_attribute("reasoning", reasoning[:500])
                        turn_ctx.set_attribute("tool_calls_count", len(tool_calls_raw))
                        turn_ctx.set_attribute("is_final", is_final)

                    tool_calls: list[ToolCall] = []

                    for tc in tool_calls_raw:
                        call_signature, arguments, is_duplicate = self._prepare_tool_arguments(case, tc, tools_called_history)
                        tool_name = tc.get("tool", "")

                        if is_duplicate:
                            tool_calls.append(ToolCall(
                                tool=tool_name,
                                arguments=arguments,
                                result={"status": "blocked", "message": "Duplicate call prohibited — already attempted"},
                                is_error=True,
                            ))
                            error_count += 1
                            continue

                        tool_span = tracer.start_span(
                            f"tool.{tool_name}",
                            attributes={"tool.name": tool_name, "payment_id": case.payment.payment_id},
                        ) if tracer else None
                        try:
                            block_reason = self._check_tool_allowed(case, tool_name, arguments)
                            if block_reason:
                                exec_result = {"status": "blocked", "message": block_reason}
                                is_error = True
                            else:
                                exec_result = execute_tool(tool_name, arguments)
                                is_error = exec_result.get("status") in ("error", "unavailable")

                            if tool_span:
                                tool_span.set_attribute("tool.status", exec_result.get("status", "unknown"))
                                tool_span.set_attribute("tool.is_error", is_error)
                                if is_error:
                                    tool_span.set_attribute("tool.error", exec_result.get("message", "")[:200])
                        finally:
                            if tool_span:
                                tool_span.end()

                        tc_record = ToolCall(
                            tool=tool_name,
                            arguments=arguments,
                            result=exec_result,
                            is_error=is_error,
                        )
                        tool_calls.append(tc_record)
                        tools_called_history.append(call_signature)

                        if is_error:
                            error_count += 1
                            error_msg = exec_result.get("message", "unknown error")
                            reflection = _reflect_on_error(case, tool_name, error_msg, observations)
                            reasoning += f" [Reflection: {reflection}]"

                        self._apply_tool_result(case, tool_name, exec_result)

                    obs = Observation(
                        turn=turn + 1,
                        reasoning=reasoning,
                        tool_calls=tool_calls,
                        raw_response=result,
                    )
                    observations.append(obs)

                    if is_final:
                        status_map = {
                            "action_dispatched": "action_dispatched",
                            "recovered": "action_dispatched",
                            "escalated": "escalated",
                            "failed": "failed",
                            "max_iterations": "max_iterations",
                        }
                        final = status_map.get(final_status, "failed")
                        if final == "escalated":
                            case.status = CaseStatus.ESCALATED
                        elif final == "action_dispatched":
                            case.status = CaseStatus.AWAITING_CUSTOMER
                        else:
                            case.status = CaseStatus.STOPPED

                        self._ingest_outcome(case)
                        if ctx:
                            ctx.set_attribute("final_status", final)
                            ctx.set_attribute("total_turns", turn + 1)
                            ctx.set_attribute("tools_called_count", len(tools_called_history))
                        return HarnessResult(
                            case=case,
                            observations=observations,
                            final_status=final,
                            total_turns=turn + 1,
                            tools_called=tools_called_history,
                            error_count=error_count,
                        )

                    if turn + 1 >= MAX_HARNESS_ITERATIONS:
                        case.status = CaseStatus.STOPPED
                        self._ingest_outcome(case)
                        if ctx: ctx.set_attribute("final_status", "max_iterations")
                        return HarnessResult(
                            case=case,
                            observations=observations,
                            final_status="max_iterations",
                            total_turns=turn + 1,
                            tools_called=tools_called_history,
                            error_count=error_count,
                        )

                    if case.attempt_count >= case.max_attempts:
                        case.status = CaseStatus.STOPPED
                        self._ingest_outcome(case)
                        if ctx: ctx.set_attribute("final_status", "max_iterations")
                        return HarnessResult(
                            case=case,
                            observations=observations,
                            final_status="max_iterations",
                            total_turns=turn + 1,
                            tools_called=tools_called_history,
                            error_count=error_count,
                        )
                finally:
                    if turn_ctx:
                        turn_ctx.end()

            case.status = CaseStatus.STOPPED
            if ctx:
                ctx.set_attribute("final_status", "max_iterations")
                ctx.set_attribute("total_turns", MAX_HARNESS_ITERATIONS)
            return HarnessResult(
                case=case,
                observations=observations,
                final_status="max_iterations",
                total_turns=MAX_HARNESS_ITERATIONS,
                tools_called=tools_called_history,
                error_count=error_count,
            )
        finally:
            if ctx:
                ctx.end()

    async def run_recovery_case_async(self, case: Case) -> HarnessResult:
        """Run the full multi-turn harness loop on a case (async, non-blocking).

        All LLM calls and tool executions are offloaded to a thread pool
        via run_in_executor, so the event loop is never blocked.
        """
        from recovery_agent.agent.stopping import check_stopping_rules, transition_to_active_tier

        try:
            from opentelemetry import trace
            tracer = trace.get_tracer("recovery-agent-harness-async")
        except Exception:
            tracer = None

        observations: list[Observation] = []
        tools_called_history: list[str] = []
        error_count = 0

        attrs = {
            "payment_id": case.payment.payment_id,
            "amount": case.payment.amount,
            "failure_code": case.payment.failure_code or "",
            "failure_reason": case.payment.failure_reason or "",
            "customer_id": case.payment.customer_id or "",
            "max_attempts": case.max_attempts,
            "recovery_tier": case.recovery_tier.value if case.recovery_tier else "",
            "async": True,
        }
        ctx = tracer.start_span("harness.run_recovery_case_async", attributes=attrs) if tracer else None
        try:
            for turn in range(MAX_HARNESS_ITERATIONS):
                turn_ctx = tracer.start_span(
                    f"harness.turn_{turn}",
                    attributes={"turn": turn, "payment_id": case.payment.payment_id},
                ) if tracer else None
                try:
                    should_stop, stop_reason = check_stopping_rules(case)
                    if should_stop:
                        case.status = CaseStatus.STOPPED
                        self._ingest_outcome(case)
                        if turn_ctx:
                            turn_ctx.set_attribute("stop_reason", stop_reason)
                            turn_ctx.set_attribute("final_status", "stopped")
                        return HarnessResult(
                            case=case, observations=observations,
                            final_status="stopped", total_turns=turn,
                            tools_called=tools_called_history, error_count=error_count,
                        )
                    if stop_reason in ("SILENT_RETRY_FAILED", "SILENT_TIER_EXHAUSTED"):
                        case = transition_to_active_tier(case, stop_reason)
                        observations.append(Observation(
                            turn=turn + 1,
                            reasoning=f"Tier transition: {stop_reason}",
                            tool_calls=[], raw_response={"tier_transition": stop_reason},
                        ))

                    similar_ctx = self.vector_memory.get_decision_context(case)
                    prompt = _build_harness_prompt(case, observations, tools_called_history, similar_ctx)

                    result = await invoke_llm_json_async(
                        prompt=prompt,
                        system=HARNESS_SYSTEM_PROMPT,
                        temperature=0,
                        max_tokens=1024,
                    )

                    if result is None:
                        self._ingest_outcome(case)
                        if turn_ctx:
                            turn_ctx.set_attribute("final_status", "llm_failed")
                        return HarnessResult(
                            case=case,
                            observations=observations,
                            final_status="failed",
                            total_turns=turn,
                            tools_called=tools_called_history,
                            error_count=error_count,
                        )

                    reasoning = result.get("reasoning", "")
                    tool_calls_raw = result.get("tool_calls", [])
                    is_final = result.get("is_final", False)
                    final_status = result.get("status", "in_progress")

                    if turn_ctx:
                        turn_ctx.set_attribute("reasoning", reasoning[:500])
                        turn_ctx.set_attribute("tool_calls_count", len(tool_calls_raw))
                        turn_ctx.set_attribute("is_final", is_final)

                    tool_calls: list[ToolCall] = []

                    for tc in tool_calls_raw:
                        call_signature, arguments, is_duplicate = self._prepare_tool_arguments(case, tc, tools_called_history)
                        tool_name = tc.get("tool", "")

                        if is_duplicate:
                            tool_calls.append(ToolCall(
                                tool=tool_name,
                                arguments=arguments,
                                result={"status": "blocked", "message": "Duplicate call prohibited — already attempted"},
                                is_error=True,
                            ))
                            error_count += 1
                            continue

                        tool_span = tracer.start_span(
                            f"tool.{tool_name}",
                            attributes={"tool.name": tool_name, "payment_id": case.payment.payment_id},
                        ) if tracer else None
                        try:
                            block_reason = self._check_tool_allowed(case, tool_name, arguments)
                            if block_reason:
                                exec_result = {"status": "blocked", "message": block_reason}
                                is_error = True
                            else:
                                exec_result = await execute_tool_async(tool_name, arguments)
                                is_error = exec_result.get("status") in ("error", "unavailable")

                            if tool_span:
                                tool_span.set_attribute("tool.status", exec_result.get("status", "unknown"))
                                tool_span.set_attribute("tool.is_error", is_error)
                                if is_error:
                                    tool_span.set_attribute("tool.error", exec_result.get("message", "")[:200])
                        finally:
                            if tool_span:
                                tool_span.end()

                        tc_record = ToolCall(
                            tool=tool_name,
                            arguments=arguments,
                            result=exec_result,
                            is_error=is_error,
                        )
                        tool_calls.append(tc_record)
                        tools_called_history.append(call_signature)

                        if is_error:
                            error_count += 1
                            error_msg = exec_result.get("message", "unknown error")
                            reflection = await _reflect_on_error_async(case, tool_name, error_msg, observations)
                            reasoning += f" [Reflection: {reflection}]"

                        self._apply_tool_result(case, tool_name, exec_result)

                    obs = Observation(
                        turn=turn + 1,
                        reasoning=reasoning,
                        tool_calls=tool_calls,
                        raw_response=result,
                    )
                    observations.append(obs)

                    if is_final:
                        status_map = {
                            "action_dispatched": "action_dispatched",
                            "recovered": "action_dispatched",
                            "escalated": "escalated",
                            "failed": "failed",
                            "max_iterations": "max_iterations",
                        }
                        final = status_map.get(final_status, "failed")
                        if final == "escalated":
                            case.status = CaseStatus.ESCALATED
                        elif final == "action_dispatched":
                            case.status = CaseStatus.AWAITING_CUSTOMER
                        else:
                            case.status = CaseStatus.STOPPED

                        self._ingest_outcome(case)
                        if ctx:
                            ctx.set_attribute("final_status", final)
                            ctx.set_attribute("total_turns", turn + 1)
                            ctx.set_attribute("tools_called_count", len(tools_called_history))
                        return HarnessResult(
                            case=case,
                            observations=observations,
                            final_status=final,
                            total_turns=turn + 1,
                            tools_called=tools_called_history,
                            error_count=error_count,
                        )

                    if turn + 1 >= MAX_HARNESS_ITERATIONS:
                        case.status = CaseStatus.STOPPED
                        self._ingest_outcome(case)
                        if ctx: ctx.set_attribute("final_status", "max_iterations")
                        return HarnessResult(
                            case=case,
                            observations=observations,
                            final_status="max_iterations",
                            total_turns=turn + 1,
                            tools_called=tools_called_history,
                            error_count=error_count,
                        )

                    if case.attempt_count >= case.max_attempts:
                        case.status = CaseStatus.STOPPED
                        self._ingest_outcome(case)
                        if ctx: ctx.set_attribute("final_status", "max_iterations")
                        return HarnessResult(
                            case=case,
                            observations=observations,
                            final_status="max_iterations",
                            total_turns=turn + 1,
                            tools_called=tools_called_history,
                            error_count=error_count,
                        )
                finally:
                    if turn_ctx:
                        turn_ctx.end()

            case.status = CaseStatus.STOPPED
            self._ingest_outcome(case)
            if ctx:
                ctx.set_attribute("final_status", "max_iterations")
                ctx.set_attribute("total_turns", MAX_HARNESS_ITERATIONS)
            return HarnessResult(
                case=case,
                observations=observations,
                final_status="max_iterations",
                total_turns=MAX_HARNESS_ITERATIONS,
                tools_called=tools_called_history,
                error_count=error_count,
            )
        finally:
            if ctx:
                ctx.end()

    # ── Shared helper methods (used by both sync and async loops) ──

    def _prepare_tool_arguments(self, case: Case, tc: dict, tools_called_history: list[str]) -> tuple[str, dict, bool]:
        """Prepare tool arguments: dedup check + cached metadata injection.

        Returns (call_signature, arguments, is_duplicate).
        """
        tool_name = tc.get("tool", "")
        arguments = tc.get("arguments", {})

        call_signature = f"{tool_name}({json.dumps(arguments, sort_keys=True)})"
        if call_signature in tools_called_history:
            return call_signature, arguments, True

        cached_metadata = {
            "method": case.payment.metadata.get("method", ""),
            "provider": case.payment.metadata.get("provider", ""),
            "error_code": case.payment.metadata.get("error_code", case.payment.failure_code),
            "error_description": case.payment.metadata.get("error_description", case.payment.failure_reason),
            "error_source": case.payment.metadata.get("error_source", ""),
            "error_step": case.payment.metadata.get("error_step", ""),
            "failure_reason": case.payment.failure_reason,
            "amount": case.payment.amount,
            "bank": case.payment.metadata.get("bank", ""),
            "card_network": case.payment.metadata.get("card_network", ""),
        }
        if cached_metadata["method"] or cached_metadata["error_code"]:
            arguments["cached_metadata"] = cached_metadata

        return call_signature, arguments, False

    def _check_tool_allowed(self, case: Case, tool_name: str, arguments: dict) -> str | None:
        """Check if tool execution is allowed. Returns error message or None."""
        decided_action = case.payment.metadata.get("decided_action", "")
        if tool_name == "escalate_to_human_agent" and decided_action != "escalate_to_human":
            return f"Escalation blocked: strategy planner decided '{decided_action}', not escalate_to_human"
        guardrail_result = self._guardrail_intercept(case, tool_name, arguments)
        if guardrail_result is not None:
            return guardrail_result.get("message", "Blocked by guardrail")
        return None

    def _ingest_outcome(self, case: Case) -> None:
        """Ingest completed case outcome into vector memory and strategy metrics."""
        try:
            self.vector_memory.ingest_outcome(case)
        except Exception:
            pass  # Never block on vector memory ingestion
        # Record strategy outcome for empirical learning
        try:
            if case.diagnosis and case.attempts:
                last_action = case.attempts[-1].action_type
                self.strategy_metrics.record_outcome(
                    failure_type=case.diagnosis.root_cause,
                    action=last_action,
                    recovered=case.recovered,
                )
        except Exception:
            pass  # Never block on metrics recording

    def _apply_tool_result(self, case: Case, tool_name: str, result: dict[str, Any]):
        """Apply a tool result to the case state.

        Tracks attempt counts and handles tier escalation:
        - Silent tier retry failure → auto-escalate to Active tier
        """
        status = result.get("status", "")

        if tool_name == "escalate_to_human_agent" and status == "escalated":
            case.status = CaseStatus.ESCALATED
            case.payment.metadata["escalated"] = True
            case.payment.metadata["escalation_ticket"] = result.get("ticket_id", "")

        elif tool_name == "generate_smart_recovery_link" and status == "ok":
            case.payment.metadata["recovery_link"] = result.get("link_url", "")

        elif tool_name == "schedule_payday_retry" and status == "scheduled":
            case.payment.metadata["scheduled_retry"] = result.get("target_time", "")
            # BUG FIX: Don't increment here — it's incremented below in the catch-all
            # to prevent double-counting

        elif tool_name == "check_bank_health":
            case.payment.metadata["bank_health"] = result.get("health_score", 0)
            case.payment.metadata["bank_status"] = result.get("status", "unknown")

        elif tool_name == "calculate_payday_window":
            case.payment.metadata["in_payday_window"] = result.get("in_payday_window", False)
            case.payment.metadata["next_payday"] = result.get("next_payday", "")

        elif tool_name == "query_gateway_error_details":
            # Enrich case metadata with detailed gateway info
            if status == "ok":
                case.payment.metadata["detailed_error_code"] = result.get("error_code", "")
                case.payment.metadata["detailed_error_source"] = result.get("error_source", "")
                case.payment.metadata["detailed_error_step"] = result.get("error_step", "")
                case.payment.metadata["detailed_error_reason"] = result.get("error_reason", "")
                case.payment.metadata["card_network"] = result.get("card_network", "")
                case.payment.metadata["bank_name"] = result.get("bank", "")

        # Increment attempt count for action tools ONLY if not blocked
        # BUG FIX: Guardrail-blocked actions should not count against attempt budget
        if tool_name in ("generate_smart_recovery_link", "schedule_payday_retry", "escalate_to_human_agent"):
            if status not in ("blocked", "error"):
                case.attempt_count += 1

        # Track silent attempts and escalate on failure
        # BUG FIX: Only escalate for ACTION tools, not diagnostic/query tools
        ACTION_TOOLS = ("generate_smart_recovery_link", "schedule_payday_retry", "escalate_to_human_agent", "initiate_voice_call")
        if case.recovery_tier == RecoveryTier.SILENT:
            if status in ("error", "unavailable", "blocked") and tool_name in ACTION_TOOLS:
                # Silent retry failed → escalate to Active immediately
                case.recovery_tier = RecoveryTier.ACTIVE
                case.payment.metadata["tier_transition"] = "silent_to_active"
                case.payment.metadata["tier_transition_reason"] = (
                    f"Silent tier tool '{tool_name}' failed with status '{status}'. "
                    f"Automatic escalation to Active tier."
                )
            elif tool_name in ("generate_smart_recovery_link", "schedule_payday_retry"):
                case.silent_attempts += 1
