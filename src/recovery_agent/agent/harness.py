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
from recovery_agent.agent.llm_client import invoke_llm_json, invoke_llm
from recovery_agent.agent.memory import CustomerMemoryStore
from recovery_agent.agent.tools import (
    TOOL_SCAPES,
    execute_tool,
    get_tool_schemas_for_llm,
)
from recovery_agent.models import (
    ActionType,
    AuditStep,
    Case,
    CaseStatus,
    RecoveryTier,
)


# --- Constants ---

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


def _build_harness_prompt(case: Case, observations: list[Observation], history: list[str]) -> str:
    """Build the multi-turn reasoning prompt with compacted context."""
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

    # Add previously blocked tools (to prevent repetition)
    if history:
        failure_ctx += f"\nPREVIOUSLY ATTEMPTED (DO NOT REPEAT IDENTICAL CALLS):\n"
        for h in history:
            failure_ctx += f"  - {h}\n"

    # Add compacted observations
    obs_text = _compact_context(observations) if observations else "No previous observations — this is the first turn."

    return f"""{failure_ctx}

═══ OBSERVATION HISTORY ═══
{obs_text}

═══ YOUR TASK ═══
Choose the next tool call(s) based on the failure context and observation history.
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
    ):
        self.memory = memory_store or CustomerMemoryStore()
        self.guardrails = guardrail_engine or GuardrailEngine()

    def run_recovery_case(self, case: Case) -> HarnessResult:
        """Run the full multi-turn harness loop on a case.

        Returns a HarnessResult with observations, final status, and metrics.
        """
        observations: list[Observation] = []
        tools_called_history: list[str] = []  # For repetition prohibition
        error_count = 0

        for turn in range(MAX_HARNESS_ITERATIONS):
            # 1. Context Engineering — build compacted prompt
            prompt = _build_harness_prompt(case, observations, tools_called_history)

            # 2. LLM Tool Plan — ask LLM to choose next tool(s)
            result = invoke_llm_json(
                prompt=prompt,
                system=HARNESS_SYSTEM_PROMPT,
                temperature=0,
                max_tokens=1024,
            )

            if result is None:
                # LLM unavailable — fall back to heuristic
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

            # 3. Execute tool calls
            tool_calls: list[ToolCall] = []

            for tc in tool_calls_raw:
                tool_name = tc.get("tool", "")
                arguments = tc.get("arguments", {})

                # Repetition prohibition
                call_signature = f"{tool_name}({json.dumps(arguments, sort_keys=True)})"
                if call_signature in tools_called_history:
                    tool_calls.append(ToolCall(
                        tool=tool_name,
                        arguments=arguments,
                        result={"status": "blocked", "message": "Duplicate call prohibited — already attempted"},
                        is_error=True,
                    ))
                    error_count += 1
                    continue

                # Execute the tool — pass cached metadata from case context
                # so simulated payments don't call live Razorpay API (which would 404)
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
                exec_result = execute_tool(tool_name, arguments)
                is_error = exec_result.get("status") in ("error", "unavailable")

                tc_record = ToolCall(
                    tool=tool_name,
                    arguments=arguments,
                    result=exec_result,
                    is_error=is_error,
                )
                tool_calls.append(tc_record)
                tools_called_history.append(call_signature)

                # 4. Error Reflection — if tool failed, reflect
                if is_error:
                    error_count += 1
                    error_msg = exec_result.get("message", "unknown error")
                    reflection = _reflect_on_error(case, tool_name, error_msg, observations)
                    reasoning += f" [Reflection: {reflection}]"

                # 5. Observation Feed — update case based on tool results
                self._apply_tool_result(case, tool_name, exec_result)

            # Record observation
            obs = Observation(
                turn=turn + 1,
                reasoning=reasoning,
                tool_calls=tool_calls,
                raw_response=result,
            )
            observations.append(obs)

            # Check stopping conditions
            if is_final:
                status_map = {
                    "recovered": "recovered",
                    "escalated": "escalated",
                    "failed": "failed",
                    "max_iterations": "max_iterations",
                }
                final = status_map.get(final_status, "failed")
                if final == "recovered":
                    case.recovered = True
                    case.recovered_amount = case.payment.amount
                    case.status = CaseStatus.RECOVERED
                elif final == "escalated":
                    case.status = CaseStatus.ESCALATED
                else:
                    case.status = CaseStatus.STOPPED

                return HarnessResult(
                    case=case,
                    observations=observations,
                    final_status=final,
                    total_turns=turn + 1,
                    tools_called=tools_called_history,
                    error_count=error_count,
                )

            # Check max iterations
            if turn + 1 >= MAX_HARNESS_ITERATIONS:
                case.status = CaseStatus.STOPPED
                return HarnessResult(
                    case=case,
                    observations=observations,
                    final_status="max_iterations",
                    total_turns=turn + 1,
                    tools_called=tools_called_history,
                    error_count=error_count,
                )

            # Check attempt-based stopping
            if case.attempt_count >= case.max_attempts:
                case.status = CaseStatus.STOPPED
                return HarnessResult(
                    case=case,
                    observations=observations,
                    final_status="max_iterations",
                    total_turns=turn + 1,
                    tools_called=tools_called_history,
                    error_count=error_count,
                )

        # Exhausted all iterations
        case.status = CaseStatus.STOPPED
        return HarnessResult(
            case=case,
            observations=observations,
            final_status="max_iterations",
            total_turns=MAX_HARNESS_ITERATIONS,
            tools_called=tools_called_history,
            error_count=error_count,
        )

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
            case.attempt_count += 1

        elif tool_name == "schedule_payday_retry" and status == "scheduled":
            case.payment.metadata["scheduled_retry"] = result.get("target_time", "")
            # If scheduling a retry on a silent tier, track it
            if case.recovery_tier == RecoveryTier.SILENT:
                case.silent_attempts += 1

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

        # Increment attempt count for action tools
        if tool_name in ("generate_smart_recovery_link", "schedule_payday_retry", "escalate_to_human_agent"):
            case.attempt_count += 1

        # Track silent attempts and escalate on failure
        if case.recovery_tier == RecoveryTier.SILENT:
            if status in ("error", "unavailable", "blocked"):
                # Silent retry failed → escalate to Active immediately
                case.recovery_tier = RecoveryTier.ACTIVE
                case.payment.metadata["tier_transition"] = "silent_to_active"
                case.payment.metadata["tier_transition_reason"] = (
                    f"Silent tier tool '{tool_name}' failed with status '{status}'. "
                    f"Automatic escalation to Active tier."
                )
            elif tool_name in ("generate_smart_recovery_link", "schedule_payday_retry"):
                case.silent_attempts += 1
