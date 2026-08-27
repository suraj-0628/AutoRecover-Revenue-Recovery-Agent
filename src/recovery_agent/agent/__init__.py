"""Agent loop — the main recovery agent using LangGraph.

Source: LangGraph components (Nodes, Edges, State, Conditional Edges)
        AI Agents in LangGraph — Harrison Chase
        https://www.deeplearning.ai/courses/ai-agents-in-langgraph
"""
from __future__ import annotations

import time
from typing import Literal

from langgraph.graph import END, StateGraph

from recovery_agent.agent.decision import run_decision
from recovery_agent.agent.diagnosis import run_diagnosis
from recovery_agent.agent.execution import run_execution
from recovery_agent.agent.guardrails import GuardrailEngine
from recovery_agent.agent.harness import AgentHarness, HarnessResult
from recovery_agent.agent.memory import CustomerMemoryStore
from recovery_agent.agent.squad import SquadOrchestrator
from recovery_agent.agent.stopping import check_stopping_rules, run_stopping_check
from recovery_agent.logging import AuditLogger
from recovery_agent.models import (
    AgentState,
    AuditStep,
    Case,
    RecoveryTier,
)


class RecoveryAgent:
    """Revenue recovery agent built with LangGraph.

    Architecture:
        detect -> diagnose -> decide -> [guardrail] -> act -> observe (stop?) -> loop or end

    Supports both monolithic mode and squad mode (SquadOrchestrator).

    Source: LangGraph components — Nodes, Edges, Conditional Edges
    https://www.deeplearning.ai/courses/ai-agents-in-langgraph
    """

    def __init__(
        self,
        audit_logger: AuditLogger | None = None,
        memory_store: CustomerMemoryStore | None = None,
        guardrail_engine: GuardrailEngine | None = None,
        use_squad: bool = False,
        use_harness: bool = False,
    ):
        self.logger = audit_logger or AuditLogger()
        self.memory = memory_store or CustomerMemoryStore()
        self.guardrails = guardrail_engine or GuardrailEngine()
        self.use_squad = use_squad
        self.use_harness = use_harness
        if use_squad:
            self.squad = SquadOrchestrator(
                memory_store=self.memory,
                guardrail_engine=self.guardrails,
            )
        if use_harness:
            self.harness = AgentHarness(
                memory_store=self.memory,
                guardrail_engine=self.guardrails,
            )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        graph = StateGraph(AgentState)

        # Add nodes (processing steps)
        graph.add_node("detect", self._detect)
        graph.add_node("diagnose", self._diagnose)
        graph.add_node("decide", self._decide)
        graph.add_node("act", self._act)
        graph.add_node("observe", self._observe)

        # Set entry point
        graph.set_entry_point("detect")

        # Add edges (transitions)
        graph.add_edge("detect", "diagnose")
        graph.add_edge("diagnose", "decide")
        graph.add_edge("decide", "act")
        graph.add_edge("act", "observe")

        # Conditional edge from observe: loop back or stop
        graph.add_conditional_edges(
            "observe",
            self._should_continue,
            {
                "continue": "diagnose",  # Loop back for next attempt
                "stop": END,
            },
        )

        return graph.compile()

    def _detect(self, state: AgentState) -> dict:
        """Step 1: Detect — confirm we have a revenue-at-risk case."""
        start = time.time()
        case = state.case

        self.logger.log_step(
            case=case,
            step=AuditStep.DETECT,
            input_data={
                "payment_id": case.payment.payment_id,
                "amount": case.payment.amount,
                "failure_reason": case.payment.failure_reason,
            },
            reasoning=f"Payment {case.payment.payment_id} failed: {case.payment.failure_reason}. "
                      f"Amount: {case.payment.currency} {case.payment.amount}. "
                      f"Opening recovery case.",
            output_data={"case_id": case.id, "status": "open"},
            duration_ms=int((time.time() - start) * 1000),
        )

        return {"case": case, "current_step": AuditStep.DETECT}

    def _diagnose(self, state: AgentState) -> dict:
        """Step 2: Diagnose — determine root cause of failure."""
        start = time.time()
        case = state.case

        case = run_diagnosis(case)

        self.logger.log_step(
            case=case,
            step=AuditStep.DIAGNOSE,
            input_data={
                "failure_reason": case.payment.failure_reason,
                "failure_code": case.payment.failure_code,
            },
            reasoning=f"Diagnosis: {case.diagnosis.root_cause.value} "
                      f"(confidence: {case.diagnosis.confidence:.0%}). "
                      f"{case.diagnosis.reasoning}",
            output_data={
                "root_cause": case.diagnosis.root_cause.value,
                "confidence": case.diagnosis.confidence,
            },
            duration_ms=int((time.time() - start) * 1000),
        )

        return {"case": case, "current_step": AuditStep.DIAGNOSE}

    def _decide(self, state: AgentState) -> dict:
        """Step 3: Decide — choose intervention based on diagnosis, memory, and tier."""
        start = time.time()
        case = state.case

        # Load customer profile from memory
        profile = self.memory.get_or_create_profile(case.payment.customer_id)

        case = run_decision(case, profile=profile, memory=self.memory)
        action = case.payment.metadata.get("decided_action", "unknown")
        tier = case.recovery_tier.value

        # Log tier assignment
        tier_reasoning = case.payment.metadata.get("strategy_reasoning", "")
        tier_enforced = case.payment.metadata.get("tier_enforced", False)
        hard_decline_blocked = case.payment.metadata.get("hard_decline_blocked", False)

        self.logger.log_step(
            case=case,
            step=AuditStep.DECIDE,
            input_data={
                "root_cause": case.diagnosis.root_cause.value if case.diagnosis else "unknown",
                "attempt_count": case.attempt_count,
                "memory_enhanced": True,
                "preferred_channel": profile.preferred_channel,
                "recovery_tier": tier,
                "silent_attempts": case.silent_attempts,
                "penalties_prevented": case.penalties_prevented,
            },
            reasoning=f"Attempt #{case.attempt_count + 1}. "
                      f"Cause: {case.diagnosis.root_cause.value if case.diagnosis else 'unknown'}. "
                      f"Tier: {tier.upper()}. "
                      f"Channel: {profile.preferred_channel or 'default'}. "
                      f"Chosen action: {action}"
                      f"{' [TIER ENFORCED]' if tier_enforced else ''}"
                      f"{' [HARD DECLINE BLOCKED]' if hard_decline_blocked else ''}",
            output_data={
                "action": action,
                "recovery_tier": tier,
                "tier_enforced": tier_enforced,
                "hard_decline_blocked": hard_decline_blocked,
                "penalties_prevented": case.penalties_prevented,
            },
            duration_ms=int((time.time() - start) * 1000),
        )

        return {"case": case, "current_step": AuditStep.DECIDE}

    def _act(self, state: AgentState) -> dict:
        """Step 4: Act — execute the chosen intervention with guardrail interception.

        When use_squad=True, delegates to SquadOrchestrator for full agent coordination.
        Logs tier information for observability.
        """
        start = time.time()
        case = state.case

        # Load customer profile for guardrail checks
        profile = self.memory.get_or_create_profile(case.payment.customer_id)

        if self.use_squad:
            # Squad mode: run full Diagnose → Plan → Guard → Execute pipeline
            result = self.squad.run_step(case, profile=profile)
            case = result.next_case
            last_attempt = case.attempts[-1] if case.attempts else None

            self.logger.log_step(
                case=case,
                step=AuditStep.ACT,
                input_data={
                    "action": result.action_taken,
                    "attempt_number": case.attempt_count,
                    "verdict": result.verdict,
                    "squad_mode": True,
                    "recovery_tier": case.recovery_tier.value,
                    "silent_attempts": case.silent_attempts,
                    "penalties_prevented": case.penalties_prevented,
                },
                reasoning=f"Squad executed: {result.action_taken}. "
                          f"Verdict: {result.verdict}. "
                          f"Tier: {case.recovery_tier.value.upper()}. "
                          f"{last_attempt.action_details.get('detail', '') if last_attempt else ''}",
                output_data={
                    "action": result.action_taken,
                    "verdict": result.verdict,
                    "squad_mode": True,
                    "recovery_tier": case.recovery_tier.value,
                },
                duration_ms=int((time.time() - start) * 1000),
            )
        else:
            # Monolithic mode: original flow
            case = run_execution(case, guardrail_engine=self.guardrails, profile=profile)
            last_attempt = case.attempts[-1] if case.attempts else None

            guardrail_checks = case.payment.metadata.get("guardrail_checks", [])
            guardrail_final = case.payment.metadata.get("guardrail_final_action", "")

            self.logger.log_step(
                case=case,
                step=AuditStep.ACT,
                input_data={
                    "action": last_attempt.action_type.value if last_attempt else "none",
                    "attempt_number": case.attempt_count,
                    "guardrail_checks": len(guardrail_checks),
                    "guardrail_final_action": guardrail_final,
                    "recovery_tier": case.recovery_tier.value,
                    "silent_attempts": case.silent_attempts,
                    "penalties_prevented": case.penalties_prevented,
                },
                reasoning=f"Executed: {last_attempt.action_type.value}. "
                          f"Result: {last_attempt.result}. "
                          f"Tier: {case.recovery_tier.value.upper()}. "
                          f"Guardrail: {guardrail_final or 'none'}. "
                          f"{last_attempt.action_details.get('detail', '')}",
                output_data={
                    "action": last_attempt.action_type.value if last_attempt else "none",
                    "result": last_attempt.result if last_attempt else "none",
                    "guardrail_final_action": guardrail_final,
                    "recovery_tier": case.recovery_tier.value,
                },
                duration_ms=int((time.time() - start) * 1000),
            )

        return {"case": case, "current_step": AuditStep.ACT}

    def _observe(self, state: AgentState) -> dict:
        """Step 5: Observe — evaluate outcome, check stopping rules, update memory, handle tier transitions."""
        start = time.time()
        case = state.case

        # Record tier before stopping check (may transition)
        previous_tier = case.recovery_tier

        case = run_stopping_check(case)
        should_stop, stop_reason = check_stopping_rules(case)

        # Check for tier transition
        tier_transitioned = case.recovery_tier != previous_tier

        # Update customer memory with this attempt's outcome
        if case.attempts:
            last_attempt = case.attempts[-1]
            channel = case.payment.metadata.get("decided_action", "")
            self.memory.update_profile_after_attempt(
                customer_id=case.payment.customer_id,
                attempt={
                    "payment_id": case.payment.payment_id,
                    "amount": case.payment.amount,
                    "failure_type": case.diagnosis.root_cause.value if case.diagnosis else "",
                },
                success=case.recovered,
                channel=channel,
            )

        self.logger.log_step(
            case=case,
            step=AuditStep.OBSERVE,
            input_data={
                "recovered": case.recovered,
                "attempt_count": case.attempt_count,
                "status": case.status.value,
                "recovery_tier": case.recovery_tier.value,
                "silent_attempts": case.silent_attempts,
                "tier_transitioned": tier_transitioned,
                "penalties_prevented": case.penalties_prevented,
            },
            reasoning=f"After attempt #{case.attempt_count}: "
                      f"recovered={case.recovered}, "
                      f"status={case.status.value}, "
                      f"tier={case.recovery_tier.value.upper()}, "
                      f"silent_attempts={case.silent_attempts}, "
                      f"penalties_prevented={case.penalties_prevented}. "
                      f"{'TIER TRANSITION: ' + case.recovery_tier.value.upper() if tier_transitioned else ''} "
                      f"{'STOP: ' + stop_reason if should_stop else 'CONTINUE'}",
            output_data={
                "should_stop": should_stop,
                "stop_reason": stop_reason,
                "status": case.status.value,
                "recovery_tier": case.recovery_tier.value,
                "tier_transitioned": tier_transitioned,
                "penalties_prevented": case.penalties_prevented,
            },
            duration_ms=int((time.time() - start) * 1000),
        )

        if should_stop:
            self.logger.log_step(
                case=case,
                step=AuditStep.STOP,
                input_data={"status": case.status.value},
                reasoning=f"Agent stopped. Reason: {stop_reason}. "
                          f"Total attempts: {case.attempt_count}. "
                          f"Recovered: {case.recovered}. "
                          f"Recovered amount: {case.recovered_amount}. "
                          f"Penalties prevented: {case.penalties_prevented}. "
                          f"Final tier: {case.recovery_tier.value.upper()}",
                output_data={
                    "final_status": case.status.value,
                    "total_attempts": case.attempt_count,
                    "recovered": case.recovered,
                    "recovered_amount": case.recovered_amount,
                    "penalties_prevented": case.penalties_prevented,
                    "final_tier": case.recovery_tier.value,
                },
                duration_ms=0,
            )

        return {
            "case": case,
            "current_step": AuditStep.OBSERVE,
            "should_stop": should_stop,
            "loop_count": state.loop_count + 1,
        }

    def _should_continue(self, state: AgentState) -> Literal["continue", "stop"]:
        """Decide whether to loop or stop."""
        if state.should_stop:
            return "stop"
        if state.loop_count >= state.case.max_attempts:
            return "stop"
        return "continue"

    def run(self, case: Case) -> Case:
        """Run the full recovery loop on a case."""
        if self.use_harness:
            return self.run_harness(case)
        initial_state = AgentState(case=case)
        final_state = self.graph.invoke(initial_state)
        return final_state["case"]

    def run_harness(self, case: Case) -> Case:
        """Run the TrueForge-style AgentHarness on a case.

        Executes multi-turn reasoning with tool-calling, error reflection,
        and context compaction. Returns the case with enriched metadata.
        """
        start = time.time()

        # Step 1: Detect — log case opening
        self.logger.log_step(
            case=case,
            step=AuditStep.DETECT,
            input_data={
                "payment_id": case.payment.payment_id,
                "amount": case.payment.amount,
                "failure_reason": case.payment.failure_reason,
            },
            reasoning=f"Payment {case.payment.payment_id} failed: {case.payment.failure_reason}. "
                      f"Amount: {case.payment.currency} {case.payment.amount}. "
                      f"Opening recovery case via AgentHarness.",
            output_data={"case_id": case.id, "status": "open", "mode": "harness"},
            duration_ms=0,
        )

        # Step 2: Diagnose
        case = run_diagnosis(case)
        self.logger.log_step(
            case=case,
            step=AuditStep.DIAGNOSE,
            input_data={
                "failure_reason": case.payment.failure_reason,
                "failure_code": case.payment.failure_code,
            },
            reasoning=f"Diagnosis: {case.diagnosis.root_cause.value} "
                      f"(confidence: {case.diagnosis.confidence:.0%}). "
                      f"{case.diagnosis.reasoning}",
            output_data={
                "root_cause": case.diagnosis.root_cause.value,
                "confidence": case.diagnosis.confidence,
                "mode": "harness",
            },
            duration_ms=0,
        )

        # Step 3: Run the multi-turn harness loop
        harness_result = self.harness.run_recovery_case(case)

        # Step 4: Log harness results
        self._log_harness_result(case, harness_result, start)

        return case

    def _log_harness_result(self, case: Case, result: HarnessResult, start_time: float):
        """Log the harness run results to the audit trail."""
        tools_summary = ", ".join(set(result.tools_called)) if result.tools_called else "none"
        error_summary = f"{result.error_count} errors" if result.error_count else "clean run"

        # Log the decide step (tool planning happened here)
        self.logger.log_step(
            case=case,
            step=AuditStep.DECIDE,
            input_data={
                "mode": "harness",
                "total_turns": result.total_turns,
                "tools_called": result.tools_called,
                "error_count": result.error_count,
                "recovery_tier": case.recovery_tier.value,
            },
            reasoning=f"AgentHarness completed {result.total_turns} turn(s). "
                      f"Tools used: {tools_summary}. "
                      f"Errors: {error_summary}. "
                      f"Final status: {result.final_status}. "
                      f"Tier: {case.recovery_tier.value.upper()}",
            output_data={
                "mode": "harness",
                "total_turns": result.total_turns,
                "final_status": result.final_status,
                "tools_called": result.tools_called,
                "error_count": result.error_count,
                "recovery_tier": case.recovery_tier.value,
            },
            duration_ms=int((time.time() - start_time) * 1000),
        )

        # Log the stop step
        self.logger.log_step(
            case=case,
            step=AuditStep.STOP,
            input_data={
                "status": case.status.value,
                "mode": "harness",
                "final_status": result.final_status,
            },
            reasoning=f"AgentHarness stopped. Status: {result.final_status}. "
                      f"Total turns: {result.total_turns}. "
                      f"Recovered: {case.recovered}. "
                      f"Recovered amount: {case.recovered_amount}. "
                      f"Penalties prevented: {case.penalties_prevented}. "
                      f"Tools used: {tools_summary}",
            output_data={
                "mode": "harness",
                "final_status": result.final_status,
                "total_turns": result.total_turns,
                "recovered": case.recovered,
                "recovered_amount": case.recovered_amount,
                "penalties_prevented": case.penalties_prevented,
                "tools_called": result.tools_called,
                "error_count": result.error_count,
            },
            duration_ms=int((time.time() - start_time) * 1000),
        )
