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
from recovery_agent.agent.stopping import check_stopping_rules, run_stopping_check
from recovery_agent.logging import AuditLogger
from recovery_agent.models import (
    AgentState,
    AuditStep,
    Case,
)


class RecoveryAgent:
    """Revenue recovery agent built with LangGraph.

    Architecture:
        detect -> diagnose -> decide -> act -> observe (stop?) -> loop or end

    Source: LangGraph components — Nodes, Edges, Conditional Edges
    https://www.deeplearning.ai/courses/ai-agents-in-langgraph
    """

    def __init__(self, audit_logger: AuditLogger | None = None):
        self.logger = audit_logger or AuditLogger()
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
        """Step 3: Decide — choose intervention based on diagnosis."""
        start = time.time()
        case = state.case

        case = run_decision(case)
        action = case.payment.metadata.get("decided_action", "unknown")

        self.logger.log_step(
            case=case,
            step=AuditStep.DECIDE,
            input_data={
                "root_cause": case.diagnosis.root_cause.value if case.diagnosis else "unknown",
                "attempt_count": case.attempt_count,
            },
            reasoning=f"Attempt #{case.attempt_count + 1}. "
                      f"Cause: {case.diagnosis.root_cause.value if case.diagnosis else 'unknown'}. "
                      f"Chosen action: {action}",
            output_data={"action": action},
            duration_ms=int((time.time() - start) * 1000),
        )

        return {"case": case, "current_step": AuditStep.DECIDE}

    def _act(self, state: AgentState) -> dict:
        """Step 4: Act — execute the chosen intervention."""
        start = time.time()
        case = state.case

        case = run_execution(case)
        last_attempt = case.attempts[-1] if case.attempts else None

        self.logger.log_step(
            case=case,
            step=AuditStep.ACT,
            input_data={
                "action": last_attempt.action_type.value if last_attempt else "none",
                "attempt_number": case.attempt_count,
            },
            reasoning=f"Executed: {last_attempt.action_type.value}. "
                      f"Result: {last_attempt.result}. "
                      f"{last_attempt.action_details.get('detail', '')}",
            output_data={
                "action": last_attempt.action_type.value if last_attempt else "none",
                "result": last_attempt.result if last_attempt else "none",
            },
            duration_ms=int((time.time() - start) * 1000),
        )

        return {"case": case, "current_step": AuditStep.ACT}

    def _observe(self, state: AgentState) -> dict:
        """Step 5: Observe — evaluate outcome and check stopping rules."""
        start = time.time()
        case = state.case

        case = run_stopping_check(case)
        should_stop, stop_reason = check_stopping_rules(case)

        self.logger.log_step(
            case=case,
            step=AuditStep.OBSERVE,
            input_data={
                "recovered": case.recovered,
                "attempt_count": case.attempt_count,
                "status": case.status.value,
            },
            reasoning=f"After attempt #{case.attempt_count}: "
                      f"recovered={case.recovered}, "
                      f"status={case.status.value}. "
                      f"{'STOP: ' + stop_reason if should_stop else 'CONTINUE'}",
            output_data={
                "should_stop": should_stop,
                "stop_reason": stop_reason,
                "status": case.status.value,
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
                          f"Recovered amount: {case.recovered_amount}",
                output_data={
                    "final_status": case.status.value,
                    "total_attempts": case.attempt_count,
                    "recovered": case.recovered,
                    "recovered_amount": case.recovered_amount,
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
        initial_state = AgentState(case=case)
        final_state = self.graph.invoke(initial_state)
        return final_state["case"]
