"""RecoveryAgent — thin wrapper over the LangGraph ReAct agent.

The agent is a true ReAct loop with memory:
    LLM → tool_calls → execute tools → LLM decides next

This module is just a convenience wrapper that:
1. Creates the graph (with InMemoryStore)
2. Builds initial state from a Case
3. Invokes the graph
4. Stores episodes for future learning
5. Returns the result
"""
from __future__ import annotations

import time
import logging

from recovery_agent.agent.graph import get_graph, build_initial_state, get_memory_store
from recovery_agent.agent.governance import AGENT_VERSION
from recovery_agent.logging import AuditLogger
from recovery_agent.models import Case, CaseStatus

logger = logging.getLogger(__name__)


class RecoveryAgent:
    """Revenue recovery agent — thin wrapper over LangGraph ReAct pipeline.

    The LLM reasons about failed payments, calls tools to diagnose and recover,
    and loops until the payment is recovered or escalation is needed.
    """

    def __init__(
        self,
        audit_logger: AuditLogger | None = None,
        guardrail_engine=None,
    ):
        self.logger = audit_logger or AuditLogger()
        self.guardrails = guardrail_engine
        self.graph = get_graph()

    def run(self, case: Case) -> Case:
        """Run the ReAct agent on a case (synchronous).

        Invokes the LangGraph agent loop: LLM → tools → LLM → ... → END
        Episodes are stored in the memory store for future learning.
        """
        start = time.time()

        self.logger.log_step(
            case=case,
            step="detect",
            input_data={
                "payment_id": case.payment.payment_id,
                "amount": case.payment.amount,
                "failure_reason": case.payment.failure_reason,
            },
            reasoning=f"Payment {case.payment.payment_id} failed: {case.payment.failure_reason}. "
                      f"Amount: {case.payment.currency} {case.payment.amount}. "
                      f"Opening recovery case.",
            output_data={"case_id": case.id, "status": "open"},
            duration_ms=0,
        )

        initial_state = build_initial_state(case)
        from recovery_agent.agent.tools import ns_safe
        config = {
            "configurable": {
                # One session per payment — see frontend.py for why this is not
                # keyed on case.id.
                "thread_id": f"case:{case.payment.payment_id}",
                "payment_id": case.payment.payment_id,
                # langmem uses this as a namespace label, which rejects periods —
                # an unsanitised email killed every memory write.
                "customer_id": ns_safe(case.payment.customer_id),
            }
        }

        from recovery_agent.agent.tools import RecoveryContext
        context = RecoveryContext(guardrail_engine=self.guardrails, case=case)

        try:
            final_state = self.graph.invoke(initial_state, config=config, context=context)

            # Extract the agent's final response
            messages = final_state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    case.payment.metadata["agent_summary"] = last_msg.content

            # Count tool calls for metrics
            tool_call_count = sum(
                1 for m in messages
                if hasattr(m, "tool_calls") and m.tool_calls
            )
            case.payment.metadata["tool_call_count"] = tool_call_count
            case.payment.metadata["strategy_source"] = "react_agent"

            # Store episode in memory store for future learning
            self._store_episode(case, messages)

        except Exception as e:
            logger.error(f"[Agent] Execution failed: {e}")
            case.status = CaseStatus.STOPPED
            case.payment.metadata["agent_error"] = str(e)

        self._log_result(case, start)
        return case

    async def run_async(self, case: Case) -> Case:
        """Run the agent asynchronously (non-blocking)."""
        import asyncio
        from functools import partial
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self.run, case))

    def _store_episode(self, case: Case, messages: list):
        """Store the recovery episode in the LangGraph memory store.

        This allows the agent to search for past episodes when handling
        similar failures in the future.
        """
        try:
            store = get_memory_store()

            # Extract tool calls from messages
            tool_calls = []
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(tc["name"])

            # Extract final summary
            summary = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                    summary = msg.content
                    break

            episode = {
                "payment_id": case.payment.payment_id,
                "customer_id": case.payment.customer_id,
                "amount": case.payment.amount,
                "failure_code": case.payment.failure_code,
                "failure_reason": case.payment.failure_reason,
                "recovery_tier": case.recovery_tier.value,
                "attempt": case.attempt_count + 1,
                "tool_calls": tool_calls,
                "summary": summary[:500],
                "status": case.status.value,
                "agent_version": AGENT_VERSION,
            }

            store.put(
                ("episodes", case.payment.customer_id),
                case.payment.payment_id,
                episode,
            )
            logger.info(f"[Agent] Stored episode for {case.payment.payment_id}")
        except Exception as e:
            logger.debug(f"[Agent] Failed to store episode: {e}")

    def _log_result(self, case: Case, start_time: float):
        """Log the agent run results to the audit trail."""
        tool_count = case.payment.metadata.get("tool_call_count", 0)
        summary = case.payment.metadata.get("agent_summary", "")

        self.logger.log_step(
            case=case,
            step="decide",
            input_data={
                "recovery_tier": case.recovery_tier.value,
                "tool_call_count": tool_count,
            },
            reasoning=f"ReAct agent completed. "
                      f"Tool calls: {tool_count}. "
                      f"Summary: {summary[:200]}",
            output_data={
                "final_status": case.status.value,
                "tool_call_count": tool_count,
                "strategy_source": "react_agent",
            },
            duration_ms=int((time.time() - start_time) * 1000),
        )

    def run_component_evals(self):
        """Run component-level evaluations (pydantic-evals)."""
        from recovery_agent.eval.component_eval import (
            run_all_component_evals,
            DIAGNOSIS_DATASET,
            TOOL_SELECTION_DATASET,
            run_diagnosis_target,
            run_tool_selection_target,
        )

        print("\nRunning component evaluations...")
        diag_report = DIAGNOSIS_DATASET.evaluate_sync(run_diagnosis_target)
        tool_report = TOOL_SELECTION_DATASET.evaluate_sync(run_tool_selection_target)

        diag_report.print(include_input=True, include_output=True)
        tool_report.print(include_input=True, include_output=True)

        return {"diagnosis": diag_report, "tool_selection": tool_report}

    def run_error_analysis(self, cases: list[Case] | None = None):
        """Run error analysis using arize-phoenix-evals."""
        from recovery_agent.eval.error_analysis import (
            analyze_errors_with_phoenix,
            get_error_summary,
            save_error_analysis,
        )

        if cases is None:
            cases = []

        print("\nRunning error analysis...")
        results_df = analyze_errors_with_phoenix(cases)
        summary = get_error_summary(results_df)
        save_error_analysis(results_df)

        print(summary)
        return {"results": results_df, "summary": summary}
