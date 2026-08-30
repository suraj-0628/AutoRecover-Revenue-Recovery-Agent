"""Multi-Agent Squad — decoupled specialized agents coordinating via structured messages.

Orchestrates 4 specialized agent roles:
1. DiagnosticAgent: 3-layer root cause analysis
2. StrategyPlannerAgent: Memory + KG-aware intervention planning
3. ComplianceOverseerAgent: Pre-execution guardrail interception
4. ToolExecutionAgent: Deterministic SDK invocation with observable outcomes
"""
from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from recovery_agent.agent.decision import decide_intervention
from recovery_agent.agent.diagnosis import run_diagnosis
from recovery_agent.agent.execution import execute_action
from recovery_agent.agent.guardrails import GuardrailEngine, GuardrailCheckResult
from recovery_agent.agent.kg_router import RazorpayKnowledgeGraph
from recovery_agent.agent.memory import CustomerMemoryStore
from recovery_agent.agent.strategy_metrics import StrategyMetricsStore, ThompsonBandit
from recovery_agent.agent.vector_memory import VectorMemoryStore
from recovery_agent.models import (
    ActionType,
    Attempt,
    Case,
    CaseStatus,
    CustomerProfile,
    Diagnosis,
    FailureType,
)


# --- Message Types ---

class AgentMessage(BaseModel):
    """Structured message passed between agents."""
    sender: str
    receiver: str
    content: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class SquadStepResult(BaseModel):
    """Output of one squad orchestration step."""
    next_case: Case
    action_taken: str
    verdict: str  # "pass", "modified", "blocked"
    guardrail_checks: list[dict] = Field(default_factory=list)
    trajectory_step: dict[str, Any] = Field(default_factory=dict)


# --- Specialized Agents ---

class DiagnosticAgent:
    """Performs 3-layer root cause error analysis.

    Layer 1: Razorpay error code mapping
    Layer 2: LLM-powered reasoning (optional)
    Layer 3: Rule-based fallback heuristics
    """

    def diagnose(self, case: Case) -> Diagnosis:
        """Run diagnosis on a case. Returns Diagnosis with root cause and confidence."""
        case = run_diagnosis(case)
        return case.diagnosis


class StrategyPlannerAgent:
    """Generates multi-step recovery plans using Memory + KG Router + Vector Memory + Bandit.

    Uses customer payment history (Pillar 1), Knowledge Graph API
    discovery (Pillar 2), vector memory context, and Thompson Bandit
    empirical evidence to select the optimal intervention.
    """

    def __init__(
        self,
        vector_memory: VectorMemoryStore | None = None,
        strategy_metrics: StrategyMetricsStore | None = None,
        bandit: ThompsonBandit | None = None,
    ):
        self.vector_memory = vector_memory
        self.strategy_metrics = strategy_metrics
        self.bandit = bandit

    def plan(
        self,
        case: Case,
        profile: CustomerProfile,
        kg_router: RazorpayKnowledgeGraph,
        memory: CustomerMemoryStore,
    ) -> ActionType:
        """Choose intervention based on diagnosis, memory, KG, vector memory, and bandit."""
        action = decide_intervention(
            case,
            profile=profile,
            memory=memory,
            kg_router=kg_router,
            vector_memory=self.vector_memory,
            strategy_metrics=self.strategy_metrics,
            bandit=self.bandit,
        )
        # Store the decided action in case metadata for downstream use
        case.payment.metadata["decided_action"] = action.value
        return action


class ComplianceOverseerAgent:
    """Enforces NVIDIA NAT style guardrail interception before tool execution.

    Runs all 5 guardrails (quiet hours, frequency cap, double-debit lock,
    opt-out, monetary cap) and modifies/blocks non-compliant actions.
    """

    def __init__(self, guardrail_engine: GuardrailEngine | None = None):
        self.engine = guardrail_engine or GuardrailEngine()

    def intercept(
        self,
        case: Case,
        action: ActionType,
        profile: CustomerProfile,
    ) -> tuple[ActionType, list[GuardrailCheckResult]]:
        """Validate action against all guardrails. Returns (approved_action, checks)."""
        return self.engine.validate_action(case=case, action=action, profile=profile)


class ToolExecutionAgent:
    """Executes deterministic Razorpay SDK tools with observable outcomes.

    Invokes the correct API, updates customer memory, and logs results.
    """

    def execute(
        self,
        case: Case,
        action: ActionType,
        profile: CustomerProfile | None = None,
    ) -> dict:
        """Execute the action and return observable outcome dict."""
        cause_value = case.diagnosis.root_cause.value if case.diagnosis else "unknown"
        return execute_action(
            action,
            cause_value,
            case.payment.amount,
            payment_id=case.payment.payment_id,
            customer_email=case.payment.metadata.get("customer_email", ""),
            customer_phone=case.payment.metadata.get("customer_phone", ""),
        )


# --- Squad Orchestrator ---

class SquadOrchestrator:
    """Coordinates the 4 specialized agents in sequence.

    Flow: Diagnose → Plan → Intercept & Guard → Execute → Observe & Remember
    """

    def __init__(
        self,
        memory_store: CustomerMemoryStore | None = None,
        kg_router: RazorpayKnowledgeGraph | None = None,
        guardrail_engine: GuardrailEngine | None = None,
        vector_memory: VectorMemoryStore | None = None,
        strategy_metrics: StrategyMetricsStore | None = None,
        bandit: ThompsonBandit | None = None,
    ):
        self.memory = memory_store or CustomerMemoryStore()
        self.kg_router = kg_router or RazorpayKnowledgeGraph()
        self.guardrails = guardrail_engine or GuardrailEngine()

        # Instantiate specialized agents
        self.diagnostic = DiagnosticAgent()
        self.planner = StrategyPlannerAgent(
            vector_memory=vector_memory,
            strategy_metrics=strategy_metrics,
            bandit=bandit,
        )
        self.compliance = ComplianceOverseerAgent(self.guardrails)
        self.executor = ToolExecutionAgent()

    def run_step(
        self,
        case: Case,
        profile: CustomerProfile | None = None,
    ) -> SquadStepResult:
        """Execute one full squad step: Diagnose → Plan → Guard → Execute.

        Returns SquadStepResult with updated case, action, verdict, and trajectory.
        """
        start = time.time()
        customer_id = case.payment.customer_id

        # Ensure we have a profile
        if profile is None:
            profile = self.memory.get_or_create_profile(customer_id)

        # --- Step 1: Diagnose ---
        diagnosis = self.diagnostic.diagnose(case)
        case.diagnosis = diagnosis

        # --- Step 2: Plan (Memory + KG aware) ---
        proposed_action = self.planner.plan(
            case, profile, self.kg_router, self.memory,
        )

        # --- Step 3: Compliance Intercept ---
        approved_action, checks = self.compliance.intercept(
            case, proposed_action, profile,
        )

        # Determine verdict
        if approved_action != proposed_action:
            blocked_checks = [c for c in checks if c.verdict.value == "blocked"]
            modified_checks = [c for c in checks if c.verdict.value == "modified"]
            if blocked_checks:
                verdict = "blocked"
            elif modified_checks:
                verdict = "modified"
            else:
                verdict = "pass"
        else:
            verdict = "pass"

        # --- Step 4: Execute ---
        execution_result = self.executor.execute(case, approved_action, profile)

        # Record the attempt
        from recovery_agent.models import Attempt as AttemptModel
        attempt = AttemptModel(
            action_type=approved_action,
            action_details={
                "cause": diagnosis.root_cause.value,
                "amount": case.payment.amount,
                "detail": execution_result.get("detail", ""),
                "proposed_action": proposed_action.value,
                "verdict": verdict,
            },
            result="pending",
        )
        case.attempts.append(attempt)
        case.attempt_count += 1

        # Update metadata
        case.payment.metadata["decided_action"] = approved_action.value
        case.payment.metadata["guardrail_checks"] = [c.model_dump() for c in checks]
        case.payment.metadata["guardrail_final_action"] = approved_action.value

        # Enrich with KG rail info
        recommended_rail = case.payment.metadata.get("recommended_api_rail", "")
        if recommended_rail:
            execution_result["recommended_rail"] = recommended_rail

        # --- Build trajectory step ---
        elapsed_ms = int((time.time() - start) * 1000)
        trajectory_step = {
            "step": case.attempt_count,
            "diagnosis": diagnosis.root_cause.value,
            "diagnosis_confidence": diagnosis.confidence,
            "proposed_action": proposed_action.value,
            "approved_action": approved_action.value,
            "verdict": verdict,
            "guardrail_checks": len(checks),
            "execution_result": execution_result.get("action", "unknown"),
            "elapsed_ms": elapsed_ms,
        }

        return SquadStepResult(
            next_case=case,
            action_taken=approved_action.value,
            verdict=verdict,
            guardrail_checks=[c.model_dump() for c in checks],
            trajectory_step=trajectory_step,
        )
