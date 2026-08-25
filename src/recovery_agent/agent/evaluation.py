"""Batch evaluation — runs multiple cases and measures recovery.

Source: Evaluation framework from Evaluating AI Agents (Arize AI)
        Gold-standard datasets from NeMo Agent Toolkit
"""
from __future__ import annotations

from dataclasses import dataclass, field

from recovery_agent.agent import RecoveryAgent
from recovery_agent.agent.test_generator import generate_batch, get_known_outcomes
from recovery_agent.models import Case, CaseStatus


@dataclass
class BatchResult:
    """Results from a batch evaluation run."""
    total_cases: int = 0
    recovered: int = 0
    stopped: int = 0
    escalated: int = 0
    total_amount: float = 0.0
    recovered_amount: float = 0.0
    avg_attempts: float = 0.0
    total_attempts: int = 0
    cases: list[dict] = field(default_factory=list)
    by_failure_type: dict[str, dict] = field(default_factory=dict)

    @property
    def recovery_rate(self) -> float:
        return self.recovered / self.total_cases if self.total_cases > 0 else 0.0

    @property
    def recovery_amount_rate(self) -> float:
        return self.recovered_amount / self.total_amount if self.total_amount > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "BATCH EVALUATION RESULTS",
            "=" * 60,
            f"Total cases:     {self.total_cases}",
            f"Recovered:       {self.recovered} ({self.recovery_rate:.1%})",
            f"Stopped:         {self.stopped}",
            f"Escalated:       {self.escalated}",
            f"Total amount:    INR {self.total_amount:,.2f}",
            f"Recovered amount: INR {self.recovered_amount:,.2f} ({self.recovery_amount_rate:.1%})",
            f"Avg attempts:    {self.avg_attempts:.1f}",
            "-" * 60,
            "BY FAILURE TYPE:",
        ]
        for ftype, stats in self.by_failure_type.items():
            lines.append(
                f"  {ftype}: {stats['recovered']}/{stats['total']} recovered "
                f"({stats['rate']:.0%})"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def run_batch_evaluation(
    num_cases: int = 30,
    seed: int | None = None,
) -> BatchResult:
    """Run a batch of cases through the recovery agent and measure results.

    Source: Evaluation framework — structured experiments
    https://www.deeplearning.ai/courses/evaluating-ai-agents
    """
    if seed is not None:
        import random
        random.seed(seed)

    # Generate test batch
    events = generate_batch(num_cases)
    known_outcomes = get_known_outcomes(events)

    # Run each case through the agent
    agent = RecoveryAgent()
    result = BatchResult()

    for event in events:
        case = Case(payment=event)
        final_case = agent.run(case)

        # Record result
        case_outcome = {
            "case_id": final_case.id,
            "payment_id": final_case.payment.payment_id,
            "failure_type": known_outcomes[event.payment_id]["failure_type"],
            "recoverable": known_outcomes[event.payment_id]["recoverable"],
            "status": final_case.status.value,
            "recovered": final_case.recovered,
            "recovered_amount": final_case.recovered_amount,
            "attempts": final_case.attempt_count,
            "amount": final_case.payment.amount,
        }
        result.cases.append(case_outcome)

        # Aggregate
        result.total_cases += 1
        result.total_amount += final_case.payment.amount
        result.total_attempts += final_case.attempt_count

        if final_case.recovered:
            result.recovered += 1
            result.recovered_amount += final_case.recovered_amount
        elif final_case.status == CaseStatus.STOPPED:
            result.stopped += 1
        elif final_case.status == CaseStatus.ESCALATED:
            result.escalated += 1

        # By failure type
        ftype = known_outcomes[event.payment_id]["failure_type"]
        if ftype not in result.by_failure_type:
            result.by_failure_type[ftype] = {"total": 0, "recovered": 0, "rate": 0.0}
        result.by_failure_type[ftype]["total"] += 1
        if final_case.recovered:
            result.by_failure_type[ftype]["recovered"] += 1

    # Calculate rates
    result.avg_attempts = result.total_attempts / result.total_cases if result.total_cases > 0 else 0
    for stats in result.by_failure_type.values():
        stats["rate"] = stats["recovered"] / stats["total"] if stats["total"] > 0 else 0.0

    return result
