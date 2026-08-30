"""Batch evaluation — runs multiple cases and measures recovery.

Source: Evaluation framework from Evaluating AI Agents (Arize AI)
        Gold-standard datasets from NeMo Agent Toolkit
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from recovery_agent.agent import RecoveryAgent
from recovery_agent.models import Case, CaseStatus, PaymentEvent

BATCH_CSV_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "batch_transactions.csv"

# failure_code -> recoverable mapping
_RECOVERABLE_MAP: dict[str, bool] = {
    "insufficient_funds": True,
    "card_expired": True,
    "risk_block": False,
}


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


def _load_batch_from_csv(csv_path: Path | None = None) -> list[PaymentEvent]:
    """Load payment events from a static CSV file."""
    path = csv_path or BATCH_CSV_PATH
    events: list[PaymentEvent] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(PaymentEvent(
                payment_id=row["payment_id"],
                customer_id=row["customer_id"],
                amount=float(row["amount"]),
                currency=row["currency"],
                failure_code=row["failure_code"],
                failure_reason=row["failure_reason"],
            ))
    return events


def _get_known_outcomes(events: list[PaymentEvent]) -> dict[str, dict]:
    """Derive known properties for each event from failure_code."""
    outcomes: dict[str, dict] = {}
    for event in events:
        outcomes[event.payment_id] = {
            "failure_type": event.failure_code,
            "recoverable": _RECOVERABLE_MAP.get(event.failure_code, False),
            "amount": event.amount,
        }
    return outcomes


def run_batch_evaluation(
    num_cases: int = 30,
    seed: int | None = None,
    csv_path: Path | None = None,
) -> BatchResult:
    """Run a batch of cases through the recovery agent and measure results.

    Reads from data/batch_transactions.csv by default. The seed parameter is
    kept for backward compatibility but is no longer used (static dataset).
    """
    events = _load_batch_from_csv(csv_path)
    known_outcomes = _get_known_outcomes(events)

    agent = RecoveryAgent()
    result = BatchResult()

    for event in events:
        case = Case(payment=event)
        final_case = agent.run(case)

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

        ftype = known_outcomes[event.payment_id]["failure_type"]
        if ftype not in result.by_failure_type:
            result.by_failure_type[ftype] = {"total": 0, "recovered": 0, "rate": 0.0}
        result.by_failure_type[ftype]["total"] += 1
        if final_case.recovered:
            result.by_failure_type[ftype]["recovered"] += 1

    result.avg_attempts = result.total_attempts / result.total_cases if result.total_cases > 0 else 0
    for stats in result.by_failure_type.values():
        stats["rate"] = stats["recovered"] / stats["total"] if stats["total"] > 0 else 0.0

    return result
