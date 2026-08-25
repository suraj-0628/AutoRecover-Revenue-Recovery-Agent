"""Audit logger — records every decision for full traceability.

Source: Observability pillar from Governing AI Agents
        OpenTelemetry tracing concept from NeMo Agent Toolkit
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from recovery_agent.models import AuditEntry, AuditStep, Case


class AuditLogger:
    """Structured audit logger that writes JSONL per case."""

    def __init__(self, log_dir: str = "data/audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_entry(self, case_id: str, entry: AuditEntry) -> None:
        """Append a single audit entry to the case's log file."""
        log_file = self.log_dir / f"case_{case_id}.jsonl"
        with open(log_file, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def log_step(
        self,
        case: Case,
        step: AuditStep,
        input_data: dict[str, Any],
        reasoning: str,
        output_data: dict[str, Any],
        duration_ms: int = 0,
    ) -> AuditEntry:
        """Create and log an audit entry for a step."""
        entry = AuditEntry(
            step=step,
            input_data=input_data,
            reasoning=reasoning,
            output_data=output_data,
            duration_ms=duration_ms,
        )
        case.audit_log.append(entry)
        self.log_entry(case.id, entry)
        return entry

    def get_case_log(self, case_id: str) -> list[AuditEntry]:
        """Retrieve all audit entries for a case."""
        log_file = self.log_dir / f"case_{case_id}.jsonl"
        if not log_file.exists():
            return []
        entries = []
        for line in log_file.read_text().splitlines():
            if line.strip():
                entries.append(AuditEntry.model_validate_json(line))
        return entries

    def get_summary(self, case_id: str) -> dict[str, Any]:
        """Generate a summary of the case's audit trail."""
        entries = self.get_case_log(case_id)
        return {
            "case_id": case_id,
            "total_steps": len(entries),
            "steps": [e.step.value for e in entries],
            "total_duration_ms": sum(e.duration_ms for e in entries),
            "timestamps": {
                "first": entries[0].timestamp.isoformat() if entries else None,
                "last": entries[-1].timestamp.isoformat() if entries else None,
            },
        }
