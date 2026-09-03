"""Case state machine — which transitions are legal, and why.

Block D2 of REBUILD-PLAN.md.

This module is deliberately *pure*: it depends only on `CaseStatus` and knows
nothing about storage. The evidence rules that need to read a case's history
live in `ledger.py`, where the history is. That split keeps this table readable
and independently testable.

The two rules the old system lacked
----------------------------------
1. **Terminal is terminal.** Nothing leaves RECOVERED / STOPPED / ESCALATED.
2. **Terminal is reachable.** Every non-terminal state can reach all three.
   The old `check_stopping_rules()` could not return `should_stop=True` for any
   input a real run produced (AUDIT-FINDINGS S0-1), so cases died at
   `MAX_TURNS` in `ACTING` forever.

Shape of the graph
------------------
    OPEN ─┬─> DIAGNOSING ──> DIAGNOSED ─┐
          │                             │
          └─────────────────────────────┴──> ACTING ─┬──> AWAITING_CUSTOMER
                                                     └──> SCHEDULED
    AWAITING_CUSTOMER <──> SCHEDULED <──> ACTING     (work can cycle)

    every non-terminal ──> RECOVERED | STOPPED | ESCALATED   (always allowed)
"""
from __future__ import annotations

from recovery_agent.models import CaseStatus

TERMINAL: frozenset[CaseStatus] = frozenset({
    CaseStatus.RECOVERED,
    CaseStatus.STOPPED,
    CaseStatus.ESCALATED,
})

#: States where the case is parked waiting for the outside world.
WAITING: frozenset[CaseStatus] = frozenset({
    CaseStatus.AWAITING_CUSTOMER,   # waiting on a person — the sensor polls these
    CaseStatus.SCHEDULED,           # waiting on a clock  — the scheduler wakes these
})

# Any non-terminal case may always be given up on, escalated, or found recovered.
# Recovery from ANY state is intentional: a customer can pay of their own accord
# before the agent acts. Whether the agent *caused* it is attribution, not state
# (see CaseRecord.attributed_to_agent) — conflating the two inflates the metric.
_ALWAYS: frozenset[CaseStatus] = frozenset({
    CaseStatus.RECOVERED,
    CaseStatus.STOPPED,
    CaseStatus.ESCALATED,
})

_WORK: dict[CaseStatus, frozenset[CaseStatus]] = {
    CaseStatus.OPEN: frozenset({
        CaseStatus.DIAGNOSING, CaseStatus.DIAGNOSED, CaseStatus.ACTING,
        CaseStatus.SCHEDULED,
    }),
    CaseStatus.DIAGNOSING: frozenset({CaseStatus.DIAGNOSED, CaseStatus.ACTING}),
    CaseStatus.DIAGNOSED: frozenset({CaseStatus.ACTING, CaseStatus.SCHEDULED}),
    CaseStatus.ACTING: frozenset({
        CaseStatus.ACTING,                 # several effectors in one turn
        CaseStatus.AWAITING_CUSTOMER,
        CaseStatus.SCHEDULED,
    }),
    CaseStatus.AWAITING_CUSTOMER: frozenset({
        CaseStatus.ACTING, CaseStatus.SCHEDULED,
    }),
    CaseStatus.SCHEDULED: frozenset({CaseStatus.ACTING}),
    CaseStatus.RECOVERED: frozenset(),
    CaseStatus.STOPPED: frozenset(),
    CaseStatus.ESCALATED: frozenset(),
}

#: The complete legal transition graph.
LEGAL_TRANSITIONS: dict[CaseStatus, frozenset[CaseStatus]] = {
    state: (allowed if state in TERMINAL else allowed | _ALWAYS)
    for state, allowed in _WORK.items()
}


class IllegalTransition(ValueError):
    """The transition is not in the state graph."""

    def __init__(self, frm: CaseStatus, to: CaseStatus, detail: str = ""):
        self.frm, self.to = frm, to
        allowed = sorted(s.value for s in LEGAL_TRANSITIONS.get(frm, frozenset()))
        msg = f"illegal transition {frm.value} -> {to.value}"
        if frm in TERMINAL:
            msg += f": {frm.value} is terminal, nothing follows it"
        else:
            msg += f": legal targets are {allowed or '(none)'}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


class MissingEvidence(IllegalTransition):
    """The transition is in the graph, but the case has not earned it.

    Subclasses `IllegalTransition` so callers can catch either, while keeping the
    two failures distinguishable: one means "that is not a legal move", the other
    means "that move is legal but you have not proved it".
    """

    def __init__(self, frm: CaseStatus, to: CaseStatus, detail: str):
        self.frm, self.to = frm, to
        ValueError.__init__(
            self, f"{frm.value} -> {to.value} requires evidence: {detail}"
        )


def is_terminal(status: CaseStatus) -> bool:
    return status in TERMINAL


def is_waiting(status: CaseStatus) -> bool:
    """True if the case is parked on the outside world rather than on the agent."""
    return status in WAITING


def can_transition(frm: CaseStatus, to: CaseStatus) -> bool:
    return to in LEGAL_TRANSITIONS.get(frm, frozenset())


def assert_transition(frm: CaseStatus, to: CaseStatus, detail: str = "") -> None:
    """Raise `IllegalTransition` unless `frm -> to` is legal."""
    if not can_transition(frm, to):
        raise IllegalTransition(frm, to, detail)


def reachable_from(frm: CaseStatus) -> frozenset[CaseStatus]:
    """Every status reachable from `frm`, transitively."""
    seen: set[CaseStatus] = set()
    stack = [frm]
    while stack:
        cur = stack.pop()
        for nxt in LEGAL_TRANSITIONS.get(cur, frozenset()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return frozenset(seen)
