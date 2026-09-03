"""D2 gate tests for the pure state graph.

The property that matters most here is the one the old system failed:
**every non-terminal state can reach a terminal one.** `check_stopping_rules()`
could not return should_stop=True for any input a real run produced
(AUDIT-FINDINGS S0-1), so cases sat in ACTING until MAX_TURNS killed them.
"""
from __future__ import annotations

import pytest

from recovery_agent.models import CaseStatus
from recovery_agent.statemachine import (
    LEGAL_TRANSITIONS,
    TERMINAL,
    WAITING,
    IllegalTransition,
    MissingEvidence,
    assert_transition,
    can_transition,
    is_terminal,
    is_waiting,
    reachable_from,
)

NON_TERMINAL = [s for s in CaseStatus if s not in TERMINAL]


def test_every_status_is_in_the_graph():
    """A status missing from the table would be a silent dead end."""
    assert set(LEGAL_TRANSITIONS) == set(CaseStatus)


def test_graph_is_closed():
    """No transition points at a status the graph does not define."""
    for frm, targets in LEGAL_TRANSITIONS.items():
        for to in targets:
            assert to in LEGAL_TRANSITIONS, f"{frm.value} -> {to.value} is off-graph"


@pytest.mark.parametrize("status", sorted(TERMINAL, key=lambda s: s.value))
def test_terminal_states_have_no_exit(status):
    assert LEGAL_TRANSITIONS[status] == frozenset()
    assert is_terminal(status)
    for other in CaseStatus:
        assert not can_transition(status, other)


@pytest.mark.parametrize("status", NON_TERMINAL)
def test_every_non_terminal_reaches_every_terminal(status):
    """THE fix for S0-1: no state can be a trap."""
    reach = reachable_from(status)
    for end in TERMINAL:
        assert end in reach, f"{status.value} cannot reach {end.value}"


@pytest.mark.parametrize("status", NON_TERMINAL)
def test_every_non_terminal_can_stop_immediately(status):
    """Giving up or escalating is always one step away — never a multi-hop dance."""
    assert can_transition(status, CaseStatus.STOPPED)
    assert can_transition(status, CaseStatus.ESCALATED)
    assert can_transition(status, CaseStatus.RECOVERED)


def test_waiting_states_are_distinct():
    """The sensor polls one, the scheduler wakes the other (S1-1)."""
    assert WAITING == {CaseStatus.AWAITING_CUSTOMER, CaseStatus.SCHEDULED}
    assert is_waiting(CaseStatus.AWAITING_CUSTOMER)
    assert is_waiting(CaseStatus.SCHEDULED)
    assert not is_waiting(CaseStatus.ACTING)
    assert not is_waiting(CaseStatus.RECOVERED)


def test_work_can_cycle_between_acting_and_waiting():
    """A case may act, wait, act again — dunning is not one-shot."""
    assert can_transition(CaseStatus.ACTING, CaseStatus.AWAITING_CUSTOMER)
    assert can_transition(CaseStatus.AWAITING_CUSTOMER, CaseStatus.ACTING)
    assert can_transition(CaseStatus.ACTING, CaseStatus.SCHEDULED)
    assert can_transition(CaseStatus.SCHEDULED, CaseStatus.ACTING)


def test_assert_transition_raises_with_useful_message():
    with pytest.raises(IllegalTransition, match="terminal"):
        assert_transition(CaseStatus.RECOVERED, CaseStatus.ACTING)
    with pytest.raises(IllegalTransition, match="legal targets"):
        assert_transition(CaseStatus.SCHEDULED, CaseStatus.AWAITING_CUSTOMER)
    assert_transition(CaseStatus.OPEN, CaseStatus.ACTING)  # no raise


def test_missing_evidence_is_catchable_as_illegal_transition():
    err = MissingEvidence(CaseStatus.OPEN, CaseStatus.RECOVERED, "no observation")
    assert isinstance(err, IllegalTransition)
    assert "requires evidence" in str(err)
