"""The agent must know when to stop, not merely be prevented from continuing.

Withholding tools from a settled case stops the damage without addressing the
cause: the agent had no perception, only narration. Everything it "knew" arrived
as prose the orchestrator chose to write into a message, so it stopped when told
"RECOVERED" and chased a paid customer whenever it was not told.

The tool that could have answered "has this been paid?" ended with
`"do not call this tool again for this case"`. We taught it that checking was
pointless, then caged it for acting on what it therefore could not know.
"""
from pathlib import Path

import pytest

from recovery_agent.agent.perception import as_briefing, ground_truth
from recovery_agent.state_store import StateStore

GRAPH = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent" / "agent"
         / "graph.py").read_text()
TOOLS = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent" / "agent"
         / "tools.py").read_text()


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """`ground_truth` builds its own StateStore, which reads the module default.
    Point that at a temp directory or these tests read the real data/."""
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    state_store.StateStore.reset_instances()
    yield
    state_store.StateStore.reset_instances()


def _case(tmp_path, **over):
    rec = {"payment_id": "p", "amount": 2499.0, "status": "recovering",
           "recovered_amount": 0, "customer": {"email": "a@b.com"}, "ladder": {}}
    rec.update(over)
    s = StateStore(tmp_path)
    s.save_payment("p", rec)
    s.flush()
    return s.get_payment("p")


# ── the facts ───────────────────────────────────────────────────────────

def test_money_received_settles_the_case(tmp_path):
    _case(tmp_path, status="recovered", recovered_amount=2374.05,
          recovered_payment_id="pay_X")
    f = ground_truth("p")
    assert f["settled"] and f["outstanding"] == 0.0


def test_a_discounted_recovery_leaves_nothing_outstanding(tmp_path):
    """An authorised offer means the reduced figure IS what is owed now."""
    _case(tmp_path, status="recovered", recovered_amount=2374.05)
    assert ground_truth("p")["outstanding"] == 0.0


def test_an_amount_received_settles_it_even_if_the_status_lags(tmp_path):
    """A status is something a writer chose; an amount is something that
    happened. Two live cases had a real recovery rewritten to `failed`."""
    _case(tmp_path, status="failed", recovered_amount=2374.05)
    assert ground_truth("p")["settled"]


def test_an_open_case_reads_as_open(tmp_path):
    _case(tmp_path)
    f = ground_truth("p")
    assert not f["settled"] and f["outstanding"] == 2499.0


def test_an_unknown_payment_does_not_claim_to_know(tmp_path):
    StateStore(tmp_path)
    assert ground_truth("nope")["known"] is False


# ── the briefing ────────────────────────────────────────────────────────

def test_the_briefing_tells_a_settled_case_to_stop(tmp_path):
    _case(tmp_path, status="recovered", recovered_amount=2374.05,
          recovered_payment_id="pay_X")
    b = as_briefing(ground_truth("p"))
    assert "SETTLED" in b and "stop" in b
    assert "already paid you" in b


def test_the_briefing_tells_an_open_case_what_is_left(tmp_path):
    _case(tmp_path, ladder={"page_push": {"at": "x"}},
          actions_tried=["page_push:plain"])
    b = as_briefing(ground_truth("p"))
    assert "NOT back" in b
    assert "Already tried: page_push:plain" in b
    assert "Do not repeat" in b


def test_the_briefing_says_when_escalation_is_finally_allowed(tmp_path):
    _case(tmp_path, customer={}, customer_email="", ladder={"page_push": {"at": "x"}})
    assert "escalate_to_human will accept" in as_briefing(ground_truth("p"))


# ── perception is part of the loop, not something to remember ───────────

def test_the_briefing_is_injected_on_every_turn():
    i = GRAPH.index("PERCEPTION — recomputed every turn")
    body = GRAPH[i:i + 1600]
    assert "as_briefing(ground_truth(pid))" in body
    assert "SystemMessage" in body


def test_perception_failing_does_not_stop_the_agent_working():
    i = GRAPH.index("PERCEPTION — recomputed every turn")
    body = GRAPH[i:i + 1600]
    assert "except Exception:" in body


def test_the_prompt_points_at_the_facts_over_the_narrative():
    i = GRAPH.index("WHAT IS TRUE RIGHT NOW")
    body = GRAPH[i:i + 900]
    assert "not the narrative below" in body
    assert "Checking is never wasted" in body


# ── checking must always be worth it ────────────────────────────────────

def test_check_payment_status_answers_whether_the_case_is_settled():
    i = TOOLS.index("def check_payment_status")
    body = TOOLS[i:i + 3000]
    assert "ground_truth(payment_id, verify=True)" in body


def test_the_agent_is_never_told_to_stop_checking():
    """One line — "do not call this tool again for this case" — is why it could
    not know when to stop."""
    i = TOOLS.index("def check_payment_status")
    body = TOOLS[i:i + 3000]
    assert "do not call this tool again" not in body
    assert "still worth calling later" in body
