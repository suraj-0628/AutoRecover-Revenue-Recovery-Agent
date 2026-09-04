"""The choke points that feed the audit trail.

A log is only as good as the places that write to it, and every one of these
writes sits inside a `try` that swallows its own failure — correctly, because a
customer must not go without their payment link because the logging broke. The
cost of that choice is that a hook can stop working in total silence. These
tests are what makes it not silent.
"""
import json

import pytest

from recovery_agent import audit


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path, raising=False)
    state_store.StateStore.reset_instances()
    audit.AuditLog.reset_instances()
    audit._default = None
    s = state_store.StateStore()
    yield s
    state_store.StateStore.reset_instances()
    audit.AuditLog.reset_instances()
    audit._default = None


def case(store, **kw):
    rec = {"payment_id": "pay_1", "amount": 2999.0, "status": "failed",
           "decline_strategy": "card_expired",
           "customer": {"email": "a@b.com", "name": "A"}, **kw}
    store.save_payment(rec["payment_id"], rec)
    store.flush()
    return rec


# ── the ladder ───────────────────────────────────────────────────────────

def test_a_rung_climbed_is_written_to_the_log(store):
    """An escalation ticket later claims 'already tried: page_push, offer' on
    the strength of the record's ladder dict. The same fact goes somewhere it
    cannot be quietly rewritten."""
    from recovery_agent.agent import ladder
    case(store)
    ladder.record_rung("pay_1", "offer", "emailed a link at full price")

    events = [e for e in audit.log().for_payment("pay_1")
              if e["kind"] == audit.LADDER_RUNG]
    assert len(events) == 1
    assert events[0]["action"] == "offer"
    assert "full price" in events[0]["reason"]


def test_climbing_the_same_rung_twice_logs_once(store):
    """`record_rung` is idempotent by design; the log must not claim two
    contacts where the record shows one."""
    from recovery_agent.agent import ladder
    case(store)
    ladder.record_rung("pay_1", "offer")
    ladder.record_rung("pay_1", "offer")
    assert len([e for e in audit.log().for_payment("pay_1")
                if e["kind"] == audit.LADDER_RUNG]) == 1


def test_a_rung_climbed_inside_a_batch_carries_the_run(store):
    from recovery_agent.agent import ladder
    case(store, batch_run_id="run_x")
    ladder.record_rung("pay_1", "offer")
    assert audit.log().for_run("run_x")[0]["kind"] == audit.LADDER_RUNG


# ── escalation ───────────────────────────────────────────────────────────

def test_an_escalation_records_who_and_on_what_grounds(store, monkeypatch,
                                                       tmp_path):
    """Compliant escalation has to be demonstrable, not asserted: the log shows
    the escalation came at the end of the ladder rather than instead of it."""
    from recovery_agent import escalation_queue
    monkeypatch.setattr(escalation_queue, "_QUEUE_PATH",
                        tmp_path / "escalations" / "queue.jsonl")
    case(store, ladder={"offer": {"at": "2026-09-04T10:00:00+00:00"},
                        "voice_call": {"at": "2026-09-04T10:30:00+00:00"}})

    escalation_queue.enqueue("pay_1", "the ladder is exhausted", amount=2999.0,
                             customer_signals=["dismissed the offer in 4s"])
    events = [e for e in audit.log().for_payment("pay_1")
              if e["kind"] == audit.ESCALATION_RAISED]
    assert len(events) == 1
    assert events[0]["reason"] == "the ladder is exhausted"
    assert events[0]["amount_paise"] == 299900
    assert events[0]["payload"]["climbed"] == ["offer", "voice_call"]


def test_a_duplicate_escalation_does_not_double_log(store, monkeypatch,
                                                    tmp_path):
    from recovery_agent import escalation_queue
    monkeypatch.setattr(escalation_queue, "_QUEUE_PATH",
                        tmp_path / "escalations" / "queue.jsonl")
    case(store)
    for _ in range(3):
        escalation_queue.enqueue("pay_1", "exhausted", amount=2999.0)
    assert len([e for e in audit.log().for_payment("pay_1")
                if e["kind"] == audit.ESCALATION_RAISED]) == 1


# ── the closure ──────────────────────────────────────────────────────────

def test_a_closed_case_records_the_outcome_and_the_figure(store, monkeypatch):
    """The stopping rule, in the record: what the agent decided, why, and how
    much actually came back — not the amount that was owed."""
    from recovery_agent.agent import tools
    case(store, status="recovered", recovered_amount=2699.0,
         recovered_payment_id="pay_cap_1")
    monkeypatch.setattr(tools, "_still_worth_doing", lambda *a, **k: "",
                        raising=False)

    tools.TOOLS_BY_NAME["close_case"].func(
        payment_id="pay_1", outcome="recovered",
        what_happened="They paid the discounted link.", runtime=None)

    events = [e for e in audit.log().for_payment("pay_1")
              if e["kind"] == audit.CASE_CLOSED]
    assert len(events) == 1
    assert events[0]["result"] == "recovered"
    assert events[0]["amount_paise"] == 269900, "the figure, not the debt"


# ── the money join, which the whole batch report rests on ────────────────

def test_recovered_money_carries_the_run_that_earned_it(store, monkeypatch):
    """Attribution is a join, not a timestamp heuristic. The executor stamps the
    run on the record; this is the other half."""
    from recovery_agent import frontend
    monkeypatch.setattr(frontend, "store", store, raising=False)
    monkeypatch.setattr(frontend, "push_event", lambda *a, **k: None)
    monkeypatch.setattr(frontend.socketio, "emit", lambda *a, **k: None)
    monkeypatch.setattr(frontend, "_notify_agent_of_recovery",
                        lambda *a, **k: None)
    case(store, batch_run_id="run_x")

    assert frontend._mark_recovered("pay_1", 2699.0, "pay_cap_1", 42, "link paid")
    events = [e for e in audit.log().for_run("run_x")
              if e["kind"] == audit.MONEY_RECOVERED]
    assert len(events) == 1
    assert events[0]["amount_paise"] == 269900
    assert events[0]["payment_id"] == "pay_1"


def test_a_case_recovered_outside_a_batch_counts_against_no_run(store,
                                                                monkeypatch):
    from recovery_agent import frontend
    monkeypatch.setattr(frontend, "store", store, raising=False)
    monkeypatch.setattr(frontend, "push_event", lambda *a, **k: None)
    monkeypatch.setattr(frontend.socketio, "emit", lambda *a, **k: None)
    monkeypatch.setattr(frontend, "_notify_agent_of_recovery",
                        lambda *a, **k: None)
    case(store)

    frontend._mark_recovered("pay_1", 2999.0, "pay_cap_1", 10, "live path")
    event = [e for e in audit.log().for_payment("pay_1")
             if e["kind"] == audit.MONEY_RECOVERED][0]
    assert event["batch_run_id"] == ""


def test_the_same_recovery_reported_twice_is_logged_once(store, monkeypatch):
    """A poller and a browser callback racing on one payment must not be able
    to make the recovered total say twice what arrived."""
    from recovery_agent import frontend
    monkeypatch.setattr(frontend, "store", store, raising=False)
    monkeypatch.setattr(frontend, "push_event", lambda *a, **k: None)
    monkeypatch.setattr(frontend.socketio, "emit", lambda *a, **k: None)
    monkeypatch.setattr(frontend, "_notify_agent_of_recovery",
                        lambda *a, **k: None)
    case(store, batch_run_id="run_x")

    assert frontend._mark_recovered("pay_1", 2699.0, "cap_1", 1, "poller")
    assert not frontend._mark_recovered("pay_1", 2699.0, "cap_1", 1, "browser")
    assert audit.log().recovered_paise("run_x") == 269900
