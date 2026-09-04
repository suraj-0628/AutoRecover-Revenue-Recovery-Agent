"""Memory must close its loop: written on close, perceived at start.

The agent had two memory systems and no guarantee either changed a decision —
langmem tools the model could forget to call, and customer profiles built in
RAM that died with the process. These tests pin the closed loop: every
close_case writes a structured, PII-masked episode; deliveries and outcomes
update the persistent profile; and perception folds both into the briefing as
measured numbers, so memory is something the agent SEES, not something it has
to remember to ask for.
"""
import json

import pytest
from langgraph.store.memory import InMemoryStore

from recovery_agent.agent import graph as graph_mod
from recovery_agent.agent.memory import CustomerMemoryStore
from recovery_agent.agent.perception import as_briefing, ground_truth
from recovery_agent.agent.tools import TOOLS_BY_NAME
from recovery_agent.state_store import StateStore

PID = "pay_mem1"
EMAIL = "mem@test.com"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setattr(graph_mod, "_memory_store", InMemoryStore())
    state_store.StateStore.reset_instances()
    CustomerMemoryStore.reset_live()
    yield
    state_store.StateStore.reset_instances()
    CustomerMemoryStore.reset_live()


def _seed(**over):
    rec = {"payment_id": PID, "amount": 2000.0, "status": "recovering",
           "recovered_amount": 0, "customer": {"email": EMAIL},
           "customer_email": EMAIL, "failure_code": "bank_declined",
           "ladder": {}}
    rec.update(over)
    s = StateStore()
    s.save_payment(PID, rec)
    s.flush()


def _close(outcome="recovered", lesson=""):
    return json.loads(TOOLS_BY_NAME["close_case"].invoke({
        "payment_id": PID, "outcome": outcome,
        "what_happened": "x", "lesson": lesson,
    }))


def _episodes(kind="method"):
    return [i.value for i in
            graph_mod.get_memory_store().search(("recovery", "episodes", kind),
                                                limit=50)]


# ── writing ─────────────────────────────────────────────────────────────

def test_every_closure_writes_an_episode_lesson_or_not():
    _seed(recovered_amount=1900.0,
          contacts=[{"channel": "email", "at": "t"}])
    out = _close()
    assert out["status"] == "closed"
    eps = _episodes()
    assert len(eps) == 1
    e = eps[0]
    assert e["outcome"] == "recovered"
    assert e["failure_kind"] == "method"
    assert e["discount_pct"] == 5.0          # 100 off 2000
    assert e["contacts"] == ["email"]


def test_full_price_episode_records_zero_discount():
    _seed(recovered_amount=2000.0)
    _close()
    assert _episodes()[0]["discount_pct"] == 0.0


def test_the_lesson_is_masked_before_it_outlives_the_case():
    _seed(recovered_amount=2000.0)
    _close(lesson=f"customer {EMAIL} answers fast, phone 98765 43210")
    e = _episodes()[0]
    assert EMAIL not in e["lesson"]
    assert "98765 43210" not in e["lesson"]


def test_closure_updates_the_customer_profile():
    _seed(recovered_amount=1900.0,
          contacts=[{"channel": "email", "at": "t"}])
    _close()
    profile = CustomerMemoryStore.live().get_or_create_profile(EMAIL)
    assert profile.total_attempts == 1
    assert any(p.status == "success" and p.channel_used == "email"
               for p in profile.payment_history)


def test_a_broken_memory_store_never_blocks_a_closure(monkeypatch):
    _seed(recovered_amount=2000.0)
    def _boom():
        raise RuntimeError("store down")
    monkeypatch.setattr(graph_mod, "get_memory_store", _boom)
    assert _close()["status"] == "closed"


# ── perceiving ──────────────────────────────────────────────────────────

def test_briefing_carries_the_customer_channel_record():
    _seed()
    CustomerMemoryStore.live().update_profile_after_attempt(
        EMAIL, attempt={"payment_id": "pay_old", "amount": 900}, success=True,
        channel="email")
    briefing = as_briefing(ground_truth(PID))
    assert "WHAT HAS WORKED BEFORE" in briefing
    assert "email recovered 1/1" in briefing


def test_briefing_carries_similar_case_outcomes():
    _seed()
    graph_mod.get_memory_store().put(
        ("recovery", "episodes", "method"), "e1",
        {"failure_kind": "method", "outcome": "recovered", "discount_pct": 0.0,
         "payment_id": "pay_other"})
    graph_mod.get_memory_store().put(
        ("recovery", "episodes", "method"), "e2",
        {"failure_kind": "method", "outcome": "escalated", "discount_pct": 0.0,
         "payment_id": "pay_other2"})
    briefing = as_briefing(ground_truth(PID))
    assert "of 2 past method case(s), 1 recovered" in briefing
    assert "1 at full price" in briefing


def test_this_cases_own_episode_is_not_its_own_evidence():
    _seed()
    graph_mod.get_memory_store().put(
        ("recovery", "episodes", "method"), "e1",
        {"failure_kind": "method", "outcome": "recovered", "discount_pct": 0.0,
         "payment_id": PID})
    assert "history" not in ground_truth(PID)


def test_no_memory_means_no_memory_lines():
    """A briefing line built on zero data is noise wearing a memory's
    clothes — with nothing learned, the section must be absent."""
    _seed()
    assert "WHAT HAS WORKED BEFORE" not in as_briefing(ground_truth(PID))


def test_memory_changes_the_briefing_deterministically():
    """The A/B the eval replays at model level, pinned here at briefing level:
    same case, memory on vs off, different text in front of the model."""
    _seed()
    without = as_briefing(ground_truth(PID))
    graph_mod.get_memory_store().put(
        ("recovery", "episodes", "method"), "e1",
        {"failure_kind": "method", "outcome": "recovered", "discount_pct": 0.0,
         "payment_id": "pay_other"})
    with_mem = as_briefing(ground_truth(PID))
    assert without != with_mem
    assert "WHAT HAS WORKED BEFORE" in with_mem


def test_settled_cases_do_not_advertise_history():
    _seed(recovered_amount=2000.0, status="recovered")
    graph_mod.get_memory_store().put(
        ("recovery", "episodes", "method"), "e1",
        {"failure_kind": "method", "outcome": "recovered", "discount_pct": 0.0,
         "payment_id": "pay_other"})
    briefing = as_briefing(ground_truth(PID))
    assert "WHAT HAS WORKED BEFORE" not in briefing
    assert "SETTLED" in briefing
