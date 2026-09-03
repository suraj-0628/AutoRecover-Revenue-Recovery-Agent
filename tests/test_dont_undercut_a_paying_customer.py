"""Do not discount a customer who is already paying you.

`pay_sjqofnsp6` (INR 99,975):

    04:16:46  push sent
    04:16:51  customer ACTED on it
    04:16:54  agent fetches history
    04:16:55  agent gets a 5% offer
    04:17:02  agent creates a DISCOUNTED link for INR 94,976.25
    04:17:08  customer pays INR 99,975.00 — full price

All of it inside one turn, so nothing had a chance to re-brief the agent. It
offered INR 4,998.75 off to someone already paying in full, and spent one of the
account's thirty lifetime payment links doing it.
"""
import json
import pathlib
import tempfile

import pytest

GRAPH = (pathlib.Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
         / "agent" / "graph.py").read_text()


@pytest.fixture(autouse=True)
def _store(monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR",
                        pathlib.Path(tempfile.mkdtemp()))
    state_store.StateStore.reset_instances()
    yield
    state_store.StateStore.reset_instances()


def _case(**over):
    from recovery_agent.state_store import StateStore
    rec = {"payment_id": "p", "amount": 99975.0, "status": "recovering",
           "recovered_amount": 0, "customer": {"email": "a@b.com"},
           "customer_email": "a@b.com", "ladder": {}}
    rec.update(over)
    s = StateStore()
    s.save_payment("p", rec)
    s.flush()


def _call(name, **args):
    from recovery_agent.agent.tools import TOOLS_BY_NAME
    return json.loads(TOOLS_BY_NAME[name].invoke({"payment_id": "p", **args}))


# ── mid-payment ─────────────────────────────────────────────────────────

def test_no_discounted_link_while_the_customer_is_completing():
    _case(pending_push={"sent_at": "2026-09-03T04:16:46Z"},
          push_outcome={"action": "acted", "at": "2026-09-03T04:16:51Z"})
    r = _call("generate_recovery_payment_link", amount=94976.25,
              customer_email="a@b.com")
    assert r["status"] == "blocked"
    assert "middle of paying" in r["reason"]


def test_a_dismissal_does_not_block_the_next_rung():
    """Only *acting* means they are completing. A dismissal is the signal to
    move on, and must not freeze the ladder."""
    _case(pending_push={"sent_at": "x"},
          push_outcome={"action": "dismissed", "at": "y"})
    r = _call("generate_recovery_payment_link", amount=94976.25,
              customer_email="a@b.com")
    assert r["status"] != "blocked"


# ── already settled ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name,args", [
    ("generate_recovery_payment_link", {"amount": 94976.25,
                                        "customer_email": "a@b.com"}),
    ("show_page_offer", {"headline": "5% off", "body": "x",
                         "payable_amount": 94976.25,
                         "payment_link": "https://rzp.io/x"}),
    ("send_recovery_notification", {"message": "pay now",
                                    "customer_email": "a@b.com",
                                    "customer_phone": "9000000000"}),
    ("initiate_voice_call", {"customer_name": "A",
                             "customer_phone": "9000000000",
                             "amount": 99975.0}),
])
def test_nothing_reaches_a_customer_who_has_already_paid(name, args):
    _case(status="recovered", recovered_amount=99975.0,
          recovered_payment_id="pay_X")
    r = _call(name, **args)
    assert r["status"] == "blocked"
    assert "already settled" in r["reason"]
    assert "close_case" in r["guidance"]


# ── re-checking is not repetition ───────────────────────────────────────

def test_check_payment_status_is_exempt_from_the_repetition_guard():
    """Asking again whether the money arrived is the point — the answer changes.
    Blocking it taught the agent that checking was a mistake, which is the habit
    that lets it spend on a customer who has already paid."""
    assert '_RECHECKABLE = {"check_payment_status"}' in GRAPH
    i = GRAPH.index("Layer 1: Exact repetition")
    assert "if tool_name in _RECHECKABLE:" in GRAPH[i:i + 1200]


def test_the_prompt_tells_it_to_recheck_before_spending():
    i = GRAPH.index("The ONE exception is check_payment_status")
    body = GRAPH[i:i + 500]
    assert "before you spend money or contact the customer again" in body
