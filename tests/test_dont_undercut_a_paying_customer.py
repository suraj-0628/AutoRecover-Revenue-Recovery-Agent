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
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    _case(pending_push={"sent_at": now},
          push_outcome={"action": "acted", "at": now})
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


# ── "mid-payment" is a window, not a state ──────────────────────────────

def test_an_old_click_does_not_freeze_the_case():
    """Live: the customer clicked, their bank declined, and for the rest of the
    case every action was refused with "the customer is in the middle of paying"
    — three runs in a row, blocked on a click that had already failed."""
    from datetime import datetime, timedelta, timezone
    old = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    _case(pending_push={"sent_at": old},
          push_outcome={"action": "acted", "at": old})
    r = _call("generate_recovery_payment_link", amount=2374.05,
              customer_email="a@b.com")
    assert "middle of paying" not in str(r.get("reason", ""))


def test_a_new_failure_clears_the_previous_click():
    """A new failure means the previous attempt is over."""
    FRONTEND = (pathlib.Path(__file__).resolve().parents[1] / "src"
                / "recovery_agent" / "frontend.py").read_text()
    i = FRONTEND.index("def payment_failed")
    # Window sized to hold the route body; the signal-precedence block sits
    # between the def and the push_outcome reset it pins.
    # Function-bounded, not byte-bounded — see the note in
    # test_bank_decline_is_not_a_dropoff for why the fixed window was wrong.
    body = FRONTEND[i:FRONTEND.index("\n@app.route", i)]
    assert "push_outcome=None" in body
    assert "failure_code=failure_code" in body, (
        "the code was never stored, so the record said failure_code: None and "
        "a bank decline survived only as prose in a message"
    )


# ── the failure KIND decides which lever is relevant ────────────────────

def test_a_bank_decline_is_named_as_a_method_problem():
    """The agent reached for a 5% discount on a bank decline while the
    customer's own history showed netbanking succeeding 2 times out of 2. Money
    off does not fix plumbing."""
    from recovery_agent.agent.perception import as_briefing, ground_truth
    _case(failure_code="bank_declined",
          failure_reason="declined by the bank")
    b = as_briefing(ground_truth("p"))
    assert "failed at the BANK, not on price" in b
    assert "get_customer_payment_history" in b


def test_a_cancellation_is_named_as_a_choice_not_a_fault():
    from recovery_agent.agent.perception import as_briefing, ground_truth
    _case(failure_code="customer_cancelled", failure_reason="cancelled")
    assert "chose not to complete" in as_briefing(ground_truth("p"))


# ── a refused plan is remembered into the next turn ─────────────────────

def test_repeated_refusals_are_surfaced_to_the_next_run():
    """`tool_call_history` resets each run, so a plan refused in one run looked
    brand new in the next — the agent proposed the same blocked link three runs
    running, and said so itself."""
    from recovery_agent.agent.perception import as_briefing, ground_truth
    _case(status="recovered", recovered_amount=2499.0)
    for _ in range(2):
        _call("generate_recovery_payment_link", amount=2374.05,
              customer_email="a@b.com")
    b = as_briefing(ground_truth("p"))
    assert "already been refused" in b
    assert "has to change" in b


# ── a discount is a lever for reluctance, not for a refused instrument ──

def test_no_discount_is_offered_for_a_bank_decline():
    """The agent switched to the customer's working rail — correctly — and then
    discounted it anyway, giving away INR 124.95 to solve a problem that was
    never about price. get_recovery_offer answered "what is the maximum
    allowed?" when the question was "what is warranted here?"."""
    _case(amount=2499.0, failure_code="bank_declined",
          failure_reason="declined by the bank",
          actions_tried=["page_push:plain"])
    r = _call("get_recovery_offer", amount=2499.0, stage="ui_offer")
    assert r["status"] == "not_indicated"
    assert r["allowed"] is False
    assert "does not make a declined instrument work" in r["reason"]
    assert "FULL INR 2,499.00" in r["do_this_instead"]


def test_the_discount_returns_once_a_full_price_rail_switch_has_failed():
    """If a working rail also fails, the reluctance is real and price is back on
    the table."""
    _case(amount=2499.0, failure_code="bank_declined",
          failure_reason="declined by the bank",
          actions_tried=["page_push:plain", "link:netbanking:2499.00"])
    r = _call("get_recovery_offer", amount=2499.0, stage="ui_offer")
    assert r["status"] == "ok" and r["discount_pct"] == 5.0


def test_a_cancellation_still_gets_an_offer_immediately():
    """Someone who chose not to pay IS a price problem — do not over-correct."""
    _case(amount=2499.0, failure_code="customer_cancelled",
          failure_reason="cancelled", actions_tried=["page_push:plain"])
    r = _call("get_recovery_offer", amount=2499.0, stage="ui_offer")
    assert r["status"] == "ok" and r["discount_pct"] == 5.0


def test_the_offer_tool_still_works_without_a_payment_id():
    """It must stay usable for a case that has no record yet."""
    import json as _json
    from recovery_agent.agent.tools import TOOLS_BY_NAME
    r = _json.loads(TOOLS_BY_NAME["get_recovery_offer"].invoke(
        {"amount": 2499.0, "stage": "ui_offer"}))
    assert r["status"] in ("ok", "no_data")
