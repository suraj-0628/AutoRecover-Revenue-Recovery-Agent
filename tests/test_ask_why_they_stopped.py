"""Ask the customer why, instead of inferring it wrongly.

Two customers abandon after the SAME bank decline and look identical here:

  - one had no balance in the wallet they tried. A discount is useless to
    them — they need time — and a recovery link sent now is a link nobody can
    pay.
  - one found it cheaper elsewhere. A discount is exactly the lever, and
    waiting is a lost sale.

No error code separates those, so guessing loses money in both directions.
The checkout asks one question before any recovery surface, and the answer is
testimony about the customer's own intent — it outranks every inference.
"""
import json
import tempfile
from pathlib import Path

import pytest

import recovery_agent.state_store as state_store
from recovery_agent import drop_reasons
from recovery_agent.agent import ladder
from recovery_agent.agent.classify import failure_kind

#: The same bank decline underneath both journeys.
BANK = {"payment_id": "p", "amount": 2499.0, "status": "recovering",
        "failure_code": "BAD_REQUEST_ERROR",
        "failure_reason": "declined by the bank",
        "customer": {"email": "a@b.com", "contact": "9000000000"},
        "ladder": {"page_push": {"at": "x"}}}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    state_store.StateStore.reset_instances()
    yield state_store.StateStore()
    state_store.StateStore.reset_instances()


def said(code, text=""):
    return {**BANK, "drop_reason": {"code": code, "text": text}}


# ── the same failure, opposite handling ─────────────────────────────────────

def test_no_balance_becomes_a_timing_problem():
    assert failure_kind(said("no_balance")) == "funds"
    rungs = [k for k, _ in ladder.rungs_for(said("no_balance"))]
    assert rungs[0] == "silent_retry", "they need a different day, not a message"


def test_a_better_price_elsewhere_becomes_a_price_problem():
    assert failure_kind(said("better_price")) == "dropoff"
    rungs = [k for k, _ in ladder.rungs_for(said("better_price"))]
    # FIRST, not second. This asserted index 1 — the discount sitting behind
    # the drop-off ladder's opening page push — which is how pay_p9c6jv4x1 met
    # a stated price objection with "You left this payment incomplete. Can we
    # help?" and a five-minute wait. The intent in the message below was always
    # right; the position encoded the opposite of it.
    assert rungs.index("offer") == 0, "the discount is the answer here"
    assert "page_push" not in rungs


def test_the_two_diverge_from_an_identical_failure():
    """The whole point: same error code, opposite ladders."""
    a = [k for k, _ in ladder.rungs_for(said("no_balance"))]
    b = [k for k, _ in ladder.rungs_for(said("better_price"))]
    assert a != b
    assert "silent_retry" in a and "silent_retry" not in b


def test_the_payment_kept_failing_stays_a_method_problem():
    assert failure_kind(said("payment_kept_failing")) == "method"


# ── the discount follows the testimony, not the code ────────────────────────

def _offer(store, code, text=""):
    from recovery_agent.agent.tools import get_recovery_offer
    store.save_payment("p", said(code, text))
    store.flush()
    return json.loads(get_recovery_offer.invoke(
        {"amount": 2499.0, "stage": "ui_offer", "payment_id": "p"}))


def test_no_balance_is_refused_a_discount(store):
    r = _offer(store, "no_balance")
    assert r["allowed"] is False
    assert "short" in r["reason"]
    assert "retry" in r["do_this_instead"].lower()


def test_a_stated_price_objection_gets_the_discount_at_once(store):
    """No need to prove full price fails — they said the price was the problem."""
    r = _offer(store, "better_price")
    assert r["allowed"] is True and r["discount_pct"] > 0


def test_i_will_pay_later_is_not_bought_off(store):
    """Discounting someone who already said they intend to pay buys nothing."""
    r = _offer(store, "will_pay_later")
    assert r["allowed"] is False


def test_an_unanticipated_reason_does_not_unlock_money(store):
    r = _offer(store, "other", "my bank is down until Monday")
    assert r["allowed"] is False, "a human reads this before money moves"


# ── free text goes to a person, not to a keyword matcher ────────────────────

def test_other_is_flagged_for_human_review(store, monkeypatch, tmp_path):
    import recovery_agent.frontend as F
    import recovery_agent.escalation_queue as eq
    monkeypatch.setattr(F, "store", store)
    monkeypatch.setattr(F, "_handoff_to_agent", lambda *a, **k: None)
    monkeypatch.setenv("ESCALATION_QUEUE_PATH", str(tmp_path / "q.jsonl"))
    filed = []
    monkeypatch.setattr(eq, "enqueue",
                        lambda **kw: filed.append(kw) or {"ticket_id": "T1"})

    store.save_payment("p", dict(BANK))
    store.flush()
    resp = F.app.test_client().post("/api/drop-reason", json={
        "payment_id": "p", "code": "other",
        "text": "the site charged me twice last month"})
    assert resp.get_json()["flagged_for_review"] is True
    assert filed and "charged me twice" in filed[0]["reason"]


def test_a_known_reason_is_not_escalated(store, monkeypatch):
    import recovery_agent.frontend as F
    import recovery_agent.escalation_queue as eq
    monkeypatch.setattr(F, "store", store)
    monkeypatch.setattr(F, "_handoff_to_agent", lambda *a, **k: None)
    filed = []
    monkeypatch.setattr(eq, "enqueue", lambda **kw: filed.append(kw))

    store.save_payment("p", dict(BANK))
    store.flush()
    r = F.app.test_client().post("/api/drop-reason",
                                 json={"payment_id": "p", "code": "no_balance"})
    assert r.get_json()["flagged_for_review"] is False
    assert filed == []


# ── the answer reaches the agent ────────────────────────────────────────────

def test_answering_hands_the_case_straight_to_the_agent(store, monkeypatch):
    import recovery_agent.frontend as F
    monkeypatch.setattr(F, "store", store)
    handoffs = []
    monkeypatch.setattr(F, "_handoff_to_agent",
                        lambda pid, obs, scenario: handoffs.append((obs, scenario)))
    store.save_payment("p", dict(BANK))
    store.flush()
    F.app.test_client().post("/api/drop-reason",
                             json={"payment_id": "p", "code": "no_balance"})
    obs, scenario = handoffs[0]
    assert scenario == "stated_reason:no_balance"
    assert "TOLD YOU WHY" in obs
    assert "does not put money in their account" in obs


def test_the_briefing_presents_it_as_testimony_not_a_guess():
    line = drop_reasons.briefing_line({"code": "better_price"})
    assert "testimony" in line and "outranks" in line


def test_free_text_is_quoted_back_verbatim():
    line = drop_reasons.briefing_line({"code": "other", "text": "bank is down"})
    assert '"bank is down"' in line
    assert "Do NOT pattern-match" in line


def test_an_unknown_code_is_refused(store, monkeypatch):
    import recovery_agent.frontend as F
    monkeypatch.setattr(F, "store", store)
    store.save_payment("p", dict(BANK)); store.flush()
    r = F.app.test_client().post("/api/drop-reason",
                                 json={"payment_id": "p", "code": "made_up"})
    assert r.status_code == 400


# ── every answer is actionable ──────────────────────────────────────────────

def test_every_reason_says_what_it_means_and_what_to_do():
    for r in drop_reasons.REASONS:
        assert r["means"] and r["do"], f"{r['code']} is not actionable"
        assert r["kind"] in ("funds", "method", "dropoff", "")


def test_the_checkout_asks_before_it_pushes():
    """The question must come BEFORE the recovery notification — asking after
    we have already nudged them is asking a question we did not wait for."""
    import recovery_agent.frontend as F
    text = Path(F.__file__).read_text()
    i = text.index("if (_closedAfterFailure) {")
    body = text[i:i + 1200]
    assert "askWhyTheyStopped();" in body
    assert body.index("askWhyTheyStopped();") < body.index("return;")
    assert "_flushHeldPush" not in body.split("askWhyTheyStopped();")[0], \
        "the held notification must wait for the answer"


def test_skipping_is_always_allowed():
    """A customer who does not want to answer must not be trapped."""
    import recovery_agent.frontend as F
    text = Path(F.__file__).read_text()
    assert 'id="why-skip"' in text
    assert 'getElementById("why-skip").onclick' in text
    # Skip means "no answer", never a recorded one.
    assert "done(false)" in text


def test_no_answer_still_releases_the_agent():
    """The question now HOLDS the agent, so every way out of it has to let
    the agent go: skipping, the timeout, an empty choice list, and a failed
    fetch. A case that waits forever for an answer nobody is coming back to
    give is worse than one that guessed."""
    import recovery_agent.frontend as F
    text = Path(F.__file__).read_text()
    assert "/api/drop-reason/skip" in text
    # released from the card, and from the paths where it never rendered
    assert text.count("release()") >= 3
    assert "setTimeout(function () { done(false); }, 120000)" in text


def test_the_agent_is_held_until_the_customer_answers():
    """The question and the agent used to race, and the agent won: on
    pay_96fxy62mc it minted a link, gave 5% away and emailed the offer
    BEFORE the customer said "I didn't have enough balance" — the one answer
    for which a discount is useless."""
    import recovery_agent.frontend as F
    text = Path(F.__file__).read_text()
    # the checkout asks for the hold on the cancel-after-failure path
    assert 'source:"customer", step:"payment_processing"}, true);' in text
    # and the server honours it instead of starting the agent
    assert 'if bool(data.get("defer_agent")):' in text
    assert '"status": "awaiting_reason"' in text


# ── a repeated call must not lose the reason it was refused for ─────────────

def test_a_repeat_after_a_transient_refusal_still_names_the_real_reason(store):
    """pay_l477mhjef: the link tool was refused because the customer was
    mid-payment — a transient condition whose refusal said "let them finish".
    The agent re-proposed it and the repetition guard answered "you already
    made this exact call", replacing an actionable reason with one the agent
    could not use. Its summary then said the block was because it "had already
    made the request" — the real cause was gone."""
    from types import SimpleNamespace
    from langchain_core.messages import AIMessage
    from recovery_agent.agent.graph import tool_repetition_guard
    from recovery_agent.agent.tools import RecoveryContext
    from recovery_agent.models import Case, PaymentEvent

    store.save_payment("p", {
        **BANK,
        "refusals": {"creating a payment link: the customer was mid-payment": 1},
    })
    store.flush()

    call = {"name": "generate_recovery_payment_link",
            "args": {"payment_id": "p", "amount": 2374.05}, "id": "c1"}
    case = Case(payment=PaymentEvent(payment_id="p", customer_id="c",
                                     amount=2499.0))
    ctx = RecoveryContext(case=case, guardrail_engine=None)
    config = {"configurable": {"__pregel_runtime": SimpleNamespace(context=ctx)}}

    from recovery_agent.agent.graph import _hash_tool_args
    state = {"messages": [AIMessage(content="", tool_calls=[call])],
             "tool_call_history": [{"name": call["name"],
                                    "args_hash": _hash_tool_args(call["args"])}]}
    out = tool_repetition_guard(state, config)

    body = json.loads(out["messages"][0].content)
    assert body["status"] == "blocked"
    assert "mid-payment" in body["reason"], \
        "the agent must be told what actually blocks it, not just that it repeated"


def test_every_way_of_walking_away_asks_first():
    """There are two ways out of the checkout and only one of them asked.

    A dismissal after a failed attempt was handled; a plain walk-away was
    not — which had it backwards, because a plain dismissal carries NO error
    code at all, so the customer's answer is the only evidence that will ever
    exist. Live (pay_kaagj53tv), someone closed the modal and got a "Complete
    your order" push before being asked anything.
    """
    import re
    import recovery_agent.frontend as F
    page = Path(F.__file__).read_text()
    page = page[page.index('PAY_PAGE = r"""'):]

    calls = []
    for m in re.finditer(r"triggerRecovery\(\{", page):
        seg = page[m.start():m.start() + 340]
        calls.append(" ".join(seg[:seg.find(");") + 2].split()))

    leaving = [c for c in calls if "customer_cancelled" in c]
    assert len(leaving) == 2, "both exits from the checkout must be covered"
    for c in leaving:
        assert c.rstrip().endswith(", true);"), f"does not hold the agent: {c}"


def test_a_customer_actively_switching_rails_is_not_interrogated():
    """Still-trying and still-present are not the same thing.

    This rule used to cover the gateway failure too, on the reasoning that a
    decline the customer is looking at is a live signal worth acting on. In
    practice that let the agent reason to a conclusion and commit to a page
    push while they were still inside the Razorpay modal — a push that renders
    under the iframe (pay_glpfpyq90) — and it did so without the one piece of
    evidence that separates "no balance" from "found it cheaper", because
    nobody had been asked yet. Holding the notification did not help: the
    DECISION was already made, and a decision taken before the reason is known
    cannot use it.

    So the gateway failure now waits for the close, which is the customer's own
    signal that they are done trying. A METHOD SWITCH still does not wait — a
    customer reaching for UPI themselves is actively paying, and interrupting
    that to ask why they stopped would be both wrong and rude.
    """
    import re
    import recovery_agent.frontend as F
    page = Path(F.__file__).read_text()
    page = page[page.index('PAY_PAGE = r"""'):]

    switches = []
    for m in re.finditer(r"triggerRecovery\(\{", page):
        seg = page[m.start():m.start() + 340]
        call = " ".join(seg[:seg.find(");") + 2].split())
        if "method_switch" in call:
            switches.append(call)

    assert switches, "the method-switch path still exists"
    for c in switches:
        assert not c.rstrip().endswith(", true);"), f"should not hold: {c[:90]}"
