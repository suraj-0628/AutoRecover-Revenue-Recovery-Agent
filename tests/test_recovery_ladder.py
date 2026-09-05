"""Escalation is rung 6, not the answer to a blocked tool.

`pay_rxfo0unaq`: the approval gate refused a discounted link at rung 2 and the
agent filed a human ticket for a INR 1,19,970 order that had been contacted
exactly once — one silent in-page push. No email, no call, no offer. A human
queue is the last resort; it had become the fallback for any refusal.
"""
import os
from datetime import datetime, timedelta, timezone

import pytest

from recovery_agent.agent import ladder


def case(**over) -> dict:
    rec = {
        "payment_id": "pay_t", "amount": 2499.0, "failure_code": "customer_cancelled",
        "customer": {"email": "a@b.com", "name": "A", "contact": "9000000000"},
        "ladder": {},
    }
    rec.update(over)
    return rec


def climb(rec: dict, *rungs: str) -> dict:
    rec["ladder"] = dict(rec.get("ladder") or {})
    for r in rungs:
        rec["ladder"][r] = {"at": datetime.now(timezone.utc).isoformat(), "detail": ""}
    return rec


# ── order ───────────────────────────────────────────────────────────────

def test_the_first_rung_is_the_silent_push_while_the_page_is_open():
    """page_push only exists while the customer is on the checkout."""
    from recovery_agent import presence
    presence.reset()
    presence.watch("sid-1", "pay_t")
    try:
        assert ladder.next_rung(case(payment_id="pay_t"))["rung"] == "page_push"
    finally:
        presence.reset()


def test_the_ladder_starts_at_the_offer_once_the_customer_has_left():
    """Which is every batch case. Claiming page_push for someone who closed the
    tab hours ago advances the ladder past a contact that never happened."""
    from recovery_agent import presence
    presence.reset()
    assert ladder.next_rung(case(payment_id="pay_t"))["rung"] == "offer"
    unavailable = {u["rung"] for u in ladder.state(case(payment_id="pay_t"))["unavailable"]}
    assert "page_push" in unavailable


def test_the_offer_follows_the_push():
    assert ladder.next_rung(climb(case(), "page_push"))["rung"] == "offer"


def test_nothing_is_left_once_every_possible_rung_is_climbed(monkeypatch):
    monkeypatch.setenv("VOICE_CALLS_ENABLED", "1")
    rec = climb(case(), "page_push", "offer", "voice_call", "post_call_email",
                "alternate_path")
    assert ladder.exhausted(rec)
    assert ladder.next_rung(rec) is None


# ── availability, not silent completion ─────────────────────────────────

def test_voice_is_unavailable_rather_than_done_when_calls_are_off(monkeypatch):
    monkeypatch.delenv("VOICE_CALLS_ENABLED", raising=False)
    st = ladder.state(climb(case(), "page_push", "offer"))
    unavailable = {r["rung"] for r in st["unavailable"]}
    assert "voice_call" in unavailable, "a switched-off channel is not a climbed rung"
    assert "voice_call" not in st["climbed"]
    assert any("switched off" in r["why_not"] for r in st["unavailable"])


def test_a_call_that_cannot_be_made_does_not_block_escalation_forever(monkeypatch):
    """Otherwise a deployment with voice off could never escalate anything."""
    monkeypatch.delenv("VOICE_CALLS_ENABLED", raising=False)
    rec = climb(case(), "page_push", "offer", "alternate_path")
    assert ladder.exhausted(rec)


def test_a_customer_with_no_contact_details_cannot_be_emailed():
    rec = case(customer={}, customer_email="", customer_phone="")
    st = ladder.state(rec)
    assert {r["rung"] for r in st["unavailable"]} >= {"offer", "alternate_path"}


# ── the 15-minute gap before a call ─────────────────────────────────────

def test_a_call_is_not_allowed_immediately_after_the_offer():
    rec = climb(case(), "page_push", "offer")
    assert ladder.voice_wait_remaining_minutes(rec) > 14


def test_the_call_opens_up_once_the_offer_has_had_its_window():
    rec = case()
    old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    rec["ladder"] = {"page_push": {"at": old}, "offer": {"at": old}}
    assert ladder.voice_wait_remaining_minutes(rec) == 0


def test_no_wait_applies_before_an_offer_has_gone_out():
    assert ladder.voice_wait_remaining_minutes(climb(case(), "page_push")) == 0


# ── the one carve-out ───────────────────────────────────────────────────

@pytest.mark.parametrize("code", ["fraud_suspected", "risk_declined",
                                  "dispute_raised", "chargeback"])
def test_do_not_pursue_cases_skip_the_ladder(code):
    assert ladder.pursuit_barred(case(failure_code=code))


def test_a_dead_card_is_not_a_do_not_pursue_case():
    """The recovery for a dead instrument is another rail, not a human ticket."""
    assert not ladder.pursuit_barred(case(failure_code="card_expired"))
    assert not ladder.pursuit_barred(case(failure_code="customer_cancelled"))


def test_an_opted_out_customer_is_not_chased():
    assert ladder.pursuit_barred(case(opted_out=True))


# ── the tool actually enforces it ───────────────────────────────────────

def test_escalation_is_refused_while_rungs_remain(tmp_path, monkeypatch):
    """A reachable customer with an untried offer must not go to a human."""
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    state_store.StateStore.reset_instances()
    import json
    from recovery_agent.state_store import StateStore
    from recovery_agent.agent.tools import TOOLS_BY_NAME
    st = StateStore()
    st.save_payment("pay_reachable", {"payment_id": "pay_reachable", "amount": 2499.0,
                                      "status": "awaiting_customer",
                                      "customer": {"email": "a@b.com"},
                                      "decline_strategy": "customer_cancelled",
                                      "ladder": {}})
    st.flush()
    out = json.loads(TOOLS_BY_NAME["escalate_to_human"].invoke(
        {"payment_id": "pay_reachable", "reason": "a tool refused me"}))
    state_store.StateStore.reset_instances()
    assert out["status"] == "blocked"
    assert out["next_rung"] == "offer"
    assert "final step" in out["reason"]


def test_a_case_with_nothing_left_to_try_may_escalate(tmp_path, monkeypatch):
    """No record, no contact, no live page — there is genuinely no rung to climb,
    and refusing escalation would strand the case."""
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    state_store.StateStore.reset_instances()
    import json
    from recovery_agent.agent.tools import TOOLS_BY_NAME
    out = json.loads(TOOLS_BY_NAME["escalate_to_human"].invoke(
        {"payment_id": "pay_never_seen", "reason": "nothing is possible here"}))
    state_store.StateStore.reset_instances()
    assert out["status"] == "escalated"


def test_the_refusal_names_what_to_do_instead(tmp_path, monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    state_store.StateStore.reset_instances()
    import json
    from recovery_agent.state_store import StateStore
    from recovery_agent.agent.tools import TOOLS_BY_NAME
    st = StateStore()
    st.save_payment("pay_reach2", {"payment_id": "pay_reach2", "amount": 2499.0,
                                   "status": "awaiting_customer",
                                   "customer": {"email": "a@b.com"},
                                   "decline_strategy": "customer_cancelled",
                                   "ladder": {}})
    st.flush()
    out = json.loads(TOOLS_BY_NAME["escalate_to_human"].invoke(
        {"payment_id": "pay_reach2", "reason": "blocked"}))
    state_store.StateStore.reset_instances()
    assert out["next_step"], "a refusal must name the next move"
    assert "guidance" in out


# ── rung 5 is whatever the agent picks, so long as it is new ────────────

def test_a_new_kind_of_action_after_the_offer_is_the_alternate_path(tmp_path, monkeypatch):
    """There is no fixed tool for rung 5. Which route is worth trying after an
    offer has failed depends on the customer and the failure, and that judgement
    is the agent's — the ledger only notices that it did something new."""
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    from recovery_agent.state_store import StateStore
    s = StateStore(tmp_path)
    s.save_payment("p1", {"payment_id": "p1", "amount": 2499,
                          "customer": {"email": "a@b.com"}, "ladder": {}})
    s.flush()

    ladder.record_rung("p1", "page_push", "")
    ladder.record_action("p1", "page_push:plain", is_rung=True)
    ladder.record_rung("p1", "offer", "")
    ladder.record_action("p1", "notify:email:https://rzp.io/x", is_rung=True)

    rec = StateStore(tmp_path).get_payment("p1")
    assert not ladder.climbed(rec, "alternate_path"), (
        "the offer email is the offer rung, not a different path"
    )

    ladder.record_action("p1", "retry:24.0h")          # agent's own choice
    rec = StateStore(tmp_path).get_payment("p1")
    assert ladder.climbed(rec, "alternate_path")


def test_repeating_an_action_is_not_a_different_path(tmp_path, monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    from recovery_agent.state_store import StateStore
    s = StateStore(tmp_path)
    s.save_payment("p2", {"payment_id": "p2", "amount": 2499,
                          "customer": {"email": "a@b.com"}, "ladder": {}})
    s.flush()
    ladder.record_rung("p2", "page_push", "")
    ladder.record_action("p2", "notify:email:https://rzp.io/x")
    ladder.record_rung("p2", "offer", "")
    ladder.record_action("p2", "notify:email:https://rzp.io/x")   # same again

    rec = StateStore(tmp_path).get_payment("p2")
    assert rec["actions_tried"].count("notify:email:https://rzp.io/x") == 1
    assert not ladder.climbed(rec, "alternate_path")


def test_the_agent_is_told_what_it_has_already_tried():
    """"Do not repeat a channel that failed" is advice the agent cannot act on
    unless it can see the list — and a Case is rebuilt on every hand-off run."""
    GRAPH = (__import__("pathlib").Path(__file__).resolve().parents[1] / "src"
             / "recovery_agent" / "agent" / "graph.py").read_text()
    i = GRAPH.index("def build_initial_state")
    body = GRAPH[i:i + 3000]
    assert "WHERE THIS CASE STANDS ON THE LADDER" in body
    assert "already tried" in body
    assert "do NOT repeat any of these" in body


def test_the_followup_context_no_longer_points_straight_at_escalation():
    GRAPH = (__import__("pathlib").Path(__file__).resolve().parents[1] / "src"
             / "recovery_agent" / "agent" / "graph.py").read_text()
    i = GRAPH.index("FOLLOW-UP: a recovery attempt was ALREADY delivered")
    body = GRAPH[i:i + 900]
    assert "next rung of the" in body
    assert "Otherwise retry_in_hours, or " not in body


# ── the turn cap must not eat the ladder ────────────────────────────────

def test_turns_are_counted_per_run_not_per_session():
    """A session is one continuous thread per payment, so counting every turn
    the case had ever taken meant the cap tripped part-way up the ladder. Live,
    `retry_in_hours` was called and never executed, the next call came back
    "not a valid tool", and a INR 2,499 case went quiet at rung 2 — neither
    recovered nor escalated."""
    from langchain_core.messages import AIMessage, HumanMessage
    from recovery_agent.agent.graph import should_continue

    names = ["search_memory", "diagnose_payment_failure", "query_knowledge_base",
             "get_customer_payment_history", "check_payment_status",
             "discover_recovery_rail", "get_recovery_offer"]

    def turn(n):
        # Distinct tools, or the repetition detector stops the run for a
        # different reason and the test proves nothing about the turn cap.
        return AIMessage(content="", tool_calls=[
            {"name": names[n % len(names)], "args": {"i": n}, "id": f"c{n}"}])

    # Six turns spent on earlier rungs, then a fresh run begins.
    history = [HumanMessage(content="failed")] + [turn(i) for i in range(6)]
    state = {"messages": history + [HumanMessage(content="dismissed"), turn(99)],
             "tool_call_history": [], "blocked_rounds": 0}
    assert should_continue(state) != "stopping_check", (
        "a new run starts with a fresh budget; the case's past turns are not its own"
    )


def test_one_run_that_loops_is_still_capped():
    from langchain_core.messages import AIMessage, HumanMessage
    from recovery_agent.agent.graph import should_continue, MAX_TURNS_PER_RUN

    names = ["search_memory", "diagnose_payment_failure", "query_knowledge_base",
             "get_customer_payment_history", "check_payment_status",
             "discover_recovery_rail", "get_recovery_offer", "send_page_push",
             "show_page_offer"]
    msgs = [HumanMessage(content="failed")]
    # One past the cap, so the single grace round for a finishing call is
    # already spent — a run that keeps looping is stopped regardless.
    for i in range(MAX_TURNS_PER_RUN + 1):
        msgs.append(AIMessage(content="", tool_calls=[
            {"name": names[i % len(names)], "args": {"i": i}, "id": f"x{i}"}]))
    state = {"messages": msgs, "tool_call_history": [], "blocked_rounds": 0}
    assert should_continue(state) == "stopping_check"


# ── the safety net obeys the same rule the agent does ───────────────────

def test_the_frontend_safety_net_will_not_escalate_mid_ladder():
    """The net fired on any run ending `failed`/`stopped`, whichever rung the
    case was on — so it filed a human ticket for a case whose agent had just
    scheduled a 24h retry, walking straight around the gate that exists to stop
    exactly that."""
    FRONTEND = (__import__("pathlib").Path(__file__).resolve().parents[1] / "src"
                / "recovery_agent" / "frontend.py").read_text()
    i = FRONTEND.index('if (final_status in ("escalated", "failed", "stopped")')
    body = FRONTEND[i - 1400:i + 1600]
    assert "_rungs_left" in body, "the net must consult the ladder"
    assert "_handoff_to_agent(" in body, (
        "rungs remaining means hand back to the agent, not file a ticket — "
        "nothing else would restart the case and the alternative is silence"
    )


def test_the_net_still_escalates_once_the_ladder_is_done():
    FRONTEND = (__import__("pathlib").Path(__file__).resolve().parents[1] / "src"
                / "recovery_agent" / "frontend.py").read_text()
    i = FRONTEND.index('if (final_status in ("escalated", "failed", "stopped")')
    body = FRONTEND[i:i + 2200]
    assert "elif (final_status in" in body and "enqueue(" in body


def test_a_do_not_pursue_case_is_not_handed_back_to_climb_rungs():
    FRONTEND = (__import__("pathlib").Path(__file__).resolve().parents[1] / "src"
                / "recovery_agent" / "frontend.py").read_text()
    i = FRONTEND.index("_rungs_left = (")
    assert "pursuit_barred" in FRONTEND[i:i + 300]


def test_the_nets_ticket_carries_the_ladder_too():
    FRONTEND = (__import__("pathlib").Path(__file__).resolve().parents[1] / "src"
                / "recovery_agent" / "frontend.py").read_text()
    i = FRONTEND.index("source=\"ladder_exhausted\"")
    body = FRONTEND[i - 1400:i]
    assert "ladder climbed:" in body
    assert "never tried —" in body


# ── the case must reach a conclusion, never silence ─────────────────────

def test_a_daemon_retry_is_watched_to_a_conclusion():
    """The retry is the last rung most cases reach, and it was where they
    stopped: the daemon created the order, said so, and nothing watched it. A
    customer who paid was never noticed; one who did not left the case at
    `scheduled` for good — never recovered, never escalated, never seen."""
    FRONTEND = (__import__("pathlib").Path(__file__).resolve().parents[1] / "src"
                / "recovery_agent" / "frontend.py").read_text()
    i = FRONTEND.index("def daemon_retry_complete")
    # Window sized to hold the whole handler: the wake_agent branch (which
    # honours the agent's own wait_for_customer) sits ahead of the retry watch.
    body = FRONTEND[i:i + 4200]
    assert "_watch_for_recovery" in body
    assert '"retry_created", "link_created"' in body


def test_the_timeout_handoff_tells_the_agent_where_it_stands():
    FRONTEND = (__import__("pathlib").Path(__file__).resolve().parents[1] / "src"
                / "recovery_agent" / "frontend.py").read_text()
    i = FRONTEND.index("No payment after {minutes} minute(s)")
    body = FRONTEND[i - 900:i + 900]
    assert "Already climbed" in body and "Already tried" in body
    assert "escalate_to_human exists for" in body, (
        "an exhausted ladder must say so; otherwise the agent reasons about a "
        "next channel that does not exist"
    )


# ── a blocked channel must not become a stuck case ──────────────────────

def test_a_spent_link_quota_makes_the_offer_rung_impossible():
    """A test account allows 30 payment links, ever. Once spent, every case
    jams at rung 2: no link can be made, no email may be sent without one, and
    escalation is refused because the ladder is not exhausted."""
    rec = case(links_unavailable=True)
    possible, why = ladder.rung_possible("offer", rec)
    assert not possible
    assert "30 payment links" in why


def test_the_ladder_can_still_finish_when_links_are_gone(monkeypatch):
    monkeypatch.delenv("VOICE_CALLS_ENABLED", raising=False)
    rec = climb(case(links_unavailable=True), "page_push", "alternate_path")
    assert ladder.exhausted(rec), (
        "with the offer impossible and voice off, an alternate route must be "
        "enough to reach escalation"
    )


def test_the_quota_failure_tells_the_agent_not_to_retry():
    TOOLS = (__import__("pathlib").Path(__file__).resolve().parents[1] / "src"
             / "recovery_agent" / "agent" / "tools.py").read_text()
    i = TOOLS.index("allowance of 30 payment links")
    # Join the source's line wraps before matching: the guidance is written as
    # `"... — do "` / `"not retry, and ..."`, so the sentence the agent actually
    # reads never appears contiguously in the file.
    import re
    body = re.sub(r'"\s*\n\s*"', "", TOOLS[i - 1200:i + 2000])
    assert '"status": "unavailable"' in body
    assert "do not retry" in body
    assert "promising a link that does not exist" in body
    assert "links_unavailable=True" in body
