"""Unit economics — the cost of a recovery must be measured, not vibes.

Tokens, calls, contacts, minted links and accepted discounts are recorded as
they happen; this suite pins the arithmetic that turns them into "what did it
cost to bring back ₹100", and the /api/economics contract the ops view reads.
"""
import json

import pytest

from recovery_agent import economics
from recovery_agent.state_store import StateStore

PID = "pay_eco1"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from recovery_agent import state_store
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    # Pin prices so the arithmetic is exact.
    monkeypatch.setenv("LLM_COST_INR_PER_MTOK_IN", "10")
    monkeypatch.setenv("LLM_COST_INR_PER_MTOK_OUT", "30")
    monkeypatch.setenv("VOICE_COST_INR_PER_CALL", "20")
    monkeypatch.setenv("EMAIL_COST_INR", "0")
    monkeypatch.setenv("SMS_COST_INR", "0")
    state_store.StateStore.reset_instances()
    yield
    state_store.StateStore.reset_instances()


def _seed(**over):
    rec = {"payment_id": PID, "amount": 2000.0, "status": "recovering",
           "recovered_amount": 0}
    rec.update(over)
    s = StateStore()
    s.save_payment(PID, rec)
    s.flush()
    return rec


# ── recording ───────────────────────────────────────────────────────────

def test_llm_usage_accumulates_across_calls():
    _seed()
    economics.record_llm_usage(PID, "gemma", {"input_tokens": 1000, "output_tokens": 200})
    economics.record_llm_usage(PID, "gemini", {"input_tokens": 500, "output_tokens": 100})
    u = StateStore().get_payment(PID)["llm_usage"]
    assert u["calls"] == 2
    assert u["input_tokens"] == 1500
    assert u["output_tokens"] == 300
    assert u["by_model"] == {"gemma": 1, "gemini": 1}


def test_a_call_with_no_usage_metadata_still_counts():
    _seed()
    economics.record_llm_usage(PID, "gemma", None)
    u = StateStore().get_payment(PID)["llm_usage"]
    assert u["calls"] == 1
    assert u["input_tokens"] == 0


def test_wall_time_accumulates():
    _seed()
    economics.record_run_wall(PID, 12.5)
    economics.record_run_wall(PID, 7.5)
    u = StateStore().get_payment(PID)["llm_usage"]
    assert u["wall_seconds"] == 20.0
    assert u["runs"] == 2


def test_recording_against_an_unknown_case_is_a_noop():
    economics.record_llm_usage("pay_nope", "m", {"input_tokens": 1})
    assert StateStore().get_payment("pay_nope") is None


# ── arithmetic ──────────────────────────────────────────────────────────

def test_case_economics_prices_the_full_anatomy():
    rec = _seed(
        recovered_amount=1900.0,          # paid a 5% discounted link
        llm_usage={"calls": 4, "input_tokens": 1_000_000, "output_tokens": 100_000},
        contacts=[{"channel": "email", "at": "t"}, {"channel": "voice", "at": "t"}],
        recovery_links=[{"link_id": "pl_1", "amount": 1900.0, "at": "t"}],
        failure_code="customer_dropped",
    )
    e = economics.case_economics(rec)
    assert e["llm_cost"] == 13.0            # 1M*10/1M + 0.1M*30/1M
    assert e["comms_cost"] == 20.0          # one voice call
    assert e["discount_given"] == 100.0     # 2000 − 1900
    assert e["total_cost"] == 133.0
    assert e["net_recovered"] == 1900.0 - 133.0
    assert e["links_minted"] == 1
    assert e["voice_calls"] == 1 and e["messages"] == 1


def test_no_discount_charged_while_nothing_recovered():
    rec = _seed(recovered_amount=0.0)
    assert economics.case_economics(rec)["discount_given"] == 0.0


def test_full_price_recovery_gives_nothing_away():
    rec = _seed(recovered_amount=2000.0)
    assert economics.case_economics(rec)["discount_given"] == 0.0


# ── aggregation ─────────────────────────────────────────────────────────

def test_summarise_buckets_by_failure_kind_and_skips_untouched():
    worked = {"payment_id": "a", "amount": 1000, "recovered_amount": 1000,
              "status": "recovered", "failure_code": "bank_declined_51",
              "llm_usage": {"calls": 2, "input_tokens": 100_000, "output_tokens": 10_000}}
    untouched = {"payment_id": "b", "amount": 500, "status": "paid"}
    out = economics.summarise([worked, untouched])
    assert out["totals"]["cases"] == 1
    assert out["totals"]["recovered"] == 1000
    assert out["totals"]["recovered_cases"] == 1
    assert len(out["by_failure_kind"]) == 1
    assert out["totals"]["cost_per_100_recovered"] is not None


def test_cost_per_100_is_none_when_nothing_recovered():
    worked = {"payment_id": "a", "amount": 1000, "recovered_amount": 0,
              "status": "recovering",
              "llm_usage": {"calls": 1, "input_tokens": 1000, "output_tokens": 100}}
    out = economics.summarise([worked])
    assert out["totals"]["cost_per_100_recovered"] is None


# ── the endpoint the ops view reads ─────────────────────────────────────

def test_api_economics_serves_all_four_panels():
    _seed(recovered_amount=1900.0,
          llm_usage={"calls": 1, "input_tokens": 1000, "output_tokens": 100})
    from recovery_agent.frontend import app
    body = json.loads(app.test_client().get("/api/economics").data)
    assert "economics" in body and "totals" in body["economics"]
    assert "guardrails" in body and "tallies" in body["guardrails"]
    assert "memory" in body
    assert "evals" in body  # may be null until a run exists


# ── two populations, never silently averaged ────────────────────────────

def _live_case(pid="pay_live1"):
    """Agent actually ran, against a real gateway order."""
    return {"payment_id": pid, "amount": 2000.0, "recovered_amount": 2000.0,
            "status": "recovered", "razorpay_order_id": "order_XYZ",
            "llm_usage": {"calls": 4, "input_tokens": 40_000,
                          "output_tokens": 2_000}}


def _seeded_case(pid="pay_seed1", **over):
    rec = {"payment_id": pid, "amount": 2000.0, "recovered_amount": 0,
           "status": "recovering"}
    rec.update(over)
    return rec


def test_a_case_the_agent_worked_end_to_end_is_live():
    assert economics.case_origin(_live_case()) == "live"


def test_demo_bar_triggers_are_never_live():
    """`/api/simulate` cases can look complete — agent ran, order exists —
    but they are volume we created, not traffic that arrived."""
    sim = dict(_live_case(), payment_id="pay_sim_4242")
    assert economics.case_origin(sim) == "seeded"


def test_a_case_the_agent_never_worked_is_seeded():
    assert economics.case_origin(_seeded_case()) == "seeded"
    # An imported row with an order but no agent run is still just volume.
    assert economics.case_origin(
        _seeded_case(razorpay_order_id="order_A")) == "seeded"


def test_scope_filters_the_population():
    records = [_live_case("pay_l1"), _live_case("pay_l2"),
               _seeded_case("pay_s1"), _seeded_case("pay_s2")]
    assert economics.summarise(records, scope="live")["totals"]["cases"] == 2
    assert economics.summarise(records, scope="seeded")["totals"]["cases"] == 2
    assert economics.summarise(records, scope="all")["totals"]["cases"] == 4


def test_blending_the_populations_moves_the_headline_number():
    """The reason the split exists: the two populations have genuinely
    different economics, so any blended average describes neither. On the
    real data live recovery runs far CHEAPER per rupee than seeded batch
    volume (which carries large discounts on little recovery); with these
    fixtures it runs dearer. Which way it moves is not the point — that it
    moves, silently, is."""
    records = [_live_case("pay_l1"),
               _seeded_case("pay_s1", recovered_amount=2000.0,
                            status="recovered")]
    live = economics.summarise(records, scope="live")["totals"]
    blended = economics.summarise(records, scope="all")["totals"]
    assert live["cost_per_100_recovered"] > blended["cost_per_100_recovered"]


def test_the_summary_declares_its_scope_and_test_mode(monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_abc")
    out = economics.summarise([_live_case()], scope="live")
    assert out["scope"] == "live"
    assert out["test_mode"] is True
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_live_abc")
    assert economics.summarise([_live_case()])["test_mode"] is False


def test_every_case_row_carries_its_origin():
    row = economics.case_economics(_live_case())
    assert row["origin"] == "live"


def test_the_endpoint_defaults_to_live_and_reports_both_counts():
    from recovery_agent.frontend import app
    body = json.loads(app.test_client().get("/api/economics").data)
    assert body["economics"]["scope"] == "live"
    assert set(body["scopes"]["counts"]) == {"live", "seeded"}
    seeded = json.loads(
        app.test_client().get("/api/economics?scope=seeded").data)
    assert seeded["economics"]["scope"] == "seeded"
    bad = json.loads(app.test_client().get("/api/economics?scope=nonsense").data)
    assert bad["economics"]["scope"] == "live"
