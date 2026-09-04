"""The wave cycle: decide once per bin, apply, watch, re-bin, repeat.

The customers in these tests are played by the `on_decision` hook — the seam
the web layer uses to watch links is also the seam a test uses to make a
customer pay, bounce, or ignore. That keeps every test on the real code path:
nothing here reaches into the cycle's internals.
"""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from recovery_agent import audit
from recovery_agent.batch import run as batch_run
from recovery_agent.batch import waves
from recovery_agent.batch.plan import BatchBudget

DAYTIME = datetime(2026, 9, 4, 10, 0)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from recovery_agent import state_store
    from recovery_agent.agent import guardrails, tools

    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path, raising=False)
    monkeypatch.delenv("RAZORPAY_WRITES_OK", raising=False)
    state_store.StateStore.reset_instances()
    audit.AuditLog.reset_instances()
    audit._default = None

    real = guardrails.GuardrailEngine.validate_action
    monkeypatch.setattr(guardrails.GuardrailEngine, "validate_action",
                        lambda self, c, a, p=None, now=None: real(
                            self, c, a, p, now or DAYTIME))

    calls: list[tuple] = []

    def tool(name, payload):
        def func(**kw):
            calls.append((name, kw))
            return json.dumps(payload)
        return SimpleNamespace(name=name, func=func)

    # The real tools climb the ladder on success; faithful fakes must too,
    # or wave 2 sees yesterday's rung and the tests exercise a system that
    # cannot exist.
    from recovery_agent.agent import ladder as _ladder

    def notify(**kw):
        calls.append(("send_recovery_notification", kw))
        rec = state_store.StateStore().get_payment(kw.get("payment_id")) or {}
        nxt = _ladder.next_rung(rec)
        if nxt:
            _ladder.record_rung(kw["payment_id"], nxt["rung"], "batch send")
        return json.dumps({"status": "ok", "channels": ["email"]})

    def retry(**kw):
        calls.append(("retry_in_hours", kw))
        _ladder.record_rung(kw["payment_id"], "silent_retry",
                            f"retry scheduled in {kw.get('hours', 24)}h")
        return json.dumps({"status": "scheduled"})

    monkeypatch.setattr(tools, "TOOLS_BY_NAME", {
        "generate_recovery_payment_link": tool(
            "generate_recovery_payment_link",
            {"status": "ok", "link_url": "https://rzp.io/l/fake",
             "link_id": "plink_fake"}),
        "send_recovery_notification": SimpleNamespace(
            name="send_recovery_notification", func=notify),
        "retry_in_hours": SimpleNamespace(name="retry_in_hours", func=retry),
    })

    waves._CYCLES.clear()
    store = state_store.StateStore()
    yield SimpleNamespace(calls=calls, store=store, path=tmp_path)
    waves._CYCLES.clear()
    state_store.StateStore.reset_instances()
    audit.AuditLog.reset_instances()
    audit._default = None


def seed(store, n=3, *, strategy="user_dropoff", amount=7999.0, prefix="pay",
         **extra):
    ids = []
    for i in range(n):
        pid = f"{prefix}_{i}"
        store.save_payment(pid, {
            "payment_id": pid, "amount": amount, "currency": "INR",
            "status": "failed", "decline_strategy": strategy,
            "customer": {"email": f"c{i}@x.com", "name": f"C{i}"}, **extra})
        ids.append(pid)
    store.flush()
    return ids


def config(**kw):
    kw.setdefault("settle_seconds", 0.01)
    kw.setdefault("max_waves", 3)
    return waves.WaveConfig(**kw)


def payer(store):
    """A customer who pays the moment they are contacted — via the same hook
    the web layer uses to start link watchers."""
    def on_decision(decision):
        if decision.outcome != "acted" or decision.action == "retry":
            return
        rec = store.get_payment(decision.payment_id) or {}
        store.update_payment(decision.payment_id, status="recovered",
                             recovered_amount=decision.charged_paise / 100)
        store.flush()
        audit.record(audit.MONEY_RECOVERED, payment_id=decision.payment_id,
                     batch_run_id=rec.get("batch_run_id") or "",
                     amount_paise=decision.charged_paise)
    return on_decision


# ── the loop itself ──────────────────────────────────────────────────────

def test_everyone_paying_ends_the_cycle_after_one_wave(env):
    ids = seed(env.store, 3)
    cycle = waves.WaveCycle(payment_ids=ids, config=config(),
                            on_decision=payer(env.store))
    report = cycle.execute()
    assert report["stop_reason"] == "all_settled"
    assert len(report["waves"]) == 1
    assert report["recovered_paise"] == 3 * 799900
    assert report["links_created"] == 3


def test_ignoring_customers_do_not_get_the_same_wave_twice(env):
    """Wave 2 must not re-send wave 1: the ladder forbids repeating a rung,
    and the post-offer rungs route to the agent. An ignored batch therefore
    converges to the agent's desk, not to a louder inbox."""
    ids = seed(env.store, 3)
    cycle = waves.WaveCycle(payment_ids=ids, config=config())
    report = cycle.execute()

    emails = [kw for name, kw in env.calls
              if name == "send_recovery_notification"]
    assert len(emails) == 3, "wave 1 contacted each customer exactly once"
    assert report["stop_reason"] in ("dry", "max_waves")
    assert len(report["exceptions"]) == 3, "wave 2 handed them to the agent"
    assert all(e["batch_run_id"] for e in report["exceptions"]), (
        "each referral names the run that sent it, so an agent rescue still "
        "counts for the cycle's money")


def test_a_case_that_fails_differently_is_re_binned_and_re_treated(env):
    """The user's exact scenario: a dropoff gets a link, the attempt bounces
    on funds, and the next wave treats it as the funds case it now is."""
    ids = seed(env.store, 1)

    def bounce(decision):
        if decision.outcome == "acted" and decision.action == "link_and_notify":
            env.store.update_payment(decision.payment_id,
                                     decline_strategy="insufficient_funds",
                                     failure_reason="link attempt: no funds")
            env.store.flush()

    cycle = waves.WaveCycle(payment_ids=ids, config=config(),
                            on_decision=bounce)
    cycle.execute()

    keys = {batch_run.projection(rid)["batch_key"] for rid in cycle.run_ids}
    assert keys == {"dropoff", "insufficient_funds"}, (
        "the same case was worked under both causes, in order")
    assert [n for n, _ in env.calls].count("retry_in_hours") == 1, (
        "the funds wave scheduled the retry the new cause calls for")


def test_recovered_cases_drop_out_between_waves(env):
    ids = seed(env.store, 2) + seed(env.store, 1, prefix="late",
                                    strategy="insufficient_funds")
    cycle = waves.WaveCycle(payment_ids=ids, config=config(max_waves=2),
                            on_decision=payer(env.store))
    report = cycle.execute()
    sends = [kw.get("payment_id") for n, kw in env.calls
             if n == "send_recovery_notification"]
    assert sends.count("pay_0") == 1 and sends.count("pay_1") == 1, (
        "a customer who paid in wave 1 is never messaged in wave 2")
    assert report["recovered_paise"] == 2 * 799900


def test_an_opted_out_customer_is_never_contacted_in_any_wave(env):
    ids = seed(env.store, 2) + seed(env.store, 1, prefix="optout",
                                    opted_out=True)
    cycle = waves.WaveCycle(payment_ids=ids, config=config(max_waves=2))
    cycle.execute()
    touched = {kw.get("payment_id") for _, kw in env.calls}
    assert "optout_0" not in touched


def test_abort_stops_everything_and_says_so(env):
    ids = seed(env.store, 3)
    cycle = waves.WaveCycle(payment_ids=ids, config=config())
    cycle.abort("operator said stop")
    report = cycle.execute()
    assert report["status"] == waves.ABORTED
    assert report["stop_reason"] == "operator said stop"
    assert env.calls == []


def test_the_cycle_is_findable_and_abortable_by_id(env):
    cycle = waves.register(waves.WaveCycle(payment_ids=[], config=config()))
    assert waves.get(cycle.cycle_id) is cycle
    waves.get(cycle.cycle_id).abort()
    assert cycle.aborted


# ── the report outlives the process ──────────────────────────────────────

def test_a_finished_cycle_survives_a_restart(env):
    """The live object dies with its process; the report is rebuilt from the
    audit log — the only copy an auditor ever asks for anyway."""
    ids = seed(env.store, 3)
    cycle = waves.WaveCycle(payment_ids=ids, config=config(max_waves=2),
                            on_decision=payer(env.store))
    live = cycle.execute()

    rebuilt = waves.cycle_projection(cycle.cycle_id)   # no registry involved
    assert rebuilt is not None and rebuilt["rebuilt_from_audit"]
    for key in ("status", "stop_reason", "cases", "runs",
                "recovered_paise", "links_created"):
        assert rebuilt[key] == live[key], key
    assert len(rebuilt["waves"]) == len(live["waves"])
    assert [w["acted"] for w in rebuilt["waves"]] ==            [w["acted"] for w in live["waves"]]


def test_a_cycle_s_exceptions_survive_too(env):
    ids = seed(env.store, 2)
    cycle = waves.WaveCycle(payment_ids=ids, config=config(max_waves=2))
    live = cycle.execute()
    rebuilt = waves.cycle_projection(cycle.cycle_id)
    assert {e["payment_id"] for e in rebuilt["exceptions"]} ==            {e["payment_id"] for e in live["exceptions"]}
    assert all(e["batch_run_id"] for e in rebuilt["exceptions"])


def test_an_unknown_cycle_projects_to_none(env):
    assert waves.cycle_projection("cyc_never_happened") is None


def test_the_log_lists_cycles_newest_first(env):
    a = waves.WaveCycle(payment_ids=seed(env.store, 1, prefix="a"),
                        config=config(max_waves=1))
    a.execute()
    b = waves.WaveCycle(payment_ids=seed(env.store, 1, prefix="b"),
                        config=config(max_waves=1))
    b.execute()
    known = waves.known_cycle_ids()
    assert known.index(b.cycle_id) < known.index(a.cycle_id)


# ── the champion path ────────────────────────────────────────────────────

def champion_runner_for(store, calls):
    """Plays the live agent: climbs a rung, makes a link, sends a message —
    by mutating the record the way the real session's tools do."""
    def runner(record, context):
        calls.append((record["payment_id"], context))
        pid = record["payment_id"]
        rec = store.get_payment(pid) or {}
        store.update_payment(
            pid,
            ladder={**(rec.get("ladder") or {}),
                    "rail_switch": {"at": "2026-09-04T10:00:00+00:00",
                                    "detail": "UPI link, full price"}},
            recovery_links=[{"link_id": "plink_champ", "amount": rec["amount"]}],
            contacts=[{"channel": "email", "at": "2026-09-04T10:00:01+00:00"}])
        store.flush()
    return runner


def test_one_champion_decides_and_the_rest_follow_its_plan(env):
    ids = seed(env.store, 5, strategy="card_expired", amount=7999.0)
    sessions = []
    cycle = waves.WaveCycle(
        payment_ids=ids, config=config(champion_mode="live", max_waves=1),
        champion_runner=champion_runner_for(env.store, sessions))
    report = cycle.execute()

    assert len(sessions) == 1, "one real agent session for a bin of five"
    assert "representative of a batch of 5" in sessions[0][1]
    champ = sessions[0][0]
    assert report["champions"] == [champ]

    plan = cycle._plans[("bank_declined", "medium")]
    assert plan.provenance["source"] == "champion"
    assert plan.provenance["champion"] == champ

    worked = [kw.get("payment_id") for n, kw in env.calls
              if n == "generate_recovery_payment_link"]
    assert champ not in worked, "the champion's session WAS its treatment"
    assert len(worked) == 4, "the other four got the champion's plan"
    assert (env.store.get_payment(champ) or {}).get("batch_run_id"), (
        "the champion is stamped so its money still lands on the run")


def test_a_failed_champion_falls_back_to_the_policy_default(env):
    ids = seed(env.store, 3, strategy="card_expired")

    def broken(record, context):
        raise RuntimeError("the proxy is down")

    cycle = waves.WaveCycle(
        payment_ids=ids, config=config(champion_mode="live", max_waves=1),
        champion_runner=broken)
    report = cycle.execute()
    plan = cycle._plans[("bank_declined", "medium")]
    assert plan.provenance["source"] == "default", (
        "the system must not stop because the model did")
    assert report["waves"][0]["acted"] == 3, (
        "every case was still worked, on the safe floor")


def test_champion_off_uses_the_defaults_and_runs_no_sessions(env):
    ids = seed(env.store, 3)
    sessions = []
    cycle = waves.WaveCycle(
        payment_ids=ids, config=config(champion_mode="off"),
        champion_runner=champion_runner_for(env.store, sessions),
        on_decision=payer(env.store))
    cycle.execute()
    assert sessions == [], "CI mode: zero LLM, defaults carry every bin"


# ── the autopilot's judgement, without a server ──────────────────────────

def test_the_autopilot_never_starts_over_a_live_cycle_or_at_night(env,
                                                                  monkeypatch):
    """'Deferred' means 'not now', and this is the someone who comes back —
    but only when coming back is allowed."""
    monkeypatch.delenv("GUARDRAIL_QUIET_DISABLED", raising=False)
    import recovery_agent.frontend as frontend

    live = waves.register(waves.WaveCycle(payment_ids=[], config=config()))
    try:
        assert frontend._wave_autopilot_blocked(
            datetime(2026, 9, 4, 10, 0)) == "a cycle is already running"
    finally:
        live.status = waves.DONE

    assert frontend._wave_autopilot_blocked(
        datetime(2026, 9, 4, 23, 0)) == "quiet hours", (
        "an unattended pass must not message anyone at night")
    assert frontend._wave_autopilot_blocked(datetime(2026, 9, 4, 10, 0)) == ""


def test_the_autopilot_is_off_unless_a_merchant_turns_it_on():
    import pathlib
    src = (pathlib.Path(waves.__file__).resolve().parents[2].parent
           / "src" / "recovery_agent" / "frontend.py").read_text()
    assert 'os.getenv("WAVE_AUTOPILOT_MINUTES", "0")' in src, (
        "an unattended sender is a thing a merchant turns on, never a thing "
        "that turns itself on")
