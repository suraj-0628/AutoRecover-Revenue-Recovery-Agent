"""The run: measured money across a batch, with stopping rules that stop.

The report is a fold over the append-only log rather than a counter someone
remembered to increment, so the assertions below mostly take the same form —
run something, then check the log agrees. That is the point of the design: there
is only one number, not a displayed one and a recorded one that can drift.
"""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from recovery_agent import audit
from recovery_agent.batch import executor as ex
from recovery_agent.batch import run as batch_run
from recovery_agent.batch.plan import BatchBudget, BatchPlan, Step

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

    monkeypatch.setattr(tools, "TOOLS_BY_NAME", {
        "generate_recovery_payment_link": tool(
            "generate_recovery_payment_link",
            {"status": "ok", "link_url": "https://rzp.io/l/fake",
             "link_id": "plink_fake"}),
        "send_recovery_notification": tool(
            "send_recovery_notification", {"status": "ok", "channels": ["email"]}),
        "retry_in_hours": tool("retry_in_hours", {"status": "scheduled"}),
    })
    yield SimpleNamespace(calls=calls, path=tmp_path)
    state_store.StateStore.reset_instances()
    audit.AuditLog.reset_instances()
    audit._default = None


def plan(tier="medium", **kw):
    base = dict(batch_key="dropoff", tier=tier, cause="they walked away",
                steps={"offer": Step("link_and_notify")},
                body="Hi {name}, {amount} is due: {link}")
    base.update(kw)
    return BatchPlan(**base)


def cases(n, *, amount=7999, prefix="pay"):
    return [{"payment_id": f"{prefix}_{i}", "amount": amount, "currency": "INR",
             "status": "failed", "decline_strategy": "user_dropoff",
             "customer": {"email": f"c{i}@x.com", "name": f"C{i}"}}
            for i in range(n)]


def make(**kw) -> batch_run.BatchRun:
    kw.setdefault("plans", {"medium": plan()})
    kw.setdefault("batch_key", "dropoff")
    return batch_run.BatchRun(**kw)


# ── it does the work ─────────────────────────────────────────────────────

def test_a_run_works_every_case_and_reports_what_it_did(env):
    report = make().execute(cases(4))
    assert report["status"] == batch_run.DONE
    assert (report["candidates"], report["acted"]) == (4, 4)
    assert report["links_created"] == 4 and report["emails_sent"] == 4
    assert report["at_risk_rupees"] == 4 * 7999
    assert env.calls and len(env.calls) == 8


def test_bands_are_planned_for_separately(env):
    """`merchant_dunning_rules.md` prescribes a different window, channel set and
    incentive per band, so one plan across two bands would apply one band's
    policy to the other band's customers."""
    run = make(plans={"medium": plan("medium"), "small": plan("small")})
    report = run.execute(cases(2) + cases(2, amount=1200, prefix="sm"))
    assert report["acted"] == 4
    assert report["exceptions"] == 0


def test_a_band_nobody_planned_for_goes_to_the_agent_not_the_bin(env):
    report = make().execute(cases(2) + cases(3, amount=1200, prefix="sm"))
    assert report["acted"] == 2
    assert report["exceptions"] == 3
    assert "no plan for tier small" in json.dumps(report["exception_by_reason"])
    assert len(env.calls) == 4          # nothing was sent to the unplanned band


# ── stopping rules ───────────────────────────────────────────────────────

def test_the_link_budget_binds_and_the_run_says_which_resource(env):
    """A Razorpay test account allows 30 links for its lifetime and cancelling
    does not return one, so the run's cap has to bind before the gateway's."""
    report = make(budget=BatchBudget(max_links=2)).execute(cases(5))
    assert report["links_created"] == 2
    assert report["stop_reason"] == "budget_links"
    assert [n for n, _ in env.calls].count("generate_recovery_payment_link") == 2


def test_an_abort_stops_the_run_and_is_recorded_as_an_abort(env):
    run = make()
    run.abort("stopped from the dashboard")
    report = run.execute(cases(5))
    assert report["status"] == batch_run.ABORTED
    assert report["stop_reason"] == "stopped from the dashboard"
    assert env.calls == []
    kinds = [e["kind"] for e in audit.log().for_run(run.run_id)]
    assert audit.BATCH_ABORTED in kinds and audit.BATCH_FINISHED not in kinds


def test_a_deadline_stops_a_run_that_overruns(env):
    # A zero-second allowance is expressible on purpose: it is the honest way
    # to say "this run has no time left", and it is the same code path a run
    # that overruns fifteen minutes later takes.
    run = make(budget=BatchBudget(max_wallclock_seconds=0))
    report = run.execute(cases(3))
    assert report["acted"] == 0
    assert run.stop_reason == "stopped: wallclock"
    assert env.calls == []


def test_repeated_failures_stop_the_run_rather_than_grinding_through(env,
                                                                     monkeypatch):
    """An expired key fails every case identically. Discovering that once is
    diagnosis; discovering it twenty-five times is just spending the budget."""
    from recovery_agent.agent import tools
    broken = dict(tools.TOOLS_BY_NAME)
    broken["generate_recovery_payment_link"] = SimpleNamespace(
        func=lambda **kw: json.dumps({"status": "error", "reason": "key expired"}))
    monkeypatch.setattr(tools, "TOOLS_BY_NAME", broken)

    run = make(budget=BatchBudget(abort_after_consecutive_failures=2))
    report = run.execute(cases(6))
    assert report["exceptions"] < 6
    assert run.stop_reason == "stopped: consecutive_failures"


def test_clicking_run_twice_spends_once(env):
    """The budget lives on the run, and every case must also clear
    `already_in_a_run` — so a second click creates a run that acts on nothing."""
    from recovery_agent.state_store import StateStore
    store = StateStore()
    for rec in cases(3):
        store.save_payment(rec["payment_id"], rec)
    store.flush()

    first = make().execute(cases(3))
    fresh = [dict(store.get_payment(f"pay_{i}") or {}) for i in range(3)]
    second = make().execute(fresh)

    assert first["acted"] == 3 and second["acted"] == 0
    assert second["skipped_by_reason"] == {"already_in_a_run": 3}


# ── the report is the log ────────────────────────────────────────────────

def test_the_projection_rebuilt_from_events_equals_the_live_report(env):
    run = make(budget=BatchBudget(max_links=3))
    live = run.execute(cases(5))
    rebuilt = batch_run.projection(run.run_id)
    for key in ("candidates", "acted", "skipped", "exceptions",
                "at_risk_paise", "acted_paise", "links_created",
                "recovered_paise", "status"):
        assert rebuilt[key] == live[key], key


def test_recovered_money_is_resolved_on_read_and_climbs_afterwards(env):
    """A batch finishes sending in seconds; customers pay over the following
    minutes. A total frozen at `finished_at` would be wrong in the direction
    that flatters us."""
    run = make()
    run.execute(cases(3))
    assert batch_run.projection(run.run_id)["recovered_paise"] == 0

    audit.record(audit.MONEY_RECOVERED, payment_id="pay_0",
                 batch_run_id=run.run_id, amount_rupees=7999)
    after = batch_run.projection(run.run_id)
    assert after["recovered_rupees"] == 7999
    assert after["status"] == batch_run.DONE      # still closed
    assert after["recovery_rate"] == round(1 / 3, 4)


def test_money_counts_only_against_the_run_that_earned_it(env):
    a, b = make(), make()
    a.execute(cases(2, prefix="a"))
    b.execute(cases(2, prefix="b"))
    audit.record(audit.MONEY_RECOVERED, payment_id="a_0",
                 batch_run_id=a.run_id, amount_rupees=7999)
    audit.record(audit.MONEY_RECOVERED, payment_id="live_case",
                 amount_rupees=50_000)            # not a batch at all
    assert batch_run.projection(a.run_id)["recovered_rupees"] == 7999
    assert batch_run.projection(b.run_id)["recovered_rupees"] == 0


def test_the_same_settlement_reported_twice_is_counted_once(env):
    """A webhook and a poll can both see one payment. Counting it twice would
    overstate recovery in a number presented as measured."""
    run = make()
    run.execute(cases(1))
    for _ in range(3):
        audit.record(audit.MONEY_RECOVERED, payment_id="pay_0",
                     batch_run_id=run.run_id, amount_rupees=7999)
    assert batch_run.projection(run.run_id)["recovered_rupees"] == 7999


def test_gross_discount_and_net_are_reported_separately(env):
    """Money given away is not money recovered. A single headline figure hides
    exactly the number a merchant would want to see."""
    run = make()
    run.execute(cases(1))
    audit.record(audit.MONEY_RECOVERED, payment_id="pay_0",
                 batch_run_id=run.run_id, amount_rupees=7599)
    audit.record(audit.ACTION_RESULT, payment_id="pay_0",
                 batch_run_id=run.run_id, discount_paise=40000)
    report = batch_run.projection(run.run_id)
    assert (report["recovered_rupees"], report["discount_rupees"],
            report["net_rupees"]) == (7599, 400, 7199)


def test_a_compliant_pause_is_not_reported_as_a_case_we_gave_up_on(env,
                                                                   monkeypatch):
    """A case held back by policy is DEFERRED, never SKIPPED — the merchant has
    to be able to tell "we chose not to, for now" from "we gave up".

    The trigger used to be quiet hours at 23:00, which no longer defers an
    email: quiet hours gate what INTERRUPTS (a call rings, an SMS buzzes) and
    an email waits in an inbox. Opt-out is used here instead — a customer who
    has said "stop contacting me" is the cleanest compliant pause there is.
    """
    from recovery_agent.agent import guardrails
    from recovery_agent.models import ActionType

    def _opted_out(self, case, action, profile=None, now=None):
        return ActionType.WAIT_AND_RETRY, [guardrails.GuardrailCheckResult(
            guardrail="opt_out", verdict=guardrails.GuardrailVerdict.BLOCKED,
            reason="Customer has opted out of automated contact",
            original_action=action.value)]

    monkeypatch.setattr(guardrails.GuardrailEngine, "validate_action", _opted_out)
    report = make().execute(cases(3))
    assert report["deferred"] == 3 and report["skipped"] == 0
    assert list(report["skipped_by_reason"]) == ["opt_out"]
    assert env.calls == []


def test_an_overnight_batch_still_sends_its_email(env, monkeypatch):
    """The other half of the same policy: 23:00 does not stop a message that
    wakes nobody. Live (pay_woo85c9gh), deferring here cost a real recovery."""
    from recovery_agent.agent import guardrails
    real = guardrails.GuardrailEngine.validate_action
    monkeypatch.setattr(guardrails.GuardrailEngine, "validate_action",
                        lambda self, c, a, p=None, now=None: real(
                            self, c, a, p, datetime(2026, 9, 4, 23, 0)))
    report = make().execute(cases(3))
    assert report["deferred"] == 0
    assert env.calls, "the email still goes out overnight"


# ── the dry run ──────────────────────────────────────────────────────────

def test_a_dry_run_decides_the_same_things_and_does_none_of_them(env):
    """It costs nothing and owns no payment links, which is what makes it
    possible to show a 200-case batch's plan without spending a quota on it."""
    dry = make(dry_run=True).execute(cases(4) + cases(2, amount=1200, prefix="sm"))
    assert env.calls == []
    assert dry["dry_run"] is True
    # It reports what it WOULD do. Reading zero here — because a dry run writes
    # no action events — would make a run that found four workable cases look
    # like one that found none.
    assert dry["acted"] == 4 and dry["links_created"] == 4

    live = make().execute(cases(4) + cases(2, amount=1200, prefix="sm"))
    assert [d["outcome"] for d in dry["decisions"]] == \
           [d["outcome"] for d in live["decisions"]]
    assert [d["charged_paise"] for d in dry["decisions"]] == \
           [d["charged_paise"] for d in live["decisions"]]


# ── the callback the web layer watches links through ────────────────────

def test_every_link_created_is_handed_to_the_caller_to_watch(env):
    """The integration rig caught this one: the batch created three links,
    three customers paid them, and the run reported zero recovered — because
    the recovery watcher is started on the agent path and nothing started it
    here. A send-only batch cannot measure money.

    The executor must not reach into the web layer to fix that, so it hands
    every acted decision back and the caller starts the watcher.
    """
    seen = []
    make().execute(cases(3), on_decision=seen.append)
    acted = [d for d in seen if d.outcome == ex.ACTED]
    assert len(acted) == 3
    for decision in acted:
        assert decision.detail["link_id"], "no id to watch this link by"
        assert decision.detail["link_url"]


def test_a_listener_that_throws_does_not_stop_the_run(env):
    """A watcher failing to start is a case whose money goes uncounted. A
    watcher failing to start and taking the batch with it is worse."""
    def boom(decision):
        raise RuntimeError("the socket layer is down")
    report = make().execute(cases(2), on_decision=boom)
    assert report["acted"] == 2


# ── the registry the abort button needs ──────────────────────────────────

def test_a_live_run_can_be_found_and_stopped_by_id(env):
    run = batch_run.register(make())
    assert batch_run.get(run.run_id) is run
    assert run in batch_run.live()
    batch_run.get(run.run_id).abort()
    assert run.aborted
    run.execute(cases(1))
    assert batch_run.live() == []
    batch_run.forget(run.run_id)
    assert batch_run.get(run.run_id) is None
