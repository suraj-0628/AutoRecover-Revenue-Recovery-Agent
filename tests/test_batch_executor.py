"""The gate in front of every batch side effect.

The executor's whole claim is that a decision made once can be applied many
times *safely*. That claim rests entirely on `precheck`: twelve conditions, each
of which, if wrong, is wrong for every case in the batch at once. So there is a
test per condition, and each asserts two things — the decision, and that nothing
happened. A skip that still sent the email is the failure mode worth catching.
"""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from recovery_agent import audit
from recovery_agent.batch import executor as ex
from recovery_agent.batch.plan import BatchBudget, BatchPlan, Step


#: A weekday mid-morning, comfortably outside the merchant's 21:00-08:00 quiet
#: hours. Pinned because otherwise this suite passes or fails by the hour it is
#: run — which is itself proof the guardrail is wired in.
DAYTIME = datetime(2026, 9, 4, 10, 0)
NIGHT = datetime(2026, 9, 4, 23, 0)


def _pin(monkeypatch, when):
    from recovery_agent.agent import guardrails
    real = guardrails.GuardrailEngine.validate_action
    monkeypatch.setattr(
        guardrails.GuardrailEngine, "validate_action",
        lambda self, c, a, p=None, now=None: real(self, c, a, p, now or when))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A store and an audit log of our own. No real account is reachable."""
    from recovery_agent import state_store
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path, raising=False)
    monkeypatch.delenv("RAZORPAY_WRITES_OK", raising=False)
    state_store.StateStore.reset_instances()
    audit.AuditLog.reset_instances()
    audit._default = None
    _pin(monkeypatch, DAYTIME)
    yield tmp_path
    state_store.StateStore.reset_instances()
    audit.AuditLog.reset_instances()
    audit._default = None


class Calls:
    """A stand-in tool registry that records instead of acting."""

    def __init__(self):
        self.log = []

    def install(self, monkeypatch, *, link_status="ok", notify_status="ok"):
        from recovery_agent.agent import tools

        def tool(name, payload):
            def func(**kwargs):
                self.log.append((name, kwargs))
                return json.dumps(payload)
            return SimpleNamespace(name=name, func=func)

        monkeypatch.setattr(tools, "TOOLS_BY_NAME", {
            "generate_recovery_payment_link": tool(
                "generate_recovery_payment_link",
                {"status": link_status, "link_url": "https://rzp.io/l/fake",
                 "link_id": "plink_fake", "reason": "quota"}),
            "send_recovery_notification": tool(
                "send_recovery_notification",
                {"status": notify_status, "channels": ["email"]}),
            "retry_in_hours": tool("retry_in_hours", {"status": "scheduled"}),
        })
        return self

    @property
    def names(self):
        return [n for n, _ in self.log]


def plan(**kw) -> BatchPlan:
    base = dict(
        batch_key="dropoff", tier="medium", cause="they walked away",
        steps={"offer": Step(action="link_and_notify", full_price=True)},
        body="Hi {name}, {amount} is due: {link}", subject="Your order")
    base.update(kw)
    return BatchPlan(**base)


def case(**kw) -> dict:
    rec = {"payment_id": "pay_1", "amount": 7999, "currency": "INR",
           "status": "failed", "decline_strategy": "user_dropoff",
           "customer": {"email": "a@b.com", "name": "Asha"}}
    rec.update(kw)
    return rec


def run(record, p=None, *, budget=None, spend=None, dry_run=False,
        run_id="run_1"):
    return ex.work_case(record, p or plan(), run_id=run_id,
                        budget=budget or BatchBudget(),
                        spend=spend or ex.Spend(), dry_run=dry_run)


# ── the happy path, so the refusals below mean something ─────────────────

def test_an_eligible_case_gets_a_link_and_a_message(env, monkeypatch):
    calls = Calls().install(monkeypatch)
    d = run(case())
    assert d.outcome == ex.ACTED
    assert calls.names == ["generate_recovery_payment_link",
                           "send_recovery_notification"]
    assert d.detail["rung"] == "offer"


def test_the_message_carries_this_case_s_own_figures(env, monkeypatch):
    """A shared template must never leak one customer's amount into another's
    message — every number is re-derived from the record being worked."""
    calls = Calls().install(monkeypatch)
    run(case(amount=7999, customer={"email": "a@b.com", "name": "Asha"}))
    msg = dict(calls.log[1][1])["message"]
    assert "Asha" in msg and "7,999" in msg and "rzp.io" in msg


# ── one test per refusal, each asserting nothing happened ────────────────

@pytest.mark.parametrize("record,outcome,reason", [
    (case(decline_strategy="card_expired"), ex.SKIPPED, "reclassified"),
    (case(amount=250), ex.SKIPPED, "wrong_tier"),
    # Fraud is refused twice over too: the classifier puts it in the
    # un-runnable `risk` batch before `pursuit_barred` behind it gets a turn.
    (case(failure_reason="suspected fraud"), ex.SKIPPED, "reclassified"),
    (case(batch_run_id="another_run"), ex.SKIPPED, "already_in_a_run"),
    # A contactless case is refused twice over: `classify` puts it in no batch
    # at all, so it never even reaches the contact check behind it.
    (case(customer={}), ex.SKIPPED, "reclassified"),
    (case(payment_id=""), ex.SKIPPED, "no payment id"),
    # The ladder already knows an offer needs a link, so a quota-exhausted case
    # arrives here having been advanced past `offer` rather than sitting on it.
    (case(links_unavailable=True), ex.EXCEPTION, "ladder_advanced"),
])
def test_a_refused_case_is_not_touched(env, monkeypatch, record, outcome, reason):
    calls = Calls().install(monkeypatch)
    d = run(record)
    assert (d.outcome, reason in d.reason) == (outcome, True), d.reason
    assert calls.log == []


def test_fraud_and_disputes_are_never_chased(env, monkeypatch):
    """The check behind the classifier, exercised directly. Contacting someone
    about a payment their bank is disputing is worse than doing nothing."""
    calls = Calls().install(monkeypatch)
    record = case(failure_reason="suspected fraud")
    d = ex.work_case(record, plan(batch_key="risk"), run_id="run_1",
                     budget=BatchBudget(), spend=ex.Spend())
    assert d.outcome == ex.SKIPPED and "pursuit_barred" in d.reason
    assert calls.log == []


def test_a_case_that_has_already_paid_is_left_alone(env, monkeypatch):
    """The most expensive mistake available: chasing money already received."""
    from recovery_agent.agent import perception
    calls = Calls().install(monkeypatch)
    monkeypatch.setattr(perception, "ground_truth", lambda pid: {"settled": True})
    d = run(case())
    assert (d.outcome, d.reason) == (ex.SKIPPED, "already_paid")
    assert calls.log == []


def test_a_case_past_the_plan_s_rungs_becomes_an_exception(env, monkeypatch):
    """Not an improvisation. A case that has moved on since planning is exactly
    the one a shared decision no longer fits, so it goes to the agent."""
    calls = Calls().install(monkeypatch)
    d = run(case(ladder={"offer": {"at": "2026-09-01T10:00:00+00:00"}}))
    assert d.outcome == ex.EXCEPTION
    assert "ladder_advanced" in d.reason
    assert calls.log == []


def test_a_plan_may_route_a_rung_to_the_agent_or_skip_it(env, monkeypatch):
    calls = Calls().install(monkeypatch)
    assert run(case(), plan(steps={"offer": Step("exception", why="needs a human")})
               ).outcome == ex.EXCEPTION
    assert run(case(), plan(steps={"offer": Step("skip", why="out of policy")})
               ).outcome == ex.SKIPPED
    assert calls.log == []


# ── compliance: the merchant's own committed rules ───────────────────────

def test_a_night_batch_still_emails_but_never_rings(env, monkeypatch):
    """Quiet hours gate what INTERRUPTS, not all recovery.

    This used to assert the whole case was DEFERRED at 23:00, citing
    `merchant_dunning_rules.md` — but that file documents no quiet-hours rule
    at all (it covers frequency caps and channel tiers); the 21:00-08:00 window
    is a code-side default. Deferring email overnight cost real recovery: live
    on pay_woo85c9gh a customer was refused at 06:30 IST *thirteen seconds*
    after dismissing an in-page notification, i.e. wide awake at their own
    keyboard. An email waits in an inbox and wakes nobody, so it still goes;
    the SMS leg is suppressed inside the dispatcher, and a voice call — the
    thing that actually rings — is still refused."""
    calls = Calls().install(monkeypatch)
    _pin(monkeypatch, NIGHT)
    d = run(case())
    assert d.outcome == ex.ACTED, "an overnight email is not a policy breach"
    assert calls.names == ["generate_recovery_payment_link",
                           "send_recovery_notification"]


def test_a_voice_call_is_still_refused_at_night(env, monkeypatch):
    """The rule quiet hours actually exist for."""
    from recovery_agent.agent.guardrails import GuardrailVerdict, QuietHourGuardrail
    from recovery_agent.models import ActionType
    r = QuietHourGuardrail().check(ActionType.VOICE_CALL, now=NIGHT)
    assert r.verdict == GuardrailVerdict.MODIFIED


def test_a_guardrail_failure_never_blocks_a_recovery(env, monkeypatch):
    """A bug in the compliance check must not become a customer who never got
    their link. It fails open, deliberately, and the audit trail records it."""
    from recovery_agent.agent import guardrails
    calls = Calls().install(monkeypatch)
    monkeypatch.setattr(guardrails, "GuardrailEngine",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert run(case()).outcome == ex.ACTED
    assert calls.names[0] == "generate_recovery_payment_link"


# ── the approval ceiling the graph node cannot enforce here ──────────────

def test_a_discount_over_the_ceiling_needs_a_human_not_a_link(env, monkeypatch):
    """`human_approval_gate` is a graph *node*, not a tool guard. An executor
    calling the link tool directly walks straight past it, so the ceiling has to
    be enforced here or the batch path gives money away the live path would
    have stopped."""
    from recovery_agent.agent import graph
    calls = Calls().install(monkeypatch)
    monkeypatch.setattr(graph, "_APPROVAL_DISCOUNT_THRESHOLD", 1, raising=False)
    d = run(case(), plan(offer_stage="email_offer",
                         steps={"offer": Step("link_and_notify", full_price=False)}))
    assert d.outcome == ex.EXCEPTION
    assert d.reason == "needs_approval"
    assert calls.log == []


# ── the runtime is real, which is what keeps the tool guards on ──────────

def test_the_case_the_tools_see_is_the_real_one(env):
    """`generate_recovery_payment_link` computes its rupees-vs-paise range check
    from `runtime.context.case`. With `runtime=None` that check silently does
    not happen and the 100x overcharge guard is off."""
    rt = ex.build_runtime(case(amount=7999))
    assert rt.context.case.payment.amount == 7999
    assert rt.context.case.payment.metadata["customer_email"] == "a@b.com"


def test_the_real_link_tool_refuses_more_than_the_debt(env, monkeypatch):
    """The test that catches `runtime=None`: with a real runtime the tool has a
    reference amount and refuses; without one `expected` is None, the whole
    range check switches off, and the customer is billed 100x.

    A fake client is installed only so the check is reachable — reaching it at
    all would already be the bug."""
    from recovery_agent import razorpay_client
    created = []
    monkeypatch.setattr(razorpay_client, "RazorpayClient", lambda: SimpleNamespace(
        is_configured=True,
        create_payment_link=lambda **kw: created.append(kw)))

    from recovery_agent.agent.tools import TOOLS_BY_NAME
    out = ex._call(TOOLS_BY_NAME["generate_recovery_payment_link"],
                   ex.build_runtime(case(amount=7999)),
                   payment_id="pay_1", amount=799900,   # paise, passed as rupees
                   customer_email="a@b.com")
    assert out["status"] != "ok"
    assert "MORE than" in json.dumps(out)
    assert created == [], "the tool reached the gateway with a 100x amount"


def test_a_settled_record_builds_a_settled_case(env):
    rt = ex.build_runtime(case(status="recovered", recovered_amount=7999))
    assert rt.context.case.recovered is True
    assert rt.context.case.recovered_amount == 7999


# ── stopping rules ───────────────────────────────────────────────────────

def test_the_link_budget_stops_the_run_and_names_the_resource(env, monkeypatch):
    """A Razorpay test account allows 30 payment links for its lifetime and
    cancelling does not return one. The cap has to bind before the gateway's."""
    calls = Calls().install(monkeypatch)
    budget, spend = BatchBudget(max_links=2), ex.Spend()
    outcomes = [run(case(payment_id=f"pay_{i}"), budget=budget, spend=spend)
                for i in range(5)]
    assert calls.names.count("generate_recovery_payment_link") == 2
    assert [o.outcome for o in outcomes[2:]] == [ex.BUDGET] * 3
    assert outcomes[-1].reason == "budget_links"


def test_the_case_cap_stops_the_run(env, monkeypatch):
    calls = Calls().install(monkeypatch)
    budget, spend = BatchBudget(max_cases=1, max_links=99), ex.Spend()
    outs = [run(case(payment_id=f"pay_{i}"), budget=budget, spend=spend)
            for i in range(3)]
    assert [o.outcome for o in outs] == [ex.ACTED, ex.BUDGET, ex.BUDGET]
    assert calls.names.count("generate_recovery_payment_link") == 1


def test_a_budget_may_be_tightened_but_never_loosened():
    assert BatchBudget.from_request({"max_links": 2}).max_links == 2
    assert BatchBudget.from_request({"max_links": 500}).max_links == \
        BatchBudget().max_links


# ── idempotency: clicking twice must not spend twice ─────────────────────

def test_a_second_run_over_the_same_batch_acts_on_nothing(env, monkeypatch):
    from recovery_agent.state_store import StateStore
    calls = Calls().install(monkeypatch)
    store = StateStore()
    store.save_payment("pay_1", case())
    store.flush()

    assert run(case(), run_id="run_1").outcome == ex.ACTED
    record = dict(store.get_payment("pay_1") or {})
    assert record["batch_run_id"] == "run_1"

    before = len(calls.log)
    assert run(record, run_id="run_2").outcome == ex.SKIPPED
    assert len(calls.log) == before


# ── dry run: the same twelve checks, zero side effects ───────────────────

def test_a_dry_run_decides_identically_and_does_nothing(env, monkeypatch):
    calls = Calls().install(monkeypatch)
    records = [case(payment_id="pay_ok"),
               case(payment_id="pay_paid", amount=250),
               case(payment_id="pay_bare", customer={})]

    dry = [run(r, dry_run=True, spend=ex.Spend()) for r in records]
    assert calls.log == []
    assert [d.outcome for d in dry] == [ex.ACTED, ex.SKIPPED, ex.SKIPPED]
    assert dry[0].detail["dry_run"] is True

    live = [run(r, spend=ex.Spend()) for r in records]
    assert [d.outcome for d in live] == [d.outcome for d in dry]
    assert [d.charged_paise for d in live] == [d.charged_paise for d in dry]


# ── every decision leaves a trace ────────────────────────────────────────

def test_every_outcome_is_written_to_the_audit_log(env, monkeypatch):
    Calls().install(monkeypatch)
    run(case(payment_id="pay_acted"))
    run(case(payment_id="pay_skipped", amount=250))      # wrong band

    acted = [e["kind"] for e in audit.log().for_payment("pay_acted")]
    assert audit.ACTION_ATTEMPTED in acted and audit.ACTION_RESULT in acted
    skipped = audit.log().for_payment("pay_skipped")
    assert skipped[0]["kind"] == audit.CASE_SKIPPED
    assert skipped[0]["reason"] == "wrong_tier"
    assert audit.log().for_run("run_1")


def test_a_failed_link_is_an_exception_and_sends_no_message(env, monkeypatch):
    """Half an action is worse than none: no link means no email about a link."""
    calls = Calls().install(monkeypatch, link_status="error")
    d = run(case())
    assert d.outcome == ex.EXCEPTION
    assert calls.names == ["generate_recovery_payment_link"]


# ── what the executor is not allowed to do ───────────────────────────────

def test_the_executor_never_closes_or_escalates_a_case():
    """Closure and escalation live behind the ladder gating in the agent. A
    batch must not become a second door into either."""
    src = (__import__("pathlib").Path(ex.__file__)).read_text()
    assert "close_case" not in src.split('"""', 2)[2]
    assert "escalate_to_human" not in src.split('"""', 2)[2]
