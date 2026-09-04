"""The email allowance is a real limit the agent has to work inside.

The free Brevo plan sends 300 transactional emails a day, and the 301st does
not fail loudly — it simply never arrives, so a customer promised a payment
link never gets one and the case stalls looking like they ignored it. A batch
of two hundred failed payments can reach that ceiling in one click.

These tests pin the meter (which must work with or without an API key), the
rule that the worse of the two counts wins, and the budget guardrail that
turns the ceiling into something the agent is told about rather than
something it discovers.

Brevo is never called: every response here is a fixture.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from recovery_agent import email_quota
from recovery_agent.integrations import brevo_client

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.delenv("OUTBOX_DIR", raising=False)
    monkeypatch.delenv("BREVO_API_KEY", raising=False)
    monkeypatch.delenv("BREVO_DAILY_LIMIT", raising=False)
    monkeypatch.delenv("EMAIL_QUOTA_RESERVE", raising=False)
    brevo_client.reset_client()
    yield
    brevo_client.reset_client()


def _log(tmp_path, entries):
    path = tmp_path / "outbox" / "dispatch_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))


def _sent(when, channel="email", delivered=True):
    return {"payment_id": "p", "results": [
        {"channel": channel, "delivered": delivered,
         "timestamp": when.isoformat()}]}


# ── the local meter ─────────────────────────────────────────────────────

def test_todays_delivered_emails_are_counted(tmp_path):
    _log(tmp_path, [_sent(NOW), _sent(NOW - timedelta(hours=3)),
                    _sent(NOW - timedelta(days=1))])   # yesterday: not today
    assert email_quota.sent_today(now=NOW) == 2


def test_an_attempt_that_reached_nobody_spends_nothing(tmp_path):
    _log(tmp_path, [_sent(NOW), _sent(NOW, delivered=False)])
    assert email_quota.sent_today(now=NOW) == 1


def test_sms_does_not_count_against_the_email_allowance(tmp_path):
    _log(tmp_path, [_sent(NOW), _sent(NOW, channel="sms")])
    assert email_quota.sent_today(now=NOW) == 1


def test_a_missing_or_corrupt_log_never_raises(tmp_path):
    assert email_quota.sent_today(now=NOW) == 0
    path = tmp_path / "outbox" / "dispatch_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json\n" + json.dumps(_sent(NOW)) + "\n")
    assert email_quota.sent_today(now=NOW) == 1


# ── merging the provider's view ─────────────────────────────────────────

def test_without_an_api_key_the_local_count_stands_alone(tmp_path):
    _log(tmp_path, [_sent(NOW)] * 5)
    s = email_quota.status(now=NOW)
    assert s["source"] == "local"
    assert s["sent_today"] == 5
    assert s["sent_today_provider"] is None
    assert s["remaining"] == 295


def test_the_worse_of_the_two_counts_wins(tmp_path, monkeypatch):
    """Allowance spent by anything else on the account is still spent."""
    _log(tmp_path, [_sent(NOW)] * 5)
    monkeypatch.setattr(email_quota, "_provider_sent_today",
                        lambda now=None: (120, {"delivered": 118}))
    s = email_quota.status(now=NOW)
    assert s["sent_today"] == 120        # provider knows about more
    assert s["source"] == "provider+local"

    # ...and our own log wins when the provider lags behind it.
    monkeypatch.setattr(email_quota, "_provider_sent_today",
                        lambda now=None: (2, {}))
    assert email_quota.status(now=NOW)["sent_today"] == 5


def test_a_provider_outage_falls_back_instead_of_blocking(monkeypatch, tmp_path):
    _log(tmp_path, [_sent(NOW)] * 3)
    monkeypatch.setattr(email_quota, "_provider_sent_today",
                        lambda now=None: (None, {"status": "error"}))
    s = email_quota.status(now=NOW)
    assert s["sent_today"] == 3 and s["source"] == "local"


# ── the reserve ─────────────────────────────────────────────────────────

def test_the_reserve_is_held_back_from_the_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("BREVO_DAILY_LIMIT", "100")
    monkeypatch.setenv("EMAIL_QUOTA_RESERVE", "20")
    _log(tmp_path, [_sent(NOW)] * 75)
    s = email_quota.status(now=NOW)
    assert s["remaining"] == 25 and s["spendable"] == 5
    assert s["exhausted"] is False


def test_the_agent_is_stopped_before_the_ceiling_not_at_it(tmp_path, monkeypatch):
    monkeypatch.setenv("BREVO_DAILY_LIMIT", "100")
    monkeypatch.setenv("EMAIL_QUOTA_RESERVE", "20")
    _log(tmp_path, [_sent(NOW)] * 80)
    allowed, why = email_quota.may_send(now=NOW)
    assert allowed is False
    assert "held in reserve" in why
    # 20 sends still exist for a human or the demo.
    assert email_quota.status(now=NOW)["remaining"] == 20


def test_a_zero_reserve_disables_the_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("BREVO_DAILY_LIMIT", "10")
    monkeypatch.setenv("EMAIL_QUOTA_RESERVE", "0")
    _log(tmp_path, [_sent(NOW)] * 50)
    assert email_quota.may_send(now=NOW)[0] is True


# ── the guardrail on the tool path ──────────────────────────────────────

def test_the_policy_gate_refuses_a_send_with_no_allowance_left(monkeypatch):
    from recovery_agent.agent import policy_gate
    monkeypatch.setattr(policy_gate, "log_verdict", lambda *a, **k: None)
    monkeypatch.setattr(policy_gate, "_record_refusal", lambda *a, **k: None)
    monkeypatch.setattr(email_quota, "may_send",
                        lambda *a, **k: (False, "today's email allowance is spent"))
    refusal = policy_gate._budget_refusal("send_recovery_notification", "pay_1")
    assert refusal["guardrail"] == "email_budget"
    assert "retry_in_hours" in refusal["guidance"]
    assert "show_page_offer" in refusal["guidance"]


def test_the_budget_only_governs_tools_that_spend_it(monkeypatch):
    from recovery_agent.agent import policy_gate
    monkeypatch.setattr(email_quota, "may_send", lambda *a, **k: (False, "spent"))
    for tool in ("initiate_voice_call", "generate_recovery_payment_link",
                 "retry_in_hours", "send_page_push"):
        assert policy_gate._budget_refusal(tool, "pay_1") is None


def test_a_broken_meter_never_blocks_a_recovery(monkeypatch):
    from recovery_agent.agent import policy_gate

    def boom(*a, **k):
        raise RuntimeError("meter down")
    monkeypatch.setattr(email_quota, "may_send", boom)
    assert policy_gate._budget_refusal("send_recovery_notification", "p") is None


# ── the Brevo client is read-only and degrades quietly ──────────────────

def test_without_a_key_every_read_skips_rather_than_failing():
    c = brevo_client.BrevoClient(api_key="")
    assert c.can_read is False
    for out in (c.get_account(), c.get_smtp_statistics()):
        assert out["status"] == "skipped"
        assert "not an API key" in out["detail"] or "BREVO_API_KEY" in out["detail"]


def test_the_smtp_relay_password_is_not_an_api_key():
    """`xsmtpsib-…` authenticates the relay, not the v3 API — a distinction
    that silently returns 401s if confused."""
    doc = brevo_client.__doc__ or ""
    assert "xsmtpsib" in doc and "xkeysib" in doc


def test_statistics_are_parsed_into_plain_day_rows(monkeypatch):
    payload = {"reports": [{"date": "2026-09-04", "requests": 12,
                            "delivered": 11, "hardBounces": 1,
                            "softBounces": 0, "blocked": 0}]}
    c = brevo_client.BrevoClient(api_key="k")
    monkeypatch.setattr(c, "_get", lambda *a, **k: {"status": "ok", "data": payload})
    out = c.get_smtp_statistics(days=1)
    assert out["reports"][0]["requests"] == 12
    assert out["reports"][0]["hard_bounces"] == 1


def test_account_credits_are_extracted_from_the_plan(monkeypatch):
    payload = {"email": "a@b.com", "companyName": "X",
               "plan": [{"type": "free", "creditsType": "sendLimit",
                         "credits": 300}]}
    c = brevo_client.BrevoClient(api_key="k")
    monkeypatch.setattr(c, "_get", lambda *a, **k: {"status": "ok", "data": payload})
    out = c.get_account()
    assert out["email_credits"] == 300
    assert out["plans"][0]["type"] == "free"
