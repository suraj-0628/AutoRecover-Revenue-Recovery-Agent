"""Billed costs are invoice lines, not our arithmetic — and must say so.

`economics.py` prices what we metered against rates somebody chose. Voice is
the one surface where the provider will tell us the real figure, so it is the
one place "is that number real?" has a strong answer. These tests pin the
ledger that keeps the two kinds of number apart, the reconciliation that can
be polled without double-counting, and the rule that an invoice line always
beats an estimate.

Every SuperU interaction here is a fixture. The real client is never called:
credits are reserved for the judges' demo.
"""
import json

import pytest

from recovery_agent import cost_ledger
from recovery_agent.integrations import superu_reconcile as rec

# One page of SuperU's real response shape, values taken from an actual
# read of the account (a 68s call, a 3.5s call, a zero-length hangup, and
# one call this system did not place).
CALLS = [
    {"id": "call-a", "campaign_id": "recovery_pay_aaa", "cost": 0.04,
     "telecom_total_cost": 1.3656692799999999, "call_duration_seconds": 68.3,
     "status": "ended", "endedReason": "Customer Hangup"},
    {"id": "call-b", "campaign_id": "recovery_pay_bbb", "cost": 0.02,
     "telecom_total_cost": 0.07039296, "call_duration_seconds": 3.5,
     "status": "ended", "endedReason": "Customer Hangup"},
    {"id": "call-c", "campaign_id": "recovery_pay_ccc", "cost": 0.0,
     "telecom_total_cost": 0.0, "call_duration_seconds": 0.0,
     "status": "ended", "endedReason": "Customer Hangup"},
    {"id": "call-d", "campaign_id": "test", "cost": 0.02,
     "telecom_total_cost": 0.5, "call_duration_seconds": 10.0,
     "status": "ended", "endedReason": "Agent Hangup"},
]


class FakeClient:
    """Answers log reads from a fixture and has no way to place a call."""

    def __init__(self, calls=None, status="ok"):
        self.calls = CALLS if calls is None else calls
        self.status = status
        self.reads = 0

    can_read = True

    def get_call_logs(self, **kw):
        self.reads += 1
        if self.status != "ok":
            return {"status": self.status, "error": "boom", "calls": []}
        return {"status": "ok", "calls": self.calls, "total": len(self.calls),
                "total_cost": sum(c["cost"] for c in self.calls)}


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setenv("USD_TO_INR", "88")
    monkeypatch.delenv("SUPERU_COST_CURRENCY", raising=False)
    monkeypatch.delenv("SUPERU_TELECOM_CURRENCY", raising=False)
    yield


# ── the ledger ──────────────────────────────────────────────────────────

def test_an_event_records_its_provenance(tmp_path):
    cost_ledger.record(cost_ledger.SURFACE_VOICE, 4.89, cost_ledger.BILLED,
                       payment_id="pay_1", source_ref="call-a")
    events = cost_ledger.read_events()
    assert len(events) == 1
    assert events[0]["provenance"] == "BILLED"
    assert events[0]["inr"] == 4.89
    assert events[0]["surface"] == "voice"


def test_the_same_source_is_never_recorded_twice():
    """The reconciler is meant to be polled. Double-billing a case because a
    timer fired twice would make the whole ledger worthless."""
    assert cost_ledger.record("voice", 4.89, "BILLED", source_ref="call-a")
    assert not cost_ledger.record("voice", 4.89, "BILLED", source_ref="call-a")
    assert len(cost_ledger.read_events()) == 1


def test_events_without_a_source_ref_are_always_kept():
    """Only provider-keyed events dedupe; a measured event has no invoice."""
    assert cost_ledger.record("llm", 1.0, "MEASURED")
    assert cost_ledger.record("llm", 1.0, "MEASURED")
    assert len(cost_ledger.read_events()) == 2


def test_totals_roll_up_per_payment_and_surface():
    cost_ledger.record("voice", 2.0, "BILLED", payment_id="p1", source_ref="a")
    cost_ledger.record("voice", 3.0, "BILLED", payment_id="p1", source_ref="b")
    cost_ledger.record("voice", 5.0, "BILLED", payment_id="p2", source_ref="c")
    cost_ledger.record("llm", 9.0, "MEASURED", payment_id="p1")
    assert cost_ledger.by_payment("voice") == {"p1": 5.0, "p2": 5.0}
    summary = cost_ledger.summarise()
    assert summary["surfaces"]["voice"]["inr"] == 10.0
    assert summary["surfaces"]["voice"]["provenance"] == {"BILLED": 3}
    assert summary["billed_inr"] == 10.0


def test_a_corrupt_line_does_not_poison_the_ledger():
    cost_ledger.record("voice", 1.0, "BILLED", source_ref="ok")
    with open(cost_ledger.ledger_path(), "a") as f:
        f.write("{not json\n")
    assert len(cost_ledger.read_events()) == 1


# ── converting SuperU's two meters ──────────────────────────────────────

def test_only_the_platform_charge_is_billed_to_us():
    """SuperU's published price is $0.02 per connected minute with telephony
    INCLUDED, and `cost` matches that meter exactly. `telecom_total_cost` is
    their own carrier bill — roughly 4x the platform charge — so adding it
    would quadruple what a merchant appears to have spent."""
    inr, raw = rec.call_cost_inr(CALLS[0])       # 68.3s -> two minutes
    assert inr == pytest.approx(0.04 * 88, abs=0.01)
    assert raw["cost"] == 0.04
    # The carrier figure is kept as evidence, flagged as not ours to pay.
    assert raw["telecom_total_cost"] == pytest.approx(1.36566928)
    assert raw["telecom_billed_to_us"] is False


def test_every_sub_minute_call_costs_the_same_one_minute_charge():
    """The meter is per connected MINUTE, so a 3s call and a 27s call are
    billed identically — which is what the observed data shows."""
    short, _ = rec.call_cost_inr(CALLS[1])       # 3.5s
    assert short == pytest.approx(0.02 * 88, abs=0.01)


def test_telecom_can_be_included_if_a_plan_ever_passes_it_through(monkeypatch):
    monkeypatch.setenv("SUPERU_INCLUDE_TELECOM", "1")
    inr, raw = rec.call_cost_inr(CALLS[1])
    assert raw["telecom_billed_to_us"] is True
    assert inr == pytest.approx(0.02 * 88 + 0.07039296, abs=0.01)


def test_the_raw_provider_figures_are_kept_verbatim():
    """SuperU labels neither meter with a currency, so our conversion is an
    assumption. Keeping the raw values means a wrong assumption is fixed by
    re-deriving, not by re-fetching."""
    _, raw = rec.call_cost_inr(CALLS[1])
    assert raw["cost_currency"] == "USD"
    assert raw["telecom_currency"] == "INR"
    assert raw["usd_to_inr"] == 88.0
    assert raw["call_duration_seconds"] == 3.5


def test_the_conversion_rate_is_configurable(monkeypatch):
    monkeypatch.setenv("USD_TO_INR", "90")
    inr, raw = rec.call_cost_inr(CALLS[1])
    assert raw["usd_to_inr"] == 90.0
    assert inr == pytest.approx(0.02 * 90, abs=0.01)


def test_a_free_call_costs_nothing_but_is_still_recorded():
    inr, _ = rec.call_cost_inr(CALLS[2])
    assert inr == 0.0


def test_only_calls_this_system_placed_map_to_a_case():
    assert rec.payment_id_from_campaign("recovery_pay_xyz") == "pay_xyz"
    assert rec.payment_id_from_campaign("test") == ""
    assert rec.payment_id_from_campaign(None) == ""


# ── reconciliation ──────────────────────────────────────────────────────

def test_reconcile_records_billed_costs_against_the_right_cases():
    out = rec.reconcile(client=FakeClient())
    assert out["status"] == "ok"
    assert out["recorded"] == 3          # a, b, c — not the "test" campaign
    assert out["unmatched"] == 1
    billed = cost_ledger.by_payment("voice")
    assert set(billed) == {"pay_aaa", "pay_bbb", "pay_ccc"}
    assert billed["pay_aaa"] == pytest.approx(0.04 * 88, abs=0.02)


def test_reconcile_is_safe_to_run_on_a_timer():
    client = FakeClient()
    first = rec.reconcile(client=client)
    second = rec.reconcile(client=client)
    assert first["recorded"] == 3 and second["recorded"] == 0
    assert second["already_known"] == 3
    assert len(cost_ledger.read_events()) == 3


def test_a_failed_read_reports_instead_of_recording_zeroes():
    out = rec.reconcile(client=FakeClient(status="error"))
    assert out["status"] == "error"
    assert out["recorded"] == 0
    assert cost_ledger.read_events() == []


# ── an invoice beats an estimate ────────────────────────────────────────

def _case(pid="pay_aaa", **over):
    rec_ = {"payment_id": pid, "amount": 5000.0, "status": "recovered",
            "recovered_amount": 5000.0,
            "contacts": [{"channel": "voice", "at": "t"}]}
    rec_.update(over)
    return rec_


def test_billed_voice_cost_replaces_the_per_call_estimate(monkeypatch):
    from recovery_agent import economics
    monkeypatch.setenv("VOICE_COST_INR_PER_CALL", "15")
    estimated = economics.case_economics(_case())
    assert estimated["voice_cost"] == 15.0
    assert estimated["voice_provenance"] == "MEASURED"

    billed = economics.case_economics(_case(), billed_voice={"pay_aaa": 4.89})
    assert billed["voice_cost"] == 4.89
    assert billed["voice_provenance"] == "BILLED"
    assert billed["total_cost"] < estimated["total_cost"]


def test_an_invoice_counts_even_when_the_case_never_logged_the_contact():
    """Calls placed before contact tracking existed still cost money."""
    from recovery_agent import economics
    out = economics.case_economics(_case(contacts=[]),
                                   billed_voice={"pay_aaa": 2.31})
    assert out["voice_cost"] == 2.31
    assert out["voice_provenance"] == "BILLED"


def test_a_case_with_no_voice_at_all_claims_no_provenance():
    from recovery_agent import economics
    out = economics.case_economics(_case(contacts=[{"channel": "email", "at": "t"}]))
    assert out["voice_cost"] == 0.0
    assert out["voice_provenance"] == "NONE"


def test_the_summary_reports_how_much_of_it_is_billed():
    from recovery_agent import economics
    rec.reconcile(client=FakeClient())
    out = economics.summarise([_case("pay_aaa"), _case("pay_zzz")])
    prov = out["provenance"]
    assert prov["voice_billed_cases"] == 1
    assert prov["voice_billed_inr"] == pytest.approx(0.04 * 88, abs=0.02)
    assert prov["ledger"]["surfaces"]["voice"]["provenance"] == {"BILLED": 3}


def test_billed_spend_survives_the_loss_of_its_case_record():
    """A purged data dir does not un-spend the money. Counting only what
    still has a record would understate real spend — the exact opposite of
    what this view is for."""
    from recovery_agent import economics
    rec.reconcile(client=FakeClient())          # bills pay_aaa/bbb/ccc
    out = economics.summarise([_case("pay_aaa")])   # only one still exists
    prov = out["provenance"]
    assert prov["voice_billed_cases"] == 1
    assert prov["voice_billed_inr"] == pytest.approx(0.04 * 88, abs=0.02)
    # pay_bbb and pay_ccc are gone, but their charges are still reported.
    assert prov["voice_billed_unattributed_inr"] > 0
    assert prov["voice_billed_total_inr"] == pytest.approx(
        prov["voice_billed_inr"] + prov["voice_billed_unattributed_inr"],
        abs=0.02)


# ── the client never spends credits on a read ───────────────────────────

def test_the_log_read_is_not_the_call_endpoint():
    from recovery_agent.integrations import superu_client as sc
    assert sc.SUPERU_CALL_LOGS_URL.endswith("/call-logs")
    assert "create_call" not in sc.SUPERU_CALL_LOGS_URL


def test_reading_logs_needs_only_an_api_key(monkeypatch):
    """A deployment that cannot place calls can still account for the ones it
    already placed."""
    from recovery_agent.integrations.superu_client import SuperUClient
    monkeypatch.setenv("SUPERU_API_KEY", "k")
    monkeypatch.setenv("SUPERU_ASSISTANT_ID", "")
    monkeypatch.setenv("SUPERU_FROM_PHONE", "")
    c = SuperUClient()
    assert c.can_read is True
    assert c.is_enabled is False


def test_an_unconfigured_client_skips_rather_than_failing(monkeypatch):
    from recovery_agent.integrations.superu_client import SuperUClient
    monkeypatch.setenv("SUPERU_API_KEY", "")
    out = SuperUClient().get_call_logs()
    assert out["status"] == "skipped"
    assert out["calls"] == []
