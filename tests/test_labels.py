"""The operator's verdict loop: label why it did not land, re-bin, run again.

The one thing these tests guard hardest is the money rule: no label — no
sequence of labels, no click — can add a rupee to recovered totals. A batch
report a human can inflate is not a measurement.
"""
import json
from datetime import datetime, timezone

import pytest

from recovery_agent import audit, labels
from recovery_agent.agent.classify import classify, failure_kind


@pytest.fixture()
def web(tmp_path, monkeypatch):
    from recovery_agent import frontend, state_store
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path, raising=False)
    state_store.StateStore.reset_instances()
    audit.AuditLog.reset_instances()
    audit._default = None
    store = state_store.StateStore()
    monkeypatch.setattr(frontend, "store", store, raising=False)
    monkeypatch.setattr(frontend, "push_event", lambda *a, **k: None)
    client = frontend.app.test_client()
    yield type("Web", (), {"client": client, "store": store})()
    state_store.StateStore.reset_instances()
    audit.AuditLog.reset_instances()
    audit._default = None


def case(store, pid="pay_1", **kw):
    rec = {"payment_id": pid, "amount": 2499.0, "status": "failed",
           "decline_strategy": "card_expired",
           "customer": {"email": "a@b.com", "name": "Asha"}, **kw}
    store.save_payment(pid, rec)
    store.flush()
    return rec


def label(web, pid, code, note=""):
    return web.client.post(f"/api/cases/{pid}/label",
                           json={"code": code, "note": note})


# ── the vocabulary itself ────────────────────────────────────────────────

def test_there_is_no_label_that_marks_money_recovered():
    """Success verdicts belong to the gateway alone. The vocabulary must not
    even be able to express one."""
    for entry in labels.choices():
        assert entry["code"] != "recovered"
        assert "success" not in entry["code"]
        if entry.get("settles"):
            assert entry["code"] == "paid_outside", (
                "only the outside-rail close may settle, and it counts in "
                "its own column")


def test_every_label_lands_somewhere_deliberate():
    """Each verdict either re-bins (a kind), routes to the agent, opts out,
    or settles outside — never nothing, never more than one destiny unclear."""
    for entry in labels.choices():
        assert entry.get("kind") or entry.get("to_agent") \
            or entry.get("opts_out") or entry.get("settles"), entry["code"]
        assert entry.get("next"), f"{entry['code']} does not say what happens"


# ── precedence: the newest testimony wins ────────────────────────────────

def test_a_label_outranks_the_customer_s_old_answer():
    """The drop-reason describes intent before any attempt; the label
    describes what happened to the latest one. Recency wins."""
    rec = {"payment_id": "p", "amount": 100, "status": "failed",
           "decline_strategy": "card_expired",
           "drop_reason": {"code": "no_balance"},
           "operator_label": {"code": "instrument_broken"},
           "customer": {"email": "a@b.com"}}
    assert failure_kind(rec) == "method"
    rec.pop("operator_label")
    assert failure_kind(rec) == "funds"


@pytest.mark.parametrize("code,batch", [
    ("promised_later", "insufficient_funds"),
    ("instrument_broken", "bank_declined"),
    ("checkout_glitch", "transient"),
    ("dispute", "risk"),
    ("other", "unclassified"),
])
def test_a_label_re_bins_the_case(code, batch):
    rec = {"payment_id": "p", "amount": 100, "status": "failed",
           "decline_strategy": "user_dropoff",
           "operator_label": {"code": code},
           "customer": {"email": "a@b.com"}}
    assert classify(rec) == batch


# ── the endpoint ─────────────────────────────────────────────────────────

def test_labeling_writes_the_record_and_the_log(web):
    case(web.store, batch_run_id="run_x")
    r = label(web, "pay_1", "promised_later")
    assert r.status_code == 200
    assert r.get_json()["rebins_to"] == "insufficient_funds"

    rec = web.store.get_payment("pay_1")
    assert rec["operator_label"]["code"] == "promised_later"
    events = [e for e in audit.log().for_payment("pay_1")
              if e["kind"] == audit.CASE_LABELED]
    assert len(events) == 1
    assert events[0]["actor"] == "operator"
    assert events[0]["batch_run_id"] == "run_x"


def test_do_not_contact_bars_pursuit_everywhere(web):
    from recovery_agent.agent.ladder import pursuit_barred
    case(web.store)
    label(web, "pay_1", "do_not_contact")
    rec = web.store.get_payment("pay_1")
    assert rec["opted_out"] is True
    assert pursuit_barred(rec)


def test_paid_outside_closes_but_never_counts_as_recovered_money(web):
    """The money rule, end to end: the case leaves the batches, the audit
    trail says so, and the run's recovered total does not move a paisa."""
    case(web.store, batch_run_id="run_x")
    before = audit.log().recovered_paise("run_x")
    r = label(web, "pay_1", "paid_outside", note="bank transfer, ref 4471")
    assert r.status_code == 200

    rec = web.store.get_payment("pay_1")
    assert rec["status"] == "settled_outside"
    assert classify(rec) is None, "it has left the batches"
    assert not rec.get("recovered_amount"), "claimed money is not recorded money"
    assert audit.log().recovered_paise("run_x") == before == 0


def test_other_requires_the_note_it_exists_for(web):
    case(web.store)
    assert label(web, "pay_1", "other").status_code == 400
    assert label(web, "pay_1", "other", note="wants COD").status_code == 200


def test_an_unknown_label_is_refused(web):
    case(web.store)
    assert label(web, "pay_1", "it_worked_trust_me").status_code == 400
    assert label(web, "missing", "dispute").status_code == 404


# ── the verdict queue ────────────────────────────────────────────────────

def test_only_acted_and_unpaid_cases_need_a_verdict(web):
    now = datetime.now(timezone.utc).isoformat()
    case(web.store, pid="pay_acted", batch_run_id="run_x",
         batch_attributed_at=now)
    case(web.store, pid="pay_untouched")
    case(web.store, pid="pay_paid", batch_run_id="run_x",
         batch_attributed_at=now, status="recovered")

    got = web.client.get("/api/batch-verdicts").get_json()["verdicts"]
    assert [v["payment_id"] for v in got] == ["pay_acted"]


def test_a_verdict_already_given_drops_off_the_queue(web):
    now = datetime.now(timezone.utc).isoformat()
    case(web.store, pid="pay_1", batch_run_id="run_x",
         batch_attributed_at=now)
    label(web, "pay_1", "promised_later")
    got = web.client.get("/api/batch-verdicts").get_json()["verdicts"]
    assert got == []


# ── the loop closes: label, re-bin, next wave treats accordingly ─────────

def test_a_judgement_label_goes_to_the_desk_not_a_bin(web):
    from recovery_agent.batch import waves
    case(web.store, decline_strategy="user_dropoff")
    label(web, "pay_1", "too_expensive")

    cycle = waves.WaveCycle(payment_ids=["pay_1"],
                            config=waves.WaveConfig(max_waves=1,
                                                    settle_seconds=0))
    report = cycle.execute()
    assert [e["payment_id"] for e in report["exceptions"]] == ["pay_1"]
    assert "labeled by a person" in report["exceptions"][0]["why"]


def test_the_activity_feed_carries_the_label(web):
    case(web.store, batch_run_id="run_x")
    label(web, "pay_1", "dispute")
    feed = web.client.get("/api/batch-activity").get_json()
    kinds = [e["kind"] for e in feed["events"]]
    assert audit.CASE_LABELED in kinds
    assert feed["cursor"] > 0
    later = web.client.get(
        f"/api/batch-activity?since={feed['cursor']}").get_json()
    assert later["events"] == []
