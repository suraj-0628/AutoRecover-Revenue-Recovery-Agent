"""D4 gate tests for the recovery sensor — the keystone block.

The gate: *pay the offer by hand, and the case flips to RECOVERED with no human
touching the system.* That end-to-end run is in the live script; here the same
path is exercised against a fake Razorpay so every branch is covered.

Two properties matter beyond "it detects payment":

* **Silence is normal.** A sensor polling every 15s must write nothing while
  nothing changes, or the event log drowns in "still pending".
* **The sensor never decides strategy.** A dead offer goes back on the work
  queue; choosing what to do next belongs to the agent (D6).
"""
from __future__ import annotations

import pytest

from recovery_agent.effectors import RecoveryOrderEffector, send_recovery_offer
from recovery_agent.ledger import EventKind, Ledger, to_paise
from recovery_agent.models import CaseStatus
from recovery_agent.sensor import (
    CANCELLED,
    EXPIRED,
    PAID,
    PARTIALLY_PAID,
    PENDING,
    OrderProbe,
    PaymentLinkProbe,
    RecoverySensor,
)
from tests.test_effectors import FakeOrderAPI


class PayableOrderAPI(FakeOrderAPI):
    """FakeOrderAPI that can be told an order got paid."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.paid: dict[str, tuple[int, str]] = {}   # order_id -> (paise, payment_id)
        self.fetch_error: Exception | None = None

    def pay(self, order_id: str, paise: int, payment_id: str = "pay_recovery_1"):
        self.paid[order_id] = (paise, payment_id)

    def fetch(self, order_id):
        if self.fetch_error:
            raise self.fetch_error
        for o in self._by_receipt.values():
            if o["id"] == order_id:
                out = dict(o)
                if order_id in self.paid:
                    paise, _ = self.paid[order_id]
                    out["amount_paid"] = paise
                    out["status"] = "paid" if paise >= out["amount"] else "attempted"
                else:
                    out["amount_paid"] = 0
                return out
        raise Exception("order not found")

    def payments(self, order_id):
        if order_id not in self.paid:
            return {"count": 0, "items": []}
        paise, pid = self.paid[order_id]
        return {"count": 1, "items": [
            {"id": pid, "status": "captured", "amount": paise, "method": "card"}
        ]}


@pytest.fixture
def ledger(tmp_path):
    return Ledger(db_path=tmp_path / "ledger.db")


@pytest.fixture
def api():
    return PayableOrderAPI()


@pytest.fixture
def sensor(ledger, api):
    return RecoverySensor(ledger=ledger, probes=[OrderProbe(api=api)])


@pytest.fixture
def awaiting(ledger, api):
    """A case with a real outstanding recovery offer."""
    case = ledger.open_case(payment_id="pay_orig", amount_paise=to_paise(2999),
                            customer_id="a@b.com", failure_code="card_expired")
    return send_recovery_offer(ledger, case,
                               effector=RecoveryOrderEffector(api=api))


def _order_id(ledger, case_id):
    for ev in reversed(ledger.events(case_id)):
        if ev.kind is EventKind.ATTEMPT and ev.result == "ok":
            return ev.payload["receipt"]["order_id"]
    raise AssertionError("no order")


# ══ THE GATE ═══════════════════════════════════════════════════════════════

def test_paying_the_offer_flips_the_case_to_recovered(ledger, api, sensor, awaiting):
    """D4 gate: money moves, and the case closes itself."""
    assert awaiting.status == CaseStatus.AWAITING_CUSTOMER

    api.pay(_order_id(ledger, awaiting.case_id), to_paise(2999), "pay_recovery_99")
    run = sensor.poll_once()

    got = ledger.require_case(awaiting.case_id)
    assert got.status == CaseStatus.RECOVERED
    assert got.is_terminal
    assert got.recovery_payment_id == "pay_recovery_99"
    assert got.recovered_amount_paise == 299900
    assert got.attributed_to_agent is True
    assert run.recovered == 1
    assert ledger.verify(got.case_id)


def test_recovery_links_a_new_payment_not_the_original(ledger, api, sensor, awaiting):
    api.pay(_order_id(ledger, awaiting.case_id), to_paise(2999), "pay_recovery_99")
    sensor.poll_once()
    got = ledger.require_case(awaiting.case_id)
    assert got.payment_id == "pay_orig"            # original stays failed forever
    assert got.recovery_payment_id == "pay_recovery_99"


# ══ Silence is normal ══════════════════════════════════════════════════════

def test_polling_an_unpaid_case_writes_nothing(ledger, sensor, awaiting):
    before = len(ledger.events(awaiting.case_id))
    seq_before = awaiting.seq

    for _ in range(25):
        run = sensor.poll_once()

    assert len(ledger.events(awaiting.case_id)) == before, "sensor wrote noise"
    assert ledger.require_case(awaiting.case_id).seq == seq_before
    assert run.unchanged == 1 and run.changed == 0


def test_repeated_partial_payment_is_recorded_once(ledger, api, sensor, awaiting):
    api.pay(_order_id(ledger, awaiting.case_id), to_paise(1000), "pay_part")
    for _ in range(5):
        sensor.poll_once()

    obs = [e for e in ledger.events(awaiting.case_id) if e.kind is EventKind.OBSERVATION]
    assert len(obs) == 1
    assert obs[0].result == PARTIALLY_PAID


def test_a_changed_partial_amount_is_recorded_again(ledger, api, sensor, awaiting):
    oid = _order_id(ledger, awaiting.case_id)
    api.pay(oid, to_paise(1000), "pay_part")
    sensor.poll_once()
    api.pay(oid, to_paise(2000), "pay_part")
    sensor.poll_once()
    obs = [e for e in ledger.events(awaiting.case_id) if e.kind is EventKind.OBSERVATION]
    assert len(obs) == 2


# ══ Partial payment is not recovery ════════════════════════════════════════

def test_partial_payment_keeps_the_case_open(ledger, api, sensor, awaiting):
    api.pay(_order_id(ledger, awaiting.case_id), to_paise(1000), "pay_part")
    run = sensor.poll_once()

    got = ledger.require_case(awaiting.case_id)
    assert got.status == CaseStatus.AWAITING_CUSTOMER
    assert got.recovered_amount_paise == 0
    assert run.recovered == 0 and run.changed == 1
    assert ledger.verify(got.case_id)


def test_overpayment_still_counts_as_recovered(ledger, api, sensor, awaiting):
    api.pay(_order_id(ledger, awaiting.case_id), to_paise(3500), "pay_over")
    sensor.poll_once()
    assert ledger.require_case(awaiting.case_id).status == CaseStatus.RECOVERED


# ══ Errors must never look like an outcome ═════════════════════════════════

def test_api_failure_changes_nothing(ledger, api, sensor, awaiting):
    api.fetch_error = Exception("razorpay 503")
    before = len(ledger.events(awaiting.case_id))
    run = sensor.poll_once()

    assert run.errors == 1 and run.changed == 0
    assert len(ledger.events(awaiting.case_id)) == before
    assert ledger.require_case(awaiting.case_id).status == CaseStatus.AWAITING_CUSTOMER


def test_a_case_with_no_receipt_is_an_error_not_a_recovery(ledger, sensor):
    case = ledger.open_case(payment_id="pay_bare", amount_paise=to_paise(10))
    obs = sensor.observe(case)
    assert obs.is_error and "no receipt" in obs.error


def test_unknown_receipt_shape_is_an_error(ledger, sensor):
    case = ledger.open_case(payment_id="pay_weird", amount_paise=to_paise(10))
    ledger.record_transition(case.case_id, CaseStatus.ACTING)
    ledger.act(case.case_id, action="mystery", to_status=CaseStatus.AWAITING_CUSTOMER,
               result="ok", receipt={"carrier_pigeon_id": "p1"})
    obs = sensor.observe(ledger.require_case(case.case_id))
    assert obs.is_error and "no probe handles" in obs.error


# ══ The sensor does not decide strategy ════════════════════════════════════

def test_a_dead_offer_goes_back_on_the_work_queue(ledger, tmp_path):
    """Expired offer -> SCHEDULED(now), for the agent to reconsider. Not STOPPED."""
    from tests.test_effectors import FakeLinkAPI
    from recovery_agent.effectors import PaymentLinkEffector

    class ExpiringLinkAPI(FakeLinkAPI):
        def fetch(self, link_id):
            link = dict(super().fetch(link_id))
            link["status"] = "expired"
            return link

    led = Ledger(db_path=tmp_path / "l2.db")
    api = ExpiringLinkAPI()
    case = led.open_case(payment_id="pay_exp", amount_paise=to_paise(500))
    send_recovery_offer(led, case, effector=PaymentLinkEffector(api=api))

    sensor = RecoverySensor(ledger=led, probes=[PaymentLinkProbe(api=api)])
    run = sensor.poll_once()

    got = led.require_case(case.case_id)
    assert got.status == CaseStatus.SCHEDULED
    assert got.wake_at is not None
    assert not got.is_terminal, "the sensor must not close a case the agent could retry"
    assert run.dead == 1
    assert {c.case_id for c in led.due_cases()} == {case.case_id}
    assert led.verify(got.case_id)


# ══ Queue behaviour ════════════════════════════════════════════════════════

def test_only_awaiting_cases_are_polled(ledger, api, sensor, awaiting):
    ledger.open_case(payment_id="pay_open", amount_paise=to_paise(10))
    other = ledger.open_case(payment_id="pay_sched", amount_paise=to_paise(10))
    from datetime import datetime, timezone
    ledger.record_transition(other.case_id, CaseStatus.SCHEDULED,
                             wake_at=datetime.now(timezone.utc))
    assert sensor.poll_once().checked == 1


def test_recovered_cases_are_not_polled_again(ledger, api, sensor, awaiting):
    api.pay(_order_id(ledger, awaiting.case_id), to_paise(2999))
    sensor.poll_once()
    assert sensor.poll_once().checked == 0


def test_run_forever_is_bounded_for_tests(ledger, api, sensor, awaiting):
    total = sensor.run_forever(interval=0, iterations=3)
    assert total.checked == 3 and total.changed == 0


def test_two_sensors_do_not_double_record(ledger, api, awaiting):
    """Second writer loses on expected_seq rather than duplicating the recovery."""
    a = RecoverySensor(ledger=ledger, probes=[OrderProbe(api=api)])
    b = RecoverySensor(ledger=ledger, probes=[OrderProbe(api=api)])
    api.pay(_order_id(ledger, awaiting.case_id), to_paise(2999), "pay_rec")

    stale = ledger.require_case(awaiting.case_id)
    run_a = a.poll_case(stale)
    run_b = b.poll_case(stale)          # same stale snapshot

    got = ledger.require_case(awaiting.case_id)
    assert got.status == CaseStatus.RECOVERED
    recs = [e for e in ledger.events(got.case_id)
            if e.kind is EventKind.TRANSITION and e.to_status is CaseStatus.RECOVERED]
    assert len(recs) == 1, "recovery recorded twice"
    assert ledger.verify(got.case_id)
