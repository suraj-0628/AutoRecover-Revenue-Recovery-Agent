"""D3 gate tests for the payment-link effector.

Two properties dominate, both taken straight from the audit:

* **No fabrication.** An unconfigured client raises instead of inventing a
  plausible-but-dead payment link (AUDIT-FINDINGS S3-4).
* **Paise in, paise out.** The old chain multiplied by 100 twice, so a Rs 2,999
  link was created for Rs 29,99,000 — a 10,000x error.

Everything here runs against an injected fake API. The live Razorpay call is in
`test_effectors_live.py`, which is skipped unless credentials are present.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recovery_agent.effectors import (
    DEFAULT_LINK_TTL,
    EffectorError,
    NotConfigured,
    PaymentLinkEffector,
    last_receipt,
    reference_id_for,
    send_recovery_link,
)
from recovery_agent.ledger import EventKind, Ledger, to_paise
from recovery_agent.models import CaseStatus
from recovery_agent.statemachine import MissingEvidence


class FakeLinkAPI:
    """Records what was sent and returns Razorpay-shaped responses."""

    def __init__(self, fail_with: Exception | None = None):
        self.created: list[dict] = []
        self.fail_with = fail_with
        self._by_ref: dict[str, dict] = {}
        self._n = 0

    def create(self, params):
        if self.fail_with is not None:
            raise self.fail_with
        ref = params.get("reference_id", "")
        if ref in self._by_ref:
            raise Exception("BAD_REQUEST_ERROR: reference_id already exists")
        self.created.append(params)
        self._n += 1
        link = {
            "id": f"plink_fake{self._n}",
            "short_url": f"https://rzp.io/i/fake{self._n}",
            "reference_id": ref,
            "status": "created",
            "amount": params["amount"],
            "currency": params["currency"],
            "expire_by": params.get("expire_by"),
        }
        self._by_ref[ref] = link
        return link

    def fetch(self, link_id):
        for link in self._by_ref.values():
            if link["id"] == link_id:
                return link
        raise Exception("not found")

    def all(self, params=None):
        ref = (params or {}).get("reference_id")
        items = [self._by_ref[ref]] if ref in self._by_ref else []
        return {"count": len(items), "items": items}


@pytest.fixture
def ledger(tmp_path):
    return Ledger(db_path=tmp_path / "ledger.db")


@pytest.fixture
def case(ledger):
    return ledger.open_case(
        payment_id="pay_orig_1", amount_paise=to_paise(2999),
        customer_id="rahul@example.com", failure_code="card_expired",
        metadata={"customer_name": "Rahul", "customer_phone": "+919999999999"},
    )


# ── No fabrication ───────────────────────────────────────────────────────────

def test_unconfigured_client_raises_instead_of_faking(case):
    """S3-4: the old client invented a real-looking but dead rzp.io URL."""
    class Unconfigured:
        is_configured = False

    eff = PaymentLinkEffector(client=Unconfigured())
    with pytest.raises(NotConfigured, match="does not simulate"):
        eff.run(case)


def test_api_failure_is_reported_not_invented(case):
    eff = PaymentLinkEffector(api=FakeLinkAPI(fail_with=Exception("gateway down")))
    result = eff.run(case)
    assert result.ok is False
    assert "gateway down" in result.error
    assert result.receipt == {}
    with pytest.raises(EffectorError):
        result.raise_for_status()


def test_missing_short_url_is_a_failure(case):
    class NoUrl(FakeLinkAPI):
        def create(self, params):
            link = super().create(params)
            link["short_url"] = ""
            return link

    result = PaymentLinkEffector(api=NoUrl()).run(case)
    assert result.ok is False
    assert "short_url" in result.error


# ── Paise in, paise out (the 10,000x regression) ─────────────────────────────

def test_amount_is_sent_as_paise_exactly_once(case):
    api = FakeLinkAPI()
    PaymentLinkEffector(api=api).run(case)
    assert api.created[0]["amount"] == 299900          # not 29_990_000
    assert case.amount_paise == 299900


@pytest.mark.parametrize("rupees,paise", [(1, 100), (2999, 299900), (99.5, 9950)])
def test_no_amount_drift_for_any_value(ledger, rupees, paise):
    c = ledger.open_case(payment_id=f"pay_{rupees}", amount_paise=to_paise(rupees))
    api = FakeLinkAPI()
    result = PaymentLinkEffector(api=api).run(c)
    assert api.created[0]["amount"] == paise
    assert result.receipt["amount_paise"] == paise


# ── Idempotency ──────────────────────────────────────────────────────────────

def test_reference_id_is_stable_for_a_case(ledger, case):
    """It must NOT be derived from attempt_count — the attempt increments it."""
    before = reference_id_for(case)
    ledger.record_attempt(case.case_id, action="create_payment_link", result="ok")
    after = reference_id_for(ledger.require_case(case.case_id))
    assert before == after
    assert case.case_id in before


def test_an_explicit_intent_mints_a_different_link(case):
    assert reference_id_for(case, "retry-2") != reference_id_for(case)


def test_a_deliberate_second_link_is_possible(ledger, case):
    """One link per case by default, but the caller can choose to make another."""
    api = FakeLinkAPI()
    eff = PaymentLinkEffector(api=api)
    first = eff.run(case)
    second = eff.run(case, intent="after-expiry")
    assert second.ok and not second.reused
    assert second.receipt["link_id"] != first.receipt["link_id"]
    assert len(api.created) == 2


def test_duplicate_reference_reuses_the_existing_link(case):
    """A crash between the API call and the ledger write must not double-create."""
    api = FakeLinkAPI()
    eff = PaymentLinkEffector(api=api)

    first = eff.run(case)
    second = eff.run(case)          # same case, same attempt -> same reference_id

    assert first.ok and second.ok
    assert second.reused is True
    assert second.receipt["link_id"] == first.receipt["link_id"]
    assert len(api.created) == 1, "a second link was created for the same debt"


def test_send_recovery_link_is_idempotent(ledger, case):
    api = FakeLinkAPI()
    eff = PaymentLinkEffector(api=api)
    for _ in range(3):
        got = send_recovery_link(ledger, ledger.require_case(case.case_id),
                                 effector=eff)
    assert got.attempt_count == 1
    assert len(api.created) == 1
    assert ledger.verify(case.case_id)


# ── Ledger wiring ────────────────────────────────────────────────────────────

def test_success_moves_case_to_awaiting_customer_with_receipt(ledger, case):
    got = send_recovery_link(ledger, case, effector=PaymentLinkEffector(api=FakeLinkAPI()))

    assert got.status == CaseStatus.AWAITING_CUSTOMER
    assert got.attempt_count == 1
    receipt = last_receipt(ledger, got.case_id)
    assert receipt["link_url"].startswith("https://rzp.io/i/")
    assert receipt["link_id"].startswith("plink_")
    assert receipt["amount_paise"] == 299900
    assert ledger.verify(got.case_id)


def test_failure_keeps_case_in_acting_and_records_the_error(ledger, case):
    eff = PaymentLinkEffector(api=FakeLinkAPI(fail_with=Exception("boom")))
    got = send_recovery_link(ledger, case, effector=eff)

    assert got.status == CaseStatus.ACTING
    errors = [e for e in ledger.events(got.case_id)
              if e.kind is EventKind.ATTEMPT and e.result == "error"]
    assert len(errors) == 1
    assert "boom" in errors[0].payload["receipt"]["error"]
    assert ledger.verify(got.case_id)


def test_a_failed_send_cannot_satisfy_the_awaiting_evidence_rule(ledger, case):
    """D2 rule 2 holds against a real effector failure, not just a synthetic one."""
    eff = PaymentLinkEffector(api=FakeLinkAPI(fail_with=Exception("boom")))
    send_recovery_link(ledger, case, effector=eff)
    with pytest.raises(MissingEvidence):
        ledger.record_transition(case.case_id, CaseStatus.AWAITING_CUSTOMER)


def test_attempt_and_transition_land_together(ledger, case):
    send_recovery_link(ledger, case, effector=PaymentLinkEffector(api=FakeLinkAPI()))
    kinds = [e.kind for e in ledger.events(case.case_id)]
    assert kinds[-2:] == [EventKind.ATTEMPT, EventKind.TRANSITION]


# ── The link itself ──────────────────────────────────────────────────────────

def test_notes_carry_what_the_sensor_needs_to_correlate(ledger, case):
    api = FakeLinkAPI()
    PaymentLinkEffector(api=api).run(case)
    notes = api.created[0]["notes"]
    assert notes["case_id"] == case.case_id
    assert notes["original_payment_id"] == "pay_orig_1"


def test_customer_block_only_includes_valid_fields(ledger, case):
    api = FakeLinkAPI()
    PaymentLinkEffector(api=api).run(case, customer_name="Rahul")
    cust = api.created[0]["customer"]
    assert cust["email"] == "rahul@example.com"
    assert cust["contact"] == "+919999999999"
    assert cust["name"] == "Rahul"


def test_invalid_email_is_omitted_rather_than_sent(ledger):
    c = ledger.open_case(payment_id="pay_noemail", amount_paise=to_paise(10),
                         customer_id="cust_12345")     # an id, not an email
    api = FakeLinkAPI()
    PaymentLinkEffector(api=api).run(c)
    assert "email" not in api.created[0].get("customer", {})


def test_link_expiry_is_clamped_above_razorpay_minimum(ledger, case):
    api = FakeLinkAPI()
    now = datetime.now(timezone.utc)
    PaymentLinkEffector(api=api).run(case, ttl=timedelta(seconds=5), now=now)
    expire_by = api.created[0]["expire_by"]
    assert expire_by - int(now.timestamp()) >= 15 * 60


def test_default_expiry_is_24h(ledger, case):
    api = FakeLinkAPI()
    now = datetime.now(timezone.utc)
    PaymentLinkEffector(api=api).run(case, now=now)
    delta = api.created[0]["expire_by"] - int(now.timestamp())
    assert abs(delta - DEFAULT_LINK_TTL.total_seconds()) < 5


def test_effector_does_not_send_notifications_itself(ledger, case):
    """Dispatch is D7's job; the link must not silently email the customer."""
    api = FakeLinkAPI()
    PaymentLinkEffector(api=api).run(case)
    assert api.created[0]["notify"] == {"sms": False, "email": False}
    assert api.created[0]["reminder_enable"] is False


# ── Replay must be a true no-op, not a partial one ───────────────────────────

def test_replay_changes_no_state_at_all(ledger, case):
    """A repeat call must not park the case in ACTING on its way to a no-op."""
    api = FakeLinkAPI()
    eff = PaymentLinkEffector(api=api)

    first = send_recovery_link(ledger, case, effector=eff)
    assert first.status == CaseStatus.AWAITING_CUSTOMER
    seq_after_first = first.seq
    events_after_first = len(ledger.events(case.case_id))

    for _ in range(3):
        again = send_recovery_link(ledger, ledger.require_case(case.case_id),
                                   effector=eff)

    assert again.status == CaseStatus.AWAITING_CUSTOMER, "replay left the case in ACTING"
    assert again.seq == seq_after_first, "replay wrote events"
    assert len(ledger.events(case.case_id)) == events_after_first
    assert len(api.created) == 1
    assert ledger.verify(case.case_id)


def test_has_attempt_detects_the_key(ledger, case):
    api = FakeLinkAPI()
    eff = PaymentLinkEffector(api=api)
    key = eff.idempotency_key(case)
    assert ledger.has_attempt(case.case_id, key) is False
    send_recovery_link(ledger, case, effector=eff)
    assert ledger.has_attempt(case.case_id, key) is True
    assert ledger.has_attempt(case.case_id, "") is False


# ═══════════════════════════════════════════════════════════════════════════
# RecoveryOrderEffector — the offer used when the payment-link quota is spent
# ═══════════════════════════════════════════════════════════════════════════

from recovery_agent.effectors import RecoveryOrderEffector, default_effector  # noqa: E402


class FakeOrderAPI:
    def __init__(self, fail_with: Exception | None = None):
        self.created: list[dict] = []
        self.fail_with = fail_with
        self._by_receipt: dict[str, dict] = {}
        self._n = 0

    def create(self, params):
        if self.fail_with is not None:
            raise self.fail_with
        self.created.append(params)
        self._n += 1
        order = {
            "id": f"order_fake{self._n}", "entity": "order",
            "amount": params["amount"], "currency": params["currency"],
            "receipt": params.get("receipt", ""), "status": "created",
            "notes": params.get("notes", {}),
        }
        self._by_receipt[order["receipt"]] = order
        return order

    def fetch(self, order_id):
        for o in self._by_receipt.values():
            if o["id"] == order_id:
                return o
        raise Exception("not found")

    def all(self, params=None):
        rcpt = (params or {}).get("receipt")
        items = [self._by_receipt[rcpt]] if rcpt in self._by_receipt else []
        return {"count": len(items), "items": items}


def test_order_amount_is_paise_exactly_once(case):
    api = FakeOrderAPI()
    result = RecoveryOrderEffector(api=api).run(case)
    assert api.created[0]["amount"] == 299900
    assert result.receipt["amount_paise"] == 299900


def test_order_receipt_carries_a_payable_checkout_url(case):
    eff = RecoveryOrderEffector(api=FakeOrderAPI(),
                                checkout_base_url="https://shop.example/pay")
    r = eff.run(case)
    assert r.ok
    assert r.receipt["order_id"].startswith("order_")
    assert r.receipt["link_url"] == f"https://shop.example/pay?order_id={r.receipt['order_id']}"


def test_order_notes_let_the_sensor_correlate(case):
    api = FakeOrderAPI()
    RecoveryOrderEffector(api=api).run(case)
    notes = api.created[0]["notes"]
    assert notes["case_id"] == case.case_id
    assert notes["original_payment_id"] == "pay_orig_1"


def test_order_effector_reuses_an_existing_order(case):
    """Covers a crash between order creation and the ledger write."""
    api = FakeOrderAPI()
    eff = RecoveryOrderEffector(api=api)
    first = eff.run(case)
    second = eff.run(case)
    assert second.reused is True
    assert second.receipt["order_id"] == first.receipt["order_id"]
    assert len(api.created) == 1


def test_order_effector_does_not_fabricate_when_unconfigured(case):
    class Unconfigured:
        is_configured = False
    with pytest.raises(NotConfigured):
        RecoveryOrderEffector(client=Unconfigured()).run(case)


def test_order_api_failure_is_recorded_not_invented(ledger, case):
    eff = RecoveryOrderEffector(api=FakeOrderAPI(fail_with=Exception("rzp 500")))
    got = send_recovery_link(ledger, case, effector=eff)
    assert got.status == CaseStatus.ACTING
    assert last_receipt(ledger, got.case_id) == {}, "a failed call must leave no receipt"
    errors = [e for e in ledger.events(got.case_id)
              if e.kind is EventKind.ATTEMPT and e.result == "error"]
    assert len(errors) == 1 and "rzp 500" in errors[0].payload["receipt"]["error"]


def test_order_flow_end_to_end_through_the_ledger(ledger, case):
    got = send_recovery_link(ledger, case,
                             effector=RecoveryOrderEffector(api=FakeOrderAPI()))
    assert got.status == CaseStatus.AWAITING_CUSTOMER
    assert got.attempt_count == 1
    r = last_receipt(ledger, got.case_id)
    assert r["order_id"].startswith("order_")
    assert "order_id=" in r["link_url"]
    assert ledger.verify(got.case_id)


def test_default_effector_is_order_unless_overridden(monkeypatch):
    monkeypatch.delenv("RECOVERY_EFFECTOR", raising=False)
    assert isinstance(default_effector(), RecoveryOrderEffector)
    monkeypatch.setenv("RECOVERY_EFFECTOR", "payment_link")
    assert isinstance(default_effector(), PaymentLinkEffector)


def test_checkout_url_is_resolved_at_call_time_not_import_time(case, monkeypatch):
    """A URL frozen at import points at a port nothing is listening on."""
    from recovery_agent.effectors import checkout_base_url
    monkeypatch.setenv("CHECKOUT_BASE_URL", "http://localhost:39017/pay")
    assert checkout_base_url() == "http://localhost:39017/pay"
    r = RecoveryOrderEffector(api=FakeOrderAPI()).run(case)
    assert r.receipt["link_url"].startswith("http://localhost:39017/pay?order_id=")


def test_an_explicit_base_url_still_wins(case, monkeypatch):
    monkeypatch.setenv("CHECKOUT_BASE_URL", "http://env.example/pay")
    eff = RecoveryOrderEffector(api=FakeOrderAPI(), checkout_base_url="https://x.test/p")
    assert eff.run(case).receipt["link_url"].startswith("https://x.test/p?order_id=")
