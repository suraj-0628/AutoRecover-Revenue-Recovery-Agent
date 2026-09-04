"""A drop-off that stops to ask "why?" must not forget who it was asking.

`/api/payment-failed` created the case without the customer. The immediate
path hands `customer` to the agent as an argument and so never noticed; the
deferred path (used whenever the checkout asks why the customer stopped)
restarts the agent from the stored record, read `{}`, and the agent escalated
to a human on its first move for want of an email address it had been given.
"""
import json

import pytest


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Bind the endpoint's own store, as the sibling suites do.

    `frontend.store` is captured at import, so resetting the singleton is not
    enough — whichever test ran last leaves that name pointing at its own
    temp directory, and these tests then read a different store from the one
    the route writes to.
    """
    from recovery_agent import state_store
    from recovery_agent import frontend as F
    monkeypatch.setattr(state_store, "_DATA_DIR", tmp_path)
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    state_store.StateStore.reset_instances()
    monkeypatch.setattr(F, "store", state_store.StateStore())
    yield
    state_store.StateStore.reset_instances()


CUST = {"name": "suraj", "email": "s@example.com", "contact": "9363064502"}


def _fail(client, pid, **over):
    body = {"payment_id": pid, "amount": 1499.0,
            "failure_code": "customer_cancelled",
            "failure_reason": "Payment cancelled by customer",
            "customer": CUST}
    body.update(over)
    return client.post("/api/payment-failed", json=body)


def test_the_customer_is_stored_on_the_case(monkeypatch):
    from recovery_agent import frontend
    monkeypatch.setattr(frontend.socketio, "start_background_task",
                        lambda *a, **k: None)
    _fail(frontend.app.test_client(), "pay_c1")
    assert (frontend.store.get_payment("pay_c1") or {}).get("customer") == CUST


def test_a_deferred_case_still_knows_who_to_contact(monkeypatch):
    """The regression itself: hold for the reason, then release."""
    from recovery_agent import frontend
    started = {}
    monkeypatch.setattr(frontend.socketio, "start_background_task",
                        lambda *a, **k: started.setdefault("args", a))
    c = frontend.app.test_client()
    _fail(c, "pay_c2", defer_agent=True)
    rel = c.post("/api/drop-reason/skip", json={"payment_id": "pay_c2"})
    assert rel.get_json().get("status") == "released", rel.get_json()
    # run_agent_for_payment(payment_id, amount, reason, customer, ...)
    assert started["args"][4] == CUST, "agent was released without the customer"


def test_a_later_failure_without_contact_does_not_erase_it(monkeypatch):
    """Retries often carry no customer block; that is not a reason to forget."""
    from recovery_agent import frontend
    monkeypatch.setattr(frontend.socketio, "start_background_task",
                        lambda *a, **k: None)
    c = frontend.app.test_client()
    _fail(c, "pay_c3")
    frontend.store.update_payment("pay_c3", status="awaiting_customer")
    _fail(c, "pay_c3", customer={}, failure_code="insufficient_funds")
    assert (frontend.store.get_payment("pay_c3") or {}).get("customer") == CUST
