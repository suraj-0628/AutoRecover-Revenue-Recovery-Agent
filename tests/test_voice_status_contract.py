"""A success the harness doesn't recognise is a success that never happened.

`superu_client.initiate_recovery_call` returns `{"status": "call_initiated"}`
when a call goes out. The tool checked for `("ok", "initiated", "queued",
"success")` — so a successful PAID call recorded nothing: the voice_call rung
stayed unclimbed and the same customer could be rung again. Mirror-image of
the blocked-escalation bug: there a refusal counted as success, here a success
counted as nothing.

NO real SuperU call is ever made here. `requests.post` is stubbed before the
test-environment guard is bypassed, so nothing can reach the network — and the
guard itself is pinned by its own test below.
"""
import recovery_agent.integrations.superu_client as superu
from recovery_agent.agent.tools import VOICE_CALL_OK_STATUSES


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"call_uuid": "uuid-1", "campaign_id": "recovery_pay_x"}


def test_the_clients_success_status_is_one_the_tool_accepts(monkeypatch):
    """The contract join: whatever the client says on success, the tool must
    read as success — otherwise the ladder rung is silently lost."""
    calls = []

    def _fake_post(url, **kwargs):
        calls.append(url)
        return _FakeResponse()

    monkeypatch.setattr(superu.requests, "post", _fake_post)
    # Bypass the test-env guard ONLY because the network is already stubbed.
    monkeypatch.setattr(superu, "_is_test_environment", lambda: False)

    client = superu.SuperUClient()
    client.api_key, client.assistant_id, client.from_phone = "k", "a", "+910000000000"
    client._enabled = True

    result = client.initiate_recovery_call(
        payment_id="pay_x", customer_name="Test", customer_phone="9999999999",
        amount=9999.0, failure_reason="card_expired",
    )

    assert calls, "the stubbed transport was never used"
    assert result["status"] == "call_initiated"
    assert result["status"] in VOICE_CALL_OK_STATUSES


def test_no_failure_status_reads_as_success():
    for status in ("call_failed", "skipped", "disabled", "blocked",
                   "too_soon", "error"):
        assert status not in VOICE_CALL_OK_STATUSES


def test_the_test_environment_guard_still_blocks_real_calls():
    """The guard the credits depend on. Under pytest it must refuse outright —
    weakening this is never an acceptable fix."""
    client = superu.SuperUClient()
    client._enabled = True
    result = client.initiate_recovery_call(
        payment_id="pay_x", customer_name="Test", customer_phone="9999999999",
        amount=9999.0,
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "test_environment"
