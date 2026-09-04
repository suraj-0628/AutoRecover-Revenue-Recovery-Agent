"""Guardrail policy the operator can move, from the screen they watch it in.

The six guardrails were env-only: the only way to change a boundary was shell
access and a restart of four services. That is the wrong audience and the wrong
moment. Live 2026-09-04 (pay_ls1dep23k, INR 12,495) a genuine bank-decline
recovery was refused at "4/3 contacts in 24h" — the cap had counted earlier
test contacts against the same customer — and the operator watching it fail in
the dashboard could do nothing from there.

Precedence is deliberate: an operator's change beats the deployment's env,
which beats the code default.
"""
import json

import pytest

import recovery_agent.guardrail_config as gc


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_DIR", str(tmp_path))
    for spec in gc.SETTINGS:
        if spec.get("env"):
            monkeypatch.delenv(spec["env"], raising=False)
    yield


# ── precedence ──────────────────────────────────────────────────────────────

def test_the_default_is_the_policy_of_last_resort():
    assert gc.get("max_contacts_24h") == 5


def test_env_configures_a_fresh_install(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_MAX_CONTACTS_24H", "7")
    assert gc.get("max_contacts_24h") == 7


def test_an_operator_change_beats_the_environment(monkeypatch):
    monkeypatch.setenv("GUARDRAIL_MAX_CONTACTS_24H", "7")
    gc.update({"max_contacts_24h": 9})
    assert gc.get("max_contacts_24h") == 9


def test_reset_returns_to_env_then_default(monkeypatch):
    gc.update({"max_contacts_24h": 9})
    assert gc.get("max_contacts_24h") == 9
    gc.reset()
    assert gc.get("max_contacts_24h") == 5


def test_the_source_is_reported_so_the_screen_can_say_where_it_came_from(monkeypatch):
    assert {s["key"]: s["source"] for s in gc.describe()}["max_contacts_24h"] == "default"
    monkeypatch.setenv("GUARDRAIL_MAX_CONTACTS_24H", "7")
    assert {s["key"]: s["source"] for s in gc.describe()}["max_contacts_24h"] == "env"
    gc.update({"max_contacts_24h": 9})
    assert {s["key"]: s["source"] for s in gc.describe()}["max_contacts_24h"] == "operator"


# ── values are coerced and bounded ──────────────────────────────────────────

def test_a_slider_is_clamped_to_its_range():
    gc.update({"max_contacts_24h": 999})
    assert gc.get("max_contacts_24h") == 12
    gc.update({"max_contacts_24h": 0})
    assert gc.get("max_contacts_24h") == 1


def test_an_hour_stays_inside_the_clock():
    gc.update({"quiet_start": 47})
    assert gc.get("quiet_start") == 23


def test_money_is_a_float_not_a_bounded_count():
    gc.update({"max_discount_giveaway": "7500.50"})
    assert gc.get("max_discount_giveaway") == 7500.50


def test_a_toggle_accepts_what_a_form_actually_sends():
    for raw in (True, "true", "1", "on"):
        gc.update({"voice_enabled": raw})
        assert gc.get("voice_enabled") is True
    for raw in (False, "false", "0", "off"):
        gc.update({"voice_enabled": raw})
        assert gc.get("voice_enabled") is False


def test_nonsense_is_rejected_by_name_not_swallowed():
    _, rejected = gc.update({"max_contacts_24h": "abc"})
    assert rejected and "max_contacts_24h" in rejected[0]
    _, rejected = gc.update({"no_such_setting": 1})
    assert rejected and "unknown setting" in rejected[0]


# ── the one that must never be switchable ───────────────────────────────────

def test_opt_out_cannot_be_turned_off():
    values, rejected = gc.update({"opt_out_enabled": False})
    assert values["opt_out_enabled"] is True
    assert rejected and "legal obligation" in rejected[0]


def test_a_refused_change_is_reported_not_silently_ignored():
    """An operator who thinks they disabled a control and did not is worse off
    than one who was told no."""
    _, rejected = gc.update({"opt_out_enabled": False})
    assert len(rejected) == 1


def test_the_dangerous_toggles_are_flagged_for_the_screen():
    danger = {s["key"] for s in gc.SETTINGS if s.get("danger")}
    assert danger == {"double_debit_enabled", "hard_decline_enabled"}


# ── every setting is presentable ────────────────────────────────────────────

def test_every_setting_explains_itself():
    for s in gc.SETTINGS:
        assert s["what"], f"{s['key']} has no description"
        assert s["kind"] in ("slider", "number", "hour", "toggle")
        assert s["group"]


def test_bounded_kinds_declare_their_bounds():
    for s in gc.SETTINGS:
        if s["kind"] == "slider":
            assert s["min"] < s["max"]


def test_money_settings_carry_a_unit():
    for s in gc.SETTINGS:
        if s["kind"] == "number":
            assert s.get("unit"), f"{s['key']} is money with no unit"


# ── the engine actually reads it ────────────────────────────────────────────

def test_the_engine_picks_up_an_operator_cap():
    from recovery_agent.agent.guardrails import GuardrailEngine
    gc.update({"max_contacts_24h": 9})
    assert GuardrailEngine().frequency_cap.max_contacts == 9


def test_turning_quiet_hours_off_empties_the_window():
    from recovery_agent.agent.guardrails import GuardrailEngine
    from recovery_agent.models import ActionType
    from recovery_agent.agent.guardrails import GuardrailVerdict
    from datetime import datetime
    from recovery_agent.agent.guardrails import IST

    gc.update({"quiet_enabled": False})
    e = GuardrailEngine()
    night = datetime(2026, 9, 4, 2, 0, tzinfo=IST)
    assert e.quiet_hours.check(ActionType.VOICE_CALL, now=night).verdict \
        == GuardrailVerdict.PASS


def test_a_disabled_safety_check_is_skipped_not_faked():
    """It must not appear in the verdict list claiming it looked and approved."""
    from recovery_agent.agent.guardrails import GuardrailEngine
    from recovery_agent.models import ActionType, Case, PaymentEvent

    case = Case(payment=PaymentEvent(payment_id="p", customer_id="c",
                                     amount=1000.0, failure_code="54"))
    gc.update({"hard_decline_enabled": False})
    _action, checks = GuardrailEngine().validate_action(
        case, ActionType.RETRY_PAYMENT)
    assert "hard_decline" not in {c.guardrail for c in checks}


def test_the_operator_can_raise_the_cap_that_blocked_the_live_case():
    """pay_ls1dep23k end to end: 4 contacts refused at a cap of 3."""
    from recovery_agent.agent.guardrails import FrequencyCapGuardrail
    gc.update({"max_contacts_24h": 3})
    assert FrequencyCapGuardrail(gc.get("max_contacts_24h")).max_contacts == 3
    gc.update({"max_contacts_24h": 8})
    assert FrequencyCapGuardrail(gc.get("max_contacts_24h")).max_contacts == 8
