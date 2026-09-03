"""Revenue that cannot be recovered in real time, sorted before it is worked.

A declined card needs a different rail; an empty account needs a different day;
someone who walked away needs a reason to come back. Worked in one queue the
agent re-derives that for every case — sorting first means it decides once and
applies many times.
"""
import json
import pathlib
import tempfile

import pytest

from recovery_agent.agent.classify import (BATCHES, BATCH_BY_KEY, classify,
                                           failure_kind, summarise)

FRONTEND = (pathlib.Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
            / "frontend.py").read_text()
INDEX = (pathlib.Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
         / "templates" / "index.html").read_text()
PAY_PAGE_SRC = FRONTEND


def rec(**over):
    r = {"payment_id": "p", "amount": 2499.0, "status": "awaiting_customer",
         "customer": {"email": "a@b.com"}, "failure_code": "", "failure_reason": ""}
    r.update(over)
    return r


# ── one definition of what kind of failure this is ──────────────────────

@pytest.mark.parametrize("code,expected", [
    ("bank_declined", "method"), ("card_expired", "method"),
    ("insufficient_funds", "funds"), ("customer_cancelled", "dropoff"),
    ("fraud_suspected", "risk"), ("something_new", "unknown"),
])
def test_failure_kind(code, expected):
    assert failure_kind(rec(failure_code=code)) == expected


def test_perception_uses_the_same_definition():
    """Two copies would drift, and a case would be a bank decline in one place
    and a drop-off in the other."""
    PERC = (pathlib.Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
            / "agent" / "perception.py").read_text()
    assert "from recovery_agent.agent.classify import failure_kind" in PERC


# ── which batch a case belongs in ───────────────────────────────────────

def test_a_recovered_case_is_in_no_batch():
    assert classify(rec(status="recovered")) is None
    assert classify(rec(recovered_amount=2499.0)) is None


def test_a_case_being_worked_right_now_is_in_no_batch():
    """Batching it would put two runs on one payment, which the session model
    does not allow."""
    assert classify(rec(status="recovering")) is None


def test_a_decline_goes_to_the_rail_batch():
    assert classify(rec(failure_code="bank_declined")) == "bank_declined"


def test_an_empty_account_goes_to_its_own_batch():
    assert classify(rec(failure_code="insufficient_funds")) == "insufficient_funds"


def test_a_scheduled_retry_is_not_re_worked():
    assert classify(rec(status="scheduled",
                        failure_code="bank_declined")) == "awaiting_retry"


def test_risk_outranks_everything():
    assert classify(rec(status="scheduled",
                        failure_code="fraud_suspected")) == "risk"


def test_a_case_with_no_contact_details_is_not_a_batch_at_all():
    """The checkout will not start a payment until name, a valid email and a
    ten-digit phone are all present, so no real customer journey can produce a
    contactless record. Counting them inflated revenue-at-risk with money
    nobody ever tried to pay — 138 of 186 cases, none of it workable."""
    assert classify(rec(customer={}, customer_email="",
                        customer_phone="")) is None


def test_an_escalated_case_shows_as_with_a_human():
    assert classify(rec(status="escalated")) == "escalated"
    assert classify(rec(closed={"outcome": "escalated"})) == "escalated"


def test_an_unclassifiable_failure_is_still_counted():
    """Money at risk must never fall out of every batch and disappear — as long
    as it is money someone could actually pay."""
    assert classify(rec(failure_code="???")) is not None


# ── the summary ─────────────────────────────────────────────────────────

def test_empty_batches_are_kept():
    """An empty batch says that class of failure is not happening, and hiding
    it makes the page rearrange itself on every refresh."""
    out = summarise([rec(failure_code="bank_declined")])
    assert len(out) == len(BATCHES)
    assert {b["key"] for b in out} == set(BATCH_BY_KEY)


def test_value_at_risk_adds_up():
    out = summarise([rec(payment_id="a", amount=100.0, failure_code="bank_declined"),
                     rec(payment_id="b", amount=250.0, failure_code="bank_declined")])
    bank = [b for b in out if b["key"] == "bank_declined"][0]
    assert bank["count"] == 2 and bank["value"] == 350.0


def test_recovered_money_is_not_counted_as_at_risk():
    out = summarise([rec(status="recovered", amount=9999.0)])
    assert sum(b["value"] for b in out) == 0.0


# ── running a batch ─────────────────────────────────────────────────────

def test_a_run_never_starts_a_second_session_on_one_payment():
    i = FRONTEND.index("def api_run_batch")
    body = FRONTEND[i:i + 3000]
    assert "if pid in active_agent_payments:" in body
    assert "already being worked" in body


def test_a_run_is_capped_and_says_so():
    """A batch can hold hundreds; a run that quietly worked all of them would
    send hundreds of emails and burn the payment link quota in one click."""
    i = FRONTEND.index("def api_run_batch")
    body = FRONTEND[i:i + 3000]
    assert "BATCH_RUN_LIMIT" in FRONTEND
    assert '"limit": limit' in body and '"total_in_batch"' in body
    assert "limit " in INDEX and "per run" in INDEX, (
        "a partial run must not read as a complete one"
    )


def test_a_run_skips_cases_with_no_way_to_contact_them():
    i = FRONTEND.index("def api_run_batch")
    assert "no way to contact them" in FRONTEND[i:i + 3000]


def test_an_unknown_batch_is_rejected():
    i = FRONTEND.index("def api_run_batch")
    assert "unknown batch" in FRONTEND[i:i + 800]


# ── the view ────────────────────────────────────────────────────────────

def test_the_rail_does_not_restructure_the_live_view():
    """Fixed rather than a grid column, so the existing workspace is shifted
    rather than rebuilt — the live view stays exactly as it was."""
    assert "position: fixed" in INDEX[INDEX.index(".railnav {"):INDEX.index(".railnav {") + 300]
    assert 'showView(\'live\')' in INDEX and 'showView(\'batches\')' in INDEX


def test_the_batch_view_uses_the_existing_design_tokens():
    i = INDEX.index(".batch-card {")
    body = INDEX[i:i + 400]
    assert "var(--panel-bg)" in body and "var(--border-color)" in body


# ── one meaning of "at risk" ────────────────────────────────────────────

def test_recovered_money_is_not_counted_as_at_risk_in_the_header():
    """The header summed EVERY payment, recovered ones included, so it read INR
    9,23,031 at risk while the batch view — counting only what is still out —
    read INR 5,36,960. Two numbers for the same thing on one screen, and the
    larger one was wrong."""
    i = FRONTEND.index("def api_payments")
    body = FRONTEND[i:i + 1800]
    assert "if not _settled(p)" in body


def test_recovered_reports_what_arrived_not_what_was_owed():
    """A INR 2,499 order recovered at 5% off brought back INR 2,374.05.
    Counting the full 2,499 quietly credits the agent with the discount it gave
    away."""
    i = FRONTEND.index("def api_payments")
    body = FRONTEND[i:i + 1800]
    assert 'p.get("recovered_amount") or p.get("amount")' in body


# ── one notification per customer ───────────────────────────────────────

def test_an_offer_renders_as_a_banner_not_a_second_notification():
    """Both were drawn from the one `agent_push` event, so a discount arrived as
    a bar across the top AND another card in the corner moments after the plain
    one — two notifications for a customer who had already read the first."""
    i = PAY_PAGE_SRC.index('socket.on("agent_push"')
    body = PAY_PAGE_SRC[i:i + 6000]
    assert "if (d.offer && d.offer.payable_rupees) return;" in body


def test_the_banner_can_be_acted_on_and_turned_down():
    """It carries the offer alone now, so it needs its own button — and its own
    dismissal, because both are signals the agent reads."""
    i = PAY_PAGE_SRC.index('bar.id = "agent-offer-banner"')
    body = PAY_PAGE_SRC[i:i + 3000]
    assert 'id="offer-cta"' in body and 'id="offer-x"' in body
    assert 'reportPush("acted"' in body and 'reportPush("dismissed"' in body


def test_there_is_only_ever_one_push_per_case():
    """Two guards used to exist — "already dismissed" and "an offer is already
    on the page" — and both prevented the same thing. The rung is the simpler
    statement of it: once a push has been delivered, the silent rung is spent,
    whatever the customer did with it."""
    TOOLS = (pathlib.Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
             / "agent" / "tools.py").read_text()
    i = TOOLS.index("def send_page_push")
    body = TOOLS[i:i + 4000]
    assert 'ladder.climbed(live, "page_push")' in body
    assert "there is no second one" in body
    assert 'climbed(live, "offer")' not in body, "one rule, not three"


def test_nothing_supersedes_a_notification_any_more():
    """The replacement machinery existed only because a second push could take
    the place of a first."""
    FRONTEND_SRC = (pathlib.Path(__file__).resolve().parents[1] / "src"
                    / "recovery_agent" / "frontend.py").read_text()
    assert "superseded" not in FRONTEND_SRC


def test_the_plain_push_cannot_carry_a_link_or_an_offer():
    """It used to accept payment_link and offer_text, and the agent used them:
    at the offer rung it sent a second notification carrying the link, landing
    on top of the banner already showing the same price. The parameters are
    gone, so it is no longer something to remember not to do."""
    from recovery_agent.agent.tools import TOOLS_BY_NAME
    params = TOOLS_BY_NAME["send_page_push"].args_schema.model_json_schema()["properties"]
    assert "payment_link" not in params
    assert "offer_text" not in params


def test_the_push_card_no_longer_renders_an_offer_or_opens_a_link():
    i = PAY_PAGE_SRC.index('wrap.id = "agent-push"')
    body = PAY_PAGE_SRC[i:i + 2600]
    assert "d.offer_text" not in body
    assert "window.open(d.payment_link" not in body


# ── one message on the page at a time ───────────────────────────────────

def test_waiting_is_not_reported_to_the_customer_as_failure():
    """A run ending is not the case ending. The customer was told "Could not
    recover automatically" while a live offer sat on screen above it."""
    i = FRONTEND.index('if (data.event === "complete")')
    body = FRONTEND[i:i + 1200]
    assert "settled.indexOf(data.status) === -1) return" in body


# ── the agent does not write on the checkout page ───────────────────────

def test_the_generative_ui_overlay_is_gone_entirely():
    """It fired on every action and spoke internal labels to the customer —
    "Recovery strategy: Page Push" — with its own competing button."""
    for token in ("ui_spec_overlay", "ui-overlay", "overlay-headline",
                  "overlay-tier"):
        assert token not in FRONTEND.replace("`ui_spec_overlay`", ""), token


def test_the_agent_no_longer_rewrites_the_checkout():
    """`applyGenerativeUISpec` pasted spec.headline into the checkout title —
    which has held the agent's markdown summary verbatim — plus a second
    discount banner and a status line reading "Background Retry Active —
    customer_cancelled"."""
    assert "applyGenerativeUISpec(data.ui_spec)" not in FRONTEND
    assert "function applyGenerativeUISpec" not in FRONTEND


def test_the_generative_ui_spec_is_gone_from_the_agent_too():
    """It was never generative: `ui_type` was a dict lookup, `subtext` was the
    internal label "Recovery strategy: ...", `tone` was always "supportive",
    and `hinglish_voice_script` was a canned sentence nothing read. Its
    docstring promised "no hardcoded if/else branches"."""
    MODELS = (pathlib.Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
              / "models" / "__init__.py").read_text()
    assert "GenerativeUISpec" not in MODELS
    for token in ("ui_spec=", "ui_morph=", "GenerativeUISpec("):
        assert token not in FRONTEND, token


def test_the_case_facts_still_reach_the_merchant_hud():
    """Tier, decline strategy and penalties are facts about the case; they used
    to travel inside the fake spec and now travel as themselves."""
    assert "case_state={" in FRONTEND
    assert "data.case_state && data.case_state.recovery_tier" in INDEX
