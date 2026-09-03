"""The agent must be told what actually worked, because the lesson is permanent.

`pay_2et832rs8`: the customer paid the recovery LINK. A 24-hour retry sat
scheduled for the next day and had not fired. The observation said "paid 65s
after wait_and_retry" — `last_action` is merely the last thing recorded — so the
agent concluded the retry had worked and stored:

    "CONFIRMED WINNING STRATEGY: Silent 24h background retry on netbanking rail"

A false lesson, in memory that now survives restarts, pushing the next recovery
toward waiting a day instead of sending the link that actually worked.
"""
from pathlib import Path

from recovery_agent.frontend import _how_it_arrived

FRONTEND = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
            / "frontend.py").read_text()
DAEMON = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
          / "daemon_worker.py").read_text()


def test_a_link_payment_is_attributed_to_the_link():
    assert "recovery payment link" in _how_it_arrived(
        "link plink_TXRYNINIFFtA9S paid (poll). Payment pay_X. After 65s")


def test_a_checkout_payment_is_attributed_to_the_page():
    assert "checkout page" in _how_it_arrived("customer paid on the checkout page")


def test_an_order_payment_is_attributed_to_the_order():
    assert "original checkout order" in _how_it_arrived("order order_X paid (poll)")


def test_an_unknown_route_does_not_invent_one():
    assert _how_it_arrived("") == "the payment was captured"
    assert _how_it_arrived("something new") == "the payment was captured"


def test_the_observation_carries_the_attribution():
    i = FRONTEND.index("HOW IT ARRIVED")
    body = FRONTEND[i - 1200:i + 700]
    assert "arrival = _how_it_arrived(how)" in body
    assert "the lesson you" in body and "permanent" in body


def test_a_pending_retry_is_explicitly_ruled_out_when_a_link_was_paid():
    """Otherwise the agent sees a scheduled retry in the record and credits it."""
    i = FRONTEND.index("pending_retry = ")
    body = FRONTEND[i:i + 400]
    assert "has NOT fired" in body


def test_mark_recovered_passes_how_through():
    assert "_notify_agent_of_recovery(payment_id, amount, rzp_payment_id, seconds, how)" \
        in FRONTEND


# ── one retry, one job ──────────────────────────────────────────────────

def test_a_retry_does_not_create_two_jobs():
    """retry_in_hours schedules a job and returns its id; register_retry_job
    then created another, so one retry left two pending jobs and the daemon
    would execute it twice."""
    assert "job_id: str = \"\"," in DAEMON
    assert 'job_id = job_id or f"job_{payment_id}_{int(time.time())}"' in DAEMON
    i = FRONTEND.index("registered_job = register_retry_job(")
    assert 'job_id=sdk_res.get("job_id", "")' in FRONTEND[i:i + 700]


def test_the_recorded_reason_matches_the_agents_decision():
    """"Immediate retry — network issues are transient" was attached to a
    24-hour job the agent had chosen deliberately."""
    i = FRONTEND.index("agent scheduled a retry in")
    assert "delay_hours" in FRONTEND[i - 200:i + 300]
