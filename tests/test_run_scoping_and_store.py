"""Two defects that showed up identically in two consecutive live recoveries.

Both cases recovered real money (INR 1,299 and INR 4,748.10) and both finished
with a closing run that reported the PREVIOUS run's work:

    [TOOL CALL] manage_memory            <- all this run actually did
    [SUMMARY]   "Waiting on the customer..."          <- previous run's summary
    [EXECUTE]   Agent executed: wait_for_customer     <- previous run's action
                Primary action: page_push

And in both, the ladder ledger was empty despite the tools reporting they had
written it.
"""
from pathlib import Path

from recovery_agent.state_store import StateStore

FRONTEND = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
            / "frontend.py").read_text()


# ── one store per directory ─────────────────────────────────────────────

def test_two_stores_for_one_directory_are_the_same_object(tmp_path):
    assert StateStore(tmp_path) is StateStore(tmp_path)


def test_different_directories_stay_separate(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert StateStore(a) is not StateStore(b)


def test_a_tools_write_survives_the_frontends_next_flush(tmp_path):
    """The exact race that made the ladder inert: the frontend holds a
    long-lived store, a tool built its own, wrote, and flushed — then the
    frontend flushed its stale copy back over the top."""
    frontend = StateStore(tmp_path)
    frontend.save_payment("p", {"payment_id": "p", "status": "recovering"})
    frontend.flush()

    tool = StateStore(tmp_path)                       # what a tool does
    tool.update_payment("p", ladder={"page_push": {"at": "now"}})
    tool.flush()

    frontend.update_payment("p", last_action="page_push")
    frontend.flush()                                  # this used to erase it

    StateStore.reset_instances()
    assert StateStore(tmp_path).get_payment("p")["ladder"] == {"page_push": {"at": "now"}}


def test_reset_instances_forces_a_reload_from_disk(tmp_path):
    s = StateStore(tmp_path)
    s.save_payment("q", {"payment_id": "q"})
    s.flush()
    StateStore.reset_instances()
    assert StateStore(tmp_path) is not s
    assert StateStore(tmp_path).get_payment("q") is not None


# ── a run reports only its own work ─────────────────────────────────────

def test_the_run_filters_out_messages_restored_from_earlier_runs():
    i = FRONTEND.index("for s in agent.graph.stream(")
    body = FRONTEND[i - 2500:i + 1200]
    assert "prior_ids" in body, "the run must know which messages predate it"
    assert "agent.graph.get_state(config)" in body, (
        "read the checkpoint to find out, rather than guessing"
    )
    assert "if getattr(m, \"id\", None) not in prior_ids" in body


def test_a_failure_to_read_the_checkpoint_does_not_break_the_run():
    """Worst case it over-reports, which is what it did before. It must never
    take the whole run down."""
    i = FRONTEND.index("prior_ids: set[str] = set()")
    body = FRONTEND[i:i + 500]
    assert "except Exception:" in body and "pass" in body
