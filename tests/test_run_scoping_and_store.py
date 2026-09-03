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


# ── a scheduled retry is live work, not a loss ──────────────────────────

def test_a_scheduled_retry_is_not_recorded_as_a_failure():
    """The agent picked a 24h silent retry as its alternate path, the daemon
    registered the job — and the case was written as `failed`, because the run
    had ended. A pending recovery counted as a loss on the dashboard."""
    i = FRONTEND.index("elif case_status == CaseStatus.STOPPED:")
    body = FRONTEND[i:i + 700]
    assert 'action_val == "wait_and_retry"' in body
    assert '"scheduled" if' in body


def test_the_agents_chosen_retry_time_is_used():
    """`retry_in_hours` returns `target_time`; only `target_timestamp` was read,
    so the key was always missing and the scheduler substituted its own guess —
    the agent asked for 24 hours and the daemon registered a job 3 minutes out."""
    i = FRONTEND.index("target_ts = (sdk_res.get(")
    body = FRONTEND[i:i + 400]
    assert 'sdk_res.get("target_time")' in body


# ── money in the bank cannot be written over ────────────────────────────

def test_recovered_is_absorbing(tmp_path):
    """Twice now a real recovery has been rewritten as a loss: an LLM 404 on a
    bookkeeping run turned a captured INR 2,374.05 into `failed`, and a run that
    only scheduled a retry wrote `failed` over a live case. Money received is a
    fact, not a status."""
    s = StateStore(tmp_path)
    s.save_payment("r", {"payment_id": "r", "status": "recovered",
                         "recovered_amount": 2374.05})
    s.update_payment("r", status="failed", tier="ERROR")
    assert s.get_payment("r")["status"] == "recovered"
    assert s.get_payment("r")["tier"] == "ERROR", "other fields still apply"


def test_an_explicit_force_can_still_correct_the_record(tmp_path):
    s = StateStore(tmp_path / "f")
    s.save_payment("r", {"payment_id": "r", "status": "recovered"})
    s.update_payment("r", status="failed", force=True)
    assert s.get_payment("r")["status"] == "failed"


def test_a_non_recovered_case_moves_freely(tmp_path):
    s = StateStore(tmp_path / "n")
    s.save_payment("r", {"payment_id": "r", "status": "escalated"})
    s.update_payment("r", status="recovered")
    assert s.get_payment("r")["status"] == "recovered", (
        "a customer paying after escalation is a legitimate transition"
    )


# ── the stop control cannot be defeated by a rebuilt Case ───────────────

def test_the_tool_lockdown_consults_the_durable_record():
    """The Case object is rebuilt from scratch every run, so its status is only
    as good as whatever the caller put there. Money actually received must be
    decisive regardless of any status string."""
    GRAPH = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
             / "agent" / "graph.py").read_text()
    i = GRAPH.index("A finished case gets bookkeeping tools ONLY")
    body = GRAPH[i:i + 2600]
    assert "StateStore().get_payment" in body
    assert 'recovered_amount' in body


# ── memory survives a restart ───────────────────────────────────────────

def test_long_term_memory_is_not_process_local():
    GRAPH = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
             / "agent" / "graph.py").read_text()
    i = GRAPH.index("def _build_memory_store")
    body = GRAPH[i:i + 2200]
    assert "SqliteStore" in body
    assert "InMemoryStore" in body, "in-memory remains the fallback, not the default"


def test_sessions_are_not_process_local():
    GRAPH = (Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
             / "agent" / "graph.py").read_text()
    i = GRAPH.index("# Working memory — the message history")
    body = GRAPH[i:i + 1200]
    assert "SqliteSaver" in body
