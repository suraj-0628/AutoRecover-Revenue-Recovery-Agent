"""A hard refresh of the batch dashboard must restore the log AS IT WAS.

The feed reads the append-only audit log by event id. On a fresh page load the
client asks with since=0, and that must return the most RECENT window of the
log — not the very first events ever written. With the old oldest-first query,
once a demo logged more than `limit` batch events a refresh showed ancient
rows and had to crawl forward a page at a time to reach what was on screen.
"""
from __future__ import annotations

import pytest

from recovery_agent import audit


@pytest.fixture()
def log(tmp_path):
    audit.AuditLog.reset_instances()
    yield audit.AuditLog(tmp_path / "audit.db")
    audit.AuditLog.reset_instances()


def _run(log, pid):
    """One batch-shaped event (carries a batch_run_id, so it is in the feed)."""
    return log.record(audit.BATCH_RUN_OPENED if hasattr(audit, "BATCH_RUN_OPENED")
                      else "batch_run.opened",
                      subject_type=audit.BATCH_RUN, subject_id=pid,
                      batch_run_id=pid, reason="x")


def test_a_fresh_load_returns_the_recent_tail_not_the_oldest():
    audit.AuditLog.reset_instances()
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    lg = audit.AuditLog(d / "audit.db")
    for i in range(250):
        lg.record("batch_run.opened", subject_type=audit.BATCH_RUN,
                  subject_id=f"run_{i}", batch_run_id=f"run_{i}", reason=str(i))
    tail = lg.batch_activity(since_event_id=0, limit=200)
    assert len(tail) == 200
    # oldest-first WITHIN the window, and the window is the newest 200
    ids = [e["event_id"] for e in tail]
    assert ids == sorted(ids)
    assert tail[-1]["reason"] == "249"       # the very last event is present
    assert tail[0]["reason"] == "50"         # ...and the ancient ones are not
    audit.AuditLog.reset_instances()


def test_the_incremental_poll_still_walks_forward(log):
    first = None
    for i in range(5):
        eid = log.record("batch_run.opened", subject_type=audit.BATCH_RUN,
                         subject_id=f"r{i}", batch_run_id=f"r{i}", reason=str(i))
        if first is None:
            first = eid
    after = log.batch_activity(since_event_id=first, limit=200)
    assert [e["reason"] for e in after] == ["1", "2", "3", "4"]


def test_the_client_replaces_on_the_initial_backfill_not_appends():
    """The page must clear the restored cache before rendering the since=0
    backfill, or a refresh doubles every line."""
    from recovery_agent import frontend
    src = frontend.INDEX_HTML if hasattr(frontend, "INDEX_HTML") else ""
    if not src:
        import pathlib
        src = (pathlib.Path(frontend.__file__).resolve().parent
               / "templates" / "index.html").read_text()
    assert "restoreEngineCache" in src
    assert "engineCursor === 0" in src        # the initial-backfill guard
    assert 'querySelectorAll(".ln").forEach' in src   # it clears before render


def test_the_live_monologue_never_opens_a_batch_tab():
    """Batch runs notify the batch view via push_event("batch", ...), which
    also fans out on the live-session channel. The live monologue is
    per-payment only and must ignore that reserved id — no phantom "batch" tab."""
    from recovery_agent import frontend
    import pathlib
    src = (pathlib.Path(frontend.__file__).resolve().parent
           / "templates" / "index.html").read_text()
    i = src.index("function sessionFor(")
    body = src[i:i + 600]
    assert 'pid === "batch"' in body          # sessionFor refuses the reserved id
    # and the socket handler bails before making it the streaming target
    assert 'data.payment_id === "batch"' in src
