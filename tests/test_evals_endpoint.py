"""The /api/evals contract the EVALS view reads.

The load-bearing case here is the ABSENT one. A fresh checkout has never run
the evals, so there is no scorecard — and the demo must not 500 because of
that. Absent data has to arrive as absent, never as zeros: a zero on the
red-team bar reads as "nothing held", which is the exact opposite of "we have
not measured yet", and that is the kind of quiet lie a dashboard tells right
in front of a judge.
"""
import json
from pathlib import Path

import pytest

from recovery_agent.frontend import app


def _get(path="/api/evals"):
    return json.loads(app.test_client().get(path).data)


# ── the endpoint must survive its own data being missing ────────────────

def test_a_fresh_checkout_with_no_scorecard_says_so_instead_of_failing(
        monkeypatch, tmp_path):
    """No scorecard is a normal state, not an error."""
    import recovery_agent.frontend as fe
    monkeypatch.setattr(fe, "__file__", str(tmp_path / "pkg" / "frontend.py"))
    r = app.test_client().get("/api/evals")
    assert r.status_code == 200, "a missing scorecard must not 500 the view"
    body = json.loads(r.data)
    assert body["ok"] is False
    assert "never been run" in body["reason"]
    # No fabricated numbers alongside the refusal.
    assert "modes" not in body


def test_an_unreadable_scorecard_reports_the_reason(monkeypatch, tmp_path):
    """Half-written JSON (a run killed mid-flight) is reported, not raised."""
    root = tmp_path / "pkg"
    (root / "evals" / "results").mkdir(parents=True)
    (root / "evals" / "results" / "scorecard.json").write_text("{not json")
    import recovery_agent.frontend as fe
    monkeypatch.setattr(fe, "__file__", str(root / "src" / "x" / "frontend.py"))
    r = app.test_client().get("/api/evals")
    assert r.status_code == 200
    body = json.loads(r.data)
    assert body["ok"] is False and "unreadable" in body["reason"].lower()


# ── the real scorecard ──────────────────────────────────────────────────

@pytest.fixture
def live():
    root = Path(__file__).resolve().parents[1]
    if not (root / "evals" / "results" / "scorecard.json").exists():
        pytest.skip("no scorecard in this checkout")
    return _get()


def test_it_serves_the_five_blocks_the_view_renders(live):
    assert live["ok"] is True
    for block in ("modes", "verdict", "corpus", "counterfactual", "baseline"):
        assert block in live, f"the view cannot render without {block}"


def test_the_verdict_counts_what_actually_gates(live):
    """`verified` must follow gateability, not the presence of numbers.

    This is the distinction the whole suite exists to make: a scorecard full
    of good-looking rates that none of the gates will stand behind is NOT a
    verified build, and the strip at the top of the view must say so.
    """
    v = live["verdict"]
    gateable = sum(1 for m in live["modes"].values() if m["gateable"])
    assert v["gated"] == gateable
    assert v["verified"] is (gateable > 0)


def test_every_mode_carries_its_reason_when_it_cannot_gate(live):
    """A mode that is not gateable must say why, in words the view can show."""
    for name, m in live["modes"].items():
        if not m["gateable"]:
            assert m["reasons"], f"{name} is ungated but gives no reason"


def test_turns_and_cases_are_reported_separately(live):
    """76 turns are not 76 independent samples.

    Collapsing them is precisely how an underpowered corpus passes for a
    strong one, so the endpoint must hand the view both numbers and let it
    show the difference.
    """
    c = live["corpus"]
    assert "turns" in c and "cases" in c
    assert c["effective_n"] == c["cases"]
    assert c["cases"] <= c["turns"]
    assert "min_cases_to_gate" in c, "the view has to draw the bar somewhere"


def test_the_counterfactual_carries_the_money_column(live):
    """The money chart is the exhibit; without giveaway it is just a score."""
    cf = live["counterfactual"]
    assert "error" not in cf, cf.get("error")
    policies = {r["policy"]: r for r in cf["results"]}
    assert "agent" in policies
    for row in cf["results"]:
        assert "giveaway_rupees" in row and "acted" in row
    # do_nothing must survive into the view: a conformance column that a
    # do-nothing policy wins is not measuring recovery, and hiding it would
    # make the agent's score look more impressive than it is.
    assert "do_nothing" in policies


# ── the view is wired to the rail ───────────────────────────────────────

def _template() -> str:
    return (Path(__file__).resolve().parents[1] / "src" / "recovery_agent" /
            "templates" / "index.html").read_text()


def test_every_view_has_both_a_rail_button_and_a_section():
    """The failure this guards is silent: a rail button that lights up while
    its section stays hidden. `showView` drives both from one table, so every
    row must name ids that actually exist in the page."""
    import re
    html = _template()
    rows = re.findall(
        r'\{name:\s*"(\w+)",\s*section:\s*"([\w-]+)",\s*rail:\s*"([\w-]+)",'
        r'\s*load:\s*function\s*\(\)\s*\{\s*return\s+(\w+)\(', html)
    assert rows, "the VIEWS table is missing or its shape changed"
    names = {r[0] for r in rows}
    assert "evals" in names, "the evals view fell out of the rail"
    for name, section, rail, loader in rows:
        assert f'id="{section}"' in html, f"{name}: no section #{section}"
        assert f'id="{rail}"' in html, f"{name}: no rail button #{rail}"
        assert f"function {loader}(" in html, f"{name}: no loader {loader}()"


def test_the_evals_view_never_renders_a_bare_zero_for_absent_data():
    """`ok:false` must reach a sentence, not the card grid.

    Zeros on the red-team bar would read as "nothing held" -- the opposite of
    "we have not measured yet" -- and that is a lie told in front of a judge.
    """
    html = _template()
    render = html[html.index("function renderEvals()"):]
    render = render[:render.index("\n}")]
    assert "if (!d.ok)" in render, "renderEvals must branch on ok before drawing"
    assert render.index("if (!d.ok)") < render.index("evCards("), \
        "the absent-data branch must return before any card is drawn"


# ── the suite must never write to the demo's observability ──────────────

def test_the_test_suite_does_not_export_traces_to_phoenix():
    """Tests share a machine with the running demo, and the collector
    defaults to localhost:6006 — so without this the suite exported straight
    into the real project. Eight runs in an afternoon buried 25 genuine agent
    traces under ~5,600 fragments. Observability you cannot read is not
    observability."""
    import os
    assert os.environ.get("PHOENIX_DISABLED") == "1", \
        "tests/conftest.py must disable tracing before any agent import"


def test_no_module_registers_its_own_tracer_provider():
    """OTel honours exactly ONE global provider. A second `register()` call
    anywhere silently demotes the first, which is how a hardcoded
    localhost:6006 in agentic_rag bypassed both PHOENIX_DISABLED and a remote
    collector. observability.init_observability() is the only init point."""
    import re
    src = Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
    offenders = []
    for py in src.rglob("*.py"):
        if py.name == "observability.py":
            continue
        body = py.read_text()
        if re.search(r"^\s*register\(", body, re.M) or \
           re.search(r"\bregister\(\s*endpoint\s*=", body):
            offenders.append(str(py.relative_to(src)))
    assert not offenders, f"these register their own provider: {offenders}"


def test_the_live_rate_agrees_with_the_frozen_scorecard(live):
    """The screen shows a number it recomputed on this request. If that ever
    disagrees with what the eval runner wrote, the dashboard is quietly telling
    a different story from CI — and a judge who checks would be right to stop
    believing both.

    The first attempt at this scored per TOOL CALL and got 70/72 against the
    runner's 74/76, because a turn calling three tools is ONE decision.
    """
    rc = live.get("recomputed") or {}
    assert "error" not in rc, rc.get("error")
    card = live["modes"]["recorded"]
    assert rc["decisions"] == card["decisions"], "different unit of analysis"
    assert round(rc["rate"], 3) == round(card["rate"], 3), (
        f"dashboard says {rc['rate']}, scorecard says {card['rate']}")


def test_the_recompute_needs_no_model_and_no_network():
    """It has to be honest AND instant, or nobody will press the button in
    front of an audience. Rule-based judging over the whole corpus is ~0.03s."""
    import json as _json
    import time
    from recovery_agent.evals.conformance import judge_decision
    rows = [_json.loads(l) for l in
            (Path(__file__).resolve().parents[1] / "evals" / "corpus" /
             "decisions.jsonl").read_text().splitlines() if l.strip()]
    t0 = time.time()
    for r in rows:
        judge_decision(r.get("facts") or {}, r.get("chosen") or [])
    assert time.time() - t0 < 2.0, "too slow to recompute live in a demo"


# ── the monologue keeps its bearings across cases ───────────────────────

def test_the_system_line_is_not_inside_the_swapped_terminal():
    """selectSession() replaces terminal.innerHTML wholesale, so anything
    parked inside the terminal as static markup is destroyed the first time a
    case arrives. The [SYSTEM] line — the one thing telling the operator the
    HUD is alive — vanished exactly when the HUD started doing something."""
    html = _template()
    term = html.index('id="terminal-stream"')
    sysline = html.index('id="hud-system-line"')
    assert sysline < term, "the system line is inside the region that gets swapped"
    # and the terminal must start empty, not carrying markup that will be lost
    tail = html[term:term + 200]
    assert 'id="terminal-stream"></div>' in tail, \
        "terminal-stream still ships static children that selectSession will erase"


def test_a_background_session_says_where_it_came_from():
    """A case waking from its own scheduled wait used to appear as a bare tab
    mid-stream, so work the operator never triggered read as the system acting
    at random."""
    html = _template()
    fn = html[html.index("function sessionFor("):]
    fn = fn[:fn.index("\n    }")]
    assert "[SESSION]" in fn, "a new session buffer carries no origin line"
    assert "background" in fn, "the line must say it ran while you were elsewhere"
