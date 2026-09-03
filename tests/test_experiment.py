"""A holdout arm, so a recovery total can become a recovery claim.

Every number this system produced counted cases where money arrived after the
agent acted, and could not separate that from money that was coming anyway. On
`pay_9frnh6kjb` the customer paid in full while the agent sat blocked by a stale
flag across three runs — and that counted as a recovery.
"""
import pathlib

import pytest

from recovery_agent.experiment import CONTROL, TREATED, assign, results

FRONTEND = (pathlib.Path(__file__).resolve().parents[1] / "src" / "recovery_agent"
            / "frontend.py").read_text()


def _func(src: str, name: str) -> str:
    """The body of one function, by indentation — not a fixed-size window.

    Slicing N characters after a `def` breaks silently the moment the function
    grows, and searching the whole file finds the same string in a different
    function. Both have already happened in this repo's tests.
    """
    start = src.index(f"def {name}(")
    lines = src[start:].split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if line and not line[0].isspace() and not line.startswith(")"):
            break
        out.append(line)
    return "\n".join(out)



# ── assignment ──────────────────────────────────────────────────────────

def test_assignment_is_deterministic():
    """A case is re-entered every time the customer acts, a retry fires or a run
    hands off. A fresh coin flip each time would move cases between arms and
    destroy the comparison."""
    ids = [f"pay_{i}" for i in range(50)]
    first = [assign(i) for i in ids]
    assert [assign(i) for i in ids] == first
    assert [assign(i) for i in ids] == first


def test_the_split_is_roughly_the_configured_fraction(monkeypatch):
    monkeypatch.setenv("HOLDOUT_FRACTION", "0.2")
    n = 4000
    control = sum(assign(f"pay_{i}") == CONTROL for i in range(n))
    assert 0.16 < control / n < 0.24


def test_a_zero_fraction_disables_the_experiment(monkeypatch):
    monkeypatch.setenv("HOLDOUT_FRACTION", "0")
    assert all(assign(f"pay_{i}") == TREATED for i in range(100))


def test_the_holdout_is_capped_at_half(monkeypatch):
    """A configuration slip must not silently stop working most of the revenue."""
    monkeypatch.setenv("HOLDOUT_FRACTION", "0.95")
    from recovery_agent.experiment import holdout_fraction
    assert holdout_fraction() == 0.5


def test_a_bad_fraction_falls_back_rather_than_raising(monkeypatch):
    monkeypatch.setenv("HOLDOUT_FRACTION", "not-a-number")
    from recovery_agent.experiment import holdout_fraction
    assert holdout_fraction() == 0.2


# ── measurement ─────────────────────────────────────────────────────────

def _case(arm, recovered, amount=1000.0):
    r = {"payment_id": "x", "amount": amount, "arm": arm}
    if recovered:
        r["status"] = "recovered"
        r["recovered_amount"] = amount
    return r


def test_history_without_an_arm_is_excluded_not_counted_as_treated():
    """Everything recorded before the experiment was worked by an agent under
    active development. Folding it into the treated arm would inflate the
    denominator with runs that no longer represent the system."""
    out = results([{"payment_id": "old", "status": "recovered",
                    "recovered_amount": 500.0}])
    assert out["treated"]["n"] == 0 and out["control"]["n"] == 0


def test_lift_is_the_difference_between_the_arms():
    recs = ([_case(TREATED, True)] * 6 + [_case(TREATED, False)] * 4
            + [_case(CONTROL, True)] * 1 + [_case(CONTROL, False)] * 4)
    out = results(recs)
    assert out["treated"]["rate"] == 0.6
    assert out["control"]["rate"] == 0.2
    assert out["lift_pp"] == 40.0


def test_a_control_case_that_self_recovers_counts_for_the_control_arm():
    """That is the entire measurement — it is not a failure of the holdout."""
    out = results([_case(CONTROL, True), _case(CONTROL, False)])
    assert out["control"]["recovered"] == 1 and out["control"]["rate"] == 0.5


def test_overlapping_intervals_are_reported_as_not_yet_evidence():
    """Reporting a lift without this caveat is the thing the module exists to
    stop — at these sample sizes a point estimate alone is the same overclaim in
    a new costume."""
    out = results([_case(TREATED, True), _case(TREATED, False),
                   _case(CONTROL, False), _case(CONTROL, False)])
    assert out["conclusive"] is False
    assert "not yet separable" in out["verdict"]


def test_a_clear_separation_is_reported_as_such():
    recs = [_case(TREATED, True)] * 60 + [_case(CONTROL, False)] * 60
    out = results(recs)
    assert out["conclusive"] is True
    assert "do not overlap" in out["verdict"]


def test_no_control_cases_says_so_rather_than_showing_a_lift():
    out = results([_case(TREATED, True)])
    assert "no control cases yet" in out["verdict"]


def test_intervals_stay_inside_zero_and_one_hundred():
    """A normal approximation on 1 of 1 produces bounds outside [0,100], which
    would be worse than showing nothing."""
    out = results([_case(TREATED, True)])
    lo, hi = out["treated"]["ci"]
    assert 0.0 <= lo <= hi <= 100.0


# ── enforcement ─────────────────────────────────────────────────────────

def test_the_arm_is_written_once_and_never_recomputed():
    i = FRONTEND.index("THE HOLDOUT")
    body = FRONTEND[i:i + 1600]
    assert 'rec.get("arm") or assign(payment_id)' in body
    assert 'if not rec.get("arm"):' in body


def test_a_control_case_is_refused_where_the_work_would_start():
    """Five callers reach the agent — ingress, hand-offs, the batch runner, the
    simulator and the webhook. Guarding them one by one would break the
    experiment the first time a sixth was added."""
    body = _func(FRONTEND, "run_agent_for_payment")
    assert 'get("arm") == CONTROL' in body
    assert "not worked" in body


def test_a_control_case_is_still_watched_for_self_recovery():
    """A control case that recovers on its own IS the measurement."""
    i = FRONTEND.index("THE HOLDOUT")
    body = FRONTEND[i:i + 2200]
    assert "_watch_for_recovery" in body


def test_simulated_cases_are_excluded_from_the_experiment():
    """A demo click is not revenue; counting one in either arm would corrupt the
    only number that claims the agent caused something."""
    body = _func(FRONTEND, "simulate_scenario")
    assert '"arm": "excluded"' in body


def test_the_batch_runner_skips_the_control_arm():
    assert 'control arm — held back' in _func(FRONTEND, "api_run_batch")
