"""Every name a module imports must actually exist.

`frontend.py` imported `register_retry_job` from `daemon_worker`, and that
function had never been written. Because the import sat inside a rarely-reached
branch, nothing failed until the agent successfully scheduled a silent retry —
at which point the whole case died with

    Agent Execution Error: cannot import name 'register_retry_job'

*after* the agent had done the right thing. A function-local import is invisible
to the interpreter at load time and to every test that does not walk that exact
branch, so it is checked here statically instead.
"""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "recovery_agent"

#: Modules we can import without side effects worth avoiding in a test run.
#: Modules that cannot be imported in a bare test environment. Empty now
#: that the eval tree is gone — nothing in the package needs an exception.
_SKIP_MODULES: set[str] = set()


def _local_imports(path: pathlib.Path):
    """Every `from recovery_agent.x import a, b` in a file, at any nesting depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("recovery_agent"):
            if node.level:
                continue                      # relative imports: out of scope here
            yield node.module, [a.name for a in node.names], node.lineno


def _iter_source_files():
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def test_every_imported_name_exists():
    missing = []
    for path in _iter_source_files():
        for module, names, lineno in _local_imports(path):
            if module in _SKIP_MODULES:
                continue
            try:
                mod = importlib.import_module(module)
            except Exception as exc:                       # pragma: no cover
                missing.append(f"{path.name}:{lineno} cannot import {module}: {exc}")
                continue
            for name in names:
                if name == "*":
                    continue
                if hasattr(mod, name):
                    continue
                # `from recovery_agent import push_bus` imports a SUBMODULE, which
                # is not an attribute of the package until it has been loaded —
                # so a plain hasattr reports a false missing name.
                try:
                    importlib.import_module(f"{module}.{name}")
                    continue
                except ImportError:
                    pass
                missing.append(f"{path.name}:{lineno} {module} has no {name!r}")

    assert not missing, "imports that would fail at runtime:\n  " + "\n  ".join(missing)


def test_register_retry_job_specifically_exists():
    """The one that actually shipped broken — pinned so it cannot vanish again."""
    from recovery_agent.daemon_worker import register_retry_job
    assert callable(register_retry_job)


def test_a_registered_retry_reaches_the_daemon_queue(tmp_path, monkeypatch):
    """Registering a job must make it visible to the worker that executes it."""
    import datetime as dt

    from recovery_agent import state_store as ss

    monkeypatch.setattr(ss.StateStore, "_dir", tmp_path, raising=False)
    from recovery_agent.daemon_worker import register_retry_job

    store = ss.StateStore(data_dir=tmp_path)
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).isoformat()

    monkeypatch.setattr("recovery_agent.state_store.StateStore",
                        lambda *a, **k: store, raising=False)
    job = register_retry_job(payment_id="pay_x", amount=1499.0, target_timestamp=past,
                             customer={"email": "a@b.com"}, reason="payday window")

    assert job["status"] == "scheduled"
    assert job["job_id"] in {j["job_id"] for j in store.get_due_jobs()}
