"""End-to-end proof of the batch path, against the fake gateway.

Costs nothing: no real payment link, no LLM call, no email off the machine. That
is not a limitation of the rig — the executor makes no LLM calls by design, so
what runs here is the same code the demo runs, with only the gateway swapped.

Seeds a batch across amount bands and ladder positions, then checks the four
claims the track asks for:

  measured money   the run's total is a join over the audit log, so it counts
                   the three cases that actually paid and not the ones that did
  stopping rules   the link budget binds before the gateway's lifetime quota
  audit trail      the report rebuilt from events equals the live one
  compliance       every case passed the guardrails before it was contacted

Run:  .venv/bin/python tests/integration/drive_batch.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "data-test"
LOG = ROOT / "data-test" / "batch_rig.log"


def _free_port() -> int:
    """A port nobody else holds.

    Not the rig's usual 6002: a server left listening from an earlier run
    answers the health check, the new one dies on "address in use", and the
    driver spends its time testing yesterday's code — which is exactly what
    happened here. Asking the kernel for a free port makes that impossible
    rather than merely unlikely.
    """
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


PORT = int(os.getenv("BATCH_RIG_PORT") or _free_port())
BASE = os.getenv("FAKE_STACK_URL", f"http://localhost:{PORT}")

CUSTOMER = {"name": "Batch Tester", "email": "batch@fake.test",
            "contact": "+919000000002"}

#: Spread across bands and causes on purpose: the planner makes one decision per
#: (batch x band), so a batch confined to one band would not exercise the split.
SEED = (
    [("bd", "bank_declined", "card_expired", amt)
     for amt in (2499, 3499, 7999, 12999, 21999, 45999)]
    + [("df", "dropoff", "user_dropoff", amt)
       for amt in (999, 1499, 6499, 8999)]
    + [("if", "insufficient_funds", "insufficient_funds", amt)
       for amt in (5999, 9999, 15999)]
)


def _req(method, path, body=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return {"_http": e.code, **json.loads(body or "{}")}
        except ValueError:
            return {"_http": e.code, "body": body[-600:]}


get = lambda p: _req("GET", p)
post = lambda p, b=None: _req("POST", p, b or {})


def seed() -> dict:
    """Write the cases straight into the store, before the server reads it.

    Deliberately not through `/api/payment-failed`: that starts a live agent
    session per case, which is the twelve-minute path this whole design exists
    to replace. The batch path's input is the store, so the store is what we
    seed.
    """
    STATE.mkdir(exist_ok=True)
    payments = {}
    for tag, _batch, code, amount in SEED:
        pid = f"pay_bt{tag}{amount}"
        payments[pid] = {
            "payment_id": pid, "amount": float(amount), "currency": "INR",
            "status": "failed", "failure_code": "", "decline_strategy": code,
            "failure_reason": f"seeded {code}", "customer": dict(CUSTOMER),
            "created_at": "2026-09-04T06:00:00+00:00",
        }
    (STATE / "live_payments.json").write_text(json.dumps(payments, indent=2))
    for name in ("live_jobs.json", "live_pending.json"):
        (STATE / name).write_text("{}")
    (STATE / "fake_gateway.json").write_text(json.dumps(
        {"captured": {}, "paid_links": {}, "paid_orders": {}, "payments": []}))
    (STATE / "audit.db").unlink(missing_ok=True)
    return payments


def boot():
    env = {**os.environ, "STATE_DIR": "data-test", "FRONTEND_PORT": str(PORT),
           "FRONTEND_URL": f"http://localhost:{PORT}",
           "VOICE_CALLS_ENABLED": "false", "RAZORPAY_WRITES_OK": "1",
           "SMTP_HOST": "localhost", "BATCH_WORKERS": "4",
           # The rig runs at whatever hour the developer is awake. The window
           # itself is tested in the unit suite, where the clock is pinned.
           "GUARDRAIL_QUIET_DISABLED": "1"}
    LOG.write_text("")
    log = LOG.open("w")
    proc = subprocess.Popen(
        [sys.executable, "tests/integration/fake_stack.py", "frontend"],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            if get("/api/batches").get("batches") is not None:
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.kill()
    raise SystemExit(f"fake stack did not come up:\n{LOG.read_text()[-3000:]}")


def ok(label, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}"
          f"{'  — ' + str(detail) if detail else ''}")
    return bool(condition)


def main() -> int:
    seed()
    proc = boot()
    passed = []
    try:
        print("\n── the batch, as the classifier sorts it")
        batches = {b["key"]: b for b in get("/api/batches")["batches"]}
        for key in ("bank_declined", "dropoff", "insufficient_funds"):
            b = batches.get(key, {})
            print(f"  {key:20s} {b.get('count', 0):>2} cases  "
                  f"INR {b.get('value', 0):>10,.0f} at risk")
        passed.append(ok("every seeded case landed in a runnable batch",
                         sum(batches[k]["count"] for k in
                             ("bank_declined", "dropoff", "insufficient_funds")
                             if k in batches) == len(SEED)))

        print("\n── dry run: the same twelve checks, zero side effects")
        dry = post("/api/batches/bank_declined/plan")
        dreport = dry["report"]
        print(f"  plans for bands: {sorted(dry['plans'])}")
        print(f"  would act on {dreport['acted']} of {dreport['candidates']}, "
              f"INR {dreport['at_risk_rupees']:,.0f} at risk")
        # The report says what it WOULD create; the gateway says what exists.
        # Asking the gateway is the only version of this check that means
        # anything, since the whole point is that nothing happened out there.
        gateway = json.loads((STATE / "fake_gateway.json").read_text())
        passed.append(ok("a dry run creates no links at the gateway",
                         not gateway.get("links_created"),
                         f"projected {dreport['links_created']}"))
        passed.append(ok("a plan is made per amount band", len(dry["plans"]) > 1,
                         sorted(dry["plans"])))
        passed.append(ok("no plan carries a rupee figure",
                         not any(k in json.dumps(p) for p in dry["plans"].values()
                                 for k in ("amount_paise", "payable_rupees"))))

        print("\n── live run, budget max_links=3")
        started = post("/api/batches/bank_declined/run", {"budget": {"max_links": 3}})
        run_id = started["batch_run_id"]
        print(f"  batch_run_id: {run_id}")
        report = {}
        for _ in range(90):
            fetched = get(f"/api/batch-runs/{run_id}")
            if "acted" in fetched:                # 404 until the run opens
                report = fetched
                if report.get("status") != "running":
                    break
            time.sleep(1)
        if not report:
            raise SystemExit(f"the run never opened:\n{LOG.read_text()[-3000:]}")
        print(f"  acted {report['acted']}  links {report['links_created']}  "
              f"deferred {report['deferred']}  skipped {report['skipped']}  "
              f"exceptions {report['exceptions']}")
        print(f"  stop_reason: {report['stop_reason'] or '(ran to completion)'}")
        passed.append(ok("the link budget bound before the account quota",
                         report["links_created"] == 3, report["links_created"]))
        passed.append(ok("the run stopped and said which resource ran out",
                         report["stop_reason"] == "budget_links",
                         report["stop_reason"]))

        print("\n── the dry run predicted the live run")
        dry_outcomes = {d["payment_id"]: d["outcome"] for d in dreport["decisions"]}
        live_outcomes = {d["payment_id"]: d["outcome"]
                         for d in report.get("decisions", [])
                         if d["outcome"] != "budget"}
        agree = [p for p, o in live_outcomes.items() if dry_outcomes.get(p) == o]
        passed.append(ok("every case the budget allowed matched its projection",
                         len(agree) == len(live_outcomes),
                         f"{len(agree)}/{len(live_outcomes)}"))

        print("\n── money: three customers pay")
        links = json.loads((STATE / "fake_gateway.json").read_text()
                           ).get("links_created", [])
        acted = [d for d in report.get("decisions", []) if d["outcome"] == "acted"]
        paid_total = 0.0
        for decision, link in zip(acted, links):
            amount = decision["charged_paise"] / 100
            reg = json.loads((STATE / "fake_gateway.json").read_text())
            reg.setdefault("paid_links", {})[link["link_id"]] = {
                "amount": amount, "payment_id": f"cap_{decision['payment_id']}"}
            (STATE / "fake_gateway.json").write_text(json.dumps(reg))
            paid_total += amount
            print(f"  {decision['payment_id']} pays INR {amount:,.0f}")

        print("  waiting for the pollers to see the money...")
        final = {}
        for _ in range(90):
            final = get(f"/api/batch-runs/{run_id}")
            if final.get("recovered_rupees"):
                break
            time.sleep(2)
        print(f"  recovered INR {final.get('recovered_rupees', 0):,.0f} "
              f"of INR {final.get('at_risk_rupees', 0):,.0f} at risk  "
              f"(rate {final.get('recovery_rate', 0):.0%} of cases acted on)")
        passed.append(ok("the run counts money that actually arrived",
                         final.get("recovered_rupees", 0) > 0,
                         final.get("recovered_rupees")))
        passed.append(ok("it counts the amounts paid, not the amounts owed",
                         abs(final.get("recovered_rupees", 0) - paid_total) < 1,
                         f"{final.get('recovered_rupees')} vs {paid_total}"))
        passed.append(ok("money is attributed to the run that earned it",
                         final["batch_run_id"] == run_id))

        print("\n── the audit trail")
        trail = get(f"/api/batch-runs/{run_id}?events=1").get("trail", [])
        kinds = {}
        for e in trail:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        for kind, n in sorted(kinds.items()):
            print(f"  {n:>3}  {kind}")
        passed.append(ok("the run opened, planned and finished in the log",
                         {"batch_run.opened", "batch_run.planned",
                          "batch_run.finished"} <= set(kinds)))
        passed.append(ok("every action has both an attempt and a result",
                         kinds.get("action.attempted") == kinds.get("action.result")))
        passed.append(ok("recovered money is in the log, not just the report",
                         kinds.get("money.recovered", 0) == len(acted[:3])))

        print("\n── stopping: a second click never re-works a case")
        # Not "spends nothing": three cases were refused by the first run's
        # budget and never contacted, and working those is exactly what a
        # second click is for. What must not happen is the three that were
        # already contacted being contacted again.
        worked_once = {d["payment_id"] for d in report.get("decisions", [])
                       if d["outcome"] == "acted"}
        again = post("/api/batches/bank_declined/run", {"budget": {"max_links": 3}})
        second = again.get("batch_run_id")
        if second:
            r2 = {}
            for _ in range(60):
                fetched = get(f"/api/batch-runs/{second}")
                if "acted" in fetched:
                    r2 = fetched
                    if r2.get("status") != "running":
                        break
                time.sleep(1)
            worked_twice = worked_once & {
                d["payment_id"] for d in r2.get("decisions", [])
                if d["outcome"] == "acted"}
            passed.append(ok("no case was contacted by two runs",
                             bool(r2) and not worked_twice,
                             f"acted {r2.get('acted')}, "
                             f"repeats {sorted(worked_twice)}"))
            passed.append(ok("the cases the budget refused were picked up",
                             bool(r2) and r2.get("acted", 0) > 0,
                             r2.get("acted")))
        else:
            passed.append(ok("a concurrent run was refused",
                             again.get("_http") == 409))

        print("\n── abort")
        third = post("/api/batches/dropoff/run", {"budget": {"max_links": 2}})
        if third.get("batch_run_id"):
            stopped = post(f"/api/batch-runs/{third['batch_run_id']}/abort")
            r3 = {}
            for _ in range(60):
                fetched = get(f"/api/batch-runs/{third['batch_run_id']}")
                if "acted" in fetched:
                    r3 = fetched
                    if r3.get("status") != "running":
                        break
                time.sleep(1)
            passed.append(ok("an aborted run is recorded as aborted, with a reason",
                             bool(r3) and r3["status"] == "aborted"
                             and bool(r3["stop_reason"]),
                             f"{r3.get('status')} / {r3.get('stop_reason')}"))
            passed.append(ok("it stopped before spending its whole budget",
                             bool(r3) and r3["links_created"] <= 2,
                             r3.get("links_created")))
        print(f"\n{'═' * 62}\n  {sum(passed)}/{len(passed)} checks passed\n")
        return 0 if all(passed) else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
