"""Waves against the opposite batch — the scored end-to-end proof.

Seeds a population of simulated customers (each acting only through real
surfaces: the inbox, a payment link, a retried order), starts the rig on its
own port, runs a wave cycle over them, and scores the outcome against the
population's own ground truth. Free: zero real links, zero LLM calls in the
default mode, zero voice.

Run:  .venv/bin/python tests/integration/drive_waves.py [--champion]
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "integration"))
os.environ["STATE_DIR"] = "data-test"

import customer_sim  # noqa: E402


def _free_port() -> int:
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


PORT = int(os.getenv("BATCH_RIG_PORT") or _free_port())
BASE = f"http://localhost:{PORT}"
LOG = STATE / "waves_rig.log"


def _req(method, path, body=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return {"_http": e.code, **json.loads(raw or "{}")}
        except ValueError:
            return {"_http": e.code, "body": raw[-400:]}


get = lambda p: _req("GET", p)
post = lambda p, b=None: _req("POST", p, b or {})


def boot():
    env = {**os.environ, "STATE_DIR": "data-test",
           "FRONTEND_PORT": str(PORT), "FRONTEND_URL": BASE,
           "VOICE_CALLS_ENABLED": "false", "RAZORPAY_WRITES_OK": "1",
           "GUARDRAIL_QUIET_DISABLED": "1", "BATCH_WORKERS": "4"}
    STATE.mkdir(exist_ok=True)
    LOG.write_text("")
    proc = subprocess.Popen(
        [sys.executable, "tests/integration/fake_stack.py", "frontend"],
        cwd=ROOT, env=env, stdout=LOG.open("w"), stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            if get("/api/batches").get("batches") is not None:
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.kill()
    raise SystemExit(f"rig did not come up:\n{LOG.read_text()[-3000:]}")


def ok(label, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}"
          f"{'  — ' + str(detail) if detail else ''}")
    return bool(condition)


def main() -> int:
    champion = "--champion" in sys.argv
    population = customer_sim.build_population(per_archetype=1)
    customer_sim.seed_store(population, STATE)
    by_type = {c.archetype: c for c in population}

    proc = boot()
    sim = customer_sim.Simulator(population, STATE, BASE)
    sim.start()
    passed = []
    try:
        print(f"\n── the population ({len(population)} customers, "
              f"champion={'live' if champion else 'off'})")
        for c in population:
            print(f"  {c.archetype:18s} {c.payment_id:26s} INR {c.amount:>8,.0f}")

        started = post("/api/batches/waves",
                       {"settle_seconds": 35, "max_waves": 3,
                        "champion": champion})
        if not started.get("cycle_id"):
            raise SystemExit(f"cycle did not start: {started}")
        cycle_id = started["cycle_id"]
        print(f"\n── wave cycle {cycle_id} over {started['cases']} cases")

        report, seen_waves = {}, 0
        for _ in range(150):
            report = get(f"/api/batch-cycles/{cycle_id}")
            for w in report.get("waves", [])[seen_waves:]:
                print(f"  wave {w['wave']}: {w['cases']} cases -> "
                      f"acted {w['acted']}, skipped {w['skipped']}, "
                      f"deferred {w['deferred']}, to-agent {w['exceptions']}, "
                      f"links {w['links_created']}")
                seen_waves += 1
            if report.get("status") != "running":
                break
            time.sleep(3)
        print(f"  stopped: {report.get('stop_reason')!r} "
              f"after {len(report.get('waves', []))} wave(s)")

        # Money resolves on read; give the last watchers one poll interval.
        time.sleep(20)
        report = get(f"/api/batch-cycles/{cycle_id}")
        card = customer_sim.score(population, STATE)

        print("\n── the scorecard")
        for row in card["rows"]:
            state = ("RECOVERED INR {:,.0f}".format(row["recovered_amount"])
                     if row["recovered"] else row["final_status"])
            print(f"  {row['archetype']:18s} emails={row['emails']} "
                  f"links={row['links']}  {state}")
        print(f"\n  recovered INR {card['recovered_paise'] / 100:,.0f} of "
              f"INR {card['recoverable_paise'] / 100:,.0f} recoverable "
              f"(capture {card['capture_rate']:.0%})")

        auto = [by_type[a] for a in customer_sim.PAYS_AUTOMATICALLY]
        print("\n── the claims")
        for c in auto:
            row = next(r for r in card["rows"]
                       if r["payment_id"] == c.payment_id)
            passed.append(ok(f"{c.archetype} recovered its money",
                             row["recovered"]
                             and abs(row["recovered_amount"] - c.amount) < 1,
                             f"{row['recovered_amount']} vs {c.amount}"))

        fd = next(r for r in card["rows"]
                  if r["archetype"] == "fails_differently")
        rec = json.loads((STATE / "live_payments.json").read_text())[
            by_type["fails_differently"].payment_id]
        passed.append(ok("fails_differently was re-binned to funds and "
                         "recovered by the funds treatment",
                         rec.get("decline_strategy") == "insufficient_funds"
                         and fd["recovered"]))

        dtt = next(r for r in card["rows"] if r["archetype"] == "drop_then_try")
        exception_ids = {e["payment_id"]
                         for e in report.get("exceptions", [])}
        if champion:
            passed.append(ok("drop_then_try paid the agent-judged second touch",
                             dtt["recovered"]))
        else:
            passed.append(ok(
                "drop_then_try reached the agent's desk, not a louder inbox",
                not dtt["recovered"] and dtt["emails"] == 1
                and dtt["payment_id"] in exception_ids,
                f"emails={dtt['emails']}"))

        np_ = next(r for r in card["rows"] if r["archetype"] == "never_pays")
        passed.append(ok("never_pays was contacted once, then handed to a "
                         "person — never chased",
                         not np_["recovered"] and np_["emails"] <= 1
                         and np_["payment_id"] in exception_ids,
                         f"emails={np_['emails']}"))

        passed.append(ok("opted_out was never contacted by anything",
                         not card["violations"], card["violations"]))
        passed.append(ok("the cycle stopped for a stated reason",
                         report.get("stop_reason") in
                         ("dry", "all_settled", "max_waves"),
                         report.get("stop_reason")))
        passed.append(ok("waves ran more than once",
                         len(report.get("waves", [])) >= 2,
                         len(report.get("waves", []))))
        passed.append(ok(
            "every recovered rupee is attributed to a run in the cycle",
            report.get("recovered_paise") == card["recovered_paise"],
            f"cycle {report.get('recovered_paise')} vs "
            f"ground truth {card['recovered_paise']}"))
        registry = json.loads((STATE / "fake_gateway.json").read_text())
        passed.append(ok("zero real payment links were spent",
                         all(l["link_id"].startswith("plink_fake")
                             for l in registry.get("links_created", []))))

        print(f"\n{'═' * 62}\n  {sum(passed)}/{len(passed)} claims held\n")
        return 0 if all(passed) else 1
    finally:
        sim.stop_flag.set()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
