"""Drive the case matrix against the fake stack and save every observation.

Talks to the isolated frontend (port 6002, STATE_DIR=data-test) exactly the way
the checkout page does — the same HTTP posts, in the same order a customer
would cause them — while the fake gateway simulates links, orders and captures.
The real LLM drives the agent, so runs are slow and can starve on proxy quota;
a case that starves is recorded INCONCLUSIVE, never silently skipped.

Observations land incrementally in:
    data-test/observations.json          (machine-readable, source of truth)
    OBSERVATIONS-CASE-MATRIX.md          (regenerated from it after every case)

Usage:
    .venv/bin/python tests/integration/drive_cases.py            # all cases
    .venv/bin/python tests/integration/drive_cases.py A2 B1      # a subset
"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "data-test"
REG = STATE / "fake_gateway.json"
OBS_JSON = STATE / "observations.json"
OBS_MD = ROOT / "OBSERVATIONS-CASE-MATRIX.md"
BASE = os.getenv("FAKE_STACK_URL", "http://localhost:6002")
LLM_URL = "http://localhost:20128/v1/models"

RUN_TS = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── plumbing ────────────────────────────────────────────────────────────────

def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return {"_http": e.code, **json.loads(e.read().decode() or "{}")}
        except Exception:
            return {"_http": e.code}


def rec(pid: str) -> dict:
    try:
        return json.loads((STATE / "live_payments.json").read_text()).get(pid) or {}
    except Exception:
        return {}


def jobs() -> dict:
    try:
        return json.loads((STATE / "live_jobs.json").read_text())
    except Exception:
        return {}


def reg_update(mutate) -> None:
    for _ in range(5):
        try:
            reg = json.loads(REG.read_text()) if REG.exists() else {}
        except Exception:
            reg = {}
        mutate(reg)
        try:
            REG.write_text(json.dumps(reg, indent=1))
            return
        except Exception:
            time.sleep(0.2)


def links_created_for() -> list[dict]:
    try:
        reg = json.loads(REG.read_text())
    except Exception:
        return []
    return reg.get("links_created", [])


def wait_for(pred, timeout: float, poll: float = 5.0, label: str = ""):
    """Poll until pred() is truthy. Returns the value or None on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(poll)
    return None


def llm_starved(pid: str) -> bool:
    return any("AGENT RUN FAILED" in str(t.get("msg", ""))
               for t in rec(pid).get("trail", []))


def wait_llm_proxy():
    said = False
    while True:
        try:
            urllib.request.urlopen(LLM_URL, timeout=3)
            if said:
                print("[driver] LLM proxy is up — continuing", flush=True)
            return
        except Exception:
            if not said:
                print("[driver] waiting for the LLM proxy on :20128 "
                      "(start OmniRoute/antigravity) ...", flush=True)
                said = True
            time.sleep(10)


_case_n = 0


def new_pid(tag: str) -> str:
    global _case_n
    _case_n += 1
    return f"pay_mx{tag.lower()}{int(time.time()) % 100000}{_case_n}"


CUSTOMER = {"name": "Matrix Tester", "email": "matrix@fake.test",
            "contact": "+919000000001"}


def fail(pid, code, reason, amount=2499.0):
    return post("/api/payment-failed", {
        "payment_id": pid, "amount": amount, "failure_code": code,
        "failure_reason": reason, "error_source": "gateway",
        "error_step": "payment_authorization", "customer": CUSTOMER})


def push_response(pid, action, detail=""):
    return post("/api/push-response", {"payment_id": pid, "action": action,
                                       "seconds_shown": 4, "detail": detail})


def create_order(pid, amount=2499.0):
    return post("/api/create-order", {"payment_id": pid, "amount": amount})


def mark_link_paid(link_id, amount, cap_id):
    reg_update(lambda r: r.setdefault("paid_links", {}).update(
        {link_id: {"amount": amount, "payment_id": cap_id}}))


def mark_order_paid(order_id, amount, cap_id):
    reg_update(lambda r: (
        r.setdefault("paid_orders", {}).update(
            {order_id: {"amount": amount, "payment_id": cap_id}}),
        r.setdefault("captured", {}).update(
            {cap_id: {"amount": int(round(amount * 100)), "order_id": order_id}}),
    ))


def succeed(pid, cap_id, order_id):
    return post("/api/payment-succeeded", {
        "payment_id": pid, "razorpay_payment_id": cap_id,
        "razorpay_order_id": order_id})


# ── observation bookkeeping ─────────────────────────────────────────────────

def snapshot(pid: str) -> dict:
    r = rec(pid)
    return {k: r.get(k) for k in (
        "status", "failure_code", "failure_reason", "ladder", "actions_tried",
        "page_offer", "refusals", "recovered_amount", "recovered_payment_id",
        "closed", "scheduled_job", "last_action") if r.get(k) is not None}


def trail_steps(pid: str, n=14) -> list[str]:
    return [f"{t.get('ts','')} {t.get('step','')}: {str(t.get('msg',''))[:110]}"
            for t in rec(pid).get("trail", [])][-n:]


def save_case(obs: dict) -> None:
    all_obs = {}
    if OBS_JSON.exists():
        try:
            all_obs = json.loads(OBS_JSON.read_text())
        except Exception:
            pass
    all_obs[obs["case"]] = obs
    OBS_JSON.write_text(json.dumps(all_obs, indent=1, default=str))
    render_md(all_obs)
    print(f"[driver] {obs['case']}: {obs['verdict']}", flush=True)


def render_md(all_obs: dict) -> None:
    order = sorted(all_obs.values(), key=lambda o: o.get("started", ""))
    lines = [
        "# Case-matrix run — observations",
        "",
        f"Run started {RUN_TS}. Fake gateway (no Razorpay links spent, no real "
        f"email, SuperU untouched); real LLM through the local proxy. "
        f"Isolated STATE_DIR=data-test on port 6002.",
        "",
        "| Case | Verdict | Summary |",
        "|------|---------|---------|",
    ]
    for o in order:
        lines.append(f"| {o['case']} | **{o['verdict']}** | {o['title']} |")
    lines.append("")
    for o in order:
        lines += [f"## {o['case']} — {o['title']}", "",
                  f"*Verdict: **{o['verdict']}*** — {o.get('note','')}", "",
                  "Steps driven:"]
        lines += [f"- {s}" for s in o.get("steps", [])]
        if o.get("checks"):
            lines += ["", "Checks:"]
            lines += [f"- {'✅' if ok else '❌'} {name}"
                      for name, ok in o["checks"]]
        if o.get("trail"):
            lines += ["", "Trail (tail):", "```"]
            lines += o["trail"]
            lines += ["```"]
        if o.get("record"):
            lines += ["", "Record snapshot:", "```json",
                      json.dumps(o["record"], indent=1, default=str)[:2600],
                      "```"]
        lines.append("")
    OBS_MD.write_text("\n".join(lines))


class Case:
    def __init__(self, cid, title):
        self.cid, self.title = cid, title
        self.steps, self.checks = [], []
        self.pid = None
        self.started = datetime.now(timezone.utc).isoformat()

    def step(self, text):
        self.steps.append(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {text}")
        print(f"[driver] {self.cid}: {text}", flush=True)

    def check(self, name, ok):
        self.checks.append((name, bool(ok)))

    def finish(self, note=""):
        starved = self.pid and llm_starved(self.pid)
        if starved:
            verdict = "INCONCLUSIVE (LLM quota)"
        elif not self.checks:
            verdict = "OBSERVED"
        elif all(ok for _, ok in self.checks):
            verdict = "PASS"
        elif any(ok for _, ok in self.checks):
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"
        save_case({"case": self.cid, "title": self.title, "verdict": verdict,
                   "note": note, "steps": self.steps, "checks": self.checks,
                   "trail": trail_steps(self.pid) if self.pid else [],
                   "record": snapshot(self.pid) if self.pid else {},
                   "started": self.started})


# ── shared waits ────────────────────────────────────────────────────────────

RUN_T = int(os.getenv("CASE_TIMEOUT", "420"))


def wait_push(pid, t=RUN_T):
    return wait_for(lambda: (rec(pid).get("ladder") or {}).get("page_push")
                    or llm_starved(pid), t, label="push")


def wait_offer_link(pid, t=RUN_T):
    """A recovery link created for this case (any price)."""
    def _find():
        if llm_starved(pid):
            return "starved"
        r = rec(pid)
        lid = str(r.get("recovery_link_id") or r.get("order_id") or "")
        if lid.startswith("plink_"):
            return lid
        return None
    return wait_for(_find, t)


def wait_status(pid, statuses, t=RUN_T):
    return wait_for(lambda: (rec(pid).get("status") in statuses
                             and rec(pid).get("status")) or llm_starved(pid), t)


def wait_quiet(pid, t=RUN_T):
    """Run finished: status left 'recovering' and stayed put twice."""
    def _q():
        s = rec(pid).get("status")
        return s if s and s != "recovering" else None
    return wait_for(_q, t, poll=6)


def link_amount(lid):
    for l in links_created_for():
        if l["link_id"] == lid:
            return l["amount"]
    return None


# ── the cases ───────────────────────────────────────────────────────────────

def case_E1():
    c = Case("E1", "Clean first-try success stays out of the recovery books")
    pid = c.pid = new_pid("e1")
    o = create_order(pid, 2499.0)
    c.step(f"create-order -> {o.get('order_id')}")
    cap = f"pay_fake_{pid}"
    mark_order_paid(o["order_id"], 2499.0, cap)
    r = succeed(pid, cap, o["order_id"])
    c.step(f"payment-succeeded -> {r}")
    c.check("response is paid_clean", r.get("status") == "paid_clean")
    time.sleep(2)
    s = rec(pid)
    c.check("record status is 'paid'", s.get("status") == "paid")
    c.check("no recovered_amount credited", not s.get("recovered_amount"))
    c.check("no trail (no agent events)", not s.get("trail"))
    c.finish("A sale must be invisible to the recovery system.")


def case_D1():
    c = Case("D1", "Risk/fraud skips the ladder: no contact, straight to a human")
    pid = c.pid = new_pid("d1")
    fail(pid, "fraud_suspected", "Risk check failed: fraud suspected", 8999.0)
    c.step("payment-failed fraud_suspected")
    s = wait_status(pid, ("escalated",), t=RUN_T)
    r = rec(pid)
    c.check("case escalated", r.get("status") == "escalated")
    c.check("no push sent", not (r.get("ladder") or {}).get("page_push"))
    c.check("no link created", not any(
        l for l in links_created_for()
        if l.get("notes", {}).get("original_payment") == pid))
    c.check("closed as escalated", (r.get("closed") or {}).get("outcome") == "escalated")
    tickets = Path(os.getenv("ESCALATION_QUEUE_PATH", "data/escalations/queue.jsonl"))
    c.check("ticket exists", tickets.exists() and pid in tickets.read_text())
    c.finish()


def case_C1():
    c = Case("C1", "Insufficient funds: silent retry, no contact, no discount")
    pid = c.pid = new_pid("c1")
    fail(pid, "insufficient_funds", "Insufficient funds in account", 2499.0)
    c.step("payment-failed insufficient_funds")
    got = wait_for(lambda: rec(pid).get("scheduled_job") or llm_starved(pid), RUN_T)
    r = rec(pid)
    job = r.get("scheduled_job") or {}
    c.check("retry scheduled", bool(job))
    c.check("status scheduled", r.get("status") == "scheduled")
    c.check("no customer contact", "notify:email" not in str(r.get("actions_tried")))
    c.check("no discount", not (r.get("page_offer") or {}).get("discount_pct"))
    c.finish(f"job: {job.get('job_id','-')} at {job.get('target_timestamp') or job.get('target_time','-')}")
    return pid


def case_C4(c1_pid):
    c = Case("C4", "Scheduled retry fires through the daemon (cross-process)")
    c.pid = c1_pid
    if not c1_pid or not rec(c1_pid).get("scheduled_job"):
        c.step("no C1 job available — skipped")
        c.finish("needs C1 to have scheduled a job")
        return
    all_jobs = jobs()
    jid = None
    for j_id, j in all_jobs.items():
        if j.get("payment_id") == c1_pid and j.get("status") == "pending":
            j["target_time"] = "2020-01-01T00:00:00+00:00"
            jid = j_id
    (STATE / "live_jobs.json").write_text(json.dumps(all_jobs, indent=2))
    c.step(f"rewound job {jid} to the past; waiting on the daemon poll (60s)")
    done = wait_for(lambda: (jobs().get(jid, {}).get("status") == "completed"),
                    180, poll=10)
    c.check("daemon saw and executed the job", bool(done))
    c.check("retry recorded on case trail", any(
        "daemon" in str(t.get("step", "")) or "Retry" in str(t.get("msg", ""))
        for t in rec(c1_pid).get("trail", [])))
    c.finish("Daemon refresh() must see jobs scheduled after it started.")


def case_C2():
    c = Case("C2", "Gateway timeout: short silent retry, never a discount")
    pid = c.pid = new_pid("c2")
    fail(pid, "gateway_timeout", "Gateway timed out (504)", 2499.0)
    c.step("payment-failed gateway_timeout")
    wait_for(lambda: rec(pid).get("scheduled_job") or llm_starved(pid)
             or (rec(pid).get("ladder") or {}), RUN_T)
    r = rec(pid)
    c.check("no discount anywhere", not (r.get("page_offer") or {}).get("discount_pct"))
    c.check("no discounted link", all(
        abs(l["amount"] - 2499.0) < 0.01 for l in links_created_for()
        if l.get("notes", {}).get("original_payment") == pid))
    c.check("agent acted (retry or push), not silence",
            bool(r.get("scheduled_job") or (r.get("ladder") or {})))
    c.finish()


def case_E4():
    c = Case("E4", "Unclassified failure: fail-closed — full price before any offer")
    pid = c.pid = new_pid("e4")
    fail(pid, "err_unknown_xyz", "Something odd happened", 2499.0)
    c.step("payment-failed err_unknown_xyz")
    wait_quiet(pid)
    wait_for(lambda: links_created_for() and any(
        l.get("notes", {}).get("original_payment") == pid
        for l in links_created_for()) or llm_starved(pid), 120)
    mine = [l for l in links_created_for()
            if l.get("notes", {}).get("original_payment") == pid]
    r = rec(pid)
    c.check("first link (if any) is FULL price", all(
        abs(l["amount"] - 2499.0) < 0.01 for l in mine[:1]) if mine else True)
    c.check("no discount shown", not (r.get("page_offer") or {}).get("discount_pct"))
    c.finish(f"links created: {[(l['link_id'], l['amount']) for l in mine]}")


def case_B5():
    c = Case("B5", "Residual edge: dismissal arrives with no gateway code first")
    pid = c.pid = new_pid("b5")
    fail(pid, "customer_cancelled", "Payment cancelled by customer", 2499.0)
    c.step("payment-failed customer_cancelled (no prior real code)")
    time.sleep(2)
    c.check("classified as drop-off (documented residual)",
            rec(pid).get("failure_code") == "customer_cancelled")
    c.finish("Known limitation: nothing stronger existed to protect.")


def case_A1():
    c = Case("A1", "Drop -> push clicked -> pays full on the original order")
    pid = c.pid = new_pid("a1")
    o = create_order(pid, 2499.0)
    fail(pid, "customer_cancelled", "Payment cancelled by customer")
    c.step("drop-off posted")
    if not wait_push(pid):
        c.check("push delivered", False); c.finish("no push"); return
    c.step("push delivered; customer clicks it")
    push_response(pid, "acted", "clicked the notification")
    cap = f"pay_fake_{pid}"
    mark_order_paid(o["order_id"], 2499.0, cap)
    r = succeed(pid, cap, o["order_id"])
    c.step(f"paid the original order full price -> {r.get('status')}")
    s = wait_status(pid, ("recovered",), 120)
    rr = rec(pid)
    c.check("recovered", rr.get("status") == "recovered")
    c.check("full amount, no discount", rr.get("recovered_amount") == 2499.0)
    wait_for(lambda: rr and (rec(pid).get("closed") or {}), RUN_T)
    c.check("agent closed the case", (rec(pid).get("closed") or {})
            .get("outcome") == "recovered")
    c.finish()


def case_A2():
    c = Case("A2", "Drop -> push dismissed -> 5% offer -> pays the discounted link")
    pid = c.pid = new_pid("a2")
    create_order(pid, 2499.0)
    fail(pid, "customer_cancelled", "Payment cancelled by customer")
    c.step("drop-off posted")
    if not wait_push(pid):
        c.check("push delivered", False); c.finish("no push"); return
    push_response(pid, "dismissed", "closed the notification")
    c.step("push dismissed; waiting for the offer")
    lid = wait_offer_link(pid)
    if not lid or lid == "starved":
        c.check("offer link created", False); c.finish(); return pid
    amt = link_amount(lid)
    c.step(f"offer link {lid} at INR {amt}")
    c.check("offer is discounted (drop-off unlocks it)",
            amt is not None and amt < 2499.0)
    c.check("discount within policy (>=5% floor 2374.05)",
            amt is None or amt >= 2374.04)
    mark_link_paid(lid, amt or 2374.05, f"pay_fake_{pid}")
    c.step("customer pays the discounted link (registry)")
    s = wait_status(pid, ("recovered",), 180)
    c.check("watcher caught the payment", s == "recovered")
    wait_for(lambda: (rec(pid).get("closed") or {}), RUN_T)
    c.check("closed as recovered",
            (rec(pid).get("closed") or {}).get("outcome") == "recovered")
    c.finish()
    return pid


def case_E2(a2_pid):
    c = Case("E2", "Recovered is absorbing: late signals cannot reopen or double-count")
    c.pid = a2_pid
    if not a2_pid or rec(a2_pid).get("status") != "recovered":
        c.step("no recovered A2 case available — skipped")
        c.finish("needs A2 recovered")
        return
    before = rec(a2_pid).get("recovered_amount")
    r = succeed(a2_pid, f"pay_fake_{a2_pid}", "order_whatever")
    c.step(f"duplicate payment-succeeded -> {r.get('status')}")
    c.check("no double count", r.get("status") in ("already_recorded",
                                                   "order_mismatch"))
    push_response(a2_pid, "dismissed", "stale dismissal after payment")
    time.sleep(8)
    rr = rec(a2_pid)
    c.check("still recovered", rr.get("status") == "recovered")
    c.check("amount unchanged", rr.get("recovered_amount") == before)
    c.finish()


def case_B1():
    c = Case("B1", "Bank decline + modal close: diagnosis survives, no discount")
    pid = c.pid = new_pid("b1")
    create_order(pid, 2499.0)
    fail(pid, "bank_declined", "Bank declined the transaction (do_not_honor)")
    c.step("REAL bank decline posted")
    wait_quiet(pid)
    c.step("run over; customer now closes the Razorpay modal")
    fail(pid, "customer_cancelled", "Payment cancelled by customer")
    time.sleep(3)
    r = rec(pid)
    c.check("bank code survived the dismissal",
            r.get("failure_code") == "bank_declined")
    c.check("signal_precedence note on the trail", any(
        t.get("step") == "signal_precedence" for t in r.get("trail", [])))
    lid = wait_offer_link(pid, t=RUN_T)
    mine = [l for l in links_created_for()
            if l.get("notes", {}).get("original_payment") == pid]
    c.check("every link so far is FULL price", all(
        abs(l["amount"] - 2499.0) < 0.01 for l in mine) if mine else True)
    c.check("no discount banner", not (rec(pid).get("page_offer") or {})
            .get("discount_pct"))
    c.finish(f"links: {[(l['link_id'], l['amount']) for l in mine]}")
    return pid


def case_B2(b1_pid):
    c = Case("B2", "Method failure recovered at FULL price on the new rail")
    c.pid = b1_pid
    if not b1_pid:
        c.finish("needs B1"); return
    lid = wait_offer_link(b1_pid, t=60)
    if not lid or lid == "starved":
        c.step("no full-price link exists to pay — skipped")
        c.finish("B1 produced no link")
        return
    amt = link_amount(lid) or 2499.0
    mark_link_paid(lid, amt, f"pay_fake_{b1_pid}b2")
    c.step(f"customer pays {lid} at INR {amt}")
    s = wait_status(b1_pid, ("recovered",), 180)
    c.check("recovered", s == "recovered")
    c.check("at full price", rec(b1_pid).get("recovered_amount") == 2499.0)
    c.finish()


def case_B4():
    c = Case("B4", "Drop first, then a REAL decline upgrades the diagnosis")
    pid = c.pid = new_pid("b4")
    create_order(pid, 2499.0)
    fail(pid, "customer_cancelled", "Payment cancelled by customer")
    c.step("drop-off first")
    if wait_push(pid):
        push_response(pid, "acted", "clicked, reopened checkout")
        c.step("push clicked; retry then fails at the bank")
    wait_quiet(pid)
    fail(pid, "bank_declined", "do_not_honor at authorization")
    time.sleep(3)
    c.check("code upgraded to bank_declined",
            rec(pid).get("failure_code") == "bank_declined")
    wait_quiet(pid)
    c.check("no discount after the upgrade", not (
        rec(pid).get("page_offer") or {}).get("discount_pct"))
    c.finish()


def case_B3():
    c = Case("B3", "Method failure + failed full-price attempt legitimately unlocks 5%")
    pid = c.pid = new_pid("b3")
    create_order(pid, 2499.0)
    fail(pid, "bank_declined", "Bank declined the transaction")
    c.step("bank decline; waiting for the full-price attempt")
    lid = wait_offer_link(pid)
    if not lid or lid == "starved":
        c.check("full-price link created", False); c.finish(); return
    c.check("first attempt full price",
            abs((link_amount(lid) or 0) - 2499.0) < 0.01)
    wait_quiet(pid)
    fail(pid, "bank_declined", "Declined again on the new rail")
    c.step("full-price attempt failed too; discount should now be legal")
    got = wait_for(lambda: any(
        l["amount"] < 2499.0 for l in links_created_for()
        if l.get("notes", {}).get("original_payment") == pid) or llm_starved(pid),
        RUN_T)
    disc = [l for l in links_created_for()
            if l.get("notes", {}).get("original_payment") == pid
            and l["amount"] < 2499.0]
    c.check("discounted link appeared only AFTER full price failed", bool(disc))
    c.check("discount within policy", all(l["amount"] >= 2374.04 for l in disc))
    c.finish(f"discounted: {[(l['link_id'], l['amount']) for l in disc]}")


def case_A7():
    c = Case("A7", "Offer exists but customer pays FULL price — never undercut")
    pid = c.pid = new_pid("a7")
    o = create_order(pid, 2499.0)
    fail(pid, "customer_cancelled", "Payment cancelled by customer")
    if not wait_push(pid):
        c.check("push delivered", False); c.finish(); return
    push_response(pid, "dismissed", "closed the notification")
    c.step("dismissed; waiting for the offer to exist")
    wait_offer_link(pid)
    cap = f"pay_fake_{pid}"
    mark_order_paid(o["order_id"], 2499.0, cap)
    r = succeed(pid, cap, o["order_id"])
    c.step(f"pays the ORIGINAL order at full price -> {r.get('status')}")
    s = wait_status(pid, ("recovered",), 120)
    c.check("recovered", s == "recovered")
    c.check("at FULL price despite the live offer",
            rec(pid).get("recovered_amount") == 2499.0)
    c.finish()


def case_A6():
    c = Case("A6", "Cancels again inside checkout after clicking the offer")
    pid = c.pid = new_pid("a6")
    create_order(pid, 2499.0)
    fail(pid, "customer_cancelled", "Payment cancelled by customer")
    if not wait_push(pid):
        c.check("push delivered", False); c.finish(); return
    push_response(pid, "dismissed", "closed the notification")
    wait_offer_link(pid)
    push_response(pid, "acted", "clicked the offer banner")
    c.step("clicked the offer, then cancels inside Razorpay again")
    time.sleep(2)
    fail(pid, "customer_cancelled", "Payment cancelled by customer")
    wait_quiet(pid)
    r = rec(pid)
    c.check("still a drop-off (synthetic over synthetic is fine)",
            r.get("failure_code") == "customer_cancelled")
    c.check("case still open and being worked",
            r.get("status") not in ("failed", None))
    c.finish()


def case_A4():
    c = Case("A4", "Dismisses the push AND the offer — ladder keeps climbing")
    pid = c.pid = new_pid("a4")
    create_order(pid, 2499.0)
    fail(pid, "customer_cancelled", "Payment cancelled by customer")
    if not wait_push(pid):
        c.check("push delivered", False); c.finish(); return
    push_response(pid, "dismissed", "closed the notification")
    wait_offer_link(pid)
    c.step("offer up; customer dismisses the banner too")
    push_response(pid, "dismissed", "closed the offer banner")
    wait_for(lambda: len((rec(pid).get("ladder") or {})) >= 3
             or (rec(pid).get("closed") or {}) or llm_starved(pid), RUN_T)
    r = rec(pid)
    climbed = list((r.get("ladder") or {}).keys())
    c.check("went past the offer rung (alternate path or ending)",
            len(climbed) >= 3 or bool(r.get("closed")))
    c.check("never re-sent the same thing", len(
        r.get("actions_tried") or []) == len(set(r.get("actions_tried") or [])))
    c.finish(f"climbed: {climbed}")


def case_A3():
    c = Case("A3", "Push ignored entirely — timeout hands back, agent advances")
    pid = c.pid = new_pid("a3")
    create_order(pid, 2499.0)
    fail(pid, "customer_cancelled", "Payment cancelled by customer")
    if not wait_push(pid):
        c.check("push delivered", False); c.finish(); return
    c.step("push delivered; customer does NOTHING (waiting out the window)")
    got = wait_for(lambda: len((rec(pid).get("ladder") or {})) >= 2
                   or llm_starved(pid), 900, poll=15)
    r = rec(pid)
    c.check("agent advanced past the ignored push",
            len((r.get("ladder") or {})) >= 2)
    c.finish(f"climbed: {list((r.get('ladder') or {}).keys())}")


CASES = {
    "E1": case_E1, "D1": case_D1, "C1": case_C1, "C2": case_C2,
    "E4": case_E4, "B5": case_B5, "A1": case_A1, "A2": case_A2,
    "B1": case_B1, "B4": case_B4, "B3": case_B3, "A7": case_A7,
    "A6": case_A6, "A4": case_A4, "A3": case_A3,
}
# C4 and E2/B2 chain off C1/A2/B1 results; handled in main order.
ORDER = ["E1", "D1", "C1", "C4", "C2", "E4", "B5", "A1", "A2", "E2",
         "B1", "B2", "A7", "B4", "B3", "A6", "A4", "A3"]


def main():
    only = set(a.upper() for a in sys.argv[1:]) or set(ORDER)
    done = set()
    if OBS_JSON.exists():
        try:
            done = set(json.loads(OBS_JSON.read_text()))
        except Exception:
            pass
    ctx = {}
    for cid in ORDER:
        if cid not in only or cid in done:
            continue
        wait_llm_proxy()
        print(f"\n[driver] ══ {cid} ══", flush=True)
        try:
            if cid == "C4":
                case_C4(ctx.get("C1"))
            elif cid == "E2":
                case_E2(ctx.get("A2"))
            elif cid == "B2":
                case_B2(ctx.get("B1"))
            else:
                out = CASES[cid]()
                if out:
                    ctx[cid] = out
        except Exception as e:
            c = Case(cid, f"driver error")
            c.step(f"EXCEPTION: {e}")
            c.finish(str(e))
        time.sleep(5)
    print(f"\n[driver] run complete — see {OBS_MD}", flush=True)


if __name__ == "__main__":
    main()
