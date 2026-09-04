"""Isolated integration stack with a FAKE payment gateway.

Constraints this rig exists to honour:
- ZERO Razorpay payment links are created (30-per-account, lifetime). Links
  come back as https://fake.rzp.test/... and cost nothing.
- SuperU is never touched; voice stays disabled.
- No real email leaves the machine: SMTP is a local sink file, and delivery is
  reported honestly at the transport boundary (the dispatcher's own logic runs
  unmodified).
- STATE_DIR=data-test keeps every byte away from the live demo data,
  including the agent's SQLite sessions and memory.

The gateway is driven by a registry file the case driver writes:
    data-test/fake_gateway.json
      {"captured":   {"pay_fake_x": {"amount": <PAISE>, "order_id": "..."}},
       "paid_links": {"plink_fakeN": {"amount": <RUPEES>, "payment_id": "..."}},
       "paid_orders":{"order_fakeN": {"amount": <RUPEES>, "payment_id": "..."}},
       "payments":   [ ...items for payment.all()... ]}
Every link the agent creates is appended to "links_created" so the driver can
see the amount the customer would have been charged.

Run:  .venv/bin/python tests/integration/fake_stack.py frontend
      .venv/bin/python tests/integration/fake_stack.py daemon
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "data-test"

os.environ.setdefault("STATE_DIR", str(STATE))
os.environ["OUTBOX_DIR"] = str(STATE / "outbox")
os.environ.setdefault("FRONTEND_PORT", "6002")
os.environ.setdefault("FRONTEND_URL", "http://localhost:6002")
os.environ["SMTP_HOST"] = "fake-sink.local"   # branch taken; transport patched
# escalation_queue does NOT follow STATE_DIR (its own env var) — found live
# when a test ticket landed in the real data/escalations queue.
os.environ.setdefault("ESCALATION_QUEUE_PATH",
                      str(STATE / "escalations" / "queue.jsonl"))
# Belt and braces: even if a patch were missed, real Razorpay writes stay
# refused because the service flag is absent.
os.environ.pop("RAZORPAY_WRITES_OK", None)
# The case matrix runs at whatever hour the driver is started; a journey that
# passes at noon must not fail at 22:00 because the policy gate deferred every
# contact into the morning. Quiet hours stay ON in the live stack.
os.environ.setdefault("GUARDRAIL_QUIET_DISABLED", "1")

sys.path.insert(0, str(ROOT / "src"))

REG = STATE / "fake_gateway.json"
_seq = itertools.count(1)


def _reg() -> dict:
    try:
        return json.loads(REG.read_text())
    except Exception:
        return {}


def _reg_write(reg: dict) -> None:
    REG.parent.mkdir(parents=True, exist_ok=True)
    REG.write_text(json.dumps(reg, indent=1))


class _FakePaymentAPI:
    def fetch(self, pid):
        rec = _reg().get("captured", {}).get(pid)
        if rec:
            return {"id": pid, "status": "captured", **rec}
        return {"id": pid, "status": "failed", "amount": 0}

    def all(self, params=None):
        return {"items": list(_reg().get("payments", []))}

    def fetch_all(self, params=None):          # some call sites use this name
        return self.all(params)


class _FakeLinkAPI:
    def fetch(self, link_id):
        paid = _reg().get("paid_links", {}).get(link_id)
        if paid:
            return {"id": link_id, "status": "paid", "payments": [{
                "status": "captured",
                "payment_id": paid.get("payment_id", f"pay_fake_{link_id}"),
                "amount": int(round(float(paid.get("amount", 0)) * 100)),
            }]}
        return {"id": link_id, "status": "created", "payments": []}


class _FakeOrderAPI:
    def fetch(self, order_id):
        paid = _reg().get("paid_orders", {}).get(order_id)
        return {"id": order_id, "status": "paid" if paid else "created"}

    def payments(self, order_id):
        paid = _reg().get("paid_orders", {}).get(order_id)
        if not paid:
            return {"items": []}
        return {"items": [{
            "status": "captured",
            "id": paid.get("payment_id", f"pay_fake_{order_id}"),
            "amount": int(round(float(paid.get("amount", 0)) * 100)),
        }]}


class _FakeSDK:
    payment = _FakePaymentAPI()
    payment_link = _FakeLinkAPI()
    order = _FakeOrderAPI()


def install() -> None:
    import recovery_agent.razorpay_client as rc

    def fake_init(self, key_id=None, key_secret=None):
        self.key_id = "rzp_test_FAKE"
        self.key_secret = "fake"
        self.client = _FakeSDK()

    def fake_create_order(self, amount, currency="INR", receipt=None, notes=None):
        oid = f"order_fake{next(_seq)}_{int(time.time())}"
        return {"id": oid, "amount": int(round(float(amount) * 100)),
                "currency": currency}

    def fake_create_payment_link(self, amount, currency="INR", description=None,
                                 customer=None, notes=None, expire_by=None, **kw):
        lid = f"plink_fake{next(_seq)}_{int(time.time())}"
        reg = _reg()
        reg.setdefault("links_created", []).append({
            "link_id": lid, "amount": float(amount),
            "notes": dict(notes or {}), "expire_by": expire_by,
            "at": time.time(),
        })
        _reg_write(reg)
        print(f"[FAKE-GATEWAY] link {lid} for INR {float(amount):,.2f} "
              f"(no quota spent)", flush=True)
        return {"id": lid, "short_url": f"https://fake.rzp.test/{lid}"}

    def fake_fetch_order(self, order_id):
        return _FakeSDK.order.fetch(order_id)

    def fake_fetch_payment(self, pid):
        return _FakeSDK.payment.fetch(pid)

    rc.RazorpayClient.__init__ = fake_init
    rc.RazorpayClient.create_order = fake_create_order
    rc.RazorpayClient.create_payment_link = fake_create_payment_link
    rc.RazorpayClient.fetch_order = fake_fetch_order
    rc.RazorpayClient.fetch_payment = fake_fetch_payment

    # SMTP → local sink; the dispatcher's delivery logic runs unmodified.
    import recovery_agent.notifications as notif
    sink = STATE / "outbox" / "smtp_sink.jsonl"

    class _SinkSMTP:
        def __init__(self, host, port, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context=None):
            pass

        def login(self, user, password):
            pass

        def sendmail(self, frm, to, msg):
            sink.parent.mkdir(parents=True, exist_ok=True)
            with open(sink, "a") as f:
                f.write(json.dumps({"to": to, "at": time.time(),
                                    "bytes": len(msg)}) + "\n")

    notif.smtplib.SMTP = _SinkSMTP
    print("[FAKE-GATEWAY] installed — razorpay + smtp simulated, "
          "SuperU untouched, voice disabled", flush=True)


if __name__ == "__main__":
    STATE.mkdir(exist_ok=True)
    install()
    role = sys.argv[1] if len(sys.argv) > 1 else "frontend"
    if role == "frontend":
        from recovery_agent import frontend
        frontend.main()
    elif role == "daemon":
        from recovery_agent import daemon_worker
        daemon_worker.main()
    else:
        raise SystemExit(f"unknown role {role!r}")
