# AutoRecover — AI Revenue Recovery Agent

An autonomous AI agent that detects failed payments, diagnoses the root cause, and executes bounded recovery workflows using Razorpay APIs. Built with LangGraph, Nemotron LLM, and real Razorpay SDK integration.

## How It Works

```
Payment Fails → Agent Detects → Diagnoses Cause → Decides Action → Executes → Observes Result → Loops or Stops
```

The agent runs a continuous loop until one of these conditions is met:
- **Recovered** — payment successfully retried or customer completed checkout
- **Escalated** — case handed off to human support
- **Max attempts** — agent stops after 3 attempts
- **Abandoned** — no viable recovery path

### Diagnosis (3 Layers)

| Layer | Source | Confidence |
|-------|--------|------------|
| Razorpay Error Mapping | Error codes from Razorpay API | 95% |
| LLM Classification | Nemotron via OmniRoute | 85% |
| Rule-Based Fallback | Keyword matching | 70-90% |

### Failure Types

| Type | Strategy |
|------|----------|
| `card_expired` | Notify customer → update payment method → escalate |
| `insufficient_funds` | Wait for salary credit → retry → notify → escalate |
| `bank_declined` | Retry → notify → escalate |
| `network_timeout` | Immediate retry → wait → retry → escalate |
| `risk_block` | Escalate immediately |
| `mandate_revoked` | Notify customer → escalate |

## Setup

### Prerequisites
- Python 3.12+
- Razorpay test account ([get keys here](https://dashboard.razorpay.com/app/keys))

### Install

```bash
git clone https://github.com/yourusername/razorpay-buildathon.git
cd razorpay-buildathon
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` with your Razorpay test keys:

```
RAZORPAY_KEY_ID=rzp_test_xxxxxxxxxxxxx
RAZORPAY_KEY_SECRET=xxxxxxxxxxxxxxxxxxxx
```

## Usage

### Run a Single Recovery Case

```bash
python -m recovery_agent.main single
```

Output:
```
Payment failed: pay_test_001
  Amount: INR 37,036.86
  Reason: Card expiry date is in the past
  Code: card_expired

Case d50cecbe: recovered
  Attempts: 3
  Recovered: True
  Recovered amount: INR 37,036.86

Audit trail:
  [detect] Payment pay_test_001 failed. Opening recovery case.
  [diagnose] Diagnosis: card_expired (confidence: 85%)
  [decide] Attempt #1. Chosen action: send_notification
  [act] Executed: send_notification. Notification sent to customer.
  [observe] After attempt #1: recovered=False. CONTINUE
  [diagnose] Diagnosis: card_expired (confidence: 85%)
  [decide] Attempt #2. Chosen action: update_payment_method
  [act] Executed: update_payment_method. Payment method update link sent.
  [observe] After attempt #2: recovered=False. CONTINUE
  [diagnose] Diagnosis: card_expired (confidence: 85%)
  [decide] Attempt #3. Chosen action: escalate_to_human
  [act] Executed: escalate_to_human. Case escalated to human support.
  [observe] After attempt #3: recovered=False, status=escalated. STOP
```

### Run a Batch Evaluation

```bash
python -m recovery_agent.main batch --cases 30
```

Output:
```
============================================================
BATCH EVALUATION RESULTS
============================================================
Total cases:     30
Recovered:       23 (76.7%)
Escalated:       7
Total amount:    INR 2,190,318.80
Recovered amount: INR 1,796,775.27 (82.0%)
Avg attempts:    2.4
------------------------------------------------------------
BY FAILURE TYPE:
  network_timeout: 4/5 recovered (80%)
  bank_declined: 4/5 recovered (80%)
  risk_block: 3/5 recovered (60%)
  insufficient_funds: 5/5 recovered (100%)
  card_expired: 4/5 recovered (80%)
  mandate_revoked: 3/5 recovered (60%)
============================================================
```

### Start the Dashboard

```bash
python -m recovery_agent.main dashboard
# Open http://localhost:5001
```

Features:
- Recovery metrics and breakdown by failure type
- LangGraph state machine visualization (Mermaid)
- Per-case drill-down with step-by-step audit timeline

### Start the Full-Stack Frontend

```bash
python -m recovery_agent.main frontend
# Customer: http://localhost:5002/pay
# Merchant: http://localhost:5002/merchant
```

The customer page shows real-time agent progress. The merchant page shows live payments and agent trail via WebSocket.

### Start the Webhook Listener

```bash
python -m recovery_agent.main webhook
# Listens on http://localhost:5000/webhook
```

Handles `payment.failed` and `payment.captured` events from Razorpay.

### Start Everything

```bash
./start.sh
```

### Run Tests

```bash
python -m pytest tests/ -v
```

46 tests covering diagnosis, decision logic, stopping rules, execution, data models, and test generation.

## CLI Commands

| Command | Description |
|---------|-------------|
| `single` | Run one case with full audit trail |
| `batch --cases N` | Run batch evaluation (default: 30) |
| `dashboard` | Start recovery dashboard (port 5001) |
| `frontend` | Start customer + merchant UI (port 5002) |
| `webhook` | Start Razorpay webhook listener (port 5000) |
| `retry-schedule --failure-type X` | Show optimal retry windows |
| `communicate --failure-type X --channel Y` | Generate recovery message |

## Architecture

```
src/recovery_agent/
├── agent/
│   ├── __init__.py         # LangGraph agent loop (detect → diagnose → decide → act → observe → stop)
│   ├── diagnosis.py        # 3-layer diagnosis engine
│   ├── decision.py         # Decision matrix (cause × attempt → action)
│   ├── execution.py        # Observable execution with Razorpay SDK
│   ├── stopping.py         # Stopping rules (max attempts, escalation, abandon)
│   ├── evaluation.py       # Batch evaluation system
│   └── test_generator.py   # Synthetic test case generator
├── models/__init__.py      # Pydantic data models
├── logging/__init__.py     # JSONL audit logger
├── razorpay_client.py      # Razorpay SDK wrapper
├── webhook.py              # Razorpay webhook listener
├── dashboard.py            # Flask dashboard
├── frontend.py             # Customer checkout + merchant dashboard
├── communication.py        # Recovery message templates
├── retry_scheduler.py      # Smart retry timing
└── main.py                 # CLI entry point
tests/
└── test_recovery_agent.py  # 46 unit tests
```

## Stopping Rules

| Rule | Condition | Result |
|------|-----------|--------|
| Recovery | `recovered == True` | Stop — success |
| Max attempts | `attempt_count >= 3` | Stop — limit reached |
| Escalation | Last action was `escalate_to_human` | Stop — human takes over |
| Abandon | Last action was `abandon` | Stop — no viable path |

Escalated cases are excluded from recovery totals (`recovered=False`).

## Audit Trail

Every case generates a JSONL audit log in `data/audit_logs/`:

```json
{
  "step": "diagnose",
  "input_data": {"failure_reason": "Card expiry date is in the past"},
  "reasoning": "Diagnosis: card_expired (confidence: 85%)",
  "output_data": {"root_cause": "card_expired", "confidence": 0.85},
  "duration_ms": 1200
}
```

View logs in the dashboard at `/case/{id}`.

## Tech Stack

- **Agent Framework**: LangGraph (state machine, conditional edges, persistence)
- **LLM**: Nemotron via OmniRoute (free tier, local API)
- **Payment API**: Razorpay SDK (test mode)
- **Dashboard**: Flask + Mermaid
- **Frontend**: Flask + Flask-SocketIO (WebSocket)
- **Data Models**: Pydantic (type safety, validation)

## License

Razorpay Buildathon 2026
