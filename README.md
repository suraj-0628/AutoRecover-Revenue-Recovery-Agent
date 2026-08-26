# AutoRecover — Razorpay AI Revenue Recovery Agent & OpenCode Operations HUD

An autonomous AI agent that detects failed payments, diagnoses root causes via pure LLM diagnostic reflection, and executes bounded recovery workflows using official Razorpay APIs. Built for the Razorpay Buildathon with LangGraph, Nemotron LLM, NVIDIA NAT Guardrails, and an OpenCode-inspired Developer Operations HUD.

---

## 🎯 How It Works

```
Payment Fails → Webhook Sensing → LLM Diagnostic Reflection → Guardrail Check → Knowledge Graph Routing → Razorpay SDK Tool Execution → Generative UI Morphing
```

The agent runs a continuous loop until one of these conditions is met:
- **Recovered** — Payment retried, payment link completed, or card expiry updated & captured via Razorpay API.
- **Escalated** — Handed off to human support when attempts or risk thresholds are reached.
- **Max Attempts** — Agent stops after 3 bounded attempts.
- **Abandoned** — Stopped when customer opts out or no viable recovery path exists.

---

## 🔬 Official Razorpay Knowledge Base & Failure Normalizer

The system embeds an official Razorpay API Error Knowledge Base (`src/recovery_agent/razorpay_knowledge_base.py`) built directly from Razorpay's official API Error documentation:

- **Error Codes Catalog**: `BAD_REQUEST_PAYMENT_TEMPORARY_TECHNICAL_ISSUE`, `BAD_REQUEST_CARD_EXPIRED`, `BAD_REQUEST_PAYMENT_INSUFFICIENT_FUNDS`, `BAD_REQUEST_PAYMENT_DECLINED_BY_BANK`, `BAD_REQUEST_MANDATE_INACTIVE`, `BAD_REQUEST_CHECKOUT_ABANDONED`, `BAD_REQUEST_RISK_CHECK_FAILED`.
- **Error Taxonomy**:
  - `source`: `customer`, `business`, `gateway`, `razorpay`.
  - `step`: `payment_initiation`, `payment_authentication`, `payment_authorization`, `payment_capture`.
- **Payload Normalization**: Automatically normalizes customer-facing UI messages (e.g. *"Your payment could not be completed due to a temporary technical issue. To complete the payment, use another payment instrument."*) into full Razorpay API Failure Payloads.

---

## 🖥️ OpenCode Developer IDE Interface & Operations HUD

The Merchant Dashboard ([http://localhost:5002/merchant](http://localhost:5002/merchant)) is designed as an **OpenCode-inspired technical IDE**:

- **Monospace Typography**: `JetBrains Mono` and `Fira Code`.
- **Dark IDE Palette**: Crisp `#0d1117` background with `#30363d` 1px borders and clean status trace badges (`[DETECT]`, `[DIAGNOSE]`, `[GUARDRAIL]`, `[ACTION]`, `[SUCCESS]`).
- **Fixed Monologue Viewing Window**: Strictly bounded `380px` height (`min-height: 380px`, `max-height: 380px`) with auto-scrolling terminal logs.
- **Tool Execution Code Cards**: Displays real Razorpay SDK API JSON response objects (`RazorpaySDK.Order.create`, `RazorpaySDK.Payment.capture`).
- **Claude-Style Dual View Switcher**:
  - `[ Canvas View ]`: Dynamic Generative UI morphing cards (WhatsApp link, Card Expiry form, Hinglish Voice AI Call, 5-step progress pipeline).
  - `[ Store Checkout (/pay) ]`: Embedded live browser iframe pointing to `http://localhost:5002/pay` for testing real checkout payment flows inside the split-screen HUD.

---

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.12+
- Razorpay test account ([Razorpay API Keys](https://dashboard.razorpay.com/app/keys))

### Install & Configure

```bash
git clone https://github.com/suraj-0628/Razorpay-AI-Revenue-Recovery-Agent.git
cd Razorpay-AI-Revenue-Recovery-Agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

### Start Services

```bash
./start.sh
```

Serves:
- **Merchant Operations HUD**: [http://localhost:5002/merchant](http://localhost:5002/merchant)
- **Customer Store Checkout**: [http://localhost:5002/pay](http://localhost:5002/pay)
- **Recovery Analytics Dashboard**: [http://localhost:5001](http://localhost:5001)
- **LangGraph Agent Flow Graph**: [http://localhost:5001/graph](http://localhost:5001/graph)
- **Razorpay Webhook Listener**: [http://localhost:5000/webhook](http://localhost:5000/webhook)

---

## 🧪 Verification & Test Suite

```bash
.venv/bin/pytest tests/ -v
```

- **Unit Test Suite**: `209 PASSED` (100% pass rate).
- **Adversarial Chaos Gym**: `31 PASSED` in `10.50s`.

---

## 📁 Architecture Breakdown

```
src/recovery_agent/
├── agent/
│   ├── diagnosis.py             # LLM diagnostic reflection chain
│   ├── decision.py              # LLM Strategy Planner & Guardrail intercept
│   ├── execution.py             # Razorpay SDK action execution
│   ├── guardrails.py            # NVIDIA NAT Guardrails
│   ├── kg_router.py             # Razorpay Knowledge Graph router
│   ├── memory.py                # Customer long-term memory store & payday radar
│   └── llm_client.py            # Shared LLM client (Nemotron / Claude / Gemini)
├── razorpay_knowledge_base.py   # Official Razorpay API Error Catalog & Normalizer
├── razorpay_client.py           # Razorpay SDK API wrapper
├── frontend.py                  # Merchant Operations HUD & Customer Checkout
├── webhook.py                   # Razorpay Webhook Listener
├── dashboard.py                 # Recovery Analytics Dashboard
└── templates/
    └── index.html               # OpenCode-inspired Developer Operations HUD
```

---

## 📄 License

Razorpay Buildathon 2026 — AI Revenue Recovery Track
