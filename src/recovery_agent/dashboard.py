"""Revenue recovery dashboard — visualize agent decision flow.

Flask-based dashboard with:
- LangGraph state machine diagram
- Per-case drill-down with step-by-step timeline
- Recovery metrics

Usage:
    python -m recovery_agent.dashboard
    # then open http://localhost:5001
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, render_template_string, request

app = Flask(__name__)

AUDIT_DIR = Path(os.getenv("AUDIT_DIR", "data/audit_logs"))

STEP_COLORS = {
    "detect": "#3b82f6",
    "diagnose": "#8b5cf6",
    "decide": "#f59e0b",
    "act": "#10b981",
    "observe": "#6366f1",
    "stop": "#ef4444",
}


def load_cases() -> list[dict]:
    """Load all cases from audit logs."""
    cases = []
    if not AUDIT_DIR.exists():
        return cases
    for log_file in AUDIT_DIR.glob("*.jsonl"):
        case_data = _parse_audit_log(log_file)
        if case_data:
            cases.append(case_data)
    return cases


def _parse_audit_log(log_file: Path) -> dict | None:
    """Parse a single audit log file into structured case data."""
    entries = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        return None

    case_id = log_file.stem.replace("case_", "")
    amount = 0.0
    root_cause = "unknown"
    recovered = False
    recovered_amount = 0.0
    path = []
    steps_detail = []

    for entry in entries:
        step = entry.get("step", "")
        reasoning = entry.get("reasoning", "")
        duration = entry.get("duration_ms", 0)

        path.append(step)

        steps_detail.append({
            "step": step,
            "reasoning": reasoning,
            "duration_ms": duration,
            "input_data": entry.get("input_data", {}),
            "output_data": entry.get("output_data", {}),
        })

        if step == "detect":
            payment = entry.get("output_data", {}).get("payment", {})
            amount = payment.get("amount", 0.0)
        elif step == "diagnose":
            root_cause = entry.get("output_data", {}).get("root_cause", "unknown")
        elif step == "observe":
            recovered = entry.get("output_data", {}).get("recovered", False)
            if recovered:
                recovered_amount = amount

    return {
        "id": case_id,
        "case_id": case_id,
        "amount": amount,
        "root_cause": root_cause,
        "recovered": recovered,
        "recovered_amount": recovered_amount,
        "attempts": len([s for s in path if s == "act"]),
        "status": "recovered" if recovered else "failed",
        "path": path,
        "steps": steps_detail,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/legacy")
def legacy_dashboard():
    cases = load_cases()
    total = len(cases)
    recovered = sum(1 for c in cases if c["recovered"])
    rate = (recovered / total * 100) if total > 0 else 0
    total_recovered = sum(c["recovered_amount"] for c in cases)

    by_type = defaultdict(lambda: {"total": 0, "recovered": 0})
    for c in cases:
        by_type[c["root_cause"]]["total"] += 1
        if c["recovered"]:
            by_type[c["root_cause"]]["recovered"] += 1
    for stats in by_type.values():
        stats["rate"] = round((stats["recovered"] / stats["total"] * 100) if stats["total"] > 0 else 0, 1)

    return render_template_string(MAIN_TEMPLATE, total_cases=total, recovered_cases=recovered,
        recovery_rate=f"{rate:.1f}", total_recovered=total_recovered,
        by_type=dict(by_type), recent_cases=cases[-20:])


@app.route("/case/<case_id>")
def case_detail(case_id):
    cases = load_cases()
    case = next((c for c in cases if c["id"] == case_id), None)
    if not case:
        return "Case not found", 404
    return render_template_string(CASE_TEMPLATE, case=case, step_colors=STEP_COLORS)


@app.route("/graph")
def graph_view():
    return render_template_string(GRAPH_TEMPLATE)


@app.route("/api/metrics")
def api_metrics():
    cases = load_cases()
    total = len(cases)
    recovered = sum(1 for c in cases if c.get("recovered", False))
    rate = (recovered / total * 100) if total > 0 else 0.0
    total_recovered = sum(c.get("recovered_amount", 0.0) for c in cases)

    # Dynamic policy violation count: guardrail intercepted = blocked or modified action
    policy_violations = 0
    for c in cases:
        for step in c.get("steps", []):
            if step.get("step") != "act":
                continue
            # Check input_data for guardrail interception signals
            input_data = step.get("input_data", {})
            if isinstance(input_data, str):
                continue
            guardrail_final = input_data.get("guardrail_final_action", "")
            guardrail_checks = input_data.get("guardrail_checks", 0)
            # Violation: guardrail modified/blocked the action (final differs from intended)
            if guardrail_final and guardrail_final in ("wait_and_retry", "escalate_to_human"):
                if guardrail_checks and guardrail_checks > 0:
                    policy_violations += 1
                    break  # count once per case

    policy_compliance_rate = 1.0 - (policy_violations / total) if total > 0 else 1.0

    # Live Memory Store Stats
    from recovery_agent.agent.memory import CustomerMemoryStore
    mem_store = CustomerMemoryStore()
    mem_stats = mem_store.get_stats()

    # Live Knowledge Graph Stats
    from recovery_agent.agent.kg_router import RazorpayKnowledgeGraph
    kg = RazorpayKnowledgeGraph()

    return jsonify({
        "total_cases": total,
        "recovered_cases": recovered,
        "recovery_rate": round(rate, 1),
        "total_recovered": round(total_recovered, 2),
        "memory_customers_tracked": mem_stats.get("total_customers", 0),
        "kg_rails_count": len(kg.graph.nodes()),
        "kg_available_rails": len(kg.graph.edges()),
        "policy_violations": policy_violations,
        "policy_compliance_rate": round(policy_compliance_rate, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/cases")
def api_cases():
    return jsonify({"cases": load_cases()})


@app.route("/api/case/<case_id>")
def api_case(case_id):
    cases = load_cases()
    case = next((c for c in cases if c["id"] == case_id), None)
    if not case:
        return jsonify({"error": "not found"}), 404
    return jsonify(case)


def main():
    port = int(os.getenv("DASHBOARD_PORT", "5001"))
    print(f"\n  ⚡ AutoRecover Enterprise Arena & Multi-Agent HUD")
    print(f"  http://localhost:{port}")
    print(f"  Legacy Dashboard: http://localhost:{port}/legacy\n")
    app.run(host="0.0.0.0", port=port)


MAIN_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Revenue Recovery Agent</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0}
.nav{background:#1e293b;padding:12px 24px;display:flex;gap:24px;align-items:center;border-bottom:1px solid #334155}
.nav a{color:#94a3b8;text-decoration:none;font-size:14px;padding:6px 12px;border-radius:6px}
.nav a:hover,.nav a.active{background:#334155;color:#e2e8f0}
.nav .logo{font-weight:700;font-size:16px;color:#3b82f6;margin-right:16px}
.container{max-width:1400px;margin:0 auto;padding:24px}
.card{background:#1e293b;border-radius:12px;padding:24px;margin-bottom:20px;border:1px solid #334155}
.card h2{font-size:14px;color:#64748b;margin-bottom:16px;text-transform:uppercase;letter-spacing:1px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px}
.metric{background:#1e293b;border-radius:12px;padding:24px;text-align:center;border:1px solid #334155}
.metric-value{font-size:2.5em;font-weight:700;color:#3b82f6}
.sv{color:#10b981}.wv{color:#f59e0b}
.metric-label{color:#64748b;font-size:13px;margin-top:4px}
.graph-wrap{background:#fff;border-radius:8px;padding:24px;overflow-x:auto}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:12px 16px;border-bottom:1px solid #334155}
th{color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
tr:hover{background:#0f172a}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600}
.bs{background:#065f46;color:#10b981}.bf{background:#7f1d1d;color:#ef4444}.ba{background:#1e3a5f;color:#3b82f6}
.bar{height:6px;background:#334155;border-radius:3px;overflow:hidden;width:100px}
.bar-fill{height:100%;background:#10b981;border-radius:3px}
a.cl{color:#3b82f6;cursor:pointer;text-decoration:none;font-size:13px}
a.cl:hover{text-decoration:underline}
.dp{display:flex;gap:4px;align-items:center;flex-wrap:wrap}
.dn{background:#0f172a;border:1px solid #334155;border-radius:4px;padding:2px 8px;font-size:11px;color:#94a3b8}
.da{color:#475569;font-size:10px}
</style>
</head>
<body>
<nav class="nav">
<span class="logo">Revenue Recovery Agent</span>
<a href="/" class="active">Dashboard</a>
<a href="/graph">Agent Flow</a>
<a href="/api/metrics" target="_blank">API</a>
</nav>
<div class="container">
<div class="metrics">
<div class="metric"><div class="metric-value">{{total_cases}}</div><div class="metric-label">Total Cases</div></div>
<div class="metric"><div class="metric-value sv">{{recovered_cases}}</div><div class="metric-label">Recovered</div></div>
<div class="metric"><div class="metric-value wv">{{recovery_rate}}%</div><div class="metric-label">Recovery Rate</div></div>
<div class="metric"><div class="metric-value">INR {{ "{:,.0f}".format(total_recovered) }}</div><div class="metric-label">Revenue Recovered</div></div>
</div>
<div class="grid">
<div class="card">
<h2>Agent State Machine</h2>
<div class="graph-wrap"><pre class="mermaid">
graph TD
    S([Start]):::first
    D1["1. DETECT<br/>Confirm failure"]:::step
    D2["2. DIAGNOSE<br/>LLM + Rules"]:::step
    D3["3. DECIDE<br/>Select action"]:::step
    D4["4. ACT<br/>Execute"]:::step
    D5["5. OBSERVE<br/>Check outcome"]:::step
    E([END]):::last
    S-->D1-->D2-->D3-->D4-->D5
    D5-. continue .->D2
    D5-. stop .->E
    classDef step fill:#1e3a5f,stroke:#3b82f6,stroke-width:2px,color:#e2e8f0
    classDef first fill:#065f46,stroke:#10b981,color:#e2e8f0
    classDef last fill:#7f1d1d,stroke:#ef4444,color:#e2e8f0
</pre></div>
<p style="color:#64748b;font-size:12px;margin-top:8px">Loop: recovered | max_attempts(3) | escalated | abandoned</p>
</div>
<div class="card">
<h2>Recovery by Failure Type</h2>
<table>
<tr><th>Type</th><th>Cases</th><th>Recovered</th><th>Rate</th><th></th></tr>
{% for type, stats in by_type.items() %}
<tr>
<td><code style="color:#94a3b8">{{type}}</code></td>
<td>{{stats.total}}</td>
<td>{{stats.recovered}}</td>
<td>{{stats.rate}}%</td>
<td><div class="bar"><div class="bar-fill" style="width:{{stats.rate}}%"></div></div></td>
</tr>
{% endfor %}
</table>
</div>
</div>
<div class="card">
<h2>Cases — Click to View Decision Flow</h2>
<table>
<tr><th>Case</th><th>Amount</th><th>Root Cause</th><th>Path</th><th>Attempts</th><th>Status</th><th></th></tr>
{% for c in recent_cases %}
<tr>
<td><code style="color:#94a3b8">{{c.id[:12]}}</code></td>
<td>INR {{ "{:,.0f}".format(c.amount) }}</td>
<td><span class="ba badge">{{c.root_cause}}</span></td>
<td><div class="dp">{% for s in c.path %}<span class="dn">{{s}}</span>{% if not loop.last %}<span class="da">&rarr;</span>{% endif %}{% endfor %}</div></td>
<td>{{c.attempts}}</td>
<td>{% if c.recovered %}<span class="bs badge">RECOVERED</span>{% else %}<span class="bf badge">FAILED</span>{% endif %}</td>
<td><a class="cl" href="/case/{{c.id}}">View Flow &rarr;</a></td>
</tr>
{% endfor %}
</table>
</div>
</div>
<script>mermaid.initialize({startOnLoad:true,theme:'default',flowchart:{curve:'linear'}})</script>
</body></html>"""


CASE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Case {{case.id[:12]}} — Decision Flow</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0}
.nav{background:#1e293b;padding:12px 24px;display:flex;gap:24px;align-items:center;border-bottom:1px solid #334155}
.nav a{color:#94a3b8;text-decoration:none;font-size:14px;padding:6px 12px;border-radius:6px}
.nav a:hover{background:#334155;color:#e2e8f0}
.nav .logo{font-weight:700;font-size:16px;color:#3b82f6;margin-right:16px}
.container{max-width:900px;margin:0 auto;padding:24px}
.card{background:#1e293b;border-radius:12px;padding:24px;margin-bottom:20px;border:1px solid #334155}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600}
.bs{background:#065f46;color:#10b981}.bf{background:#7f1d1d;color:#ef4444}.ba{background:#1e3a5f;color:#3b82f6}
.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}
.stat{background:#0f172a;border-radius:8px;padding:16px;text-align:center;border:1px solid #334155}
.stat-v{font-size:1.5em;font-weight:700;color:#3b82f6}
.stat-l{color:#64748b;font-size:11px;margin-top:2px}
.timeline{position:relative;padding:20px 0 20px 0}
.tl-line{position:absolute;left:19px;top:0;bottom:0;width:2px;background:#334155}
.tl-step{position:relative;padding-left:56px;margin-bottom:20px}
.tl-dot{position:absolute;left:8px;top:8px;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#fff;z-index:1}
.tl-content{background:#0f172a;border-radius:8px;padding:16px;border:1px solid #334155;border-left:3px solid #334155}
.tl-content.success{border-left-color:#10b981}
.tl-content.failed{border-left-color:#ef4444}
.tl-content.active{border-left-color:#3b82f6}
.tl-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.tl-name{font-weight:600;font-size:14px;text-transform:uppercase}
.tl-time{color:#475569;font-size:11px}
.tl-body{color:#94a3b8;font-size:13px;line-height:1.7}
.tl-body code{background:#334155;padding:1px 6px;border-radius:3px;font-size:12px;color:#e2e8f0}
.tl-body .label{color:#64748b;font-size:11px;text-transform:uppercase;display:block;margin-top:8px}
.decision-path{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:16px 0}
.dn{background:#0f172a;border:1px solid #334155;border-radius:4px;padding:4px 10px;font-size:12px}
.da{color:#475569;font-size:11px}
.loop-indicator{background:#7f1d1d;color:#fca5a5;border:1px solid #991b1b;border-radius:8px;padding:12px 16px;margin:16px 0;font-size:13px}
</style>
</head>
<body>
<nav class="nav">
<span class="logo">Revenue Recovery Agent</span>
<a href="/">Dashboard</a>
<a href="/graph">Agent Flow</a>
</nav>
<div class="container">
<div class="card">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
<div>
<h1 style="font-size:20px;margin-bottom:4px">Case <code style="color:#3b82f6">{{case.id[:12]}}</code></h1>
<p style="color:#64748b;font-size:13px">{{case.timestamp}}</p>
</div>
{% if case.recovered %}<span class="bs badge" style="font-size:14px;padding:6px 16px">RECOVERED</span>
{% else %}<span class="bf badge" style="font-size:14px;padding:6px 16px">FAILED</span>{% endif %}
</div>
<div class="summary">
<div class="stat"><div class="stat-v">INR {{ "{:,.0f}".format(case.amount) }}</div><div class="stat-l">Amount</div></div>
<div class="stat"><div class="stat-v" style="color:#8b5cf6">{{case.root_cause}}</div><div class="stat-l">Root Cause</div></div>
<div class="stat"><div class="stat-v">{{case.attempts}}</div><div class="stat-l">Attempts</div></div>
<div class="stat"><div class="stat-v" style="color:#10b981">INR {{ "{:,.0f}".format(case.recovered_amount) }}</div><div class="stat-l">Recovered</div></div>
</div>
<div style="margin-bottom:8px"><span style="color:#64748b;font-size:12px;text-transform:uppercase">Decision Path:</span></div>
<div class="decision-path">
{% for s in case.path %}
<span class="dn" style="border-color:{{step_colors.get(s,'#334155')}};color:{{step_colors.get(s,'#94a3b8')}}">{{s}}</span>
{% if not loop.last %}<span class="da">&rarr;</span>{% endif %}
{% endfor %}
</div>
</div>

<h2 style="font-size:14px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:16px">Step-by-Step Flow</h2>
<div class="card" style="padding:8px 24px 24px 24px">
<div class="timeline">
<div class="tl-line"></div>
{% for step in case.steps %}
<div class="tl-step">
<div class="tl-dot" style="background:{{step_colors.get(step.step,'#334155')}}">{{loop.index}}</div>
<div class="tl-content {% if step.step == 'act' %}{% if 'success' in step.reasoning.lower() or 'created' in step.reasoning.lower() %}success{% elif 'failed' in step.reasoning.lower() %}failed{% endif %}{% elif step.step == 'stop' %}{% if 'Recovered: True' in step.reasoning %}success{% else %}failed{% endif %}{% endif %}">
<div class="tl-hdr">
<span class="tl-name" style="color:{{step_colors.get(step.step,'#e2e8f0')}}">{{step.step}}</span>
<span class="tl-time">{% if step.duration_ms %}{{step.duration_ms}}ms{% endif %}</span>
</div>
<div class="tl-body">{{step.reasoning}}</div>
{% if step.input and step.input|length > 0 %}
<span class="label">Input</span>
<div class="tl-body"><code>{{step.input}}</code></div>
{% endif %}
{% if step.output and step.output|length > 0 %}
<span class="label">Output</span>
<div class="tl-body"><code>{{step.output}}</code></div>
{% endif %}
</div>
</div>
{% endfor %}
</div>
</div>

{% if case.path|length > 3 %}
<div class="loop-indicator">
<strong>Agent looped {{ (case.path|length - 3) }} time(s)</strong> — re-diagnosed and tried different actions before stopping.
</div>
{% endif %}

</div>
</body></html>"""


GRAPH_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Agent Flow — LangGraph State Machine</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0}
.nav{background:#1e293b;padding:12px 24px;display:flex;gap:24px;align-items:center;border-bottom:1px solid #334155}
.nav a{color:#94a3b8;text-decoration:none;font-size:14px;padding:6px 12px;border-radius:6px}
.nav a:hover,.nav a.active{background:#334155;color:#e2e8f0}
.nav .logo{font-weight:700;font-size:16px;color:#3b82f6;margin-right:16px}
.container{max-width:1200px;margin:0 auto;padding:24px}
.card{background:#1e293b;border-radius:12px;padding:24px;margin-bottom:20px;border:1px solid #334155}
.card h2{font-size:14px;color:#64748b;margin-bottom:16px;text-transform:uppercase;letter-spacing:1px}
.graph-wrap{background:#fff;border-radius:8px;padding:32px;overflow-x:auto;text-align:center}
.legend{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}
.legend-item{display:flex;align-items:center;gap:8px;font-size:13px;color:#94a3b8}
.legend-dot{width:12px;height:12px;border-radius:50%}
.info{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
.info-card{background:#0f172a;border-radius:8px;padding:16px;border:1px solid #334155}
.info-card h3{font-size:13px;color:#3b82f6;margin-bottom:8px}
.info-card p{font-size:13px;color:#94a3b8;line-height:1.6}
.info-card code{background:#334155;padding:1px 6px;border-radius:3px;font-size:12px;color:#e2e8f0}
</style>
</head>
<body>
<nav class="nav">
<span class="logo">Revenue Recovery Agent</span>
<a href="/">Dashboard</a>
<a href="/graph" class="active">Agent Flow</a>
</nav>
<div class="container">
<div class="card">
<h2>Agent State Machine — Built with LangGraph</h2>
<div class="graph-wrap"><pre class="mermaid">
graph TD
    START(([START])):::entry
    DETECT["<b>1. DETECT</b><br/>----------------<br/>Confirm payment failure<br/>Extract: amount, reason, code<br/>Open recovery case"]:::s1
    DIAGNOSE["<b>2. DIAGNOSE</b><br/>----------------<br/>Razorpay error mapping<br/>LLM classification (Nemotron)<br/>Rule-based fallback<br/>Output: root_cause + confidence"]:::s2
    DECIDE["<b>3. DECIDE</b><br/>----------------<br/>Decision tree:<br/>cause x attempt_count<br/>Map to action type"]:::s3
    ACT["<b>4. ACT</b><br/>----------------<br/>Execute intervention<br/>Razorpay API (real)<br/>or simulation<br/>Record attempt"]:::s4
    OBSERVE["<b>5. OBSERVE</b><br/>----------------<br/>Check outcome<br/>Apply stopping rules<br/>Should we continue?"]:::s5
    STOP(([END])):::exit

    START --> DETECT
    DETECT --> DIAGNOSE
    DIAGNOSE --> DECIDE
    DECIDE --> ACT
    ACT --> OBSERVE
    OBSERVE -. "CONTINUE<br/>(attempt < 3 & not recovered)" .-> DIAGNOSE
    OBSERVE -. "STOP<br/>(recovered / max / escalated)" .-> STOP

    classDef entry fill:#065f46,stroke:#10b981,stroke-width:3px,color:#fff
    classDef exit fill:#7f1d1d,stroke:#ef4444,stroke-width:3px,color:#fff
    classDef s1 fill:#1e3a5f,stroke:#3b82f6,stroke-width:2px,color:#e2e8f0
    classDef s2 fill:#2d1f5e,stroke:#8b5cf6,stroke-width:2px,color:#e2e8f0
    classDef s3 fill:#4a3000,stroke:#f59e0b,stroke-width:2px,color:#e2e8f0
    classDef s4 fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#e2e8f0
    classDef s5 fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#e2e8f0
</pre></div>
<div class="legend">
<div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>1. DETECT — Identify failure</div>
<div class="legend-item"><div class="legend-dot" style="background:#8b5cf6"></div>2. DIAGNOSE — Classify root cause</div>
<div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>3. DECIDE — Select intervention</div>
<div class="legend-item"><div class="legend-dot" style="background:#10b981"></div>4. ACT — Execute action</div>
<div class="legend-item"><div class="legend-dot" style="background:#6366f1"></div>5. OBSERVE — Check outcome</div>
<div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>STOP — Loop exit conditions</div>
</div>
</div>
<div class="info">
<div class="info-card">
<h3>Diagnosis (3 layers)</h3>
<p>1. <code>Razorpay error mapping</code> — real API error codes (confidence: 95%)<br/>
2. <code>LLM classification</code> — Nemotron via OmniRoute (confidence: 85%)<br/>
3. <code>Rule-based fallback</code> — keyword matching (confidence: 70-90%)</p>
</div>
<div class="info-card">
<h3>Decision Matrix</h3>
<p><code>cause x attempt_count -> action</code><br/>
Attempt 1: notification / wait_and_retry<br/>
Attempt 2: update_payment_method / retry<br/>
Attempt 3: escalate_to_human</p>
</div>
<div class="info-card">
<h3>Stopping Rules</h3>
<p><code>recovered -> STOP</code><br/>
<code>attempt_count >= 3 -> STOP</code><br/>
<code>escalated -> STOP</code><br/>
<code>abandoned -> STOP</code></p>
</div>
<div class="info-card">
<h3>Execution</h3>
<p><code>Razorpay API</code> — real order creation for retries<br/>
<code>Simulation</code> — notifications, escalations<br/>
Every step logged to JSONL audit trail</p>
</div>
</div>
</div>
<script>mermaid.initialize({startOnLoad:true,theme:'default',flowchart:{curve:'linear',padding:20}})</script>
</body></html>"""


if __name__ == "__main__":
    main()
