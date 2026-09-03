"""Main entry point — run the recovery agent."""
from __future__ import annotations

import argparse

from recovery_agent.agent import RecoveryAgent
from recovery_agent.agent.evaluation import run_batch_evaluation
from recovery_agent.models import Case


def run_single(payment_id: str = "pay_test_001", use_harness: bool = False) -> None:
    """Run a single case for demo."""
    from tests.test_generator import generate_payment_event, FAILURE_SCENARIOS

    # Pick a specific scenario for demo
    scenario = FAILURE_SCENARIOS[0]  # Card expired
    event = generate_payment_event(scenario=scenario, payment_id=payment_id)

    print(f"Payment failed: {event.payment_id}")
    print(f"  Amount: {event.currency} {event.amount}")
    print(f"  Reason: {event.failure_reason}")
    print(f"  Code: {event.failure_code}")
    print(f"  Mode: {'AgentHarness' if use_harness else 'LangGraph'}")
    print()

    agent = RecoveryAgent(use_harness=use_harness)
    case = Case(payment=event)
    final_case = agent.run(case)

    print(f"Case {final_case.id}: {final_case.status.value}")
    print(f"  Attempts: {final_case.attempt_count}")
    print(f"  Recovered: {final_case.recovered}")
    print(f"  Recovered amount: INR {final_case.recovered_amount:,.2f}")
    print()
    print("Audit trail:")
    for entry in final_case.audit_log:
        print(f"  [{entry.step.value}] {entry.reasoning}")


def run_batch(num_cases: int = 30, seed: int = 42) -> None:
    """Run batch evaluation."""
    result = run_batch_evaluation(num_cases=num_cases, seed=seed)
    print(result.summary())


def run_webhook():
    """Start the Razorpay webhook listener."""
    from recovery_agent.webhook import main
    main()


def run_dashboard():
    """Start the revenue recovery dashboard."""
    from recovery_agent.dashboard import main
    main()


def run_retry_schedule(failure_type: str = "network_timeout", attempt: int = 1):
    """Show retry schedule for a failure type."""
    from recovery_agent.retry_scheduler import format_retry_schedule
    from recovery_agent.models import FailureType

    ft = FailureType(failure_type)
    print(format_retry_schedule(ft, attempt))


def run_communicate(failure_type: str = "card_expired", channel: str = "email"):
    """Generate recovery message."""
    from recovery_agent.communication import generate_recovery_message
    from recovery_agent.models import FailureType

    ft = FailureType(failure_type)
    msg = generate_recovery_message(
        failure_type=ft,
        channel=channel,
        customer_name="Rahul Kumar",
        amount=2999.0,
        card_last4="4242",
    )

    if msg:
        print(f"Channel: {msg.channel}")
        if msg.subject:
            print(f"Subject: {msg.subject}")
        print(f"Tone: {msg.tone}")
        if msg.cta:
            print(f"CTA: {msg.cta}")
        print()
        print(msg.body)
    else:
        print(f"No {channel} message for {failure_type}")


def run_frontend():
    """Start the full-stack frontend (customer + merchant)."""
    from recovery_agent.frontend import main
    main()


def run_chaos_gym(episodes: int = 10, seed: int | None = None, use_harness: bool = False):
    """Run the Adversarial Chaos Gym evaluation with trajectory benchmarking."""
    from recovery_agent.eval.chaos_gym import run_chaos_gym as _run_chaos_gym
    from recovery_agent.eval.trajectory_benchmark import TrajectoryBenchmark

    result = _run_chaos_gym(episodes=episodes, seed=seed, use_harness=use_harness)

    # Run trajectory benchmarking on all episodes
    benchmark = TrajectoryBenchmark()
    all_metrics = []
    for ep in result["episodes_data"]:
        metrics = benchmark.evaluate_trajectory(ep.get("benchmark_trajectory", ep["trajectory"]))
        all_metrics.append(metrics)
    agg = benchmark.aggregate_metrics(all_metrics)

    print("=" * 60)
    print("ADVERSARIAL CHAOS GYM RESULTS")
    print("=" * 60)
    print(f"Episodes:            {result['episodes']}")
    print(f"Recovered:           {result['recovered']}/{result['episodes']} ({result['recovery_rate']:.1%})")
    print(f"Total Amount:        INR {result['total_amount']:,.2f}")
    print(f"Recovered Amount:    INR {result['total_recovered_amount']:,.2f} ({result['recovery_amount_rate']:.1%})")
    print(f"Avg Reward:          {result['avg_reward']}")
    print(f"Total Reward:        {result['total_reward']}")
    print(f"Avg Steps:           {result['avg_steps']}")
    print(f"Friction Index:      {result['avg_friction_index']}")
    print(f"Policy Violations:   {result['policy_violations']}")
    print("-" * 60)
    print("BY PERSONA:")
    for persona, stats in result["by_persona"].items():
        print(f"  {persona}: {stats['recovered']}/{stats['total']} recovered ({stats['recovery_rate']:.0%})")
    print("-" * 60)
    print("TRAJECTORY BENCHMARK:")
    print(f"  Avg Step Efficiency:    {agg['avg_step_efficiency']:.3f}")
    print(f"  Avg Friction Score:     {agg['avg_friction_score']:.3f}")
    print(f"  Avg Compliance Rate:    {agg['avg_policy_compliance']:.3f}")
    print(f"  Avg Trajectory Score:   {agg['avg_trajectory_score']:.3f}")
    if agg.get("total_invasive_steps", 0) > 0:
        print(f"  Total Invasive Steps:   {agg['total_invasive_steps']}")
    print("=" * 60)


def run_phoenix_eval(payment_id: str | None = None) -> None:
    """Run Phoenix agent evaluations."""
    from recovery_agent.eval.phoenix_evals import PhoenixEvaluator

    evaluator = PhoenixEvaluator()
    report = evaluator.run_evaluation(payment_id=payment_id)

    print("=" * 60)
    print("PHOENIX AGENT EVALUATION REPORT")
    print("=" * 60)
    print(f"Total evaluations: {report['total_evaluations']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    print(f"Pass rate: {report['pass_rate']}")
    print(f"Annotations written to Phoenix: {report['annotations_written']}")
    print()
    print("By Evaluator:")
    for name, stats in report["by_evaluator"].items():
        total = stats["total"]
        passed = stats["pass"]
        pct = (passed / total * 100) if total else 0
        print(f"  {name}: {passed}/{total} passed ({pct:.0f}%)")
    if report["failed_evaluations"]:
        print()
        print("Failures:")
        for f in report["failed_evaluations"]:
            print(f"  [{f['evaluator']}] {f['payment_id']}: {f['explanation']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Revenue Recovery Agent")
    parser.add_argument(
        "command",
        choices=["single", "batch", "webhook", "dashboard", "frontend", "retry-schedule", "communicate", "chaos-gym", "phoenix-eval"],
        help="Command to run",
    )
    parser.add_argument("--cases", type=int, default=30, help="Number of cases for batch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--payment-id", type=str, default="pay_test_001", help="Payment ID for single run")
    parser.add_argument("--failure-type", type=str, default="card_expired", help="Failure type for retry-schedule/communicate")
    parser.add_argument("--channel", type=str, default="email", help="Channel for communicate command")
    parser.add_argument("--attempt", type=int, default=1, help="Attempt number for retry-schedule")
    parser.add_argument("--harness", action="store_true", help="Use TrueForge AgentHarness (multi-turn tool-calling loop)")

    args = parser.parse_args()

    if args.command == "single":
        run_single(args.payment_id, use_harness=args.harness)
    elif args.command == "batch":
        run_batch(args.cases, args.seed)
    elif args.command == "webhook":
        run_webhook()
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "frontend":
        run_frontend()
    elif args.command == "retry-schedule":
        run_retry_schedule(args.failure_type, args.attempt)
    elif args.command == "communicate":
        run_communicate(args.failure_type, args.channel)
    elif args.command == "chaos-gym":
        run_chaos_gym(args.cases, args.seed, use_harness=args.harness)
    elif args.command == "phoenix-eval":
        run_phoenix_eval(args.payment_id if args.payment_id != "pay_test_001" else None)


if __name__ == "__main__":
    main()
