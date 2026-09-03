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


def main():
    parser = argparse.ArgumentParser(description="Revenue Recovery Agent")
    parser.add_argument(
        "command",
        choices=["single", "batch", "webhook", "dashboard", "frontend", "retry-schedule", "communicate"],
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


if __name__ == "__main__":
    main()
