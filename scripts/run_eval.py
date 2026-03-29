"""
scripts/run_eval.py
===================
CLI entry point for the evaluation framework.

Usage:
    # Run default count from config (EVAL_TEST_COUNT, default 15)
    python scripts/run_eval.py

    # Override test count
    python scripts/run_eval.py --count 8

    # Run only one tier (runs entire tier pool, ignores --count)
    python scripts/run_eval.py --tier simple

    # Run a single test case by ID
    python scripts/run_eval.py --case medium_03

    # List all test cases without running
    python scripts/run_eval.py --list

    # Preview distribution for a count without running
    python scripts/run_eval.py --count 12 --list
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `app` and `evals` can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env before importing anything that reads env vars
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from app.config import EVAL_TEST_COUNT, EVAL_JUDGE_MODEL
from evals.runner import EvalRunner, _RESULTS_DIR
from evals.report import generate_eval_report
from evals.test_cases import (
    select_test_cases,
    get_all_cases,
    get_tier_pool,
    get_case_by_id,
    _distribute,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger(__name__)


def _print_case_table(cases) -> None:
    """Print a formatted table of test cases."""
    print(f"\n  {'ID':<18} {'Tier':<12} {'Agents':>6}  Query")
    print(f"  {'-'*18} {'-'*12} {'-'*6}  {'-'*50}")
    for c in cases:
        query_short = c.query[:50] + ("…" if len(c.query) > 50 else "")
        print(f"  {c.id:<18} {c.tier:<12} {c.expected_sub_agents:>6}  {query_short}")
    print(f"\n  Total: {len(cases)} case(s)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation suite for the Multi-Agent AI Research System.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help=f"Number of test cases to run (default: EVAL_TEST_COUNT={EVAL_TEST_COUNT}). "
             "Auto-distributed across tiers (27/27/27/19 %%).",
    )
    parser.add_argument(
        "--tier",
        choices=["simple", "medium", "complex", "adversarial"],
        default=None,
        help="Run all cases in a specific tier only (overrides --count).",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Run a single test case by ID (e.g. medium_03).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List test cases that would be selected without running them.",
    )
    args = parser.parse_args()

    count = args.count or EVAL_TEST_COUNT

    # ── Resolve which cases to run ────────────────────────────────────────────
    if args.case:
        try:
            cases = [get_case_by_id(args.case)]
        except ValueError as exc:
            print(f"❌ {exc}")
            sys.exit(1)

    elif args.tier:
        cases = get_tier_pool(args.tier)

    else:
        cases = select_test_cases(count)

    # ── --list mode: just show what would run ─────────────────────────────────
    if args.list:
        if args.case:
            print(f"\n  Single case: {args.case}")
        elif args.tier:
            print(f"\n  All cases in tier '{args.tier}' ({len(cases)} cases):")
        else:
            dist = _distribute(count)
            print(f"\n  Distribution for --count {count}:")
            for tier, n in dist.items():
                print(f"    {tier:<14}: {n}")
            print()
        _print_case_table(cases)
        return

    # ── Validate environment ───────────────────────────────────────────────────
    missing = []
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY (search provider)")
    if missing:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing)}")
        print("   Set them in .env before running evals.\n")
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"\n  Judge model : {EVAL_JUDGE_MODEL}")
    print(f"  Cases to run: {len(cases)}")
    _print_case_table(cases)

    runner = EvalRunner()
    summary = runner.run_all(cases)

    # ── Report ────────────────────────────────────────────────────────────────
    run_dir = _RESULTS_DIR / summary.run_id
    generate_eval_report(summary, run_dir=run_dir)

    print(f"\n  Results saved to: {run_dir}")
    print(f"  Run complete. Pass rate: {summary.pass_rate*100:.0f}%  "
          f"Avg score: {summary.avg_weighted_score:.3f}  "
          f"Total cost: ${summary.total_cost:.4f}\n")


if __name__ == "__main__":
    main()
