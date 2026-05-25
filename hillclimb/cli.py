"""Unified CLI for hill climbing prototypes."""

from __future__ import annotations

import argparse
import subprocess
import sys

PROTOTYPES = {
    # Software
    "rl_interface": "hillclimb.software.rl_interface.run",
    "ocr_self_iterate": "hillclimb.software.ocr_self_iterate.run",
    "sift_coding": "hillclimb.software.sift_coding.run",
    "config_discovery": "hillclimb.software.config_discovery.run",
    # Science
    "finance_research": "hillclimb.science.finance_research.run",
    "lean_prover": "hillclimb.science.lean_prover.run",
    "hypothesis_tournament": "hillclimb.science.hypothesis_tournament.run",
    "portfolio_optimizer": "hillclimb.science.portfolio_optimizer.run",
}

DEMO_ARGS = {
    "rl_interface": ["--compare", "--hard", "--rounds", "8"],
    "ocr_self_iterate": ["--rounds", "8", "--docs", "3"],
    "sift_coding": ["--task", "reverse_string"],
    "config_discovery": ["--seed", "42"],
    "finance_research": [],
    "lean_prover": [],
    "hypothesis_tournament": [],
    "portfolio_optimizer": [],
}


def run_prototype(name: str, extra_args: list[str] | None = None) -> int:
    if name not in PROTOTYPES:
        print(f"Unknown prototype: {name}")
        print(f"Available: {', '.join(PROTOTYPES)}")
        return 1
    module = PROTOTYPES[name]
    args = [sys.executable, "-m", module] + (extra_args or [])
    return subprocess.call(args)


def run_all_demos() -> int:
    failed = []
    for name, demo_args in DEMO_ARGS.items():
        print(f"\n{'='*60}")
        print(f"  Demo: {name}")
        print(f"{'='*60}\n")
        rc = run_prototype(name, demo_args)
        if rc != 0:
            failed.append(name)
    if failed:
        print(f"\nFailed demos: {', '.join(failed)}")
        return 1
    print("\nAll demos completed successfully.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hill climbing AI agent experimentation harness"
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run a prototype demo")
    run_p.add_argument("prototype", choices=list(PROTOTYPES.keys()))
    run_p.add_argument("extra", nargs=argparse.REMAINDER, help="Args passed to prototype")

    sub.add_parser("demo", help="Run all prototype demos").add_argument(
        "--all", action="store_true", default=True
    )

    sub.add_parser("list", help="List available prototypes")

    args = parser.parse_args(argv)

    if args.command == "run":
        extra = args.extra
        if extra and extra[0] == "--":
            extra = extra[1:]
        return run_prototype(args.prototype, extra)

    if args.command == "demo":
        return run_all_demos()

    if args.command == "list":
        print("Software Development:")
        for name in ["rl_interface", "ocr_self_iterate", "sift_coding", "config_discovery"]:
            print(f"  {name:25s} → python -m {PROTOTYPES[name]}")
        print("\nScience Discovery:")
        for name in ["finance_research", "lean_prover", "hypothesis_tournament", "portfolio_optimizer"]:
            print(f"  {name:25s} → python -m {PROTOTYPES[name]}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
