"""Greedy hill climb over RandomForest hyperparameters."""

from __future__ import annotations

import random

from hillclimb.core.harness import HillClimber
from hillclimb.core.types import AcceptPolicy, Candidate
from hillclimb.software.config_discovery.evaluator import ConfigEvaluator
from hillclimb.software.config_discovery.proposer import ConfigProposer
from hillclimb.software.config_discovery.search_space import random_config


def run_climb(
    *,
    seed: int = 42,
    max_rounds: int = 25,
    early_stop_patience: int = 5,
) -> dict:
    rng = random.Random(seed)
    initial_config = random_config(rng)
    initial = Candidate(state=initial_config, metadata={"source": "random_baseline"})

    climber = HillClimber(
        proposer=ConfigProposer(),
        evaluator=ConfigEvaluator(),
        accept_policy=AcceptPolicy.GREEDY,
        max_rounds=max_rounds,
        early_stop_patience=early_stop_patience,
    )
    result = climber.climb(initial)

    initial_acc = result.history[0][1].diagnostics.get("accuracy_mean", 0.0)
    best_acc = result.best_score

    return {
        "rounds": result.rounds,
        "converged": result.converged,
        "initial_accuracy": initial_acc,
        "best_accuracy": best_acc,
        "improvement": best_acc - initial_acc,
        "best_config": result.best.state,
        "history_len": len(result.history),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Config discovery hill climb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=25)
    args = parser.parse_args()

    summary = run_climb(seed=args.seed, max_rounds=args.max_rounds)
    print("RandomForest on breast cancer (5-fold CV accuracy)")
    print(f"Initial accuracy: {summary['initial_accuracy']:.4f}")
    print(f"Best accuracy:    {summary['best_accuracy']:.4f}")
    print(f"Improvement:      {summary['improvement']:+.4f}")
    print(f"Rounds: {summary['rounds']}  converged={summary['converged']}")
    print("Best config:")
    for key, value in summary["best_config"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
