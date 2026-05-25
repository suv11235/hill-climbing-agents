"""Demo: hill climb on strategy Sharpe ratio."""

from __future__ import annotations

import random

from hillclimb.core.harness import HillClimber
from hillclimb.core.types import Candidate
from hillclimb.science.finance_research.data import generate_synthetic_prices
from hillclimb.science.finance_research.researcher import (
    FinanceEvaluator,
    FinanceProposer,
    FinanceResearcher,
)


def run(seed: int = 42, max_rounds: int = 15) -> None:
    market = generate_synthetic_prices(seed=seed)
    researcher = FinanceResearcher(market=market, rng=random.Random(seed))
    initial_hypothesis = researcher.propose_initial()

    climber = HillClimber(
        proposer=FinanceProposer(researcher),
        evaluator=FinanceEvaluator(researcher),
        max_rounds=max_rounds,
        early_stop_patience=4,
    )

    result = climber.climb(Candidate(state=initial_hypothesis))
    best = result.best.state

    print("=== Financial Strategy Deep Research Demo ===")
    print(f"Rounds: {result.rounds}  Converged: {result.converged}")
    print(f"Best Sharpe: {result.best_score:.3f}")
    print(f"Best strategy: {best.to_dict()}")

    if result.history:
        init_score = result.history[0][1].score
        print(f"Initial Sharpe: {init_score:.3f}")
        print(f"Improvement: {result.best_score - init_score:+.3f}")

    print("\nResearch log:")
    for i, step in enumerate(researcher.research_log, 1):
        print(
            f"  {i}. {step.hypothesis.kind.value} "
            f"lb={step.hypothesis.lookback} "
            f"Sharpe={step.backtest.sharpe:.3f} "
            f"| {step.reflection[:60]}..."
        )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
