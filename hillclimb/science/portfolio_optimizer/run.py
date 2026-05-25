from __future__ import annotations

from typing import Any

import numpy as np

from hillclimb.core.harness import HillClimber, ClimbResult
from hillclimb.core.types import Candidate
from hillclimb.science.portfolio_optimizer.allocator import AllocationParams
from hillclimb.science.portfolio_optimizer.data import generate_synthetic_returns
from hillclimb.science.portfolio_optimizer.evaluator import PortfolioEvaluator, evaluate_portfolio
from hillclimb.science.portfolio_optimizer.proposer import PortfolioProposer
from hillclimb.science.portfolio_optimizer.regime import detect_regimes, regime_accuracy


def run_portfolio_optimization(
    *,
    n_days: int = 504,
    seed: int = 42,
    max_rounds: int = 15,
    use_detected_regimes: bool = True,
) -> dict[str, Any]:
    """Hill-climb portfolio Sharpe across detected market regimes."""
    data = generate_synthetic_returns(n_days, seed=seed)
    returns = data["returns"]
    true_labels = data["regime_labels"]
    detected = detect_regimes(returns)

    initial = Candidate(state=AllocationParams(), metadata={"source": "default_template"})
    climber = HillClimber(
        proposer=PortfolioProposer(rng=np.random.default_rng(seed)),
        evaluator=PortfolioEvaluator(
            returns,
            true_labels,
            use_detected_regimes=use_detected_regimes,
        ),
        max_rounds=max_rounds,
        early_stop_patience=3,
    )
    result: ClimbResult = climber.climb(initial)

    best_params: AllocationParams = result.best.state
    labels = detected if use_detected_regimes else true_labels
    _, final_diag = evaluate_portfolio(returns, labels, best_params)

    return {
        "best_score": result.best_score,
        "rounds": result.rounds,
        "converged": result.converged,
        "best_params": best_params,
        "diagnostics": final_diag,
        "regime_detection_accuracy": regime_accuracy(detected, true_labels),
        "history_scores": [ev.score for _, ev in result.history],
    }


def main() -> None:
    outcome = run_portfolio_optimization()
    params = outcome["best_params"]
    print("Regime-Aware Portfolio Optimizer")
    print(f"Best net Sharpe: {outcome['best_score']:.4f} ({outcome['rounds']} rounds)")
    print(f"Regime detection accuracy: {outcome['regime_detection_accuracy']:.2%}")
    print(f"Bull weights:   {params.bull_weights}")
    print(f"Bear weights:   {params.bear_weights}")
    print(f"Sideways weights: {params.sideways_weights}")
    print(f"Diagnostics: {outcome['diagnostics']}")


if __name__ == "__main__":
    main()
