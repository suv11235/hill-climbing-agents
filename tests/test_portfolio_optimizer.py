from __future__ import annotations

import numpy as np

from hillclimb.science.portfolio_optimizer.allocator import (
    AllocationParams,
    simulate_allocation,
    weights_for_regime,
)
from hillclimb.science.portfolio_optimizer.data import generate_synthetic_returns
from hillclimb.science.portfolio_optimizer.evaluator import (
    evaluate_portfolio,
    sharpe_ratio,
    turnover_penalty,
)
from hillclimb.science.portfolio_optimizer.proposer import PortfolioProposer
from hillclimb.science.portfolio_optimizer.regime import detect_regimes, regime_accuracy
from hillclimb.science.portfolio_optimizer.run import run_portfolio_optimization
from hillclimb.core.types import Candidate, Evaluation


def test_synthetic_data_shape_and_regimes():
    data = generate_synthetic_returns(n_days=300, seed=1)
    assert data["returns"].shape == (300, 3)
    assert len(data["regime_labels"]) == 300
    assert set(np.unique(data["regime_labels"])) <= {0, 1, 2}


def test_regime_detector_outputs_valid_labels():
    data = generate_synthetic_returns(n_days=252, seed=2)
    detected = detect_regimes(data["returns"])
    assert detected.shape == (252,)
    assert set(np.unique(detected[~np.isnan(detected.astype(float))])) <= {0, 1, 2}


def test_allocator_weights_sum_to_one():
    params = AllocationParams()
    for regime in (0, 1, 2):
        w = weights_for_regime(regime, params)
        assert np.isclose(w.sum(), 1.0)
        assert np.all(w >= 0)


def test_simulate_allocation_returns_portfolio_series():
    data = generate_synthetic_returns(n_days=100, seed=3)
    params = AllocationParams()
    port, weights = simulate_allocation(data["returns"], data["regime_labels"], params)
    assert port.shape == (100,)
    assert weights.shape == (100, 3)


def test_sharpe_and_turnover_metrics():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, size=200)
    assert sharpe_ratio(returns) != 0.0
    weights = np.tile(np.array([0.5, 0.3, 0.2]), (50, 1))
    weights[25:] = np.array([0.2, 0.6, 0.2])
    assert turnover_penalty(weights) > 0.0


def test_evaluate_portfolio_returns_diagnostics():
    data = generate_synthetic_returns(n_days=252, seed=4)
    params = AllocationParams()
    score, diag = evaluate_portfolio(data["returns"], data["regime_labels"], params)
    assert "raw_sharpe" in diag
    assert "net_sharpe" in diag
    assert "regime_sharpes" in diag
    assert isinstance(score, float)


def test_proposer_mutates_allocation_params():
    proposer = PortfolioProposer(step=0.05, rng=np.random.default_rng(0))
    current = Candidate(state=AllocationParams())
    history = [
        (
            current,
            Evaluation(
                score=0.5,
                diagnostics={
                    "regime_sharpes": {0: 1.0, 1: -0.5, 2: 0.2},
                    "turnover": 0.2,
                },
            ),
        )
    ]
    proposal = proposer.propose(current, history)
    assert isinstance(proposal.state, AllocationParams)
    assert proposal.state.bull_weights != current.state.bull_weights or (
        proposal.state.bear_weights != current.state.bear_weights
    )


def test_run_portfolio_optimization_improves_or_completes():
    outcome = run_portfolio_optimization(n_days=252, max_rounds=8, seed=5)
    assert outcome["rounds"] >= 1
    assert isinstance(outcome["best_score"], float)
    assert isinstance(outcome["best_params"], AllocationParams)
    assert 0.0 <= outcome["regime_detection_accuracy"] <= 1.0
    assert len(outcome["history_scores"]) >= 1
