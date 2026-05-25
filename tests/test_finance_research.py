"""Tests for financial strategy deep-research prototype."""

from __future__ import annotations

import random

import numpy as np

from hillclimb.core.harness import HillClimber
from hillclimb.core.types import Candidate
from hillclimb.science.finance_research.backtest import walk_forward_backtest
from hillclimb.science.finance_research.data import Regime, generate_synthetic_prices
from hillclimb.science.finance_research.hypothesis import (
    StrategyHypothesis,
    StrategyKind,
    mutate_hypothesis,
    random_hypothesis,
)
from hillclimb.science.finance_research.researcher import (
    FinanceEvaluator,
    FinanceProposer,
    FinanceResearcher,
    reflect_on_diagnostics,
)
from hillclimb.science.finance_research.strategies import generate_positions


def test_synthetic_prices_have_regimes():
    market = generate_synthetic_prices(n_days=400, seed=1)
    assert market.n_days == 400
    assert len(market.prices) == 400
    assert len(market.returns) == 399
    unique = set(market.regimes.tolist())
    assert Regime.BULL.value in unique
    assert Regime.BEAR.value in unique
    assert Regime.SIDEWAYS.value in unique


def test_strategy_positions_bounded():
    market = generate_synthetic_prices(seed=2)
    hyp = StrategyHypothesis(
        kind=StrategyKind.MOMENTUM, lookback=10, max_leverage=1.5
    )
    positions = generate_positions(hyp, market.prices)
    assert len(positions) == len(market.prices)
    assert np.all(np.abs(positions) <= 1.5 + 1e-9)


def test_walk_forward_backtest_metrics():
    market = generate_synthetic_prices(seed=3)
    hyp = StrategyHypothesis(kind=StrategyKind.MEAN_REVERSION, lookback=20)
    result = walk_forward_backtest(hyp, market)
    assert isinstance(result.sharpe, float)
    assert result.max_drawdown <= 0.0
    assert result.turnover >= 0.0
    assert result.n_folds >= 1


def test_hypothesis_mutation_responds_to_diagnostics():
    hyp = StrategyHypothesis(
        kind=StrategyKind.MOMENTUM, lookback=10, max_leverage=2.0
    )
    refined = mutate_hypothesis(
        hyp,
        {"sharpe": -0.5, "turnover": 3.0, "max_drawdown": -0.30},
        rng=random.Random(0),
    )
    assert refined.lookback >= hyp.lookback
    assert refined.max_leverage <= hyp.max_leverage


def test_researcher_reflect_on_diagnostics():
    market = generate_synthetic_prices(seed=4)
    researcher = FinanceResearcher(market=market, rng=random.Random(0))
    step = researcher.deep_research_round()
    assert step.hypothesis.kind in StrategyKind
    assert isinstance(step.reflection, str)
    assert len(step.reflection) > 0


def test_finance_hill_climb_improves_or_runs():
    market = generate_synthetic_prices(seed=5)
    researcher = FinanceResearcher(market=market, rng=random.Random(5))
    initial = researcher.propose_initial()

    climber = HillClimber(
        proposer=FinanceProposer(researcher),
        evaluator=FinanceEvaluator(researcher),
        max_rounds=8,
        early_stop_patience=3,
    )
    result = climber.climb(Candidate(state=initial))
    assert result.rounds >= 1
    assert len(result.history) >= 2
    assert isinstance(result.best_score, float)


def test_all_strategy_kinds_produce_positions():
    market = generate_synthetic_prices(seed=6)
    for kind in StrategyKind:
        hyp = StrategyHypothesis(kind=kind, lookback=15, target_vol=0.1)
        positions = generate_positions(hyp, market.prices)
        assert len(positions) == len(market.prices)
