from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hillclimb.science.finance_research.data import MarketData
from hillclimb.science.finance_research.hypothesis import StrategyHypothesis
from hillclimb.science.finance_research.strategies import generate_positions


@dataclass(frozen=True)
class BacktestResult:
    sharpe: float
    max_drawdown: float
    turnover: float
    cumulative_return: float
    n_folds: int
    fold_sharpes: tuple[float, ...]

    def to_diagnostics(self) -> dict:
        return {
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "cumulative_return": self.cumulative_return,
            "n_folds": self.n_folds,
            "fold_sharpes": list(self.fold_sharpes),
        }


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))


def _annualized_sharpe(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std < 1e-10:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def _turnover(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.abs(np.diff(positions)).mean())


def walk_forward_backtest(
    hypothesis: StrategyHypothesis,
    market: MarketData,
    periods_per_year: float = 252.0,
) -> BacktestResult:
    """
    Walk-forward backtest: train on rolling window, evaluate on next test window.

    Positions are lagged by one day to avoid look-ahead bias.
    """
    prices = market.prices
    n = len(prices)
    train = hypothesis.train_window
    test = hypothesis.test_window

    fold_sharpes: list[float] = []
    all_strategy_returns: list[float] = []
    all_positions: list[float] = []
    equity = 1.0
    equity_curve: list[float] = [equity]

    start = train
    while start + test <= n:
        window_prices = prices[:start]
        positions = generate_positions(hypothesis, window_prices)
        full_positions = generate_positions(hypothesis, prices[: start + test])
        lagged_positions = full_positions[start - 1 : start + test - 1]

        window_returns = np.diff(prices[start - 1 : start + test]) / prices[
            start - 1 : start + test - 1
        ]
        strategy_returns = lagged_positions * window_returns
        fold_sharpes.append(_annualized_sharpe(strategy_returns, periods_per_year))
        all_strategy_returns.extend(strategy_returns.tolist())
        all_positions.extend(lagged_positions.tolist())

        for r in strategy_returns:
            equity *= 1.0 + r
            equity_curve.append(equity)

        start += test

    if not fold_sharpes:
        positions = generate_positions(hypothesis, prices)
        asset_returns = np.diff(prices) / prices[:-1]
        strategy_returns = positions[:-1] * asset_returns
        fold_sharpes = [_annualized_sharpe(strategy_returns, periods_per_year)]
        all_strategy_returns = strategy_returns.tolist()
        all_positions = positions.tolist()
        equity_curve = np.cumprod(1.0 + strategy_returns).tolist()

    sharpe = _annualized_sharpe(np.array(all_strategy_returns), periods_per_year)
    return BacktestResult(
        sharpe=sharpe,
        max_drawdown=_max_drawdown(np.array(equity_curve)),
        turnover=_turnover(np.array(all_positions)),
        cumulative_return=float(equity_curve[-1] - 1.0) if equity_curve else 0.0,
        n_folds=len(fold_sharpes),
        fold_sharpes=tuple(fold_sharpes),
    )
