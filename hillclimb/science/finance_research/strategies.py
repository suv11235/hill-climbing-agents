from __future__ import annotations

from typing import Callable

import numpy as np

from hillclimb.science.finance_research.hypothesis import StrategyHypothesis, StrategyKind


def _clip_position(position: np.ndarray, max_leverage: float) -> np.ndarray:
    return np.clip(position, -max_leverage, max_leverage)


def momentum_signals(prices: np.ndarray, lookback: int, max_leverage: float) -> np.ndarray:
    """Long when past lookback return is positive, short when negative."""
    n = len(prices)
    positions = np.zeros(n)
    if lookback >= n:
        return positions

    past_return = prices[lookback:] / prices[:-lookback] - 1.0
    positions[lookback:] = np.sign(past_return)
    return _clip_position(positions, max_leverage)


def mean_reversion_signals(
    prices: np.ndarray, lookback: int, max_leverage: float
) -> np.ndarray:
    """Fade z-score deviations from a rolling mean."""
    n = len(prices)
    positions = np.zeros(n)
    if lookback >= n:
        return positions

    window = np.lib.stride_tricks.sliding_window_view(prices, lookback)
    means = window.mean(axis=1)
    stds = window.std(axis=1)
    stds[stds < 1e-8] = 1e-8
    z = (prices[lookback - 1 :] - means) / stds
    positions[lookback - 1 :] = -np.tanh(z)
    return _clip_position(positions, max_leverage)


def vol_targeting_signals(
    returns: np.ndarray, lookback: int, target_vol: float, max_leverage: float
) -> np.ndarray:
    """Scale exposure inversely with realized volatility."""
    n = len(returns)
    positions = np.zeros(n + 1)
    if lookback >= n:
        return positions

    window = np.lib.stride_tricks.sliding_window_view(returns, lookback)
    realized = window.std(axis=1)
    realized[realized < 1e-8] = 1e-8
    scale = target_vol / realized
    positions[lookback:] = np.clip(scale, 0.0, max_leverage)
    return positions


STRATEGY_FNS: dict[StrategyKind, Callable[..., np.ndarray]] = {
    StrategyKind.MOMENTUM: momentum_signals,
    StrategyKind.MEAN_REVERSION: mean_reversion_signals,
    StrategyKind.VOL_TARGETING: vol_targeting_signals,
}


def generate_positions(hypothesis: StrategyHypothesis, prices: np.ndarray) -> np.ndarray:
    """Build position series for a strategy hypothesis."""
    returns = np.diff(prices) / prices[:-1]
    fn = STRATEGY_FNS[hypothesis.kind]

    if hypothesis.kind == StrategyKind.VOL_TARGETING:
        positions = fn(
            returns,
            hypothesis.lookback,
            hypothesis.target_vol,
            hypothesis.max_leverage,
        )
    else:
        positions = fn(prices, hypothesis.lookback, hypothesis.max_leverage)

    return positions[: len(prices)]
