from __future__ import annotations

import numpy as np

from hillclimb.science.portfolio_optimizer.data import REGIME_NAMES


def rolling_volatility(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Annualized rolling volatility of portfolio or single-asset returns."""
    if returns.ndim == 1:
        series = returns
    else:
        series = returns.mean(axis=1)
    vol = np.full(len(series), np.nan)
    for t in range(window - 1, len(series)):
        vol[t] = np.std(series[t - window + 1 : t + 1], ddof=1) * np.sqrt(252)
    return vol


def rolling_trend(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Cumulative return over rolling window (annualized proxy)."""
    if returns.ndim == 1:
        series = returns
    else:
        series = returns[:, 0]  # equity as trend signal
    trend = np.full(len(series), np.nan)
    for t in range(window - 1, len(series)):
        trend[t] = np.sum(series[t - window + 1 : t + 1]) * (252 / window)
    return trend


def detect_regimes(
    returns: np.ndarray,
    *,
    vol_window: int = 21,
    trend_window: int = 21,
    vol_threshold: float = 0.18,
    trend_threshold: float = 0.02,
) -> np.ndarray:
    """
    Classify each day into bull_low_vol (0), bear_high_vol (1), or sideways (2).

    Uses rolling volatility and equity trend heuristics.
    """
    vol = rolling_volatility(returns, vol_window)
    trend = rolling_trend(returns, trend_window)
    n = len(returns)
    detected = np.full(n, 2, dtype=int)  # default sideways

    for t in range(n):
        if np.isnan(vol[t]) or np.isnan(trend[t]):
            continue
        if vol[t] >= vol_threshold and trend[t] < -trend_threshold:
            detected[t] = 1  # bear_high_vol
        elif vol[t] < vol_threshold and trend[t] > trend_threshold:
            detected[t] = 0  # bull_low_vol
        else:
            detected[t] = 2  # sideways

    return detected


def regime_accuracy(detected: np.ndarray, true_labels: np.ndarray) -> float:
    mask = ~np.isnan(detected.astype(float))
    if not np.any(mask):
        return 0.0
    return float(np.mean(detected[mask] == true_labels[mask]))


def regime_name(regime_id: int) -> str:
    return REGIME_NAMES[regime_id] if 0 <= regime_id < len(REGIME_NAMES) else "unknown"
