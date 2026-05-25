from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Regime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


@dataclass(frozen=True)
class MarketData:
    """Synthetic price series with labeled market regimes."""

    prices: np.ndarray
    returns: np.ndarray
    regimes: np.ndarray
    dates: np.ndarray

    @property
    def n_days(self) -> int:
        return len(self.prices)


def _regime_params(regime: Regime) -> tuple[float, float]:
    """Return (daily_drift, daily_volatility) for each regime."""
    if regime == Regime.BULL:
        return 0.0008, 0.012
    if regime == Regime.BEAR:
        return -0.0006, 0.018
    return 0.0, 0.008


def generate_synthetic_prices(
    n_days: int = 756,
    seed: int = 42,
    regime_lengths: tuple[int, ...] | None = None,
) -> MarketData:
    """
    Generate a synthetic price path switching across bull/bear/sideways regimes.

    Default length is ~3 years of trading days. Regime blocks repeat cyclically.
    """
    rng = np.random.default_rng(seed)
    if regime_lengths is None:
        regime_lengths = (180, 120, 150)

    cycle = [Regime.BULL, Regime.BEAR, Regime.SIDEWAYS]
    regimes: list[Regime] = []
    idx = 0
    while len(regimes) < n_days:
        regime = cycle[idx % len(cycle)]
        block = min(regime_lengths[idx % len(regime_lengths)], n_days - len(regimes))
        regimes.extend([regime] * block)
        idx += 1

    regime_arr = np.array([r.value for r in regimes[:n_days]], dtype=object)
    drifts = np.zeros(n_days)
    vols = np.zeros(n_days)
    for regime in Regime:
        mask = regime_arr == regime.value
        drift, vol = _regime_params(regime)
        drifts[mask] = drift
        vols[mask] = vol

    shocks = rng.normal(0.0, 1.0, n_days)
    log_returns = drifts + vols * shocks
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    dates = np.arange(n_days)

    return MarketData(
        prices=prices,
        returns=np.diff(prices) / prices[:-1],
        regimes=regime_arr,
        dates=dates,
    )
