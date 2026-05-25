from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AllocationParams:
    """Portfolio weight templates per regime (equity, bond, commodity)."""

    bull_weights: tuple[float, float, float] = (0.70, 0.20, 0.10)
    bear_weights: tuple[float, float, float] = (0.20, 0.60, 0.20)
    sideways_weights: tuple[float, float, float] = (0.40, 0.40, 0.20)
    rebalance_threshold: float = 0.05

    def as_dict(self) -> dict[str, tuple[float, float, float]]:
        return {
            "bull_low_vol": self.bull_weights,
            "bear_high_vol": self.bear_weights,
            "sideways": self.sideways_weights,
        }

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                *self.bull_weights,
                *self.bear_weights,
                *self.sideways_weights,
                self.rebalance_threshold,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> AllocationParams:
        vec = np.clip(vec, 0.0, 1.0)
        bull = _normalize(vec[0:3])
        bear = _normalize(vec[3:6])
        sideways = _normalize(vec[6:9])
        threshold = float(np.clip(vec[9] if len(vec) > 9 else 0.05, 0.01, 0.20))
        return cls(
            bull_weights=tuple(bull),
            bear_weights=tuple(bear),
            sideways_weights=tuple(sideways),
            rebalance_threshold=threshold,
        )


def _normalize(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    if total <= 0:
        return np.array([1 / 3, 1 / 3, 1 / 3])
    return weights / total


def weights_for_regime(regime: int, params: AllocationParams) -> np.ndarray:
    templates = [params.bull_weights, params.bear_weights, params.sideways_weights]
    idx = int(np.clip(regime, 0, 2))
    return np.array(templates[idx], dtype=float)


def simulate_allocation(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    params: AllocationParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backtest regime-switching allocation.

    Returns (portfolio_returns, weight_history) where weight_history is (T, 3).
    """
    n_days, n_assets = returns.shape
    weights = np.zeros((n_days, n_assets))
    port_returns = np.zeros(n_days)
    current_w = weights_for_regime(int(regime_labels[0]), params)

    for t in range(n_days):
        target = weights_for_regime(int(regime_labels[t]), params)
        if t == 0 or np.max(np.abs(target - current_w)) > params.rebalance_threshold:
            current_w = target
        weights[t] = current_w
        port_returns[t] = float(np.dot(current_w, returns[t]))

    return port_returns, weights
