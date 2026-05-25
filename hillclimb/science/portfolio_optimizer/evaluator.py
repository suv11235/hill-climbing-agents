from __future__ import annotations

import numpy as np

from hillclimb.core.types import Candidate, Evaluation
from hillclimb.science.portfolio_optimizer.allocator import (
    AllocationParams,
    simulate_allocation,
)
from hillclimb.science.portfolio_optimizer.regime import detect_regimes


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / 252.0
    std = np.std(excess, ddof=1)
    if std <= 1e-12:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def turnover_penalty(weight_history: np.ndarray) -> float:
    if len(weight_history) < 2:
        return 0.0
    changes = np.abs(np.diff(weight_history, axis=0)).sum(axis=1)
    return float(np.mean(changes))


def evaluate_portfolio(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    params: AllocationParams,
    *,
    turnover_cost: float = 0.001,
) -> tuple[float, dict]:
    port_returns, weights = simulate_allocation(returns, regime_labels, params)
    raw_sharpe = sharpe_ratio(port_returns)
    turnover = turnover_penalty(weights)
    net_sharpe = raw_sharpe - turnover_cost * turnover * 252

    regime_sharpes = {}
    for regime_id in (0, 1, 2):
        mask = regime_labels == regime_id
        if np.sum(mask) > 5:
            regime_sharpes[regime_id] = sharpe_ratio(port_returns[mask])
        else:
            regime_sharpes[regime_id] = 0.0

    diagnostics = {
        "raw_sharpe": round(raw_sharpe, 4),
        "turnover": round(turnover, 4),
        "net_sharpe": round(net_sharpe, 4),
        "regime_sharpes": {int(k): round(v, 4) for k, v in regime_sharpes.items()},
        "mean_return": round(float(np.mean(port_returns)) * 252, 4),
        "volatility": round(float(np.std(port_returns, ddof=1)) * np.sqrt(252), 4),
    }
    return net_sharpe, diagnostics


class PortfolioEvaluator:
    """Scores allocation params using Sharpe minus turnover penalty."""

    def __init__(
        self,
        returns: np.ndarray,
        true_regime_labels: np.ndarray | None = None,
        *,
        use_detected_regimes: bool = True,
        turnover_cost: float = 0.001,
    ) -> None:
        self.returns = returns
        self.true_regime_labels = true_regime_labels
        self.use_detected_regimes = use_detected_regimes
        self.turnover_cost = turnover_cost

    def _regime_labels(self) -> np.ndarray:
        if self.use_detected_regimes:
            return detect_regimes(self.returns)
        if self.true_regime_labels is not None:
            return self.true_regime_labels
        return detect_regimes(self.returns)

    def evaluate(self, candidate: Candidate) -> Evaluation:
        params: AllocationParams = candidate.state
        labels = self._regime_labels()
        score, diagnostics = evaluate_portfolio(
            self.returns,
            labels,
            params,
            turnover_cost=self.turnover_cost,
        )
        return Evaluation(score=score, diagnostics=diagnostics)
