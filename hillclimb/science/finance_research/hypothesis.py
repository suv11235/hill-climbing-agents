from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StrategyKind(str, Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOL_TARGETING = "vol_targeting"


@dataclass
class StrategyHypothesis:
    """Parameterized trading strategy candidate."""

    kind: StrategyKind
    lookback: int = 20
    target_vol: float = 0.10
    max_leverage: float = 1.0
    train_window: int = 252
    test_window: int = 63
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "lookback": self.lookback,
            "target_vol": self.target_vol,
            "max_leverage": self.max_leverage,
            "train_window": self.train_window,
            "test_window": self.test_window,
            **self.metadata,
        }


def random_hypothesis(rng: random.Random | None = None) -> StrategyHypothesis:
    """Sample a random strategy hypothesis."""
    rng = rng or random.Random()
    kind = rng.choice(list(StrategyKind))
    lookback = rng.randint(5, 60)
    target_vol = rng.uniform(0.05, 0.20)
    max_leverage = rng.uniform(0.5, 2.0)
    train_window = rng.choice([126, 189, 252])
    test_window = rng.choice([21, 42, 63])
    return StrategyHypothesis(
        kind=kind,
        lookback=lookback,
        target_vol=target_vol,
        max_leverage=max_leverage,
        train_window=train_window,
        test_window=test_window,
    )


def mutate_hypothesis(
    hypothesis: StrategyHypothesis,
    diagnostics: dict[str, Any],
    rng: random.Random | None = None,
) -> StrategyHypothesis:
    """Refine a hypothesis using backtest diagnostics."""
    rng = rng or random.Random()
    new = StrategyHypothesis(
        kind=hypothesis.kind,
        lookback=hypothesis.lookback,
        target_vol=hypothesis.target_vol,
        max_leverage=hypothesis.max_leverage,
        train_window=hypothesis.train_window,
        test_window=hypothesis.test_window,
        metadata=dict(hypothesis.metadata),
    )

    sharpe = diagnostics.get("sharpe", 0.0)
    turnover = diagnostics.get("turnover", 0.0)
    max_dd = diagnostics.get("max_drawdown", 0.0)

    if sharpe < 0:
        new.kind = rng.choice(list(StrategyKind))

    if turnover > 2.0:
        new.lookback = min(120, new.lookback + rng.randint(3, 10))
    elif sharpe > 0.5:
        new.lookback = max(5, new.lookback + rng.randint(-3, 3))

    if max_dd < -0.25:
        new.max_leverage = max(0.25, new.max_leverage * 0.85)
        new.target_vol = max(0.03, new.target_vol * 0.9)

    if new.kind == StrategyKind.VOL_TARGETING:
        new.target_vol = float(np_clip(new.target_vol + rng.uniform(-0.02, 0.02), 0.03, 0.25))

    new.metadata["parent_sharpe"] = sharpe
    new.metadata["refinement"] = diagnostics.get("reflection", "parameter_tweak")
    return new


def np_clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
