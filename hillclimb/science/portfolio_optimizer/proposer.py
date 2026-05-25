from __future__ import annotations

import numpy as np

from hillclimb.core.types import Candidate, Evaluation
from hillclimb.science.portfolio_optimizer.allocator import AllocationParams


class PortfolioProposer:
    """Adjusts allocation params based on regime Sharpe diagnostics."""

    def __init__(self, step: float = 0.05, rng: np.random.Generator | None = None) -> None:
        self.step = step
        self.rng = rng or np.random.default_rng()

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        params: AllocationParams = current.state
        diagnostics = history[-1][1].diagnostics if history else {}
        regime_sharpes = diagnostics.get("regime_sharpes", {})

        vec = params.to_vector().copy()
        bull = vec[0:3]
        bear = vec[3:6]
        sideways = vec[6:9]

        # Shift toward defensive assets when bear-regime Sharpe is weak
        bear_sharpe = regime_sharpes.get(1, regime_sharpes.get("1", 0.0))
        bull_sharpe = regime_sharpes.get(0, regime_sharpes.get("0", 0.0))

        if bear_sharpe < bull_sharpe:
            bear[0] -= self.step
            bear[1] += self.step
        else:
            bull[0] += self.step
            bull[1] -= self.step * 0.5

        if diagnostics.get("turnover", 0.0) > 0.15:
            vec[9] = min(0.20, vec[9] + self.step)
        else:
            vec[9] = max(0.01, vec[9] - self.step * 0.5)

        # Small exploratory noise on sideways template
        sideways += self.rng.normal(0, self.step * 0.3, size=3)

        vec[0:3] = bull
        vec[3:6] = bear
        vec[6:9] = sideways

        new_params = AllocationParams.from_vector(vec)
        return Candidate(
            state=new_params,
            metadata={"source": "regime_aware_proposer"},
        )
