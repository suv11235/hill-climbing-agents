from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from hillclimb.core.types import Candidate, Evaluation
from hillclimb.science.finance_research.backtest import BacktestResult, walk_forward_backtest
from hillclimb.science.finance_research.data import MarketData
from hillclimb.science.finance_research.hypothesis import (
    StrategyHypothesis,
    mutate_hypothesis,
    random_hypothesis,
)


@dataclass
class ResearchStep:
    hypothesis: StrategyHypothesis
    backtest: BacktestResult
    reflection: str


def reflect_on_diagnostics(result: BacktestResult) -> str:
    """Generate a human-readable reflection from backtest metrics."""
    notes: list[str] = []
    if result.sharpe < 0:
        notes.append("Negative Sharpe suggests wrong strategy regime fit.")
    elif result.sharpe > 1.0:
        notes.append("Strong risk-adjusted returns; refine cautiously.")
    else:
        notes.append("Modest edge; parameter tuning may help.")

    if result.max_drawdown < -0.20:
        notes.append("Drawdown exceeds 20%; reduce leverage or vol target.")
    if result.turnover > 2.0:
        notes.append("High turnover; lengthen lookback to cut costs.")
    if result.n_folds > 1:
        spread = max(result.fold_sharpes) - min(result.fold_sharpes)
        if spread > 1.5:
            notes.append("Fold instability detected; strategy may be overfit.")

    return " ".join(notes)


@dataclass
class FinanceResearcher:
    """
    Multi-step deep research loop:
    propose hypothesis → backtest → reflect → refine.
    """

    market: MarketData
    rng: random.Random = field(default_factory=random.Random)
    research_log: list[ResearchStep] = field(default_factory=list)

    def propose_initial(self) -> StrategyHypothesis:
        return random_hypothesis(self.rng)

    def evaluate_hypothesis(self, hypothesis: StrategyHypothesis) -> tuple[BacktestResult, str]:
        result = walk_forward_backtest(hypothesis, self.market)
        reflection = reflect_on_diagnostics(result)
        self.research_log.append(
            ResearchStep(hypothesis=hypothesis, backtest=result, reflection=reflection)
        )
        return result, reflection

    def refine(
        self, hypothesis: StrategyHypothesis, diagnostics: dict[str, Any]
    ) -> StrategyHypothesis:
        enriched = dict(diagnostics)
        enriched["reflection"] = reflect_on_diagnostics_from_dict(diagnostics)
        return mutate_hypothesis(hypothesis, enriched, self.rng)

    def deep_research_round(self, hypothesis: StrategyHypothesis | None = None) -> ResearchStep:
        current = hypothesis or self.propose_initial()
        result, reflection = self.evaluate_hypothesis(current)
        return ResearchStep(hypothesis=current, backtest=result, reflection=reflection)


def reflect_on_diagnostics_from_dict(diagnostics: dict[str, Any]) -> str:
    return reflect_on_diagnostics(
        BacktestResult(
            sharpe=diagnostics.get("sharpe", 0.0),
            max_drawdown=diagnostics.get("max_drawdown", 0.0),
            turnover=diagnostics.get("turnover", 0.0),
            cumulative_return=diagnostics.get("cumulative_return", 0.0),
            n_folds=diagnostics.get("n_folds", 0),
            fold_sharpes=tuple(diagnostics.get("fold_sharpes", [])),
        )
    )


class FinanceProposer:
    """Hill-climb proposer that refines strategy hypotheses from diagnostics."""

    def __init__(self, researcher: FinanceResearcher) -> None:
        self.researcher = researcher

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        diagnostics = history[-1][1].diagnostics if history else {}
        hypothesis: StrategyHypothesis = current.state
        refined = self.researcher.refine(hypothesis, diagnostics)
        return Candidate(
            state=refined,
            metadata={"source": "finance_researcher", "parent": hypothesis.to_dict()},
        )


class FinanceEvaluator:
    """Scores strategy hypotheses by walk-forward Sharpe ratio."""

    def __init__(self, researcher: FinanceResearcher) -> None:
        self.researcher = researcher

    def evaluate(self, candidate: Candidate) -> Evaluation:
        hypothesis: StrategyHypothesis = candidate.state
        result, reflection = self.researcher.evaluate_hypothesis(hypothesis)
        diagnostics = result.to_diagnostics()
        diagnostics["reflection"] = reflection
        diagnostics["hypothesis"] = hypothesis.to_dict()
        return Evaluation(
            score=result.sharpe,
            diagnostics=diagnostics,
            passed=result.sharpe > -1.0,
        )
