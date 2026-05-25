"""Financial strategy deep-research prototype (Co-Scientist / TiMi inspired)."""

from hillclimb.science.finance_research.backtest import BacktestResult, walk_forward_backtest
from hillclimb.science.finance_research.data import MarketData, generate_synthetic_prices
from hillclimb.science.finance_research.hypothesis import StrategyHypothesis, random_hypothesis
from hillclimb.science.finance_research.researcher import FinanceResearcher

__all__ = [
    "BacktestResult",
    "FinanceResearcher",
    "MarketData",
    "StrategyHypothesis",
    "generate_synthetic_prices",
    "random_hypothesis",
    "walk_forward_backtest",
]
