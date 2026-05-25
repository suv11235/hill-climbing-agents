"""Regime-aware agentic portfolio optimization."""

__all__ = ["run_portfolio_optimization"]


def run_portfolio_optimization(*args, **kwargs):
    from hillclimb.science.portfolio_optimizer.run import run_portfolio_optimization as _run

    return _run(*args, **kwargs)
