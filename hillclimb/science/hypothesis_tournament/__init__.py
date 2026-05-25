"""Co-Scientist-inspired hypothesis tournament evolution."""

from hillclimb.science.hypothesis_tournament.hypothesis import Hypothesis

__all__ = ["Hypothesis", "run_tournament_evolution"]


def run_tournament_evolution(*args, **kwargs):
    from hillclimb.science.hypothesis_tournament.run import run_tournament_evolution as _run

    return _run(*args, **kwargs)
