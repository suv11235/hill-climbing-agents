from __future__ import annotations

import random

from hillclimb.science.hypothesis_tournament.hypothesis import Hypothesis
from hillclimb.science.hypothesis_tournament.evaluator import composite_score, score_hypothesis


def expected_score(rating_a: float, rating_b: float, k: float = 32.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(winner: Hypothesis, loser: Hypothesis, k: float = 32.0) -> None:
    ea = expected_score(winner.elo, loser.elo, k)
    eb = expected_score(loser.elo, winner.elo, k)
    winner.elo += k * (1.0 - ea)
    loser.elo += k * (0.0 - eb)


def compare_pair(
    a: Hypothesis,
    b: Hypothesis,
    *,
    research_question: str = "",
    evidence_context: str = "",
) -> Hypothesis:
    """Rule-based pairwise judge: higher composite heuristic wins."""
    sa = composite_score(
        score_hypothesis(a, research_question=research_question, evidence_context=evidence_context)
    )
    sb = composite_score(
        score_hypothesis(b, research_question=research_question, evidence_context=evidence_context)
    )
    if sa == sb:
        return a if a.elo >= b.elo else b
    return a if sa > sb else b


def run_pairwise_tournament(
    hypotheses: list[Hypothesis],
    *,
    research_question: str = "",
    evidence_context: str = "",
    rounds: int | None = None,
    k: float = 32.0,
    rng: random.Random | None = None,
) -> list[Hypothesis]:
    """Run Elo-style pairwise tournament; returns hypotheses sorted by Elo."""
    if len(hypotheses) < 2:
        return list(hypotheses)

    rng = rng or random.Random()
    pool = list(hypotheses)
    n_rounds = rounds or max(3, len(pool) * 2)

    for _ in range(n_rounds):
        a, b = rng.sample(pool, 2)
        winner = compare_pair(
            a, b, research_question=research_question, evidence_context=evidence_context
        )
        loser = b if winner is a else a
        update_elo(winner, loser, k=k)

    return sorted(pool, key=lambda h: h.elo, reverse=True)
