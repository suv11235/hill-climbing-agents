from __future__ import annotations

import re

from hillclimb.core.types import Evaluation
from hillclimb.science.hypothesis_tournament.hypothesis import Hypothesis


_TESTABILITY_KEYWORDS = (
    "measure",
    "compare",
    "assay",
    "knockout",
    "inhibit",
    "dose",
    "cohort",
    "randomized",
    "expression",
    "survival",
    "response rate",
)

_EVIDENCE_KEYWORDS = (
    "mutation",
    "pathway",
    "resistance",
    "efflux",
    "target",
    "binding",
    "clinical",
    "preclinical",
    "biomarker",
    "mechanism",
)

_MECHANISM_KEYWORDS = (
    "via",
    "through",
    "mediated",
    "pathway",
    "signaling",
    "regulation",
    "expression",
    "interaction",
)


def _novelty_score(text: str, seen_claims: set[str]) -> float:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    if normalized in seen_claims:
        return 0.2
    overlap = sum(1 for prior in seen_claims if _jaccard(normalized, prior) > 0.6)
    base = 1.0 - min(0.8, overlap * 0.25)
    unique_tokens = len(set(normalized.split()))
    length_bonus = min(0.2, unique_tokens / 40.0)
    return min(1.0, base + length_bonus)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _keyword_density(text: str, keywords: tuple[str, ...]) -> float:
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw in lower)
    return min(1.0, hits / max(1, len(keywords) // 3))


def score_hypothesis(
    hypothesis: Hypothesis,
    *,
    research_question: str = "",
    evidence_context: str = "",
    seen_claims: set[str] | None = None,
) -> dict[str, float]:
    """Heuristic scores in [0, 1] for novelty, testability, evidence alignment."""
    seen = seen_claims or set()
    full_text = hypothesis.as_text()
    context = f"{research_question} {evidence_context}".lower()

    testability = _keyword_density(hypothesis.testable_prediction, _TESTABILITY_KEYWORDS)
    if re.search(r"\d", hypothesis.testable_prediction):
        testability = min(1.0, testability + 0.15)
    if len(hypothesis.testable_prediction.split()) >= 8:
        testability = min(1.0, testability + 0.1)

    evidence = _keyword_density(full_text, _EVIDENCE_KEYWORDS)
    if context:
        context_hits = sum(1 for tok in full_text.lower().split() if tok in context and len(tok) > 4)
        evidence = min(1.0, evidence + context_hits * 0.05)
    mechanism = _keyword_density(hypothesis.mechanism, _MECHANISM_KEYWORDS)
    evidence = min(1.0, 0.6 * evidence + 0.4 * mechanism)

    novelty = _novelty_score(hypothesis.claim, seen)

    return {
        "novelty": round(novelty, 4),
        "testability": round(testability, 4),
        "evidence_alignment": round(evidence, 4),
    }


def composite_score(scores: dict[str, float]) -> float:
    return (
        0.35 * scores["novelty"]
        + 0.35 * scores["testability"]
        + 0.30 * scores["evidence_alignment"]
    )


class HypothesisEvaluator:
    """Scores a population of hypotheses; higher is better."""

    def __init__(
        self,
        research_question: str,
        evidence_context: str = "",
    ) -> None:
        self.research_question = research_question
        self.evidence_context = evidence_context

    def evaluate_population(self, hypotheses: list[Hypothesis]) -> tuple[float, dict[str, float]]:
        seen: set[str] = set()
        totals = {"novelty": 0.0, "testability": 0.0, "evidence_alignment": 0.0}
        for hyp in hypotheses:
            scores = score_hypothesis(
                hyp,
                research_question=self.research_question,
                evidence_context=self.evidence_context,
                seen_claims=seen,
            )
            for key in totals:
                totals[key] += scores[key]
            seen.add(re.sub(r"\s+", " ", hyp.claim.lower().strip()))
        n = max(1, len(hypotheses))
        avg = {k: round(v / n, 4) for k, v in totals.items()}
        return composite_score(avg), avg

    def evaluate_candidate(self, state: dict) -> Evaluation:
        hypotheses: list[Hypothesis] = state["hypotheses"]
        score, breakdown = self.evaluate_population(hypotheses)
        top = max(
            hypotheses,
            key=lambda h: composite_score(
                score_hypothesis(
                    h,
                    research_question=self.research_question,
                    evidence_context=self.evidence_context,
                )
            ),
        )
        return Evaluation(
            score=score,
            diagnostics={
                "breakdown": breakdown,
                "population_size": len(hypotheses),
                "best_hypothesis_id": top.id,
                "best_claim": top.claim,
            },
        )
