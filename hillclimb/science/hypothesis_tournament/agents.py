from __future__ import annotations

import random
import re

from hillclimb.science.hypothesis_tournament.hypothesis import Hypothesis


_DRUG_PATTERNS = (
    (r"drug\s+(\w+)", "drug"),
    (r"resistance\s+to\s+(\w+)", "resistance target"),
    (r"(\w+)\s+resistance", "resistance target"),
)

_MECHANISM_TEMPLATES = (
    "Increased efflux pump activity reduces intracellular {target} concentration",
    "Target-site {target} mutations decrease binding affinity",
    "Upregulation of bypass signaling restores survival despite {target} inhibition",
    "Epigenetic silencing of apoptotic pathways promotes persistence under {target}",
    "Metabolic reprogramming lowers effective {target} exposure in resistant cells",
)

_CLAIM_TEMPLATES = (
    "{target} resistance is driven primarily by {mechanism_short}",
    "Clinical failure on {target} reflects {mechanism_short}",
    "Acquired {target} resistance emerges through {mechanism_short}",
)

_PREDICTION_TEMPLATES = (
    "Knockout of the proposed mediator will restore {target} sensitivity in resistant lines",
    "Resistant cohorts will show higher {biomarker} than responders in a randomized assay",
    "Combining {target} with a {pathway} inhibitor will improve response rate by >=20%",
    "Single-cell RNA-seq will reveal elevated {biomarker} in resistant clones",
)


def _extract_target(research_question: str) -> str:
    q = research_question.lower()
    for pattern, _ in _DRUG_PATTERNS:
        match = re.search(pattern, q)
        if match:
            return match.group(1)
    tokens = [t for t in re.findall(r"[a-z0-9]+", q) if len(t) > 2]
    return tokens[-1] if tokens else "drug X"


def _short_mechanism(mechanism: str) -> str:
    words = mechanism.split()
    return " ".join(words[:6]) + ("..." if len(words) > 6 else "")


class GenerationAgent:
    """Rule-based hypothesis generator from a research question."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def generate(
        self,
        research_question: str,
        *,
        count: int = 4,
        generation: int = 0,
    ) -> list[Hypothesis]:
        target = _extract_target(research_question)
        biomarkers = ("ABC transporter", "BCL2", "MAPK", "PI3K", "EGFR", "MDR1")
        pathways = ("MAPK", "PI3K/AKT", "JAK/STAT", "Wnt", "NF-kB")
        results: list[Hypothesis] = []

        for i in range(count):
            mechanism = self.rng.choice(_MECHANISM_TEMPLATES).format(target=target)
            claim = self.rng.choice(_CLAIM_TEMPLATES).format(
                target=target,
                mechanism_short=_short_mechanism(mechanism),
            )
            prediction = self.rng.choice(_PREDICTION_TEMPLATES).format(
                target=target,
                biomarker=self.rng.choice(biomarkers),
                pathway=self.rng.choice(pathways),
            )
            results.append(
                Hypothesis(
                    claim=claim,
                    mechanism=mechanism,
                    testable_prediction=prediction,
                    generation=generation,
                )
            )
        return results


class ReflectionAgent:
    """Critiques and refines a hypothesis using rule-based edits."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def reflect(self, hypothesis: Hypothesis, research_question: str) -> Hypothesis:
        target = _extract_target(research_question)
        claim = hypothesis.claim
        mechanism = hypothesis.mechanism
        prediction = hypothesis.testable_prediction

        if "primarily" not in claim.lower():
            claim = claim.replace("is driven", "is primarily driven")
        if "measure" not in prediction.lower() and "assay" not in prediction.lower():
            prediction = f"Measure {target} IC50 shift after perturbation: {prediction}"
        if len(mechanism.split()) < 8:
            mechanism = f"{mechanism}; mediated through adaptive stress-response signaling"

        return Hypothesis(
            claim=claim,
            mechanism=mechanism,
            testable_prediction=prediction,
            generation=hypothesis.generation + 1,
            parent_id=hypothesis.id,
            metadata={"reflected": True},
        )


class RankingAgent:
    """Ranks hypotheses using heuristic composite scores."""

    def rank(
        self,
        hypotheses: list[Hypothesis],
        *,
        research_question: str = "",
        evidence_context: str = "",
    ) -> list[Hypothesis]:
        from hillclimb.science.hypothesis_tournament.evaluator import (
            composite_score,
            score_hypothesis,
        )

        return sorted(
            hypotheses,
            key=lambda h: composite_score(
                score_hypothesis(
                    h,
                    research_question=research_question,
                    evidence_context=evidence_context,
                )
            ),
            reverse=True,
        )


class EvolutionAgent:
    """Combines tournament winners and applies reflection mutations."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.reflection = ReflectionAgent(self.rng)
        self.generation = GenerationAgent(self.rng)

    def evolve(
        self,
        ranked: list[Hypothesis],
        research_question: str,
        *,
        population_size: int = 4,
    ) -> list[Hypothesis]:
        if not ranked:
            return self.generation.generate(research_question, count=population_size)

        survivors = ranked[: max(2, population_size // 2)]
        children: list[Hypothesis] = []

        for parent in survivors:
            child = self.reflection.reflect(parent, research_question)
            if self.rng.random() < 0.5:
                sibling = survivors[(survivors.index(parent) + 1) % len(survivors)]
                child = Hypothesis(
                    claim=f"{child.claim} (combining insights from {sibling.claim[:40]}...)",
                    mechanism=f"{child.mechanism}; cross-talk with {sibling.mechanism[:50]}",
                    testable_prediction=child.testable_prediction,
                    generation=child.generation,
                    parent_id=parent.id,
                    metadata={"crossover": sibling.id},
                )
            children.append(child)

        while len(children) < population_size:
            children.append(
                self.generation.generate(research_question, count=1, generation=children[0].generation)[0]
            )

        return children[:population_size]
