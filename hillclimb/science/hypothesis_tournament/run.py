from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from hillclimb.core.harness import HillClimber, ClimbResult
from hillclimb.core.types import Candidate, Evaluation
from hillclimb.science.hypothesis_tournament.agents import EvolutionAgent, GenerationAgent
from hillclimb.science.hypothesis_tournament.evaluator import HypothesisEvaluator
from hillclimb.science.hypothesis_tournament.hypothesis import Hypothesis
from hillclimb.science.hypothesis_tournament.tournament import run_pairwise_tournament


@dataclass
class TournamentState:
    research_question: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    evidence_context: str = ""
    tournament_round: int = 0


class TournamentProposer:
    """Hill-climb proposer: tournament rank → evolve population."""

    def __init__(
        self,
        *,
        population_size: int = 4,
        tournament_rounds: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.population_size = population_size
        self.tournament_rounds = tournament_rounds
        self.rng = rng or random.Random()
        self.evolution = EvolutionAgent(self.rng)

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        state: TournamentState = current.state
        ranked = run_pairwise_tournament(
            state.hypotheses,
            research_question=state.research_question,
            evidence_context=state.evidence_context,
            rounds=self.tournament_rounds,
            rng=self.rng,
        )
        evolved = self.evolution.evolve(
            ranked,
            state.research_question,
            population_size=self.population_size,
        )
        new_state = TournamentState(
            research_question=state.research_question,
            hypotheses=evolved,
            evidence_context=state.evidence_context,
            tournament_round=state.tournament_round + 1,
        )
        return Candidate(
            state=new_state,
            metadata={"top_elo": ranked[0].elo if ranked else 0.0},
        )


class TournamentPopulationEvaluator:
    """Wraps HypothesisEvaluator for the hill-climbing harness."""

    def __init__(self, research_question: str, evidence_context: str = "") -> None:
        self._inner = HypothesisEvaluator(research_question, evidence_context)

    def evaluate(self, candidate: Candidate) -> Evaluation:
        state: TournamentState = candidate.state
        return self._inner.evaluate_candidate(
            {
                "hypotheses": state.hypotheses,
                "research_question": state.research_question,
            }
        )


def run_tournament_evolution(
    research_question: str,
    *,
    evidence_context: str = "",
    population_size: int = 4,
    max_rounds: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run Co-Scientist-style tournament evolution for a research question.

    Example: "What causes resistance to drug X?"
    """
    rng = random.Random(seed)
    generator = GenerationAgent(rng)
    initial_hyps = generator.generate(research_question, count=population_size)

    initial_state = TournamentState(
        research_question=research_question,
        hypotheses=initial_hyps,
        evidence_context=evidence_context,
    )
    initial = Candidate(state=initial_state, metadata={"source": "initial_generation"})

    climber = HillClimber(
        proposer=TournamentProposer(
            population_size=population_size,
            rng=rng,
        ),
        evaluator=TournamentPopulationEvaluator(research_question, evidence_context),
        max_rounds=max_rounds,
        early_stop_patience=3,
    )
    result: ClimbResult = climber.climb(initial)

    final_state: TournamentState = result.best.state
    ranked = run_pairwise_tournament(
        final_state.hypotheses,
        research_question=research_question,
        evidence_context=evidence_context,
        rng=rng,
    )

    return {
        "research_question": research_question,
        "best_score": result.best_score,
        "rounds": result.rounds,
        "converged": result.converged,
        "best_hypothesis": ranked[0] if ranked else None,
        "final_population": ranked,
        "history_scores": [ev.score for _, ev in result.history],
    }


def main() -> None:
    question = "What causes resistance to drug X?"
    outcome = run_tournament_evolution(question)
    best = outcome["best_hypothesis"]
    print(f"Research question: {question}")
    print(f"Best score: {outcome['best_score']:.4f} ({outcome['rounds']} rounds)")
    if best:
        print(f"Top hypothesis [{best.id}, Elo={best.elo:.0f}]:")
        print(f"  Claim: {best.claim}")
        print(f"  Mechanism: {best.mechanism}")
        print(f"  Prediction: {best.testable_prediction}")


if __name__ == "__main__":
    main()
