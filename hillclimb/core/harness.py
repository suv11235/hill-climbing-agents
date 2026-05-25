from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from hillclimb.core.types import AcceptPolicy, Candidate, Evaluation


class Proposer(Protocol):
    """Generates candidate solutions from current state and diagnostics."""

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate: ...


class Evaluator(Protocol):
    """Scores a candidate; higher is better unless maximize=False."""

    def evaluate(self, candidate: Candidate) -> Evaluation: ...


@dataclass
class ClimbResult:
    best: Candidate
    best_score: float
    history: list[tuple[Candidate, Evaluation]] = field(default_factory=list)
    rounds: int = 0
    converged: bool = False


@dataclass
class HillClimber:
    """
    Universal hill-climbing harness inspired by Li et al. (2026):
    greedy acceptance with early stopping is the strong default.
    """

    proposer: Proposer
    evaluator: Evaluator
    accept_policy: AcceptPolicy = AcceptPolicy.GREEDY
    max_rounds: int = 20
    early_stop_patience: int = 3
    sa_temperature: float = 1.0
    sa_cooling: float = 0.95
    maximize: bool = True

    def climb(self, initial: Candidate) -> ClimbResult:
        current = initial
        current_eval = self.evaluator.evaluate(current)
        best = current
        best_score = current_eval.score
        history: list[tuple[Candidate, Evaluation]] = [(current, current_eval)]
        no_improve = 0
        temperature = self.sa_temperature

        for round_num in range(1, self.max_rounds + 1):
            proposal = self.proposer.propose(current, history)
            proposal.generation = round_num
            proposal.parent_id = current.id

            try:
                proposal_eval = self.evaluator.evaluate(proposal)
            except Exception as exc:
                proposal_eval = Evaluation(
                    score=best_score if self.maximize else float("inf"),
                    diagnostics={"exception": str(exc)},
                    passed=False,
                    error=str(exc),
                )

            delta = proposal_eval.score - current_eval.score
            if not self.maximize:
                delta = -delta

            improved = delta > 0
            proposal_eval.diagnostics["improved"] = improved
            proposal_eval.diagnostics["delta"] = delta
            history.append((proposal, proposal_eval))

            if self._accept(current_eval.score, proposal_eval.score, temperature):
                current = proposal
                current_eval = proposal_eval
                if (self.maximize and proposal_eval.score > best_score) or (
                    not self.maximize and proposal_eval.score < best_score
                ):
                    best = proposal
                    best_score = proposal_eval.score
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if no_improve >= self.early_stop_patience:
                return ClimbResult(
                    best=best,
                    best_score=best_score,
                    history=history,
                    rounds=round_num,
                    converged=True,
                )

            temperature *= self.sa_cooling

        return ClimbResult(
            best=best,
            best_score=best_score,
            history=history,
            rounds=self.max_rounds,
            converged=False,
        )

    def _accept(self, current_score: float, proposal_score: float, temperature: float) -> bool:
        if self.accept_policy == AcceptPolicy.ALWAYS:
            return True
        delta = proposal_score - current_score
        if not self.maximize:
            delta = -delta
        if delta > 0:
            return True
        if self.accept_policy == AcceptPolicy.GREEDY:
            return False
        # Simulated annealing
        if temperature <= 0:
            return False
        prob = math.exp(delta / temperature)
        return random.random() < prob


def rule_based_proposer(
    mutate_fn: Callable[[Any, dict], Any],
) -> Proposer:
    """Wrap a mutation function as a Proposer."""

    class _Proposer:
        def propose(
            self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
        ) -> Candidate:
            diagnostics = history[-1][1].diagnostics if history else {}
            new_state = mutate_fn(current.state, diagnostics)
            return Candidate(state=new_state, metadata={"source": "rule_based"})

    return _Proposer()
