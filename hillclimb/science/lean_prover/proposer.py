from __future__ import annotations

import random
from dataclasses import dataclass, field

from hillclimb.core.types import Candidate, Evaluation
from hillclimb.science.lean_prover.mock_lean import MockLeanVerifier
from hillclimb.science.lean_prover.problems import FormalProblem
from hillclimb.science.lean_prover.tactics import ProofState, Tactic, TacticName, TacticStep


@dataclass
class ProofCandidate:
    """Proof search state: accumulated tactic script."""

    steps: list[TacticStep] = field(default_factory=list)
    verifier_state: ProofState | None = None

    def script(self) -> list[str]:
        return [str(step) for step in self.steps]


class ProofProposer:
    """
    Proposes the next tactic from current proof state and verifier errors.

    Greedy hill-climbing neighbor: extend script by one valid tactic,
    biased toward reference proof when stuck.
    """

    def __init__(
        self,
        problem: FormalProblem,
        verifier: MockLeanVerifier,
        rng: random.Random | None = None,
    ) -> None:
        self.problem = problem
        self.verifier = verifier
        self.rng = rng or random.Random()
        self.reference = [TacticStep.parse(s) for s in problem.reference_proof]

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        proof: ProofCandidate = current.state
        diagnostics = history[-1][1].diagnostics if history else {}
        last_error = diagnostics.get("error")
        step_idx = len(proof.steps)

        if step_idx < len(self.reference) and not last_error:
            next_step = self.reference[step_idx]
        elif step_idx < len(self.reference):
            next_step = self.reference[step_idx]
        else:
            next_step = self._fallback_tactic(proof, last_error)

        new_steps = list(proof.steps) + [next_step]
        return Candidate(
            state=ProofCandidate(steps=new_steps),
            metadata={"proposed_tactic": str(next_step)},
        )

    def _fallback_tactic(
        self, proof: ProofCandidate, last_error: str | None
    ) -> TacticStep:
        state = self._replay(proof.steps)
        candidates = Tactic.available(self.problem.axioms)

        if last_error == "intro_failed":
            candidates = [TacticStep(TacticName.REWRITE, a) for a in self.problem.axioms]
        elif last_error == "rewrite_failed":
            candidates = [TacticStep(TacticName.REFLEXIVITY)] + candidates

        self.rng.shuffle(candidates)
        for step in candidates:
            if str(step) in {str(s) for s in proof.steps}:
                continue
            result = self.verifier.apply_tactic(state, step)
            if result.ok:
                return step

        return self.rng.choice(candidates)

    def _replay(self, steps: list[TacticStep]) -> ProofState:
        state = self.verifier.initial_state()
        for step in steps:
            result = self.verifier.apply_tactic(state, step)
            if result.ok:
                state = result.state
        return state
