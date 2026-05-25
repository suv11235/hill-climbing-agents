from __future__ import annotations

from hillclimb.core.types import Candidate, Evaluation
from hillclimb.science.lean_prover.mock_lean import MockLeanVerifier
from hillclimb.science.lean_prover.problems import FormalProblem
from hillclimb.science.lean_prover.proposer import ProofCandidate
from hillclimb.science.lean_prover.tactics import TacticStep


class ProofEvaluator:
    """
    Scores proof candidates by progress toward QED.

    Score = 1.0 when the proof is complete; otherwise fractional progress
    based on goals closed and valid steps applied.
    """

    def __init__(self, problem: FormalProblem, verifier: MockLeanVerifier) -> None:
        self.problem = problem
        self.verifier = verifier
        self.reference_len = max(len(problem.reference_proof), 1)

    def evaluate(self, candidate: Candidate) -> Evaluation:
        proof: ProofCandidate = candidate.state
        result = self.verifier.verify_script(proof.steps)
        state = result.state

        if result.ok and state.solved:
            score = 1.0
        else:
            score = self.verifier.score_state(state, self.reference_len)

        diagnostics = {
            "solved": state.solved,
            "n_steps": len(proof.steps),
            "n_goals": len(state.goals),
            "goals": list(state.goals),
            "script": proof.script(),
            "error": result.error,
            "message": result.message,
        }

        return Evaluation(
            score=score,
            diagnostics=diagnostics,
            passed=state.solved,
            error=result.error,
        )

    @staticmethod
    def parse_steps(lines: list[str]) -> list[TacticStep]:
        return [TacticStep.parse(line) for line in lines]
