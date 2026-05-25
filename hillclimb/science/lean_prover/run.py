"""Demo: hill climb proof search with mock Lean verification."""

from __future__ import annotations

import random

from hillclimb.core.harness import HillClimber
from hillclimb.core.types import Candidate
from hillclimb.science.lean_prover.evaluator import ProofEvaluator
from hillclimb.science.lean_prover.mock_lean import MockLeanVerifier
from hillclimb.science.lean_prover.problems import all_problems, get_problem
from hillclimb.science.lean_prover.proposer import ProofCandidate, ProofProposer


def run(problem_name: str = "two_plus_two", seed: int = 0, max_rounds: int = 10) -> None:
    problem = get_problem(problem_name)
    verifier = MockLeanVerifier(problem)
    proposer = ProofProposer(problem, verifier, rng=random.Random(seed))
    evaluator = ProofEvaluator(problem, verifier)

    climber = HillClimber(
        proposer=proposer,
        evaluator=evaluator,
        max_rounds=max_rounds,
        early_stop_patience=3,
    )

    initial = Candidate(state=ProofCandidate())
    result = climber.climb(initial)

    print("=== Lean Theorem Proving Hill Climber Demo ===")
    print(f"Problem: {problem.name} — {problem.goal}")
    print(f"Rounds: {result.rounds}  Converged: {result.converged}")
    print(f"Best score: {result.best_score:.3f}")

    best: ProofCandidate = result.best.state
    print(f"Proof script ({len(best.steps)} steps):")
    for step in best.steps:
        print(f"  {step}")

    final_eval = result.history[-1][1]
    if final_eval.diagnostics.get("solved"):
        print("Status: QED ✓")
    else:
        print(f"Status: incomplete — {final_eval.diagnostics.get('message', '')}")


def main() -> None:
    print("Available problems:", [p.name for p in all_problems()])
    run("two_plus_two")
    print()
    run("add_zero_left")


if __name__ == "__main__":
    main()
