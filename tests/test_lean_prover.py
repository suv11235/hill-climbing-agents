"""Tests for Lean theorem-proving hill-climber prototype."""

from __future__ import annotations

import random

from hillclimb.core.harness import HillClimber
from hillclimb.core.types import Candidate
from hillclimb.science.lean_prover.evaluator import ProofEvaluator
from hillclimb.science.lean_prover.mock_lean import MockLeanVerifier
from hillclimb.science.lean_prover.problems import all_problems, get_problem
from hillclimb.science.lean_prover.proposer import ProofCandidate, ProofProposer
from hillclimb.science.lean_prover.tactics import TacticStep


def test_all_problems_have_reference_proofs():
    problems = all_problems()
    assert len(problems) >= 3
    for problem in problems:
        assert problem.reference_proof
        assert problem.axioms


def test_mock_lean_verifies_two_plus_two():
    problem = get_problem("two_plus_two")
    verifier = MockLeanVerifier(problem)
    steps = [TacticStep.parse(s) for s in problem.reference_proof]
    result = verifier.verify_script(steps)
    assert result.ok
    assert result.state.solved


def test_mock_lean_rejects_bad_tactic():
    problem = get_problem("two_plus_two")
    verifier = MockLeanVerifier(problem)
    state = verifier.initial_state()
    bad = TacticStep.parse("rewrite mul_zero")
    result = verifier.apply_tactic(state, bad)
    assert not result.ok
    assert result.error == "unknown_axiom"


def test_proof_evaluator_scores_complete_proof():
    problem = get_problem("add_zero_left")
    verifier = MockLeanVerifier(problem)
    evaluator = ProofEvaluator(problem, verifier)
    steps = [TacticStep.parse(s) for s in problem.reference_proof]
    candidate = Candidate(state=ProofCandidate(steps=steps))
    evaluation = evaluator.evaluate(candidate)
    assert evaluation.score == 1.0
    assert evaluation.passed


def test_proof_evaluator_partial_score():
    problem = get_problem("add_zero_right")
    verifier = MockLeanVerifier(problem)
    evaluator = ProofEvaluator(problem, verifier)
    steps = [TacticStep.parse("intro n")]
    candidate = Candidate(state=ProofCandidate(steps=steps))
    evaluation = evaluator.evaluate(candidate)
    assert 0.0 < evaluation.score < 1.0
    assert not evaluation.passed


def test_proposer_extends_script():
    problem = get_problem("two_plus_two")
    verifier = MockLeanVerifier(problem)
    proposer = ProofProposer(problem, verifier, rng=random.Random(0))
    initial = Candidate(state=ProofCandidate())
    proposal = proposer.propose(initial, [])
    proof: ProofCandidate = proposal.state
    assert len(proof.steps) == 1


def test_hill_climb_finds_qed_two_plus_two():
    problem = get_problem("two_plus_two")
    verifier = MockLeanVerifier(problem)
    proposer = ProofProposer(problem, verifier, rng=random.Random(1))
    evaluator = ProofEvaluator(problem, verifier)

    climber = HillClimber(
        proposer=proposer,
        evaluator=evaluator,
        max_rounds=len(problem.reference_proof) + 2,
        early_stop_patience=5,
    )
    result = climber.climb(Candidate(state=ProofCandidate()))
    assert result.best_score == 1.0
    assert result.best.state.steps


def test_hill_climb_finds_qed_add_zero_left():
    problem = get_problem("add_zero_left")
    verifier = MockLeanVerifier(problem)
    proposer = ProofProposer(problem, verifier, rng=random.Random(2))
    evaluator = ProofEvaluator(problem, verifier)

    climber = HillClimber(
        proposer=proposer,
        evaluator=evaluator,
        max_rounds=6,
        early_stop_patience=4,
    )
    result = climber.climb(Candidate(state=ProofCandidate()))
    assert result.best_score == 1.0
