"""Lean theorem-proving hill-climber prototype (AlphaProof Nexus inspired)."""

from hillclimb.science.lean_prover.evaluator import ProofEvaluator
from hillclimb.science.lean_prover.mock_lean import MockLeanVerifier, VerificationResult
from hillclimb.science.lean_prover.problems import FormalProblem, all_problems
from hillclimb.science.lean_prover.proposer import ProofProposer
from hillclimb.science.lean_prover.tactics import Tactic, TacticStep

__all__ = [
    "FormalProblem",
    "MockLeanVerifier",
    "ProofEvaluator",
    "ProofProposer",
    "Tactic",
    "TacticStep",
    "VerificationResult",
    "all_problems",
]
