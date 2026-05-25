"""Hybrid LLM + rule-based proposer base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from hillclimb.core.types import Candidate, Evaluation


class ProposerLike(Protocol):
    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate: ...


@dataclass
class LLMEnhancedProposer:
    """
    Try LLM proposal first; fall back to rule-based proposer when LLM fails
    or returns an unchanged state. Guarantees at least baseline progress.
    """

    llm: ProposerLike
    fallback: ProposerLike
    name: str = "llm_enhanced"

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        try:
            llm_candidate = self.llm.propose(current, history)
            if not _states_equal(llm_candidate.state, current.state):
                llm_candidate.metadata.setdefault("source", f"{self.name}_llm")
                return llm_candidate
        except Exception:
            pass

        fallback_candidate = self.fallback.propose(current, history)
        fallback_candidate.metadata.setdefault("source", f"{self.name}_fallback")
        return fallback_candidate


def _states_equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if type(a) != type(b):
        return False
    if hasattr(a, "__dataclass_fields__") and hasattr(b, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(a) == asdict(b)
    if isinstance(a, dict):
        return a == b
    if isinstance(a, str):
        return a.strip() == b.strip()
    return a == b
