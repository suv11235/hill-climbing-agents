"""Tests for LLM hybrid proposer and benchmark infrastructure."""

from __future__ import annotations

from dataclasses import dataclass

from hillclimb.benchmarks.hybrid import LLMEnhancedProposer, _states_equal
from hillclimb.core.types import Candidate, Evaluation


@dataclass
class _FakeState:
    value: int


class _FixedProposer:
    def __init__(self, delta: int = 1) -> None:
        self.delta = delta

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        state: _FakeState = current.state
        return Candidate(state=_FakeState(state.value + self.delta))


class _FailingProposer:
    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        raise RuntimeError("LLM unavailable")


def test_states_equal_dataclass():
    assert _states_equal(_FakeState(1), _FakeState(1))
    assert not _states_equal(_FakeState(1), _FakeState(2))


def test_hybrid_falls_back_on_llm_failure():
    proposer = LLMEnhancedProposer(llm=_FailingProposer(), fallback=_FixedProposer(1))
    current = Candidate(state=_FakeState(0))
    next_c = proposer.propose(current, [])
    assert next_c.state.value == 1
    assert next_c.metadata["source"] == "llm_enhanced_fallback"


def test_hybrid_uses_llm_when_it_changes_state():
    proposer = LLMEnhancedProposer(llm=_FixedProposer(5), fallback=_FixedProposer(1))
    current = Candidate(state=_FakeState(0))
    next_c = proposer.propose(current, [])
    assert next_c.state.value == 5
    assert next_c.metadata["source"] == "llm_enhanced_llm"


def test_hybrid_falls_back_when_llm_unchanged():
    proposer = LLMEnhancedProposer(llm=_FixedProposer(0), fallback=_FixedProposer(2))
    current = Candidate(state=_FakeState(5))
    # LLM returns same value (5+0=5) which equals current
    next_c = proposer.propose(current, [])
    assert next_c.state.value == 7
