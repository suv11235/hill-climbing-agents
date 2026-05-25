from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TacticName(str, Enum):
    INTRO = "intro"
    APPLY = "apply"
    REWRITE = "rewrite"
    CASES = "cases"
    INDUCTION = "induction"
    REFLEXIVITY = "rfl"


@dataclass(frozen=True)
class TacticStep:
    """Single tactic invocation."""

    name: TacticName
    arg: str = ""

    def __str__(self) -> str:
        if self.arg:
            return f"{self.name.value} {self.arg}"
        return self.name.value

    @classmethod
    def parse(cls, line: str) -> TacticStep:
        parts = line.strip().split(maxsplit=1)
        name = TacticName(parts[0])
        arg = parts[1] if len(parts) > 1 else ""
        return cls(name=name, arg=arg)


@dataclass
class ProofState:
    """Current proof obligations and context."""

    goals: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    solved: bool = False
    steps_applied: list[TacticStep] = field(default_factory=list)

    def primary_goal(self) -> str | None:
        return self.goals[0] if self.goals else None


class Tactic:
    """Library of proof tactics for natural-number arithmetic."""

    RULES: dict[str, str] = {
        "add_zero": "n + 0 = n",
        "add_succ": "n + suc(m) = suc(n + m)",
        "mul_zero": "n * 0 = 0",
        "mul_succ": "n * suc(m) = n * m + n",
    }

    @staticmethod
    def available(problem_axioms: list[str]) -> list[TacticStep]:
        base = [
            TacticStep(TacticName.INTRO, "n"),
            TacticStep(TacticName.INDUCTION, "n"),
            TacticStep(TacticName.CASES, "n"),
            TacticStep(TacticName.REFLEXIVITY),
        ]
        for axiom in problem_axioms:
            base.append(TacticStep(TacticName.REWRITE, axiom))
            base.append(TacticStep(TacticName.APPLY, axiom))
        return base

    @staticmethod
    def describe(step: TacticStep) -> str:
        if step.name == TacticName.INTRO:
            return f"Introduce variable `{step.arg}`."
        if step.name == TacticName.APPLY:
            return f"Apply lemma `{step.arg}`."
        if step.name == TacticName.REWRITE:
            return f"Rewrite using `{step.arg}`."
        if step.name == TacticName.CASES:
            return f"Case split on `{step.arg}`."
        if step.name == TacticName.INDUCTION:
            return f"Induct on `{step.arg}`."
        if step.name == TacticName.REFLEXIVITY:
            return "Close goal by reflexivity."
        return str(step)
