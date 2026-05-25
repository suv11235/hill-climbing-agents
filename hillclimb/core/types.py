from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AcceptPolicy(str, Enum):
    GREEDY = "greedy"
    SIMULATED_ANNEALING = "simulated_annealing"
    ALWAYS = "always"


@dataclass
class Candidate:
    """A proposed solution in the search space."""

    state: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent_id: str | None = None
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"gen{self.generation}_{id(self.state) & 0xFFFF:04x}"


@dataclass
class Evaluation:
    """Result of evaluating a candidate against the objective."""

    score: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    error: str | None = None

    @property
    def improved(self) -> bool:
        return self.diagnostics.get("improved", False)
