from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hypothesis:
    """Scientific hypothesis with claim, mechanism, and testable prediction."""

    claim: str
    mechanism: str
    testable_prediction: str
    id: str = ""
    elo: float = 1500.0
    generation: int = 0
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            tag = abs(hash((self.claim, self.mechanism, self.testable_prediction))) % 0xFFFF
            self.id = f"h{self.generation}_{tag:04x}"

    def as_text(self) -> str:
        return f"{self.claim} | {self.mechanism} | {self.testable_prediction}"
