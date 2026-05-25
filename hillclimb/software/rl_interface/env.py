from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import random


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass
class GridWorldConfig:
    size: int = 5
    hard_mode: bool = False
    max_steps: int = 50
    seed: int | None = None


@dataclass
class RawState:
    """Environment state before interface mapping."""

    x: int
    y: int
    step: int
    at_goal: bool

    def to_features(self, hard_mode: bool = False) -> list[float]:
        """Flatten state into a feature vector for observation programs."""
        if not hard_mode:
            return [float(self.x), float(self.y), float(self.step), float(self.at_goal)]

        # Hard mode: true position hidden at indices 2,3; baseline picks decoys at 0,1.
        decoy_a = float((self.x * 3 + 1) % 7) / 6.0
        decoy_b = float((self.y * 2 + 3) % 6) / 5.0
        distractor_a = float((self.x + self.y) % 5)
        distractor_b = float(4 - self.x)
        distractor_c = float(self.x * self.y)
        parity = float((self.x + self.y) % 2)
        mirror_y = float(4 - self.y)
        return [
            decoy_a,
            decoy_b,
            float(self.x),
            float(self.y),
            distractor_a,
            distractor_b,
            distractor_c,
            parity,
            mirror_y,
        ]


class GridWorld:
    """Simple grid navigation: start (0,0), goal (size-1, size-1)."""

    def __init__(self, config: GridWorldConfig | None = None) -> None:
        self.config = config or GridWorldConfig()
        self._rng = random.Random(self.config.seed)
        self.x = 0
        self.y = 0
        self.steps = 0

    @property
    def goal(self) -> tuple[int, int]:
        s = self.config.size - 1
        return s, s

    def reset(self, seed: int | None = None) -> RawState:
        if seed is not None:
            self._rng = random.Random(seed)
        self.x = 0
        self.y = 0
        self.steps = 0
        return self.raw_state

    @property
    def raw_state(self) -> RawState:
        gx, gy = self.goal
        return RawState(
            x=self.x,
            y=self.y,
            step=self.steps,
            at_goal=self.x == gx and self.y == gy,
        )

    def step(self, action: Action) -> tuple[RawState, bool]:
        if action == Action.UP:
            self.y = max(0, self.y - 1)
        elif action == Action.DOWN:
            self.y = min(self.config.size - 1, self.y + 1)
        elif action == Action.LEFT:
            self.x = max(0, self.x - 1)
        elif action == Action.RIGHT:
            self.x = min(self.config.size - 1, self.x + 1)

        self.steps += 1
        done = self.raw_state.at_goal or self.steps >= self.config.max_steps
        return self.raw_state, done

    def manhattan_to_goal(self) -> int:
        gx, gy = self.goal
        return abs(self.x - gx) + abs(self.y - gy)
