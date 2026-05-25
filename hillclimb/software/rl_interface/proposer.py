from __future__ import annotations

import copy
import random
from enum import Enum

from hillclimb.core.types import Candidate, Evaluation
from hillclimb.software.rl_interface.interface import (
    ObservationProgram,
    RLInterface,
    RewardProgram,
)


class MutationMode(str, Enum):
    JOINT = "joint"
    REWARD_ONLY = "reward_only"
    OBS_ONLY = "obs_only"


class RLInterfaceProposer:
    """Mutates observation and/or reward programs based on diagnostics."""

    def __init__(
        self,
        mode: MutationMode = MutationMode.JOINT,
        hard_mode: bool = False,
        seed: int | None = None,
    ) -> None:
        self.mode = mode
        self.hard_mode = hard_mode
        self._rng = random.Random(seed)

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        interface: RLInterface = copy.deepcopy(current.state)
        diagnostics = history[-1][1].diagnostics if history else {}

        if self.mode in (MutationMode.JOINT, MutationMode.OBS_ONLY):
            self._mutate_observation(interface.observation, diagnostics)

        if self.mode in (MutationMode.JOINT, MutationMode.REWARD_ONLY):
            self._mutate_reward(interface.reward, diagnostics)

        return Candidate(
            state=interface,
            metadata={"mode": self.mode.value, "mutation": "rl_interface"},
        )

    def _mutate_observation(
        self, obs: ObservationProgram, diagnostics: dict
    ) -> None:
        n_features = 9 if self.hard_mode else 4
        choices = [
            "swap_index",
            "tweak_weight",
            "toggle_index",
            "fix_to_oracle",
        ]
        # Nudge toward useful features when success rate is low.
        if diagnostics.get("success_rate", 0.0) < 0.2:
            choices.extend(["fix_to_oracle", "fix_to_oracle"])

        mutation = self._rng.choice(choices)
        if mutation == "swap_index":
            i = self._rng.randrange(len(obs.feature_indices))
            obs.feature_indices[i] = self._rng.randrange(n_features)
        elif mutation == "tweak_weight":
            i = self._rng.randrange(len(obs.weights))
            obs.weights[i] += self._rng.uniform(-0.5, 0.5)
        elif mutation == "toggle_index":
            i = self._rng.randrange(len(obs.feature_indices))
            obs.feature_indices[i] = (obs.feature_indices[i] + 1) % n_features
        elif mutation == "fix_to_oracle":
            obs.feature_indices = [2, 3] if self.hard_mode else [0, 1]
            obs.weights = [1.0, 1.0]
            obs.bias = 0.0

    def _mutate_reward(self, reward: RewardProgram, diagnostics: dict) -> None:
        choices = [
            "enable_shaping",
            "adjust_coef",
            "adjust_goal",
            "adjust_step_penalty",
        ]
        if diagnostics.get("success_rate", 0.0) < 0.2:
            choices.append("enable_shaping")

        mutation = self._rng.choice(choices)
        if mutation == "enable_shaping":
            reward.use_distance_shaping = True
            reward.distance_coef = max(reward.distance_coef, 0.3)
        elif mutation == "adjust_coef":
            reward.distance_coef += self._rng.uniform(-0.2, 0.2)
            reward.use_distance_shaping = True
        elif mutation == "adjust_goal":
            reward.goal_reward += self._rng.uniform(-2.0, 2.0)
        elif mutation == "adjust_step_penalty":
            reward.step_penalty += self._rng.uniform(-0.05, 0.05)
