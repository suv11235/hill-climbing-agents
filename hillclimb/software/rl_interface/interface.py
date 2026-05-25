from __future__ import annotations

from dataclasses import dataclass, field

from hillclimb.software.rl_interface.env import RawState


@dataclass
class ObservationProgram:
    """
    Linear feature selector + quantizer producing a discrete observation key.

    Each entry in `feature_indices` picks a raw feature; `weights` scale it.
    """

    feature_indices: list[int] = field(default_factory=lambda: [0, 1])
    weights: list[float] = field(default_factory=lambda: [1.0, 1.0])
    bias: float = 0.0
    num_bins: int = 5

    def observe(self, raw: RawState, hard_mode: bool = False) -> tuple[int, int]:
        features = raw.to_features(hard_mode=hard_mode)
        coords: list[int] = []
        for idx, weight in zip(self.feature_indices, self.weights, strict=False):
            if idx < 0 or idx >= len(features):
                value = 0.0
            else:
                value = features[idx] * weight + self.bias
            clamped = max(0, min(self.num_bins - 1, int(round(value))))
            coords.append(clamped)
        if len(coords) == 1:
            coords.append(0)
        return coords[0], coords[1]

    def copy(self) -> ObservationProgram:
        return ObservationProgram(
            feature_indices=list(self.feature_indices),
            weights=list(self.weights),
            bias=self.bias,
            num_bins=self.num_bins,
        )


@dataclass
class RewardProgram:
    """Shaping + terminal reward computed from raw transition info."""

    goal_reward: float = 10.0
    step_penalty: float = -0.1
    distance_coef: float = 0.0
    action_penalty: float = 0.0
    use_distance_shaping: bool = False

    def compute(
        self,
        prev: RawState,
        nxt: RawState,
        action: int,
        prev_dist: int,
        nxt_dist: int,
    ) -> float:
        reward = self.step_penalty + self.action_penalty * float(action)
        if nxt.at_goal:
            reward += self.goal_reward
        if self.use_distance_shaping:
            reward += self.distance_coef * float(prev_dist - nxt_dist)
        return reward

    def copy(self) -> RewardProgram:
        return RewardProgram(
            goal_reward=self.goal_reward,
            step_penalty=self.step_penalty,
            distance_coef=self.distance_coef,
            action_penalty=self.action_penalty,
            use_distance_shaping=self.use_distance_shaping,
        )


@dataclass
class RLInterface:
    observation: ObservationProgram
    reward: RewardProgram
    hard_mode: bool = False

    def copy(self) -> RLInterface:
        return RLInterface(
            observation=self.observation.copy(),
            reward=self.reward.copy(),
            hard_mode=self.hard_mode,
        )


def baseline_interface(hard_mode: bool = False) -> RLInterface:
    """Deliberately weak mapping: uses distractor features in hard mode."""
    if hard_mode:
        obs = ObservationProgram(
            feature_indices=[0, 1],
            weights=[1.0, 1.0],
            bias=0.0,
        )
        reward = RewardProgram(
            goal_reward=10.0,
            step_penalty=-0.1,
            use_distance_shaping=False,
        )
    else:
        obs = ObservationProgram(feature_indices=[0, 1], weights=[1.0, 1.0])
        reward = RewardProgram(goal_reward=10.0, step_penalty=-0.1)
    return RLInterface(observation=obs, reward=reward, hard_mode=hard_mode)


def oracle_interface(hard_mode: bool = False) -> RLInterface:
    """Strong hand-crafted interface for sanity checks."""
    if hard_mode:
        obs = ObservationProgram(feature_indices=[2, 3], weights=[1.0, 1.0])
    else:
        obs = ObservationProgram(feature_indices=[0, 1], weights=[1.0, 1.0])
    reward = RewardProgram(
        goal_reward=10.0,
        step_penalty=-0.05,
        distance_coef=0.5,
        use_distance_shaping=True,
    )
    return RLInterface(observation=obs, reward=reward, hard_mode=hard_mode)
