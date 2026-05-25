from __future__ import annotations

import random
from dataclasses import dataclass

from hillclimb.core.types import Candidate, Evaluation
from hillclimb.software.rl_interface.env import Action, GridWorld, GridWorldConfig
from hillclimb.software.rl_interface.interface import RLInterface


@dataclass
class QLearningConfig:
    episodes: int = 120
    max_steps: int = 50
    alpha: float = 0.3
    gamma: float = 0.95
    epsilon: float = 0.2
    eval_episodes: int = 40
    eval_epsilon: float = 0.05
    seed: int = 0


class RLInterfaceEvaluator:
    """Train tabular Q-learning on an interface; score = goal success rate."""

    def __init__(
        self,
        env_config: GridWorldConfig | None = None,
        ql_config: QLearningConfig | None = None,
    ) -> None:
        self.env_config = env_config or GridWorldConfig()
        self.ql_config = ql_config or QLearningConfig()

    def evaluate(self, candidate: Candidate) -> Evaluation:
        interface: RLInterface = candidate.state
        train_seed = self.ql_config.seed
        q_table = self._train(interface, seed=train_seed)
        success_rate = self._evaluate_policy(interface, q_table, seed=train_seed + 1)

        return Evaluation(
            score=success_rate,
            diagnostics={
                "success_rate": success_rate,
                "episodes_trained": self.ql_config.episodes,
                "hard_mode": interface.hard_mode,
                "obs_indices": list(interface.observation.feature_indices),
                "reward_shaping": interface.reward.use_distance_shaping,
            },
            passed=True,
        )

    def _train(self, interface: RLInterface, seed: int) -> dict[tuple, list[float]]:
        rng = random.Random(seed)
        q: dict[tuple, list[float]] = {}
        cfg = self.ql_config

        for ep in range(cfg.episodes):
            env = GridWorld(self.env_config)
            state = env.reset(seed=seed + ep)
            obs = interface.observation.observe(state, hard_mode=interface.hard_mode)
            self._ensure_state(q, obs)

            for _ in range(cfg.max_steps):
                if rng.random() < cfg.epsilon:
                    action = rng.randrange(4)
                else:
                    action = max(range(4), key=lambda a: q[obs][a])

                prev_dist = env.manhattan_to_goal()
                nxt, done = env.step(Action(action))
                nxt_obs = interface.observation.observe(nxt, hard_mode=interface.hard_mode)
                self._ensure_state(q, nxt_obs)

                reward = interface.reward.compute(
                    state, nxt, action, prev_dist, env.manhattan_to_goal()
                )
                best_next = max(q[nxt_obs])
                q[obs][action] += cfg.alpha * (
                    reward + cfg.gamma * best_next * (0.0 if done else 1.0) - q[obs][action]
                )
                obs = nxt_obs
                state = nxt
                if done:
                    break

        return q

    def _evaluate_policy(
        self, interface: RLInterface, q: dict[tuple, list[float]], seed: int
    ) -> float:
        rng = random.Random(seed)
        successes = 0
        cfg = self.ql_config

        for ep in range(cfg.eval_episodes):
            env = GridWorld(self.env_config)
            state = env.reset(seed=seed + ep)
            obs = interface.observation.observe(state, hard_mode=interface.hard_mode)
            reached = False

            for _ in range(cfg.max_steps):
                if rng.random() < cfg.eval_epsilon:
                    action = rng.randrange(4)
                else:
                    self._ensure_state(q, obs)
                    action = max(range(4), key=lambda a: q[obs][a])

                state, done = env.step(Action(action))
                obs = interface.observation.observe(state, hard_mode=interface.hard_mode)
                if state.at_goal:
                    reached = True
                if done:
                    break

            if reached:
                successes += 1

        return successes / cfg.eval_episodes

    @staticmethod
    def _ensure_state(q: dict[tuple, list[float]], obs: tuple[int, int]) -> None:
        if obs not in q:
            q[obs] = [0.0, 0.0, 0.0, 0.0]
