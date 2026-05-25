from __future__ import annotations

from hillclimb.core.harness import HillClimber
from hillclimb.core.types import AcceptPolicy, Candidate
from hillclimb.software.rl_interface.env import Action, GridWorld, GridWorldConfig
from hillclimb.software.rl_interface.evaluator import QLearningConfig, RLInterfaceEvaluator
from hillclimb.software.rl_interface.interface import (
    ObservationProgram,
    RLInterface,
    RewardProgram,
    baseline_interface,
    oracle_interface,
)
from hillclimb.software.rl_interface.proposer import MutationMode, RLInterfaceProposer


def test_gridworld_reaches_goal():
    env = GridWorld(GridWorldConfig(size=5, max_steps=20))
    env.reset()
    for _ in range(10):
        env.step(Action.RIGHT)
    for _ in range(10):
        env.step(Action.DOWN)
    assert env.raw_state.at_goal


def test_hard_mode_features_include_position():
    state = GridWorld().raw_state
    feats = state.to_features(hard_mode=True)
    assert feats[2] == float(state.x)
    assert feats[3] == float(state.y)
    assert len(feats) == 9


def test_observation_program_discretizes():
    obs = ObservationProgram(feature_indices=[0, 1], weights=[1.0, 1.0])
    state = GridWorld().raw_state
    key = obs.observe(state, hard_mode=False)
    assert isinstance(key, tuple)
    assert len(key) == 2


def test_evaluator_returns_score_in_unit_interval():
    interface = oracle_interface(hard_mode=False)
    evaluator = RLInterfaceEvaluator(
        env_config=GridWorldConfig(hard_mode=False),
        ql_config=QLearningConfig(episodes=60, eval_episodes=20, seed=1),
    )
    result = evaluator.evaluate(Candidate(state=interface))
    assert 0.0 <= result.score <= 1.0
    assert result.diagnostics["success_rate"] == result.score


def test_oracle_beats_baseline_on_easy_grid():
    env_cfg = GridWorldConfig(hard_mode=False)
    ql_cfg = QLearningConfig(episodes=80, eval_episodes=30, seed=2)
    evaluator = RLInterfaceEvaluator(env_config=env_cfg, ql_config=ql_cfg)

    baseline_score = evaluator.evaluate(
        Candidate(state=baseline_interface(hard_mode=False))
    ).score
    oracle_score = evaluator.evaluate(
        Candidate(state=oracle_interface(hard_mode=False))
    ).score
    assert oracle_score >= baseline_score


def test_joint_outperforms_reward_only_on_hard_grid():
    env_cfg = GridWorldConfig(hard_mode=True)
    ql_cfg = QLearningConfig(episodes=100, eval_episodes=30, seed=3)
    max_rounds = 10

    def climb(mode: MutationMode) -> float:
        proposer = RLInterfaceProposer(mode=mode, hard_mode=True, seed=3)
        evaluator = RLInterfaceEvaluator(env_config=env_cfg, ql_config=ql_cfg)
        climber = HillClimber(
            proposer=proposer,
            evaluator=evaluator,
            max_rounds=max_rounds,
            early_stop_patience=4,
            accept_policy=AcceptPolicy.GREEDY,
        )
        result = climber.climb(Candidate(state=baseline_interface(hard_mode=True)))
        return result.best_score

    joint_score = climb(MutationMode.JOINT)
    reward_only_score = climb(MutationMode.REWARD_ONLY)
    assert joint_score > reward_only_score
