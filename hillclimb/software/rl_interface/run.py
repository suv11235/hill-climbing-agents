from __future__ import annotations

import argparse
import random

from hillclimb.core.harness import HillClimber
from hillclimb.core.types import AcceptPolicy, Candidate
from hillclimb.software.rl_interface.env import GridWorldConfig
from hillclimb.software.rl_interface.evaluator import QLearningConfig, RLInterfaceEvaluator
from hillclimb.software.rl_interface.interface import baseline_interface
from hillclimb.software.rl_interface.proposer import MutationMode, RLInterfaceProposer


def run_climb(
    mode: MutationMode,
    hard_mode: bool,
    max_rounds: int,
    seed: int,
) -> tuple[float, list[tuple[float, dict]]]:
    env_config = GridWorldConfig(hard_mode=hard_mode, seed=seed)
    ql_config = QLearningConfig(seed=seed, episodes=100 if hard_mode else 80)

    proposer = RLInterfaceProposer(mode=mode, hard_mode=hard_mode, seed=seed)
    evaluator = RLInterfaceEvaluator(env_config=env_config, ql_config=ql_config)
    climber = HillClimber(
        proposer=proposer,
        evaluator=evaluator,
        max_rounds=max_rounds,
        early_stop_patience=4,
        accept_policy=AcceptPolicy.GREEDY,
    )

    initial = Candidate(state=baseline_interface(hard_mode=hard_mode))
    result = climber.climb(initial)

    trajectory: list[tuple[float, dict]] = []
    for cand, ev in result.history:
        trajectory.append((ev.score, ev.diagnostics))

    return result.best_score, trajectory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hill-climb RL observation/reward interfaces (LIMEN-inspired toy demo)."
    )
    parser.add_argument("--hard", action="store_true", help="Use obfuscated raw features.")
    parser.add_argument("--rounds", type=int, default=12, help="Max hill-climbing rounds.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare joint vs reward-only modes on the selected task.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    if args.compare:
        print(f"=== RL Interface Discovery (hard={args.hard}) ===")
        for mode in (MutationMode.REWARD_ONLY, MutationMode.JOINT):
            best, trajectory = run_climb(mode, args.hard, args.rounds, args.seed)
            print(f"\nMode: {mode.value}")
            for i, (score, diag) in enumerate(trajectory):
                obs = diag.get("obs_indices", "?")
                shaping = diag.get("reward_shaping", "?")
                print(f"  round {i:02d}: success={score:.3f} obs={obs} shaping={shaping}")
            print(f"  best success rate: {best:.3f}")
        return

    best, trajectory = run_climb(MutationMode.JOINT, args.hard, args.rounds, args.seed)
    print(f"RL interface hill climb (hard={args.hard}, joint mode)")
    for i, (score, diag) in enumerate(trajectory):
        obs = diag.get("obs_indices", "?")
        shaping = diag.get("reward_shaping", "?")
        print(f"round {i:02d}: success={score:.3f} obs={obs} shaping={shaping}")
    print(f"best success rate: {best:.3f}")


if __name__ == "__main__":
    main()
