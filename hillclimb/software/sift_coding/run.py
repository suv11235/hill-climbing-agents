"""Greedy hill climb on code patches with judge pre-filtering."""

from __future__ import annotations

from dataclasses import dataclass

from hillclimb.core.harness import HillClimber
from hillclimb.core.types import AcceptPolicy, Candidate, Evaluation
from hillclimb.software.sift_coding.evaluator import evaluate_code
from hillclimb.software.sift_coding.judge import judge_code
from hillclimb.software.sift_coding.patcher import generate_candidates
from hillclimb.software.sift_coding.task import CodingTask, TASKS, get_task


@dataclass
class SiftCodingProposer:
    task: CodingTask

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        diagnostics = history[-1][1].diagnostics if history else {}
        candidates = generate_candidates(current.state, self.task, diagnostics)

        best_code = candidates[0]
        best_judge = -1.0
        for code in candidates:
            quick_eval = evaluate_code(code, self.task)
            verdict = judge_code(
                code,
                self.task,
                test_pass_rate=quick_eval.diagnostics.get("pass_rate", 0.0),
            )
            if verdict.accepted and verdict.score > best_judge:
                best_judge = verdict.score
                best_code = code

        return Candidate(
            state=best_code,
            metadata={"judge_score": best_judge, "source": "sift_patcher"},
        )


@dataclass
class SiftCodingEvaluator:
    task: CodingTask
    judge_weight: float = 0.15

    def evaluate(self, candidate: Candidate) -> Evaluation:
        result = evaluate_code(candidate.state, self.task)
        pass_rate = result.diagnostics.get("pass_rate", 0.0)
        verdict = judge_code(candidate.state, self.task, test_pass_rate=pass_rate)

        combined = (1.0 - self.judge_weight) * pass_rate + self.judge_weight * verdict.score
        result.score = combined
        result.diagnostics["judge_score"] = verdict.score
        result.diagnostics["judge_reasons"] = list(verdict.reasons)
        result.diagnostics["combined_score"] = combined
        result.diagnostics["judge_accepted"] = verdict.accepted
        return result


def run_climb(
    task_name: str = "reverse_string",
    *,
    max_rounds: int = 10,
    early_stop_patience: int = 3,
) -> dict:
    task = get_task(task_name)
    initial = Candidate(
        state=task.starter_code,
        metadata={"source": "starter"},
    )

    climber = HillClimber(
        proposer=SiftCodingProposer(task=task),
        evaluator=SiftCodingEvaluator(task=task),
        accept_policy=AcceptPolicy.GREEDY,
        max_rounds=max_rounds,
        early_stop_patience=early_stop_patience,
    )
    result = climber.climb(initial)

    initial_pass = result.history[0][1].diagnostics.get("pass_rate", 0.0)
    best_pass_rate = max(
        ev.diagnostics.get("pass_rate", 0.0) for _, ev in result.history
    )

    return {
        "task": task.name,
        "rounds": result.rounds,
        "converged": result.converged,
        "initial_pass_rate": initial_pass,
        "best_pass_rate": best_pass_rate,
        "best_score": result.best_score,
        "best_code": result.best.state,
        "history_len": len(result.history),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SIFT coding self-improvement hill climb")
    parser.add_argument("--task", default="reverse_string", choices=sorted(TASKS))
    parser.add_argument("--max-rounds", type=int, default=10)
    args = parser.parse_args()

    summary = run_climb(args.task, max_rounds=args.max_rounds)
    print(f"Task: {summary['task']}")
    print(f"Initial pass rate: {summary['initial_pass_rate']:.0%}")
    print(f"Best pass rate:    {summary['best_pass_rate']:.0%}")
    print(f"Combined score:    {summary['best_score']:.3f}")
    print(f"Rounds: {summary['rounds']}  converged={summary['converged']}")
    print("--- best code ---")
    print(summary["best_code"])


if __name__ == "__main__":
    main()
