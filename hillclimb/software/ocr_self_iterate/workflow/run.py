from __future__ import annotations

import argparse

from hillclimb.core.harness import HillClimber
from hillclimb.core.types import AcceptPolicy, Candidate, Evaluation
from hillclimb.software.ocr_self_iterate.judge import judge_fields
from hillclimb.software.ocr_self_iterate.schema import INVOICE_SCHEMA
from hillclimb.software.ocr_self_iterate.synthetic import generate_invoice
from hillclimb.software.ocr_self_iterate.workflow.config import baseline_workflow_config
from hillclimb.software.ocr_self_iterate.workflow.proposer import WorkflowProposer
from hillclimb.software.ocr_self_iterate.workflow.rewards import HybridRewardFramework
from hillclimb.software.ocr_self_iterate.workflow.runner import run_workflow


class WorkflowBatchEvaluator:
    """Scores workflow configs via hybrid static+dynamic rewards on a doc batch."""

    def __init__(
        self,
        documents: list[tuple[object, dict[str, str], str]],
        reward_framework: HybridRewardFramework | None = None,
    ) -> None:
        self.documents = documents
        self.reward_framework = reward_framework or HybridRewardFramework()

    def evaluate(self, candidate: Candidate) -> Evaluation:
        config = candidate.state
        rewards: list[float] = []
        accuracies: list[float] = []
        confusion_count = 0
        missing_count = 0
        refine_rounds: list[int] = []
        all_errors: list[dict] = []

        for image, truth, fallback in self.documents:
            result = run_workflow(
                image,
                truth,
                config,
                fallback_text=fallback,
                reward_framework=self.reward_framework,
            )
            rewards.append(result.total_reward)
            acc, diags = judge_fields(result.context.extracted_fields, truth, INVOICE_SCHEMA)
            accuracies.append(acc)
            confusion_count += sum(
                1 for e in result.context.validation_errors if e.get("type") == "confusion"
            )
            missing_count += sum(1 for d in diags if not d.correct)
            refine_rounds.append(result.context.refine_rounds)
            all_errors.extend(result.context.validation_errors)

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0

        return Evaluation(
            score=mean_reward,
            diagnostics={
                "mean_hybrid_reward": mean_reward,
                "mean_field_accuracy": mean_acc,
                "per_doc_rewards": rewards,
                "per_doc_accuracy": accuracies,
                "confusion_count": confusion_count,
                "missing_field_count": missing_count,
                "avg_refine_rounds": sum(refine_rounds) / len(refine_rounds) if refine_rounds else 0,
                "validation_errors": all_errors[:8],
            },
            passed=True,
        )


def run_workflow_climb(
    *,
    docs: int = 4,
    seed: int = 42,
    max_rounds: int = 12,
) -> dict:
    batch = []
    for i in range(docs):
        doc = generate_invoice(seed=seed + i)
        batch.append((doc.image, doc.ground_truth, doc.rendered_text))

    evaluator = WorkflowBatchEvaluator(batch)
    proposer = WorkflowProposer(seed=seed)
    climber = HillClimber(
        proposer=proposer,
        evaluator=evaluator,
        max_rounds=max_rounds,
        early_stop_patience=3,
        accept_policy=AcceptPolicy.GREEDY,
    )

    initial = Candidate(state=baseline_workflow_config())
    result = climber.climb(initial)
    best: WorkflowConfig = result.best.state

    return {
        "rounds": result.rounds,
        "converged": result.converged,
        "initial_reward": result.history[0][1].score,
        "best_reward": result.best_score,
        "initial_accuracy": result.history[0][1].diagnostics.get("mean_field_accuracy", 0),
        "best_accuracy": max(
            ev.diagnostics.get("mean_field_accuracy", 0) for _, ev in result.history
        ),
        "best_orchestrator_prompt": best.orchestrator_prompt[:80],
        "best_max_refine_rounds": best.orchestrator_params.get("max_refine_rounds"),
        "best_extractor_prompt": best.agents["extractor"].system_prompt[:80],
        "history": [
            {
                "round": i,
                "reward": ev.score,
                "accuracy": ev.diagnostics.get("mean_field_accuracy"),
                "confusion": ev.diagnostics.get("confusion_count"),
            }
            for i, (_, ev) in enumerate(result.history)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Workflow-level OCR hill climbing with orchestrator + sub-agents."
    )
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--docs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_workflow_climb(docs=args.docs, seed=args.seed, max_rounds=args.rounds)

    print("=== OCR Workflow Hill Climb ===")
    print(f"Initial: reward={summary['initial_reward']:.3f}  accuracy={summary['initial_accuracy']:.3f}")
    print(f"Best:    reward={summary['best_reward']:.3f}  accuracy={summary['best_accuracy']:.3f}")
    print(f"Rounds: {summary['rounds']}  converged={summary['converged']}")
    print(f"Orchestrator refine rounds: {summary['best_max_refine_rounds']}")
    print(f"Orchestrator prompt: {summary['best_orchestrator_prompt']}...")
    print(f"Extractor prompt: {summary['best_extractor_prompt']}...")
    print("\nTrajectory:")
    for step in summary["history"]:
        print(
            f"  round {step['round']:02d}: reward={step['reward']:.3f} "
            f"acc={step['accuracy']:.3f} confusion={step['confusion']}"
        )


if __name__ == "__main__":
    main()
