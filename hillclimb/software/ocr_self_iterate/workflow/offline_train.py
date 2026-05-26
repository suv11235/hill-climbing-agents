from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hillclimb.software.ocr_self_iterate.judge import judge_fields
from hillclimb.software.ocr_self_iterate.schema import INVOICE_SCHEMA
from hillclimb.software.ocr_self_iterate.workflow.config import WorkflowConfig, baseline_workflow_config
from hillclimb.software.ocr_self_iterate.workflow.feedback import (
    FeedbackBatch,
    collect_output_feedback,
    propagate_feedback_backward,
)
from hillclimb.software.ocr_self_iterate.workflow.registry import (
    WorkflowMetrics,
    WorkflowRegistry,
    WorkflowStatus,
)
from hillclimb.software.ocr_self_iterate.workflow.rewards import HybridRewardFramework
from hillclimb.software.ocr_self_iterate.workflow.runner import run_workflow


DocumentBatch = list[tuple[Any, dict[str, str], str]]


@dataclass
class OfflineTrainConfig:
    epochs: int = 8
    min_accuracy_for_stable: float = 0.95
    early_stop_accuracy: float = 0.99
    registry_path: Path = field(default_factory=lambda: Path("results/ocr_workflow_registry.json"))


@dataclass
class OfflineWorkflowTrainer:
    """
    Offline training loop: forward pass → output feedback → backward propagation.

    Feature/prompt engineering is minimal — params are derived from output failures.
    Workflows transition experimental → candidate → stable as metrics plateau.
    """

    documents: DocumentBatch
    train_config: OfflineTrainConfig = field(default_factory=OfflineTrainConfig)
    reward_framework: HybridRewardFramework = field(default_factory=HybridRewardFramework)
    registry: WorkflowRegistry = field(default_factory=WorkflowRegistry)

    def train(self, initial: WorkflowConfig | None = None) -> dict[str, Any]:
        config = initial.copy() if initial else baseline_workflow_config()
        history: list[dict[str, Any]] = []

        for epoch in range(self.train_config.epochs):
            batch_feedback, metrics = self._forward_pass(config, epoch)
            version = self.registry.register(
                config,
                WorkflowMetrics(
                    mean_accuracy=metrics["mean_accuracy"],
                    mean_reward=metrics["mean_reward"],
                    epoch=epoch,
                ),
                version_id="offline_training",
            )

            history.append(
                {
                    "epoch": epoch,
                    "status": version.status.value,
                    "mean_accuracy": metrics["mean_accuracy"],
                    "mean_reward": metrics["mean_reward"],
                    "feedback_count": len(batch_feedback.items),
                    "stable": self.registry.stable_version_id,
                }
            )

            stable = self.registry.get_stable()
            if stable and metrics["mean_accuracy"] >= self.train_config.early_stop_accuracy:
                self.registry.freeze_stable()
                break

            if not version.is_mutable:
                config = stable.config.copy() if stable else config
                break

            config = propagate_feedback_backward(config, batch_feedback)

        self.registry.save(self.train_config.registry_path)
        stable = self.registry.get_stable() or self.registry.get_best_candidate()

        return {
            "epochs_run": len(history),
            "stable_version_id": self.registry.stable_version_id,
            "final_status": stable.status.value if stable else "experimental",
            "final_accuracy": stable.latest_accuracy if stable else history[-1]["mean_accuracy"],
            "history": history,
            "registry_path": str(self.train_config.registry_path),
            "stable_config": stable.config if stable else config,
        }

    def _forward_pass(
        self, config: WorkflowConfig, epoch: int
    ) -> tuple[FeedbackBatch, dict[str, float]]:
        feedback_items = []
        rewards: list[float] = []
        accuracies: list[float] = []
        refine_helped = 0
        refine_attempted = 0

        for image, truth, fallback in self.documents:
            result = run_workflow(
                image,
                truth,
                config,
                fallback_text=fallback,
                reward_framework=self.reward_framework,
            )
            reward, _ = self.reward_framework.evaluate(result, truth)
            acc, _ = judge_fields(result.context.extracted_fields, truth, INVOICE_SCHEMA)

            rewards.append(reward)
            accuracies.append(acc)

            doc_feedback = collect_output_feedback(result, truth)
            feedback_items.extend(doc_feedback)

            if result.context.refine_rounds > 0:
                refine_attempted += 1
                if acc >= 0.99:
                    refine_helped += 1

        batch = FeedbackBatch(
            items=feedback_items,
            mean_accuracy=sum(accuracies) / len(accuracies) if accuracies else 0.0,
            mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
            refine_success_rate=refine_helped / refine_attempted if refine_attempted else 0.0,
        )
        metrics = {
            "mean_accuracy": batch.mean_accuracy,
            "mean_reward": batch.mean_reward,
            "epoch": epoch,
        }
        return batch, metrics

    @classmethod
    def load_stable(cls, registry_path: Path, documents: DocumentBatch) -> WorkflowConfig | None:
        if not registry_path.exists():
            return None
        registry = WorkflowRegistry.load(registry_path)
        stable = registry.get_stable()
        return stable.config if stable else None
