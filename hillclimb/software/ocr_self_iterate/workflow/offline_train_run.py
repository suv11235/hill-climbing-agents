from __future__ import annotations

import argparse

from hillclimb.software.ocr_self_iterate.synthetic import generate_invoice
from hillclimb.software.ocr_self_iterate.workflow.config import baseline_workflow_config
from hillclimb.software.ocr_self_iterate.workflow.offline_train import OfflineTrainConfig, OfflineWorkflowTrainer
from hillclimb.software.ocr_self_iterate.workflow.registry import WorkflowRegistry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline OCR workflow training with backward feedback propagation."
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--docs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--registry",
        default="results/ocr_workflow_registry.json",
        help="Path to save/load workflow registry",
    )
    parser.add_argument(
        "--resume-stable",
        action="store_true",
        help="Resume from stable workflow in registry if available",
    )
    args = parser.parse_args()

    batch = []
    for i in range(args.docs):
        doc = generate_invoice(seed=args.seed + i)
        batch.append((doc.image, doc.ground_truth, doc.rendered_text))

    train_cfg = OfflineTrainConfig(epochs=args.epochs, registry_path=__import__("pathlib").Path(args.registry))

    initial = None
    if args.resume_stable:
        initial = OfflineWorkflowTrainer.load_stable(train_cfg.registry_path, batch)

    trainer = OfflineWorkflowTrainer(documents=batch, train_config=train_cfg)
    summary = trainer.train(initial=initial or baseline_workflow_config())

    print("=== Offline Workflow Training ===")
    print(f"Epochs: {summary['epochs_run']}")
    print(f"Stable version: {summary['stable_version_id']}")
    print(f"Final status: {summary['final_status']}")
    print(f"Final accuracy: {summary['final_accuracy']:.3f}")
    print(f"Registry: {summary['registry_path']}")
    print("\nEpoch history:")
    for step in summary["history"]:
        print(
            f"  epoch {step['epoch']:02d}: acc={step['mean_accuracy']:.3f} "
            f"reward={step['mean_reward']:.3f} status={step['status']} "
            f"feedback={step['feedback_count']} stable={step['stable']}"
        )


if __name__ == "__main__":
    main()
