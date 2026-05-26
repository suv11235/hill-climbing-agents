from __future__ import annotations

import argparse
from pathlib import Path

from hillclimb.core.frontier import FrontierProvider, get_frontier
from hillclimb.software.ocr_self_iterate.synthetic import generate_invoice
from hillclimb.software.ocr_self_iterate.workflow.blackbox.offline import BlackboxOfflineTrainer
from hillclimb.software.ocr_self_iterate.workflow.config import baseline_workflow_config
from hillclimb.software.ocr_self_iterate.workflow.offline_train import OfflineTrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Black-box OCR workflow training via frontier models (Gemini). No model training."
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--docs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "mock"],
        default="gemini",
        help="Frontier provider (default: gemini; falls back to mock without API key)",
    )
    parser.add_argument("--model", default="", help="Frontier model id (e.g. gemini-2.0-flash)")
    parser.add_argument(
        "--registry",
        default="results/blackbox_workflow_registry.json",
    )
    parser.add_argument("--resume-stable", action="store_true")
    args = parser.parse_args()

    batch = []
    for i in range(args.docs):
        doc = generate_invoice(seed=args.seed + i)
        batch.append((doc.image, doc.ground_truth, doc.rendered_text))

    frontier = get_frontier(
        FrontierProvider(args.provider),
        model=args.model or None,
    )
    train_cfg = OfflineTrainConfig(epochs=args.epochs, registry_path=Path(args.registry))

    initial = None
    if args.resume_stable:
        initial = BlackboxOfflineTrainer.load_stable(train_cfg.registry_path, batch, frontier)

    trainer = BlackboxOfflineTrainer(
        documents=batch,
        frontier=frontier,
        train_config=train_cfg,
    )
    summary = trainer.train(initial=initial or baseline_workflow_config())

    print("=== Black-Box OCR Workflow (no model training) ===")
    print(f"Frontier: {summary['frontier_provider']}")
    print(f"Epochs: {summary['epochs_run']}")
    print(f"Stable: {summary['stable_version_id']}  status={summary['final_status']}")
    print(f"Accuracy: {summary['final_accuracy']:.3f}")
    print(f"Registry: {summary['registry_path']}")
    for step in summary["history"]:
        print(
            f"  epoch {step['epoch']:02d}: acc={step['mean_accuracy']:.3f} "
            f"status={step['status']} feedback={step['feedback_count']}"
        )


if __name__ == "__main__":
    main()
