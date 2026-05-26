from __future__ import annotations

from pathlib import Path

from hillclimb.core.frontier import FrontierProvider, MockFrontierModel, get_frontier
from hillclimb.software.ocr_self_iterate.judge import judge_fields
from hillclimb.software.ocr_self_iterate.schema import INVOICE_SCHEMA
from hillclimb.software.ocr_self_iterate.synthetic import generate_invoice
from hillclimb.software.ocr_self_iterate.workflow.blackbox.offline import BlackboxOfflineTrainer
from hillclimb.software.ocr_self_iterate.workflow.blackbox.runner import (
    BlackboxRunConfig,
    propagate_via_frontier,
    run_blackbox_workflow,
)
from hillclimb.software.ocr_self_iterate.workflow.config import baseline_workflow_config
from hillclimb.software.ocr_self_iterate.workflow.feedback import (
    FeedbackBatch,
    collect_output_feedback,
)


def test_get_frontier_mock_without_api_key():
    model = get_frontier(FrontierProvider.GEMINI)
    assert model.available


def test_blackbox_workflow_refines_total_with_mock_frontier():
    doc = generate_invoice(seed=20)
    frontier = MockFrontierModel()
    result = run_blackbox_workflow(
        doc.image,
        baseline_workflow_config(),
        frontier,
        ground_truth=doc.ground_truth,
        fallback_text=doc.rendered_text,
        run_config=BlackboxRunConfig(max_frontier_refine_rounds=2),
    )
    acc, _ = judge_fields(result.context.extracted_fields, doc.ground_truth, INVOICE_SCHEMA)
    assert acc >= 0.8


def test_propagate_via_frontier_patches_confusion():
    doc = generate_invoice(seed=21)
    cfg = baseline_workflow_config()
    from hillclimb.software.ocr_self_iterate.workflow.runner import run_workflow

    base_result = run_workflow(doc.image, doc.ground_truth, cfg, fallback_text=doc.rendered_text)
    items = collect_output_feedback(base_result, doc.ground_truth)
    assert items, "baseline should yield feedback"
    batch = FeedbackBatch(items=items, mean_accuracy=0.8)
    updated = propagate_via_frontier(cfg, batch, MockFrontierModel())
    pattern = updated.agents["extractor"].params.get("field_patterns", {}).get("total", "")
    assert "(?<!Sub)" in pattern or updated.orchestrator_params.get("max_refine_rounds", 0) > 0


def test_blackbox_offline_trainer_reaches_stable(tmp_path: Path):
    batch = [
        (generate_invoice(i).image, generate_invoice(i).ground_truth, generate_invoice(i).rendered_text)
        for i in range(4)
    ]
    from hillclimb.software.ocr_self_iterate.workflow.offline_train import OfflineTrainConfig

    trainer = BlackboxOfflineTrainer(
        documents=batch,
        frontier=MockFrontierModel(),
        train_config=OfflineTrainConfig(
            epochs=6,
            early_stop_accuracy=0.95,
            registry_path=tmp_path / "bb.json",
        ),
    )
    summary = trainer.train()
    assert summary["final_accuracy"] >= 0.95
    assert summary["frontier_provider"] == "MockFrontierModel"


def test_no_training_surface_in_blackbox_runner():
    """Black-box path must not expose train/fine-tune hooks."""
    import hillclimb.software.ocr_self_iterate.workflow.blackbox.runner as mod

    source = Path(mod.__file__).read_text()
    assert "fine_tune" not in source.lower()
    assert "backward()" not in source
    assert "gradient" not in source.lower()
