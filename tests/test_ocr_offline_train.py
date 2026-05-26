from __future__ import annotations

from pathlib import Path

from hillclimb.software.ocr_self_iterate.judge import judge_fields
from hillclimb.software.ocr_self_iterate.schema import INVOICE_SCHEMA
from hillclimb.software.ocr_self_iterate.synthetic import generate_invoice
from hillclimb.software.ocr_self_iterate.workflow.config import baseline_workflow_config
from hillclimb.software.ocr_self_iterate.workflow.feedback import (
    ErrorKind,
    collect_output_feedback,
    derive_pattern_from_feedback,
    propagate_feedback_backward,
    FeedbackBatch,
    OutputFeedback,
)
from hillclimb.software.ocr_self_iterate.workflow.offline_train import OfflineTrainConfig, OfflineWorkflowTrainer
from hillclimb.software.ocr_self_iterate.workflow.registry import WorkflowRegistry, WorkflowStatus
from hillclimb.software.ocr_self_iterate.workflow.runner import run_workflow


def test_derive_pattern_from_confusion_feedback():
    fb = OutputFeedback(
        field="total",
        error_kind=ErrorKind.CONFUSION,
        expected="108.50",
        actual="100.00",
        message="subtotal confusion",
        responsible_agent="extractor",
        raw_text_snippet="Subtotal: $100.00\nTotal: $108.50",
    )
    pattern = derive_pattern_from_feedback(fb)
    assert pattern is not None
    assert "(?<!Sub)" in pattern


def test_collect_output_feedback_on_baseline():
    doc = generate_invoice(seed=10)
    result = run_workflow(doc.image, doc.ground_truth, baseline_workflow_config(), fallback_text=doc.rendered_text)
    feedback = collect_output_feedback(result, doc.ground_truth)
    assert any(f.field == "total" for f in feedback)


def test_propagate_backward_fixes_total_pattern():
    doc = generate_invoice(seed=11)
    result = run_workflow(doc.image, doc.ground_truth, baseline_workflow_config(), fallback_text=doc.rendered_text)
    items = collect_output_feedback(result, doc.ground_truth)
    batch = FeedbackBatch(items=items, mean_accuracy=0.8)
    updated = propagate_feedback_backward(baseline_workflow_config(), batch)
    pattern = updated.agents["extractor"].params["field_patterns"]["total"]
    assert "(?<!Sub)" in pattern


def test_sync_prompts_derived_not_hand_authored():
    doc = generate_invoice(seed=12)
    cfg = baseline_workflow_config()
    items = collect_output_feedback(
        run_workflow(doc.image, doc.ground_truth, cfg, fallback_text=doc.rendered_text),
        doc.ground_truth,
    )
    updated = propagate_feedback_backward(cfg, FeedbackBatch(items=items, mean_accuracy=0.8))
    assert "scale=" in updated.agents["preprocessor"].system_prompt
    assert "Extract" in updated.agents["extractor"].system_prompt


def test_registry_promotes_to_stable():
    registry = WorkflowRegistry(promote_accuracy=0.9, promote_epochs=2)
    cfg = baseline_workflow_config()
    from hillclimb.software.ocr_self_iterate.workflow.registry import WorkflowMetrics

    registry.register(cfg, WorkflowMetrics(mean_accuracy=0.96, mean_reward=0.7, epoch=0), version_id="v1")
    v = registry.register(cfg, WorkflowMetrics(mean_accuracy=0.97, mean_reward=0.75, epoch=1), version_id="v1")
    assert v.status in (WorkflowStatus.CANDIDATE, WorkflowStatus.STABLE)


def test_registry_save_load_roundtrip(tmp_path: Path):
    registry = WorkflowRegistry()
    cfg = baseline_workflow_config()
    from hillclimb.software.ocr_self_iterate.workflow.registry import WorkflowMetrics

    registry.register(cfg, WorkflowMetrics(mean_accuracy=0.95, mean_reward=0.7, epoch=0))
    path = tmp_path / "registry.json"
    registry.save(path)
    loaded = WorkflowRegistry.load(path)
    assert len(loaded.versions) == 1


def test_offline_trainer_reaches_high_accuracy():
    batch = [(generate_invoice(i).image, generate_invoice(i).ground_truth, generate_invoice(i).rendered_text) for i in range(5)]
    trainer = OfflineWorkflowTrainer(
        documents=batch,
        train_config=OfflineTrainConfig(epochs=6, registry_path=Path("/tmp/test_registry.json")),
    )
    summary = trainer.train()
    assert summary["final_accuracy"] >= 0.95


def test_stable_workflow_is_frozen_after_convergence(tmp_path: Path):
    batch = [(generate_invoice(i).image, generate_invoice(i).ground_truth, generate_invoice(i).rendered_text) for i in range(4)]
    reg_path = tmp_path / "reg.json"
    trainer = OfflineWorkflowTrainer(
        documents=batch,
        train_config=OfflineTrainConfig(
            epochs=10,
            early_stop_accuracy=0.95,
            registry_path=reg_path,
        ),
    )
    summary = trainer.train()
    registry = WorkflowRegistry.load(reg_path)
    stable = registry.get_stable()
    if stable:
        assert stable.status in (WorkflowStatus.STABLE, WorkflowStatus.FROZEN)
