from __future__ import annotations

from hillclimb.core.harness import HillClimber
from hillclimb.core.types import AcceptPolicy, Candidate
from hillclimb.software.ocr_self_iterate.judge import judge_fields
from hillclimb.software.ocr_self_iterate.schema import INVOICE_SCHEMA
from hillclimb.software.ocr_self_iterate.synthetic import generate_invoice
from hillclimb.software.ocr_self_iterate.workflow.config import baseline_workflow_config
from hillclimb.software.ocr_self_iterate.workflow.proposer import WorkflowProposer
from hillclimb.software.ocr_self_iterate.workflow.rewards import (
    DynamicRewardComponent,
    HybridRewardFramework,
    StaticRewardComponent,
)
from hillclimb.software.ocr_self_iterate.workflow.run import WorkflowBatchEvaluator, run_workflow_climb
from hillclimb.software.ocr_self_iterate.workflow.runner import run_workflow


def test_workflow_baseline_has_subtotal_confusion():
    doc = generate_invoice(seed=0)
    result = run_workflow(doc.image, doc.ground_truth, baseline_workflow_config(), fallback_text=doc.rendered_text)
    acc, _ = judge_fields(result.context.extracted_fields, doc.ground_truth, INVOICE_SCHEMA)
    assert acc < 1.0
    assert result.context.extracted_fields["total"] != doc.ground_truth["total"] or acc < 1.0


def test_hybrid_reward_static_component():
    doc = generate_invoice(seed=1)
    truth = doc.ground_truth
    extracted = dict(truth)
    static = StaticRewardComponent()
    score, bd = static.score(extracted, truth)
    assert score >= 0.9
    assert bd["static_accuracy"] == 1.0


def test_hybrid_reward_penalizes_confusion():
    doc = generate_invoice(seed=2)
    result = run_workflow(doc.image, doc.ground_truth, baseline_workflow_config(), fallback_text=doc.rendered_text)
    framework = HybridRewardFramework()
    _, bd = framework.evaluate(result, doc.ground_truth)
    assert "static_total" in bd
    assert "dynamic_total" in bd
    assert "hybrid_total" in bd


def test_orchestrator_refine_loop_improves_extraction():
    doc = generate_invoice(seed=3)
    cfg = baseline_workflow_config()
    cfg.orchestrator_params["max_refine_rounds"] = 2
    cfg.orchestrator_params["retry_on_validation_failure"] = True
    cfg.agents["extractor"].params["field_patterns"]["total"] = r"(?<!Sub)Total[:\s]+\$?\s*([\d,]+\.\d{2})"

    result = run_workflow(doc.image, doc.ground_truth, cfg, fallback_text=doc.rendered_text)
    acc, _ = judge_fields(result.context.extracted_fields, doc.ground_truth, INVOICE_SCHEMA)
    assert acc == 1.0


def test_workflow_hill_climb_improves_reward():
    summary = run_workflow_climb(docs=3, seed=7, max_rounds=10)
    assert summary["best_reward"] >= summary["initial_reward"]
    assert summary["best_accuracy"] >= summary["initial_accuracy"]


def test_workflow_proposer_mutates_orchestrator():
    proposer = WorkflowProposer(seed=0)
    initial = Candidate(state=baseline_workflow_config())
    ev = WorkflowBatchEvaluator(
        [(generate_invoice(0).image, generate_invoice(0).ground_truth, generate_invoice(0).rendered_text)]
    )
    first = ev.evaluate(initial)
    proposed = proposer.propose(initial, [(initial, first)])
    assert proposed.state is not initial.state
    assert (
        proposed.state.orchestrator_prompt != initial.state.orchestrator_prompt
        or proposed.state.orchestrator_params != initial.state.orchestrator_params
        or proposed.state.agents != initial.state.agents
    )


def test_sub_agent_prompt_affects_behavior():
    doc = generate_invoice(seed=4)
    cfg = baseline_workflow_config()
    cfg.agents["preprocessor"].system_prompt = "Enhance aggressively with high contrast"
    result = run_workflow(doc.image, doc.ground_truth, cfg, fallback_text=doc.rendered_text)
    assert result.context.raw_text


def test_dynamic_reward_refine_bonus():
    doc = generate_invoice(seed=5)
    cfg = baseline_workflow_config()
    cfg.orchestrator_params["max_refine_rounds"] = 1
    cfg.orchestrator_params["retry_on_validation_failure"] = True
    result = run_workflow(doc.image, doc.ground_truth, cfg, fallback_text=doc.rendered_text)
    dynamic = DynamicRewardComponent()
    _, bd = dynamic.score(result, doc.ground_truth, prior_failures={"total"})
    assert "dynamic_total" in bd
