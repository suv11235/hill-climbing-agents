from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from hillclimb.core.frontier import FrontierModel
from hillclimb.software.ocr_self_iterate.schema import DocumentSchema, INVOICE_SCHEMA
from hillclimb.software.ocr_self_iterate.workflow.config import WorkflowConfig
from hillclimb.software.ocr_self_iterate.workflow.feedback import (
    FeedbackBatch,
    propagate_feedback_backward,
)
from hillclimb.software.ocr_self_iterate.workflow.rewards import HybridRewardFramework
from hillclimb.software.ocr_self_iterate.workflow.runner import run_workflow
from hillclimb.software.ocr_self_iterate.workflow.state import AgentResult, WorkflowResult, WorkflowStage


@dataclass
class BlackboxRunConfig:
    """Runtime options for black-box workflow — no model training."""

    use_frontier_extract: bool = True
    use_frontier_validate: bool = True
    use_frontier_refine: bool = True
    max_frontier_refine_rounds: int = 2


def run_blackbox_workflow(
    image,
    config: WorkflowConfig,
    frontier: FrontierModel,
    *,
    ground_truth: dict[str, str] | None = None,
    fallback_text: str = "",
    schema: DocumentSchema = INVOICE_SCHEMA,
    run_config: BlackboxRunConfig | None = None,
    reward_framework: HybridRewardFramework | None = None,
) -> WorkflowResult:
    """
    Execute OCR workflow with black-box frontier calls at key stages.

    Rule-based pipeline handles preprocessing/layout; frontier model handles
    extract → validate → refine. Model weights are never updated.
    """
    rc = run_config or BlackboxRunConfig()
    framework = reward_framework or HybridRewardFramework()

    base = run_workflow(
        image,
        ground_truth or {},
        config,
        fallback_text=fallback_text,
        schema=schema,
        reward_framework=None,
    )
    ctx = base.context

    if rc.use_frontier_extract:
        ctx.extracted_fields = _frontier_extract(
            frontier, config, ctx, schema, fallback_text
        )

    if rc.use_frontier_validate:
        ctx.validation_errors = _frontier_validate(frontier, config, ctx, schema)

    if rc.use_frontier_refine and ctx.validation_errors:
        for _ in range(rc.max_frontier_refine_rounds):
            if not ctx.validation_errors:
                break
            ctx.extracted_fields = _frontier_refine(frontier, config, ctx, schema)
            ctx.refine_rounds += 1
            ctx.validation_errors = _frontier_validate(frontier, config, ctx, schema)

    result = WorkflowResult(context=ctx, agent_trace=list(base.agent_trace))
    result.agent_trace.append(
        AgentResult(
            agent_name="frontier",
            stage=WorkflowStage.EXTRACT,
            success=not ctx.validation_errors,
            message="blackbox frontier extract/validate/refine",
            diagnostics={"validation_errors": ctx.validation_errors},
        )
    )

    if ground_truth:
        total, breakdown = framework.evaluate(result, ground_truth)
        result.total_reward = total
        result.static_reward = breakdown.get("static_total", 0.0)
        result.dynamic_reward = breakdown.get("dynamic_total", 0.0)
        result.reward_breakdown = breakdown

    return result


def propagate_via_frontier(
    config: WorkflowConfig,
    batch: FeedbackBatch,
    frontier: FrontierModel,
) -> WorkflowConfig:
    """Black-box backward pass via frontier model; rule fallback on failure."""
    if not batch.items:
        return config

    feedback_payload = [
        {
            "field": fb.field,
            "error_kind": fb.error_kind.value,
            "expected": fb.expected,
            "actual": fb.actual,
            "message": fb.message,
            "responsible_agent": fb.responsible_agent,
            "raw_text_snippet": fb.raw_text_snippet,
        }
        for fb in batch.items[:12]
    ]

    system = (
        "You improve an OCR agentic workflow without training any model. "
        "Given output-level feedback from document parsing failures, return JSON "
        "patches to the workflow config only. Keys: "
        "orchestrator_params (max_refine_rounds int, retry_on_validation_failure bool), "
        "agent_updates (dict agent_name -> {system_prompt str, params dict}). "
        "For Subtotal/Total confusion set total pattern with (?<!Sub) negative lookbehind. "
        "Do not suggest model fine-tuning."
    )
    user = json.dumps(
        {
            "mean_accuracy": batch.mean_accuracy,
            "feedback": feedback_payload,
            "current_orchestrator_params": config.orchestrator_params,
            "agent_roles": {n: a.role for n, a in config.agents.items()},
        },
        indent=2,
    )

    try:
        patch = frontier.complete_json(system, user)
        updated = _apply_frontier_patch(config, patch)
    except Exception:
        updated = config.copy()

    # Always merge rule-based backward pass — no model training, dual black-box + deterministic
    if batch.items:
        updated = propagate_feedback_backward(updated, batch)
    return updated


def _frontier_extract(
    frontier: FrontierModel,
    config: WorkflowConfig,
    ctx: Any,
    schema: DocumentSchema,
    fallback_text: str,
) -> dict[str, str | None]:
    agent = config.agents.get("extractor")
    system = (
        f"{agent.system_prompt if agent else 'Extract structured fields.'}\n"
        'Return JSON: {"extracted_fields": {field: value|null}}. Inference only — no training.'
    )
    user = json.dumps(
        {
            "fields": schema.field_names,
            "ocr_text": ctx.raw_text or fallback_text,
            "fallback_text": fallback_text,
            "field_patterns": (
                config.agents.get("extractor", {}).params.get("field_patterns", {})
                if config.agents.get("extractor")
                else {}
            ),
        }
    )
    images = [ctx.preprocessed_image or ctx.image]
    try:
        result = frontier.complete_json(system, user, images=images)
        fields = result.get("extracted_fields", {})
        return {k: fields.get(k) for k in schema.field_names}
    except Exception:
        return ctx.extracted_fields


def _frontier_validate(
    frontier: FrontierModel,
    config: WorkflowConfig,
    ctx: Any,
    schema: DocumentSchema,
) -> list[dict[str, Any]]:
    agent = config.agents.get("validator")
    system = (
        f"{agent.system_prompt if agent else 'Validate extracted fields.'}\n"
        'Return JSON: {"validation_errors": [...], "passed": bool}. '
        "Each error: field, type, message."
    )
    user = json.dumps(
        {
            "extracted_fields": ctx.extracted_fields,
            "ocr_text": ctx.raw_text,
            "required_fields": schema.field_names,
        }
    )
    try:
        result = frontier.complete_json(system, user)
        return list(result.get("validation_errors", []))
    except Exception:
        return ctx.validation_errors


def _frontier_refine(
    frontier: FrontierModel,
    config: WorkflowConfig,
    ctx: Any,
    schema: DocumentSchema,
) -> dict[str, str | None]:
    agent = config.agents.get("refiner")
    system = (
        f"{agent.system_prompt if agent else 'Refine extracted fields.'}\n"
        'Return JSON: {"extracted_fields": {...}} with corrected values.'
    )
    user = json.dumps(
        {
            "extracted_fields": ctx.extracted_fields,
            "validation_errors": ctx.validation_errors,
            "ocr_text": ctx.raw_text,
        }
    )
    try:
        result = frontier.complete_json(system, user, images=[ctx.image])
        fields = result.get("extracted_fields", ctx.extracted_fields)
        return {k: fields.get(k, ctx.extracted_fields.get(k)) for k in schema.field_names}
    except Exception:
        return ctx.extracted_fields


def _apply_frontier_patch(config: WorkflowConfig, patch: dict[str, Any]) -> WorkflowConfig:
    updated = config.copy()
    if orch := patch.get("orchestrator_params"):
        updated.orchestrator_params.update(orch)
    for agent_name, agent_patch in patch.get("agent_updates", {}).items():
        if agent_name not in updated.agents:
            continue
        agent = updated.agents[agent_name]
        if prompt := agent_patch.get("system_prompt"):
            agent.system_prompt = prompt
        if params := agent_patch.get("params"):
            merged = dict(agent.params)
            for key, val in params.items():
                if key == "field_patterns" and isinstance(val, dict):
                    merged["field_patterns"] = {
                        **agent.params.get("field_patterns", {}),
                        **val,
                    }
                else:
                    merged[key] = val
            agent.params = merged
    return updated
