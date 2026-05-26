from __future__ import annotations

from hillclimb.software.ocr_self_iterate.schema import DocumentSchema, INVOICE_SCHEMA
from hillclimb.software.ocr_self_iterate.workflow.config import WorkflowConfig
from hillclimb.software.ocr_self_iterate.workflow.orchestrator import OrchestratorAgent
from hillclimb.software.ocr_self_iterate.workflow.rewards import HybridRewardFramework
from hillclimb.software.ocr_self_iterate.workflow.state import DocumentContext, WorkflowResult


def run_workflow(
    image,
    ground_truth: dict[str, str],
    config: WorkflowConfig,
    *,
    fallback_text: str = "",
    schema: DocumentSchema = INVOICE_SCHEMA,
    reward_framework: HybridRewardFramework | None = None,
    prior_failures: set[str] | None = None,
) -> WorkflowResult:
    """Execute orchestrated OCR workflow on a single document."""
    ctx = DocumentContext(
        image=image,
        ground_truth=ground_truth,
        fallback_text=fallback_text,
        doc_type=schema.doc_type.value,
    )
    orchestrator = OrchestratorAgent(config=config, schema=schema)
    result = orchestrator.run(ctx)

    framework = reward_framework or HybridRewardFramework()
    total, breakdown = framework.evaluate(result, ground_truth, prior_failures)
    result.static_reward = breakdown.get("static_total", 0.0)
    result.dynamic_reward = breakdown.get("dynamic_total", 0.0)
    result.total_reward = total
    result.reward_breakdown = breakdown
    return result
