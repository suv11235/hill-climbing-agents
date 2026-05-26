from __future__ import annotations

from dataclasses import dataclass

from hillclimb.software.ocr_self_iterate.schema import DocumentSchema, INVOICE_SCHEMA
from hillclimb.software.ocr_self_iterate.workflow.agents import build_agent
from hillclimb.software.ocr_self_iterate.workflow.config import WorkflowConfig
from hillclimb.software.ocr_self_iterate.workflow.state import (
    AgentResult,
    DocumentContext,
    WorkflowResult,
    WorkflowStage,
)


@dataclass
class OrchestratorAgent:
    """
    Workflow orchestrator: routes documents through mutable sub-agents.

    The orchestrator's system prompt and params control retry policy, pipeline
    ordering, and refine-loop behavior — all hill-climbable surfaces.
    """

    config: WorkflowConfig
    schema: DocumentSchema = INVOICE_SCHEMA

    def run(self, ctx: DocumentContext) -> WorkflowResult:
        trace: list[AgentResult] = []
        pipeline: list[str] = list(
            self.config.orchestrator_params.get(
                "pipeline", ["preprocess", "layout", "extract", "validate"]
            )
        )
        max_refine = int(self.config.orchestrator_params.get("max_refine_rounds", 0))
        retry = bool(self.config.orchestrator_params.get("retry_on_validation_failure", False))

        ctx.trace.append(
            {
                "stage": WorkflowStage.INGEST.value,
                "orchestrator_prompt": self.config.orchestrator_prompt[:120],
            }
        )

        for stage_name in pipeline:
            agent_cfg = self.config.get_agent(stage_name)
            if agent_cfg is None:
                continue
            agent = build_agent(agent_cfg)
            result = agent.run(ctx, self.schema)
            trace.append(result)
            ctx.trace.append(
                {
                    "stage": stage_name,
                    "agent": agent_cfg.name,
                    "success": result.success,
                    "message": result.message,
                }
            )

        validation_failed = any(r.stage == WorkflowStage.VALIDATE and not r.success for r in trace)

        if retry and validation_failed and max_refine > 0:
            refiner_cfg = self.config.get_agent("refine")
            if refiner_cfg:
                for _ in range(max_refine):
                    if not ctx.validation_errors:
                        break
                    refiner = build_agent(refiner_cfg)
                    refine_result = refiner.run(ctx, self.schema)
                    trace.append(refine_result)

                    validator_cfg = self.config.get_agent("validate")
                    if validator_cfg:
                        validator = build_agent(validator_cfg)
                        val_result = validator.run(ctx, self.schema)
                        trace.append(val_result)
                        if val_result.success:
                            break

        return WorkflowResult(context=ctx, agent_trace=trace)
