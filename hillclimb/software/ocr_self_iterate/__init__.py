"""OCR self-iteration via hill climbing (MinerU / OCR-Agent inspired)."""

from hillclimb.software.ocr_self_iterate.parser import ParserConfig, parse_document
from hillclimb.software.ocr_self_iterate.judge import judge_fields, field_accuracy
from hillclimb.software.ocr_self_iterate.refiner import OCRRefiner
from hillclimb.software.ocr_self_iterate.schema import DocumentSchema, INVOICE_SCHEMA, FORM_SCHEMA
from hillclimb.software.ocr_self_iterate.workflow import (
    HybridRewardFramework,
    OrchestratorAgent,
    SubAgentConfig,
    WorkflowConfig,
    run_workflow,
)

__all__ = [
    "ParserConfig",
    "parse_document",
    "judge_fields",
    "field_accuracy",
    "OCRRefiner",
    "DocumentSchema",
    "INVOICE_SCHEMA",
    "FORM_SCHEMA",
    "SubAgentConfig",
    "WorkflowConfig",
    "OrchestratorAgent",
    "HybridRewardFramework",
    "run_workflow",
]
