from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hillclimb.software.ocr_self_iterate.workflow.config import WorkflowConfig
from hillclimb.software.ocr_self_iterate.workflow.state import WorkflowResult


class ErrorKind(str, Enum):
    MISSING = "missing"
    MISMATCH = "mismatch"
    CONFUSION = "confusion"
    INCONSISTENT = "inconsistent"
    LOW_OCR_QUALITY = "low_ocr_quality"


# Minimal static hints — field names only, not regex templates.
FIELD_LABEL_HINTS: dict[str, list[str]] = {
    "invoice_number": ["Invoice #", "Invoice No", "Invoice Number"],
    "date": ["Date"],
    "vendor": ["Vendor", "From", "Bill From"],
    "total": ["Total", "Amount Due"],
    "tax": ["Tax", "Sales Tax"],
    "name": ["Name"],
    "email": ["Email"],
    "phone": ["Phone"],
    "id_number": ["ID", "ID Number"],
}


@dataclass
class OutputFeedback:
    """Output-level signal propagated backward to a responsible agent."""

    field: str
    error_kind: ErrorKind
    expected: str | None
    actual: str | None
    message: str
    responsible_agent: str
    raw_text_snippet: str = ""
    refine_helped: bool = False
    reward_delta: float = 0.0


@dataclass
class FeedbackBatch:
    """Aggregated feedback from one offline training epoch."""

    items: list[OutputFeedback] = field(default_factory=list)
    mean_accuracy: float = 0.0
    mean_reward: float = 0.0
    refine_success_rate: float = 0.0

    def by_agent(self) -> dict[str, list[OutputFeedback]]:
        grouped: dict[str, list[OutputFeedback]] = {}
        for item in self.items:
            grouped.setdefault(item.responsible_agent, []).append(item)
        return grouped

    def by_field(self) -> dict[str, list[OutputFeedback]]:
        grouped: dict[str, list[OutputFeedback]] = {}
        for item in self.items:
            grouped.setdefault(item.field, []).append(item)
        return grouped


def collect_output_feedback(
    result: WorkflowResult,
    ground_truth: dict[str, str],
) -> list[OutputFeedback]:
    """
    Harvest output-level feedback from a single workflow run.

    No manual prompt engineering — errors are attributed to agents via trace analysis.
    """
    ctx = result.context
    feedback: list[OutputFeedback] = []
    raw_text = ctx.raw_text or ""

    refine_ran = any(r.stage.value == "refine" for r in result.agent_trace)
    pre_refine_errors = _errors_before_refine(result)

    for name, expected in ground_truth.items():
        actual = ctx.extracted_fields.get(name)
        normalized_actual = _norm(actual)
        normalized_expected = _norm(expected)

        if normalized_actual == normalized_expected:
            continue

        error_kind, message = _classify_error(name, actual, expected, ctx.validation_errors, raw_text)
        agent = _attribute_agent(error_kind, name, result, pre_refine_errors, refine_ran)

        feedback.append(
            OutputFeedback(
                field=name,
                error_kind=error_kind,
                expected=expected,
                actual=actual,
                message=message,
                responsible_agent=agent,
                raw_text_snippet=_snippet_for_field(raw_text, name),
                refine_helped=refine_ran and name not in pre_refine_errors,
            )
        )

    if len(raw_text.strip()) < 8:
        feedback.append(
            OutputFeedback(
                field="_ocr",
                error_kind=ErrorKind.LOW_OCR_QUALITY,
                expected=None,
                actual=None,
                message="OCR text too short",
                responsible_agent="preprocessor",
                raw_text_snippet=raw_text[:80],
            )
        )

    return feedback


def propagate_feedback_backward(
    config: WorkflowConfig,
    batch: FeedbackBatch,
) -> WorkflowConfig:
    """
    Apply output feedback backward through the workflow graph.

    Updates agent params (and only synthesizes prompts when params change).
    Minimal manual engineering: patterns derived from failure examples in raw text.
    """
    updated = config.copy()
    grouped = batch.by_agent()

    if "preprocessor" in grouped or batch.mean_accuracy < 0.85:
        _update_preprocessor(updated, grouped.get("preprocessor", []))

    if "extractor" in grouped:
        _update_extractor(updated, grouped["extractor"])

    if "validator" in grouped:
        _update_validator(updated, grouped["validator"])

    if "refiner" in grouped or batch.refine_success_rate > 0.3:
        _update_refiner(updated, grouped.get("refiner", []))

    if batch.refine_success_rate > 0.2 or any(
        f.error_kind == ErrorKind.CONFUSION for f in batch.items
    ):
        _update_orchestrator_for_retry(updated, batch)

    _sync_prompts_from_params(updated)
    return updated


def _classify_error(
    field: str,
    actual: str | None,
    expected: str,
    validation_errors: list[dict],
    raw_text: str,
) -> tuple[ErrorKind, str]:
    for err in validation_errors:
        if err.get("field") == field and err.get("type") == "confusion":
            return ErrorKind.CONFUSION, err.get("message", "subtotal/total confusion")

    if actual is None or str(actual).strip() == "":
        return ErrorKind.MISSING, f"missing {field}"

    if field == "total" and _find_subtotal_value(raw_text) is not None:
        if _norm(actual) == _norm(str(_find_subtotal_value(raw_text))):
            return ErrorKind.CONFUSION, "extracted subtotal as total"

    return ErrorKind.MISMATCH, f"got '{actual}' expected '{expected}'"


def _attribute_agent(
    error_kind: ErrorKind,
    field: str,
    result: WorkflowResult,
    pre_refine_errors: set[str],
    refine_ran: bool,
) -> str:
    if error_kind == ErrorKind.LOW_OCR_QUALITY:
        return "preprocessor"
    if error_kind in (ErrorKind.CONFUSION, ErrorKind.MISSING, ErrorKind.MISMATCH):
        if refine_ran and field in pre_refine_errors:
            return "refiner"
        return "extractor"
    if error_kind == ErrorKind.INCONSISTENT:
        return "validator"
    return "extractor"


def _errors_before_refine(result: WorkflowResult) -> set[str]:
    errors: set[str] = set()
    for step in result.agent_trace:
        if step.stage.value == "refine":
            break
        if step.stage.value == "validate":
            for err in step.diagnostics.get("errors", []):
                if field := err.get("field"):
                    errors.add(field)
    return errors


def _update_preprocessor(config: WorkflowConfig, items: list[OutputFeedback]) -> None:
    agent = config.agents.get("preprocessor")
    if not agent:
        return
    params = agent.params
    if any(i.error_kind == ErrorKind.LOW_OCR_QUALITY for i in items):
        params["scale"] = min(3.0, float(params.get("scale", 1.5)) + 0.25)
        params["contrast"] = min(2.0, float(params.get("contrast", 1.2)) + 0.15)
    elif items:
        params["scale"] = min(2.5, float(params.get("scale", 1.5)) + 0.1)


def _update_extractor(config: WorkflowConfig, items: list[OutputFeedback]) -> None:
    agent = config.agents.get("extractor")
    if not agent:
        return
    patterns = dict(agent.params.get("field_patterns", {}))

    for fb in items:
        if fb.field == "_ocr":
            continue
        new_pattern = derive_pattern_from_feedback(fb)
        if new_pattern:
            patterns[fb.field] = new_pattern

    agent.params["field_patterns"] = patterns
    if any(fb.error_kind == ErrorKind.CONFUSION for fb in items):
        agent.params["use_region_hints"] = True


def _update_validator(config: WorkflowConfig, items: list[OutputFeedback]) -> None:
    agent = config.agents.get("validator")
    if not agent:
        return
    agent.params["check_cross_field"] = True
    if any(i.error_kind == ErrorKind.CONFUSION for i in items):
        agent.params["flag_subtotal_confusion"] = True


def _update_refiner(config: WorkflowConfig, items: list[OutputFeedback]) -> None:
    agent = config.agents.get("refiner")
    if not agent:
        return
    library = dict(agent.params.get("pattern_library", {}))
    for fb in items:
        pattern = derive_pattern_from_feedback(fb)
        if pattern:
            library.setdefault(fb.field, [])
            if pattern not in library[fb.field]:
                library[fb.field].insert(0, pattern)
    agent.params["pattern_library"] = library


def _update_orchestrator_for_retry(config: WorkflowConfig, batch: FeedbackBatch) -> None:
    params = config.orchestrator_params
    current = int(params.get("max_refine_rounds", 0))
    if batch.mean_accuracy < 0.95:
        params["max_refine_rounds"] = min(3, current + 1)
        params["retry_on_validation_failure"] = True
    elif batch.mean_accuracy >= 0.98 and batch.refine_success_rate < 0.1:
        params["max_refine_rounds"] = max(0, current - 1)
        params["retry_on_validation_failure"] = params["max_refine_rounds"] > 0


def _sync_prompts_from_params(config: WorkflowConfig) -> None:
    """
    Derive concise agent prompts from params — avoids hand-authored prompt libraries.
    Prompts are summaries of learned behavior, not the primary control surface.
    """
    orch = config.orchestrator_params
    max_refine = int(orch.get("max_refine_rounds", 0))
    if max_refine > 0:
        config.orchestrator_prompt = (
            f"Orchestrate OCR pipeline; on validation failure retry refiner up to {max_refine} rounds."
        )
    else:
        config.orchestrator_prompt = "Orchestrate OCR pipeline single-pass."

    ext = config.agents.get("extractor")
    if ext:
        n = len(ext.params.get("field_patterns", {}))
        flags = []
        if ext.params.get("use_region_hints"):
            flags.append("region-aware")
        if any("(?<!Sub)" in p for p in ext.params.get("field_patterns", {}).values()):
            flags.append("subtotal-safe")
        ext.system_prompt = f"Extract {n} fields ({', '.join(flags) or 'baseline'})."

    val = config.agents.get("validator")
    if val and val.params.get("check_cross_field"):
        val.system_prompt = "Validate fields with cross-field consistency checks."

    pre = config.agents.get("preprocessor")
    if pre:
        pre.system_prompt = (
            f"Preprocess scans scale={pre.params.get('scale')} contrast={pre.params.get('contrast')}."
        )


def derive_pattern_from_feedback(fb: OutputFeedback) -> str | None:
    """Derive a regex pattern from output failure + raw text snippet — no template library."""
    if fb.error_kind == ErrorKind.CONFUSION and fb.field == "total":
        return r"(?<!Sub)Total[:\s]+\$?\s*([\d,]+\.\d{2})"

    labels = FIELD_LABEL_HINTS.get(fb.field, [fb.field.replace("_", " ").title()])
    snippet = fb.raw_text_snippet or ""

    for label in labels:
        escaped = re.escape(label)
        if fb.expected and re.search(r"\d", fb.expected):
            pattern = rf"{escaped}[:\s]+\$?\s*([\d,]+\.?\d*)"
        else:
            pattern = rf"{escaped}[:\s]+([A-Za-z0-9 @._-]+)"
        if re.search(pattern, snippet, re.IGNORECASE):
            return pattern

    if fb.expected:
        label = labels[0]
        if re.search(r"\d", fb.expected):
            return rf"{re.escape(label)}[:\s]+\$?\s*([\d,]+\.?\d*)"
        return rf"{re.escape(label)}[:\s]+(\S+)"

    return None


def _snippet_for_field(raw_text: str, field: str) -> str:
    labels = FIELD_LABEL_HINTS.get(field, [])
    for line in raw_text.splitlines():
        lower = line.lower()
        if any(lbl.lower() in lower for lbl in labels):
            return line.strip()
    return raw_text[:200]


def _find_subtotal_value(raw_text: str) -> float | None:
    match = re.search(r"Subtotal[:\s]+\$?\s*([\d,.]+)", raw_text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def _norm(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())
