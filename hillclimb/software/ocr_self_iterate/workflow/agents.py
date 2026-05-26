from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from hillclimb.software.ocr_self_iterate.schema import INVOICE_SCHEMA, DocumentSchema
from hillclimb.software.ocr_self_iterate.workflow.config import SubAgentConfig
from hillclimb.software.ocr_self_iterate.workflow.state import AgentResult, DocumentContext, WorkflowStage


def _ocr_text(image: Image.Image, fallback: str = "") -> str:
    try:
        import pytesseract

        text = pytesseract.image_to_string(image).strip()
        if len(text) >= 8:
            return text
    except Exception:
        pass
    return fallback


def _apply_prompt_hints(params: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Map prompt keywords to param nudges (adaptive prompt → behavior)."""
    out = dict(params)
    lower = prompt.lower()
    if "aggressive" in lower or "high contrast" in lower:
        out["scale"] = max(float(out.get("scale", 1.5)), 2.0)
        out["contrast"] = max(float(out.get("contrast", 1.2)), 1.8)
    if "negative lookbehind" in lower or "subtotal" in lower:
        patterns = out.get("field_patterns", {})
        if "total" in patterns and "(?<!Sub)" not in patterns["total"]:
            patterns = dict(patterns)
            patterns["total"] = r"(?<!Sub)Total[:\s]+\$?\s*([\d,]+\.\d{2})"
            out["field_patterns"] = patterns
    if "cross-field" in lower or "consistency" in lower:
        out["check_cross_field"] = True
    if "region" in lower or "layout" in lower:
        out["use_region_hints"] = True
    return out


class SubAgent(ABC):
    def __init__(self, config: SubAgentConfig) -> None:
        self.config = config

    @abstractmethod
    def run(self, ctx: DocumentContext, schema: DocumentSchema) -> AgentResult: ...

    def effective_params(self) -> dict[str, Any]:
        return _apply_prompt_hints(self.config.params, self.config.system_prompt)


class PreprocessorAgent(SubAgent):
    def run(self, ctx: DocumentContext, schema: DocumentSchema) -> AgentResult:
        p = self.effective_params()
        img = ctx.image.convert("L")
        scale = float(p.get("scale", 1.5))
        if scale != 1.0:
            w, h = img.size
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        if p.get("invert"):
            img = ImageOps.invert(img)
        img = ImageEnhance.Contrast(img).enhance(float(p.get("contrast", 1.2)))
        if p.get("sharpen"):
            img = img.filter(ImageFilter.SHARPEN)
        threshold = int(p.get("threshold", 160))
        img = img.point(lambda px: 255 if px > threshold else 0)
        ctx.preprocessed_image = img
        text = _ocr_text(img, fallback=ctx.fallback_text)
        ctx.raw_text = text
        return AgentResult(
            agent_name=self.config.name,
            stage=WorkflowStage.PREPROCESS,
            success=True,
            message=f"preprocessed scale={scale} threshold={threshold}",
            artifacts={"text_len": len(text)},
            diagnostics={"scale": scale, "contrast": p.get("contrast"), "threshold": threshold},
        )


class LayoutAgent(SubAgent):
    def run(self, ctx: DocumentContext, schema: DocumentSchema) -> AgentResult:
        p = self.effective_params()
        strategy = p.get("strategy", "full_page")
        lines = [ln for ln in ctx.raw_text.splitlines() if ln.strip()]
        regions: list[dict[str, Any]] = []

        if strategy == "header_body":
            split = max(1, int(len(lines) * float(p.get("header_ratio", 0.35))))
            regions = [
                {"name": "header", "lines": lines[:split]},
                {"name": "body", "lines": lines[split:]},
            ]
        elif strategy == "line_by_line":
            regions = [{"name": f"line_{i}", "lines": [ln]} for i, ln in enumerate(lines)]
        else:
            regions = [{"name": "full_page", "lines": lines}]

        ctx.layout_regions = regions
        return AgentResult(
            agent_name=self.config.name,
            stage=WorkflowStage.LAYOUT,
            success=True,
            message=f"layout strategy={strategy} regions={len(regions)}",
            artifacts={"regions": len(regions)},
            diagnostics={"strategy": strategy},
        )


class ExtractorAgent(SubAgent):
    def run(self, ctx: DocumentContext, schema: DocumentSchema) -> AgentResult:
        p = self.effective_params()
        patterns: dict[str, str] = dict(p.get("field_patterns", {}))
        use_regions = bool(p.get("use_region_hints", False))
        text_sources = [ctx.raw_text]
        if use_regions and ctx.layout_regions:
            text_sources.extend("\n".join(r["lines"]) for r in ctx.layout_regions)

        extracted: dict[str, str | None] = {}
        for field_spec in schema.fields:
            pattern = patterns.get(field_spec.name, "")
            value = None
            if pattern:
                for source in text_sources:
                    match = re.search(pattern, source, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        break
            extracted[field_spec.name] = value

        ctx.extracted_fields = extracted
        missing = [k for k, v in extracted.items() if v is None]
        return AgentResult(
            agent_name=self.config.name,
            stage=WorkflowStage.EXTRACT,
            success=len(missing) == 0,
            message=f"extracted {len(extracted) - len(missing)}/{len(extracted)} fields",
            artifacts={"fields": extracted},
            diagnostics={"missing_fields": missing, "patterns_used": list(patterns.keys())},
        )


class ValidatorAgent(SubAgent):
    def run(self, ctx: DocumentContext, schema: DocumentSchema) -> AgentResult:
        p = self.effective_params()
        required = p.get("required_fields", schema.field_names)
        errors: list[dict[str, Any]] = []

        for name in required:
            val = ctx.extracted_fields.get(name)
            if val is None or str(val).strip() == "":
                errors.append({"field": name, "type": "missing", "message": "required field missing"})

        if p.get("check_cross_field"):
            total = _parse_amount(ctx.extracted_fields.get("total"))
            tax = _parse_amount(ctx.extracted_fields.get("tax"))
            subtotal_hint = _find_subtotal(ctx.raw_text)
            if total is not None and tax is not None and total < tax:
                errors.append(
                    {"field": "total", "type": "inconsistent", "message": "total < tax"}
                )
            if total is not None and subtotal_hint is not None and abs(total - subtotal_hint) < 0.01:
                errors.append(
                    {
                        "field": "total",
                        "type": "confusion",
                        "message": "total matches subtotal — likely Subtotal/Total regex bug",
                    }
                )

        ctx.validation_errors = errors
        return AgentResult(
            agent_name=self.config.name,
            stage=WorkflowStage.VALIDATE,
            success=len(errors) == 0,
            message=f"validation {'passed' if not errors else 'failed'} ({len(errors)} issues)",
            artifacts={"errors": errors},
            diagnostics={"error_count": len(errors), "errors": errors[:5]},
        )


class RefinerAgent(SubAgent):
    """Patch extractor patterns / preprocessing based on validation diagnostics."""

    def run(self, ctx: DocumentContext, schema: DocumentSchema) -> AgentResult:
        p = self.effective_params()
        strategy = p.get("strategy", "pattern_library")
        library: dict[str, list[str]] = p.get("pattern_library", {})
        patches: list[str] = []

        extractor_cfg = None
        for err in ctx.validation_errors:
            field = err.get("field")
            if not field:
                continue
            if strategy == "pattern_library" and field in library:
                options = library[field]
                current = ctx.extracted_fields.get(field)
                for pattern in options:
                    match = re.search(pattern, ctx.raw_text, re.IGNORECASE)
                    if match:
                        val = match.group(1).strip()
                        if val != current:
                            ctx.extracted_fields[field] = val
                            patches.append(f"{field}: {pattern}")
                            break

        ctx.refine_rounds += 1
        ctx.validation_errors = [
            e
            for e in ctx.validation_errors
            if ctx.extracted_fields.get(e.get("field")) is None
            or (
                e.get("type") == "confusion"
                and _parse_amount(ctx.extracted_fields.get("total")) is not None
            )
        ]

        return AgentResult(
            agent_name=self.config.name,
            stage=WorkflowStage.REFINE,
            success=len(patches) > 0,
            message=f"refine applied {len(patches)} patches",
            artifacts={"patches": patches},
            diagnostics={"strategy": strategy, "refine_round": ctx.refine_rounds},
        )


def _parse_amount(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _find_subtotal(text: str) -> float | None:
    match = re.search(r"Subtotal[:\s]+\$?\s*([\d,.]+)", text, re.IGNORECASE)
    if match:
        return _parse_amount(match.group(1))
    return None


AGENT_CLASSES: dict[str, type[SubAgent]] = {
    "preprocess": PreprocessorAgent,
    "layout": LayoutAgent,
    "extract": ExtractorAgent,
    "validate": ValidatorAgent,
    "refine": RefinerAgent,
}


def build_agent(config: SubAgentConfig) -> SubAgent:
    cls = AGENT_CLASSES.get(config.role)
    if cls is None:
        raise ValueError(f"Unknown agent role: {config.role}")
    return cls(config)
