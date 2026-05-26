from __future__ import annotations

from dataclasses import dataclass, field

from hillclimb.software.ocr_self_iterate.judge import field_accuracy, judge_fields
from hillclimb.software.ocr_self_iterate.schema import DocumentSchema, INVOICE_SCHEMA
from hillclimb.software.ocr_self_iterate.workflow.state import WorkflowResult


@dataclass
class StaticRewardComponent:
    """
    Fixed, ground-truth-based rewards — stable across hill-climb rounds.

    - field accuracy vs labeled data
    - required field coverage
    - format compliance (numeric fields parseable)
    """

    schema: DocumentSchema = INVOICE_SCHEMA
    accuracy_weight: float = 0.55
    coverage_weight: float = 0.25
    format_weight: float = 0.20

    def score(
        self,
        extracted: dict[str, str | None],
        ground_truth: dict[str, str],
    ) -> tuple[float, dict[str, float]]:
        accuracy = field_accuracy(extracted, ground_truth, self.schema)
        required = self.schema.field_names
        present = sum(1 for n in required if extracted.get(n) not in (None, ""))
        coverage = present / len(required) if required else 0.0

        numeric_fields = {"total", "tax"}
        format_ok = 0
        format_total = 0
        for name in numeric_fields:
            if name not in ground_truth:
                continue
            format_total += 1
            val = extracted.get(name)
            if val and _is_numeric(val):
                format_ok += 1
        format_score = format_ok / format_total if format_total else 1.0

        total = (
            self.accuracy_weight * accuracy
            + self.coverage_weight * coverage
            + self.format_weight * format_score
        )
        breakdown = {
            "static_accuracy": accuracy,
            "static_coverage": coverage,
            "static_format": format_score,
            "static_total": total,
        }
        return total, breakdown


@dataclass
class DynamicRewardComponent:
    """
    Diagnostic-driven rewards that shift as the workflow learns.

    - confusion penalties (Subtotal/Total)
    - cross-field consistency bonus
    - refinement efficiency (reward fixing errors, penalize excess loops)
    - validation clearance bonus
    """

    confusion_penalty: float = 0.15
    consistency_bonus: float = 0.10
    refine_efficiency_weight: float = 0.08
    validation_clear_bonus: float = 0.07

    def score(
        self,
        result: WorkflowResult,
        ground_truth: dict[str, str],
        prior_failures: set[str] | None = None,
    ) -> tuple[float, dict[str, float]]:
        ctx = result.context
        extracted = ctx.extracted_fields
        prior_failures = prior_failures or set()

        breakdown: dict[str, float] = {}
        dynamic = 0.0

        confusion = 0.0
        for err in ctx.validation_errors:
            if err.get("type") == "confusion":
                confusion += self.confusion_penalty
        breakdown["dynamic_confusion_penalty"] = -confusion
        dynamic -= confusion

        consistency = 0.0
        total = _parse_amount(extracted.get("total"))
        tax = _parse_amount(extracted.get("tax"))
        gt_total = _parse_amount(ground_truth.get("total"))
        if total is not None and tax is not None and total >= tax:
            consistency += self.consistency_bonus * 0.5
        if total is not None and gt_total is not None and abs(total - gt_total) < 0.02:
            consistency += self.consistency_bonus * 0.5
        breakdown["dynamic_consistency"] = consistency
        dynamic += consistency

        fixed_fields = 0
        _, diags = judge_fields(extracted, ground_truth, INVOICE_SCHEMA)
        current_failures = {d.field for d in diags if not d.correct}
        for field in prior_failures:
            if field not in current_failures:
                fixed_fields += 1
        refine_bonus = fixed_fields * self.refine_efficiency_weight
        loop_penalty = max(0, ctx.refine_rounds - 1) * 0.03
        breakdown["dynamic_refine_fix_bonus"] = refine_bonus
        breakdown["dynamic_refine_loop_penalty"] = -loop_penalty
        dynamic += refine_bonus - loop_penalty

        if not ctx.validation_errors:
            breakdown["dynamic_validation_clear"] = self.validation_clear_bonus
            dynamic += self.validation_clear_bonus
        else:
            breakdown["dynamic_validation_clear"] = 0.0

        breakdown["dynamic_total"] = dynamic
        return dynamic, breakdown


@dataclass
class HybridRewardFramework:
    """
    Hybrid static + dynamic reward for workflow-level hill climbing.

    total = static_weight * static + dynamic_weight * dynamic
    """

    static: StaticRewardComponent = field(default_factory=StaticRewardComponent)
    dynamic: DynamicRewardComponent = field(default_factory=DynamicRewardComponent)
    static_weight: float = 0.72
    dynamic_weight: float = 0.28

    def evaluate(
        self,
        result: WorkflowResult,
        ground_truth: dict[str, str],
        prior_failures: set[str] | None = None,
    ) -> tuple[float, dict[str, float]]:
        static_score, static_bd = self.static.score(result.context.extracted_fields, ground_truth)
        dynamic_score, dynamic_bd = self.dynamic.score(result, ground_truth, prior_failures)

        total = self.static_weight * static_score + self.dynamic_weight * dynamic_score
        breakdown = {**static_bd, **dynamic_bd, "hybrid_total": total}
        return total, breakdown


def _parse_amount(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _is_numeric(value: str) -> bool:
    try:
        float(str(value).replace(",", "").replace("$", "").strip())
        return True
    except ValueError:
        return False
