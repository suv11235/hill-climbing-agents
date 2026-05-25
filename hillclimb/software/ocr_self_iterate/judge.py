from __future__ import annotations

from dataclasses import dataclass

from hillclimb.software.ocr_self_iterate.schema import DocumentSchema


@dataclass
class FieldDiagnostic:
    field: str
    expected: str | None
    actual: str | None
    correct: bool
    message: str


def normalize_value(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def field_accuracy(
    parsed: dict[str, str | None],
    ground_truth: dict[str, str],
    schema: DocumentSchema,
) -> float:
    if not schema.field_names:
        return 0.0
    correct = sum(
        1
        for name in schema.field_names
        if normalize_value(parsed.get(name)) == normalize_value(ground_truth.get(name))
    )
    return correct / len(schema.field_names)


def judge_fields(
    parsed: dict[str, str | None],
    ground_truth: dict[str, str],
    schema: DocumentSchema,
) -> tuple[float, list[FieldDiagnostic]]:
    """Compare parsed output to ground truth; return score and diagnostics."""
    diagnostics: list[FieldDiagnostic] = []
    for field_spec in schema.fields:
        name = field_spec.name
        expected = ground_truth.get(name)
        actual = parsed.get(name)
        ok = normalize_value(actual) == normalize_value(expected)
        if ok:
            msg = "match"
        elif actual is None:
            msg = "missing extraction"
        else:
            msg = f"mismatch: got '{actual}' expected '{expected}'"
        diagnostics.append(
            FieldDiagnostic(
                field=name,
                expected=expected,
                actual=actual,
                correct=ok,
                message=msg,
            )
        )

    score = field_accuracy(parsed, ground_truth, schema)
    return score, diagnostics
