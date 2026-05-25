from __future__ import annotations

from hillclimb.core.harness import HillClimber
from hillclimb.core.types import AcceptPolicy, Candidate
from hillclimb.software.ocr_self_iterate.judge import field_accuracy, judge_fields
from hillclimb.software.ocr_self_iterate.parser import (
    ParserConfig,
    baseline_parser_config,
    parse_document,
)
from hillclimb.software.ocr_self_iterate.refiner import OCRRefiner
from hillclimb.software.ocr_self_iterate.run import OCRBatchEvaluator
from hillclimb.software.ocr_self_iterate.schema import FORM_SCHEMA, INVOICE_SCHEMA
from hillclimb.software.ocr_self_iterate.synthetic import generate_form, generate_invoice


SAMPLE_INVOICE_TEXT = """
INVOICE
Invoice #: INV-1234
Date: 2026-05-01
Vendor: Acme Corp

Subtotal: $100.00
Tax: $8.50
Total: $108.50
"""

SAMPLE_FORM_TEXT = """
Name: Jane Doe
Email: jane@example.com
Phone: 555-0100
ID Number: ID-54321
"""


def test_judge_fields_perfect_match():
    truth = {
        "invoice_number": "INV-1234",
        "date": "2026-05-01",
        "vendor": "Acme Corp",
        "total": "108.50",
        "tax": "8.50",
    }
    score, diags = judge_fields(truth, truth, INVOICE_SCHEMA)
    assert score == 1.0
    assert all(d.correct for d in diags)


def test_field_accuracy_partial():
    parsed = {"invoice_number": "INV-1", "date": None, "vendor": "X", "total": "1", "tax": "0"}
    truth = {"invoice_number": "INV-1", "date": "2026-01-01", "vendor": "X", "total": "1", "tax": "0"}
    assert field_accuracy(parsed, truth, INVOICE_SCHEMA) == 0.8


def test_parse_document_with_injected_text():
    config = baseline_parser_config()
    config.field_patterns["invoice_number"] = r"Invoice\s+#\s*:?\s*([A-Z0-9-]+)"
    doc = generate_invoice(seed=0)
    parsed = parse_document(doc.image, config, schema=INVOICE_SCHEMA, ocr_text=SAMPLE_INVOICE_TEXT)
    assert parsed["invoice_number"] == "INV-1234"
    assert parsed["vendor"] == "Acme Corp"


def test_parse_form_fields():
    config = baseline_parser_config(doc_type=FORM_SCHEMA.doc_type)
    parsed = parse_document(
        generate_form(seed=1).image,
        config,
        schema=FORM_SCHEMA,
        ocr_text=SAMPLE_FORM_TEXT,
    )
    assert parsed["name"] == "Jane Doe"
    assert parsed["email"] == "jane@example.com"


def test_hill_climb_improves_parser_config():
    docs = [generate_invoice(seed=i) for i in range(3)]
    batch = [(d.image, d.ground_truth) for d in docs]

    # Inject OCR text so tests do not depend on tesseract binary.
    def evaluate_with_text(candidate: Candidate):
        config: ParserConfig = candidate.state
        scores = []
        field_diags = []
        for image, truth in batch:
            text = (
                f"INVOICE\nInvoice #: {truth['invoice_number']}\n"
                f"Date: {truth['date']}\nVendor: {truth['vendor']}\n"
                f"Tax: ${truth['tax']}\nTotal: ${truth['total']}\n"
            )
            parsed = parse_document(image, config, schema=INVOICE_SCHEMA, ocr_text=text)
            score, diags = judge_fields(parsed, truth, INVOICE_SCHEMA)
            scores.append(score)
            field_diags.extend({"field": d.field, "correct": d.correct} for d in diags)

        from hillclimb.core.types import Evaluation

        mean = sum(scores) / len(scores)
        return Evaluation(
            score=mean,
            diagnostics={"field_diagnostics": [d for d in field_diags if not d["correct"]]},
        )

    class _Eval:
        evaluate = staticmethod(evaluate_with_text)

    climber = HillClimber(
        proposer=OCRRefiner(seed=5),
        evaluator=_Eval(),
        max_rounds=8,
        early_stop_patience=3,
        accept_policy=AcceptPolicy.GREEDY,
    )
    initial = Candidate(state=baseline_parser_config())
    initial_score = evaluate_with_text(initial).score
    result = climber.climb(initial)
    assert result.best_score >= initial_score


def test_synthetic_invoice_ground_truth_keys():
    doc = generate_invoice(seed=9)
    assert set(doc.ground_truth.keys()) == set(INVOICE_SCHEMA.field_names)
