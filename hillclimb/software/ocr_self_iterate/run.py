from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from hillclimb.core.harness import HillClimber
from hillclimb.core.types import AcceptPolicy, Candidate, Evaluation
from hillclimb.software.ocr_self_iterate.judge import judge_fields
from hillclimb.software.ocr_self_iterate.parser import (
    ParserConfig,
    baseline_parser_config,
    parse_document,
)
from hillclimb.software.ocr_self_iterate.refiner import OCRRefiner
from hillclimb.software.ocr_self_iterate.schema import INVOICE_SCHEMA
from hillclimb.software.ocr_self_iterate.synthetic import generate_invoice, save_synthetic_batch


class OCRBatchEvaluator:
    """Scores parser configs by mean field accuracy on synthetic invoices."""

    def __init__(
        self,
        documents: list[tuple[object, dict[str, str], str]],
        seed: int = 0,
    ) -> None:
        self.documents = documents
        self.seed = seed

    def evaluate(self, candidate: Candidate) -> Evaluation:
        config: ParserConfig = candidate.state
        scores: list[float] = []
        all_field_diags: list[dict] = []

        for image, truth, fallback_text in self.documents:
            parsed = parse_document(
                image,
                config,
                schema=INVOICE_SCHEMA,
                fallback_text=fallback_text,
            )
            score, diags = judge_fields(parsed, truth, INVOICE_SCHEMA)
            scores.append(score)
            all_field_diags.extend(asdict(d) for d in diags)

        mean_score = sum(scores) / len(scores) if scores else 0.0
        missed = [d for d in all_field_diags if not d["correct"]]

        return Evaluation(
            score=mean_score,
            diagnostics={
                "mean_field_accuracy": mean_score,
                "per_doc_scores": scores,
                "field_diagnostics": missed[:10],
                "num_missed_fields": len(missed),
            },
            passed=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-iterate OCR parsing rules on synthetic documents."
    )
    parser.add_argument("--rounds", type=int, default=10, help="Hill-climbing rounds.")
    parser.add_argument("--docs", type=int, default=4, help="Synthetic invoices in eval batch.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Optional directory to write synthetic PNGs.",
    )
    args = parser.parse_args()

    if args.save_dir:
        docs = save_synthetic_batch(Path(args.save_dir), args.docs, args.seed)
        batch = [(d.image, d.ground_truth, d.rendered_text) for d in docs]
    else:
        batch = []
        for i in range(args.docs):
            doc = generate_invoice(seed=args.seed + i)
            batch.append((doc.image, doc.ground_truth, doc.rendered_text))

    evaluator = OCRBatchEvaluator(batch, seed=args.seed)
    refiner = OCRRefiner(seed=args.seed)
    climber = HillClimber(
        proposer=refiner,
        evaluator=evaluator,
        max_rounds=args.rounds,
        early_stop_patience=3,
        accept_policy=AcceptPolicy.GREEDY,
    )

    initial = Candidate(state=baseline_parser_config())
    result = climber.climb(initial)

    print("OCR self-iteration (synthetic invoices)")
    for i, (_, ev) in enumerate(result.history):
        print(
            f"round {i:02d}: field_accuracy={ev.score:.3f} "
            f"missed={ev.diagnostics.get('num_missed_fields', '?')}"
        )
    print(f"best field accuracy: {result.best_score:.3f}")
    best: ParserConfig = result.best.state
    print(f"best patterns: {best.field_patterns}")
    print(f"preprocess: scale={best.scale} contrast={best.contrast} threshold={best.threshold}")


if __name__ == "__main__":
    main()
