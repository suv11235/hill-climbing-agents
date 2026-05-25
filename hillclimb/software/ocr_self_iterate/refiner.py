from __future__ import annotations

import copy
import random

from hillclimb.core.types import Candidate, Evaluation
from hillclimb.software.ocr_self_iterate.parser import ParserConfig


# Improved regex templates keyed by field name.
PATTERN_LIBRARY: dict[str, list[str]] = {
    "invoice_number": [
        r"Invoice\s*#?\s*:?\s*([A-Z0-9-]+)",
        r"Invoice\s*No\.?\s*([A-Z0-9-]+)",
        r"#\s*([A-Z0-9-]{4,})",
    ],
    "date": [
        r"Date[:\s]+(\d{4}-\d{2}-\d{2})",
        r"Date[:\s]+([\d/-]+)",
        r"(\d{4}-\d{2}-\d{2})",
    ],
    "vendor": [
        r"Vendor[:\s]+([A-Za-z0-9 &]+)",
        r"From[:\s]+([A-Za-z0-9 &]+)",
        r"Bill From[:\s]+([A-Za-z0-9 &]+)",
    ],
    "total": [
        r"(?<!Sub)Total[:\s]+\$?\s*([\d,]+\.\d{2})",
        r"Amount Due[:\s]+\$?\s*([\d,]+\.\d{2})",
        r"TOTAL[:\s]+\$?\s*([\d.]+)",
    ],
    "tax": [
        r"Tax[:\s]+\$?\s*([\d,]+\.\d{2})",
        r"Sales Tax[:\s]+\$?\s*([\d.]+)",
        r"TAX[:\s]+\$?\s*([\d.]+)",
    ],
    "name": [r"Name[:\s]+([A-Za-z .'-]+)", r"Full Name[:\s]+([A-Za-z .'-]+)"],
    "email": [r"Email[:\s]+([\w.+-]+@[\w.-]+\.[A-Za-z]{2,})"],
    "phone": [r"Phone[:\s]+([\d\-()+ ]{7,})"],
    "id_number": [r"ID(?: Number)?[:\s#]+([A-Z0-9-]+)"],
}


class OCRRefiner:
    """Proposes parser-config mutations from field-level judge diagnostics."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        config: ParserConfig = copy.deepcopy(current.state)
        diagnostics = history[-1][1].diagnostics if history else {}
        field_diags = diagnostics.get("field_diagnostics", [])

        missed = [d for d in field_diags if not d.get("correct", False)]
        if missed:
            target = self._rng.choice(missed)["field"]
            self._improve_field_pattern(config, target)
        else:
            self._mutate_preprocessing(config)

        return Candidate(state=config, metadata={"source": "ocr_refiner"})

    def _improve_field_pattern(self, config: ParserConfig, field: str) -> None:
        options = PATTERN_LIBRARY.get(field, [])
        if options:
            current = config.field_patterns.get(field, "")
            candidates = [p for p in options if p != current] or options
            config.field_patterns[field] = self._rng.choice(candidates)
        else:
            config.field_patterns[field] = config.field_patterns.get(field, r"(\w+)")

    def _mutate_preprocessing(self, config: ParserConfig) -> None:
        choice = self._rng.choice(
            ["scale_up", "scale_down", "contrast_up", "threshold", "sharpen", "invert"]
        )
        if choice == "scale_up":
            config.scale = min(3.5, config.scale + 0.25)
        elif choice == "scale_down":
            config.scale = max(1.0, config.scale - 0.25)
        elif choice == "contrast_up":
            config.contrast = min(2.5, config.contrast + 0.2)
        elif choice == "threshold":
            config.threshold = int(max(100, min(220, config.threshold + self._rng.randint(-15, 15))))
        elif choice == "sharpen":
            config.sharpen = not config.sharpen
        elif choice == "invert":
            config.invert = not config.invert
