from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from hillclimb.software.ocr_self_iterate.schema import DocumentSchema, DocumentType, INVOICE_SCHEMA


@dataclass
class ParserConfig:
    """Candidate state: regex patterns + OCR preprocessing knobs."""

    doc_type: DocumentType = DocumentType.INVOICE
    field_patterns: dict[str, str] = field(default_factory=dict)
    scale: float = 2.0
    contrast: float = 1.5
    threshold: int = 160
    sharpen: bool = False
    invert: bool = False

    def copy(self) -> ParserConfig:
        return ParserConfig(
            doc_type=self.doc_type,
            field_patterns=dict(self.field_patterns),
            scale=self.scale,
            contrast=self.contrast,
            threshold=self.threshold,
            sharpen=self.sharpen,
            invert=self.invert,
        )


def baseline_parser_config(doc_type: DocumentType = DocumentType.INVOICE) -> ParserConfig:
    """Weak regex patterns that miss common synthetic layouts."""
    if doc_type == DocumentType.INVOICE:
        patterns = {
            "invoice_number": r"Invoice\s*#:\s*([A-Z0-9-]+)",
            "date": r"Date[:\s]+([\d/-]+)",
            "vendor": r"Vendor[:\s]+([A-Za-z ]+)",
            "total": r"Total[:\s]+\$?\s*([\d.]+)",
            "tax": r"Tax[:\s]+\$?([\d.]+)",
        }
    else:
        patterns = {
            "name": r"Name[:\s]+([A-Za-z ]+)",
            "email": r"Email[:\s]+([\w@.+-]+)",
            "phone": r"Phone[:\s]+([\d\-() ]+)",
            "id_number": r"ID[:\s#]+(\w+)",
        }
    return ParserConfig(doc_type=doc_type, field_patterns=patterns, scale=1.5, contrast=1.2)


def _preprocess(image: Image.Image, config: ParserConfig) -> Image.Image:
    img = image.convert("L")
    if config.scale != 1.0:
        w, h = img.size
        img = img.resize((int(w * config.scale), int(h * config.scale)), Image.Resampling.LANCZOS)
    if config.invert:
        img = ImageOps.invert(img)
    img = ImageEnhance.Contrast(img).enhance(config.contrast)
    if config.sharpen:
        img = img.filter(ImageFilter.SHARPEN)
    img = img.point(lambda p: 255 if p > config.threshold else 0)
    return img


def _ocr_text(image: Image.Image, config: ParserConfig, fallback_text: str = "") -> str:
    processed = _preprocess(image, config)
    text = ""
    try:
        import pytesseract

        text = pytesseract.image_to_string(processed)
    except Exception:
        text = ""

    cleaned = text.strip()
    if len(cleaned) < 8 and fallback_text:
        return fallback_text
    return text


def parse_document(
    image: Image.Image,
    config: ParserConfig,
    schema: DocumentSchema | None = None,
    ocr_text: str | None = None,
    fallback_text: str = "",
) -> dict[str, str | None]:
    """Extract structured fields from a document image."""
    schema = schema or INVOICE_SCHEMA
    if ocr_text is None:
        ocr_text = _ocr_text(image, config, fallback_text=fallback_text)
    normalized = " ".join(ocr_text.split())

    parsed: dict[str, str | None] = {}
    for field_spec in schema.fields:
        pattern = config.field_patterns.get(field_spec.name, "")
        if not pattern:
            parsed[field_spec.name] = None
            continue
        match = re.search(pattern, ocr_text, re.IGNORECASE) or re.search(
            pattern, normalized, re.IGNORECASE
        )
        parsed[field_spec.name] = match.group(1).strip() if match else None

    return parsed


def config_summary(config: ParserConfig) -> dict[str, Any]:
    return {
        "scale": config.scale,
        "contrast": config.contrast,
        "threshold": config.threshold,
        "sharpen": config.sharpen,
        "patterns": list(config.field_patterns.keys()),
    }
