from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont


@dataclass
class SyntheticDocument:
    image: Image.Image
    ground_truth: dict[str, str]
    doc_type: str = "invoice"
    rendered_text: str = ""


def _load_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_invoice(
    seed: int | None = None,
    size: tuple[int, int] = (640, 480),
) -> SyntheticDocument:
    rng = random.Random(seed)
    invoice_number = f"INV-{rng.randint(1000, 9999)}"
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    vendor = rng.choice(["Acme Corp", "Globex LLC", "Initech", "Umbrella Co"])
    tax = round(rng.uniform(5.0, 40.0), 2)
    total = round(rng.uniform(100.0, 500.0), 2)

    ground_truth = {
        "invoice_number": invoice_number,
        "date": date,
        "vendor": vendor,
        "total": f"{total:.2f}",
        "tax": f"{tax:.2f}",
    }

    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(20)
    small = _load_font(16)

    lines = [
        ("INVOICE", font),
        (f"Invoice #: {invoice_number}", small),
        (f"Date: {date}", small),
        (f"Vendor: {vendor}", small),
        ("", small),
        (f"Subtotal: ${total - tax:.2f}", small),
        (f"Tax: ${tax:.2f}", small),
        (f"Total: ${total:.2f}", font),
    ]

    rendered_lines: list[str] = []
    y = 30
    for text, fnt in lines:
        if text:
            rendered_lines.append(text)
        draw.text((40, y), text, fill="black", font=fnt)
        y += 36 if fnt == font else 28

    return SyntheticDocument(
        image=img,
        ground_truth=ground_truth,
        doc_type="invoice",
        rendered_text="\n".join(rendered_lines),
    )


def generate_form(seed: int | None = None) -> SyntheticDocument:
    rng = random.Random(seed)
    ground_truth = {
        "name": rng.choice(["Jane Doe", "John Smith", "Alex Rivera"]),
        "email": rng.choice(["jane@example.com", "john@test.org", "alex@mail.io"]),
        "phone": rng.choice(["555-0100", "(555) 010-1234", "555-0199"]),
        "id_number": f"ID-{rng.randint(10000, 99999)}",
    }

    img = Image.new("RGB", (640, 400), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(18)

    y = 30
    rendered_lines: list[str] = []
    for label, value in [
        ("Name", ground_truth["name"]),
        ("Email", ground_truth["email"]),
        ("Phone", ground_truth["phone"]),
        ("ID Number", ground_truth["id_number"]),
    ]:
        line = f"{label}: {value}"
        rendered_lines.append(line)
        draw.text((40, y), line, fill="black", font=font)
        y += 40

    return SyntheticDocument(
        image=img,
        ground_truth=ground_truth,
        doc_type="form",
        rendered_text="\n".join(rendered_lines),
    )


def save_synthetic_batch(out_dir: Path, count: int = 5, seed: int = 0) -> list[SyntheticDocument]:
    out_dir.mkdir(parents=True, exist_ok=True)
    docs: list[SyntheticDocument] = []
    for i in range(count):
        doc = generate_invoice(seed=seed + i)
        path = out_dir / f"invoice_{i:02d}.png"
        doc.image.save(path)
        docs.append(doc)
    return docs
