"""
Black-box frontier model interface — no weight training, API-only inference.

Supports Gemini (primary) and OpenAI (fallback). Used for OCR workflow agents
that hill-climb over prompts/config at test time only.
"""

from __future__ import annotations

import io
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from PIL import Image


class FrontierProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    MOCK = "mock"


class FrontierModel(Protocol):
    """Black-box model: complete / complete_json, optionally with vision."""

    @property
    def available(self) -> bool: ...

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        images: list[Image.Image] | None = None,
    ) -> dict[str, Any]: ...


@dataclass
class GeminiClient:
    """Google Gemini via google-genai SDK — no fine-tuning, inference only."""

    model: str = "gemini-2.0-flash"
    api_key: str | None = None
    temperature: float = 0.1
    _client: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self.api_key = (
            self.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai required; install with: pip install 'hillclimb[gemini]'"
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        images: list[Image.Image] | None = None,
    ) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set")

        from google.genai import types

        client = self._ensure_client()
        parts: list[Any] = [types.Part.from_text(text=user)]
        for img in images or []:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))

        response = client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                response_mime_type="application/json",
            ),
        )
        text = response.text
        if not text:
            raise RuntimeError("Empty Gemini response")
        return json.loads(text.strip())


@dataclass
class OpenAIFrontierClient:
    """OpenAI as frontier fallback (text-only path)."""

    model: str = "gpt-4o-mini"
    api_key: str | None = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        images: list[Image.Image] | None = None,
    ) -> dict[str, Any]:
        from hillclimb.core.llm import LLMClient

        client = LLMClient(model=self.model, api_key=self.api_key)
        if images:
            raise RuntimeError("OpenAI frontier fallback is text-only; use Gemini for vision")
        return client.complete_json(system, user)


@dataclass
class MockFrontierModel:
    """Deterministic mock for tests — no API, no training."""

    @property
    def available(self) -> bool:
        return True

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        images: list[Image.Image] | None = None,
    ) -> dict[str, Any]:
        payload = json.loads(user) if user.strip().startswith("{") else {"raw": user}

        if "extract" in system.lower() or "fields" in system.lower():
            text = payload.get("ocr_text") or payload.get("fallback_text") or ""
            custom = payload.get("field_patterns") or {}
            return _mock_extract(text, payload.get("fields", []), custom)

        if "validate" in system.lower():
            return _mock_validate(payload.get("extracted_fields", {}), payload.get("ocr_text", ""))

        if "refine" in system.lower() or "patch" in system.lower():
            return _mock_refine(payload)

        if "workflow" in system.lower() or "backward" in system.lower():
            return _mock_workflow_patch(payload)

        if "orchestrat" in system.lower():
            return {"max_refine_rounds": 2, "retry_on_validation_failure": True}

        return {}


def _mock_extract(
    text: str, fields: list[str], custom_patterns: dict[str, str] | None = None
) -> dict[str, Any]:
    import re

    result: dict[str, str | None] = {}
    defaults = {
        "invoice_number": r"Invoice\s*#?:?\s*([A-Z0-9-]+)",
        "date": r"Date[:\s]+([\d-]+)",
        "vendor": r"Vendor[:\s]+([A-Za-z0-9 &]+?)(?:\s*$|\s*\n)",
        "total": r"(?<!Sub)Total[:\s]+\$?\s*([\d,]+\.\d{2})",
        "tax": r"Tax[:\s]+\$?\s*([\d,]+\.\d{2})",
    }
    patterns = {**defaults, **(custom_patterns or {})}
    for f in fields or list(patterns.keys()):
        pat = patterns.get(f, "")
        m = re.search(pat, text, re.I | re.M) if pat else None
        result[f] = m.group(1).strip() if m else None
    return {"extracted_fields": result}


def _mock_validate(extracted: dict, text: str) -> dict[str, Any]:
    errors = []
    total = extracted.get("total")
    if total and "Subtotal" in text:
        import re

        sub = re.search(r"Subtotal[:\s]+\$?\s*([\d.]+)", text, re.I)
        if sub and sub.group(1) == str(total):
            errors.append(
                {"field": "total", "type": "confusion", "message": "matched subtotal as total"}
            )
    return {"validation_errors": errors, "passed": len(errors) == 0}


def _mock_refine(payload: dict) -> dict[str, Any]:
    import re

    text = payload.get("ocr_text", "")
    fields = dict(payload.get("extracted_fields", {}))
    m = re.search(r"(?<!Sub)Total[:\s]+\$?\s*([\d.]+)", text, re.I)
    if m:
        fields["total"] = m.group(1)
    return {"extracted_fields": fields, "patches_applied": ["total_regex_fix"]}


def _mock_workflow_patch(payload: dict) -> dict[str, Any]:
    feedback = payload.get("feedback", [])
    patch: dict[str, Any] = {
        "orchestrator_params": {
            "max_refine_rounds": 2,
            "retry_on_validation_failure": True,
        },
        "agent_updates": {
            "extractor": {
                "params": {
                    "field_patterns": {
                        "total": r"(?<!Sub)Total[:\s]+\$?\s*([\d,]+\.\d{2})"
                    }
                }
            }
        },
    }
    if not feedback:
        patch["orchestrator_params"] = {"max_refine_rounds": 0, "retry_on_validation_failure": False}
    return patch


def get_frontier(
    provider: FrontierProvider | str = FrontierProvider.GEMINI,
    model: str | None = None,
) -> FrontierModel:
    """
    Resolve a black-box frontier model. No training — inference-only API calls.

    Priority when provider=gemini: Gemini → OpenAI text fallback → Mock.
    """
    if isinstance(provider, str):
        provider = FrontierProvider(provider)

    if provider == FrontierProvider.MOCK:
        return MockFrontierModel()

    if provider == FrontierProvider.GEMINI:
        gemini = GeminiClient(model=model or "gemini-2.0-flash")
        if gemini.available:
            return gemini
        openai = OpenAIFrontierClient(model=model or "gpt-4o-mini")
        if openai.available:
            return openai
        return MockFrontierModel()

    if provider == FrontierProvider.OPENAI:
        openai = OpenAIFrontierClient(model=model or "gpt-4o-mini")
        if openai.available:
            return openai
        return MockFrontierModel()

    return MockFrontierModel()
