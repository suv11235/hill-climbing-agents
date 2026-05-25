"""OpenAI LLM client for hill-climbing proposers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMClient:
    """Thin wrapper around the OpenAI chat completions API."""

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 2048
    _client: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai package required; install with: pip install 'hillclimb[llm]'"
                ) from exc
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
    ) -> str:
        if not self.available:
            raise RuntimeError("OPENAI_API_KEY not set")

        client = self._ensure_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Empty LLM response")
        return content.strip()

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        raw = self.complete(system, user, json_mode=True)
        return json.loads(raw)


def get_llm(model: str = "gpt-4o-mini") -> LLMClient | None:
    """Return an LLM client if OPENAI_API_KEY is configured, else None."""
    client = LLMClient(model=model)
    return client if client.available else None
