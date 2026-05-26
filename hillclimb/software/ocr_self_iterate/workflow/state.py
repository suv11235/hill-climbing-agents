from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from PIL import Image


class WorkflowStage(str, Enum):
    INGEST = "ingest"
    PREPROCESS = "preprocess"
    LAYOUT = "layout"
    EXTRACT = "extract"
    VALIDATE = "validate"
    REFINE = "refine"
    DONE = "done"


@dataclass
class DocumentContext:
    """Mutable document state as it moves through the workflow."""

    image: Image.Image
    ground_truth: dict[str, str]
    fallback_text: str = ""
    doc_type: str = "invoice"
    raw_text: str = ""
    preprocessed_image: Image.Image | None = None
    layout_regions: list[dict[str, Any]] = field(default_factory=list)
    extracted_fields: dict[str, str | None] = field(default_factory=dict)
    validation_errors: list[dict[str, Any]] = field(default_factory=list)
    refine_rounds: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentResult:
    """Output from a sub-agent invocation."""

    agent_name: str
    stage: WorkflowStage
    success: bool
    message: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """End-to-end workflow output for one document."""

    context: DocumentContext
    agent_trace: list[AgentResult] = field(default_factory=list)
    static_reward: float = 0.0
    dynamic_reward: float = 0.0
    total_reward: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
