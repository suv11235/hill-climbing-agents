from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubAgentConfig:
    """
    Mutable sub-agent definition.

    `system_prompt` is the hill-climb surface (adaptive persona/instructions).
    `params` are structured knobs the prompt maps to at execution time.
    """

    name: str
    role: str
    system_prompt: str
    params: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> SubAgentConfig:
        return SubAgentConfig(
            name=self.name,
            role=self.role,
            system_prompt=self.system_prompt,
            params=copy.deepcopy(self.params),
        )


@dataclass
class WorkflowConfig:
    """Hill-climbing candidate: orchestrator + sub-agent fleet."""

    orchestrator_prompt: str
    orchestrator_params: dict[str, Any] = field(default_factory=dict)
    agents: dict[str, SubAgentConfig] = field(default_factory=dict)

    def copy(self) -> WorkflowConfig:
        return WorkflowConfig(
            orchestrator_prompt=self.orchestrator_prompt,
            orchestrator_params=copy.deepcopy(self.orchestrator_params),
            agents={k: v.copy() for k, v in self.agents.items()},
        )

    def get_agent(self, role: str) -> SubAgentConfig | None:
        for agent in self.agents.values():
            if agent.role == role:
                return agent
        return None


def baseline_workflow_config() -> WorkflowConfig:
    """
    Weak baseline workflow — mirrors the old single-parser limitations:
    naive total regex, minimal preprocessing, no refine loop by default.
    """
    return WorkflowConfig(
        orchestrator_prompt=(
            "You are a document OCR orchestrator. Run preprocess, layout, extract, "
            "validate once. Do not retry on failure."
        ),
        orchestrator_params={
            "max_refine_rounds": 0,
            "retry_on_validation_failure": False,
            "pipeline": ["preprocess", "layout", "extract", "validate"],
        },
        agents={
            "preprocessor": SubAgentConfig(
                name="preprocessor",
                role="preprocess",
                system_prompt=(
                    "Enhance invoice scans lightly. Prefer minimal scaling to avoid blur."
                ),
                params={"scale": 1.5, "contrast": 1.2, "threshold": 160, "sharpen": False},
            ),
            "layout": SubAgentConfig(
                name="layout",
                role="layout",
                system_prompt=(
                    "Segment document into header and body regions for field extraction."
                ),
                params={"strategy": "header_body", "header_ratio": 0.35},
            ),
            "extractor": SubAgentConfig(
                name="extractor",
                role="extract",
                system_prompt=(
                    "Extract invoice fields with simple line-anchored regex patterns."
                ),
                params={
                    "field_patterns": {
                        "invoice_number": r"Invoice\s*#:\s*([A-Z0-9-]+)",
                        "date": r"Date[:\s]+([\d/-]+)",
                        "vendor": r"Vendor[:\s]+([A-Za-z ]+)",
                        "total": r"Total[:\s]+\$?\s*([\d.]+)",
                        "tax": r"Tax[:\s]+\$?([\d.]+)",
                    },
                    "use_region_hints": False,
                },
            ),
            "validator": SubAgentConfig(
                name="validator",
                role="validate",
                system_prompt=(
                    "Validate required fields and basic numeric consistency."
                ),
                params={
                    "check_cross_field": False,
                    "required_fields": [
                        "invoice_number",
                        "date",
                        "vendor",
                        "total",
                        "tax",
                    ],
                },
            ),
            "refiner": SubAgentConfig(
                name="refiner",
                role="refine",
                system_prompt=(
                    "When validation fails, patch extractor patterns for missed fields. "
                    "Use negative lookbehind to avoid Subtotal/Total confusion."
                ),
                params={
                    "strategy": "pattern_library",
                    "pattern_library": {
                        "total": [
                            r"(?<!Sub)Total[:\s]+\$?\s*([\d,]+\.\d{2})",
                            r"Amount Due[:\s]+\$?\s*([\d,]+\.\d{2})",
                        ],
                        "tax": [
                            r"Tax[:\s]+\$?\s*([\d,]+\.\d{2})",
                            r"Sales Tax[:\s]+\$?\s*([\d.]+)",
                        ],
                    },
                },
            ),
        },
    )


def improved_workflow_config() -> WorkflowConfig:
    """Stronger starting point with refine loop enabled (oracle for tests)."""
    cfg = baseline_workflow_config()
    cfg.orchestrator_params["max_refine_rounds"] = 2
    cfg.orchestrator_params["retry_on_validation_failure"] = True
    cfg.agents["extractor"].params["field_patterns"]["total"] = (
        r"(?<!Sub)Total[:\s]+\$?\s*([\d,]+\.\d{2})"
    )
    cfg.agents["preprocessor"].params["scale"] = 2.0
    cfg.agents["validator"].params["check_cross_field"] = True
    return cfg
