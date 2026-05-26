from __future__ import annotations

import copy
import random

from hillclimb.core.types import Candidate, Evaluation
from hillclimb.software.ocr_self_iterate.workflow.config import WorkflowConfig


PROMPT_MUTATIONS: dict[str, list[str]] = {
    "orchestrator": [
        "You are a document OCR orchestrator. On validation failure, invoke refiner up to 2 rounds.",
        "You are a document OCR orchestrator. Retry with refiner when total/subtotal confusion detected.",
        "You are a document OCR orchestrator. Run preprocess, layout, extract, validate once.",
    ],
    "preprocess": [
        "Enhance invoice scans aggressively with high contrast and scaling.",
        "Enhance invoice scans lightly. Prefer minimal scaling to avoid blur.",
        "Apply strong binarization for faint scans; sharpen edges.",
    ],
    "layout": [
        "Segment document into header and body regions for field extraction.",
        "Use line-by-line layout to isolate Total and Tax lines.",
        "Treat full page as single region for robust regex matching.",
    ],
    "extract": [
        "Extract invoice fields with simple line-anchored regex patterns.",
        "Extract fields using negative lookbehind to avoid Subtotal/Total confusion.",
        "Use region-aware extraction with layout hints for totals.",
    ],
    "validate": [
        "Validate required fields and basic numeric consistency.",
        "Validate required fields with cross-field consistency checks.",
        "Strict validation: flag total/subtotal confusion and total < tax.",
    ],
    "refine": [
        "Patch extractor patterns from pattern library when validation fails.",
        "When validation fails, use negative lookbehind for Total field fixes.",
        "Iteratively refine regex patterns for missed numeric fields.",
    ],
}

PARAM_MUTATIONS: dict[str, dict[str, list]] = {
    "orchestrator": {
        "max_refine_rounds": [0, 1, 2, 3],
        "retry_on_validation_failure": [False, True],
    },
    "preprocess": {
        "scale": [1.5, 2.0, 2.5],
        "contrast": [1.2, 1.5, 1.8],
        "threshold": [140, 160, 180],
        "sharpen": [False, True],
    },
    "extract": {
        "use_region_hints": [False, True],
    },
    "validate": {
        "check_cross_field": [False, True],
    },
}


class WorkflowProposer:
    """
    Hill-climb proposer at workflow level.

    Mutates orchestrator prompt/params and sub-agent system prompts based on
    hybrid reward diagnostics from the previous evaluation.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        config: WorkflowConfig = copy.deepcopy(current.state)
        diagnostics = history[-1][1].diagnostics if history else {}

        target = self._select_mutation_target(diagnostics)
        if target == "orchestrator":
            self._mutate_orchestrator(config)
        elif target in config.agents:
            self._mutate_agent(config.agents[target])
        else:
            self._mutate_orchestrator(config)

        return Candidate(state=config, metadata={"mutation_target": target})

    def _select_mutation_target(self, diagnostics: dict) -> str:
        confusion = diagnostics.get("confusion_count", 0)
        missing = diagnostics.get("missing_field_count", 0)
        low_accuracy = diagnostics.get("mean_field_accuracy", 1.0) < 0.9
        no_refine = diagnostics.get("avg_refine_rounds", 0) == 0 and low_accuracy

        if confusion > 0:
            return self._rng.choice(["extractor", "refiner", "validator"])
        if missing > 0:
            return self._rng.choice(["extractor", "refiner", "preprocessor"])
        if no_refine:
            return "orchestrator"
        if low_accuracy:
            return self._rng.choice(["preprocessor", "layout", "extractor"])
        return self._rng.choice(["orchestrator", "preprocessor", "extractor", "validator", "refiner"])

    def _mutate_orchestrator(self, config: WorkflowConfig) -> None:
        options = PROMPT_MUTATIONS["orchestrator"]
        config.orchestrator_prompt = self._rng.choice(
            [p for p in options if p != config.orchestrator_prompt] or options
        )
        params = config.orchestrator_params
        if "max_refine_rounds" in PARAM_MUTATIONS["orchestrator"]:
            params["max_refine_rounds"] = self._rng.choice(
                PARAM_MUTATIONS["orchestrator"]["max_refine_rounds"]
            )
        params["retry_on_validation_failure"] = params.get("max_refine_rounds", 0) > 0

    def _mutate_agent(self, agent) -> None:
        role_key = agent.role if agent.role in PROMPT_MUTATIONS else agent.name
        prompts = PROMPT_MUTATIONS.get(role_key, PROMPT_MUTATIONS.get(agent.name, []))
        if prompts:
            agent.system_prompt = self._rng.choice(
                [p for p in prompts if p != agent.system_prompt] or prompts
            )

        param_space = PARAM_MUTATIONS.get(agent.role, {})
        for key, choices in param_space.items():
            agent.params[key] = self._rng.choice(choices)

        if agent.role == "extract":
            patterns = dict(agent.params.get("field_patterns", {}))
            if "total" in patterns and "(?<!Sub)" not in patterns["total"]:
                patterns["total"] = r"(?<!Sub)Total[:\s]+\$?\s*([\d,]+\.\d{2})"
                agent.params["field_patterns"] = patterns
            agent.params["use_region_hints"] = "region" in agent.system_prompt.lower()

        if agent.role == "validate":
            agent.params["check_cross_field"] = (
                "cross-field" in agent.system_prompt.lower()
                or "consistency" in agent.system_prompt.lower()
                or "strict" in agent.system_prompt.lower()
            )

        if agent.role == "preprocess":
            if "aggressive" in agent.system_prompt.lower():
                agent.params["scale"] = max(float(agent.params.get("scale", 1.5)), 2.0)
                agent.params["contrast"] = max(float(agent.params.get("contrast", 1.2)), 1.6)
