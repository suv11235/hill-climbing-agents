"""Diagnostic-aware hyperparameter mutations."""

from __future__ import annotations

import copy
import random
from typing import Any

from hillclimb.core.types import Candidate, Evaluation
from hillclimb.software.config_discovery.search_space import (
    BOUNDS,
    CRITERION_CHOICES,
    MAX_FEATURES_CHOICES,
    clamp_config,
)


def _mutate_n_estimators(config: dict[str, Any], delta: int) -> None:
    lo, hi = BOUNDS["n_estimators"]
    config["n_estimators"] = int(max(lo, min(hi, config["n_estimators"] + delta)))


def _mutate_max_depth(config: dict[str, Any], delta: int) -> None:
    lo, hi = BOUNDS["max_depth"]
    config["max_depth"] = int(max(lo, min(hi, config["max_depth"] + delta)))


def _mutate_min_samples_split(config: dict[str, Any], delta: int) -> None:
    lo, hi = BOUNDS["min_samples_split"]
    config["min_samples_split"] = int(
        max(lo, min(hi, config["min_samples_split"] + delta))
    )


def _mutate_min_samples_leaf(config: dict[str, Any], delta: int) -> None:
    lo, hi = BOUNDS["min_samples_leaf"]
    config["min_samples_leaf"] = int(
        max(lo, min(hi, config["min_samples_leaf"] + delta))
    )


def _mutate_max_features(config: dict[str, Any]) -> None:
    choices = list(MAX_FEATURES_CHOICES)
    idx = choices.index(config["max_features"])
    config["max_features"] = choices[(idx + 1) % len(choices)]


def _mutate_criterion(config: dict[str, Any]) -> None:
    choices = list(CRITERION_CHOICES)
    idx = choices.index(config["criterion"])
    config["criterion"] = choices[(idx + 1) % len(choices)]


def mutate_config(config: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    """
    Apply CliffSearch-style correction mutations informed by CV diagnostics.

    High variance -> regularize (shallower trees, larger min_samples).
    Low accuracy with low variance -> increase capacity (more trees, deeper).
    """
    new_config = copy.deepcopy(config)
    acc_std = diagnostics.get("accuracy_std", 0.0)
    acc_mean = diagnostics.get("accuracy_mean", 0.0)

    if acc_std > 0.04:
        _mutate_max_depth(new_config, -1)
        _mutate_min_samples_leaf(new_config, +1)
        _mutate_min_samples_split(new_config, +1)
    elif acc_mean < 0.92:
        _mutate_n_estimators(new_config, +15)
        _mutate_max_depth(new_config, +1)
    else:
        # Fine-tune: small random walk on one axis
        axis = random.choice(
            ["n_estimators", "max_depth", "min_samples_split", "max_features", "criterion"]
        )
        if axis == "n_estimators":
            _mutate_n_estimators(new_config, random.choice([-10, 10, 20]))
        elif axis == "max_depth":
            _mutate_max_depth(new_config, random.choice([-1, 1]))
        elif axis == "min_samples_split":
            _mutate_min_samples_split(new_config, random.choice([-1, 1]))
        elif axis == "max_features":
            _mutate_max_features(new_config)
        else:
            _mutate_criterion(new_config)

    return clamp_config(new_config)


class ConfigProposer:
    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        diagnostics = history[-1][1].diagnostics if history else {}
        new_state = mutate_config(current.state, diagnostics)
        return Candidate(state=new_state, metadata={"source": "diagnostic_mutation"})
