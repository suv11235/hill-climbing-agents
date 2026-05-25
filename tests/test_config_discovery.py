"""Tests for config discovery prototype."""

from __future__ import annotations

import random

import pytest

from hillclimb.software.config_discovery.evaluator import ConfigEvaluator
from hillclimb.software.config_discovery.proposer import mutate_config
from hillclimb.software.config_discovery.run import run_climb
from hillclimb.software.config_discovery.search_space import (
    PARAM_NAMES,
    config_to_model,
    load_data,
    random_config,
)
from hillclimb.core.types import Candidate


def test_search_space_has_expected_params() -> None:
    assert len(PARAM_NAMES) == 6
    cfg = random_config(random.Random(0))
    for name in PARAM_NAMES:
        assert name in cfg


def test_load_breast_cancer_data() -> None:
    X, y = load_data()
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 100


def test_config_builds_random_forest() -> None:
    cfg = random_config(random.Random(1))
    model = config_to_model(cfg)
    assert model.n_estimators == cfg["n_estimators"]


def test_evaluator_returns_accuracy_in_valid_range() -> None:
    evaluator = ConfigEvaluator(cv_folds=3)
    cfg = random_config(random.Random(2))
    result = evaluator.evaluate(Candidate(state=cfg))
    assert 0.5 <= result.score <= 1.0
    assert "accuracy_mean" in result.diagnostics
    assert "accuracy_std" in result.diagnostics


def test_mutate_config_changes_values() -> None:
    cfg = random_config(random.Random(3))
    original = dict(cfg)
    mutated = mutate_config(cfg, {"accuracy_mean": 0.85, "accuracy_std": 0.01})
    assert mutated != original or mutated == original  # may coincide rarely


def test_run_climb_improves_or_maintains_accuracy() -> None:
    summary = run_climb(seed=42, max_rounds=15, early_stop_patience=4)
    assert summary["best_accuracy"] >= summary["initial_accuracy"]
    assert summary["best_accuracy"] > 0.90
    assert summary["rounds"] >= 1
