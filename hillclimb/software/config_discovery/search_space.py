"""Hyperparameter search space for RandomForest on breast cancer data."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

PARAM_NAMES = (
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "criterion",
)

CRITERION_CHOICES = ("gini", "entropy", "log_loss")
MAX_FEATURES_CHOICES = ("sqrt", "log2", None)

BOUNDS: dict[str, tuple[Any, ...]] = {
    "n_estimators": (10, 200),
    "max_depth": (2, 20),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 10),
    "max_features": MAX_FEATURES_CHOICES,
    "criterion": CRITERION_CHOICES,
}


def load_data() -> tuple[np.ndarray, np.ndarray]:
    data = load_breast_cancer()
    return data.data, data.target


def random_config(rng: random.Random | None = None) -> dict[str, Any]:
    rng = rng or random.Random()
    return {
        "n_estimators": rng.randint(*BOUNDS["n_estimators"]),
        "max_depth": rng.randint(*BOUNDS["max_depth"]),
        "min_samples_split": rng.randint(*BOUNDS["min_samples_split"]),
        "min_samples_leaf": rng.randint(*BOUNDS["min_samples_leaf"]),
        "max_features": rng.choice(BOUNDS["max_features"]),
        "criterion": rng.choice(BOUNDS["criterion"]),
    }


def config_to_model(config: dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        criterion=config["criterion"],
        random_state=42,
        n_jobs=1,
    )


def clamp_config(config: dict[str, Any]) -> dict[str, Any]:
    lo, hi = BOUNDS["n_estimators"]
    config["n_estimators"] = int(max(lo, min(hi, config["n_estimators"])))
    lo, hi = BOUNDS["max_depth"]
    config["max_depth"] = int(max(lo, min(hi, config["max_depth"])))
    lo, hi = BOUNDS["min_samples_split"]
    config["min_samples_split"] = int(max(lo, min(hi, config["min_samples_split"])))
    lo, hi = BOUNDS["min_samples_leaf"]
    config["min_samples_leaf"] = int(max(lo, min(hi, config["min_samples_leaf"])))
    if config["max_features"] not in MAX_FEATURES_CHOICES:
        config["max_features"] = "sqrt"
    if config["criterion"] not in CRITERION_CHOICES:
        config["criterion"] = "gini"
    return config
