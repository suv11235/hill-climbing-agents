"""Cross-validation accuracy evaluator for config candidates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import cross_val_score

from hillclimb.core.types import Candidate, Evaluation
from hillclimb.software.config_discovery.search_space import (
    config_to_model,
    load_data,
)


@dataclass
class ConfigEvaluator:
    cv_folds: int = 5
    _X: np.ndarray = field(init=False, repr=False)
    _y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._X, self._y = load_data()

    def evaluate(self, candidate: Candidate) -> Evaluation:
        config = candidate.state
        model = config_to_model(config)
        try:
            scores = cross_val_score(
                model,
                self._X,
                self._y,
                cv=self.cv_folds,
                scoring="accuracy",
                n_jobs=1,
            )
        except Exception as exc:
            return Evaluation(
                score=0.0,
                diagnostics={"exception": str(exc), "config": dict(config)},
                passed=False,
                error=str(exc),
            )

        mean_acc = float(np.mean(scores))
        std_acc = float(np.std(scores))
        return Evaluation(
            score=mean_acc,
            diagnostics={
                "accuracy_mean": mean_acc,
                "accuracy_std": std_acc,
                "cv_scores": scores.tolist(),
                "config": dict(config),
            },
            passed=True,
        )
