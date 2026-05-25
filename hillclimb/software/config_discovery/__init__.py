"""CliffSearch-inspired hyperparameter discovery for sklearn RandomForest."""

from hillclimb.software.config_discovery.search_space import (
    PARAM_NAMES,
    random_config,
)

__all__ = ["PARAM_NAMES", "random_config"]
