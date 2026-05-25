from __future__ import annotations

import numpy as np


REGIME_NAMES = ("bull_low_vol", "bear_high_vol", "sideways")

ASSET_NAMES = ("equity", "bond", "commodity")


def generate_synthetic_returns(
    n_days: int = 504,
    *,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Generate multi-asset daily returns with latent regime labels.

    Returns dict with keys: returns (T, 3), regime_labels (T,), dates index.
    """
    rng = np.random.default_rng(seed)
    returns = np.zeros((n_days, 3))
    regime_labels = np.zeros(n_days, dtype=int)

    # Regime segments: bull (0), bear (1), sideways (2)
    segment_lengths = [n_days // 3, n_days // 3, n_days - 2 * (n_days // 3)]
    regimes = [0, 1, 2]
    idx = 0
    for regime, length in zip(regimes, segment_lengths):
        regime_labels[idx : idx + length] = regime
        idx += length

    means = {
        0: np.array([0.0008, 0.0002, 0.0003]),   # bull: equity leads
        1: np.array([-0.0012, 0.0004, 0.0001]),  # bear: equity falls
        2: np.array([0.0001, 0.0001, 0.0000]),   # sideways
    }
    vols = {
        0: np.array([0.012, 0.004, 0.010]),
        1: np.array([0.025, 0.006, 0.018]),
        2: np.array([0.008, 0.003, 0.007]),
    }
    corr = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, -0.1], [0.4, -0.1, 1.0]])

    for t in range(n_days):
        regime = int(regime_labels[t])
        vol = vols[regime]
        cov = np.outer(vol, vol) * corr
        shock = rng.multivariate_normal(means[regime], cov)
        returns[t] = shock

    return {
        "returns": returns,
        "regime_labels": regime_labels,
        "asset_names": np.array(ASSET_NAMES),
        "regime_names": np.array(REGIME_NAMES),
    }
