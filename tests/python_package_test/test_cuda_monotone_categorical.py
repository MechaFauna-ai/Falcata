"""CUDA monotone constraints hold on data with categorical features.

A categorical feature has no monotone direction of its own, but a categorical split can sit below a monotone split,
and its two child outputs must stay inside that leaf's [min, max] (CPU FindBestThresholdCategoricalInner with USE_MC).
The CUDA categorical split finders computed the children's outputs and gains without the leaf's bounds, so a model
trained on CUDA could break a monotone constraint that the CPU model keeps.
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only regression test")


def _data(seed, objective, n=3000, m=20, num_cat=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m))
    p = 1.0 / np.arange(1, 301) ** 1.1
    for j in range(num_cat):
        X[:, j] = rng.choice(300, size=n, p=p / p.sum())
    z = X[:, 5] - X[:, 6] + np.sin(X[:, 0])
    y = rng.poisson(np.exp(0.5 * z)).astype(float) if objective == "poisson" else z + 0.3 * rng.standard_normal(n)
    constraints = [0] * num_cat + [[-1, 0, 1][j % 3] for j in range(num_cat, m)]
    return X, y, constraints


def _worst_violation(booster, X, constraints):
    base = X[:30]
    grid = np.linspace(-3, 3, 25)
    worst = 0.0
    for j, c in enumerate(constraints):
        if c == 0:
            continue
        Xg = np.repeat(base, len(grid), axis=0)
        Xg[:, j] = np.tile(grid, len(base))
        steps = np.diff(booster.predict(Xg, raw_score=True).reshape(len(base), len(grid)), axis=1) * c
        worst = min(worst, float(steps.min()))
    return worst


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("objective", ["regression", "poisson"])
@pytest.mark.parametrize("max_bin", [255, 1023])  # 1023: a category count over 256 takes the global-memory finder
def test_cuda_monotone_constraints_hold_with_categorical_features(seed, objective, max_bin):
    X, y, constraints = _data(seed, objective)
    params = {
        "objective": objective,
        "device_type": "cuda",
        "monotone_constraints": constraints,
        "num_leaves": 31,
        "max_bin": max_bin,
        "learning_rate": 0.1,
        "quant_mode": "none",
        "deterministic": True,
        "verbose": -1,
    }
    booster = flc.train(params, flc.Dataset(X, y, categorical_feature=[0, 1, 2], params=params), num_boost_round=10)
    assert _worst_violation(booster, X, constraints) >= -1e-12
