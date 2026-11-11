"""Softplus must remain finite before applying small observation weights."""

import falcata as flc
import numpy as np
import pytest


@pytest.mark.parametrize("objective", ["none", "cross_entropy_lambda"])
@pytest.mark.parametrize("score", [1000.0, 10000.0])
def test_weighted_xentropy_large_raw_score(objective, score):
    weight = 1.0 / score
    data = flc.Dataset(
        np.arange(40.0).reshape(-1, 1),
        label=np.full(40, 0.5),
        weight=np.full(40, weight),
        init_score=np.full(40, score),
    )
    model = flc.Booster(
        {"objective": objective, "metric": "xentlambda", "num_threads": 1, "verbosity": -1}, train_set=data
    )
    actual = model.eval_train()[0][2]
    probability = -np.expm1(-1.0)
    expected = -0.5 * np.log(probability) - 0.5 * np.log1p(-probability)
    assert actual == pytest.approx(expected, rel=1e-6)
