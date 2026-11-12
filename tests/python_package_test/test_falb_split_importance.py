"""Split importance of a model loaded from FALB without diagnostics.

save_model(format="falb") and model_to_binary() leave the diagnostics
section (split gains) out by default. Every split_gain then reads 0, and
split importance -- which counts splits with a positive gain -- reported 0
for every feature of a model that splits on them (gain importance is
documented as needing the diagnostics; split importance is not).
"""

import numpy as np
import pytest

import falcata as flc


@pytest.fixture(scope="module")
def booster():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 4))
    y = 2 * x[:, 0] + x[:, 2]
    return flc.train({"verbose": -1, "num_threads": 1, "num_leaves": 7}, flc.Dataset(x, label=y), num_boost_round=4)


@pytest.mark.parametrize("kwargs", [{}, {"with_stats": True}, {"with_diagnostics": True}, {"compress_level": 0}])
def test_split_importance_survives_falb(booster, kwargs):
    loaded = flc.Booster(model_bin=booster.model_to_binary(**kwargs))
    np.testing.assert_array_equal(loaded.feature_importance("split"), booster.feature_importance("split"))
