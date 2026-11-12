import numpy as np
import pytest

import falcata as flc

# With non-finite gradients (poisson on a 1e30 label) the categorical split
# finder computed NaN gains. "NaN <= min_gain_shift" is false, so it marked the
# feature splittable without ever recording a best threshold, and emitted a
# split on category bin -1 + offset: SplitInner then indexed the category
# table far out of range (ASan SEGV in BinMapper::BinToValue). Since #322
# (RoundInt maps NaN counts to 0 instead of a huge negative value) the
# candidates are no longer filtered out first, so master crashed where
# 856772a2 trained a NaN model. NaN gains are now never splittable.


@pytest.mark.parametrize("onehot", [True, False])
def test_categorical_split_with_non_finite_gradients(onehot):
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.integers(-1, 2 if onehot else 40, (n, 1)).astype(float)
    y = np.floor(np.exp(rng.standard_normal(n)) * rng.random(n) * 2)
    y[0] = 1e30
    params = {
        "objective": "poisson",
        "boost_from_average": False,
        "min_data_in_leaf": 0,
        "max_cat_to_onehot": 4 if onehot else 1,
        "verbose": -1,
        "device_type": "cpu",
    }
    error = ""
    try:
        bst = flc.train(params, flc.Dataset(X, y, categorical_feature=[0]), num_boost_round=2)
    except flc.basic.FalcataError as e:
        error = str(e)
    # either it trains, or the non-finite leaf guard stops it with a clean error: never a crash
    if error:
        assert "non-finite leaf output" in error
    else:
        assert bst.num_trees() >= 1
