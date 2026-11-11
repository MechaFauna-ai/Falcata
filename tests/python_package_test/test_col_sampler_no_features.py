import numpy as np
import pytest

import falcata as flc

# With feature_fraction < 1 and a dataset whose features are all unusable
# (here: constant), ColSampler::ResetByTree memset() an empty vector, passing
# a null pointer (UBSan: "null pointer passed as argument 1, which is declared
# to never be null", col_sampler.hpp). The check fires under the sanitizer
# build; the test also pins that training stays well-defined.


@pytest.mark.parametrize(
    "extra",
    [{}, {"boosting": "rf", "bagging_fraction": 0.5, "bagging_freq": 1, "linear_tree": True}],
)
def test_feature_fraction_without_usable_features(extra):
    X = np.full((50, 3), 2.0)
    y = np.arange(50.0)
    params = {"objective": "regression", "feature_fraction": 0.5, "verbose": -1, **extra}
    bst = flc.train(params, flc.Dataset(X, y), num_boost_round=3)
    preds = bst.predict(X)
    assert np.all(np.isfinite(preds))
    if not extra:
        np.testing.assert_allclose(preds, y.mean())
