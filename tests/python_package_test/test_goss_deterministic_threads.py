import numpy as np
import pytest

import falcata as flc

# GOSS picks its top_rate threshold and draws the other rows per block of the
# parallel partition runner, whose blocks follow num_threads. deterministic=true
# promises the same model for different num_threads on CPU (docs), but GOSS
# models differed once sampling started (after 1 / learning_rate rounds).


@pytest.mark.parametrize("goss", [{"boosting": "goss"}, {"data_sample_strategy": "goss"}])
@pytest.mark.parametrize("objective", ["regression", "multiclassova"])
def test_goss_deterministic_across_num_threads(goss, objective):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((7000, 10))
    if objective == "regression":
        y = X[:, 0] + rng.standard_normal(7000)
    else:
        y = np.digitize(X[:, 0] + 0.5 * rng.standard_normal(7000), [-1, 0, 1]).astype(float)
    params = {
        "objective": objective,
        "deterministic": True,
        "force_row_wise": True,
        "device_type": "cpu",
        "verbose": -1,
        **goss,
    }
    if objective != "regression":
        params["num_class"] = 4
    preds = [
        flc.train({**params, "num_threads": t}, flc.Dataset(X, y), num_boost_round=20).predict(X, raw_score=True)
        for t in (1, 3, 5)
    ]
    np.testing.assert_array_equal(preds[0], preds[1])
    np.testing.assert_array_equal(preds[0], preds[2])
