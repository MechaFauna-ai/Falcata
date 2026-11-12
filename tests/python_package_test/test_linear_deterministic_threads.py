import numpy as np
import pytest

import falcata as flc

# The linear leaf fit sums X^T H X and X^T g per thread and then adds the
# per-thread sums, so with more than 1024 rows the coefficients depended on
# num_threads (float grouping), although deterministic=true promises the same
# model for any num_threads on CPU.


@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_linear_tree_deterministic_across_num_threads(objective):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5000, 6))
    y = 2 * X[:, 0] + np.sin(X[:, 1]) + 0.1 * rng.standard_normal(5000)
    if objective == "binary":
        y = (y > 0).astype(float)
    params = {
        "objective": objective,
        "linear_tree": True,
        "deterministic": True,
        "force_row_wise": True,
        "device_type": "cpu",
        "verbose": -1,
    }
    preds = [
        flc.train({**params, "num_threads": t}, flc.Dataset(X, y), num_boost_round=10).predict(X, raw_score=True)
        for t in (1, 3, 5)
    ]
    np.testing.assert_array_equal(preds[0], preds[1])
    np.testing.assert_array_equal(preds[0], preds[2])
