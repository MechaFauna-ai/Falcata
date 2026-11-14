"""Linear-tree training survives a prediction that raised the process-wide thread count.

LinearTreeLearner sized its per-thread normal-equation buffers for the thread count at
setup. A prediction between iterations runs with its own num_threads (default: all cores),
which sets the process-wide OpenMP thread count; the next iteration then indexed the
buffers with thread ids past their size (heap-buffer-overflow / use-after-free in
CalculateLinear under ASan, a crash or wrong coefficients otherwise).
"""

import numpy as np

import falcata as flc


def test_predict_between_linear_iterations():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(81, 4))
    X[rng.random(X.shape) < 0.1] = np.nan  # the NaN path of CalculateLinear
    y = np.nan_to_num(X[:, 0]) * 3
    params = {"objective": "regression", "linear_tree": True, "num_threads": 1, "verbose": -1}
    ds = flc.Dataset(X, y, params={"linear_tree": True, "verbose": -1}, free_raw_data=False)
    booster = flc.Booster(params, ds)
    reference = flc.Booster(params, flc.Dataset(X, y, params={"linear_tree": True, "verbose": -1}))
    for _ in range(4):
        booster.update()
        reference.update()
        booster.predict(X[:19], num_threads=4)
    np.testing.assert_allclose(booster.predict(X, num_threads=1), reference.predict(X, num_threads=1))
