"""Averaged models have zero raw output for an empty iteration range."""

import numpy as np
import pytest
from scipy import sparse

import falcata as flc


@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("objective", ["regression", "binary"])
def test_rf_empty_prediction_range(as_sparse, objective):
    x = np.random.default_rng(9).normal(size=(100, 2))
    model = flc.train(
        {
            "objective": objective,
            "boosting_type": "rf",
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "feature_fraction": 0.8,
            "min_data_in_leaf": 3,
            "num_threads": 1,
            "verbosity": -1,
        },
        flc.Dataset(x, label=(x[:, 0] > 0).astype(float) if objective == "binary" else x[:, 0]),
        num_boost_round=3,
    )
    data = sparse.csr_matrix(x) if as_sparse else x
    raw = model.predict(data, start_iteration=3, raw_score=True)
    result = model.predict(data, start_iteration=3)
    np.testing.assert_array_equal(raw, np.zeros(100))
    np.testing.assert_array_equal(result, np.full(100, 0.5 if objective == "binary" else 0.0))
