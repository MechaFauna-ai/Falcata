import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("zero_count", [1, 3])
def test_average_precision_ignores_zero_weight_thresholds(zero_count):
    x = np.arange(20.0, dtype=float).reshape(-1, 1)
    labels = np.tile([0.0, 1.0], 10)
    scores = np.arange(20.0, dtype=float)[::-1].copy()
    weights = np.ones(20)
    weights[:zero_count] = 0
    params = {
        "objective": "none",
        "metric": "average_precision",
        "num_threads": 1,
        "min_data_in_leaf": 1,
        "verbosity": -1,
    }

    def evaluate(keep):
        data = flc.Dataset(x[keep], label=labels[keep], weight=weights[keep], init_score=scores[keep])
        model = flc.Booster(params, train_set=data)
        return model.eval_train()[0][2]

    actual = evaluate(np.arange(20))
    expected = evaluate(np.flatnonzero(weights))
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected)
