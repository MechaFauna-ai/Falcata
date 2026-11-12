import numpy as np
import pytest

import falcata as flc

# The gamma / gamma_deviance metrics and the quantile objective refused bad
# input with bare CHECK macros ("Check failed: (label) > (0)", "Check failed:
# alpha_ > 0 && alpha_ < 1") that name neither the metric nor the parameter.
# The gamma metrics were the most frequent error in the overnight fuzzer
# (e.g. metric=gamma on a regression dataset with a zero label).


def _data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = np.abs(X[:, 0]) + 0.5
    return X, y


@pytest.mark.parametrize("metric", ["gamma", "gamma_deviance"])
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_gamma_metrics_name_themselves(metric, bad):
    X, y = _data()
    y[4] = bad
    with pytest.raises(flc.basic.FalcataError, match=f"Metric {metric} requires positive labels, but found {int(bad)}"):
        flc.train(
            {"objective": "regression", "metric": metric, "device_type": "cpu", "verbose": -1}, flc.Dataset(X, y), 2
        )


@pytest.mark.parametrize("alpha", [1.0, 1.5, 7.0])  # alpha <= 0 is refused by the config check
def test_quantile_alpha_out_of_range(alpha):
    X, y = _data()
    with pytest.raises(flc.basic.FalcataError, match="objective=quantile requires 0 < alpha < 1"):
        flc.train({"objective": "quantile", "alpha": alpha, "device_type": "cpu", "verbose": -1}, flc.Dataset(X, y), 2)


def test_valid_gamma_metric_and_quantile_still_train():
    X, y = _data()
    evals = {}
    ds = flc.Dataset(X, y)
    flc.train(
        {
            "objective": "quantile",
            "alpha": 0.3,
            "metric": ["gamma", "gamma_deviance"],
            "device_type": "cpu",
            "verbose": -1,
        },
        ds,
        3,
        valid_sets=[ds],
        callbacks=[flc.record_evaluation(evals)],
    )
    assert np.all(np.isfinite(evals["training"]["gamma"]))
