# coding: utf-8
"""An unknown metric name is an error, not a silently missing metric.

Metric::CreateMetric returned nullptr for any name it did not know and the callers skipped
it, so metric="aucc" (a typo) trained with no metric at all: nothing was evaluated and
early stopping had nothing to watch.
"""

import numpy as np
import pytest

import falcata as flc


def _data():
    rng = np.random.default_rng(0)
    X = rng.random((300, 3))
    return X, (X[:, 0] > 0.5).astype(float)


@pytest.mark.parametrize("metric", ["aucc", ["binary_logloss", "aucc"]])
def test_unknown_metric_is_rejected(metric):
    X, y = _data()
    ds = flc.Dataset(X, y)
    with pytest.raises(flc.basic.FalcataError, match="Unknown metric type name: aucc"):
        flc.train({"verbose": -1, "objective": "binary", "metric": metric}, ds, 2, valid_sets=[ds])


@pytest.mark.parametrize("metric", ["None", "custom", "na", "auc", "", ["auc", "binary_error"]])
def test_known_or_disabled_metrics_still_work(metric):
    X, y = _data()
    ds = flc.Dataset(X, y)
    flc.train({"verbose": -1, "objective": "binary", "metric": metric}, ds, 2, valid_sets=[ds])
