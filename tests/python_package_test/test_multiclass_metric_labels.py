import numpy as np
import pytest

import falcata as flc

# Multiclass metrics index each row's num_class scores (and AUC-mu its
# per-class tables) by the label. The objective checks the training labels,
# but a validation set's labels were never checked, and AUC-mu's Init runs
# before the objective's: an out-of-range label read or wrote out of bounds.


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = rng.integers(0, 3, n).astype(float)
    return X, y


@pytest.mark.parametrize("metric", ["multi_logloss", "multi_error", "auc_mu"])
@pytest.mark.parametrize("bad_label", [-1.0, 3.0, 7.0])
def test_out_of_range_validation_label_is_rejected(metric, bad_label):
    X, y = _data()
    Xv, yv = _data(seed=1)
    yv[5] = bad_label
    params = {"objective": "multiclass", "num_class": 3, "metric": metric, "verbose": -1}
    train = flc.Dataset(X, y)
    valid = flc.Dataset(Xv, yv, reference=train)
    with pytest.raises(flc.basic.FalcataError, match=rf"Metric {metric}: label must be in \[0, 3\)"):
        flc.train(params, train, num_boost_round=2, valid_sets=[valid])


@pytest.mark.parametrize("objective", ["multiclass", "multiclassova"])
def test_negative_training_label_with_auc_mu_is_rejected(objective):
    X, y = _data()
    y[0] = -2.0
    params = {"objective": objective, "num_class": 3, "metric": "auc_mu", "verbose": -1}
    with pytest.raises(flc.basic.FalcataError, match="(?i)label must be in"):
        flc.train(params, flc.Dataset(X, y), num_boost_round=2)


def test_in_range_labels_still_evaluate():
    X, y = _data()
    params = {"objective": "multiclass", "num_class": 3, "metric": ["multi_logloss", "auc_mu"], "verbose": -1}
    train = flc.Dataset(X, y)
    evals = {}
    flc.train(params, train, 3, valid_sets=[train], callbacks=[flc.record_evaluation(evals)])
    assert np.all(np.isfinite(evals["training"]["multi_logloss"]))
    assert np.all(np.isfinite(evals["training"]["auc_mu"]))
