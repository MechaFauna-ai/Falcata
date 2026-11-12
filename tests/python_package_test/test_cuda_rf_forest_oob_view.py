"""RF on CUDA: the trees a forest group grows ahead are scored on their out-of-bag rows later, when the packed
per-tree column view in force is the last grown tree's. Under feature_fraction that view leaves the earlier trees'
columns null, so reading the train score (a training metric, eval_train, update(train_set=...)) hit an illegal
memory access.
"""

import os

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata (TASK=cuda)"
)


def _data():
    r = np.random.default_rng(1)
    n = 1800
    X = r.standard_normal((n, 8)).astype(np.float32)
    z = X @ r.standard_normal(8)
    y = np.digitize(z + 0.5 * r.standard_normal(n), np.quantile(z, [1 / 3, 2 / 3])).astype(float)
    return X, y


def _params(**extra):
    p = {
        "objective": "multiclass",
        "num_class": 3,
        "boosting": "rf",
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "feature_fraction": 0.7,
        "max_bin": 3,
        "num_leaves": 7,
        "min_data_in_leaf": 50,
        "device_type": "cuda",
        "deterministic": True,
        "num_threads": 2,
        "verbose": -1,
    }
    p.update(extra)
    return p


def test_rf_training_metric_with_feature_fraction():
    X, y = _data()
    params = _params(metric="multi_logloss")
    ds = flc.Dataset(X, y, params=params, free_raw_data=False)
    evals = {}
    booster = flc.train(params, ds, 6, valid_sets=[ds], valid_names=["train"], callbacks=[flc.record_evaluation(evals)])
    assert len(evals["train"]["multi_logloss"]) == 6
    assert np.all(np.isfinite(booster.predict(X)))
    # the same forest as growing every tree in turn (no group grown ahead)
    seq = flc.train(
        _params(metric="multi_logloss", cuda_plan="auto,forest_batched:off"),
        flc.Dataset(X, y, params=params, free_raw_data=False),
        6,
        valid_sets=[ds],
    )
    np.testing.assert_allclose(booster.predict(X, raw_score=True), seq.predict(X, raw_score=True), rtol=0, atol=1e-12)


def test_rf_update_train_set_with_feature_fraction():
    X, y = _data()
    params = _params()
    ds = flc.Dataset(X[:1200], y[:1200], params=params, free_raw_data=False)
    booster = flc.Booster(params=params, train_set=ds)
    booster.update()
    booster.update(train_set=flc.Dataset(X[1200:], y[1200:], params=params, reference=ds, free_raw_data=False))
    for _ in range(3):
        booster.update()
    assert np.all(np.isfinite(booster.predict(X)))
