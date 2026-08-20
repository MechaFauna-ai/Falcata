# coding: utf-8
"""ObliquePool: the projection pool must (a) be exactly reproducible and
persistable, and (b) actually help on a rotated decision boundary -- the
workload class it exists for."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import falcata as lgb
from falcata import ObliquePool


def _rotated_problem(n=20_000, p=20, seed=7):
    """y depends only on x0 - x1: the worst case for axis-aligned splits,
    padded with distractor features and some NaNs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = ((X[:, 0] - X[:, 1]) > 0).astype(np.float64)
    X[rng.random(X.shape) < 0.02] = np.nan
    return X, y


def _auc(X_tr, y_tr, X_te, y_te):
    params = {
        "objective": "binary",
        "device_type": "cpu",
        "num_leaves": 4,
        "max_depth": 3,
        "learning_rate": 0.3,
        "feature_fraction": 0.5,
        "verbosity": -1,
        "seed": 0,
        "deterministic": True,
        "num_threads": 4,
    }
    booster = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=12)
    return roc_auc_score(y_te, booster.predict(X_te))


def test_pool_beats_axis_aligned_on_rotated_boundary():
    # small train set: the axis-aligned staircase around the diagonal is
    # data-starved, while one pool projection landing near x0 - x1 nails it
    X, y = _rotated_problem(n=6_000)
    X_tr, X_te = X[:4_000], X[4_000:]
    y_tr, y_te = y[:4_000], y[4_000:]

    pool = ObliquePool(num_projections=64, density=2.0, seed=3)
    auc_axis = _auc(X_tr, y_tr, X_te, y_te)
    auc_oblique = _auc(pool.fit_transform(X_tr), y_tr, pool.transform(X_te), y_te)
    assert auc_oblique > auc_axis + 0.01, (auc_oblique, auc_axis)


def test_transform_reproducible_and_persistable(tmp_path):
    X, _ = _rotated_problem(n=2_000)
    pool = ObliquePool(num_projections=16, density=3.0, seed=11)
    Z1 = pool.fit_transform(X)

    same = ObliquePool(num_projections=16, density=3.0, seed=11).fit(X)
    np.testing.assert_array_equal(Z1, same.transform(X))

    path = tmp_path / "pool.json"
    pool.save(path)
    loaded = ObliquePool.load(path)
    np.testing.assert_allclose(Z1, loaded.transform(X), rtol=1e-6)

    assert Z1.shape == (X.shape[0], X.shape[1] + 16)
    # augmented matrix must be NaN-free in the projection block
    assert np.isfinite(Z1[:, X.shape[1] :]).all()


def test_shape_mismatch_rejected():
    X, _ = _rotated_problem(n=1_000, p=10)
    pool = ObliquePool(num_projections=4, seed=0).fit(X)
    with pytest.raises(ValueError, match="fitted on 10"):
        pool.transform(X[:, :7])
