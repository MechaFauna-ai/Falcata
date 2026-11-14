"""A LibSVM file whose first rows have no nonzero feature is detected as LibSVM.

Format detection looked for an index:value token only in the first two lines, so a sparse
LibSVM file starting with two label-only rows (rows whose features are all zero) failed with
"Unknown format of training data".
"""

import numpy as np
import scipy.sparse as sp

import falcata as flc


def test_libsvm_with_label_only_first_rows(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 2, size=(80, 3))
    X[rng.random(X.shape) < 0.5] = 0.0
    X[:3] = 0.0  # the first rows have no nonzero feature
    X[10, 2] = 1.5  # keep the last feature present
    y = X.sum(axis=1)
    lines = []
    for i in range(len(X)):
        feats = " ".join(f"{j}:{float(X[i, j])!r}" for j in range(3) if X[i, j] != 0)
        lines.append(f"{float(y[i])!r} {feats}".rstrip())
    path = tmp_path / "t.svm"
    path.write_text("\n".join(lines) + "\n")
    params = {"objective": "regression", "verbose": -1, "min_data_in_leaf": 2, "min_data_in_bin": 1}
    ds = flc.Dataset(str(path), params=params)
    booster = flc.train(params, ds, num_boost_round=3)
    assert ds.num_feature() == 3
    reference = flc.train(params, flc.Dataset(sp.csr_matrix(X), y, params=params), num_boost_round=3)
    np.testing.assert_allclose(booster.predict(X), reference.predict(X))
