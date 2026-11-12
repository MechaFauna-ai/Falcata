"""A LibSVM file whose only feature has index 0 loads.

GetNumColFromLIBSVMFile ended with CHECK_GT(max_col_idx, 0), so a valid zero-based LibSVM
file with a single feature (every token "0:value") failed with "Check failed:
(max_col_idx) > (0)".
"""

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("all_rows_have_the_feature", [True, False])
def test_single_feature_libsvm_trains(tmp_path, all_rows_have_the_feature):
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 2, 60)
    y = 2 * x
    lines = []
    for i in range(60):
        if all_rows_have_the_feature or i % 3:
            lines.append(f"{float(y[i])!r} 0:{float(x[i])!r}")
        else:
            lines.append(f"{0.0!r} 0:0")
            x[i], y[i] = 0.0, 0.0
    path = tmp_path / "one.svm"
    path.write_text("\n".join(lines) + "\n")
    params = {"objective": "regression", "verbose": -1, "min_data_in_leaf": 2, "min_data_in_bin": 1}
    ds = flc.Dataset(str(path), params=params)
    booster = flc.train(params, ds, num_boost_round=3)
    assert ds.num_feature() == 1
    np.testing.assert_allclose(booster.predict(str(path)), booster.predict(x.reshape(-1, 1)))


def test_libsvm_without_features_is_rejected(tmp_path):
    path = tmp_path / "none.svm"
    path.write_text("1\n0\n1\n")
    with pytest.raises(flc.basic.FalcataError, match="has no feature"):
        flc.Dataset(str(path), params={"verbose": -1}).construct()
