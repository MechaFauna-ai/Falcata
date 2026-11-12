import numpy as np
import pytest

import falcata as flc

# Labels and weights set through the API go through Common::AvoidInf (NaN -> 0,
# +-inf clamped), but the text-file loaders set them row by row with
# Metadata::SetLabelAt / SetWeightAt, which stored NaN as is: the same data
# trained differently from a file, and a NaN label gave NaN gradients (an RF
# model predicting NaN, or an out-of-bounds index in multiclass metrics).


def _write(tmp_path, a, name):
    path = tmp_path / name
    np.savetxt(path, a, delimiter=",", fmt="%.10g")
    return path


@pytest.mark.parametrize("boosting", ["gbdt", "rf"])
def test_nan_label_from_file_matches_numpy(tmp_path, boosting):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    y = X[:, 0] + 0.1 * rng.standard_normal(200)
    y[[3, 50]] = np.nan
    path = _write(tmp_path, np.column_stack([y, X]), "train.csv")
    params = {
        "objective": "regression",
        "boosting": boosting,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "deterministic": True,
        "num_threads": 1,
        "verbose": -1,
    }
    from_file = flc.train(params, flc.Dataset(str(path), params={"label_column": 0}), 5)
    from_numpy = flc.train(params, flc.Dataset(X, y), 5)
    p_file, p_numpy = from_file.predict(X), from_numpy.predict(X)
    assert np.all(np.isfinite(p_file))
    np.testing.assert_allclose(p_file, p_numpy, rtol=1e-6, atol=1e-6)


def test_nan_weight_column_from_file_matches_numpy(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 3))
    y = (X[:, 0] > 0).astype(float)
    w = rng.random(200)
    w[7] = np.nan
    path = _write(tmp_path, np.column_stack([y, w, X]), "train.csv")
    params = {"objective": "binary", "deterministic": True, "num_threads": 1, "verbose": -1}
    from_file = flc.train(params, flc.Dataset(str(path), params={"label_column": 0, "weight_column": 0}), 5)
    from_numpy = flc.train(params, flc.Dataset(X, y, weight=w), 5)
    # weight_column does not count the label column; the weight keeps its slot in the file model's input
    file_input = np.column_stack([np.zeros(200), X])
    np.testing.assert_allclose(from_file.predict(file_input), from_numpy.predict(X), rtol=1e-6, atol=1e-6)


def test_nan_label_from_file_with_auc_mu(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.standard_normal((150, 3))
    y = rng.integers(0, 3, 150).astype(float)
    y[0] = np.nan
    path = _write(tmp_path, np.column_stack([y, X]), "train.csv")
    params = {"objective": "multiclass", "num_class": 3, "metric": "auc_mu", "verbose": -1}
    bst = flc.train(params, flc.Dataset(str(path), params={"label_column": 0}), 3)
    assert np.all(np.isfinite(bst.predict(X)))
