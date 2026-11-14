"""Predicting a text file with a header whose column count differs from its data lines.

With a header, the predictor maps the file's columns to the model's features by name
through a vector sized by the number of fields. A data line with more fields than that
read past the vector, and a header naming more columns than the data lines hold wrote
past it (heap-buffer-overflow under ASan).
"""

import numpy as np

import falcata as flc


def _train(tmp_path, names):
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(60, len(names)))
    path = tmp_path / "train.csv"
    path.write_text(",".join(["y", *names]) + "\n" + "".join(f"{x[0] * 3},{','.join(map(str, x))}\n" for x in X))
    ds = flc.Dataset(str(path), params={"header": True, "label_column": "name:y", "verbose": -1})
    return flc.train({"objective": "regression", "min_data_in_leaf": 2, "verbose": -1}, ds, num_boost_round=3)


def _rows(rng, n, k):
    return [",".join(f"{v:.4f}" for v in rng.uniform(size=k)) for _ in range(n)]


def test_data_line_longer_than_header(tmp_path):
    booster = _train(tmp_path, ["a", "b"])
    rng = np.random.default_rng(1)
    rows = _rows(rng, 60, 3)
    long_row = rows[0] + ",0.4" * 500
    path = tmp_path / "pred.csv"
    path.write_text("y,b,a\n" + "\n".join(rows + [long_row]) + "\n")
    pred = booster.predict(str(path), data_has_header=True)
    assert pred.shape == (61,)
    # the extra fields are dropped: the long row predicts like its first three fields
    np.testing.assert_allclose(pred[-1], pred[0])


def test_header_longer_than_data_lines(tmp_path):
    booster = _train(tmp_path, ["a", "b", "c"])
    rng = np.random.default_rng(2)
    path = tmp_path / "pred.csv"
    path.write_text("y,c,b,a,d,e\n" + "\n".join(_rows(rng, 30, 2)) + "\n")
    pred = booster.predict(str(path), data_has_header=True)
    assert pred.shape == (30,)
    assert np.all(np.isfinite(pred))
