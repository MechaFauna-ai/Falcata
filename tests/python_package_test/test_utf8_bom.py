"""Text data files that start with a UTF-8 byte order mark load like files without one.

Excel and Windows write "CSV UTF-8" files with a leading BOM. It used to stick to the first
value of a headerless file ("Unknown token \\ufeff0.844 in data file") and to the first
header name, so ``label_column=name:<first column>`` was not found and a model trained on
the file named its first feature "\\ufeff<name>" (predicting a headered file without the
BOM then silently treated that feature as missing).
"""

import numpy as np
import pytest

import falcata as flc

BOM = b"\xef\xbb\xbf"


def _write(path, text, bom):
    path.write_bytes((BOM if bom else b"") + text.encode())
    return str(path)


def _rows(n=60):
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(n, 2))
    y = 3 * X[:, 0] + X[:, 1]
    return "".join(f"{y[i]:.6f},{X[i, 0]:.6f},{X[i, 1]:.6f}\n" for i in range(n)), X


@pytest.mark.parametrize("header", [False, True])
def test_bom_file_trains_like_plain_file(tmp_path, header):
    body, X = _rows()
    text = ("y,a,b\n" if header else "") + body
    params = {"header": header, "verbose": -1, "min_data_in_leaf": 2}
    if header:
        params["label_column"] = "name:y"
    models = []
    for bom in (False, True):
        path = _write(tmp_path / f"t{int(bom)}.csv", text, bom)
        ds = flc.Dataset(path, params=params)
        booster = flc.train({"objective": "regression", **params}, ds, num_boost_round=3)
        models.append(booster)
        if header:
            assert booster.feature_name() == ["a", "b"]
    np.testing.assert_allclose(models[0].predict(X), models[1].predict(X))


def test_bom_file_predicts(tmp_path):
    body, X = _rows()
    plain = _write(tmp_path / "p.csv", "y,a,b\n" + body, False)
    bom = _write(tmp_path / "b.csv", "y,a,b\n" + body, True)
    booster = flc.train(
        {"objective": "regression", "verbose": -1, "min_data_in_leaf": 2},
        flc.Dataset(plain, params={"header": True, "label_column": "name:y"}),
        num_boost_round=3,
    )
    np.testing.assert_allclose(booster.predict(bom, data_has_header=True), booster.predict(plain, data_has_header=True))
