"""An empty last field of a CSV/TSV line is a missing value, like an empty field anywhere else.

An empty field in the middle of a line ("1,,2") is read as NaN, but after the line's final delimiter ("1,2,") the
parser reached the end of the line and emitted nothing, so that column read as 0. A CSV whose last column has
missing values trained a model that learned them as zeros, and predicting such a file sent them down the zero
branch.
"""

import numpy as np
import pytest

import falcata as flc

# test_empty_last_field_trains_as_missing compares two trainings bit for bit: on CUDA (unset device_type with a usable
# GPU) non-quantized training repeats only with the deterministic histograms (opt-in since 1.2; CPU ignores the key)
_PARAMS = {"verbose": -1, "num_leaves": 4, "min_data_in_leaf": 5, "cuda_plan": "auto,nonquant_deterministic:on"}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    f0 = rng.standard_normal(600)
    f1 = rng.standard_normal(600)
    f1[::3] = np.nan
    y = np.where(np.isnan(f1), 10.0, np.where(f1 > 0, 1.0, 0.0))
    return f0, f1, y


def _write(path, sep, f0, f1, y, missing):
    lines = [
        sep.join([repr(float(yy)), repr(float(a)), missing if np.isnan(c) else repr(float(c))])
        for yy, a, c in zip(y, f0, f1, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("sep", [",", "\t"])
def test_empty_last_field_trains_as_missing(tmp_path, data, sep):
    f0, f1, y = data
    # TSV sniffing trims each sampled line, which drops an empty last field, so keep the first 32 lines complete
    head = 0 if sep == "," else 40
    f1 = f1.copy()
    f1[:head][np.isnan(f1[:head])] = 0.25
    preds = {}
    for missing in ("", "nan"):
        path = tmp_path / f"train_{missing or 'empty'}.txt"
        _write(path, sep, f0, f1, y, missing)
        booster = flc.train(_PARAMS, flc.Dataset(str(path), params={"verbose": -1}), 10)
        preds[missing] = booster.predict(np.array([[0.3, np.nan], [0.3, 0.0], [0.3, 1.0]]))
    np.testing.assert_array_equal(preds[""], preds["nan"])


@pytest.mark.parametrize("sep", [",", "\t"])
def test_empty_last_field_predicts_as_missing(tmp_path, data, sep):
    f0, f1, y = data
    booster = flc.train(_PARAMS, flc.Dataset(np.c_[f0, f1], y), 10)
    rows = np.array([[0.5, 1.0]] * 40 + [[0.3, np.nan], [0.3, 0.0], [-1.0, np.nan]])
    path = tmp_path / "predict.txt"
    lines = [sep.join([repr(float(a)), "" if np.isnan(c) else repr(float(c))]) for a, c in rows]
    path.write_text("\n".join(lines) + "\n")
    np.testing.assert_array_equal(booster.predict(str(path)), booster.predict(rows))
