# coding: utf-8
"""Prediction from a data file comes back in memory, as the in-memory paths do."""

import ctypes

import numpy as np
import pytest

import falcata as lgb
from falcata.basic import _LIB, _c_str, _safe_call

KINDS = [{}, {"raw_score": True}, {"pred_leaf": True}, {"pred_contrib": True}]


@pytest.fixture(scope="module")
def multiclass_booster():
    rng = np.random.default_rng(3)
    X = rng.random((3000, 10))
    y = (X[:, 0] * 3).astype(int)
    booster = lgb.train({"objective": "multiclass", "num_class": 3, "verbosity": -1, "seed": 1}, lgb.Dataset(X, y), 10)
    return booster, X


@pytest.mark.parametrize("kwargs", KINDS)
def test_file_predictions_equal_in_memory_predictions(tmp_path, multiclass_booster, kwargs):
    booster, X = multiclass_booster
    data_file = tmp_path / "data.csv"
    np.savetxt(data_file, X, delimiter=",", fmt="%.17g")
    from_file = booster.predict(data_file, **kwargs)
    in_memory = booster.predict(np.loadtxt(data_file, delimiter=","), **kwargs)
    assert from_file.shape == in_memory.shape
    np.testing.assert_array_equal(from_file, in_memory)


# a one-row file used to come back flat ((num_class,) or (k, 1)) because np.loadtxt drops the row axis
@pytest.mark.parametrize("kwargs", KINDS)
def test_one_row_file_keeps_its_row_axis(tmp_path, multiclass_booster, kwargs):
    booster, X = multiclass_booster
    data_file = tmp_path / "one_row.csv"
    np.savetxt(data_file, X[:1], delimiter=",", fmt="%.17g")
    from_file = booster.predict(data_file, **kwargs)
    in_memory = booster.predict(X[:1], **kwargs)
    assert from_file.shape == in_memory.shape
    np.testing.assert_array_equal(from_file, in_memory)


def test_c_api_file_result_matches_the_array_entry_point(tmp_path, multiclass_booster):
    # FLC_BoosterPredictForFile still writes the same values, at round-trip precision
    booster, X = multiclass_booster
    data_file = tmp_path / "data.csv"
    result_file = tmp_path / "result.txt"
    np.savetxt(data_file, X, delimiter=",", fmt="%.17g")
    _safe_call(
        _LIB.FLC_BoosterPredictForFile(
            booster._handle,
            _c_str(str(data_file)),
            ctypes.c_int(0),
            ctypes.c_int(0),
            ctypes.c_int(0),
            ctypes.c_int(-1),
            _c_str(""),
            _c_str(str(result_file)),
        )
    )
    np.testing.assert_array_equal(np.loadtxt(result_file), booster.predict(data_file))


def test_array_entry_point_reports_rows_and_length(tmp_path, multiclass_booster):
    booster, _ = multiclass_booster
    num_row = ctypes.c_int64(-1)
    length = ctypes.c_int64(-1)
    result = ctypes.POINTER(ctypes.c_double)()
    data_file = tmp_path / "data.csv"
    np.savetxt(data_file, np.zeros((5, 10)), delimiter=",")
    _safe_call(
        _LIB.FLC_BoosterPredictForFileToArray(
            booster._handle,
            _c_str(str(data_file)),
            ctypes.c_int(0),
            ctypes.c_int(0),
            ctypes.c_int(0),
            ctypes.c_int(-1),
            _c_str(""),
            ctypes.byref(num_row),
            ctypes.byref(length),
            ctypes.byref(result),
        )
    )
    assert num_row.value == 5
    assert length.value == 15
    _safe_call(_LIB.FLC_BoosterFreePredictForFileResult(result))
    _safe_call(_LIB.FLC_BoosterFreePredictForFileResult(ctypes.POINTER(ctypes.c_double)()))
