# coding: utf-8
"""The push API's finish state.

A Dataset filled with FLC_DatasetPushRows* finishes loading (FinishLoad merges the per-thread push
buffers) when a push reaches its last row, or on FLC_DatasetMarkFinished when
FLC_DatasetSetWaitForManualFinish is set. Two ways to get it wrong were silent:
- rows pushed after the finish were ignored (pushing the last rows first dropped the rest);
- an unfinished Dataset could be trained on, which read uninitialized bin state.
Both are errors now.
"""

import ctypes

import numpy as np
import pytest

import falcata as flc
from falcata.basic import _LIB, FalcataError, _c_str, _safe_call

NROW, NCOL = 1000, 3
PARAMS = {"objective": "binary", "num_leaves": 7, "min_data_in_leaf": 5, "verbose": -1, "device_type": "cpu"}


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(NROW, NCOL))
    X[rng.random(X.shape) < 0.8] = 0.0  # sparse columns: their bins only exist after FinishLoad
    return X, (X[:, 0] > 0).astype(np.float32)


def _reference(X, y):
    return flc.Dataset(X, label=y, params={"verbose": -1, "min_data_in_leaf": 5}, free_raw_data=False).construct()


def _empty_like(reference, manual_finish=False):
    handle = ctypes.c_void_p()
    _safe_call(_LIB.FLC_DatasetCreateByReference(reference._handle, ctypes.c_int64(NROW), ctypes.byref(handle)))
    if manual_finish:
        _safe_call(_LIB.FLC_DatasetSetWaitForManualFinish(handle, ctypes.c_int(1)))
    return handle


def _push(handle, X, start, nrow):
    block = np.ascontiguousarray(X[start : start + nrow])
    _safe_call(
        _LIB.FLC_DatasetPushRows(
            handle,
            block.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(1),
            ctypes.c_int32(nrow),
            ctypes.c_int32(NCOL),
            ctypes.c_int32(start),
        )
    )


def _train(handle, y):
    _safe_call(
        _LIB.FLC_DatasetSetField(
            handle, _c_str("label"), y.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(NROW), ctypes.c_int(0)
        )
    )
    booster = ctypes.c_void_p()
    params = " ".join(f"{k}={v}" for k, v in PARAMS.items())
    _safe_call(_LIB.FLC_BoosterCreate(handle, _c_str(params), ctypes.byref(booster)))
    finished = ctypes.c_int()
    for _ in range(5):
        _safe_call(_LIB.FLC_BoosterUpdateOneIter(booster, ctypes.byref(finished)))
    out_len = ctypes.c_int64()
    buffer = ctypes.create_string_buffer(1 << 20)
    _safe_call(
        _LIB.FLC_BoosterSaveModelToString(
            booster,
            ctypes.c_int(0),
            ctypes.c_int(-1),
            ctypes.c_int(0),
            ctypes.c_int64(1 << 20),
            ctypes.byref(out_len),
            buffer,
        )
    )
    _safe_call(_LIB.FLC_BoosterFree(booster))
    return flc.Booster(model_str=buffer.value.decode())


def test_push_after_the_automatic_finish_is_an_error():
    X, y = _data()
    handle = _empty_like(_reference(X, y))
    try:
        _push(handle, X, 500, 500)  # reaches the last row: the Dataset finishes
        with pytest.raises(FalcataError, match="already finished loading"):
            _push(handle, X, 0, 500)  # used to be ignored, leaving rows 0-499 empty
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))


def test_pushing_into_a_constructed_dataset_is_an_error():
    X, y = _data()
    reference = _reference(X, y)
    with pytest.raises(FalcataError, match="already finished loading"):
        _push(reference._handle, X, 0, 10)


@pytest.mark.parametrize("manual_finish", [False, True])
def test_training_on_an_unfinished_dataset_is_an_error(manual_finish):
    X, y = _data()
    handle = _empty_like(_reference(X, y), manual_finish=manual_finish)
    try:
        # all rows without MarkFinished, or only the first half without the manual flag
        _push(handle, X, 0, NROW if manual_finish else 500)
        with pytest.raises(FalcataError, match="has not finished loading"):
            _train(handle, y)
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))


def test_manual_finish_accepts_rows_in_any_order():
    X, y = _data()
    reference = _reference(X, y)
    expected = flc.train(PARAMS, reference, 5).predict(X)
    handle = _empty_like(reference, manual_finish=True)
    try:
        _push(handle, X, 500, 500)
        _push(handle, X, 0, 500)
        _safe_call(_LIB.FLC_DatasetMarkFinished(handle))
        np.testing.assert_allclose(_train(handle, y).predict(X), expected)
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))
