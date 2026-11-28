"""Tests for O(N) 256-bucket counting FindBin on int8/uint8 inputs and
double-buffered pinned async staging in UploadDense4BitColumnsColMajor.

1. For int8 and uint8 dense matrices (row-major, column-major, multi-matrix),
   Dataset construction via the 256-bucket counting path must produce
   byte-for-byte identical serialized Datasets and identical trained models to
   converting the same matrix to float64 (which runs the classic double
   radix-sort FindBin path).
2. For wide 4-bit dense CUDA datasets exceeding the pinned staging threshold,
   double-buffered pinned async column-major upload must produce bit-identical
   trained models to colmajor_direct:off in both compact and mask regimes, for
   both even and odd row counts.
"""

import ctypes
import hashlib
import os

import falcata as flc
import numpy as np
import pytest
from falcata.basic import _LIB, _c_float_array, _c_str, _np2d_to_np1d, _safe_call

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)


def _dataset_binary_md5(X, params, tmp_path, tag, categorical_feature="auto"):
    ds = flc.Dataset(
        X, params=params, categorical_feature=categorical_feature, free_raw_data=False
    )
    ds.construct()
    path = tmp_path / f"{tag}.bin"
    ds.save_binary(str(path))
    return hashlib.md5(path.read_bytes()).hexdigest()


def _model_md5(X, y, params, rounds=8, categorical_feature="auto"):
    ds = flc.Dataset(X, label=y, params=params, categorical_feature=categorical_feature)
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [
        line
        for line in booster.model_to_string().splitlines()
        if "cuda_plan" not in line
    ]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _make_edge_case_matrix(dtype, rows=4096, seed=17, sentinel=None):
    rng = np.random.default_rng(seed)
    is_signed = np.issubdtype(dtype, np.signedinteger)
    lo = -128 if is_signed else 0
    hi = 128 if is_signed else 256

    cols = []
    # 0: full range uniform
    cols.append(rng.integers(lo, hi, size=rows, dtype=dtype))
    # 1: small cardinality (Numerai-like 0..4)
    cols.append(rng.integers(0, 5, size=rows, dtype=dtype))
    # 2: all-zero column
    cols.append(np.zeros(rows, dtype=dtype))
    # 3: constant positive column
    cols.append(np.full(rows, 7, dtype=dtype))
    # 4: constant negative (int8) or high constant (uint8)
    cols.append(np.full(rows, -5 if is_signed else 200, dtype=dtype))
    # 5: negative + positive with ZERO zeros (exercises zero-crossing insertion when zero_cnt == 0)
    if is_signed:
        neg_pos = np.where(
            rng.random(rows) < 0.5,
            rng.integers(-20, 0, size=rows),
            rng.integers(1, 21, size=rows),
        )
        cols.append(neg_pos.astype(dtype))
    else:
        cols.append(rng.integers(1, 21, size=rows, dtype=dtype))
    # 6: strictly negative with zeros (int8) or high positive with zeros (uint8)
    c6 = (
        rng.integers(-30, 0, size=rows, dtype=dtype)
        if is_signed
        else rng.integers(100, 150, size=rows, dtype=dtype)
    )
    c6[rng.random(rows) < 0.3] = 0
    cols.append(c6)
    # 7: 85% single nonzero value (most_freq_bin != default_bin -> exercises GetByteMostFreqMask)
    mask7 = rng.random(rows) < 0.15
    c7 = np.full(rows, 42, dtype=dtype)
    c7[mask7] = rng.integers(0, 10, size=int(mask7.sum()), dtype=dtype)
    cols.append(c7)
    # 8 & 9: mutually exclusive sparse columns (exercises EFB bundling)
    c8 = np.zeros(rows, dtype=dtype)
    c9 = np.zeros(rows, dtype=dtype)
    c8[: rows // 4] = rng.integers(1, 10, size=rows // 4, dtype=dtype)
    c9[rows // 2 : 3 * rows // 4] = rng.integers(1, 10, size=rows // 4, dtype=dtype)
    cols.append(c8)
    cols.append(c9)
    # 10: categorical-friendly non-negative integers (plus a few negatives for int8)
    c10 = rng.integers(0, 12, size=rows, dtype=dtype)
    if is_signed:
        c10[rng.random(rows) < 0.05] = -3
    cols.append(c10)
    # 11: sparse high-range categorical
    c11 = np.zeros(rows, dtype=dtype)
    idx11 = rng.choice(rows, size=rows // 5, replace=False)
    c11[idx11] = rng.integers(1, 120, size=len(idx11), dtype=dtype)
    cols.append(c11)

    if sentinel is not None:
        # 12: column with sentinel values mixed in
        c12 = rng.integers(lo, hi, size=rows, dtype=dtype)
        c12[rng.random(rows) < 0.2] = dtype(sentinel)
        cols.append(c12)
        # 13: all-sentinel column (non_na_cnt == 0, zero_cnt == 0 when !zero_as_missing)
        cols.append(np.full(rows, sentinel, dtype=dtype))

    return np.ascontiguousarray(np.column_stack(cols))


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("max_bin", [5, 15, 63, 255])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    ("use_missing", "zero_as_missing"),
    [(True, False), (True, True), (False, False)],
)
def test_int8_uint8_counting_findbin_matches_float64_binary(
    tmp_path, dtype, max_bin, order, use_missing, zero_as_missing
):
    X = _make_edge_case_matrix(dtype, rows=4096, seed=11)
    if order == "F":
        X = np.asfortranarray(X)
    X_f64 = np.array(X, dtype=np.float64, order=order)

    params = {
        "device_type": "cpu",
        "max_bin": max_bin,
        "min_data_in_bin": 3,
        "min_data_in_leaf": 10,
        "use_missing": use_missing,
        "zero_as_missing": zero_as_missing,
        "enable_bundle": True,
        "bin_construct_sample_cnt": 2048,
        "data_random_seed": 7,
        "verbose": -1,
    }
    cat_cols = [10, 11]
    h_byte = _dataset_binary_md5(
        X, params, tmp_path, "byte", categorical_feature=cat_cols
    )
    h_f64 = _dataset_binary_md5(
        X_f64, params, tmp_path, "f64", categorical_feature=cat_cols
    )
    assert h_byte == h_f64


@pytest.mark.parametrize(
    "dtype,sentinel", [(np.int8, -128), (np.int8, 127), (np.uint8, 255)]
)
@pytest.mark.parametrize("zero_as_missing", [False, True])
def test_int8_uint8_counting_findbin_missing_sentinel(
    tmp_path, dtype, sentinel, zero_as_missing
):
    X = _make_edge_case_matrix(dtype, rows=4096, seed=29, sentinel=sentinel)
    X_f64 = X.astype(np.float64)
    X_f64[X == dtype(sentinel)] = np.nan

    base_params = {
        "device_type": "cpu",
        "max_bin": 31,
        "min_data_in_bin": 3,
        "min_data_in_leaf": 10,
        "use_missing": True,
        "zero_as_missing": zero_as_missing,
        "enable_bundle": True,
        "bin_construct_sample_cnt": 2048,
        "data_random_seed": 13,
        "verbose": -1,
    }
    cat_cols = [10, 11]
    h_byte = _dataset_binary_md5(
        X,
        {**base_params, "missing_sentinel": str(sentinel)},
        tmp_path,
        "byte_sentinel",
        categorical_feature=cat_cols,
    )
    h_f64 = _dataset_binary_md5(
        X_f64, base_params, tmp_path, "f64_nan", categorical_feature=cat_cols
    )
    assert h_byte == h_f64


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_int8_uint8_counting_findbin_multi_matrix_mixed_layout(tmp_path, dtype):
    X = _make_edge_case_matrix(dtype, rows=4096, seed=41)
    X1 = np.ascontiguousarray(X[:2000])
    X2 = np.asfortranarray(X[2000:])
    X_f64 = X.astype(np.float64)

    params = {
        "device_type": "cpu",
        "max_bin": 63,
        "min_data_in_bin": 3,
        "min_data_in_leaf": 10,
        "enable_bundle": True,
        "bin_construct_sample_cnt": 2048,
        "data_random_seed": 19,
        "verbose": -1,
    }
    h_f64 = _dataset_binary_md5(
        X_f64, params, tmp_path, "single_f64", categorical_feature=[10, 11]
    )

    # Call FLC_DatasetCreateFromMats directly to verify mixed C/F multi-matrix int8/uint8 sampling.
    mats = [X1, X2]
    nrow = np.array([m.shape[0] for m in mats], dtype=np.int32)
    ptrs = (ctypes.c_void_p * len(mats))()
    layouts = (ctypes.c_int * len(mats))()
    holders = []
    dtype_code = -1
    for i, m in enumerate(mats):
        m1d, layout = _np2d_to_np1d(m)
        ptr, code, holder = _c_float_array(m1d)
        ptrs[i] = ctypes.cast(ptr, ctypes.c_void_p).value
        layouts[i] = layout
        dtype_code = code
        holders.append(holder)
    params_str = (
        " ".join(f"{k}={v}" for k, v in params.items()) + " categorical_column=10,11"
    )
    handle = ctypes.c_void_p()
    _safe_call(
        _LIB.FLC_DatasetCreateFromMats(
            ctypes.c_int32(len(mats)),
            ctypes.cast(ptrs, ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
            ctypes.c_int(dtype_code),
            nrow.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(X.shape[1]),
            layouts,
            _c_str(params_str),
            None,
            ctypes.byref(handle),
        )
    )
    try:
        multi_path = tmp_path / "multi.bin"
        _safe_call(_LIB.FLC_DatasetSaveBinary(handle, _c_str(str(multi_path))))
        h_multi = hashlib.md5(multi_path.read_bytes()).hexdigest()
    finally:
        _safe_call(_LIB.FLC_DatasetFree(handle))
    assert h_multi == h_f64


@_REQUIRES_CUDA
@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("max_bin", [5, 15, 63, 255])
def test_cuda_int8_uint8_counting_findbin_tree_md5_matches_float64(dtype, max_bin):
    X = _make_edge_case_matrix(dtype, rows=8192, seed=53)
    X_f64 = X.astype(np.float64)
    rng = np.random.default_rng(54)
    y = (
        0.02 * X_f64[:, 0]
        + 0.3 * X_f64[:, 1]
        - 0.05 * X_f64[:, 5]
        + 0.2 * rng.standard_normal(X.shape[0])
    ).astype(np.float32)

    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "max_bin": max_bin,
        "quant_mode": "stochastic",
        "bin_construct_sample_cnt": 4096,
        "data_random_seed": 23,
        "seed": 23,
        "verbose": -1,
    }
    md5_byte = _model_md5(X, y, params, rounds=10, categorical_feature=[10])
    md5_f64 = _model_md5(X_f64, y, params, rounds=10, categorical_feature=[10])
    assert md5_byte == md5_f64


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("rows", "cols"), [(1_100_000, 80), (1_100_001, 80), (2_200_001, 12)]
)
@pytest.mark.parametrize("feature_fraction", [0.5, 1.0])
def test_cuda_colmajor_pinned_async_upload_bit_identical(rows, cols, feature_fraction):
    """Multi-slab (<1 MiB/col pinned staged) and direct async (>=1 MiB/col) 4-bit
    column-major uploads must match colmajor_direct:off for even and odd row counts.
    """
    rng = np.random.default_rng(71)
    X = rng.integers(0, 5, size=(rows, cols), dtype=np.int8)
    y = (
        0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.15 * X[:, 2] + rng.standard_normal(rows)
    ).astype(np.float32)

    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 100,
        "learning_rate": 0.1,
        "max_bin": 15,
        "feature_fraction": feature_fraction,
        "quant_mode": "stochastic",
        "seed": 11,
        "verbose": -1,
    }
    on = _model_md5(X, y, {**params, "cuda_plan": "auto"}, rounds=5)
    off = _model_md5(
        X, y, {**params, "cuda_plan": "auto,colmajor_direct:off"}, rounds=5
    )
    assert on == off
