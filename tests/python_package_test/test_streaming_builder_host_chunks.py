# coding: utf-8
"""The streaming builder from HOST chunks is byte-identical to the host build, and its bin sample is small.

FLC_DatasetDeviceBuilder* keeps the bin-construction sample as the raw sampled cells (one element per cell,
column-major) and expands it into values and row ids one column at a time while the bin mappers are found and
again, per column, for feature bundling. These tests check that this yields exactly the dataset of the classic
path (lgb.Dataset on the whole matrix: the samplers expand every column at once) for every input type and the
options that change what the sample feeds: missing handling, zero as missing, a missing sentinel, categorical
columns, forced bins, max_bin per feature, pre-filtering, ignored columns, sub-sampling and columns that bundle;
both chunk layouts. Serialized datasets are compared byte for byte, and the models trained on them as text.

A last test checks the host memory of a wide build: the peak grows by about the sampled cells themselves, not by
the 12 more bytes per cell the all-columns expansion took. Everything needs a CUDA build (TASK=cuda); the shapes
stay tiny (a few MB of device memory).
"""

import ctypes
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import falcata as lgb
from falcata.basic import _LIB, _param_dict_to_str, _safe_call

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

ROWS = 3001
COLS = 24
CHUNK_SIZES = [941, 1, 1063, 996]
assert sum(CHUNK_SIZES) == ROWS
_DTYPE_CODE = {
    np.dtype(np.float32): 0,
    np.dtype(np.float64): 1,
    np.dtype(np.int8): 4,
    np.dtype(np.int16): 5,
    np.dtype(np.uint8): 6,
    np.dtype(np.uint16): 7,
    np.dtype(np.float16): 8,
}


def _params(**extra):
    p = {
        "objective": "regression",
        "device_type": "cuda",
        "deterministic": True,
        "seed": 42,
        "num_leaves": 15,
        "max_bin": 63,
        "verbosity": -1,
        "num_threads": 4,
    }
    p.update(extra)
    return p


def _build_from_host_chunks(X, y, params, row_major, reference=None):
    """The C API sequence a host-chunk caller runs: sample every chunk, then push every chunk."""
    n, p = X.shape
    bounds = []
    start = 0
    for size in CHUNK_SIZES * (1 + n // sum(CHUNK_SIZES)):
        if start >= n:
            break
        bounds.append((start, min(n, start + size)))
        start += size
    chunks = [np.ascontiguousarray(X[a:b] if row_major else X[a:b].T) for a, b in bounds]
    builder = ctypes.c_void_p()
    ref = None if reference is None else reference.construct()._handle
    _safe_call(
        _LIB.FLC_DatasetDeviceBuilderCreate(
            ctypes.c_int32(n),
            ctypes.c_int32(p),
            ctypes.c_int(_DTYPE_CODE[X.dtype]),
            ctypes.c_char_p(_param_dict_to_str(params).encode()),
            ref,
            ctypes.byref(builder),
        )
    )
    handle = ctypes.c_void_p()
    try:
        if reference is None:
            for c in chunks:
                m = c.shape[0] if row_major else c.shape[1]
                _safe_call(
                    _LIB.FLC_DatasetDeviceBuilderSampleChunk(
                        builder, ctypes.c_void_p(c.ctypes.data), ctypes.c_int32(m), ctypes.c_int(row_major), 0
                    )
                )
        for c in chunks:
            m = c.shape[0] if row_major else c.shape[1]
            _safe_call(
                _LIB.FLC_DatasetDeviceBuilderPushChunk(
                    builder, ctypes.c_void_p(c.ctypes.data), ctypes.c_int32(m), ctypes.c_int(row_major), 0
                )
            )
        _safe_call(_LIB.FLC_DatasetDeviceBuilderFinish(builder, ctypes.byref(handle)))
    except Exception:
        _LIB.FLC_DatasetDeviceBuilderFree(builder)
        raise
    ds = lgb.Dataset(None, reference=reference, params=dict(params), free_raw_data=False)
    ds._handle = handle
    ds.set_label(y)
    return ds


def _float_data(seed, dtype=np.float32, cols=COLS, sparse_cols=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(ROWS, cols))
    X[rng.random(X.shape) < 0.1] = 0.0
    X[rng.random(X.shape) < 0.05] = np.nan
    X[:, 1] = rng.integers(0, 3, size=ROWS)  # heavily tied
    X[:, 2] = 0.0  # constant
    X[:, 3] = np.where(rng.random(ROWS) < 0.6, 7.5, X[:, 3])  # most frequent bin is not the zero bin
    X[rng.random(ROWS) < 0.3, 4] = np.nan  # NaN-heavy
    X[0, 5] = 1e9
    if sparse_cols:
        # mostly-zero columns with nearly disjoint support: feature bundling merges them
        X[:, cols - sparse_cols :] = 0.0
        owner = rng.integers(0, sparse_cols, size=ROWS)
        on = rng.random(ROWS) < 0.5
        X[np.arange(ROWS)[on], cols - sparse_cols + owner[on]] = rng.integers(1, 5, size=on.sum())
    y = np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 3]) * 0.1 + rng.normal(size=ROWS) * 0.1
    return X.astype(dtype), y


def _int_data(seed, dtype, sentinel=None):
    rng = np.random.default_rng(seed)
    info = np.iinfo(dtype)
    X = rng.integers(max(info.min, -20), min(info.max, 40), size=(ROWS, COLS)).astype(dtype)
    X[rng.random(X.shape) < 0.15] = 0
    if sentinel is not None:
        X[rng.random(X.shape) < 0.07] = sentinel
    y = X[:, 0].astype(np.float64) + rng.normal(size=ROWS) * 0.1
    return X, y


def _half_data(seed):
    X, y = _float_data(seed, np.float64)
    X[:, 6] = -0.0  # negative zeros are skipped like zeros
    X[:3, 7] = 6e-8  # half subnormals are kept
    with np.errstate(over="ignore"):
        return X.astype(np.float16), y


def _assert_same_as_host_build(X, y, params, tmp_path, row_major):
    chunked = _build_from_host_chunks(X, y, params, row_major).construct()
    host = lgb.Dataset(X, label=y, params=dict(params), free_raw_data=False).construct()
    chunked.save_binary(tmp_path / "chunked.bin")
    host.save_binary(tmp_path / "host.bin")
    assert (tmp_path / "chunked.bin").read_bytes() == (tmp_path / "host.bin").read_bytes()
    # the datasets carry their forced bins; the wrapper rejects forcedbins_filename on a constructed handle
    train_params = {k: v for k, v in params.items() if k != "forcedbins_filename"}
    m_chunked = lgb.train(train_params, chunked, num_boost_round=6).model_to_string()
    m_host = lgb.train(train_params, host, num_boost_round=6).model_to_string()
    assert m_chunked == m_host
    # a validation set through the builder takes the training set's bins
    valid = _build_from_host_chunks(X[::3], y[::3], params, row_major, reference=chunked).construct()
    valid_host = lgb.Dataset(X[::3], label=y[::3], reference=host, params=dict(params)).construct()
    valid.save_binary(tmp_path / "valid.bin")
    valid_host.save_binary(tmp_path / "valid_host.bin")
    assert (tmp_path / "valid.bin").read_bytes() == (tmp_path / "valid_host.bin").read_bytes()


_CASES = {
    "float32": (lambda: _float_data(0), {}),
    "float64": (lambda: _float_data(1, np.float64), {}),
    "float16": (lambda: _half_data(2), {}),
    "int8_sentinel": (lambda: _int_data(3, np.int8, -1), {"missing_sentinel": "-1"}),
    "uint8": (lambda: _int_data(4, np.uint8), {}),
    "int16_sentinel": (lambda: _int_data(5, np.int16, -7), {"missing_sentinel": "-7"}),
    "uint16": (lambda: _int_data(6, np.uint16), {}),
    "sub_sample": (lambda: _float_data(7), {"bin_construct_sample_cnt": 1700}),
    "zero_as_missing": (lambda: _float_data(8), {"zero_as_missing": True}),
    "no_missing": (lambda: _float_data(9), {"use_missing": False}),
    "max_bin_by_feature": (lambda: _float_data(10), {"max_bin_by_feature": [5 + (i % 7) * 9 for i in range(COLS)]}),
    "categorical": (lambda: _float_data(11), {"categorical_feature": [1, 3]}),
    "pre_filter": (lambda: _float_data(12), {"feature_pre_filter": True, "min_data_in_leaf": 400}),
    "ignore_column": (lambda: _float_data(13), {"ignore_column": "2,5"}),
    "bundled": (lambda: _float_data(14, cols=40, sparse_cols=16), {}),
    "bundled_sub_sample": (lambda: _float_data(15, cols=40, sparse_cols=16), {"bin_construct_sample_cnt": 1500}),
    "bundle_disabled": (lambda: _float_data(16, cols=40, sparse_cols=16), {"enable_bundle": False}),
    "max_bin_255": (lambda: _float_data(17), {"max_bin": 255}),
}


@_REQUIRES_CUDA
@pytest.mark.parametrize("row_major", [0, 1], ids=["col_major_chunks", "row_major_chunks"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_host_chunk_build_matches_host_build(case, row_major, tmp_path):
    make, extra = _CASES[case]
    X, y = make()
    _assert_same_as_host_build(X, y, _params(**extra), tmp_path, row_major)


@_REQUIRES_CUDA
def test_host_chunk_build_forced_bins_matches_host_build(tmp_path):
    forced = tmp_path / "forced.json"
    forced.write_text(
        json.dumps([{"feature": 0, "bin_upper_bound": [-1.0, 0.0, 0.5]}, {"feature": 3, "bin_upper_bound": [7.0]}])
    )
    X, y = _float_data(18)
    _assert_same_as_host_build(X, y, _params(forcedbins_filename=str(forced)), tmp_path, 0)


_PEAK_SCRIPT = textwrap.dedent(
    """
    import json, resource, sys
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    import test_streaming_builder_host_chunks as t
    rows, cols = int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(rows, cols)).astype(np.float32)
    y = rng.normal(size=rows)
    # warm-up: CUDA context, library pools and the small build's own buffers
    t._build_from_host_chunks(X[:3001, :8].copy(), y[:3001], t._params(), 0).construct()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    t._build_from_host_chunks(X, y, t._params(), 0).construct()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(json.dumps({"before": before, "after": after}))
    """
)


@_REQUIRES_CUDA
def test_host_chunk_build_bin_sample_peak_is_about_the_sampled_cells():
    """Peak host growth of a 20,000 x 2,000 float32 build (every row sampled): the sampled cells (4 bytes each, 160 MB)
    plus small per-thread and per-column terms. Expanding all columns at once took about 20 bytes per cell (800 MB)."""
    rows, cols = 20000, 2000
    cells = rows * cols
    out = subprocess.run(
        [sys.executable, "-c", _PEAK_SCRIPT, os.path.dirname(__file__), str(rows), str(cols)],
        capture_output=True,
        text=True,
        check=True,
    )
    r = json.loads(out.stdout.strip().splitlines()[-1])
    growth = r["after"] - r["before"]
    # the chunk copies of the test's own builder (one transposed copy of X per pass) are freed between passes;
    # allow the block, one chunk list (4 bytes per cell) and 256 MB for the rest
    assert growth < 2 * 4 * cells + (256 << 20), f"peak host growth {growth / 2**20:.0f} MiB"
