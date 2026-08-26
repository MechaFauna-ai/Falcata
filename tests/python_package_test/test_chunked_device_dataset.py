# coding: utf-8
"""Chunked device dataset construction is byte-identical to the host build.

A dataset built from row chunks (device-resident, uneven sizes, odd start
rows) must produce exactly the bin mappers and binned values of the same
matrix built in one piece on the host: the serialized dataset binaries are
compared byte for byte, and boosters trained on them must serialize to the
same model text.

The missing_sentinel parameter has host-side semantics too (int8 codes with
a declared sentinel bin exactly like the float matrix with NaN in its
place), so those assertions run everywhere; everything touching device
memory needs a CUDA build (TASK=cuda) and stays tiny -- a few thousand rows,
well under 2 GB of device memory including the CUDA context.
"""

import ctypes
import glob
import os

import numpy as np
import pytest
import scipy.sparse

import falcata as lgb
from falcata.basic import _C_API_DTYPE_INT8, _LIB, FalcataError, _c_str, _safe_call

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

ROWS = 3001
COLS = 17
SENTINEL = -1
# uneven on purpose: chunk boundaries at odd dataset rows exercise the
# nibble-packed (4-bit) parity merge, and a one-row chunk the degenerate pair
CHUNK_SIZES = [941, 1, 1063, 996]
assert sum(CHUNK_SIZES) == ROWS


def _params(max_bin, **extra):
    p = {
        "objective": "regression",
        "device_type": "cuda",
        "deterministic": True,
        "seed": 42,
        "num_leaves": 15,
        "max_bin": max_bin,
        "verbosity": -1,
        "num_threads": 4,
    }
    p.update(extra)
    return p


# --- a minimal __cuda_array_interface__ producer -----------------------------
#
# The gate venvs have no cupy, so device buffers come straight from the CUDA
# runtime via ctypes. Any contiguous numpy array (C or Fortran order) is
# copied up; the interface reports the matching strides.


def _load_cudart():
    names = ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    names += [
        os.path.join(root, "lib64", "libcudart.so") for root in sorted(glob.glob("/usr/local/cuda*"), reverse=True)
    ]
    for name in names:
        try:
            rt = ctypes.CDLL(name)
        except OSError:
            continue
        rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        rt.cudaFree.argtypes = [ctypes.c_void_p]
        return rt
    pytest.skip("no CUDA runtime library found for device-buffer allocation")


class _DeviceMatrix:
    """Device copy of a contiguous 2-D numpy array, exposing __cuda_array_interface__."""

    _MEMCPY_H2D = 1

    def __init__(self, arr: np.ndarray):
        assert arr.ndim == 2
        assert arr.flags["C_CONTIGUOUS"] or arr.flags["F_CONTIGUOUS"]
        self._rt = _load_cudart()
        self._ptr = ctypes.c_void_p()
        status = self._rt.cudaMalloc(ctypes.byref(self._ptr), arr.nbytes)
        assert status == 0, f"cudaMalloc failed with status {status}"
        status = self._rt.cudaMemcpy(self._ptr, ctypes.c_void_p(arr.ctypes.data), arr.nbytes, self._MEMCPY_H2D)
        assert status == 0, f"cudaMemcpy failed with status {status}"
        self.__cuda_array_interface__ = {
            "shape": arr.shape,
            "typestr": arr.dtype.str,
            "data": (self._ptr.value, False),
            "strides": None if arr.flags["C_CONTIGUOUS"] else tuple(arr.strides),
            "version": 2,
        }

    def free(self):
        if self._ptr.value is not None:
            self._rt.cudaFree(self._ptr)
            self._ptr = ctypes.c_void_p()

    def __del__(self):
        self.free()


# --- data + comparison helpers ------------------------------------------------


def _float_data(seed=0):
    """Floats with zeros, NaNs, ties and outliers -- everything binning cares about."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(ROWS, COLS))
    X[rng.random(X.shape) < 0.1] = 0.0
    X[rng.random(X.shape) < 0.05] = np.nan
    X[:, 3] = rng.integers(0, 3, size=ROWS)  # heavily tied column
    X[0, 5] = 1e9
    y = np.nansum(X[:, :4], axis=1) + rng.normal(size=ROWS) * 0.1
    return X, y


def _int8_data(seed=1):
    """int8 feature codes 0..4 with a scattered missing sentinel."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 5, size=(ROWS, COLS)).astype(np.int8)
    X[rng.random(X.shape) < 0.07] = SENTINEL
    y = (X == 4).sum(axis=1) + rng.normal(size=ROWS) * 0.1
    return X, y


def _chunks(X):
    out = []
    start = 0
    for size in CHUNK_SIZES:
        out.append(X[start : start + size])
        start += size
    return out


def _dataset_bytes(ds, tmp_path, name):
    path = tmp_path / f"{name}.bin"
    ds.save_binary(path)
    return path.read_bytes()


def _model_str(ds, params):
    booster = lgb.train(params, ds, num_boost_round=8)
    return booster.model_to_string()


def _assert_dataset_parity(builders, params, tmp_path, train=True):
    """builders: name -> zero-arg callable returning a constructed Dataset."""
    blobs = {}
    models = {}
    for name, build in builders.items():
        ds = build().construct()
        blobs[name] = _dataset_bytes(ds, tmp_path, name)
        if train:
            models[name] = _model_str(ds, params)
    names = list(builders)
    reference = names[0]
    for name in names[1:]:
        assert blobs[name] == blobs[reference], f"serialized dataset of '{name}' differs from '{reference}'"
        if train:
            assert models[name] == models[reference], f"model trained on '{name}' differs from '{reference}'"


# --- device parity (CUDA build) ----------------------------------------------


@_REQUIRES_CUDA
@pytest.mark.parametrize("max_bin", [7, 63])  # 4-bit packed and 8-bit groups
def test_chunked_device_build_matches_host_and_single_device(max_bin, tmp_path):
    X, y = _float_data()
    X32 = X.astype(np.float32)
    params = _params(max_bin)
    holders = []  # keep device buffers alive across construct()

    def host():
        return lgb.Dataset(X32, label=y, params=params)

    def device_single():
        holders.append(_DeviceMatrix(X32))
        return lgb.Dataset(holders[-1], label=y, params=params)

    def device_chunked():
        mats = [_DeviceMatrix(c) for c in _chunks(X32)]
        holders.extend(mats)
        return lgb.Dataset(mats, label=y, params=params)

    def device_chunked_mixed_layout():
        # one Fortran-order chunk: the column-major device path at a row offset
        mats = []
        for i, c in enumerate(_chunks(X32)):
            mats.append(_DeviceMatrix(np.asfortranarray(c) if i == 2 else c))
        holders.extend(mats)
        return lgb.Dataset(mats, label=y, params=params)

    _assert_dataset_parity(
        {
            "host": host,
            "device_single": device_single,
            "device_chunked": device_chunked,
            "device_chunked_mixed_layout": device_chunked_mixed_layout,
        },
        params,
        tmp_path,
    )


@_REQUIRES_CUDA
def test_chunked_device_int8_sentinel_matches_float_nan_builds(tmp_path):
    """The production shape: int8 codes + sentinel streamed as device chunks
    equals both the float-with-NaN host build and the float16 host build."""
    X, y = _int8_data()
    X_nan64 = X.astype(np.float64)
    X_nan64[X == SENTINEL] = np.nan
    X_nan16 = X_nan64.astype(np.float16)
    params = _params(7, missing_sentinel=SENTINEL)
    params_float = _params(7)  # float inputs carry NaN directly
    holders = []

    def float64_nan_host():
        return lgb.Dataset(X_nan64, label=y, params=params_float)

    def float16_nan_host():
        return lgb.Dataset(X_nan16, label=y, params=params_float)

    def int8_sentinel_chunked_device():
        mats = [_DeviceMatrix(c) for c in _chunks(X)]
        holders.extend(mats)
        return lgb.Dataset(mats, label=y, params=params)

    # train only where params match across datasets; dataset bytes are the
    # real assertion and params are not part of the serialized dataset
    _assert_dataset_parity(
        {
            "float64_nan_host": float64_nan_host,
            "float16_nan_host": float16_nan_host,
            "int8_sentinel_chunked_device": int8_sentinel_chunked_device,
        },
        params,
        tmp_path,
        train=False,
    )


@_REQUIRES_CUDA
def test_chunked_device_build_with_reference_matches_host(tmp_path):
    """Valid datasets (bin mappers aligned to a reference) build chunked too."""
    X, y = _float_data()
    X32 = X.astype(np.float32)
    Xv, yv = _float_data(seed=7)
    Xv32 = Xv.astype(np.float32)
    params = _params(31)
    train_ds = lgb.Dataset(X32, label=y, params=params).construct()

    valid_host = lgb.Dataset(Xv32, label=yv, params=params, reference=train_ds).construct()
    mats = [_DeviceMatrix(c) for c in _chunks(Xv32)]
    valid_chunked = lgb.Dataset(mats, label=yv, params=params, reference=train_ds).construct()

    assert _dataset_bytes(valid_chunked, tmp_path, "valid_chunked") == _dataset_bytes(
        valid_host, tmp_path, "valid_host"
    )


@_REQUIRES_CUDA
def test_builder_protocol_enforces_full_sampling_and_row_count():
    """The C API rejects binning before the sample covers every declared row,
    and refuses to finish with rows missing."""
    X = _int8_data()[0][:100]
    dev = _DeviceMatrix(X)
    ptr = ctypes.c_void_p(dev.__cuda_array_interface__["data"][0])
    params = _c_str("device_type=cuda verbosity=-1")
    int8_code = _C_API_DTYPE_INT8

    builder = ctypes.c_void_p()
    _safe_call(
        _LIB.FLC_DatasetDeviceBuilderCreate(
            ctypes.c_int32(200),  # declares 200 rows; we only ever offer 100
            ctypes.c_int32(COLS),
            ctypes.c_int(int8_code),
            params,
            None,
            ctypes.byref(builder),
        )
    )
    _safe_call(
        _LIB.FLC_DatasetDeviceBuilderSampleChunk(builder, ptr, ctypes.c_int32(100), ctypes.c_int(1), ctypes.c_int(1))
    )
    # sampling covered 100 of 200 declared rows: binning must refuse to start
    with pytest.raises(FalcataError, match="sampling phase covered"):
        _safe_call(
            _LIB.FLC_DatasetDeviceBuilderPushChunk(builder, ptr, ctypes.c_int32(100), ctypes.c_int(1), ctypes.c_int(1))
        )
    _safe_call(_LIB.FLC_DatasetDeviceBuilderFree(builder))

    # declare the true row count but push only half: Finish must refuse
    builder = ctypes.c_void_p()
    _safe_call(
        _LIB.FLC_DatasetDeviceBuilderCreate(
            ctypes.c_int32(100),
            ctypes.c_int32(COLS),
            ctypes.c_int(int8_code),
            params,
            None,
            ctypes.byref(builder),
        )
    )
    _safe_call(
        _LIB.FLC_DatasetDeviceBuilderSampleChunk(builder, ptr, ctypes.c_int32(100), ctypes.c_int(1), ctypes.c_int(1))
    )
    _safe_call(
        _LIB.FLC_DatasetDeviceBuilderPushChunk(builder, ptr, ctypes.c_int32(50), ctypes.c_int(1), ctypes.c_int(1))
    )
    out = ctypes.c_void_p()
    with pytest.raises(FalcataError, match="50 of the declared 100"):
        _safe_call(_LIB.FLC_DatasetDeviceBuilderFinish(builder, ctypes.byref(out)))
    _safe_call(_LIB.FLC_DatasetDeviceBuilderFree(builder))
    dev.free()


@_REQUIRES_CUDA
def test_from_mats_device_one_shot_matches_host_c_api(tmp_path):
    """FLC_DatasetCreateFromMatsDevice (all chunks passed at once) equals the
    host FLC_DatasetCreateFromMat build of the concatenated matrix."""
    X = _int8_data()[0]
    params = _c_str("max_bin=7 missing_sentinel=-1 device_type=cuda verbosity=-1 num_threads=4")

    host = ctypes.c_void_p()
    _safe_call(
        _LIB.FLC_DatasetCreateFromMat(
            ctypes.c_void_p(X.ctypes.data),
            ctypes.c_int(_C_API_DTYPE_INT8),
            ctypes.c_int32(X.shape[0]),
            ctypes.c_int32(X.shape[1]),
            ctypes.c_int(1),
            params,
            None,
            ctypes.byref(host),
        )
    )

    chunks = _chunks(X)
    mats = [_DeviceMatrix(c) for c in chunks]
    ptrs = (ctypes.c_void_p * len(mats))(*[m.__cuda_array_interface__["data"][0] for m in mats])
    nrows = (ctypes.c_int32 * len(mats))(*[c.shape[0] for c in chunks])
    layouts = (ctypes.c_int * len(mats))(*([1] * len(mats)))
    dev = ctypes.c_void_p()
    _safe_call(
        _LIB.FLC_DatasetCreateFromMatsDevice(
            ctypes.c_int32(len(mats)),
            ptrs,
            ctypes.c_int(_C_API_DTYPE_INT8),
            nrows,
            ctypes.c_int32(X.shape[1]),
            layouts,
            params,
            None,
            ctypes.byref(dev),
        )
    )

    blobs = {}
    for name, handle in [("host", host), ("dev", dev)]:
        path = tmp_path / f"{name}.bin"
        _safe_call(_LIB.FLC_DatasetSaveBinary(handle, _c_str(str(path))))
        blobs[name] = path.read_bytes()
        _safe_call(_LIB.FLC_DatasetFree(handle))
    assert blobs["dev"] == blobs["host"]


# --- missing_sentinel host semantics (run everywhere) ------------------------


def test_missing_sentinel_host_build_matches_float_nan(tmp_path):
    """int8 + declared sentinel on the plain host path equals the float build
    with NaN in the sentinel's place (CPU, no device involvement)."""
    X, y = _int8_data()
    X_nan = X.astype(np.float64)
    X_nan[X == SENTINEL] = np.nan
    params_int8 = {"max_bin": 7, "verbosity": -1, "missing_sentinel": SENTINEL}
    params_float = {"max_bin": 7, "verbosity": -1}

    ds_int8 = lgb.Dataset(X, label=y, params=params_int8).construct()
    ds_float = lgb.Dataset(X_nan, label=y, params=params_float).construct()

    assert _dataset_bytes(ds_int8, tmp_path, "int8_sentinel") == _dataset_bytes(ds_float, tmp_path, "float_nan")


def test_missing_sentinel_rejects_unsupported_uses():
    X, y = _int8_data()
    # float input: the caller should write NaN directly
    with pytest.raises(FalcataError, match="integer and float16"):
        lgb.Dataset(X.astype(np.float32), label=y, params={"missing_sentinel": SENTINEL, "verbosity": -1}).construct()
    # zero sentinel is indistinguishable from the implicit zero filtering
    with pytest.raises(FalcataError, match="nonzero"):
        lgb.Dataset(X, label=y, params={"missing_sentinel": 0, "verbosity": -1}).construct()
    # sparse input formats do not implement the rewrite and must say so
    csr = scipy.sparse.csr_matrix(X.astype(np.float64))
    with pytest.raises(FalcataError, match="not supported"):
        lgb.Dataset(csr, label=y, params={"missing_sentinel": SENTINEL, "verbosity": -1}).construct()
