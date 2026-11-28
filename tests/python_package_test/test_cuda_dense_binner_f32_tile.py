"""Bit-identity tests for construct_tile_bin_f32: tiled float32/16-bit GPU binning + exact float32 bounds.

Verifies that enabling construct_tile_bin_f32 produces 100% byte-identical Dataset
binaries and identical trained models vs both construct_tile_bin_f32:off (double
bounds, non-tiled kernel) and gpu_construct:off (host binning path) across:
  - IEEE-754 float32 edge values (NaNs, +-0, +-inf, subnormals, +-FLT_MAX,
    values straddling double bin boundaries, forced non-float32 double bounds)
  - All three float32 binary-search tiers: r_init <= 32 (warp shuffle),
    32 < r_init <= 256 (per-warp shared bounds), r_init > 256 (global/L1)
  - 4-bit, 8-bit, and 16-bit group bin widths
  - zero_as_missing, use_missing, categorical columns, trivial/dropped columns
  - Row-major (tile kernel), col-major (f32-bounds chunk kernel), multi-column EFB groups
  - Chunked construction with odd start rows (parity == 1)
  - int16, uint16, and float16 tiled 64x64 LUT binning
"""

import json
import os
import re

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

BASE_PARAMS = {
    "objective": "regression",
    "device_type": "cuda",
    "num_leaves": 15,
    "max_depth": 5,
    "learning_rate": 0.1,
    "verbosity": -1,
    "seed": 7,
    "num_threads": 4,
    "quant_mode": "stochastic",
    "min_data_in_bin": 1,
}


def _build_binary(X, y, params, path, categorical_feature="auto"):
    ds = flc.Dataset(
        X,
        label=y,
        params=params,
        categorical_feature=categorical_feature,
        free_raw_data=False,
    )
    ds.construct()
    ds.save_binary(path)
    return ds, path.read_bytes()


def _train_trees(params, ds, rounds=4):
    model = flc.train(params, ds, num_boost_round=rounds).model_to_string()
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.MULTILINE).split("\nparameters:")[0]


def _edge_float32_matrix(rows, ncol, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, ncol)).astype(np.float32)
    f32_info = np.finfo(np.float32)
    tiny_sub = np.float32(f32_info.smallest_subnormal)
    fmax = np.float32(f32_info.max)

    # Column 0: dense cluster around 1.0f with adjacent ULPs (forces midpoints strictly between float32 values)
    one = np.float32(1.0)
    adj_vals = np.array(
        [
            np.nextafter(one, np.float32(0.0)),
            one,
            np.nextafter(one, np.float32(2.0)),
            np.nextafter(np.nextafter(one, np.float32(2.0)), np.float32(2.0)),
            np.float32(-0.0),
            np.float32(+0.0),
            -tiny_sub,
            tiny_sub,
            np.float32(2.0) * tiny_sub,
            -np.float32(2.0) * tiny_sub,
            -fmax,
            fmax,
            np.float32(-np.inf),
            np.float32(+np.inf),
            np.float32(np.nan),
        ],
        dtype=np.float32,
    )
    X[:, 0] = adj_vals[rng.integers(0, len(adj_vals), size=rows)]

    # Column 1: subnormals and tiny values around zero_as_missing threshold (1e-35)
    z_vals = np.array(
        [
            np.float32(-1e-35),
            np.nextafter(np.float32(-1e-35), np.float32(-1.0)),
            np.nextafter(np.float32(-1e-35), np.float32(0.0)),
            -tiny_sub,
            np.float32(-0.0),
            np.float32(+0.0),
            tiny_sub,
            np.nextafter(np.float32(1e-35), np.float32(0.0)),
            np.float32(1e-35),
            np.nextafter(np.float32(1e-35), np.float32(1.0)),
            np.float32(np.nan),
        ],
        dtype=np.float32,
    )
    X[:, 1] = z_vals[rng.integers(0, len(z_vals), size=rows)]

    # Column 2: constant (trivial / dropped column -> col_group == -1)
    if ncol > 2:
        X[:, 2] = np.float32(3.1415927)

    # Inject NaNs, +-0, +-inf, subnormals across other columns
    if ncol > 5:
        mask = rng.random((rows, ncol - 3))
        X[:, 3:] = np.where(mask < 0.03, np.float32(np.nan), X[:, 3:])
        X[:, 3:] = np.where((mask >= 0.03) & (mask < 0.04), np.float32(0.0), X[:, 3:])
        X[:, 3:] = np.where((mask >= 0.04) & (mask < 0.05), np.float32(-0.0), X[:, 3:])
        X[:, 3:] = np.where((mask >= 0.05) & (mask < 0.055), np.float32(np.inf), X[:, 3:])
        X[:, 3:] = np.where((mask >= 0.055) & (mask < 0.06), np.float32(-np.inf), X[:, 3:])
        X[:, 3:] = np.where((mask >= 0.06) & (mask < 0.065), tiny_sub, X[:, 3:])
        X[:, 3:] = np.where((mask >= 0.065) & (mask < 0.07), -tiny_sub, X[:, 3:])

    y = rng.standard_normal(rows).astype(np.float32)
    return np.ascontiguousarray(X), y


def test_construct_tile_bin_f32_key_parses():
    """construct_tile_bin_f32 accepts on/off and rejects invalid values."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 4)).astype(np.float32)
    y = rng.standard_normal(64).astype(np.float32)
    for good in ("on", "off", "1", "0"):
        p = {
            **BASE_PARAMS,
            "device_type": "cpu",
            "quant_mode": "none",
            "cuda_plan": f"auto,construct_tile_bin_f32:{good}",
        }
        flc.Dataset(X, label=y, params=p).construct()
    bad_p = {
        **BASE_PARAMS,
        "device_type": "cpu",
        "quant_mode": "none",
        "cuda_plan": "auto,construct_tile_bin_f32:maybe",
    }
    with pytest.raises(flc.basic.FalcataError, match="cuda_plan: bad value"):
        flc.Dataset(X, label=y, params=bad_p).construct()


@_REQUIRES_CUDA
@pytest.mark.parametrize("max_bin", [5, 15, 31, 63, 255, 511])
@pytest.mark.parametrize("zero_as_missing", [False, True])
def test_f32_tile_edges_and_tiers_bit_identical_cuda(max_bin, zero_as_missing, tmp_path):
    """Covers all 3 binary-search tiers (<=32 shuffle, <=256 smem, >256 global),
    4/8/16-bit output widths, trivial columns, NaNs/subnormals/+-0/+-inf/ULP boundaries,
    and zero_as_missing."""
    X, y = _edge_float32_matrix(rows=1027, ncol=70, seed=max_bin + int(zero_as_missing))
    base = {**BASE_PARAMS, "max_bin": max_bin, "zero_as_missing": zero_as_missing}
    ds_on, blob_on = _build_binary(X, y, {**base, "cuda_plan": "auto"}, tmp_path / "on.bin")
    ds_off, blob_off = _build_binary(
        X,
        y,
        {**base, "cuda_plan": "auto,construct_tile_bin_f32:off"},
        tmp_path / "off.bin",
    )
    ds_host, blob_host = _build_binary(X, y, {**base, "cuda_plan": "auto,gpu_construct:off"}, tmp_path / "host.bin")
    assert blob_on == blob_off, f"max_bin={max_bin}, zam={zero_as_missing}: tile_f32 != f32_off"
    assert blob_on == blob_host, f"max_bin={max_bin}, zam={zero_as_missing}: tile_f32 != host"
    if max_bin <= 255:
        m_on = _train_trees({**base, "cuda_plan": "auto"}, ds_on)
        m_off = _train_trees({**base, "cuda_plan": "auto,construct_tile_bin_f32:off"}, ds_off)
        m_host = _train_trees({**base, "cuda_plan": "auto,gpu_construct:off"}, ds_host)
        assert m_on == m_off == m_host


@_REQUIRES_CUDA
def test_f32_forced_double_boundaries_bit_identical_cuda(tmp_path):
    """Explicit double bin upper bounds that are NOT representable in float32,
    including out-of-float32-range finite doubles (+-1e100), subnormal-straddling
    doubles (+-1e-300, +-0.75*2^-149), and exact midpoints between adjacent float32s."""
    f32_info = np.finfo(np.float32)
    tiny = np.float32(f32_info.smallest_subnormal)
    fmax = np.float32(f32_info.max)
    one = np.float32(1.0)
    one_up = np.nextafter(one, np.float32(2.0))
    one_dn = np.nextafter(one, np.float32(0.0))

    vals = np.array(
        [
            np.float32(-np.inf),
            -fmax,
            np.nextafter(-fmax, np.float32(0.0)),
            np.float32(-1.0),
            -tiny,
            np.float32(-0.0),
            np.float32(+0.0),
            tiny,
            np.float32(2.0) * tiny,
            np.float32(0.1),
            np.nextafter(np.float32(0.1), np.float32(0.0)),
            np.nextafter(np.float32(0.1), np.float32(1.0)),
            one_dn,
            one,
            one_up,
            np.nextafter(fmax, np.float32(0.0)),
            fmax,
            np.float32(+np.inf),
            np.float32(np.nan),
        ],
        dtype=np.float32,
    )
    rows = 513
    rng = np.random.default_rng(123)
    X = np.empty((rows, 36), dtype=np.float32)
    for c in range(36):
        X[:, c] = vals[rng.integers(0, len(vals), size=rows)]
    y = rng.standard_normal(rows).astype(np.float32)

    forced_bounds = [
        -1e100,
        float(-fmax),
        float(np.nextafter(np.float64(-fmax), np.float64(-np.inf))),
        -1e-300,
        -0.75 * float(tiny),
        -0.25 * float(tiny),
        0.0,
        1e-300,
        0.25 * float(tiny),
        0.75 * float(tiny),
        0.1,  # double 0.1 != float32(0.1)
        1.0 - 2.0**-25,
        1.0 + 2.0**-24,  # exact midpoint between 1.0f and nextafterf(1.0f, 2.0f)
        float(np.nextafter(np.float64(fmax), np.float64(0.0))),
        float(fmax),
        1e100,
    ]
    forced_bounds = sorted(set(forced_bounds))
    forced_json = tmp_path / "forced_bins.json"
    forced_json.write_text(json.dumps([{"feature": c, "bin_upper_bound": forced_bounds} for c in range(36)]))

    params = {**BASE_PARAMS, "max_bin": 63, "forcedbins_filename": str(forced_json)}
    _, blob_on = _build_binary(X, y, {**params, "cuda_plan": "auto"}, tmp_path / "on.bin")
    _, blob_off = _build_binary(
        X,
        y,
        {**params, "cuda_plan": "auto,construct_tile_bin_f32:off"},
        tmp_path / "off.bin",
    )
    _, blob_host = _build_binary(X, y, {**params, "cuda_plan": "auto,gpu_construct:off"}, tmp_path / "host.bin")
    assert blob_on == blob_off
    assert blob_on == blob_host


@_REQUIRES_CUDA
@pytest.mark.parametrize(("rows", "ncol"), [(1, 1), (3, 13), (127, 32), (128, 33), (129, 65)])
@pytest.mark.parametrize("order", ["C", "F"])
def test_f32_shapes_and_orders_bit_identical_cuda(rows, ncol, order, tmp_path):
    """Small, unaligned, tile-boundary, and Fortran-order shapes."""
    X, y = _edge_float32_matrix(rows=rows, ncol=max(ncol, 2), seed=rows * 17 + ncol)
    X = X[:, :ncol]
    X = np.asfortranarray(X) if order == "F" else np.ascontiguousarray(X)
    params = {**BASE_PARAMS, "max_bin": 15}
    _, blob_on = _build_binary(X, y, {**params, "cuda_plan": "auto"}, tmp_path / "on.bin")
    _, blob_off = _build_binary(
        X,
        y,
        {**params, "cuda_plan": "auto,construct_tile_bin_f32:off"},
        tmp_path / "off.bin",
    )
    _, blob_host = _build_binary(X, y, {**params, "cuda_plan": "auto,gpu_construct:off"}, tmp_path / "host.bin")
    assert blob_on == blob_off
    assert blob_on == blob_host


@_REQUIRES_CUDA
def test_f32_categorical_and_efb_multi_col_group_bit_identical_cuda(tmp_path):
    """Categorical float32 columns in tile kernel + EFB-bundled sparse float32 columns
    exercising the non-tiled CUDADenseBinChunkKernel with USE_F32_BOUNDS=true."""
    rng = np.random.default_rng(99)
    rows = 1501
    # Mix categorical (with edge values) and numerical columns
    X = rng.standard_normal((rows, 40)).astype(np.float32)
    cats = [1, 17, 35]
    for c in cats:
        col = rng.integers(0, 20, size=rows).astype(np.float32)
        u = rng.random(rows)
        col[u < 0.05] = np.nan
        col[(u >= 0.05) & (u < 0.08)] = -1.0
        col[(u >= 0.08) & (u < 0.10)] = -0.5
        col[(u >= 0.10) & (u < 0.12)] = 3.7
        col[(u >= 0.12) & (u < 0.13)] = 3e9
        col[(u >= 0.13) & (u < 0.14)] = np.inf
        col[(u >= 0.14) & (u < 0.15)] = -np.inf
        X[:, c] = col
    y = rng.standard_normal(rows).astype(np.float32)

    params = {**BASE_PARAMS, "max_bin": 63, "enable_bundle": False}
    _, b_on = _build_binary(X, y, {**params, "cuda_plan": "auto"}, tmp_path / "cat_on.bin", cats)
    _, b_off = _build_binary(
        X,
        y,
        {**params, "cuda_plan": "auto,construct_tile_bin_f32:off"},
        tmp_path / "cat_off.bin",
        cats,
    )
    _, b_host = _build_binary(
        X,
        y,
        {**params, "cuda_plan": "auto,gpu_construct:off"},
        tmp_path / "cat_host.bin",
        cats,
    )
    assert b_on == b_off == b_host

    # Sparse mutually exclusive columns -> EFB bundles into multi-column groups
    X_efb = np.zeros((rows, 20), dtype=np.float32)
    for c in range(20):
        idx = np.arange(c, rows, 20)
        X_efb[idx, c] = rng.standard_normal(len(idx)).astype(np.float32) + 2.0
    params_efb = {
        **BASE_PARAMS,
        "max_bin": 15,
        "enable_bundle": True,
        "max_conflict_rate": 0.0,
    }
    _, e_on = _build_binary(X_efb, y, {**params_efb, "cuda_plan": "auto"}, tmp_path / "efb_on.bin")
    _, e_off = _build_binary(
        X_efb,
        y,
        {**params_efb, "cuda_plan": "auto,construct_tile_bin_f32:off"},
        tmp_path / "efb_off.bin",
    )
    _, e_host = _build_binary(
        X_efb,
        y,
        {**params_efb, "cuda_plan": "auto,gpu_construct:off"},
        tmp_path / "efb_host.bin",
    )
    assert e_on == e_off == e_host


@_REQUIRES_CUDA
@pytest.mark.parametrize("max_bin", [15, 255])
def test_f32_chunked_odd_parity_bit_identical_cuda(max_bin, tmp_path):
    """Multi-chunk construction with odd chunk boundaries (parity == 1) matches
    single-chunk and host construction byte-for-byte."""
    X, y = _edge_float32_matrix(rows=777, ncol=40, seed=42 + max_bin)
    chunks = [X[:137], X[137:290], X[290:513], X[513:]]  # odd start rows 137, 513
    params = {**BASE_PARAMS, "max_bin": max_bin}

    _, b_single = _build_binary(X, y, {**params, "cuda_plan": "auto"}, tmp_path / "single.bin")
    _, b_chunks_on = _build_binary(chunks, y, {**params, "cuda_plan": "auto"}, tmp_path / "chunks_on.bin")
    _, b_chunks_off = _build_binary(
        chunks,
        y,
        {**params, "cuda_plan": "auto,construct_tile_bin_f32:off"},
        tmp_path / "chunks_off.bin",
    )
    _, b_chunks_host = _build_binary(
        chunks,
        y,
        {**params, "cuda_plan": "auto,gpu_construct:off"},
        tmp_path / "chunks_host.bin",
    )
    assert b_single == b_chunks_on == b_chunks_off == b_chunks_host


@_REQUIRES_CUDA
@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.float16])
@pytest.mark.parametrize("max_bin", [15, 255])
def test_16bit_tile_bit_identical_cuda(dtype, max_bin, tmp_path):
    """Row-major int16/uint16/float16 tiled 64x64 LUT kernel matches off and host paths."""
    rng = np.random.default_rng(55 + max_bin)
    rows, ncol = 1029, 75  # > 64 cols (multiple tiles), odd rows
    if dtype == np.int16:
        X = rng.integers(-5000, 5000, size=(rows, ncol), dtype=np.int16)
        X[:, 3] = 42  # trivial column
    elif dtype == np.uint16:
        X = rng.integers(0, 10000, size=(rows, ncol), dtype=np.uint16)
        X[:, 3] = 7  # trivial column
    else:
        X = rng.standard_normal((rows, ncol)).astype(np.float16)
        u = rng.random((rows, ncol))
        X = np.where(u < 0.04, np.float16(np.nan), X)
        X = np.where((u >= 0.04) & (u < 0.05), np.float16(np.inf), X)
        X = np.where((u >= 0.05) & (u < 0.06), np.float16(-np.inf), X)
        X = np.where((u >= 0.06) & (u < 0.07), np.float16(-0.0), X)
        X[:, 3] = np.float16(1.5)  # trivial column
    X = np.ascontiguousarray(X)
    y = rng.standard_normal(rows).astype(np.float32)

    params = {**BASE_PARAMS, "max_bin": max_bin}
    ds_on, b_on = _build_binary(X, y, {**params, "cuda_plan": "auto"}, tmp_path / "on.bin")
    ds_off, b_off = _build_binary(
        X,
        y,
        {**params, "cuda_plan": "auto,construct_tile_bin_f32:off"},
        tmp_path / "off.bin",
    )
    ds_host, b_host = _build_binary(X, y, {**params, "cuda_plan": "auto,gpu_construct:off"}, tmp_path / "host.bin")
    assert b_on == b_off == b_host
    assert _train_trees({**params, "cuda_plan": "auto"}, ds_on) == _train_trees(
        {**params, "cuda_plan": "auto,construct_tile_bin_f32:off"}, ds_off
    )
    assert _train_trees({**params, "cuda_plan": "auto"}, ds_on) == _train_trees(
        {**params, "cuda_plan": "auto,gpu_construct:off"}, ds_host
    )
