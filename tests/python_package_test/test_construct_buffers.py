# coding: utf-8
"""Dataset construction's host buffers on CUDA never change the Dataset.

Three cuda_plan keys decide how construction from host arrays handles its buffers, and none may change a byte of the
Dataset:
  - pin_bins (auto|always|never): whether the group bin storage is page-locked. auto page-locks it only where
    CUDAColumnData copies it per column (num_groups x num_data <= 8 GB), which every shape here is; never makes it
    pageable, huge-page aligned from one transparent huge page up (the "tall" shape's 2.2 MB groups).
  - construct_staging_pool: the GPU binner's page-locked staging blocks are kept for the next construct in the process
    (best fit). The blocks a construct reuses hold the previous construct's bytes.
  - construct_h2d_overlap: the binner's chunk upload runs on its own stream.
The saved binary of every construct must equal the default build's and the host binning path's (gpu_construct:off,
which also fills the 4-bit odd-nibble buffer that the GPU path never touches), byte for byte, and a short training on
it the same model.
"""

import os
import re

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

BASE = {
    "objective": "regression",
    "device_type": "cuda",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.1,
    "verbosity": -1,
    "seed": 5,
    "num_threads": 8,
}
# quantized training: integer histogram sums, so equal Datasets train equal models run to run
QUANT = {"quant_mode": "stochastic"}

# (rows, columns, distinct values, dtype, max_bin): 4-bit and 8-bit groups, odd row counts (the pad nibble), the LUT
# (int8) and float binning kernels
SHAPES = {
    "nibble": (30_003, 300, 6, np.int8, 15),
    "byte": (20_001, 40, 200, np.float32, 255),
    "small": (5_001, 7, 12, np.float64, 15),
    # 2.2 MB groups: one transparent huge page (2 MB on x86-64) or more, so pin_bins:never takes the aligned path
    "tall": (2_200_001, 3, 100, np.float32, 255),
}

PLANS = [
    "auto",
    "auto,pin_bins:always",
    "auto,pin_bins:never",
    "auto,construct_staging_pool:off",
    "auto,construct_h2d_overlap:off",
    "auto,pin_bins:always,construct_staging_pool:off,construct_h2d_overlap:off",
    "auto,gpu_construct:off",
]


def _data(shape, seed=0):
    rows, cols, values, dtype, _ = SHAPES[shape]
    rng = np.random.default_rng(seed)
    X = rng.integers(0, values, size=(rows, cols)).astype(dtype)
    y = X[:, :3].astype(np.float64) @ rng.standard_normal(3) + rng.standard_normal(rows)
    return X, y


def _params(shape, plan):
    return {**BASE, **QUANT, "max_bin": SHAPES[shape][4], "cuda_plan": plan}


def _binary(X, y, params, path):
    ds = flc.Dataset(X, label=y, params=params, free_raw_data=False)
    ds.construct()
    ds.save_binary(path)
    return ds, path.read_bytes()


def _model(params, ds, rounds=5):
    # the parameter dump records the plan string itself
    model = flc.train(params, ds, num_boost_round=rounds).model_to_string()
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.M)


@_REQUIRES_CUDA
@pytest.mark.parametrize("shape", ["nibble", "byte", "tall"])
def test_construct_plan_keys_are_bit_identical_cuda(shape, tmp_path):
    """Every buffer key, alone and all off together, and the host binning path build the default's Dataset and train
    its model."""
    X, y = _data(shape)
    ref_ds, ref = _binary(X, y, _params(shape, "auto"), tmp_path / "ref.bin")
    ref_model = _model(_params(shape, "auto"), ref_ds)
    for i, plan in enumerate(PLANS[1:]):
        ds, blob = _binary(X, y, _params(shape, plan), tmp_path / f"{i}.bin")
        assert blob == ref, f"{shape}: the saved binary differs under {plan}"
        assert _model(_params(shape, plan), ds) == ref_model, f"{shape}: the model differs under {plan}"


@_REQUIRES_CUDA
def test_staging_pool_reuse_across_shapes_cuda(tmp_path):
    """One process constructs shapes that shrink and grow, so the pooled staging blocks are reused with a previous
    construct's bytes in them, handed to requests of other sizes, and dropped for larger ones. Each construct saves
    the bytes of the same matrix built with fresh staging blocks and of the host binning path."""
    order = ["byte", "small", "nibble", "small", "byte", "nibble", "small"]
    pooled = []
    for i, shape in enumerate(order):
        X, y = _data(shape, seed=i)
        pooled.append(_binary(X, y, _params(shape, "auto"), tmp_path / f"pool{i}.bin")[1])
    for i, shape in enumerate(order):
        X, y = _data(shape, seed=i)
        for plan in ("auto,construct_staging_pool:off", "auto,gpu_construct:off"):
            blob = _binary(X, y, _params(shape, plan), tmp_path / f"ref{i}.bin")[1]
            assert pooled[i] == blob, f"construct {i} ({shape}) differs from its {plan} build"


def test_pin_bins_plan_key_parses():
    """pin_bins takes auto|always|never; anything else is refused (the plan is resolved at Dataset construction, so
    this needs no GPU)."""
    X, y = _data("small")
    for good in ("auto", "always", "never"):
        p = {**BASE, "device_type": "cpu", "cuda_plan": f"auto,pin_bins:{good}"}
        flc.Dataset(X, label=y, params=p).construct()
    for bad in ("on", "off", "sometimes"):
        p = {**BASE, "device_type": "cpu", "cuda_plan": f"auto,pin_bins:{bad}"}
        with pytest.raises(flc.basic.FalcataError, match="cuda_plan: bad value"):
            flc.Dataset(X, label=y, params=p).construct()
