# coding: utf-8
"""dense_rowdata trains quantized sparse-ish data on the dense row-wise bin and never changes the model.

cuda_plan key dense_rowdata: a dataset whose features have an average sparse rate of at least 0.25 gets the sparse
row-wise bin on the host's heuristic, which the CUDA learner trains through its per-leaf kernels; with the key,
quantized training takes the dense bin (the batched level kernels) while it needs at most a quarter of the device's
total memory. Layout only: every quantized model must be byte-identical with the key off. Non-quantized training
keeps the sparse bin and its deterministic float construct: the key must not touch it (same model as with the key off,
and the same model on every run). The shapes cover mildly and very sparse columns, NaNs next to zeros, 4-bit columns,
few-value integer columns (a 4-value uniform column alone has a sparse rate of 0.25), bagging and multiclass.
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
    "num_leaves": 63,
    "max_depth": 6,
    "min_data_in_leaf": 5,
    "learning_rate": 0.1,
    "max_bin": 255,
    "quant_mode": "stochastic",
    "verbosity": -1,
    "seed": 5,
}

# (rows, columns, share of columns that are sparse, zero share of those, NaN share of the rest, values per column or 0, params)
SHAPES = {
    "mild": (40_001, 30, 1.0, 0.3, 0.0, 0, {}),
    "sparse_continuous": (40_001, 60, 0.5, 0.9, 0.0, 0, {}),
    "very_sparse_wide": (20_001, 400, 1.0, 0.99, 0.0, 0, {"objective": "binary"}),
    "zeros_and_nans": (40_001, 40, 0.5, 0.8, 0.3, 0, {"feature_fraction": 0.5}),
    "four_bit": (40_001, 50, 0.6, 0.8, 0.0, 4, {"quant_mode": "fixedpoint"}),
    "few_values": (40_001, 60, 0.2, 0.6, 0.0, 4, {"num_leaves": 255, "max_depth": 8, "feature_fraction": 0.5}),
    "bagged_multiclass": (
        30_001,
        30,
        0.7,
        0.7,
        0.0,
        12,
        {"objective": "multiclass", "num_class": 3, "bagging_fraction": 0.7, "bagging_freq": 1},
    ),
}

# the non-quantized checks: the fp64 default, bagged, and feature-sampled
NONQUANT = ["sparse_continuous", "zeros_and_nans", "bagged_multiclass"]

_ROW_DATA_LINE = re.compile(r"CUDARowData: \d+-bit row data|CUDARowData: 4-bit packed row data")


def _data(rows, cols, sparse_cols, zero_frac, nan_frac, levels):
    rng = np.random.default_rng(23)
    X = (
        rng.integers(0, levels, size=(rows, cols)).astype(np.float32)
        if levels
        else rng.standard_normal((rows, cols)).astype(np.float32)
    )
    n_sparse = int(round(sparse_cols * cols))
    X[:, :n_sparse][rng.random((rows, n_sparse)) < zero_frac] = 0.0
    if nan_frac:
        X[:, n_sparse:][rng.random((rows, cols - n_sparse)) < nan_frac] = np.nan
    z = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, -1]) + 0.3 * rng.standard_normal(rows)
    return X, z


def _train(X, z, params, plan, rounds=6):
    p = dict(params)
    if plan:
        p["cuda_plan"] = plan
    y = z
    if p["objective"] == "binary":
        y = (z > np.median(z)).astype(np.float32)
    elif p["objective"] == "multiclass":
        y = np.digitize(z, np.quantile(z, [1 / 3, 2 / 3])).astype(np.float32)
    ds = flc.Dataset(X, label=y, params=p)
    model = flc.train(p, ds, num_boost_round=rounds).model_to_string()
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.M)


def _dense_row_data(capfd, X, z, params, plan):
    """whether training put dense row data on the device (CUDARowData reports only the dense layout)"""
    capfd.readouterr()
    _train(X, z, {**params, "verbosity": 1}, plan, rounds=1)
    out = capfd.readouterr()
    return bool(_ROW_DATA_LINE.search(out.out + out.err))


@_REQUIRES_CUDA
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_dense_rowdata_bit_identical(shape):
    rows, cols, sparse_cols, zero_frac, nan_frac, levels, extra = SHAPES[shape]
    X, z = _data(rows, cols, sparse_cols, zero_frac, nan_frac, levels)
    params = {**BASE, **extra}
    assert _train(X, z, params, "") == _train(X, z, params, "dense_rowdata:off")


@_REQUIRES_CUDA
def test_dense_rowdata_engages_quantized_only(capfd):
    rows, cols, sparse_cols, zero_frac, nan_frac, levels, extra = SHAPES["sparse_continuous"]
    X, z = _data(rows, cols, sparse_cols, zero_frac, nan_frac, levels)
    params = {**BASE, **extra}
    assert _dense_row_data(capfd, X, z, params, "")
    assert not _dense_row_data(capfd, X, z, params, "dense_rowdata:off")
    assert not _dense_row_data(capfd, X, z, {**params, "quant_mode": "none"}, "")


@_REQUIRES_CUDA
@pytest.mark.parametrize("shape", NONQUANT)
def test_dense_rowdata_nonquantized_untouched(shape):
    rows, cols, sparse_cols, zero_frac, nan_frac, levels, extra = SHAPES[shape]
    X, z = _data(rows, cols, sparse_cols, zero_frac, nan_frac, levels)
    params = {**BASE, **extra, "quant_mode": "none"}
    first = _train(X, z, params, "")
    assert first == _train(X, z, params, "")
    assert first == _train(X, z, params, "dense_rowdata:off")


@_REQUIRES_CUDA
def test_dense_rowdata_multi_gpu_decided_once(capfd):
    """num_gpu > 1: NCCLGBDT decides the size rule ONCE for all ranks (the largest rank's rows against the smallest total
    memory over the ranks' devices) and hands it to every rank's Dataset (Dataset::SetDenseRowDataFits), so the ranks
    lay out the same row-wise bin and all-reduce histograms of one shape. On a one-GPU box the topology clamps num_gpu
    to one rank, which still trains through the per-rank Dataset (CopySubrowToDevice) and the decision. Skipped where
    the NCCL library is not loadable (it is dlopen'ed: put libnccl.so.2 on LD_LIBRARY_PATH to run it)."""
    rows, cols, sparse_cols, zero_frac, nan_frac, levels, extra = SHAPES["sparse_continuous"]
    X, z = _data(rows, cols, sparse_cols, zero_frac, nan_frac, levels)
    params = {**BASE, **extra, "num_gpu": 2, "verbosity": 2}
    capfd.readouterr()
    _train(X, z, params, "", rounds=2)
    out = capfd.readouterr()
    log = out.out + out.err
    if "needs the NCCL library" in log:
        pytest.skip("NCCL library not loadable")
    # one decision for the whole training; the rank's dense row data (the parent's single-GPU learner reports too)
    decisions = re.findall(
        r"dense_rowdata \(multi-GPU\): (\d+) rows per rank, smallest device (\d+) bytes: "
        r"dense row-wise bin (fits|does not fit)",
        log,
    )
    assert len(decisions) == 1
    assert decisions[0][2] == "fits"
    n_ranks = re.search(r"Using GPU devices (\d+(?:,\d+)*)", log).group(1).count(",") + 1
    # the rows are split over the ranks that train, not over the node's devices: rank 0's share is ceil(rows / ranks)
    assert int(decisions[0][0]) == -(-rows // n_ranks)
    assert len(_ROW_DATA_LINE.findall(log)) == 1 + n_ranks
