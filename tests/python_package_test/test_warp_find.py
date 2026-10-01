# coding: utf-8
"""The warp-per-task quantized level finder must pick exactly the splits of the block kernel.

cuda_plan key warp_find switches the quantized per-level split finder between one warp per (task, leaf) and one
256-thread block per (task, leaf). Both scan the same integer histograms with the same fp64 gain math, so the
trained model must be byte-identical whichever runs. The sweep covers what the warp kernel branches on: 16- and
32-bit leaf histograms, forward and reverse scans, NaN handling, a stored or unstored most-frequent bin, unused
features, the bagging ridge, and the parameters that switch its fp32 pruning off (max_delta_step) or move the
min-gain cutoff. Only the deterministic (quantized) modes are compared.
"""

import os
import re
import subprocess
import sys

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
    "max_depth": 8,
    "min_data_in_leaf": 20,
    "learning_rate": 0.1,
    "verbosity": -1,
    "seed": 5,
}


def _data(kind, rows, seed):
    rng = np.random.default_rng(seed)
    if kind == "dense":  # float features, up to 255 bins
        X = rng.standard_normal((rows, 30)).astype(np.float32)
    elif kind == "fewbin":  # 5 values, mostly zero: the most-frequent bin is not stored (mfb_offset 1)
        X = rng.integers(0, 5, size=(rows, 40)).astype(np.float32)
        X[rng.random(X.shape) < 0.6] = 0.0
    elif kind == "nan":  # NaN in half the columns: forward and reverse scans with NaN handling
        X = rng.standard_normal((rows, 30)).astype(np.float32)
        X[:, :15][rng.random((rows, 15)) < 0.2] = np.nan
    else:
        raise ValueError(kind)
    w = rng.standard_normal(min(8, X.shape[1]))
    y = np.nan_to_num(X[:, : len(w)]) @ w + rng.standard_normal(rows)
    return X, y.astype(np.float32)


def _train(kind, rows, params, seed=0, rounds=12, X=None, y=None):
    if X is None:
        X, y = _data(kind, rows, seed)
    p = {**BASE, **params}
    model = flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=rounds).model_to_string()
    # the parameter dump records the plan string itself
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.M)


CONFIGS = [
    pytest.param({}, id="plain"),
    pytest.param({"bagging_fraction": 0.7, "bagging_freq": 1}, id="bagging-ridge"),
    pytest.param({"max_delta_step": 0.5}, id="max-delta-step-no-pruning"),
    pytest.param({"lambda_l2": 2.0, "min_gain_to_split": 0.05}, id="l2-min-gain"),
    pytest.param({"feature_fraction": 0.5}, id="feature-sampling"),
    pytest.param({"min_data_in_leaf": 2000}, id="large-min-data"),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])
@pytest.mark.parametrize("kind", ["dense", "fewbin", "nan"])
@pytest.mark.parametrize("config", CONFIGS)
def test_warp_find_is_bit_identical_cuda(kind, config, quant):
    params = {"quant_mode": quant, **config}
    rows = 60_001
    warp = _train(kind, rows, {**params, "cuda_plan": "auto"})
    block = _train(kind, rows, {**params, "cuda_plan": "auto,warp_find:off"})
    assert warp == block


@_REQUIRES_CUDA
def test_warp_find_is_bit_identical_with_32bit_leaf_histograms_cuda():
    """Large leaves use 32-bit histogram bins (int64 packing); a few large leaves keep them in play."""
    params = {"quant_mode": "fixedpoint", "num_leaves": 15, "max_depth": 4}
    warp = _train("dense", 1_200_000, {**params, "cuda_plan": "auto"}, rounds=6)
    block = _train("dense", 1_200_000, {**params, "cuda_plan": "auto,warp_find:off"}, rounds=6)
    assert warp == block


_PROBE = """
import sys
sys.path.insert(0, {here!r})
import numpy as np
from test_warp_find import _data, _train
X, y = _data({kind!r}, 20_000, 0)
if {categorical!r}:
    X = np.column_stack([X, np.random.default_rng(1).integers(0, 6, size=len(y))]).astype(np.float32)
    _train(None, 0, {{"quant_mode": "stochastic", "verbosity": 1, "categorical_feature": [X.shape[1] - 1]}}, X=X, y=y, rounds=2)
else:
    _train(None, 0, {{"quant_mode": "stochastic", "verbosity": 1}}, X=X, y=y, rounds=2)
"""


def _finder_kernels(kind, categorical):
    code = _PROBE.format(here=os.path.dirname(os.path.abspath(__file__)), kind=kind, categorical=categorical)
    env = {**os.environ, "FALCATA_DEBUG": "diag"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    return re.findall(r"split finder: (\S+) kernel for quantized levels", out)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("kind", "categorical", "kernel"),
    [("dense", False, "warp"), ("fewbin", False, "warp"), ("nan", False, "warp"), ("dense", True, "block")],
)
def test_datasets_reach_the_intended_finder_cuda(kind, categorical, kernel):
    """The bit-identity sweep only means something if its datasets take the kernel they are meant to exercise."""
    kernels = _finder_kernels(kind, categorical)
    assert kernels, "no split-finder kernel choice was logged"
    assert all(k == kernel for k in kernels), kernels
