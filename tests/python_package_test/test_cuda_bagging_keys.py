# coding: utf-8
"""Bagged CUDA training must train the same model with or without the bagging fast paths.

cuda_plan keys bagged_code_view (a bagged tree's 4-bit fill still writes the pair-joint code view), oob_node_table
(the out-of-bag score pass reads one packed node table) and bagging_fused_scan (the device bag in three kernels without
a flag array) change only how the same values are computed, so the trained model must be byte-identical with each
key off. The sweep covers 4-bit columns with and without NaN, 8-bit columns, a column count that leaves an odd column
per byte slot, feature_fraction (the per-tree views), the regression, binary and multiclass objectives, small and
large bags, and a row count that is not a multiple of the scan block. These small datasets do not take the code view
(it engages with the column-major 4-bit store of wide data), so bagged_code_view is a no-op in that sweep;
test_bagged_code_view_engages_cuda runs wide few-value data where the diag log shows it engage, and
test_bagging_fused_scan_multi_pass_carry_cuda a bag of more than 1024 scan blocks.
"""

import hashlib
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

KEYS = ("bagged_code_view", "oob_node_table", "bagging_fused_scan")


def _model_md5(params, X, y, rounds):
    ds = flc.Dataset(X, label=y, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _data(kind, rows, cols, seed):
    rng = np.random.default_rng(seed)
    if kind == "fewbin":
        X = rng.integers(0, 5, size=(rows, cols)).astype(np.float32)
    elif kind == "fewbin-nan":
        X = rng.integers(0, 5, size=(rows, cols)).astype(np.float32)
        X[rng.random(X.shape) < 0.08] = np.nan
    else:
        X = rng.standard_normal((rows, cols)).astype(np.float32)
    return X


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "kind, cols, objective, frac, feature_fraction, leaves, depth",
    [
        ("fewbin", 7, "regression", 0.3, 1.0, 31, 5),
        ("fewbin", 120, "regression", 0.6, 0.2, 255, 8),
        ("fewbin-nan", 41, "binary", 0.8, 0.5, 63, 6),
        ("dense", 9, "binary", 0.4, 0.7, 63, 6),
        ("fewbin", 5, "multiclass", 0.5, 1.0, 15, 4),
    ],
)
def test_bagging_keys_are_bit_identical_cuda(kind, cols, objective, frac, feature_fraction, leaves, depth):
    rows = 20011
    X = _data(kind, rows, cols, seed=3)
    rng = np.random.default_rng(4)
    if objective == "multiclass":
        y = np.nan_to_num(X[:, 0]) % 3
    elif objective == "binary":
        y = (np.nan_to_num(X[:, 0]) + rng.normal(size=rows) > np.nanmean(X[:, 0])).astype(float)
    else:
        y = np.nan_to_num(X).sum(axis=1) + rng.normal(size=rows)
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": 5,
        "bagging_fraction": frac,
        "bagging_freq": 1,
        "feature_fraction": feature_fraction,
        "use_quantized_grad": True,
        "verbosity": -1,
        "seed": 7,
    }
    if objective == "multiclass":
        params["num_class"] = 3
    reference = _model_md5({**params, "cuda_plan": "auto"}, X, y, rounds=20)
    for key in KEYS:
        assert _model_md5({**params, "cuda_plan": f"auto,{key}:off"}, X, y, rounds=20) == reference, key
    off = ",".join(f"{key}:off" for key in KEYS)
    assert _model_md5({**params, "cuda_plan": f"auto,{off}"}, X, y, rounds=20) == reference


@_REQUIRES_CUDA
@pytest.mark.parametrize("frac", [0.3, 0.9])
def test_bagging_fused_scan_multi_pass_carry_cuda(frac):
    """bagging_fused_scan's block-base kernel scans the per-block kept counts in passes of 1024 blocks, carrying the
    running total from pass to pass. 3 * 2^20 + 517 rows are 3073 blocks of 1024 rows: four passes, the last one
    partial, so every carry and the partial tail are exercised (the sweep above has 20 blocks, one pass). A wrong
    carry or tail puts rows in the wrong half of the [in-bag | out-of-bag] index array, which changes the bag and so
    the model; the reference is the flag pass + global prefix sum (key off)."""
    rows = 3 * 1024 * 1024 + 517
    rng = np.random.default_rng(9)
    X = rng.standard_normal((rows, 4)).astype(np.float32)
    y = X @ np.array([1.0, -2.0, 0.5, 0.25]) + rng.normal(size=rows)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 63,
        "max_depth": 6,
        "bagging_fraction": frac,
        "bagging_freq": 1,
        "use_quantized_grad": True,
        "verbosity": -1,
        "seed": 11,
    }
    reference = _model_md5({**params, "cuda_plan": "auto,bagging_fused_scan:off"}, X, y, rounds=6)
    assert _model_md5({**params, "cuda_plan": "auto"}, X, y, rounds=6) == reference


_CODE_VIEW_PROBE = """
import hashlib
import numpy as np
import falcata as flc
rng = np.random.default_rng(5)
rows, cols, values = 8000, {cols}, {values}
X = rng.integers(0, values, size=(rows, cols)).astype(np.float64)
y = X[:, :40] @ rng.standard_normal(40) + rng.standard_normal(rows)
params = {{"device_type": "cuda", "objective": "regression", "num_leaves": 63, "max_depth": 6, "max_bin": 15,
          "feature_fraction": {ff}, "bagging_fraction": 0.6, "bagging_freq": {freq}, "use_quantized_grad": True,
          "verbosity": 1, "seed": 42, "cuda_plan": {plan!r}}}
booster = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=8)
print("MD5", hashlib.md5(booster.model_to_string().split("\\nparameters:")[0].encode()).hexdigest())
"""


def _code_view_run(cols, values, ff, freq, plan):
    """(model md5, the compact fill's code-view diag lines) of a bagged run on wide few-value data, FALCATA_DEBUG=diag."""
    code = _CODE_VIEW_PROBE.format(cols=cols, values=values, ff=ff, freq=freq, plan=plan)
    env = {**os.environ, "FALCATA_DEBUG": "diag"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    md5 = re.findall(r"^MD5 (\w+)$", out, flags=re.M)
    lines = re.findall(r"compact fill: (pair_code5 on|pair_code5 off|bagged tree without the code view)", out)
    return md5[0], lines


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("cols", "values", "ff", "freq", "expected"),
    [
        # 5 x 5 = 25-cell joint tables: the 5-bit code view on every bagged tree, a bag drawn every round or reused
        (1200, 5, 0.2, 1, "pair_code5 on"),
        (1200, 5, 0.2, 5, "pair_code5 on"),
        # 6 x 6 = 36-cell tables in a 60-byte row: the 6-bit row is no smaller, the view is declined on every tree
        (600, 6, 0.2, 1, "bagged tree without the code view"),
    ],
)
def test_bagged_code_view_engages_cuda(cols, values, ff, freq, expected):
    """bagged_code_view is only tested if its path runs: the diag log must show every bagged tree's fill taking the
    code view (or, where the columns decline it, the plain tiled fill without the fused histogram), and the key off
    must take neither (no fused fill under bagging) and train the same model."""
    on_md5, on_lines = _code_view_run(cols, values, ff, freq, "auto")
    assert len(on_lines) >= 8, on_lines
    if expected == "bagged tree without the code view":
        assert set(on_lines) == {"pair_code5 off", expected}, on_lines
    else:
        assert set(on_lines) == {expected}, on_lines
    off_md5, off_lines = _code_view_run(cols, values, ff, freq, "auto,bagged_code_view:off")
    assert off_lines == [], off_lines
    assert off_md5 == on_md5
    all_off_md5, _ = _code_view_run(cols, values, ff, freq, "auto," + ",".join(f"{key}:off" for key in KEYS))
    assert all_off_md5 == on_md5
