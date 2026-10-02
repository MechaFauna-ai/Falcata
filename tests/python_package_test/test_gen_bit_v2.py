# coding: utf-8
"""The gen_bit_v2 apply kernels must send every row to exactly the child the original kernel does.

cuda_plan key gen_bit_v2 replaces the host-launched gen-bit-vector step of the level apply with three kernels chosen
by bin source: 128-thread chunks with 8 rows per thread for column-major sources (8-bit and flat 4-bit columns), and 1024-thread chunks for the 4-bit packed compact source, in an interleaved chunk order when a level
has 2..128 splits and the plain order otherwise (single-split roots, more than 128 splits). The trained model must be
byte-identical either way; the cases below reach every kernel and every bin source, plus categorical and missing-value
splits.
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
    "max_depth": 8,
    "min_data_in_leaf": 20,
    "learning_rate": 0.1,
    "verbosity": -1,
    "seed": 13,
}


def _data(kind, rows=60_001, seed=0):
    rng = np.random.default_rng(seed)
    extra = {}
    if kind == "float255":  # 8-bit columns
        X = rng.standard_normal((rows, 30)).astype(np.float32)
        extra = {"max_bin": 255}
    elif kind == "bins1023":  # more than 256 bins: 16-bit columns
        X = rng.standard_normal((rows, 12)).astype(np.float32)
        extra = {"max_bin": 1023}
    elif kind == "int4bit":  # <= 16 bins: flat 4-bit columns, and the packed compact source under feature sampling
        X = rng.integers(0, 7, size=(rows, 120)).astype(np.float32)
        extra = {"max_bin": 15}
    elif kind == "nan":
        X = rng.standard_normal((rows, 20)).astype(np.float32)
        X[:, :10][rng.random((rows, 10)) < 0.2] = np.nan
        extra = {"max_bin": 63}
    elif kind == "categorical":
        X = np.column_stack([rng.standard_normal((rows, 12)), rng.integers(0, 30, size=rows)]).astype(np.float32)
        extra = {"max_bin": 63, "categorical_feature": [12]}
    else:
        raise ValueError(kind)
    w = rng.standard_normal(min(8, X.shape[1]))
    y = np.nan_to_num(X[:, : len(w)]) @ w + rng.standard_normal(rows)
    if kind == "categorical":
        y = y + (X[:, 12] % 5 == 0)
    return X, y.astype(np.float32), extra


def _train(kind, params, rounds=10):
    X, y, extra = _data(kind)
    p = {**BASE, **extra, **params}
    model = flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=rounds).model_to_string()
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.M)


CASES = [
    pytest.param("float255", {}, id="8bit-columns"),
    pytest.param("int4bit", {}, id="flat-4bit-columns"),
    pytest.param("int4bit", {"feature_fraction": 0.3}, id="packed-compact-interleaved"),
    pytest.param(
        "int4bit",
        {"feature_fraction": 0.3, "num_leaves": 511, "max_depth": 10, "min_data_in_leaf": 5},
        id="packed-compact-over-128-splits",
    ),
    pytest.param("nan", {}, id="missing-values"),
    pytest.param("categorical", {}, id="categorical"),
    pytest.param("float255", {"objective": "multiclass", "num_class": 3}, id="multiclass"),
    pytest.param("float255", {"num_leaves": 255, "max_depth": 12, "min_data_in_leaf": 2}, id="tiny-leaves"),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])
@pytest.mark.parametrize(("kind", "config"), CASES)
def test_gen_bit_v2_is_bit_identical_cuda(kind, config, quant):
    params = {"quant_mode": quant, **config}
    base_plan = "auto,graph_loop:off" if params["quant_mode"] == "none" else "auto"
    if params.get("objective") == "multiclass":
        X, y, extra = _data(kind)
        y = np.digitize(y, np.quantile(y, [0.33, 0.66])).astype(np.float32)

        def train(plan):
            p = {**BASE, **extra, **params, "cuda_plan": plan}
            m = flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=6).model_to_string()
            return re.sub(r"^\[cuda_plan: .*\]$", "", m, flags=re.M)

        assert train(base_plan) == train(base_plan + ",gen_bit_v2:off")
    else:
        on = _train(kind, {**params, "cuda_plan": base_plan})
        assert on == _train(kind, {**params, "cuda_plan": base_plan + ",gen_bit_v2:off"})
