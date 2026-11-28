"""Tests for Wave 4 B5 (warp_find_mixed_cat): mixed warp+block dispatch, zero_as_missing
in WarpFindBest, and fast categorical prefix scans + 64-bit group-acceptance bitmask.

Verifies 100% exact model-string (tree_md5) bit-identity between warp_find_mixed_cat:on
and warp_find_mixed_cat:off (and warp_find:off), plus FALCATA_DEBUG=diag activation across
mixed numerical+categorical, high-cardinality (>64 thresholds), multiclass, bagging,
feature_fraction, feature_fraction_bynode, and zero_as_missing workloads.
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
    "seed": 7,
}


def _make_dataset(kind, rows=30_001, seed=42):
    rng = np.random.default_rng(seed)
    extra = {}
    if kind == "1cat_50num":
        num = rng.standard_normal((rows, 50)).astype(np.float32)
        cat = rng.integers(0, 16, size=(rows, 1)).astype(np.float32)
        X = np.column_stack([num, cat])
        extra["categorical_feature"] = [50]
        y = num[:, :5] @ rng.standard_normal(5) + (cat[:, 0] % 4) * 0.6 + rng.standard_normal(rows)
    elif kind == "5cat_20num_mixed_width":
        narrow = rng.integers(1, 6, size=(rows, 10)).astype(np.float32)
        wide = rng.standard_normal((rows, 10)).astype(np.float32)
        cats = np.column_stack([rng.integers(0, c, size=rows) for c in (4, 8, 16, 32, 48)]).astype(np.float32)
        X = np.column_stack([narrow, wide, cats])
        extra["categorical_feature"] = [20, 21, 22, 23, 24]
        y = (
            narrow[:, :4] @ rng.standard_normal(4)
            + wide[:, :4] @ rng.standard_normal(4)
            + (cats[:, 0] == 1) * 1.2
            + (cats[:, 2] % 5) * 0.4
            + rng.standard_normal(rows)
        )
    elif kind == "high_card_cat":
        num = rng.standard_normal((rows, 16)).astype(np.float32)
        cats = np.column_stack([rng.integers(0, c, size=rows) for c in (150, 180)]).astype(np.float32)
        X = np.column_stack([num, cats])
        extra["categorical_feature"] = [16, 17]
        extra["max_cat_threshold"] = 96
        y = num[:, :4] @ rng.standard_normal(4) + (cats[:, 0] % 11) * 0.3 + rng.standard_normal(rows)
    elif kind == "pure_cat":
        cats = np.column_stack([rng.integers(0, c, size=rows) for c in (6, 12, 24, 48, 80)]).astype(np.float32)
        X = cats
        extra["categorical_feature"] = [0, 1, 2, 3, 4]
        y = (cats[:, 1] % 4) * 0.8 + (cats[:, 3] % 7) * 0.3 + rng.standard_normal(rows)
    elif kind == "zam_wide_signed":
        # Signed values with ~45% zeros: default_bin > 0, mfb_offset == 0
        X = rng.standard_normal((rows, 24)).astype(np.float32)
        X[rng.random(X.shape) < 0.45] = 0.0
        extra["zero_as_missing"] = True
        y = X[:, :6] @ rng.standard_normal(6) + rng.standard_normal(rows)
    elif kind == "zam_wide_nonneg":
        # Non-negative values with ~45% zeros: default_bin == 0, mfb_offset == 1
        X = np.abs(rng.standard_normal((rows, 24))).astype(np.float32)
        X[rng.random(X.shape) < 0.45] = 0.0
        extra["zero_as_missing"] = True
        y = X[:, :6] @ rng.standard_normal(6) + rng.standard_normal(rows)
    elif kind == "zam_narrow":
        # 5-bin values with ~40% zeros: hits narrow 4-lane strided tier + find_group_direction
        X = rng.integers(-2, 3, size=(rows, 20)).astype(np.float32)
        X[rng.random(X.shape) < 0.3] = 0.0
        extra["zero_as_missing"] = True
        y = X[:, :6] @ rng.standard_normal(6) + rng.standard_normal(rows)
    elif kind == "zam_plus_cat":
        num = rng.standard_normal((rows, 20)).astype(np.float32)
        num[rng.random(num.shape) < 0.4] = 0.0
        cats = np.column_stack([rng.integers(0, c, size=rows) for c in (12, 32)]).astype(np.float32)
        X = np.column_stack([num, cats])
        extra["categorical_feature"] = [20, 21]
        extra["zero_as_missing"] = True
        y = num[:, :5] @ rng.standard_normal(5) + (cats[:, 0] % 3) * 0.7 + rng.standard_normal(rows)
    else:
        raise ValueError(kind)
    return X, y.astype(np.float32), extra


def _train_model_str(X, y, params, rounds=10):
    p = {**BASE, **params}
    cat = p.pop("categorical_feature", "auto")
    ds = flc.Dataset(X, label=y, categorical_feature=cat, params=p)
    model = flc.train(p, ds, num_boost_round=rounds).model_to_string()
    return re.sub(r"^\[cuda_plan: .*\]$", "", model, flags=re.MULTILINE)


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])
@pytest.mark.parametrize(
    "kind",
    [
        "1cat_50num",
        "5cat_20num_mixed_width",
        "high_card_cat",
        "pure_cat",
        "zam_wide_signed",
        "zam_wide_nonneg",
        "zam_narrow",
        "zam_plus_cat",
    ],
)
def test_mixed_cat_and_zam_bit_identical_cuda(kind, quant):
    X, y, extra = _make_dataset(kind)
    on_model = _train_model_str(X, y, {**extra, "quant_mode": quant, "cuda_plan": "auto"})
    off_model = _train_model_str(
        X,
        y,
        {**extra, "quant_mode": quant, "cuda_plan": "auto,warp_find_mixed_cat:off"},
    )
    assert on_model == off_model


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("kind", "extra_params"),
    [
        pytest.param(
            "1cat_50num",
            {"objective": "binary"},
            id="1cat_50num-binary",
        ),
        pytest.param(
            "5cat_20num_mixed_width",
            {"objective": "multiclass", "num_class": 5},
            id="5cat_20num-multiclass",
        ),
        pytest.param(
            "5cat_20num_mixed_width",
            {"bagging_fraction": 0.7, "bagging_freq": 1, "feature_fraction": 0.5},
            id="5cat_20num-bagging-ff",
        ),
        pytest.param(
            "5cat_20num_mixed_width",
            {"feature_fraction": 0.12},
            id="5cat_20num-sparse-ff-all-branches",
        ),
        pytest.param(
            "5cat_20num_mixed_width",
            {"feature_fraction_bynode": 0.5},
            id="5cat_20num-bynode",
        ),
        pytest.param(
            "zam_wide_signed",
            {"objective": "binary", "bagging_fraction": 0.7, "bagging_freq": 1},
            id="zam-binary-bagging",
        ),
        pytest.param(
            "zam_wide_signed",
            {"max_delta_step": 0.5},
            id="zam-max-delta-step",
        ),
        pytest.param(
            "zam_plus_cat",
            {
                "feature_fraction_bynode": 0.6,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
            },
            id="zam-cat-bynode-bagging",
        ),
    ],
)
def test_mixed_cat_and_zam_configs_bit_identical_cuda(kind, extra_params):
    X, y, extra = _make_dataset(kind, rows=25_001, seed=13)
    if extra_params.get("objective") == "binary":
        y = (y > np.median(y)).astype(np.float32)
    elif extra_params.get("objective") == "multiclass":
        k = extra_params["num_class"]
        y = np.digitize(y, np.quantile(y, np.linspace(0, 1, k + 1)[1:-1])).astype(np.float32)
    params = {**extra, "quant_mode": "stochastic", **extra_params}
    on_model = _train_model_str(X, y, {**params, "cuda_plan": "auto"}, rounds=10)
    off_model = _train_model_str(X, y, {**params, "cuda_plan": "auto,warp_find_mixed_cat:off"}, rounds=10)
    assert on_model == off_model


_DIAG_PROBE = """
import sys
sys.path.insert(0, {here!r})
from test_cuda_warp_find_mixed_cat import _make_dataset, _train_model_str
X, y, extra = _make_dataset({kind!r}, rows=15_000, seed=3)
params = {{**extra, "quant_mode": "stochastic", "verbosity": 1, **{params!r}}}
_train_model_str(X, y, params, rounds={rounds!r})
"""


def _finder_diag_lines(kind, params=None, rounds=3):
    code = _DIAG_PROBE.format(
        here=os.path.dirname(os.path.abspath(__file__)),
        kind=kind,
        params=params or {},
        rounds=rounds,
    )
    env = {**os.environ, "FALCATA_DEBUG": "diag"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    found = re.findall(r"split finder: (\S+) kernel for this tree's quantized levels(.*)", out)
    return [f"{kernel}{tier}" for kernel, tier in found]


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("kind", "params", "expected"),
    [
        pytest.param("1cat_50num", {}, "warp+block (tier: wide)", id="1cat_50num-on"),
        pytest.param(
            "1cat_50num",
            {"cuda_plan": "auto,warp_find_mixed_cat:off"},
            "block",
            id="1cat_50num-off",
        ),
        pytest.param(
            "5cat_20num_mixed_width",
            {},
            "warp+block (tier: few-position + wide)",
            id="5cat_20num-mixed-tier",
        ),
        pytest.param("pure_cat", {}, "block", id="pure-cat"),
        pytest.param("zam_wide_signed", {}, "warp (tier: wide)", id="zam-wide-on"),
        pytest.param(
            "zam_wide_signed",
            {"cuda_plan": "auto,warp_find_mixed_cat:off"},
            "block",
            id="zam-wide-off",
        ),
        pytest.param("zam_narrow", {}, "warp (tier: few-position)", id="zam-narrow-on"),
        pytest.param("zam_plus_cat", {}, "warp+block (tier: wide)", id="zam-plus-cat-on"),
    ],
)
def test_mixed_cat_and_zam_diag_activation_cuda(kind, params, expected):
    lines = _finder_diag_lines(kind, params)
    assert lines, "no split-finder launch was logged"
    assert all(line == expected for line in lines), lines
