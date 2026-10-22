# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key device_levels.

The key runs the first levels of the quantized two-sync level prefix (those where the leaf budget cannot bind and the
children stay below max_depth) device-driven: the level sync's select kernel and the apply's pairs kernel replay the
host loop's decisions (the splittable leaves, the apply descriptors, the tree record, the children's search
descriptors and the construct's block map), the launches are sized for the level's worst case and read the live
counts from the device, and the host reads the levels back once. The sweep covers both quantized modes, regression,
binary and multiclass, leaf budgets that are and are not powers of two (the host loop takes over where the budget can
bind), max_depth below and above log2(num_leaves), min_data_in_leaf large enough to stop a tree inside the device
levels, min_gain_to_split stumps, bagging, GOSS, feature_fraction (the compact column view), few-bin and wide data,
and missing values; the configurations the key leaves to the host loop (per-node feature masks, categorical
features, interaction constraints) must still train the same model.

Where its depth bound holds, the complete device tree (cuda_plan key device_tree, default on) replaces the prefix, so
the prefix arms below set device_tree:off; _same also trains the default plan (the complete tree where the bound holds,
else the prefix), which must match too.
"""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)


def _model_md5(params, X, y, rounds, categorical=None):
    ds = flc.Dataset(X, label=y, params=params, categorical_feature=categorical or "auto")
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def _data(rows, cols, objective, seed, few_bin_cols=4, nan_frac=0.05):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    X[:, cols - few_bin_cols :] = rng.integers(0, 6, size=(rows, few_bin_cols))
    if nan_frac > 0:
        X[rng.random(X.shape) < nan_frac] = np.nan
    z = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) + 0.3 * rng.normal(size=rows)
    if objective == "multiclass":
        y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    elif objective == "binary":
        y = (z > 0).astype(np.float32)
    else:
        y = z.astype(np.float32)
    return X, y


def _params(objective, leaves, depth, quant, seed=3, **extra):
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": 10,
        "learning_rate": 0.1,
        "verbose": -1,
        "quant_mode": quant,
        "seed": seed,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    params.update(extra)
    return params


def _same(params, X, y, rounds, categorical=None, base_plan="auto"):
    off = _model_md5({**params, "cuda_plan": base_plan + ",device_levels:off"}, X, y, rounds, categorical)
    on = _model_md5(
        {**params, "cuda_plan": base_plan + ",device_levels:on,device_tree:off,device_level_graph:off"},
        X,
        y,
        rounds,
        categorical,
    )
    graph = _model_md5(
        {**params, "cuda_plan": base_plan + ",device_levels:on,device_tree:off,device_level_graph:on"},
        X,
        y,
        rounds,
        categorical,
    )
    default = _model_md5({**params, "cuda_plan": base_plan}, X, y, rounds, categorical)
    assert on == off
    assert graph == off
    assert default == off


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "leaves", "depth", "quant", "extra"),
    [
        ("regression", 255, 8, "stochastic", {}),
        ("binary", 511, 9, "fixedpoint", {}),
        ("multiclass", 127, 7, "fixedpoint", {}),
        ("regression", 1023, 10, "stochastic", {"bagging_fraction": 0.5, "bagging_freq": 1}),
        ("regression", 1024, 10, "fixedpoint", {}),
        ("regression", 100, 8, "stochastic", {}),
        ("binary", 300, 9, "stochastic", {"bagging_fraction": 0.7, "bagging_freq": 1}),
        ("regression", 255, 4, "stochastic", {}),
        ("regression", 255, -1, "fixedpoint", {}),
        ("binary", 255, 8, "stochastic", {"min_data_in_leaf": 3000}),
        ("regression", 255, 8, "stochastic", {"min_gain_to_split": 30.0}),
        ("regression", 255, 8, "fixedpoint", {"feature_fraction": 0.5}),
        ("binary", 255, 8, "stochastic", {"feature_fraction": 0.3, "bagging_fraction": 0.5, "bagging_freq": 1}),
        ("regression", 511, 9, "stochastic", {"min_sum_hessian_in_leaf": 5.0}),
        ("regression", 255, 8, "stochastic", {"quant_bins": 16}),
        ("regression", 255, 8, "stochastic", {"max_bin": 15}),
    ],
)
def test_device_levels_is_bit_identical_cuda(objective, leaves, depth, quant, extra):
    X, y = _data(30011, 20, objective, 29)
    _same(_params(objective, leaves, depth, quant, **extra), X, y, 12)


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])
def test_device_levels_wide_and_goss_cuda(quant):
    # more columns than one construct partition holds, and GOSS's bagged subset
    X, y = _data(20011, 700, "binary", 31, few_bin_cols=300, nan_frac=0.0)
    # Cross GOSS's ten-round unbagged warmup: the hessian ridge captured by the finder changes afterwards.
    _same(_params("binary", 255, 8, quant, data_sample_strategy="goss", top_rate=0.2, other_rate=0.1), X, y, 18)


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant", ["stochastic", "fixedpoint"])
def test_device_levels_few_values_sampled_cuda(quant):
    # Numerai-like: many five-value columns, a tenth of them sampled per tree (the compact view, the pair-joint
    # constructs, per-tree launch layouts that a cached level graph must key on)
    rng = np.random.default_rng(41)
    rows, cols = 30011, 400
    X = rng.integers(0, 5, size=(rows, cols)).astype(np.float32)
    y = (X[:, :8].sum(axis=1) + rng.normal(size=rows)).astype(np.float32)
    _same(_params("regression", 1023, 10, quant, feature_fraction=0.1, min_data_in_leaf=20), X, y, 20)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "extra",
    [
        {"feature_fraction_bynode": 0.5},
        {"interaction_constraints": [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9]]},
        {"categorical": True},
    ],
)
def test_device_levels_host_fallbacks_cuda(extra):
    # flows the key leaves to the host loop: the same model either way
    extra = dict(extra)
    categorical = None
    X, y = _data(20011, 12, "regression", 37)
    if extra.pop("categorical", False):
        X[:, 0] = np.random.default_rng(5).integers(0, 30, size=X.shape[0])
        categorical = [0]
    _same(_params("regression", 255, 8, "stochastic", **extra), X, y, 8, categorical)


@_REQUIRES_CUDA
def test_device_levels_engage_cuda():
    # the key runs where its gate holds: the hybrid diag line names the device levels (a child process: the line is
    # printed once per process)
    code = (
        "import numpy as np, falcata as flc\n"
        "rng = np.random.default_rng(1)\n"
        "X = rng.standard_normal((20000, 10)).astype(np.float32)\n"
        "y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)\n"
        "p = {'device_type': 'cuda', 'objective': 'binary', 'num_leaves': 255, 'max_depth': 8, 'verbose': -1,\n"
        "     'quant_mode': 'stochastic', 'cuda_plan': 'auto,device_levels:on,device_tree:off'}\n"
        "flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=3)\n"
    )
    env = dict(os.environ, FALCATA_DEBUG="diag")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert "device_levels: 7 device-driven levels per tree" in out.stderr


@_REQUIRES_CUDA
@pytest.mark.parametrize("max_bin", [15, 255])
@pytest.mark.parametrize("graph", ["off", "on"])
def test_device_levels_first_tree_stump_cuda(max_bin, graph):
    # Fresh learner, including both dense and pair-joint constructs: no level ever publishes a nonempty block map.
    # This also runs under compute-sanitizer initcheck in the source-specific recovery gate.
    X, y = _data(12000, 24, "regression", 73)
    p = _params("regression", 255, 8, "fixedpoint", min_gain_to_split=1e12, max_bin=max_bin)
    models = []
    for plan in ["auto,device_levels:off", "auto,device_tree:off,device_level_graph:" + graph, "auto"]:
        b = flc.train({**p, "cuda_plan": plan}, flc.Dataset(X, label=y, params=p), num_boost_round=3)
        assert b.num_trees() == 1
        assert b.dump_model()["tree_info"][0]["num_leaves"] == 1
        models.append("\n".join(line for line in b.model_to_string().splitlines() if "cuda_plan" not in line))
    assert models[0] == models[1] == models[2]


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "plan",
    [
        "pdl_level_chain:off",
        "sync_used_tasks:off",
        "sync_copy_batched:off",
        "warp_find:off",
        "fix_subtract_fused:off,fix_subtract_masked:off",
        "apply_row_batch:off,aggregate_load_first:off",
        "apply_struct_fused:off",
        "gap_copy_fused:off",
        "batch_kernels:off",
        "batch_apply:off",
        "readback_fused:off",
        "final_map_only:off,final_readback_first:off",
    ],
)
def test_device_levels_dependencies_cuda(plan):
    X, y = _data(30011, 24, "regression", 51)
    _same(_params("regression", 255, 8, "fixedpoint"), X, y, 8, base_plan="auto," + plan)


@_REQUIRES_CUDA
@pytest.mark.parametrize(("leaves", "depth"), [(3, 2), (7, 3), (8, 4), (16, 5), (63, 6), (100, 7)])
def test_device_levels_invalid_leaves_and_ties_cuda(leaves, depth):
    rng = np.random.default_rng(53)
    # Duplicate columns have exactly tied gains; repeated rows give equal child counts.
    x = rng.integers(0, 4, size=(12000, 1)).astype(np.float32)
    X = np.repeat(x, 12, axis=1)
    y = ((x[:, 0] == 1) | (x[:, 0] == 2)).astype(np.float32)
    _same(_params("binary", leaves, depth, "fixedpoint", min_data_in_leaf=2), X, y, 10)


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
@pytest.mark.parametrize("reset", ["budget", "data"])
def test_device_levels_reset_cuda(graph, reset):
    X, y = _data(30011, 16, "regression", 64)

    def train(plan):
        p = _params("regression", 63, 6, "fixedpoint", min_data_in_leaf=3, cuda_plan=plan)
        ds = flc.Dataset(X, label=y, params=p, free_raw_data=False)
        b = flc.Booster(p, train_set=ds)
        for _ in range(4):
            b.update()
        if reset == "budget":
            b.reset_parameter({"num_leaves": 511, "max_depth": 9})
            for _ in range(8):
                b.update()
        else:
            X2, y2 = _data(50013, 16, "regression", 65)
            ds2 = flc.Dataset(X2, label=y2, reference=ds, free_raw_data=False)
            b.update(train_set=ds2)
            for _ in range(7):
                b.update()
        return hashlib.md5(
            "\n".join(line for line in b.model_to_string().splitlines() if "cuda_plan" not in line).encode()
        ).hexdigest()

    off = train("auto,device_levels:off")
    assert train("auto,device_levels:on,device_tree:off,device_level_graph:" + graph) == off
    assert train("auto,device_level_graph:" + graph) == off


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
def test_device_levels_sampledwide_uncapped_cuda(graph):
    # Source-specific lattice repro: the uncapped code-byte gather and changing sampled-column layouts.
    rng = np.random.default_rng(int(hashlib.md5(b"sampledwide5").hexdigest()[:8], 16))
    X = rng.integers(0, 5, size=(8000, 1200)).astype(np.float64)
    y = X[:, :40] @ rng.standard_normal(40) + rng.standard_normal(len(X))
    p = _params("regression", 63, 6, "stochastic", seed=42, num_threads=8, max_bin=15, feature_fraction=0.15)
    plan = "auto,pair_code5_words:off,pair_capped_rows:off"
    off = _model_md5({**p, "cuda_plan": plan + ",device_levels:off"}, X, y, 25)
    for _ in range(3):
        on = _model_md5({**p, "cuda_plan": plan + ",device_tree:off,device_level_graph:" + graph}, X, y, 25)
        assert on == off
        assert _model_md5({**p, "cuda_plan": plan + ",device_level_graph:" + graph}, X, y, 25) == off


@_REQUIRES_CUDA
def test_device_levels_forced_splits_cuda(tmp_path):
    X, y = _data(20011, 16, "regression", 41)
    forced = tmp_path / "forced.json"
    forced.write_text(json.dumps({"feature": 0, "threshold": 0.0, "left": {"feature": 1, "threshold": 0.5}}))
    _same(_params("regression", 255, 8, "stochastic", forcedsplits_filename=str(forced)), X, y, 6)


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
def test_device_levels_nonquant_fallback_cuda(graph):
    X, y = _data(12000, 16, "binary", 67)
    p = _params("binary", 63, 6, "none", num_threads=1)
    plan = "auto,graph_loop:off"
    off = _model_md5({**p, "cuda_plan": plan + ",device_levels:off"}, X, y, 6)
    assert _model_md5({**p, "cuda_plan": plan + ",device_tree:off,device_level_graph:" + graph}, X, y, 6) == off
    assert _model_md5({**p, "cuda_plan": plan + ",device_level_graph:" + graph}, X, y, 6) == off


@_REQUIRES_CUDA
def test_device_levels_rf_multiclass_cuda():
    # a multiclass forest's class trees train one after another on the booster's learner (the forest slots are gone):
    # the host flow, the device_levels prefix (direct and graph) and the complete device tree grow the same trees
    X, y = _data(30011, 16, "multiclass", 69)
    _same(
        _params("multiclass", 63, 6, "fixedpoint", boosting="rf", bagging_fraction=0.632, bagging_freq=1),
        X,
        y,
        12,
    )


@_REQUIRES_CUDA
def test_device_levels_single_stream_fallback_cuda():
    code = (
        "import numpy as np, falcata as flc, hashlib\n"
        "rng = np.random.default_rng(71)\n"
        "X = rng.standard_normal((12000, 12)).astype(np.float32)\n"
        "y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)\n"
        "p = dict(device_type='cuda', objective='binary', num_leaves=63, max_depth=6, verbose=-1,\n"
        "         quant_mode='fixedpoint', seed=7)\n"
        "out = []\n"
        "for plan in ['auto,device_levels:off', 'auto,device_tree:off,device_level_graph:off',\n"
        "             'auto,device_tree:off,device_level_graph:on', 'auto']:\n"
        " p['cuda_plan'] = plan\n"
        " b = flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=5)\n"
        " text = '\\n'.join(s for s in b.model_to_string().splitlines() if 'cuda_plan' not in s)\n"
        " out.append(hashlib.md5(text.encode()).hexdigest())\n"
        "assert len(set(out)) == 1, out\n"
    )
    env = dict(os.environ, FALCATA_SINGLE_STREAM="1", FALCATA_DEBUG="diag")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert "device_levels: host fallback (legacy single stream)" in out.stderr


@_REQUIRES_CUDA
def test_device_levels_shallow_large_budget_cuda():
    # The new finder and bit-change scratch allocations must depend on the shallow prefix, not this budget.
    X, y = _data(12000, 64, "binary", 73)
    _same(_params("binary", 32767, 3, "fixedpoint"), X, y, 4)


@_REQUIRES_CUDA
def test_device_levels_deep_host_continuation_cuda():
    # A complete prefix reaches its 512-pair cap, then host levels grow the finder/descriptor allocations.
    # The next tree must recapture every pointer that moved rather than replaying the old allocation.
    rows = np.arange(32768, dtype=np.uint32)
    X = ((rows[:, None] >> np.arange(13, dtype=np.uint32)) & 1).astype(np.float32)
    y = (X @ np.arange(1, 14, dtype=np.float32)).astype(np.float32)
    _same(_params("regression", 4095, 12, "fixedpoint", min_data_in_leaf=2, quant_bins=16), X, y, 4)


@_REQUIRES_CUDA
def test_device_levels_sampled_codecs_cuda():
    rng = np.random.default_rng(74)
    # Per-tree samples can change the compact decoder even when the allocation and padded row size are reused.
    # Ten sampled columns fit in four bytes under either radix5 or radix6; selecting the last column changes codec.
    X = np.column_stack([rng.integers(0, bins, size=20011) for bins in [5] * 99 + [6]]).astype(np.float32)
    y = X[:, :8].sum(axis=1) + rng.standard_normal(len(X))
    _same(
        _params("regression", 255, 8, "fixedpoint", feature_fraction=0.1, max_bin=15),
        X,
        y,
        40,
        base_plan="auto,pack_radix5:on,pack_radix6:on",
    )
