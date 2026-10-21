# coding: utf-8
"""The per-tree feature sample must select bin-matrix columns by the features they hold.

The CUDA construct kernels and the compact view skip unsampled columns. With EFB a column bundles several features, so
a column is built when ANY of its features is in the tree's sample; reading the feature-indexed sample by column index
built the wrong columns, and the finder then searched sampled features on zero histograms. Every flow (classic per-leaf
loop, host level flow, device levels, complete device tree) must grow the trees the CPU learner grows from the same
sample.
"""

import hashlib
import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")

_PLANS = {
    "classic": "auto,hybrid:off",
    "host": "auto,device_levels:off",
    "device_levels": "auto,device_levels:on,device_tree:off",
    "device_tree": "auto,device_levels:on,device_tree:on",
}


def _data():
    # six dense 16-bin columns plus 30 mutually exclusive one-hot features, which EFB bundles into one column; the
    # signal sits in one-hot features, whose feature index exceeds the column count
    rng = np.random.default_rng(0)
    rows = 20000
    dense = np.floor(rng.random((rows, 6)) * 16).astype(np.float32)
    cat = rng.integers(0, 30, size=rows)
    onehot = np.zeros((rows, 30), dtype=np.float32)
    onehot[np.arange(rows), cat] = 1.0
    X = np.hstack([dense, onehot])
    y = 4.0 * (cat == 17) + 2.0 * (cat == 23) + 0.3 * dense[:, 0] + rng.standard_normal(rows)
    return X, y.astype(np.float32)


def _train(device, plan=None, **extra):
    X, y = _data()
    params = {
        "objective": "regression",
        "device_type": device,
        "num_leaves": 15,
        "max_depth": 4,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.5,
        "seed": 0,
        "num_threads": 8,
        "verbose": -1,
        **extra,
    }
    if plan is not None:
        params["cuda_plan"] = plan
    booster = flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=20)
    roots = [t["tree_structure"].get("split_feature", -1) for t in booster.dump_model()["tree_info"]]
    model = "\n".join(line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line)
    return roots, hashlib.md5(model.encode()).hexdigest(), booster.predict(X)


@_REQUIRES_CUDA
def test_feature_sample_selects_bundled_columns_cuda():
    X, _ = _data()
    assert flc.Dataset(X, params={"verbose": -1}).construct().num_feature() == 36
    cpu_roots, _, cpu_pred = _train("cpu")
    # the CPU learner reaches the bundled one-hot signal (features 23 and 29) in the trees that sample it
    assert {23, 29} <= set(cpu_roots)
    runs = {name: _train("cuda", plan) for name, plan in _PLANS.items()}
    for name, (roots, _, pred) in runs.items():
        assert roots == cpu_roots, f"{name}: root split features differ from the CPU learner's"
        np.testing.assert_allclose(pred, cpu_pred, rtol=1e-4, atol=1e-4, err_msg=name)
    assert runs["host"][1] == runs["device_levels"][1] == runs["device_tree"][1]
    np.testing.assert_array_equal(runs["classic"][2], runs["host"][2])


@_REQUIRES_CUDA
def test_level_flows_match_classic_quantized_cuda():
    runs = {
        name: _train("cuda", plan, use_quantized_grad=True, quant_mode="stochastic") for name, plan in _PLANS.items()
    }
    assert {23, 29} <= set(runs["classic"][0])
    assert runs["host"][1] == runs["device_levels"][1] == runs["device_tree"][1]
    # the classic loop numbers its nodes leaf-wise, the level flows level-wise: same trees, same predictions
    for name in ("host", "device_levels", "device_tree"):
        assert runs[name][0] == runs["classic"][0], name
        np.testing.assert_array_equal(runs[name][2], runs["classic"][2], err_msg=name)
