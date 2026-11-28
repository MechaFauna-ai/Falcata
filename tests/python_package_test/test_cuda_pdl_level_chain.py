# coding: utf-8
"""CUDA training must train the same model with or without cuda_plan key pdl_level_chain.

The key launches the level chain's consumer kernels (level sync, offset aggregate, split-inner partition, quantized
fix + subtract) as programmatic dependents of the kernel before them on the same stream; each consumer waits for
its producer's completion and memory before touching memory, so the model is byte-identical with the key off. The
sweep covers quantized training (stochastic, fixed point), feature sampling (the used-feature fix
+ subtract and the used-task sync), bagging, missing values, a leaf budget that binds before max_depth, and the
multiclass objective (several trees per round).
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


def _model_md5(params, X, y, rounds):
    ds = flc.Dataset(X, label=y, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "leaves", "depth", "quant", "feature_fraction", "bagging"),
    [
        ("regression", 1023, 10, "stochastic", 1.0, 1.0),
        ("binary", 255, 10, "fixedpoint", 0.5, 1.0),
        ("binary", 511, 9, "stochastic", 1.0, 0.6),
        ("multiclass", 127, 7, "fixedpoint", 1.0, 1.0),
        ("regression", 255, 8, "stochastic", 0.7, 0.5),
    ],
)
def test_pdl_level_chain_is_bit_identical_cuda(objective, leaves, depth, quant, feature_fraction, bagging):
    rows = 30011
    rng = np.random.default_rng(13)
    X = rng.standard_normal((rows, 24)).astype(np.float32)
    X[:, 20:] = rng.integers(0, 5, size=(rows, 4))
    X[rng.random(X.shape) < 0.03] = np.nan
    z = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) + 0.3 * rng.normal(size=rows)
    if objective == "multiclass":
        y = np.digitize(z, [-1.0, 0.0, 1.0]).astype(np.float32)
    elif objective == "binary":
        y = (z > 0).astype(np.float32)
    else:
        y = z.astype(np.float32)
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": leaves,
        "max_depth": depth,
        "min_data_in_leaf": 10,
        "learning_rate": 0.1,
        "feature_fraction": feature_fraction,
        "verbose": -1,
        "quant_mode": quant,
        "seed": 5,
    }
    if objective == "multiclass":
        params["num_class"] = 4
    if bagging < 1.0:
        params["bagging_fraction"] = bagging
        params["bagging_freq"] = 1
    on = _model_md5({**params, "cuda_plan": "auto"}, X, y, 12)
    off = _model_md5({**params, "cuda_plan": "auto,pdl_level_chain:off"}, X, y, 12)
    assert on == off


@_REQUIRES_CUDA
def test_pdl_graph_chain_and_map_clear_cuda():
    """Verify pdl_graph_chain captures programmatic edges inside device_level_graph and both keys are bit-identical."""
    probe = """
import hashlib, json, numpy as np, falcata as flc
rng = np.random.default_rng(19)
X = rng.standard_normal((20000, 16)).astype(np.float32)
y = (X[:, 0] + 0.7 * X[:, 1] * X[:, 2] > 0).astype(np.float32)
base = {
    "device_type": "cuda", "objective": "binary", "num_leaves": 127, "max_depth": 7,
    "quant_mode": "stochastic", "seed": 42, "verbose": -1,
}
plan = __import__("sys").argv[1]
p = {**base, "cuda_plan": plan}
ds = flc.Dataset(X, label=y, params=p)
bst = flc.train(p, ds, num_boost_round=6)
lines = [line for line in bst.model_to_string().splitlines() if "cuda_plan" not in line]
print(json.dumps({"md5": hashlib.md5("\\n".join(lines).encode()).hexdigest()}))
"""
    results = {}
    for plan in [
        "auto,pdl_level_chain:off,device_tree_map_clear:on",
        "auto,pdl_graph_chain:off,device_tree_map_clear:on",
        "auto,pdl_graph_chain:on,device_tree_map_clear:on",
        "auto,pdl_graph_chain:on,device_tree_map_clear:off",
        "auto,pdl_graph_chain:on,device_tree_map_clear:auto",
    ]:
        proc = subprocess.run(
            [sys.executable, "-c", probe, plan],
            env=dict(os.environ, FALCATA_DEBUG="diag"),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + "\nSTDERR:\n" + proc.stderr
        md5 = proc.stdout.strip().splitlines()[-1]
        edges = re.findall(
            r"device_level_graph_edges: nodes=(\d+) edges=(\d+) pdl=(\d+) default=(\d+) k2k_default=(\d+)",
            proc.stderr,
        )
        assert edges, f"expected captured device_level_graph_edges for {plan}\n{proc.stderr}"
        clears = re.findall(r"device_tree_warm_clear_graph:", proc.stderr)
        results[plan] = (md5, tuple(map(int, edges[0])), len(clears))

    md5s = {v[0] for v in results.values()}
    assert len(md5s) == 1, f"model md5 differed across plans: {results}"
    _, e_pdl_off, c_on = results["auto,pdl_level_chain:off,device_tree_map_clear:on"]
    _, e_graph_off, _ = results["auto,pdl_graph_chain:off,device_tree_map_clear:on"]
    _, e_graph_on_clr, c_clr_on = results["auto,pdl_graph_chain:on,device_tree_map_clear:on"]
    _, e_graph_on_noclr, c_clr_off = results["auto,pdl_graph_chain:on,device_tree_map_clear:off"]
    assert e_pdl_off[2] == 0
    assert c_on >= 1 and c_clr_on >= 1 and c_clr_off == 0
    if e_graph_off[2] > 0:
        # On SM >= 9.0 with PDL active, pdl_graph_chain must capture strictly more programmatic edges,
        # and skipping the warm-clear memset drops one node and turns the final aggregate->apply edge programmatic.
        assert e_graph_on_clr[2] > e_graph_off[2], (e_graph_off, e_graph_on_clr)
        assert e_graph_on_clr[4] < e_graph_off[4], (e_graph_off, e_graph_on_clr)
        assert e_graph_on_noclr[0] == e_graph_on_clr[0] - 1, (e_graph_on_clr, e_graph_on_noclr)
        assert e_graph_on_noclr[2] == e_graph_on_clr[2] + 1, (e_graph_on_clr, e_graph_on_noclr)
