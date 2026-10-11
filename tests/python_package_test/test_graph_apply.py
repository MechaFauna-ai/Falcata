"""Captured apply ports retain ordered partitions and honor plan overrides.

Fresh processes isolate the process-global plan. Deterministic fp64 and
quantized graph runs compare tree bytes and prediction bits; diagnostics prove
the captured graph ran and distinguish fused node counts. Reset and forced
split cases guard the buffer lifecycle and the fallback paths.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA build")
_KEYS = ("graph_apply_rows", "graph_apply_fused", "graph_skip_unsplittable")
_OFF = ",".join(f"{key}:off" for key in _KEYS)
_NODES = re.compile(r"\d+ bodies x (\d+) nodes, \d+ cached")
_GRAPH = "instantiated unrolled device level loop"


def _data(shape, seed=19):
    rng = np.random.default_rng(seed)
    rows, features = (8193, 600) if shape == "compact" else (8193, 20)
    X = rng.standard_normal((rows, features)).astype(np.float32)
    y = X[:, :8] @ rng.standard_normal(8) + 0.1 * rng.standard_normal(rows)
    extra = {}
    if shape == "compact":
        X = rng.integers(0, 5, size=X.shape).astype(np.int8)
        y = X[:, :8].astype(np.float64) @ rng.standard_normal(8) + rng.standard_normal(rows)
        extra = {"max_bin": 15, "feature_fraction": 0.2}
    elif shape == "missing":
        X[rng.random(X.shape) < 0.15] = np.nan
    elif shape == "gaps":
        rich = np.arange(rows) >= 6000
        X[:, 0] = rich
        y = np.where(rich, 4.0 + X[:, 1] + 0.6 * X[:, 2] ** 2, 0.0)
    elif shape == "boundary":
        row = np.arange(6400)
        X = np.column_stack((row % 2, (row // 2) % 2, (row // 4) % 2, rng.standard_normal((6400, 17))))
        X = X.astype(np.float32)
        y = 4.0 * X[:, 0] + X[:, 1] + 0.125 * X[:, 2]
        extra = {"max_bin": 15}
    return X, y.astype(np.float32), extra


def _leaf_depths(node, depth=0):
    if "leaf_index" in node or "split_index" not in node:
        return [depth]
    return _leaf_depths(node["left_child"], depth + 1) + _leaf_depths(node["right_child"], depth + 1)


def _child(spec):
    mode = spec["mode"]
    plan = "auto,graph_det:on,tuner:off" if mode == "fp64det" else "auto,graph_quant:on,tuner:off"
    if spec.get("overrides"):
        plan += "," + spec["overrides"]
    X, y, extra = _data(spec["shape"])
    params = {
        "objective": "regression",
        "device_type": "cuda",
        "quant_mode": "none" if mode == "fp64det" else "stochastic",
        "cuda_precision": "fp64",
        "num_leaves": 31,
        "max_depth": 5,
        "min_data_in_leaf": 5,
        "feature_pre_filter": False,
        "num_threads": 4,
        "seed": 42,
        "verbosity": 1,
        "cuda_plan": plan,
        **extra,
        **spec.get("params", {}),
    }
    train = flc.Dataset(X, label=y, params=params)
    bst = flc.Booster(params, train_set=train)
    for _ in range(3):
        bst.update()
    if spec.get("reset_params"):
        bst.reset_parameter(spec["reset_params"])
    if spec.get("reset_data"):
        X, y, _ = _data(spec["shape"], seed=31)
        bst.update(train_set=flc.Dataset(X, label=y, reference=train, params=params))
    for _ in range(5):
        bst.update()
    text = bst.model_to_string()
    pred = bst.predict(X)
    assert np.all(np.isfinite(pred))
    np.testing.assert_array_equal(pred, flc.Booster(model_str=text).predict(X))
    trees = bst.dump_model()["tree_info"]
    result = {
        "model": hashlib.md5(text.split("\nparameters:")[0].encode()).hexdigest(),
        "prediction": hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest(),
        "leaves": [t["num_leaves"] for t in trees],
        "depths": [_leaf_depths(t["tree_structure"]) for t in trees],
    }
    print("GRAPH_RESULT " + json.dumps(result), flush=True)


def _probe(shape="dense", mode="fp64det", overrides="", **extra):
    spec = {"shape": shape, "mode": mode, "overrides": overrides, **extra}
    env = {**os.environ, "FALCATA_DEBUG": "diag", "FALCATA_VERIFY": "1"}
    run = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe", json.dumps(spec)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    out = run.stdout + run.stderr
    result = json.loads(re.search(r"^GRAPH_RESULT (.+)$", out, re.M).group(1))
    return result, [int(n) for n in _NODES.findall(out)], out


def _assert_graph(out):
    assert _GRAPH in out, out
    assert "falling back" not in out, out
    assert "using the host level loop" not in out, out


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("shape", "mode"),
    [
        ("dense", "fp64det"),
        ("missing", "fp64det"),
        ("gaps", "fp64det"),
        ("dense", "stoch"),
        ("missing", "stoch"),
        ("gaps", "stoch"),
        ("compact", "stoch"),
    ],
)
def test_graph_ports_preserve_models_cuda(shape, mode):
    base, _, out = _probe(shape, mode, _OFF)
    _assert_graph(out)
    if shape == "gaps":
        assert any(min(depths) < max(depths) for depths in base["depths"])
    # Enable each port independently, then all three together.
    for key in (*_KEYS, None):
        override = "" if key is None else ",".join(f"{k}:off" for k in _KEYS if k != key)
        candidate, nodes, out = _probe(shape, mode, override)
        _assert_graph(out)
        assert nodes, out
        assert candidate == base


@_REQUIRES_CUDA
@pytest.mark.parametrize("override", ["graph_apply_rows:off", "apply_row_batch:off"])
def test_graph_rows_legacy_override_cuda(override):
    base, _, out = _probe()
    _assert_graph(out)
    off, _, out = _probe(overrides=override)
    _assert_graph(out)
    assert off == base
    narrow, _, out = _probe(overrides="apply_genbit_rows:off,apply_inner_rows:off")
    _assert_graph(out)
    assert narrow == base


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("override", "extra_nodes"),
    [
        ("graph_apply_fused:off", 2),
        ("apply_struct_fused:off", 1),
        ("gap_copy_fused:off", 1),
    ],
)
def test_graph_fusion_legacy_override_cuda(override, extra_nodes):
    base, base_nodes, _ = _probe(shape="gaps")
    off, nodes, out = _probe(shape="gaps", overrides=override)
    _assert_graph(out)
    assert nodes, out
    assert set(nodes) == {n + extra_nodes for n in base_nodes}, nodes
    assert off == base


@_REQUIRES_CUDA
@pytest.mark.parametrize("min_data", [1599, 1600, 1601, 1602])
def test_graph_pruning_count_margin_cuda(min_data):
    on, _, out = _probe(shape="boundary", params={"min_data_in_leaf": min_data})
    _assert_graph(out)
    off, _, out = _probe(
        shape="boundary", overrides="graph_skip_unsplittable:off", params={"min_data_in_leaf": min_data}
    )
    _assert_graph(out)
    assert on == off
    assert on["leaves"][0] == (4 if min_data <= 1600 else 2)


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", ["fp64det", "stoch"])
@pytest.mark.parametrize("leaf_budget", [7, 19])
def test_graph_ports_leaf_budget_fallback_cuda(mode, leaf_budget):
    # Capture admits only a complete max-depth prefix. A smaller leaf budget
    # must preserve the host fallback's budget and partition behavior.
    extra = {"params": {"num_leaves": leaf_budget, "max_depth": 8}}
    base, nodes, out = _probe(mode=mode, overrides=_OFF, **extra)
    assert _GRAPH not in out, out
    assert not nodes, out
    on, nodes, out = _probe(mode=mode, **extra)
    assert _GRAPH not in out, out
    assert not nodes, out
    assert on == base
    assert all(leaves == leaf_budget for leaves in on["leaves"])


@_REQUIRES_CUDA
@pytest.mark.parametrize("reset_data", [False, True])
def test_graph_ports_follow_reset_cuda(reset_data):
    extra = {"reset_data": reset_data, "reset_params": {"min_data_in_leaf": 100, "num_leaves": 63}}
    base, _, out = _probe(overrides=_OFF, **extra)
    _assert_graph(out)
    on, _, out = _probe(**extra)
    _assert_graph(out)
    assert out.count(_GRAPH) >= 2, out
    assert on == base


@_REQUIRES_CUDA
def test_graph_ports_forced_split_fallback_cuda(tmp_path):
    forced = tmp_path / "forced.json"
    forced.write_text(json.dumps({"feature": 0, "threshold": 0.5}), encoding="utf-8")
    extra = {"params": {"forcedsplits_filename": str(forced)}}
    base, _, _ = _probe(shape="gaps", overrides=_OFF, **extra)
    on, nodes, out = _probe(shape="gaps", **extra)
    assert _GRAPH not in out, out
    assert not nodes, out
    assert on == base


if __name__ == "__main__":
    assert sys.argv[1] == "--probe"
    _child(json.loads(sys.argv[2]))
