# coding: utf-8
"""A complete device tree whose level root search finds no split must still grow what the host flows grow.

The host level flows search the root again with the classic per-leaf search before their leaf-wise tail. On an
EFB-bundled multiclass dataset with feature_fraction, a sampled compact layout first seen mid-training can leave the
batched level root search without a valid split while that classic search finds one. The complete device tree has
no host tail of its own, so it must take that same single root search instead of recording an empty tree.
"""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _data():
    # Six dense 16-bin columns plus 30 mutually exclusive one-hot features, which EFB bundles into shared columns.
    rng = np.random.default_rng(0)
    rows = 20000
    dense = np.floor(rng.random((rows, 6)) * 16).astype(np.float32)
    cat = rng.integers(0, 30, size=rows)
    onehot = np.zeros((rows, 30), dtype=np.float32)
    onehot[np.arange(rows), cat] = 1.0
    X = np.hstack([dense, onehot])
    score = dense[:, 0] + 0.5 * dense[:, 1] + 3.0 * (cat % 3) + rng.standard_normal(rows)
    y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3])).astype(np.float32)
    return X, y


def _probe(graph):
    X, y = _data()
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "device_type": "cuda",
        "num_leaves": 16,
        "max_depth": 4,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.5,
        "quant_mode": "stochastic",
        "seed": 0,
        "num_threads": 8,
        "metric": "None",
        "verbose": -1,
    }
    models = []
    counts = []
    for suffix in ["device_tree:off", "device_tree:on,device_level_graph:" + graph]:
        p = {**params, "cuda_plan": "auto,device_levels:off," + suffix}
        booster = flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=24)
        models.append("\n".join(line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line))
        counts.append([t["num_leaves"] for t in booster.dump_model()["tree_info"]])
    print(json.dumps({"host": counts[0], "device": counts[1], "same_model": models[0] == models[1]}), flush=True)


@_REQUIRES_CUDA
@pytest.mark.parametrize("graph", ["off", "on"])
def test_device_tree_root_tail_matches_host_cuda(graph):
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_probe'](*sys.argv[2:])"
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, graph],
        env=dict(os.environ, FALCATA_DEBUG="diag"),
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout.splitlines()[-1])
    events = re.findall(
        r"device_tree: complete depth=(\d+) applied_levels=(\d+) splits=(\d+) "
        r"growth_readbacks=1 host_tail=(\d+) graph=(\d+) hits=(\d+) misses=(\d+)",
        result.stderr,
    )
    # every device tree engaged (no plan fallback)
    assert len(events) == len(data["device"]), result.stderr
    assert "device_tree: host fallback" not in result.stderr
    # the device must not record an empty tree where the host grows one, and the models must be identical
    stumps = [i for i, (h, d) in enumerate(zip(data["host"], data["device"], strict=True)) if d == 1 and h > 1]
    assert not stumps, f"device stumps where the host grows trees: {stumps}"
    assert data["device"] == data["host"]
    assert data["same_model"], "host/device model bytes differ"
    # the fixture reached a level root without a device split, mid-training (after earlier complete device trees
    # and, with graphs, after cache hits), and the host-tail tree is a grown tree
    tails = [i for i, e in enumerate(events) if e[3] == "1"]
    assert tails, "fixture no longer reaches a level root without a valid split"
    assert all(int(events[i][2]) == 0 for i in tails)
    assert tails[0] > 0
    if graph == "on":
        assert int(events[tails[0]][5]) > 0
    assert all(data["host"][i] > 1 for i in tails)
