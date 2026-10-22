# coding: utf-8
"""A complete device tree whose level root search finds no split must grow exactly what the host flows grow.

The host level flows search a split-less root again with the classic per-leaf search before their leaf-wise tail; the
complete device tree has no host tail, so it runs that same single search (and continues on the host only if it finds a
split). Before the per-tree feature sample was translated to bin-matrix columns, EFB data with feature_fraction built
the batched root histogram for the wrong columns, and that search then found splits the level search could not (see
test_cuda_compact_layout_feature_mask.py). Now both constructs build the same columns into a cleared slot and both
finders search the same sampled features, so every split-less root is a genuine stump (host_tail=0): this fixture
(EFB data, feature_fraction, min_gain_to_split) reaches such roots mid-training, after complete device trees and,
with graphs, after cache hits, and the device must record the same stumps and the same model as the host.
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
        "min_gain_to_split": 5.0,
        "quant_mode": "stochastic",
        "seed": 0,
        "num_threads": 8,
        "metric": "None",
        "verbose": -1,
    }
    models = []
    counts = []
    for suffix in ["device_levels:off", "device_tree:on,device_level_graph:" + graph]:
        p = {**params, "cuda_plan": "auto," + suffix}
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
    # (the host arm reports plan_off; a split-less device root continues on the classic tail: root_tail_parity)
    assert not re.search(r"device_tree: fallback \((?!plan_off\)|root_tail_parity\))", result.stderr)
    if any(event[3] == "1" for event in events):
        assert "device_tree: fallback (root_tail_parity) route=classic\n" in result.stderr
    # the device must not record an empty tree where the host grows one, and the models must be identical
    stumps = [i for i, (h, d) in enumerate(zip(data["host"], data["device"], strict=True)) if d == 1 and h > 1]
    assert not stumps, f"device stumps where the host grows trees: {stumps}"
    assert data["device"] == data["host"]
    assert data["same_model"], "host/device model bytes differ"
    # the fixture reaches split-less level roots mid-training (after earlier complete device trees and, with graphs,
    # after cache hits); each is a genuine stump: the classic root search finds no split either (host_tail=0)
    assert all(e[3] == "0" for e in events), "a classic root search found a split the level root search missed"
    roots = [i for i, e in enumerate(events) if int(e[2]) == 0]
    assert roots, "fixture no longer reaches a level root without a valid split"
    assert roots[-1] > 0
    if graph == "on":
        assert any(int(events[i][5]) > 0 for i in roots)
    assert all(data["host"][i] == 1 for i in roots)
