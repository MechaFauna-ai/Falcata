# coding: utf-8
"""A CUDA tree that throws mid-way leaves no state the next tree trusts (CUDASingleGPUTreeLearner::AbortedTreeReset).

The test hook FALCATA_TEST_ABORT_TREE=n makes the learner's n-th tree (0-based) fail once, inside the host level
loop's apply after the tree has at least 4 leaves, as a device-memory failure there would: the histograms of its first
levels are written, beyond the slots the last completed tree used (depth-limited trees that grow from tree to tree). The
caller catches the error and trains on; the next Train sees the tree still in progress and resets what it would take
from it (every histogram slot zeroed: the accumulating constructs would add the leftovers). The shapes keep no other
per-tree state that the failed tree advanced (no per-node or per-tree sampling, deterministic rounding without error
feedback), so the model must be the undisturbed one.
"""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

_CHILD = r"""
import hashlib, json, sys
import numpy as np
import falcata as flc
params, rounds = json.loads(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(3)
X = rng.standard_normal((20000, 16)).astype(np.float32)
y = (X[:, 0] - 0.5 * X[:, 1] + np.sin(X[:, 2]) + 0.3 * rng.standard_normal(20000) > 0).astype(np.float32)
booster = flc.Booster(params, flc.Dataset(X, label=y, params={"max_bin": 63, "verbose": -1}))
failed = []
while booster.current_iteration() < rounds:
    try:
        booster.update()
    except flc.basic.FalcataError as error:
        assert "injected inside a tree" in str(error), error
        failed.append(booster.current_iteration())
text = booster.model_to_string().split("parameters:")[0]
print("RESULT", json.dumps({"md5": hashlib.md5(text.encode()).hexdigest(), "failed": failed}))
"""


def _run(params, rounds, inject=None):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FALCATA_DEBUG", "FALCATA_TEST_ABORT_TREE")}
    if inject is not None:
        env["FALCATA_TEST_ABORT_TREE"] = str(inject)
    res = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(params), str(rounds)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert res.returncode == 0, res.stderr[-3000:]
    line = next(x for x in res.stdout.splitlines() if x.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


_BASE = {
    "device_type": "cuda",
    "objective": "binary",
    "num_leaves": 255,
    "max_depth": 8,
    "min_data_in_leaf": 1,
    "max_bin": 63,
    "metric": "None",
    "seed": 3,
    "verbose": -1,
}


@pytest.mark.parametrize(
    "params",
    [
        # a non-quantized forest: the host level loop with the count construct
        {**_BASE, "boosting": "rf", "quant_mode": "none", "bagging_fraction": 0.632, "bagging_freq": 1},
        # quantized gbdt with interaction constraints: the host level loop with the accumulating integer construct
        {
            **_BASE,
            "quant_mode": "fixedpoint",
            "cuda_plan": "auto,quant_ef:off",
            "interaction_constraints": [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        },
    ],
    ids=["rf-none", "gbdt-fixedpoint-interaction"],
)
def test_aborted_tree_leaves_no_state_cuda(params):
    want = _run(params, 10)
    assert want["failed"] == []
    got = _run(params, 10, inject=5)
    assert got["failed"] == [5], got  # the hook fired once, in round 5, and training went on
    assert got["md5"] == want["md5"]
