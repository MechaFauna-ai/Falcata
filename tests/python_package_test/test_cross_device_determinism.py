# coding: utf-8
"""CUDA training must give the same trees on every card model of an architecture.

Every launch-shape decision of the CUDA tree learner that reads the device's multiprocessor count (the batched
construct's saturation floor, the fused root fill's tiles per block, the device-level construct's target blocks, the
count histogram's tiling and the live-grid budget, the split finder's warps per wide task, the pair-joint construct's
device slots) goes through FalcataDeviceSMCount (include/Falcata/cuda/cuda_device_sm.hpp). FALCATA_DEBUG=smcount=N
makes all of them see N multiprocessors. A card of the same architecture with another SM count (an RTX 5090 has 170,
an RTX PRO 6000 has 188) therefore runs exactly the schedule that smcount=N runs on the card under test. These tests
require the model to be byte-identical for every N, so a decision that changes results with the card is caught on
a single card. The token is read once per process, so every run is a subprocess.

Measured 2026-11-17 for the PR that added the token: the same models on an RTX 5090 (170 SMs) and an RTX PRO 6000
Max-Q (188 SMs), and on each card with smcount=170 and 188.
"""

import os
import re
import subprocess
import sys

import pytest

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

_CHILD = """
import hashlib, re
import numpy as np
import falcata as flc

rng = np.random.default_rng(5)
n, p = {rows}, {feats}
X = rng.standard_normal((n, p)).astype(np.float32)
X[:, : p // 4] = np.round(X[:, : p // 4] * 3) / 3
if {missing}:
    X[rng.random((n, p)) < 0.3] = np.nan
y = (X[:, :8] @ rng.standard_normal(8) * 0.05 + rng.standard_normal(n)).astype(np.float32)
w = (rng.gamma(2.0, 0.5, n) + 0.05).astype(np.float32) if {weighted} else None
params = {{"objective": "regression", "device_type": "cuda", "seed": 3, "verbosity": -1,
           "min_data_in_leaf": 25, "learning_rate": 0.1, **{params!r}}}
ds = flc.Dataset(X, label=y, weight=w, params={{"max_bin": params.get("max_bin", 63), "device_type": "cuda"}})
bst = flc.train(params, ds, {rounds})
model = bst.model_to_string()
print("MODEL_MD5", hashlib.md5(model.encode()).hexdigest(), flush=True)
"""

CASES = {
    # the shape of a production fit that diverged across cards (later traced to its inputs): weighted rows, 30%
    # missing, bagging and column sampling, many leaves, fixed-point quantized gradients
    "fixedpoint_weighted": {
        "rows": 60_000,
        "feats": 120,
        "missing": True,
        "weighted": True,
        "rounds": 40,
        "params": {
            "quant_mode": "fixedpoint",
            "cuda_precision": "fp32",
            "num_leaves": 255,
            "max_bin": 51,
            "feature_fraction": 0.2,
            "bagging_fraction": 0.5,
            "bagging_freq": 1,
            "lambda_l2": 10.0,
        },
    },
    "stochastic_large": {
        "rows": 400_000,
        "feats": 60,
        "missing": True,
        "weighted": False,
        "rounds": 20,
        "params": {
            "quant_mode": "stochastic",
            "num_leaves": 127,
            "max_bin": 63,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
        },
    },
    "unquantized_fp32": {
        "rows": 400_000,
        "feats": 60,
        "missing": False,
        "weighted": True,
        "rounds": 20,
        "params": {"quant_mode": "none", "cuda_precision": "fp32", "num_leaves": 127, "max_bin": 63},
    },
    "wide_255_bins": {
        "rows": 60_000,
        "feats": 800,
        "missing": False,
        "weighted": False,
        "rounds": 15,
        "params": {"quant_mode": "fixedpoint", "num_leaves": 63, "max_bin": 255, "feature_fraction": 0.8},
    },
}

# the real count, two cards of the architecture that ships (5090, PRO 6000), and the extremes of the schedule:
# one multiprocessor (every per-device bound at its minimum) and far more than any card has
SM_COUNTS = [None, 170, 188, 1, 2048]


def _md5(case, smcount):
    c = CASES[case]
    code = _CHILD.format(**c)
    env = dict(os.environ)
    env.pop("FALCATA_DEBUG", None)
    if smcount is not None:
        env["FALCATA_DEBUG"] = f"smcount={smcount}"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
    return re.search(r"^MODEL_MD5 (\w+)$", out, re.M).group(1)


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", sorted(CASES))
def test_trees_do_not_depend_on_the_sm_count_cuda(case):
    md5 = {n: _md5(case, n) for n in SM_COUNTS}
    assert len(set(md5.values())) == 1, md5


@_REQUIRES_CUDA
def test_smcount_token_reaches_the_launch_shape_cuda():
    """The token is not a no-op: the learner logs the emulated count the first time a launch shape reads it."""
    code = _CHILD.format(**CASES["fixedpoint_weighted"]).replace('"verbosity": -1', '"verbosity": 1')
    env = {**os.environ, "FALCATA_DEBUG": "smcount=188"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    log = out.stdout + out.stderr
    assert "unknown token" not in log
    assert re.search(r"FALCATA_DEBUG smcount: launch shapes see 188 multiprocessors \(device \d+ has \d+\)", log), log[
        -2000:
    ]
