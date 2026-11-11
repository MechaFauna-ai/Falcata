# coding: utf-8
"""A custom objective with quant_mode=none and deterministic=true must train on CUDA.

Booster.update(fobj=...) resets objective=none through ResetConfig. On origin/master that reset map lacked quant_mode,
so Config::Set mapped deterministic=true to quant_mode=fixedpoint again: the booster switched to quantized training
mid-setup and its CUDA learner, built without a gradient discretizer, segfaulted in
CUDASingleGPUTreeLearner::BeforeTrain. The training runs in a child process so the crash fails the test instead of
killing the session.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")

_CHILD = textwrap.dedent(
    """
    import sys
    import numpy as np
    import falcata as flc

    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 5)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=5000) * 0.1).astype(np.float32)

    def fobj(preds, ds):
        return preds - ds.get_label(), np.ones_like(preds)

    params = {"objective": fobj, "device_type": "cuda", "num_leaves": 4, "verbose": -1, "seed": 0,
              "deterministic": sys.argv[1] == "1", "quant_mode": "none"}
    booster = flc.train(params, flc.Dataset(X, label=y), num_boost_round=3)
    pred = booster.predict(X)
    assert np.all(np.isfinite(pred))
    print("leaves", booster.dump_model()["tree_info"][0]["num_leaves"])
    """
)


@_REQUIRES_CUDA
@pytest.mark.parametrize("deterministic", ["1", "0"])
def test_custom_objective_with_quant_mode_none_trains(deterministic):
    out = subprocess.run(
        [sys.executable, "-c", _CHILD, deterministic], capture_output=True, text=True, timeout=600, check=False
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "leaves 4" in out.stdout
