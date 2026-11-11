"""CPU fallback metric transfers must refresh between CUDA evaluations."""

import ctypes
import os

import numpy as np
import pytest

import falcata as lgb
from falcata.basic import _LIB, _safe_call


@pytest.mark.skipif(os.environ.get("TASK") != "cuda", reason="requires CUDA build and GPU")
def test_cuda_metric_scores_refresh_across_updates_and_datasets():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1024, 4))
    labels = (0.5 + 0.2 * np.tanh(x[:, 0])).astype(np.float32)
    params = {
        "device_type": "cuda",
        "quant_mode": "none",
        "objective": "regression",
        "metric": ["r2", "cross_entropy", "l2"],
        "num_threads": 4,
        "num_leaves": 3,
        "verbosity": -1,
    }
    training = lgb.Dataset(x, label=labels)
    validation = lgb.Dataset(x[::2], label=labels[::2], reference=training)
    model = lgb.Booster(params, train_set=training)
    model.add_valid(validation, "validation")
    previous = None
    for _ in range(3):
        model.update()
        for data_index, (values, target, evaluate) in enumerate(
            [(x, labels, model.eval_train), (x[::2], labels[::2], model.eval_valid)]
        ):
            scores = np.empty(len(values))
            length = ctypes.c_int64()
            _safe_call(
                _LIB.FLC_BoosterGetPredict(
                    model._handle,
                    data_index,
                    ctypes.byref(length),
                    scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                )
            )
            assert length.value == len(values)
            target = target.astype(np.float64)
            expected = {
                "l2": np.mean((scores - target) ** 2),
                "r2": 1 - np.sum((scores - target) ** 2) / np.sum((target - target.mean()) ** 2),
                "cross_entropy": np.mean(-target * np.log(scores) - (1 - target) * np.log1p(-scores)),
            }
            for _ in range(2):
                actual = {entry[1]: entry[2] for entry in evaluate()}
                for metric, value in expected.items():
                    assert actual[metric] == pytest.approx(value, rel=1e-7, abs=1e-9)
        if previous is not None:
            assert actual["l2"] != previous
        previous = actual["l2"]
