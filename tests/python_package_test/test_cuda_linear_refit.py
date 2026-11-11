"""Refitting a linear-tree model on CUDA refits its leaf models, as on the CPU.

The CUDA learner's FitByExistingTree refit only the leaf outputs; a linear leaf predicts with its constant and
coefficients, so a CUDA refit returned the old model's predictions unchanged.
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only regression test")


def _leaf_models(booster):
    def walk(node):
        if "leaf_index" in node:
            yield node["leaf_index"], node.get("leaf_const"), node.get("leaf_coeff", [])
        else:
            yield from walk(node["left_child"])
            yield from walk(node["right_child"])

    return [sorted(walk(tree["tree_structure"])) for tree in booster.dump_model()["tree_info"]]


@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("decay_rate", [0.0, 0.5])
def test_cuda_linear_refit_matches_cpu(with_nan, decay_rate):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 3))
    y = x[:, 0] + x[:, 1] - x[:, 2]
    refit_y = 3 * x[:, 0] - 2 * x[:, 1] + 1
    if with_nan:
        x[rng.random(len(x)) < 0.05, 1] = np.nan
    refits = {}
    for device in ["cpu", "cuda"]:
        params = {
            "objective": "regression",
            "linear_tree": True,
            "num_leaves": 4,
            "device_type": device,
            "deterministic": True,
            "num_threads": 2,
            "verbosity": -1,
        }
        booster = flc.train(params, flc.Dataset(x, label=y), num_boost_round=4)
        refits[device] = booster.refit(x, refit_y, decay_rate=decay_rate)
    cpu, cuda = _leaf_models(refits["cpu"]), _leaf_models(refits["cuda"])
    assert [[leaf for leaf, _, _ in tree] for tree in cuda] == [[leaf for leaf, _, _ in tree] for tree in cpu]
    for cpu_tree, cuda_tree in zip(cpu, cuda, strict=True):
        for (_, cpu_const, cpu_coeff), (_, cuda_const, cuda_coeff) in zip(cpu_tree, cuda_tree, strict=True):
            # the device gram is a fixed-order sum in another order than the CPU threads', float gradients
            np.testing.assert_allclose(cuda_const, cpu_const, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(cuda_coeff, cpu_coeff, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(refits["cuda"].predict(x), refits["cpu"].predict(x), rtol=1e-4, atol=1e-5)
