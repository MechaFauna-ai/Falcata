"""quant_train_renew_leaf on CUDA sets each leaf to its full-precision Newton output.

The renewal recomputes a quantized tree's leaf outputs from the original gradients of each leaf's rows, as the CPU
GradientDiscretizer::RenewIntGradTreeOutput does: -ThresholdL1(G, lambda_l1) / (H + lambda_l2), capped at
max_delta_step. The CUDA learner reused the refit kernel for it, which blended the result with the quantized output
by refit_decay_rate (0.9 by default: 90 percent of the quantized value stayed) and ignored lambda_l1.
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only regression test")


def _leaf_values(node, out):
    if "leaf_index" in node:
        out[node["leaf_index"]] = node["leaf_value"]
    else:
        _leaf_values(node["left_child"], out)
        _leaf_values(node["right_child"], out)
    return out


@pytest.mark.parametrize("quant_mode", ["stochastic", "fixedpoint"])
@pytest.mark.parametrize(
    ("lambda_l1", "lambda_l2", "max_delta_step"), [(0.0, 0.0, 0.0), (50.0, 10.0, 0.0), (0.0, 0.0, 0.05)]
)
def test_cuda_renewed_leaves_are_the_full_precision_outputs(quant_mode, lambda_l1, lambda_l2, max_delta_step):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20000, 5))
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.3 * rng.standard_normal(20000)
    params = {
        "objective": "regression",
        "device_type": "cuda",
        "num_leaves": 15,
        "learning_rate": 0.1,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "max_delta_step": max_delta_step,
        "quant_mode": quant_mode,
        "quant_train_renew_leaf": True,
        "boost_from_average": False,
        "verbose": -1,
        "seed": 1,
    }
    booster = flc.train(params, flc.Dataset(X, y, params=params), num_boost_round=1)
    leaves = booster.predict(X, pred_leaf=True).ravel().astype(int)
    for leaf, value in _leaf_values(booster.dump_model()["tree_info"][0]["tree_structure"], {}).items():
        rows = leaves == leaf
        g, h = float(np.sum(-y[rows])), float(rows.sum())  # L2 loss at score 0: gradient -y, hessian 1
        output = -np.sign(g) * max(abs(g) - lambda_l1, 0.0) / (h + lambda_l2)
        if max_delta_step > 0:
            output = np.clip(output, -max_delta_step, max_delta_step)
        np.testing.assert_allclose(value, 0.1 * output, rtol=1e-6, atol=1e-9)
