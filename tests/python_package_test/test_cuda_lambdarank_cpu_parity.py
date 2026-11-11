# coding: utf-8
"""CUDA lambdarank must train the CPU objective's models.

The CUDA objective computes each pair's lambda exactly as LambdarankNDCG does: the discount 1 / log2(2 + rank)
from DCGCalculator's table, p_lambda from the CPU's precomputed sigmoid table (GetSigmoid), item weights, and the
item order of the CPU's stable sort of the double scores. With deterministic=True it also adds every sum in the CPU's
order, so the gradients are the CPU's bit for bit (see tests/cpp_tests/test_cuda_lambdarank_cpu_parity.cu), and
with the same gradients the two tree learners grow the same trees.

quant_mode is pinned to none: on device_type=cuda, deterministic=True resolves quant_mode=auto to fixedpoint
(quantized gradients), which the CPU learner does not have, so the trees would differ for a reason unrelated to
the objective.
"""

import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _data(group):
    rng = np.random.default_rng(0)
    rows = sum(group)
    X = rng.standard_normal((rows, 6))
    relevance = X[:, 0] + 0.5 * X[:, 1] + 0.8 * rng.standard_normal(rows)
    y = np.digitize(relevance, [-1.0, 0.0, 0.8, 1.6]).astype(np.float64)
    return X, y


def _train(device, group, extra):
    X, y = _data(group)
    params = {
        "objective": "lambdarank",
        "device_type": device,
        "deterministic": True,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "seed": 0,
        "num_threads": 8,
        "quant_mode": "none",
        "verbose": -1,
        **extra,
    }
    booster = flc.train(params, flc.Dataset(X, label=y, group=group, params=params), num_boost_round=5)
    return booster.predict(X)


_GROUPS = {"up_to_1024": [200] * 40, "up_to_2048": [1500, 1300, 1700], "sorted": [2600, 3100]}
_EXTRAS = {"default": {}, "no_norm": {"lambdarank_norm": False}, "truncation_10": {"lambdarank_truncation_level": 10}}

# The tree learners, not the objectives, part ways here: the gradients of every round are the CPU's bit for bit (fed
# the CPU model's own scores, the CUDA objective returns identical lambdas and hessians), trees 0-2 are identical, and
# tree 3's ninth split picks a different feature for the same 89-row leaf. With lambdarank_truncation_level=10 most
# hessians are near zero, and min_data_in_leaf gates on counts estimated from hessian sums: the CPU sums per-bin
# RoundInt(hess * cnt_factor), while CUDA split-finder paths round the side's summed hessian (e.g.
# cuda_best_split_finder_kernels.cuh), so the estimates differ by enough to flip the gate. With min_data_in_leaf <= 5
# the models are identical. Strict, so the case starts passing loudly once the learners agree.
_LEARNER_COUNT_ESTIMATE = pytest.mark.xfail(
    strict=True, reason="CPU/CUDA tree learners estimate min_data_in_leaf counts from hessians differently"
)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("group", "extra"),
    [
        pytest.param(
            _GROUPS[g],
            _EXTRAS[e],
            id=f"{g}-{e}",
            marks=[_LEARNER_COUNT_ESTIMATE] if (g, e) == ("sorted", "truncation_10") else [],
        )
        for g in _GROUPS
        for e in _EXTRAS
    ],
)
def test_lambdarank_matches_cpu_cuda(group, extra):
    # the same trees: identical predictions up to the order of the float additions in prediction itself
    np.testing.assert_allclose(_train("cuda", group, extra), _train("cpu", group, extra), rtol=0, atol=1e-12)


@_REQUIRES_CUDA
def test_lambdarank_matches_cpu_cuda_sorted_truncation_without_count_gate():
    """The xfailed case above with min_data_in_leaf=5: the objectives agree, so the models do."""
    extra = {"lambdarank_truncation_level": 10, "min_data_in_leaf": 5}
    np.testing.assert_allclose(
        _train("cuda", _GROUPS["sorted"], extra), _train("cpu", _GROUPS["sorted"], extra), rtol=0, atol=1e-12
    )
