# coding: utf-8
"""CUDA objective gradients must train the same model with or without their fast paths.

cuda_plan keys gradient_fp64_lean (binary logloss, weighted or not: a reduced-FP64 evaluation whose floats are stored
only where every double within its error bound rounds to the same float, else the original double expression),
softmax_registers (multiclass: the row's class scores in registers instead of a global scratch), and
softmax_fp64_lean (multiclass, num_class <= 32: cancellation-free reduced-FP64 softmax with Dekker FastTwoSum
certification in the high-confidence band and exact fallback) change only how the same stored floats are computed, so
the trained model must be byte-identical with each key off. The sweep covers sigmoid values that move many rows into
the lean path's fallbacks (large |x|, the hessian's cancellation), un-quantized and quantized gradients, sample weights
and is_unbalance (the weighted lean variants), extreme multiclass confidence / ties / subnormal / cutoff regimes via
explicit init_score matrices, and class counts on every register capacity and above them (the scratch kernel).
"""

import hashlib
import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

KEYS = ("gradient_fp64_lean", "softmax_registers", "softmax_fp64_lean")


def _model_md5(params, X, y, rounds, *, weight=None, init_score=None):
    ds = flc.Dataset(X, label=y, weight=weight, init_score=init_score, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "extra", "quantized", "weighted"),
    [
        ("binary", {}, True, False),
        ("binary", {}, False, False),
        ("binary", {"sigmoid": 0.3}, True, False),
        ("binary", {"sigmoid": 4.0}, False, False),
        ("binary", {"is_unbalance": True}, True, False),
        ("binary", {}, True, True),
        ("multiclass", {"num_class": 3}, True, False),
        ("multiclass", {"num_class": 7}, False, True),
        ("multiclass", {"num_class": 12}, True, False),
        ("multiclass", {"num_class": 20}, True, False),
        ("multiclass", {"num_class": 40}, True, False),
    ],
)
def test_gradient_keys_are_bit_identical_cuda(objective, extra, quantized, weighted):
    rows, cols = 30011, 12
    rng = np.random.default_rng(5)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    signal = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.3 * rng.standard_normal(rows)
    if objective == "multiclass":
        k = extra["num_class"]
        y = np.digitize(signal, np.quantile(signal, np.linspace(0, 1, k + 1)[1:-1])).astype(float)
    else:
        y = (signal > 0.4).astype(float)
    weight = rng.uniform(0.5, 2.0, size=rows) if weighted else None
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 63,
        "max_depth": 7,
        "min_data_in_leaf": 5,
        "learning_rate": 0.5,
        "use_quantized_grad": quantized,
        "verbosity": -1,
        "seed": 7,
        **extra,
    }
    # compared bit for bit: non-quantized, the deterministic histograms (cuda_plan key nonquant_deterministic, opt-in
    # since 1.2)
    base = "auto" if quantized else "auto,nonquant_deterministic:on"
    reference = _model_md5({**params, "cuda_plan": base}, X, y, rounds=30, weight=weight)
    for key in KEYS:
        assert _model_md5({**params, "cuda_plan": f"{base},{key}:off"}, X, y, rounds=30, weight=weight) == reference, (
            key
        )


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "extra",
    [
        {"use_quantized_grad": True, "bagging_fraction": 0.6, "bagging_freq": 1},
        # non-quantized: the deterministic histograms (opt-in since 1.2), since three trainings are compared
        {"use_quantized_grad": False, "cuda_plan": "auto,device_tree:off,nonquant_deterministic:on"},
        {"use_quantized_grad": True, "objective": "multiclassova", "num_class": 4},
    ],
)
def test_wide_weight_binary_training_is_deterministic_cuda(extra):
    """Sample weights over 2^58 (exp(U(-20, 20)) as float32) do not sum exactly in double, so the binary init score's
    label and weight sums depend on the order the per-block partials are added. That order is now the block order
    (it was the order of the blocks' atomicAdds, which made three trainings give three models): the same data must
    train the same model every time, quantized and bagged, non-quantized, and one-vs-all (the binary init per class)."""
    rows = 100_000
    rng = np.random.default_rng(1)
    X = rng.standard_normal((rows, 20)).astype(np.float32)
    signal = X[:, :8] @ rng.standard_normal(8)
    weight = np.exp(rng.uniform(-20, 20, size=rows)).astype(np.float32)
    params = {
        "device_type": "cuda",
        "objective": "binary",
        "is_unbalance": True,
        "sigmoid": 1.7,
        "num_leaves": 63,
        "max_depth": 7,
        "min_data_in_leaf": 5,
        "learning_rate": 0.2,
        "verbosity": -1,
        "seed": 3,
        **extra,
    }
    if params["objective"] == "multiclassova":
        params.pop("is_unbalance")
        y = np.digitize(signal, np.quantile(signal, [0.25, 0.5, 0.75])).astype(float)
    else:
        y = (signal + rng.standard_normal(rows) > np.median(signal)).astype(float)
    md5s = {_model_md5(params, X, y, rounds=20, weight=weight) for _ in range(3)}
    assert len(md5s) == 1, md5s


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_class", [2, 3, 7, 12, 24, 32])
@pytest.mark.parametrize("weighted", [False, True])
def test_softmax_fp64_lean_extreme_confidence_and_ties_cuda(num_class, weighted):
    """Stress softmax_fp64_lean across ties, extreme confidence (1 - p_max << 1e-5 down to < 2^-55),
    wide sample weights, and class capacities 8 / 16 / 32 in non-quantized deterministic mode (so
    1-ulp gradient/hessian differences are never masked by gradient quantization)."""
    rows, cols = 8009, 8
    rng = np.random.default_rng(100 + num_class * 13 + int(weighted))
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    # Strong linear separation + high learning_rate pushes 1 - p_max rapidly across 1e-4 .. 1e-18
    logits = (
        X[:, : min(cols, num_class)]
        if num_class <= cols
        else X @ rng.standard_normal((cols, num_class)).astype(np.float32)
    )
    y = np.argmax(logits, axis=1).astype(float)
    weight = np.exp(rng.uniform(-6.0, 6.0, size=rows)).astype(np.float32) if weighted else None
    params = {
        "device_type": "cuda",
        "objective": "multiclass",
        "num_class": num_class,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 5,
        "learning_rate": 1.2,
        "use_quantized_grad": False,
        "verbosity": -1,
        "seed": 11,
    }
    base = "auto,nonquant_deterministic:on"
    ref = _model_md5({**params, "cuda_plan": f"{base},softmax_fp64_lean:off"}, X, y, rounds=25, weight=weight)
    lean = _model_md5({**params, "cuda_plan": f"{base},softmax_fp64_lean:on"}, X, y, rounds=25, weight=weight)
    assert lean == ref, f"num_class={num_class} weighted={weighted}"


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_class", [2, 8, 9, 16, 17, 32])
@pytest.mark.parametrize("weighted", [False, True])
def test_softmax_fp64_lean_init_score_edge_cases_cuda(num_class, weighted):
    """Exercise every branch of softmax_fp64_lean via explicit init_score matrices on 1-round
    non-quantized deterministic training: exact 2-way/3-way ties, S_other < 2^-55, S_other in
    [2^-55, 1e-5] (Dekker FastTwoSum exact class-order path), float-subnormal gaps (87..105),
    weighted/unweighted exp cutoffs (> 650 / 700), and extreme sample weights (0, 1e-30, 1e30)."""
    rows, cols = 2048, 4
    rng = np.random.default_rng(700 + num_class * 17 + int(weighted))
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    y = (np.arange(rows) % num_class).astype(float)
    init = np.zeros((rows, num_class), dtype=np.float64)

    for i in range(rows):
        kmax = i % num_class
        k2 = (kmax + 1) % num_class
        k3 = (kmax + 2) % num_class
        base_score = rng.uniform(-50.0, 50.0)
        mode = i % 8
        if mode == 0:
            # Exact 2-way tie (and all-equal when num_class == 2)
            init[i, :] = base_score - rng.uniform(0.5, 5.0, size=num_class)
            init[i, kmax] = base_score
            init[i, k2] = base_score
        elif mode == 1:
            # Exact 3-way tie (or all-equal for num_class <= 3)
            init[i, :] = base_score - rng.uniform(0.5, 5.0, size=num_class)
            init[i, kmax] = base_score
            init[i, k2] = base_score
            init[i, k3] = base_score
        elif mode == 2:
            # S_other < 2^-55 (exact wsum == 1.0 path, gaps 40..60)
            init[i, :] = base_score - rng.uniform(40.0, 60.0, size=num_class)
            init[i, kmax] = base_score
        elif mode == 3:
            # S_other in [2^-55, 1e-5] (Dekker FastTwoSum exact class-order path, gaps 12..37)
            init[i, :] = base_score - rng.uniform(12.0, 37.0, size=num_class)
            init[i, kmax] = base_score
        elif mode == 4:
            # Logit gaps 87..105 (float subnormal outputs -> fallback)
            init[i, :] = base_score - rng.uniform(87.0, 105.0, size=num_class)
            init[i, kmax] = base_score
        elif mode == 5:
            # Logit gaps crossing weighted (-650) and unweighted (-700) cutoffs
            init[i, :] = base_score - rng.uniform(651.0, 750.0, size=num_class)
            init[i, kmax] = base_score
        elif mode == 6:
            # Mixed: one moderate competitor + one deep-underflow class (> 700)
            init[i, :] = base_score - rng.uniform(705.0, 900.0, size=num_class)
            init[i, kmax] = base_score
            init[i, k2] = base_score - rng.uniform(0.1, 15.0)
        else:
            # Moderate / Covtype-like regime (S_other ~ 1e-5 .. 1.0)
            init[i, :] = base_score - rng.uniform(0.0, 12.0, size=num_class)
            init[i, kmax] = base_score

    params = {
        "device_type": "cuda",
        "objective": "multiclass",
        "num_class": num_class,
        "num_leaves": 4,
        "max_depth": 2,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 0.0,
        "learning_rate": 1.0,
        "boost_from_average": False,
        "use_quantized_grad": False,
        "verbosity": -1,
        "seed": 19,
    }
    base = "auto,nonquant_deterministic:on"

    if not weighted:
        ref = _model_md5({**params, "cuda_plan": f"{base},softmax_fp64_lean:off"}, X, y, rounds=1, init_score=init)
        lean = _model_md5({**params, "cuda_plan": f"{base},softmax_fp64_lean:on"}, X, y, rounds=1, init_score=init)
        assert lean == ref, f"unweighted num_class={num_class}"
    else:
        # Pass 1: normal weights mixed with exact 0.0 and tiny 1e-30 weights
        w_small = rng.uniform(0.5, 2.0, size=rows).astype(np.float32)
        w_small[::16] = 0.0
        w_small[1::16] = 1e-30
        ref1 = _model_md5(
            {**params, "cuda_plan": f"{base},softmax_fp64_lean:off"}, X, y, rounds=1, weight=w_small, init_score=init
        )
        lean1 = _model_md5(
            {**params, "cuda_plan": f"{base},softmax_fp64_lean:on"}, X, y, rounds=1, weight=w_small, init_score=init
        )
        assert lean1 == ref1, f"weighted small/zero num_class={num_class}"

        # Pass 2: huge 1e30 weights on their own so they drive every split without drowning small weights
        w_huge = rng.uniform(0.5e30, 1.5e30, size=rows).astype(np.float32)
        ref2 = _model_md5(
            {**params, "cuda_plan": f"{base},softmax_fp64_lean:off"}, X, y, rounds=1, weight=w_huge, init_score=init
        )
        lean2 = _model_md5(
            {**params, "cuda_plan": f"{base},softmax_fp64_lean:on"}, X, y, rounds=1, weight=w_huge, init_score=init
        )
        assert lean2 == ref2, f"weighted huge num_class={num_class}"
