"""Focused correctness and bit-identity tests for Wave 4 Lever A6 CUDA objective hotfixes.

Covers:
1. Poisson: hoisted exp(max_delta_step_) + gradient_fp64_lean settled-float exp path
   (weighted/unweighted, max_delta_step=0.7 and 1.5, quantized and non-quantized deterministic,
   including high-cancellation / extreme-score fallback regimes).
2. LambdaRank: cached device_warp_size_, exact FP32 sqrtf + integer-corrected pair-index
   decoding (covering <=1024, 1025..2048, and >2048 doc queries), and gradients_no_sync.
3. Multiclass Softmax: 256-thread zero-spill MAX_CLASS=32 register kernel, 2-pass scratch
   kernel for num_class > 32, and lazy cuda_softmax_buffer_ allocation (including
   ConvertOutputCUDA with validation sets smaller and larger than training set).
"""

import hashlib
import math
import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)


def _model_md5(params, X, y, rounds, *, weight=None, group=None, init_score=None):
    ds = flc.Dataset(X, label=y, weight=weight, group=group, init_score=init_score, params=params)
    booster = flc.train(params, ds, num_boost_round=rounds)
    lines = [line for line in booster.model_to_string().splitlines() if "cuda_plan" not in line]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def test_lambdarank_pair_row_decode_exact():
    """Verify FP32 sqrtf + integer adjustment matches original FP64 pair-row decoding identically."""

    def decode_ref(pair_index, s):
        square = 2.0 * float(pair_index) + float(s * s - s)
        sqrt_result = math.floor(math.sqrt(square))
        return int(math.floor(math.sqrt(square - sqrt_result)) + 1 - s)

    def decode_32(pair_index, s):
        n = 2 * pair_index + s * (s - 1)
        m = int(np.float32(np.sqrt(np.float32(n))) + np.float32(0.5))
        if m * (m - 1) > n:
            m -= 1
        if m * (m + 1) <= n:
            m += 1
        return m - s

    def decode_64(pair_index, s):
        n = 2 * pair_index + s * (s - 1)
        m = int(np.float32(np.sqrt(np.float32(n))) + np.float32(0.5))
        while m * (m - 1) > n:
            m -= 1
        while m * (m + 1) <= n:
            m += 1
        return m - s

    # Exhaustive check for small/medium query sizes and boundary truncation levels
    for cnt in (2, 3, 4, 10, 32, 50, 128):
        for trunc in (1, 5, cnt // 2, cnt - 1, cnt + 10):
            num_items_i = min(cnt - 1, trunc)
            num_j_per_i = cnt - 1
            s = num_j_per_i - num_items_i + 1
            num_pairs = (num_j_per_i + s) * num_items_i // 2
            for pair_index in range(num_pairs):
                r_ref = decode_ref(pair_index, s)
                assert decode_32(pair_index, s) == r_ref
                assert decode_64(pair_index, s) == r_ref

    # Boundary pair indices at max shared-memory query size (2048) and sorted path (50000)
    for cnt, trunc, decoder in (
        (2048, 2048, decode_32),
        (2048, 100, decode_32),
        (50000, 100, decode_64),
    ):
        num_items_i = min(cnt - 1, trunc)
        num_j_per_i = cnt - 1
        s = num_j_per_i - num_items_i + 1
        for r in (0, 1, num_items_i // 2, num_items_i - 1):
            start = (2 * s + r - 1) * r // 2
            end = (2 * s + r) * (r + 1) // 2 - 1
            for pair_index in (start, min(start + 1, end), (start + end) // 2, end):
                assert decoder(pair_index, s) == r


@_REQUIRES_CUDA
@pytest.mark.parametrize("max_delta_step", [0.7, 1.5])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("quantized", [True, False])
def test_poisson_lean_and_hoisted_exp_bit_identical_cuda(max_delta_step, weighted, quantized):
    rows, cols = 20011, 12
    rng = np.random.default_rng(17)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    rate = np.exp(0.4 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2])
    y = rng.poisson(np.clip(rate, 0.05, 50.0)).astype(np.float32)
    weight = rng.uniform(0.5, 2.5, size=rows).astype(np.float32) if weighted else None
    base_plan = "auto" if quantized else "auto,nonquant_deterministic:on"
    params = {
        "device_type": "cuda",
        "objective": "poisson",
        "poisson_max_delta_step": max_delta_step,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 10,
        "learning_rate": 0.1,
        "use_quantized_grad": quantized,
        "verbosity": -1,
        "seed": 11,
    }
    md5_on = _model_md5({**params, "cuda_plan": base_plan}, X, y, rounds=25, weight=weight)
    md5_off = _model_md5(
        {**params, "cuda_plan": f"{base_plan},gradient_fp64_lean:off"},
        X,
        y,
        rounds=25,
        weight=weight,
    )
    assert md5_on == md5_off


@_REQUIRES_CUDA
@pytest.mark.parametrize("weighted", [False, True])
def test_poisson_extreme_init_score_and_cancellation_bit_identical_cuda(weighted):
    """Exercise Poisson lean exp() fallback paths: |score| in {60, 85} and exact exp(score) == label cancellation."""
    rows, cols = 4096, 8
    rng = np.random.default_rng(71)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    y = rng.integers(1, 20, size=rows).astype(np.float32)
    init_score = np.log(y.astype(np.float64))  # exact e == label cancellation on first round
    init_score[:256] = 60.0
    init_score[256:512] = -60.0
    init_score[512:768] = 85.0  # outside |score| <= 80 lean gate -> fallback to libdevice exp
    init_score[768:1024] = -85.0  # outside |score| <= 80 lean gate -> fallback to libdevice exp
    init_score[1024:1536] = rng.uniform(-5.0, 5.0, size=512)
    weight = rng.uniform(0.5, 2.0, size=rows).astype(np.float32) if weighted else None
    params = {
        "device_type": "cuda",
        "objective": "poisson",
        "boost_from_average": False,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 10,
        "learning_rate": 0.05,
        "use_quantized_grad": False,
        "verbosity": -1,
        "seed": 73,
    }
    base = "auto,nonquant_deterministic:on"
    md5_on = _model_md5(
        {**params, "cuda_plan": base},
        X,
        y,
        rounds=5,
        weight=weight,
        init_score=init_score,
    )
    md5_off = _model_md5(
        {**params, "cuda_plan": f"{base},gradient_fp64_lean:off"},
        X,
        y,
        rounds=5,
        weight=weight,
        init_score=init_score,
    )
    assert md5_on == md5_off


@_REQUIRES_CUDA
@pytest.mark.parametrize("objective", ["lambdarank", "rank_xendcg"])
@pytest.mark.parametrize("quantized", [True, False])
def test_rank_gradients_no_sync_bit_identical_cuda(objective, quantized):
    """Verify removing redundant LaunchGetGradientsKernel syncs keeps LambdaRank and RankXENDCG bit-identical."""
    rng = np.random.default_rng(79)
    group = [30] * 60
    rows, cols = sum(group), 10
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    signal = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.4 * rng.standard_normal(rows)
    y = np.digitize(signal, np.quantile(signal, [0.2, 0.4, 0.6, 0.8])).astype(np.float32)
    base_plan = "auto" if quantized else "auto,nonquant_deterministic:on"
    params = {
        "device_type": "cuda",
        "objective": objective,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 5,
        "learning_rate": 0.1,
        "use_quantized_grad": quantized,
        "deterministic": True,
        "verbosity": -1,
        "seed": 83,
    }
    md5_on = _model_md5(
        {**params, "cuda_plan": f"{base_plan},gradients_no_sync:on"},
        X,
        y,
        rounds=10,
        group=group,
    )
    md5_off = _model_md5(
        {**params, "cuda_plan": f"{base_plan},gradients_no_sync:off"},
        X,
        y,
        rounds=10,
        group=group,
    )
    assert md5_on == md5_off


@_REQUIRES_CUDA
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("query_sizes", [[50] * 100, [5, 25, 100, 500, 1200], [10, 2200]])
def test_lambdarank_hotfixes_cuda(weighted, deterministic, query_sizes):
    rng = np.random.default_rng(23)
    rows = sum(query_sizes)
    cols = 12
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    signal = X[:, 0] + 0.6 * X[:, 1] - 0.4 * X[:, 2] + 0.5 * rng.standard_normal(rows)
    y = np.digitize(signal, np.quantile(signal, [0.2, 0.4, 0.6, 0.8])).astype(np.float32)
    weight = rng.uniform(0.5, 2.0, size=rows).astype(np.float32) if weighted else None
    params = {
        "device_type": "cuda",
        "objective": "lambdarank",
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 5,
        "learning_rate": 0.1,
        "use_quantized_grad": True,
        "deterministic": deterministic,
        "verbosity": -1,
        "seed": 13,
    }
    ds = flc.Dataset(X, label=y, weight=weight, group=query_sizes, params=params)
    booster = flc.train(params, ds, num_boost_round=15)
    preds = booster.predict(X)
    assert np.all(np.isfinite(preds))
    if deterministic:
        md5_a = _model_md5(params, X, y, rounds=15, weight=weight, group=query_sizes)
        md5_b = _model_md5(params, X, y, rounds=15, weight=weight, group=query_sizes)
        assert md5_a == md5_b


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_class", [7, 24, 32, 40, 64, 100])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("quantized", [True, False])
def test_multiclass_softmax_hotfixes_bit_identical_cuda(num_class, weighted, quantized):
    rows, cols = 8009, 10
    rng = np.random.default_rng(31)
    X = rng.standard_normal((rows, cols)).astype(np.float32)
    signal = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.3 * rng.standard_normal(rows)
    y = np.digitize(signal, np.quantile(signal, np.linspace(0, 1, num_class + 1)[1:-1])).astype(np.float32)
    weight = rng.uniform(0.5, 2.0, size=rows).astype(np.float32) if weighted else None
    base_plan = "auto" if quantized else "auto,nonquant_deterministic:on"
    params = {
        "device_type": "cuda",
        "objective": "multiclass",
        "num_class": num_class,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 5,
        "learning_rate": 0.2,
        "use_quantized_grad": quantized,
        "verbosity": -1,
        "seed": 19,
    }
    md5_on = _model_md5({**params, "cuda_plan": base_plan}, X, y, rounds=8, weight=weight)
    md5_off = _model_md5(
        {**params, "cuda_plan": f"{base_plan},softmax_registers:off"},
        X,
        y,
        rounds=8,
        weight=weight,
    )
    assert md5_on == md5_off


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_class", [8, 32, 48])
def test_multiclass_lazy_softmax_buffer_and_convert_output_cuda(num_class):
    """Verify ConvertOutputCUDA works with lazy buffer for both smaller and larger validation sets."""
    rng = np.random.default_rng(41)
    n_train, n_val_small, n_val_large, cols = 2000, 500, 3500, 8
    X_tr = rng.standard_normal((n_train, cols)).astype(np.float32)
    y_tr = rng.integers(0, num_class, size=n_train).astype(np.float32)
    X_vs = rng.standard_normal((n_val_small, cols)).astype(np.float32)
    y_vs = rng.integers(0, num_class, size=n_val_small).astype(np.float32)
    X_vl = rng.standard_normal((n_val_large, cols)).astype(np.float32)
    y_vl = rng.integers(0, num_class, size=n_val_large).astype(np.float32)
    params = {
        "device_type": "cuda",
        "objective": "multiclass",
        "num_class": num_class,
        "metric": "multi_logloss",
        "num_leaves": 15,
        "max_depth": 4,
        "learning_rate": 0.1,
        "use_quantized_grad": True,
        "verbosity": -1,
        "seed": 42,
    }
    ds_tr = flc.Dataset(X_tr, label=y_tr, params=params)
    ds_vs = flc.Dataset(X_vs, label=y_vs, reference=ds_tr, params=params)
    ds_vl = flc.Dataset(X_vl, label=y_vl, reference=ds_tr, params=params)
    evals = {}
    booster = flc.train(
        params,
        ds_tr,
        num_boost_round=5,
        valid_sets=[ds_vs, ds_vl],
        valid_names=["small", "large"],
        callbacks=[flc.record_evaluation(evals)],
    )
    assert len(evals["small"]["multi_logloss"]) == 5
    assert len(evals["large"]["multi_logloss"]) == 5
    assert all(np.isfinite(evals["small"]["multi_logloss"]))
    assert all(np.isfinite(evals["large"]["multi_logloss"]))
    probs = booster.predict(X_vs)
    assert probs.shape == (n_val_small, num_class)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
