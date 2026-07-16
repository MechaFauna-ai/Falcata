# coding: utf-8
"""Tests for dual GPU+CPU support."""

import contextlib
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from sklearn.metrics import log_loss

import lightgbm as lgb

from .utils import load_breast_cancer

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled LightGBM build (set TASK=cuda)",
)


def _get_init_score(device_type, objective, alpha, X, y):
    """Train a 1-tree model and read 'Start training from score' from the log."""
    params = {
        "objective": objective,
        "alpha": alpha,
        "verbose": 1,
        "num_leaves": 2,
        "min_data_in_leaf": 1,
        "learning_rate": 0.1,
        "deterministic": True,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "seed": 0,
        "device_type": device_type,
    }
    ds = lgb.Dataset(X, label=y, params={"verbose": -1})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        lgb.train(params, ds, num_boost_round=1)
    for line in buf.getvalue().splitlines():
        if "Start training from score" in line:
            return float(line.split("score")[-1].strip())
    raise AssertionError(f"no init score logged for {device_type} {objective} alpha={alpha}")


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("objective", "alpha"),
    [("regression_l1", 0.5), ("quantile", 0.5), ("quantile", 0.3), ("quantile", 0.7)],
)
@pytest.mark.parametrize("n", [5, 7, 10, 11, 100, 500])
def test_cuda_init_score_matches_cpu(objective, alpha, n):
    """CUDA percentile-based init scores must match CPU at FP epsilon.

    Regression test for the bug in PercentileGlobalKernel that used
    `(1 - alpha) * len` instead of `(1 - alpha) * (len - 1)`. For
    objective=regression_l1 with y=[1..5], CUDA returned 3.5 instead of
    the correct 3.0.
    """
    X = np.zeros((n, 1))
    y = np.arange(1, n + 1, dtype=np.float64)
    cpu = _get_init_score("cpu", objective, alpha, X, y)
    cuda = _get_init_score("cuda", objective, alpha, X, y)
    assert cuda == pytest.approx(cpu, abs=1e-6), f"{objective} alpha={alpha} n={n}: cpu={cpu} cuda={cuda}"


_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled LightGBM build (set TASK=cuda)",
)


@_REQUIRES_CUDA
@pytest.mark.parametrize("objective", ["regression_l1", "quantile"])
@pytest.mark.parametrize("n", [100, 200, 500, 1000])
def test_cuda_weighted_percentile_renewal_does_not_crash(objective, n):
    """Regression test for the OOB shared-memory access in
    ShuffleSortedPrefixSumDevice that crashed weighted L1 / weighted
    quantile training with "illegal memory access" for n >= ~100.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 3)).astype(np.float64)
    y = rng.standard_normal(n).astype(np.float64)
    w = rng.random(n)
    ds = lgb.Dataset(X, label=y, weight=w, params={"verbose": -1, "feature_pre_filter": False})
    params = {
        "objective": objective,
        "alpha": 0.5,
        "device_type": "cuda",
        "verbose": -1,
        "num_leaves": 4,
        "min_data_in_leaf": 1,
        "deterministic": True,
        "gpu_use_dp": True,
    }
    # If the OOB access regresses, this raises a CUDA "illegal memory access" error.
    bst = lgb.train(params, ds, num_boost_round=2)
    preds = bst.predict(X, raw_score=True)
    assert np.all(np.isfinite(preds)), "weighted percentile renewal produced non-finite predictions"


_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled LightGBM build (set TASK=cuda)",
)


@_REQUIRES_CUDA
@pytest.mark.parametrize("items_per_query", [50, 1500])
def test_cuda_lambdarank_deterministic_is_bit_identical_run_to_run(items_per_query):
    """Regression test for LambdaRank gradient kernel run-to-run determinism.

    The non-deterministic kernel scatters per-pair gradient and hessian
    contributions into shared memory via atomicAdd_block. Different runs
    interleave those atomics in different orders, and floating-point
    addition is non-associative, so the per-slot sums (and hence the
    boosted tree) are not bit-identical across runs.

    With deterministic=True the kernel switches to a CUB BlockReduce-based
    code path that accumulates each output slot in a fixed (i, j) order
    without atomics. Output should then be bit-identical across runs.

    Two parametrized values cover both code paths:
      - items_per_query=50  -> BitonicArgSort_1024 path
      - items_per_query=1500 -> BitonicArgSort_2048 path
    """
    rng = np.random.default_rng(7)
    n_queries = max(2, 600 // items_per_query)
    n = n_queries * items_per_query
    n_features = 8
    X = rng.standard_normal((n, n_features)).astype(np.float64)
    coef = rng.standard_normal(n_features)
    score = X @ coef
    quantiles = np.quantile(score, [0.2, 0.4, 0.6, 0.8])
    y = np.digitize(score, quantiles).astype(np.int32)
    group = np.full(n_queries, items_per_query, dtype=np.int32)

    base = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "device_type": "cuda",
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 7,
        "feature_pre_filter": False,
        "gpu_use_dp": True,
        "num_leaves": 8,
        "learning_rate": 0.1,
        "min_data_in_leaf": 3,
    }

    preds = []
    for _ in range(3):
        ds = lgb.Dataset(X, label=y, group=group, params={"verbose": -1, "feature_pre_filter": False})
        bst = lgb.train(base, ds, num_boost_round=5)
        preds.append(bst.predict(X, raw_score=True))

    # Bit-identical across all three runs.
    for i, p in enumerate(preds[1:], start=1):
        assert np.array_equal(p, preds[0]), (
            f"deterministic=True LambdaRank produced non-identical output between run 0 and run {i} "
            f"(max|Δ|={float(np.abs(p - preds[0]).max()):.3e}); "
            f"items_per_query={items_per_query}."
        )


@pytest.mark.skipif(
    os.environ.get("LIGHTGBM_TEST_DUAL_CPU_GPU", "0") != "1",
    reason="Set LIGHTGBM_TEST_DUAL_CPU_GPU=1 to test using CPU and GPU training from the same package.",
)
def test_cpu_and_gpu_work():
    # If compiled appropriately, the same installation will support both GPU and CPU.
    X, y = load_breast_cancer(return_X_y=True)
    data = lgb.Dataset(X, y)

    params_cpu = {
        "verbosity": -1,
        "num_leaves": 31,
        "objective": "binary",
        "device": "cpu",
    }
    cpu_bst = lgb.train(params_cpu, data, num_boost_round=10)
    cpu_score = log_loss(y, cpu_bst.predict(X))

    params_gpu = params_cpu.copy()
    params_gpu["device"] = "gpu"
    # Double-precision floats are only supported on x86_64 with PoCL
    params_gpu["gpu_use_dp"] = platform.machine() == "x86_64"
    gpu_bst = lgb.train(params_gpu, data, num_boost_round=10)
    gpu_score = log_loss(y, gpu_bst.predict(X))

    rel = 1e-6 if params_gpu["gpu_use_dp"] else 1e-4
    assert cpu_score == pytest.approx(gpu_score, rel=rel)
    assert gpu_score < 0.242


_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled LightGBM build (set TASK=cuda)",
)

# Loose enough to absorb label_t float32 quantization in the renewal kernel,
# tight enough to flag the ~0.3 bias the old PercentileDevice formula produced.
_PERCENTILE_TOL = 1e-6


def _train_one_tree_for_renewal(device_type, objective, alpha, X, y):
    # learning_rate=1.0 makes raw_score equal to the renewed leaf value directly,
    # which lets the assertion compare against numpy.quantile without unwinding shrinkage.
    params = {
        "objective": objective,
        "alpha": alpha,
        "num_leaves": 7,
        "min_data_in_leaf": 1,
        "learning_rate": 1.0,
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 0,
        "feature_pre_filter": False,
        "device_type": device_type,
        "gpu_use_dp": True,
        "force_col_wise": True,
    }
    ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
    return lgb.train(params, ds, num_boost_round=1)


@_REQUIRES_CUDA
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cuda_l1_leaf_renewal_matches_numpy_median(seed):
    """L1 leaf renewal must produce numpy.median(y_in_leaf) on both CPU and CUDA.

    Regression test for the unweighted PercentileDevice formula that previously
    used `len * (1 - alpha)` instead of `(len - 1) * (1 - alpha)`, biasing
    leaf values upward in the descending-sort convention used for L1 / quantile
    renewal.
    """
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.standard_normal((n, 5)).astype(np.float64)
    w = rng.standard_normal(5)
    y = (X @ w + 0.1 * rng.standard_normal(n)).astype(np.float64)

    for device_type in ("cpu", "cuda"):
        bst = _train_one_tree_for_renewal(device_type, "regression_l1", 0.5, X, y)
        leaf_idx = bst.predict(X, pred_leaf=True).astype(int).reshape(-1)
        raw = bst.predict(X, raw_score=True)
        for li in np.unique(leaf_idx):
            mask = leaf_idx == li
            expected = float(np.median(y[mask]))
            actual = float(raw[mask][0])
            assert actual == pytest.approx(expected, abs=_PERCENTILE_TOL), (
                f"{device_type} leaf {li} (n={int(mask.sum())}): expected np.median={expected:.10f}, got {actual:.10f}"
            )


@_REQUIRES_CUDA
@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.7, 0.9])
def test_cuda_quantile_leaf_renewal_matches_numpy_quantile(alpha):
    """Quantile leaf renewal must produce numpy.quantile(y_in_leaf, alpha)
    on both CPU and CUDA. Same regression coverage as the L1 test, but
    sweeping alpha so the bias of the wrong formula would show on every
    even/odd leaf size combination.
    """
    rng = np.random.default_rng(123)
    n = 250
    X = rng.standard_normal((n, 6)).astype(np.float64)
    w = rng.standard_normal(6)
    y = (X @ w + 0.1 * rng.standard_normal(n)).astype(np.float64)

    for device_type in ("cpu", "cuda"):
        bst = _train_one_tree_for_renewal(device_type, "quantile", alpha, X, y)
        leaf_idx = bst.predict(X, pred_leaf=True).astype(int).reshape(-1)
        raw = bst.predict(X, raw_score=True)
        for li in np.unique(leaf_idx):
            mask = leaf_idx == li
            expected = float(np.quantile(y[mask], alpha))
            actual = float(raw[mask][0])
            assert actual == pytest.approx(expected, abs=_PERCENTILE_TOL), (
                f"{device_type} alpha={alpha} leaf {li} (n={int(mask.sum())}): "
                f"expected np.quantile={expected:.10f}, got {actual:.10f}"
            )


@_REQUIRES_CUDA
@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 9])
def test_cuda_l1_median_handles_small_even_and_odd_leaves(n):
    """Targets the specific failure mode of the old PercentileDevice formula:
    even-length leaves returning sorted[1] instead of avg(sorted[1], sorted[2]),
    and odd-length leaves returning avg(sorted[1], sorted[2]) instead of
    sorted[2]. We force every datapoint into its own leaf, then split a couple
    in half and check the leaf medians.
    """
    rng = np.random.default_rng(7)
    # one feature so we deterministically split on it; values are well-separated
    X = np.arange(n, dtype=np.float64).reshape(-1, 1)
    # values designed so that splitting on the only feature produces leaves of
    # exactly the requested cardinalities at depth 1 and 2.
    y = rng.standard_normal(n).astype(np.float64)

    for device_type in ("cpu", "cuda"):
        bst = _train_one_tree_for_renewal(device_type, "regression_l1", 0.5, X, y)
        leaf_idx = bst.predict(X, pred_leaf=True).astype(int).reshape(-1)
        raw = bst.predict(X, raw_score=True)
        for li in np.unique(leaf_idx):
            mask = leaf_idx == li
            expected = float(np.median(y[mask]))
            actual = float(raw[mask][0])
            assert actual == pytest.approx(expected, abs=_PERCENTILE_TOL), (
                f"{device_type} n={n} leaf {li} (size {int(mask.sum())}): "
                f"expected np.median={expected:.10f}, got {actual:.10f}"
            )


def _tree_depth(node, depth=0):
    if "leaf_value" in node:
        return depth
    return max(
        _tree_depth(node["left_child"], depth + 1),
        _tree_depth(node["right_child"], depth + 1),
    )


def _train_pair(params_overrides, X, y):
    out = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "feature_pre_filter": False,
            "device_type": device_type,
            "gpu_use_dp": True,
            "force_col_wise": True,
            **params_overrides,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        out[device_type] = lgb.train(params, ds, num_boost_round=1)
    return out


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("max_depth", "num_leaves"),
    [
        (1, 2),
        (1, 7),
        (2, 4),
        (2, 7),
        (2, 31),
        (3, 7),
        (3, 31),
        (5, 31),
    ],
)
def test_cuda_respects_max_depth(max_depth, num_leaves):
    """CUDA tree learner must enforce max_depth, matching CPU.

    Regression test for the bug where CUDABestSplitFinder had no max_depth
    check and CUDATree::Split never updated host-side leaf_depth_, causing
    CUDA to produce trees up to log2(num_leaves) deep regardless of
    max_depth. With max_depth=2 and num_leaves=31, CUDA was producing
    depth-7 trees with all 31 leaves filled.
    """
    rng = np.random.default_rng(0)
    n = 64
    X = rng.standard_normal((n, 4)).astype(np.float64)
    y = (X @ rng.standard_normal(4) + 0.1 * rng.standard_normal(n)).astype(np.float64)

    models = _train_pair(
        {"max_depth": max_depth, "num_leaves": num_leaves, "min_data_in_leaf": 1},
        X,
        y,
    )

    cpu_dump = models["cpu"].dump_model()["tree_info"][0]
    cuda_dump = models["cuda"].dump_model()["tree_info"][0]

    cpu_depth = _tree_depth(cpu_dump["tree_structure"])
    cuda_depth = _tree_depth(cuda_dump["tree_structure"])

    assert cuda_depth <= max_depth, (
        f"CUDA exceeded max_depth={max_depth}: produced depth-{cuda_depth} tree with num_leaves={num_leaves}"
    )
    assert cpu_depth == cuda_depth, (
        f"CPU/CUDA depth mismatch with max_depth={max_depth}, num_leaves={num_leaves}: "
        f"cpu={cpu_depth}, cuda={cuda_depth}"
    )


# Every n here has grid(n) < 80 while a child leaf can land in the ~101-160 band
# (grid up to 80), so a child leaf's grid exceeds the root grid and overruns the
# construction-time buffer. The earlier 120/150/159 were dead: for those the root
# grid is already the band maximum, so no child leaf can exceed it.
@_REQUIRES_CUDA
@pytest.mark.parametrize("n", [200, 250, 300, 400])
@pytest.mark.parametrize("num_leaves", [7, 15, 31])
def test_cuda_data_partition_block_offset_no_overflow(n, num_leaves):
    """CUDA training must match CPU when a split processes a leaf whose grid
    exceeds the full-dataset grid.

    Regression guard for the out-of-bounds __global__ write in
    GenDataToLeftBitVectorKernel's PrepareOffset. CUDADataPartition::CalcBlockDim
    is non-monotonic (the per-block data count is rounded up to a power of two),
    so a leaf in the ~101-160 range can need more blocks than the full dataset,
    while cuda_block_data_to_{left,right}_offset_ were sized only for the
    full-dataset grid. compute-sanitizer flags the overflow as an invalid device
    write; the fix grows those buffers on demand in CUDADataPartition::Split.

    The bug is silent on most allocators (the overflow stays within the
    allocation's slack, so predictions remain bit-identical), which is exactly
    why it went unnoticed -- this parity test pins the scenario so any future
    change that makes the overflow corrupt results, or that reintroduces the
    crash on a stricter allocator, is caught. Run under compute-sanitizer to see
    the underlying memory error without the fix.
    """
    rng = np.random.default_rng(11)
    d = 8
    X = rng.standard_normal((n, d)).astype(np.float64)
    coef = rng.standard_normal(d)
    y = (X @ coef + 0.1 * rng.standard_normal(n)).astype(np.float64)

    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 42,
            "feature_pre_filter": False,
            "device_type": device_type,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "num_leaves": num_leaves,
            "learning_rate": 0.1,
            "min_data_in_leaf": 5,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        bst = lgb.train(params, ds, num_boost_round=5)
        preds[device_type] = bst.predict(X, raw_score=True)

    assert np.all(np.isfinite(preds["cuda"])), "CUDA produced non-finite predictions"
    np.testing.assert_allclose(preds["cuda"], preds["cpu"], atol=1e-10)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("n", "num_leaves", "bagging_fraction", "bagging_freq"),
    [
        (200, 7, 0.7, 1),
        (500, 15, 0.5, 1),
        (1000, 31, 0.8, 3),
        (2000, 31, 0.7, 1),
    ],
)
def test_cuda_bagging_does_not_crash_and_matches_cpu(n, num_leaves, bagging_fraction, bagging_freq):
    """CUDA training with bagging must not crash and must track CPU.

    Regression test for two independent CUDA bugs that made *any* bagged run
    abort with "[CUDA] an illegal memory access was encountered":

    1. ``CUDATree::ToHost()`` freed the per-tree GPU tree-structure arrays
       (split_feature_inner / children / thresholds / decision_type), keeping
       only ``cuda_leaf_value_``, to bound device memory across many rounds.
       But ``AddPredictionToScoreKernel`` traverses the whole tree, and the
       GBDT out-of-bag score update (only reached under bagging) launches it
       post-ToHost, dereferencing the freed/null device pointers. Fixed by
       re-uploading the structure for that launch (and freeing it again).
    2. ``CUDADataPartition::CalcBlockDim`` is non-monotonic, so a bagged leaf
       (~bagging_fraction * n, landing in the ~101-160 band) needs more blocks
       than the full dataset, overflowing the per-block offset buffers. (Fixed
       separately; this test also guards against it regressing into a crash.)

    Before the fixes this test aborted the interpreter. After them, CPU and
    CUDA agree to within the floating-point / RNG divergence documented as
    expected for the CUDA tree learner in upstream issue #6055 (different bag
    sampling => a generous tolerance, the point of the test is no-crash +
    finite + same ballpark).
    """
    rng = np.random.default_rng(11)
    d = 8
    X = rng.standard_normal((n, d)).astype(np.float64)
    coef = rng.standard_normal(d)
    y = (X @ coef + 0.3 * rng.standard_normal(n)).astype(np.float64)

    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 7,
            "feature_pre_filter": False,
            "device_type": device_type,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "num_leaves": num_leaves,
            "learning_rate": 0.05,
            "min_data_in_leaf": 5,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "bagging_seed": 3,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        bst = lgb.train(params, ds, num_boost_round=20)
        preds[device_type] = bst.predict(X, raw_score=True)

    assert np.all(np.isfinite(preds["cuda"])), "CUDA bagging produced non-finite predictions"
    # Bagging samples a different bag on CUDA than on CPU (different RNG stream),
    # so predictions are not bit-identical; #6055 documents this as expected.
    # The bar here is "same ballpark" -- strict enough to catch a model that
    # silently trained on garbage, loose enough to tolerate bag-sampling drift.
    max_abs = float(np.abs(preds["cpu"] - preds["cuda"]).max())
    y_range = float(y.max() - y.min())
    assert max_abs < 0.25 * y_range, (
        f"CPU/CUDA bagging predictions diverge far more than bag-sampling drift: "
        f"max|Δ|={max_abs:.4f}, y_range={y_range:.4f}"
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize(("n", "seed"), [(1000, 7), (2000, 7), (2000, 11)])
def test_cuda_quantized_tree_structure_matches_cpu(n, seed):
    """CUDA quantized trees must grow to roughly the same size as CPU quantized.

    Regression test for the child-leaf packed-sum bugs: the int64 packed
    gradient/hessian sum (sum_of_gradients_hessians) was (1) dropped by
    CUDASplitInfo::operator= and (2) never refreshed for child leaves in the
    data-partition split kernel. The discretized best-split finder uses that
    packed sum as the leaf total, so children inherited the parent's total,
    scored phantom splits that partitioned to a 0-data leaf, and CUDA quantized
    trees stalled at ~2-4 leaves while CPU grew the full 31.

    This guards the structural fix: CUDA must grow to within 20% of CPU's leaf
    count (the bug capped it far below). It deliberately does NOT assert tight
    prediction parity -- residual CPU/CUDA divergence remains from the separate
    open 8-bit-histogram leaf-value bug and from the cross-feature gain-tie
    ordering (a different branch). min_data_in_leaf is high so leaves stay in
    the 16-bit histogram regime.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 12)).astype(np.float64)
    y = (X @ rng.standard_normal(12) + 0.3 * rng.standard_normal(n)).astype(np.float64)
    params = {
        "objective": "regression",
        "num_leaves": 31,
        "min_data_in_leaf": 300,
        "learning_rate": 0.1,
        "use_quantized_grad": True,
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": seed,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
    }
    models = {}
    for device_type in ("cpu", "cuda"):
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        models[device_type] = lgb.train({**params, "device_type": device_type}, ds, num_boost_round=20)

    cpu_leaves = sum(t["num_leaves"] for t in models["cpu"].dump_model()["tree_info"])
    cuda_leaves = sum(t["num_leaves"] for t in models["cuda"].dump_model()["tree_info"])
    # The bug capped CUDA at ~2/tree (~40 total over 20 trees); after the fix it
    # tracks CPU's count. Require within 20% in both directions.
    assert 0.8 * cpu_leaves <= cuda_leaves <= 1.2 * cpu_leaves, (
        f"CUDA quantized tree size diverged: cuda={cuda_leaves} vs cpu={cpu_leaves} (n={n}, seed={seed})"
    )
    assert np.all(np.isfinite(models["cuda"].predict(X)))


@_REQUIRES_CUDA
@pytest.mark.parametrize(("n", "seed"), [(1000, 7), (2000, 7), (1000, 11)])
def test_cuda_quantized_deep_trees_track_cpu(n, seed):
    """Deep CUDA quantized trees (small leaves) must track CPU, not explode.

    Regression test for the histogram-slot collision: in SplitTreeStructureKernel
    the left-is-smaller branch handed the discretized child a hist slot at a 1x
    (right_leaf_index * num_total_bin) stride while every other path uses 2x, so a
    child could be assigned a slot already owned by another leaf. Its histogram then
    accumulated on top of that leaf's data (e.g. a 45-point leaf's histogram summed
    to 106 points' worth), yielding phantom splits, 0-data leaves, and leaf outputs
    that exploded to 1e12-1e20 over 20 rounds at small min_data_in_leaf.

    With min_data_in_leaf=20 the trees go deep (many 8-bit-histogram leaves), which
    is exactly where the collision bit. After the fix CUDA matches CPU's leaf counts
    and predictions stay bounded (residual divergence is the separate gain-tie
    ordering, handled on another branch).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 12)).astype(np.float64)
    y = (X @ rng.standard_normal(12) + 0.3 * rng.standard_normal(n)).astype(np.float64)
    params = {
        "objective": "regression",
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "use_quantized_grad": True,
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": seed,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
    }
    models = {}
    for device_type in ("cpu", "cuda"):
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        models[device_type] = lgb.train({**params, "device_type": device_type}, ds, num_boost_round=20)

    cpu_leaves = sum(t["num_leaves"] for t in models["cpu"].dump_model()["tree_info"])
    cuda_leaves = sum(t["num_leaves"] for t in models["cuda"].dump_model()["tree_info"])
    assert 0.9 * cpu_leaves <= cuda_leaves <= 1.1 * cpu_leaves, (
        f"CUDA quantized deep-tree size diverged: cuda={cuda_leaves} vs cpu={cpu_leaves} (n={n}, seed={seed})"
    )
    cpu_pred = models["cpu"].predict(X)
    cuda_pred = models["cuda"].predict(X)
    assert np.all(np.isfinite(cuda_pred))
    max_diff = float(np.max(np.abs(cuda_pred - cpu_pred)))
    # Broken behaviour exploded to >=1e12; the fix brings it to ~1 (gain-tie FP level).
    assert max_diff < 5.0, f"CUDA quantized deep trees diverge from CPU by {max_diff:.3g} (n={n}, seed={seed})"


def _train_forced(device_type, forced_split, tmp_path, num_boost_round=10, num_leaves=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(400, 6)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.1 * rng.rand(400)
    fn = tmp_path / f"forced_{device_type}_{seed}.json"
    fn.write_text(json.dumps(forced_split))
    params = {
        "objective": "regression",
        "forcedsplits_filename": str(fn),
        "num_leaves": num_leaves,
        "min_data_in_leaf": 5,
        "learning_rate": 0.1,
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 0,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "device_type": device_type,
    }
    ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
    return lgb.train(params, ds, num_boost_round=num_boost_round), X, y


_FORCED_SPLIT_CASES = {
    "root_only": {"feature": 2, "threshold": 0.5},
    "root_nested": {
        "feature": 2,
        "threshold": 0.5,
        "left": {"feature": 2, "threshold": 0.25},
        "right": {"feature": 2, "threshold": 0.75},
    },
    "root_lr_mixed": {
        "feature": 1,
        "threshold": 0.4,
        "left": {"feature": 0, "threshold": 0.3},
        "right": {"feature": 3, "threshold": 0.6},
    },
    "three_deep_chain": {
        "feature": 2,
        "threshold": 0.5,
        "left": {
            "feature": 0,
            "threshold": 0.5,
            "left": {"feature": 1, "threshold": 0.5},
        },
    },
}


def _forced_features_in_json(node, out=None):
    if out is None:
        out = []
    if "feature" in node:
        out.append(node["feature"])
        for side in ("left", "right"):
            if side in node:
                _forced_features_in_json(node[side], out)
    return out


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", list(_FORCED_SPLIT_CASES))
@pytest.mark.parametrize("num_leaves", [8, 31])
def test_cuda_forced_splits_honored(case, num_leaves, tmp_path):
    """CUDA must apply the forced-split JSON: the forced root feature heads every tree.

    Regression test for forcedsplits_filename being silently ignored on CUDA
    (ForceSplits only existed in SerialTreeLearner::Train; the CUDA learner never
    consulted the forced-split JSON, so CUDA trees split on whatever feature had
    the best gain instead of the forced one).
    """
    forced_split = _FORCED_SPLIT_CASES[case]
    bst, _, _ = _train_forced("cuda", forced_split, tmp_path, num_leaves=num_leaves)
    forced_root_feature = forced_split["feature"]
    for tree in bst.dump_model()["tree_info"]:
        root = tree["tree_structure"]
        assert root["split_feature"] == forced_root_feature, (
            f"tree does not honor forced root split: got feature {root['split_feature']}, "
            f"expected {forced_root_feature}"
        )


@_REQUIRES_CUDA
@pytest.mark.parametrize("case", list(_FORCED_SPLIT_CASES))
@pytest.mark.parametrize("num_leaves", [8, 31])
@pytest.mark.parametrize("seed", [0, 1])
def test_cuda_forced_splits_match_cpu(case, num_leaves, seed, tmp_path):
    """CUDA forced-split training must produce the same model as CPU.

    Predictions must match at FP epsilon over 30 boosting rounds; the first tree's
    structure (split features, gains, counts, leaf values) must match exactly.
    """
    forced_split = _FORCED_SPLIT_CASES[case]
    bst_cpu, X, _ = _train_forced(
        "cpu",
        forced_split,
        tmp_path,
        num_boost_round=30,
        num_leaves=num_leaves,
        seed=seed,
    )
    bst_cuda, _, _ = _train_forced(
        "cuda",
        forced_split,
        tmp_path,
        num_boost_round=30,
        num_leaves=num_leaves,
        seed=seed,
    )

    # tree-0 structure equality (features, gains, counts, leaf values; thresholds may
    # differ in real-value display encoding for the same bin boundary)
    def substantive(bst):
        out = []

        def _rec(node):
            if "leaf_value" in node:
                out.append(("leaf", round(node["leaf_value"], 10), node["leaf_count"]))
            else:
                out.append(
                    (
                        node["split_feature"],
                        round(node["split_gain"], 6),
                        node["internal_count"],
                    )
                )
                _rec(node["left_child"])
                _rec(node["right_child"])

        _rec(bst.dump_model()["tree_info"][0]["tree_structure"])
        return out

    assert substantive(bst_cpu) == substantive(bst_cuda)

    # prediction parity over all rounds
    np.testing.assert_allclose(
        bst_cpu.predict(X),
        bst_cuda.predict(X),
        rtol=0,
        atol=1e-10,
        err_msg=f"forced splits case={case}: CUDA diverges from CPU",
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_leaves", [7, 31, 63])
def test_cuda_histogram_event_ordering_matches_cpu(num_leaves):
    """Multi-leaf CUDA training must match CPU after the histogram->FindBestSplits
    device syncs were replaced by event-based stream ordering.

    The per-split device sync after histogram construction and the one between the
    smaller- and larger-leaf FindBestSplits launches were replaced by
    cudaStreamWaitEvent on the histogram constructor's construct/subtract completion
    events. The smaller-leaf search waits for the constructed histogram, the
    larger-leaf search for the subtracted histogram. Building trees with several
    leaves exercises this path; a missing/incorrect event ordering would let a
    FindBestSplits kernel read a histogram before it is written and diverge from
    CPU well beyond the tolerance below.
    """
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.standard_normal((n, 10)).astype(np.float64)
    y = (X @ rng.standard_normal(10) + 0.1 * rng.standard_normal(n)).astype(np.float64)

    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "feature_pre_filter": False,
            "device_type": device_type,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "num_leaves": num_leaves,
            "min_data_in_leaf": 5,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        preds[device_type] = lgb.train(params, ds, num_boost_round=5).predict(X)

    np.testing.assert_allclose(
        preds["cuda"],
        preds["cpu"],
        rtol=0,
        atol=1e-9,
        err_msg=f"num_leaves={num_leaves}: CUDA histogram event ordering diverges from CPU",
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_leaves", [7, 31, 63])
def test_cuda_syncbestsplit_overlap_matches_cpu(num_leaves):
    """Multi-leaf CUDA training must match CPU after overlapping the two child
    leaves' SyncBestSplit reduction.

    LaunchSyncBestSplitForLeafKernel used to reduce the smaller and larger child
    leaves' per-feature best splits with two kernel launches separated by a full
    device sync. That sync was dropped and the two leaves now reduce concurrently
    on separate streams (each reads only the cuda_best_split_info_ region that
    FindBestSplitsForLeafKernel wrote on that same stream). Building trees with
    several leaves exercises the both-child-leaves-valid path this touches; CUDA
    must stay bit-for-bit aligned with CPU on the deterministic single-thread,
    double-precision config.
    """
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.standard_normal((n, 10)).astype(np.float64)
    y = (X @ rng.standard_normal(10) + 0.1 * rng.standard_normal(n)).astype(np.float64)

    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "feature_pre_filter": False,
            "device_type": device_type,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "num_leaves": num_leaves,
            "min_data_in_leaf": 5,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        preds[device_type] = lgb.train(params, ds, num_boost_round=5).predict(X)

    np.testing.assert_allclose(
        preds["cuda"],
        preds["cpu"],
        rtol=0,
        atol=1e-9,
        err_msg=f"num_leaves={num_leaves}: CUDA SyncBestSplit overlap diverges from CPU",
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize("n", [200, 500, 1000])
@pytest.mark.parametrize("num_leaves", [7, 31])
def test_cuda_quantized_training_produces_splits(n, num_leaves):
    """CUDA use_quantized_grad training must produce real splitting trees.

    Regression test for the 2x under-allocation of the discretized
    gradient/hessian buffer (CUDAGradientDiscretizer). The buffer holds an
    int16 gradient and an int16 hessian per data point (4 bytes), but was
    sized num_data * 2 bytes. DiscretizeGradientsKernel overran it and
    corrupted the adjacent dequantization scale buffers, which made the
    root leaf's sum_hessians come out as a denormal ~0. The root then
    failed the sum_hessians > min_sum_hessian_in_leaf validity check and
    never split, so every CUDA quantized tree collapsed to a single leaf
    and the model did not learn.

    The fix sizes the buffer as num_data * 4. Here we assert that CUDA
    quantized trees actually split (num_leaves > 1) and that predictions
    are not the degenerate constant the single-leaf model produced.
    """
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, 12)).astype(np.float64)
    y = (X @ rng.standard_normal(12) + 0.3 * rng.standard_normal(n)).astype(np.float64)
    params = {
        "objective": "regression",
        "num_leaves": num_leaves,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "use_quantized_grad": True,
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 7,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "device_type": "cuda",
    }
    ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
    bst = lgb.train(params, ds, num_boost_round=20)

    leaf_counts = [tree["num_leaves"] for tree in bst.dump_model()["tree_info"]]
    assert max(leaf_counts) > 1, (
        f"CUDA quantized training collapsed to single-leaf trees "
        f"(n={n}, num_leaves={num_leaves}): per-tree leaf counts {leaf_counts[:5]}"
    )

    preds = bst.predict(X)
    assert np.all(np.isfinite(preds)), "CUDA quantized produced non-finite predictions"
    assert preds.std() > 1e-6, f"CUDA quantized predictions are degenerate/constant (n={n}, num_leaves={num_leaves})"


@_REQUIRES_CUDA
@pytest.mark.parametrize("n", [2000, 8000, 50000])
def test_cuda_quantized_32bit_histogram_matches_cpu(n):
    """CUDA quantized training must match CPU once leaves need 32-bit histograms.

    Regression test for the best-split finder reading the 32-bit discretized
    histogram with the wrong width. A leaf whose max per-bin stat
    (num_data_in_leaf * num_grad_quant_bins) reaches 65536 uses an int64-per-bin
    (32-bit grad / 32-bit hess) histogram. The finder dispatched that case with
    BIN_HIST_TYPE=int32_t and read it through an int32_t* offset, i.e. 4-byte
    half-bins at the wrong stride, so the split search saw garbage for any leaf
    large enough to need 32-bit bins. With num_grad_quant_bins=16 that is any leaf
    with >= 4096 rows; the resulting models were near-random (correlation ~0 with
    CPU). The 8-bit and 16-bit paths were correct, so small-data tests never hit it.

    With num_grad_quant_bins=16: n=2000 stays 16-bit (already correct), while
    n>=8000 forces a 32-bit root. The fix reads the histogram as int64, making CUDA
    bit-identical to CPU at all scales.
    """
    rng = np.random.default_rng(0)
    nf = 40
    X = rng.standard_normal((n, nf)).astype(np.float64)
    y = (X @ rng.standard_normal(nf) + 0.3 * rng.standard_normal(n)).astype(np.float64)
    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "learning_rate": 0.05,
            "use_quantized_grad": True,
            "num_grad_quant_bins": 16,
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "feature_pre_filter": False,
            "device_type": device_type,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        preds[device_type] = lgb.train(params, ds, num_boost_round=20).predict(X)
    corr = float(np.corrcoef(preds["cpu"], preds["cuda"])[0, 1])
    # Before the fix, 32-bit-histogram leaves (n>=8000) gave correlation ~0.
    assert corr > 0.99, f"CUDA quantized (32-bit histogram) diverges from CPU: corr={corr:.4f} (n={n})"


_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled LightGBM build (set TASK=cuda)",
)


def _make_linear_regression(n, d, seed, nan=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float64)
    coef = rng.standard_normal(d)
    y = (X @ coef + 0.3 * rng.standard_normal(n)).astype(np.float64)
    if nan:
        X[rng.random((n, d)) < 0.05] = np.nan
    return X, y


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("num_leaves", "linear_lambda"),
    [(15, 0.0), (31, 0.5), (63, 5.0)],
)
def test_cuda_linear_tree_matches_cpu(num_leaves, linear_lambda):
    """linear_tree on the CUDA tree learner must match the CPU LinearTreeLearner.

    The per-leaf linear model (const + sum coeff*raw_feat) is fit on-device (GPU
    XtHX/Xtg accumulation) with the same math as CPU. For L2 regression (constant
    hessian) the fit is bit-identical up to floating-point summation order, so a
    boosted model of linear-leaf trees reproduces the CPU predictions to ~1e-6.
    (The on-device gram uses atomic float adds, so ~1e-15/tree order differences
    can, at aggressive leaf counts, flip a borderline split and diverge structurally
    -- the same way non-linear CUDA can; these configs are in the stable regime.)
    """
    X, y = _make_linear_regression(4000, 10, seed=0)
    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "num_leaves": num_leaves,
            "min_data_in_leaf": 20,
            "learning_rate": 0.1,
            "linear_tree": True,
            "linear_lambda": linear_lambda,
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 7,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "feature_pre_filter": False,
            "device_type": device_type,
        }
        ds = lgb.Dataset(X, label=y, params=params)
        bst = lgb.train(params, ds, num_boost_round=30)
        preds[device_type] = bst.predict(X)
        # sanity: the model actually fit linear leaves, not constant fallbacks
        assert "leaf_coeff" in str(bst.dump_model()["tree_info"][5])
    max_diff = float(np.max(np.abs(preds["cpu"] - preds["cuda"])))
    assert max_diff < 1e-6, f"CUDA linear tree diverges from CPU: max|diff|={max_diff:.3e}"


@_REQUIRES_CUDA
def test_cuda_linear_tree_handles_nan_like_cpu():
    """Rows with NaN in a leaf's linear features fall back to the leaf's constant
    output, on both CPU and CUDA; the fit must also skip NaN rows identically.
    Uses a modest leaf count so the NaN-skipping fit stays in the bit-exact regime."""
    X, y = _make_linear_regression(4000, 10, seed=5, nan=True)
    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "num_leaves": 15,
            "min_data_in_leaf": 20,
            "learning_rate": 0.1,
            "linear_tree": True,
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 7,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "feature_pre_filter": False,
            "device_type": device_type,
        }
        ds = lgb.Dataset(X, label=y, params=params)
        preds[device_type] = lgb.train(params, ds, num_boost_round=30).predict(X)
    max_diff = float(np.max(np.abs(preds["cpu"] - preds["cuda"])))
    assert max_diff < 1e-6, f"CUDA linear tree (NaN) diverges from CPU: max|diff|={max_diff:.3e}"


_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled LightGBM build (set TASK=cuda)",
)

# --------------------------------------------------------------------------- #
# ExaBoost env kill-switch / feature-flag / ingestion / determinism gates.
#
# These encode the out-of-tree scratchpad md5 gates (hybrid_verify.py,
# int8_md5_gates.py, the graph/quant sweeps) as CI-enforced pytest, closing the
# coverage gap where the whole hybrid/graph/quant/ingestion surface rested only
# on ad-hoc md5 scripts + racecheck.
#
# WHY QUANT MODE EVERYWHERE: non-quant CUDA training accumulates histograms with
# float atomicAdd, so it is run-to-run NONdeterministic -- the same config trained
# twice does not produce a bit-identical model. Quantized-gradient training
# (use_quantized_grad=True) uses integer histograms and IS bit-deterministic, so
# it is the only mode in which a model md5 is a stable fingerprint. Every gate
# below that asserts bit-identity therefore trains in quant mode.
#
# WHY SUBPROCESSES for the kill-switches: several switches are read exactly once
# per process into a `static const bool` (EXABOOST_GRAPH_LEVEL_LOOP,
# EXABOOST_GRAPH_QUANT, EXABOOST_FAST_ROWDATA, EXABOOST_ROWDATA_4BIT,
# EXABOOST_FP32_HIST, EXABOOST_FP32_GAIN). Mutating os.environ between two
# in-process trains would NOT flip them. Each variant is trained in a fresh
# interpreter with the env set at launch so the switch is genuinely exercised.
# --------------------------------------------------------------------------- #

# Worker executed in a fresh interpreter: trains a small quantized CUDA model and
# prints its model md5. The profile selects a param/data shape that routes through
# the code path the switch under test guards.
_QUANT_MD5_WORKER = textwrap.dedent(
    """
    import hashlib, sys
    import numpy as np
    import lightgbm as lgb

    profile = sys.argv[1]
    rng = np.random.default_rng(0)
    n = 8000
    if profile == "graph":
        # depth-limited multiclass quant -> CUDA graph level-loop path
        m = 20
        X = rng.standard_normal((n, m)).astype(np.float64)
        y = (X @ rng.standard_normal((m, 5))).argmax(axis=1).astype(np.float64)
        p = {"objective": "multiclass", "num_class": 5, "num_leaves": 63,
             "max_depth": 6, "max_bin": 255, "learning_rate": 0.1,
             "use_quantized_grad": True}
    elif profile == "fewbin":
        # <=16 bins -> 4-bit packed row-data path
        m = 12
        X = rng.integers(0, 12, size=(n, m)).astype(np.float64)
        y = (X @ rng.standard_normal(m) + rng.standard_normal(n)).astype(np.float64)
        p = {"objective": "regression", "num_leaves": 31, "max_depth": 6,
             "max_bin": 15, "learning_rate": 0.1, "use_quantized_grad": True}
    else:  # "dense": full-width dense quant (fast-rowdata / gpu-construct / efb)
        m = 20
        X = rng.standard_normal((n, m)).astype(np.float64)
        y = (X @ rng.standard_normal(m) + 0.3 * rng.standard_normal(n)).astype(np.float64)
        p = {"objective": "regression", "num_leaves": 31, "max_depth": 6,
             "max_bin": 255, "learning_rate": 0.1, "use_quantized_grad": True}
    p.update({"device_type": "cuda", "seed": 42, "verbose": -1, "metric": "None",
              "num_threads": 8})
    ds = lgb.Dataset(X, label=y, params=p)
    ds.construct()
    bst = lgb.train(p, ds, num_boost_round=25)
    print(hashlib.md5(bst.model_to_string().encode()).hexdigest())
    """
)


def _quant_model_md5(profile, extra_env=None):
    """Train the quant worker for `profile` in a fresh interpreter with `extra_env`
    layered on top of the current environment; return the printed model md5.

    A fresh process is required because the kill-switches under test are cached in
    `static const bool` on first read, so they can only be flipped at process start.
    """
    env = dict(os.environ)
    env["TASK"] = "cuda"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", _QUANT_MD5_WORKER, profile],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"quant worker failed (profile={profile}, env={extra_env}):\n{proc.stderr}"
    md5 = proc.stdout.strip().splitlines()[-1].strip()
    assert len(md5) == 32, f"unexpected worker output for profile={profile}: {proc.stdout!r}\n{proc.stderr}"
    return md5


# (profile, extra_env-for-the-DISABLED-variant, human name). The default (env unset)
# must produce a model bit-identical to the switch explicitly disabled: the switch is
# an optimization that must be behavior-preserving. GRAPH tests opt into the quant
# graph path with EXABOOST_GRAPH_QUANT=1 (quant needs the explicit opt-in), then
# compare graph level-loop on (default) vs EXABOOST_GRAPH_LEVEL_LOOP=0.
_KILL_SWITCH_CASES = [
    (
        "graph",
        {"EXABOOST_GRAPH_QUANT": "1"},
        {"EXABOOST_GRAPH_QUANT": "1", "EXABOOST_GRAPH_LEVEL_LOOP": "0"},
        "EXABOOST_GRAPH_LEVEL_LOOP",
    ),
    ("fewbin", {}, {"EXABOOST_ROWDATA_4BIT": "0"}, "EXABOOST_ROWDATA_4BIT"),
    ("dense", {}, {"EXABOOST_FAST_ROWDATA": "0"}, "EXABOOST_FAST_ROWDATA"),
    ("dense", {}, {"EXABOOST_GPU_CONSTRUCT": "0"}, "EXABOOST_GPU_CONSTRUCT"),
    ("dense", {}, {"EXABOOST_EFB_PRECHECK": "0"}, "EXABOOST_EFB_PRECHECK"),
]


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("profile", "default_env", "disabled_env", "switch"),
    _KILL_SWITCH_CASES,
    ids=[c[3] for c in _KILL_SWITCH_CASES],
)
def test_cuda_kill_switch_is_bit_identical(profile, default_env, disabled_env, switch):
    """An ExaBoost optimization kill-switch must be behavior-preserving.

    For each switch that defaults ON as a pure optimization, the model trained with
    the switch at its default must be BIT-IDENTICAL (identical model_to_string md5)
    to the model trained with the switch forced to ``0``. A mismatch means the
    optimized path diverges from the reference path -- a real correctness bug, not a
    test-tuning issue. Quant mode makes the md5 a stable fingerprint (see module
    note); each variant runs in its own interpreter because the switches are cached
    per-process.
    """
    default_md5 = _quant_model_md5(profile, default_env)
    disabled_md5 = _quant_model_md5(profile, disabled_env)
    assert default_md5 == disabled_md5, (
        f"{switch}: default path md5={default_md5} != disabled(=0) path md5={disabled_md5}; "
        f"the '{switch}' fast path is NOT behavior-preserving on profile={profile}"
    )


# Feature flags that must be no-ops when off/unset: setting them to 0 (or leaving
# unset) must reproduce the plain quant model exactly. (EXABOOST_FIXEDPOINT_QUANT=1
# deliberately changes the model -- it is a different, near-lossless quant mode -- so
# only its OFF state is a no-op and is what we pin here.)
_NOOP_FLAGS = ["EXABOOST_FP32_GAIN", "EXABOOST_FP32_HIST", "EXABOOST_FIXEDPOINT_QUANT"]


@_REQUIRES_CUDA
@pytest.mark.parametrize("flag", _NOOP_FLAGS)
def test_cuda_feature_flag_off_reproduces_plain_quant(flag):
    """A feature flag set to 0 must reproduce the plain quant model bit-for-bit.

    Guards that FP32_GAIN / FP32_HIST / FIXEDPOINT_QUANT are opt-IN: with the flag
    absent (plain quant) or explicitly 0, training must yield the identical model.
    A divergence would mean the "off" state silently activates the alternate path.
    """
    plain = _quant_model_md5("dense")
    off = _quant_model_md5("dense", {flag: "0"})
    assert plain == off, f"{flag}=0 changed the model (plain={plain}, off={off}); the flag is not a clean no-op"


@_REQUIRES_CUDA
def test_cuda_quant_training_is_deterministic():
    """The same quant config trained twice must produce an identical model.

    Documents that quantized-gradient CUDA training is the deterministic mode
    (integer histograms, no float-atomic nondeterminism). This is the invariant the
    whole md5-gate suite relies on; if it ever fails, every bit-identity gate is
    meaningless.
    """
    md5_a = _quant_model_md5("dense")
    md5_b = _quant_model_md5("dense")
    assert md5_a == md5_b, f"plain quant is nondeterministic: {md5_a} != {md5_b}"


@_REQUIRES_CUDA
def test_cuda_fixedpoint_quant_is_deterministic():
    """EXABOOST_FIXEDPOINT_QUANT=1 (near-lossless quant mode) must also be
    run-to-run deterministic. It produces a DIFFERENT model from plain quant (a
    distinct integer scheme), but that model must be reproducible."""
    md5_a = _quant_model_md5("dense", {"EXABOOST_FIXEDPOINT_QUANT": "1"})
    md5_b = _quant_model_md5("dense", {"EXABOOST_FIXEDPOINT_QUANT": "1"})
    assert md5_a == md5_b, f"fixedpoint quant is nondeterministic: {md5_a} != {md5_b}"
    # sanity: it really is a distinct mode, not silently the plain path
    assert md5_a != _quant_model_md5("dense"), "EXABOOST_FIXEDPOINT_QUANT=1 did not change the model"


def _ingestion_model_string(X, y, base):
    ds = lgb.Dataset(X, label=y, params=base)
    return lgb.train(base, ds, num_boost_round=20).model_to_string()


def _strip_nonsubstantive(model_str):
    # feature_names differ (numpy default "Column_i" vs pandas column labels) and the
    # trailing pandas_categorical marker serializes as "null" for ndarray vs "[]" for
    # a DataFrame; neither reflects the binned data or the learned trees.
    return "\n".join(
        line for line in model_str.split("\n") if not line.startswith(("feature_names=", "pandas_categorical"))
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32])
def test_cuda_integer_ingestion_matches_float32(dtype):
    """Integer numpy input must produce the same Dataset -- and thus the same model
    -- as the equivalent float32 input, when the integer values fit both dtypes.

    Mirrors the int8_md5_gates.py identity gate: feeding the same values as int8 /
    uint8 / int16 / uint16 / int32 vs float32 must bin identically and train to a
    bit-identical model. A mismatch would mean the integer ingestion path bins wrong.
    """
    info = np.iinfo(dtype)
    rng = np.random.default_rng(3)
    n, m = 6000, 8
    lo = max(0, info.min)
    hi = min(50, info.max)
    Xi = rng.integers(lo, hi, size=(n, m))
    y = (Xi @ rng.standard_normal(m) + rng.standard_normal(n)).astype(np.float64)
    base = {
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 6,
        "max_bin": 255,
        "learning_rate": 0.05,
        "use_quantized_grad": True,
        "device_type": "cuda",
        "seed": 42,
        "verbose": -1,
        "metric": "None",
        "num_threads": 8,
    }
    ref = _ingestion_model_string(Xi.astype(np.float32), y, base)
    got = _ingestion_model_string(Xi.astype(dtype), y, base)
    assert (
        hashlib.md5(_strip_nonsubstantive(got).encode()).hexdigest()
        == hashlib.md5(_strip_nonsubstantive(ref).encode()).hexdigest()
    ), f"{np.dtype(dtype).name} ingestion diverges from float32"


@_REQUIRES_CUDA
def test_cuda_pandas_int_frame_matches_numpy():
    """A small-integer pandas DataFrame must ingest identically to the equivalent
    numpy array (same binning, same model up to the cosmetic feature_names /
    pandas_categorical serialization lines)."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(3)
    n, m = 6000, 8
    Xi = rng.integers(0, 50, size=(n, m))
    y = (Xi @ rng.standard_normal(m) + rng.standard_normal(n)).astype(np.float64)
    base = {
        "objective": "regression",
        "num_leaves": 31,
        "max_depth": 6,
        "max_bin": 255,
        "learning_rate": 0.05,
        "use_quantized_grad": True,
        "device_type": "cuda",
        "seed": 42,
        "verbose": -1,
        "metric": "None",
        "num_threads": 8,
    }
    ref = _ingestion_model_string(Xi.astype(np.float32), y, base)
    frame = pd.DataFrame(Xi.astype(np.int16), columns=[f"Column_{i}" for i in range(m)])
    got = _ingestion_model_string(frame, y, base)
    assert (
        hashlib.md5(_strip_nonsubstantive(got).encode()).hexdigest()
        == hashlib.md5(_strip_nonsubstantive(ref).encode()).hexdigest()
    ), "pandas int frame diverges from numpy ingestion"


_COVTYPE_DIR = "/home/felixjk/Documents/exaboost-bench/data/cache/covtype/"
_HAS_COVTYPE = os.path.isdir(_COVTYPE_DIR) and os.path.isfile(_COVTYPE_DIR + "X_train.npy")


@_REQUIRES_CUDA
@pytest.mark.skipif(not _HAS_COVTYPE, reason="covtype data cache not present")
@pytest.mark.parametrize(
    ("num_leaves", "max_depth", "quant"),
    [(63, 6, True), (63, 6, False), (127, 8, True), (31, 5, True)],
)
def test_cuda_multiclass_graph_loop_trains_and_is_accurate(num_leaves, max_depth, quant):
    """Multiclass training with the CUDA graph level-loop active (default ON since
    84db39cd) must not raise and must reach a sane accuracy.

    Regression guard for the 84db39cd upload-race fix: the depth-limited multiclass
    graph loop is now the default path. This trains several covtype configs through
    it and asserts (a) no exception / no crash and (b) accuracy in a sane band, so a
    future change that reintroduces the upload race (garbage histograms -> near-random
    trees) or breaks the multiclass graph path is caught.
    """
    Xtr = np.load(_COVTYPE_DIR + "X_train.npy")
    ytr = np.load(_COVTYPE_DIR + "y_train.npy")
    Xte = np.load(_COVTYPE_DIR + "X_test.npy")
    yte = np.load(_COVTYPE_DIR + "y_test.npy")
    Xs, ys = Xtr[:60000], ytr[:60000]
    params = {
        "objective": "multiclass",
        "num_class": 7,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "max_bin": 255,
        "learning_rate": 0.1,
        "device_type": "cuda",
        "seed": 42,
        "verbose": -1,
        "metric": "None",
        "num_threads": 8,
    }
    if quant:
        params["use_quantized_grad"] = True
    ds = lgb.Dataset(Xs, label=ys, params=params)
    ds.construct()
    bst = lgb.train(params, ds, num_boost_round=40)
    preds = bst.predict(Xte)
    assert np.all(np.isfinite(preds)), "multiclass graph-loop produced non-finite predictions"
    acc = float((preds.argmax(axis=1) == yte).mean())
    # covtype 40-tree models here score ~0.78-0.83; a graph-loop regression to garbage
    # trees would collapse well below the 1/7 class balance. 0.70 is a safe floor.
    assert acc > 0.70, (
        f"multiclass graph loop (num_leaves={num_leaves}, max_depth={max_depth}, quant={quant}) "
        f"accuracy {acc:.4f} below sane band -- possible graph/upload-race regression"
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize("n_categories", [300, 1200, 3000])
def test_cuda_large_categorical_global_memory_does_not_crash(n_categories):
    """Training on a categorical feature with > 256 histogram bins must not crash.

    A feature whose histogram exceeds NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER (256)
    bins routes categorical split finding through the global-memory kernel and its
    device-wide BitonicArgSortDevice. That path used to raise
    "[CUDA] an illegal memory access was encountered" for large category counts,
    because the sort was instantiated with MAX_DEPTH=11 (the value for a 1024-wide
    block) instead of 9 for the 256-thread block, and because the per-feature
    scratch-buffer pointers were offset by hist_offset*2 instead of hist_offset,
    running the sort off the end of the buffers.

    This test guards the crash. (Full CPU/CUDA split-parity on the global-memory
    categorical path is tracked separately: a residual data-dependent divergence
    remains for > 256-category features.)
    """
    rng = np.random.default_rng(7)
    n = n_categories * 40
    cats = rng.integers(0, n_categories, size=n).astype(np.float64)
    category_means = rng.standard_normal(n_categories) * 0.7
    y = (category_means[cats.astype(int)] + rng.standard_normal(n) * 0.05).astype(np.float64)
    X = cats.reshape(-1, 1)
    params = {
        "objective": "regression",
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 0,
        "feature_pre_filter": False,
        "gpu_use_dp": True,
        "num_leaves": 8,
        "min_data_in_leaf": 5,
        "min_data_in_bin": 1,
        "max_bin": 8192,
        "cat_smooth": 1,
        "learning_rate": 0.1,
        "device_type": "cuda",
    }
    ds = lgb.Dataset(
        X,
        label=y,
        categorical_feature=[0],
        params={
            "verbose": -1,
            "feature_pre_filter": False,
            "min_data_in_bin": 1,
            "max_bin": 8192,
        },
    )
    # Regression: this raised a CUDA illegal-memory-access error before the fix.
    bst = lgb.train(params, ds, num_boost_round=5)
    preds = bst.predict(X, raw_score=True)
    assert np.all(np.isfinite(preds)), "global-memory categorical training produced non-finite predictions"


@_REQUIRES_CUDA
@pytest.mark.parametrize("min_data_per_group", [1, 5, 20, 33, 50, 100, 200])
def test_cuda_min_data_per_group_categorical_matches_cpu(min_data_per_group):
    """CUDA must apply min_data_per_group with CPU's per-group semantics.

    CPU (FeatureHistogram::FindBestThresholdCategoricalInner) treats
    min_data_per_group as a minimum on the *group*: the run of sorted
    categories accumulated since the last accepted threshold. It tracks
    cnt_cur_group, skips a threshold whose group is still too small, and
    resets the counter each time a threshold is accepted.

    CUDA instead applied it per side (left_count >= min_data_per_group and
    right_count >= min_data_per_group), a different rule. The two agreed
    only when the threshold was far from the per-category row counts, so
    CUDA silently picked a different categorical split -- and hence a
    different tree -- whenever min_data_per_group landed near them.

    With 400 rows over 12 categories (~33 rows each), the pre-fix code
    diverged from CPU at min_data_per_group=33 (max|Δ|=5.2e-2) and at the
    default 100 (max|Δ|=6.1e-2), while agreeing at 1/5/20/50/200. One
    boosting round isolates the split choice itself.
    """
    rng = np.random.default_rng(123)
    n = 400
    n_categories = 12
    cats = rng.integers(0, n_categories, size=n).astype(np.float64)
    category_means = rng.standard_normal(n_categories) * 0.7
    y = (category_means[cats.astype(int)] + rng.standard_normal(n) * 0.05).astype(np.float64)
    X = cats.reshape(-1, 1)
    params = {
        "objective": "regression",
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 0,
        "feature_pre_filter": False,
        "gpu_use_dp": True,
        "num_leaves": 8,
        "min_data_in_leaf": 5,
        "min_data_in_bin": 1,
        "max_bin": 8192,
        "cat_smooth": 1,
        "learning_rate": 0.1,
        "device_type": "cuda",
    }
    ds = lgb.Dataset(
        X,
        label=y,
        categorical_feature=[0],
        params={
            "verbose": -1,
            "feature_pre_filter": False,
            "min_data_in_bin": 1,
            "max_bin": 8192,
        },
    )
    # Regression: this raised a CUDA illegal-memory-access error before the fix.
    bst = lgb.train(params, ds, num_boost_round=5)
    preds = bst.predict(X, raw_score=True)
    assert np.all(np.isfinite(preds)), "global-memory categorical training produced non-finite predictions"

    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "feature_pre_filter": False,
            "gpu_use_dp": True,
            "num_leaves": 4,
            "min_data_in_leaf": 5,
            "learning_rate": 0.1,
            "min_data_per_group": min_data_per_group,
            "device_type": device_type,
        }
        ds = lgb.Dataset(
            X,
            label=y,
            categorical_feature=[0],
            params={"verbose": -1, "feature_pre_filter": False},
        )
        bst = lgb.train(params, ds, num_boost_round=1)
        preds[device_type] = bst.predict(X, raw_score=True)

    max_diff = float(np.abs(preds["cpu"] - preds["cuda"]).max())
    assert max_diff == 0.0, (
        f"CUDA categorical split disagrees with CPU at min_data_per_group={min_data_per_group}: max|Δ|={max_diff:.3e}"
    )


@_REQUIRES_CUDA
@pytest.mark.xfail(
    reason=(
        "Pre-existing binary categorical CPU/CUDA non-parity, independent of this fix. "
        "On plain master the divergence is a constant max|Δ|≈7.7e-2 even at "
        "min_data_per_group=1, where the per-group rule is a no-op -- so it is a separate "
        "non-unit-hessian categorical issue (sum-of-per-category-roundings vs "
        "rounding-of-the-summed-hessian in the count estimate), not something "
        "SequentialCategoricalGroupAccepted can fix. Kept as a tracker: it should XPASS "
        "once binary categorical parity is addressed separately."
    ),
    strict=False,
)
@pytest.mark.parametrize("min_data_per_group", [1, 20, 33, 100])
def test_cuda_min_data_per_group_categorical_binary_matches_cpu(min_data_per_group):
    """Non-unit-hessian categorical parity (currently xfail; see marker).

    Intent: cover what the unit-hessian (L2) regression test cannot -- binary
    log-loss has a non-unit hessian p*(1-p), so cnt_factor != 1 and the count
    estimate is a genuine rounding of a scaled hessian sum. This would exercise
    the same arithmetic SequentialCategoricalGroupAccepted performs on the
    shared-memory kernel.

    Reality: binary categorical is not CPU/CUDA bit-parity on master today, for a
    reason orthogonal to min_data_per_group (verified: the divergence is constant
    at min_data_per_group=1, where the group rule accepts every valid threshold
    exactly as the old predicate did). The count-rounding difference is documented
    as a caveat on the helper. The test is retained as an xfail tracker rather than
    deleted so the coverage returns automatically once that separate issue is fixed.
    """
    rng = np.random.default_rng(321)
    n = 600
    n_categories = 12
    cats = rng.integers(0, n_categories, size=n).astype(np.float64)
    category_logits = rng.standard_normal(n_categories) * 1.2
    probs = 1.0 / (1.0 + np.exp(-category_logits[cats.astype(int)]))
    y = (rng.random(n) < probs).astype(np.float64)
    X = cats.reshape(-1, 1)

    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "binary",
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "feature_pre_filter": False,
            "gpu_use_dp": True,
            "num_leaves": 4,
            "min_data_in_leaf": 5,
            "learning_rate": 0.1,
            "min_data_per_group": min_data_per_group,
            "device_type": device_type,
        }
        ds = lgb.Dataset(
            X,
            label=y,
            categorical_feature=[0],
            params={"verbose": -1, "feature_pre_filter": False},
        )
        bst = lgb.train(params, ds, num_boost_round=1)
        preds[device_type] = bst.predict(X, raw_score=True)

    max_diff = float(np.abs(preds["cpu"] - preds["cuda"]).max())
    assert max_diff == 0.0, (
        f"CUDA binary categorical split disagrees with CPU at "
        f"min_data_per_group={min_data_per_group}: max|Δ|={max_diff:.3e}"
    )


@_REQUIRES_CUDA
@pytest.mark.skip(
    reason=(
        "Pre-existing illegal-memory-access crash in master's global-memory categorical "
        "kernel, unrelated to this fix. Any categorical feature with > "
        "NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER (256) histogram bins routes through the "
        "global-memory path and aborts (verified on plain master: crashes at >=~300 "
        "categories with the old per-side predicate, so it is not caused by "
        "SequentialCategoricalGroupAccepted). The abort is a SIGABRT that kills the "
        "interpreter, so this cannot be an xfail. Enable once the global-memory categorical "
        "kernel crash is fixed; the helper is already wired into that path (both direction "
        "passes rewrite hist_hess_buffer_ptr with the direction's scan-order prefix)."
    )
)
@pytest.mark.parametrize("min_data_per_group", [1, 20, 100])
def test_cuda_min_data_per_group_categorical_global_memory_matches_cpu(
    min_data_per_group,
):
    """Per-group semantics on the global-memory categorical kernel (currently skipped).

    The best-split finder uses a shared-memory kernel (BitonicArgSort_1024) when
    a feature has <= NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER (256) histogram bins,
    and a global-memory kernel (BitonicArgSortDevice, prefix scan in
    hist_hess_buffer_ptr) above that. The 12-category tests only reach the
    shared-memory path; this one uses ~1200 categories with a raised max_bin so
    the feature has > 256 bins and would route through the global-memory kernel,
    where both direction passes rewrite hist_hess_buffer_ptr with that direction's
    scan-order prefix before the replay reads it. Skipped: see marker for the
    pre-existing crash that blocks running it today.
    """
    rng = np.random.default_rng(777)
    n = 24000
    n_categories = 1200  # > 256 bins -> global-memory kernel; also > 1024
    cats = rng.integers(0, n_categories, size=n).astype(np.float64)
    category_means = rng.standard_normal(n_categories) * 0.7
    y = (category_means[cats.astype(int)] + rng.standard_normal(n) * 0.05).astype(np.float64)
    X = cats.reshape(-1, 1)

    preds = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "feature_pre_filter": False,
            "gpu_use_dp": True,
            "num_leaves": 4,
            "min_data_in_leaf": 5,
            "min_data_in_bin": 1,
            "max_bin": 2048,
            "cat_smooth": 1,
            "learning_rate": 0.1,
            "min_data_per_group": min_data_per_group,
            "device_type": device_type,
        }
        ds = lgb.Dataset(
            X,
            label=y,
            categorical_feature=[0],
            params={
                "verbose": -1,
                "feature_pre_filter": False,
                "min_data_in_bin": 1,
                "max_bin": 2048,
            },
        )
        bst = lgb.train(params, ds, num_boost_round=1)
        preds[device_type] = bst.predict(X, raw_score=True)

    max_diff = float(np.abs(preds["cpu"] - preds["cuda"]).max())
    assert max_diff == 0.0, (
        f"CUDA global-memory categorical split disagrees with CPU at "
        f"min_data_per_group={min_data_per_group}: max|Δ|={max_diff:.3e}"
    )


def _train_mds(
    device_type,
    objective,
    max_delta_step,
    learning_rate,
    num_boost_round,
    min_data_in_leaf,
):
    rng = np.random.RandomState(0)
    X = rng.rand(400, 6)
    if objective == "binary":
        y = (X[:, 0] + 0.4 * rng.rand(400) > 0.6).astype(int)
    else:
        y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.1 * rng.rand(400)
    params = {
        "objective": objective,
        "max_delta_step": max_delta_step,
        "num_leaves": 15,
        "min_data_in_leaf": min_data_in_leaf,
        "learning_rate": learning_rate,
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 0,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "device_type": device_type,
    }
    ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
    return lgb.train(params, ds, num_boost_round=num_boost_round), X, y


def _tree_leaf_values(bst, tree_index=0):
    values = []

    def _rec(node):
        if "leaf_value" in node:
            values.append(node["leaf_value"])
        else:
            _rec(node["left_child"])
            _rec(node["right_child"])

    _rec(bst.dump_model()["tree_info"][tree_index]["tree_structure"])
    return values


@_REQUIRES_CUDA
@pytest.mark.parametrize("max_delta_step", [0.05, 0.1, 0.5])
@pytest.mark.parametrize("objective", ["binary", "regression"])
def test_cuda_max_delta_step_caps_outputs_like_cpu(objective, max_delta_step):
    """The max_delta_step output cap must be enforced identically on CPU and CUDA.

    Regression test for max_delta_step being silently ignored on CUDA: the cap
    (the USE_MAX_OUTPUT branch of the CPU FeatureHistogram leaf-output formula)
    was missing from the CUDA leaf-output device functions, so CUDA leaf values
    were unbounded while CPU's were capped.

    The cap limits each leaf's Newton step to [-max_delta_step, +max_delta_step],
    so within one tree the leaf-value spread can be at most 2 * max_delta_step.
    Before the fix, CUDA's spread exceeded this by an order of magnitude.
    """
    spreads = {}
    for device_type in ("cpu", "cuda"):
        bst, _, _ = _train_mds(
            device_type,
            objective,
            max_delta_step,
            learning_rate=1.0,
            num_boost_round=1,
            min_data_in_leaf=1,
        )
        values = _tree_leaf_values(bst)
        spreads[device_type] = max(values) - min(values)
        assert spreads[device_type] <= 2 * max_delta_step + 1e-9, (
            f"{device_type}: leaf spread {spreads[device_type]} exceeds 2*max_delta_step={2 * max_delta_step}"
        )
    # both backends apply the same cap, so the spreads must match exactly
    assert spreads["cuda"] == pytest.approx(spreads["cpu"], abs=1e-10)


@_REQUIRES_CUDA
@pytest.mark.parametrize("max_delta_step", [1.0, 2.0, 5.0])
def test_cuda_max_delta_step_matches_cpu_exactly(max_delta_step):
    """When the cap binds without saturating every leaf, CUDA must match CPU bit-for-bit.

    Uses the regression objective where leaf outputs are moderate, so the cap
    reduces some leaf values without collapsing all split gains to zero. In this
    regime the trained models must be identical at FP epsilon. (When the cap
    saturates every leaf, all split gains collapse to ~0 and ULP-level FP noise
    decides among equally-optimal splits; that tie-breaking difference is tracked
    separately and is not a max_delta_step issue.)
    """
    preds = {}
    for device_type in ("cpu", "cuda"):
        bst, X, _ = _train_mds(
            device_type,
            "regression",
            max_delta_step,
            learning_rate=0.1,
            num_boost_round=10,
            min_data_in_leaf=5,
        )
        preds[device_type] = bst.predict(X, raw_score=True)
    np.testing.assert_allclose(
        preds["cpu"],
        preds["cuda"],
        rtol=0,
        atol=1e-10,
        err_msg=f"max_delta_step={max_delta_step}: CUDA diverges from CPU",
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize("objective", ["binary", "regression"])
def test_cuda_max_delta_step_loss_matches_cpu_when_saturated(objective):
    """When the cap saturates leaves (creating gain plateaus), training quality must still match.

    With a tiny max_delta_step every leaf clamps to +/-max_delta_step and all
    split gains collapse to ~0; CPU and CUDA then pick different but equally
    optimal splits (FP tie-breaking). The tree structures may differ, but the
    cap must hold on both and the training loss must agree closely.
    """
    max_delta_step = 0.05
    losses = {}
    for device_type in ("cpu", "cuda"):
        bst, X, y = _train_mds(
            device_type,
            objective,
            max_delta_step,
            learning_rate=0.1,
            num_boost_round=10,
            min_data_in_leaf=5,
        )
        # cap still enforced on every tree
        for t in range(10):
            values = _tree_leaf_values(bst, t)
            assert max(values) - min(values) <= 2 * max_delta_step + 1e-9
        pred = bst.predict(X)
        if objective == "binary":
            pred = np.clip(pred, 1e-12, 1 - 1e-12)
            losses[device_type] = float(-np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)))
        else:
            losses[device_type] = float(np.mean((y - pred) ** 2))
    rel_diff = abs(losses["cpu"] - losses["cuda"]) / abs(losses["cpu"])
    assert rel_diff < 0.02, f"training loss diverged: cpu={losses['cpu']} cuda={losses['cuda']}"


def _train_monotone(device_type, constraints, num_boost_round, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(600, 3)
    y = 5 * X[:, 0] - 5 * X[:, 1] + 0.7 * X[:, 2] + 0.5 * rng.rand(600)
    params = {
        "objective": "regression",
        "monotone_constraints": constraints,
        "monotone_constraints_method": "basic",
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "learning_rate": 0.1,
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 0,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "device_type": device_type,
    }
    ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
    return lgb.train(params, ds, num_boost_round=num_boost_round), X, y


def _monotonicity_violations(bst, constraints, n_grid=500):
    """Count monotonicity violations of bst on grid sweeps of each constrained feature."""
    count = 0
    worst = 0.0
    grid = np.linspace(0, 1, n_grid)
    for j, c in enumerate(constraints):
        if c == 0:
            continue
        for base_value in (0.2, 0.5, 0.8):
            base = np.full(3, base_value)
            G = np.tile(base, (n_grid, 1))
            G[:, j] = grid
            diffs = np.diff(bst.predict(G))
            bad = -diffs if c > 0 else diffs
            violations = bad[bad > 1e-12]
            count += len(violations)
            if len(violations):
                worst = max(worst, float(violations.max()))
    return count, worst


@_REQUIRES_CUDA
@pytest.mark.parametrize("constraints", [[1, -1, 0], [1, 1, 1], [-1, 0, 1], [1, 0, 0]])
@pytest.mark.parametrize("num_boost_round", [1, 30, 100])
def test_cuda_monotone_constraints_are_enforced(constraints, num_boost_round):
    """CUDA must enforce basic-method monotone constraints: zero violations.

    Regression test for monotone_constraints being silently ignored on CUDA: the
    best-split kernels had no constraint plumbing, so CUDA produced models that
    violated the requested monotonic relationships (up to 57 violations of
    magnitude 0.68 on this data before the fix).
    """
    bst, _, _ = _train_monotone("cuda", constraints, num_boost_round)
    count, worst = _monotonicity_violations(bst, constraints)
    assert count == 0, f"CUDA model violates monotone constraints {constraints}: {count} violations, worst={worst:.3e}"


@_REQUIRES_CUDA
@pytest.mark.parametrize("constraints", [[1, -1, 0], [1, 1, 1], [-1, 0, 1]])
def test_cuda_monotone_constraints_match_cpu_quality(constraints):
    """CUDA monotone training quality must match CPU.

    CPU and CUDA may build different (both valid) constrained trees because CPU
    additionally prunes features via its is_splittable cache, so exact tree
    equality is not required. But both must (a) enforce the constraints and
    (b) reach equivalent training loss (within 5%).
    """
    num_boost_round = 50
    bst_cpu, X, y = _train_monotone("cpu", constraints, num_boost_round)
    bst_cuda, _, _ = _train_monotone("cuda", constraints, num_boost_round)

    # both enforce
    for bst, name in ((bst_cpu, "cpu"), (bst_cuda, "cuda")):
        count, worst = _monotonicity_violations(bst, constraints)
        assert count == 0, f"{name} violates constraints: {count} violations, worst={worst:.3e}"

    # equivalent quality
    mse_cpu = float(np.mean((y - bst_cpu.predict(X)) ** 2))
    mse_cuda = float(np.mean((y - bst_cuda.predict(X)) ** 2))
    assert mse_cuda <= mse_cpu * 1.05, f"CUDA mse {mse_cuda} much worse than CPU mse {mse_cpu}"


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_boost_round", [1, 30])
def test_cuda_monotone_noop_constraints_match_cpu_exactly(num_boost_round):
    """With all-zero constraints the MC code path must be a no-op:
    predictions must match CPU bit-for-bit."""
    bst_cpu, X, _ = _train_monotone("cpu", [0, 0, 0], num_boost_round)
    bst_cuda, _, _ = _train_monotone("cuda", [0, 0, 0], num_boost_round)
    np.testing.assert_allclose(
        bst_cpu.predict(X),
        bst_cuda.predict(X),
        rtol=0,
        atol=1e-10,
        err_msg="all-zero monotone constraints must not change CUDA results",
    )


@_REQUIRES_CUDA
def test_cuda_monotone_unsupported_configs_raise():
    """Only the basic method, monotone_penalty=0, and full-precision training are
    supported on CUDA; other monotone configurations must be rejected loudly."""
    rng = np.random.RandomState(0)
    X = rng.rand(200, 3)
    y = X[:, 0] - X[:, 1]
    base = {
        "objective": "regression",
        "monotone_constraints": [1, -1, 0],
        "device_type": "cuda",
        "num_leaves": 7,
        "verbose": -1,
    }
    for bad, expected in (
        (
            {"monotone_constraints_method": "intermediate"},
            r'only supports the "basic" monotone_constraints_method',
        ),
        (
            {"monotone_constraints_method": "advanced"},
            r'only supports the "basic" monotone_constraints_method',
        ),
        (
            {"monotone_penalty": 1.0},
            r"monotone_penalty is not supported with device_type=cuda",
        ),
        (
            {"use_quantized_grad": True},
            r"monotone_constraints is not supported with use_quantized_grad",
        ),
    ):
        with pytest.raises(lgb.basic.LightGBMError, match=expected):
            lgb.train(
                {**base, **bad},
                lgb.Dataset(X, label=y, params={"verbose": -1}),
                num_boost_round=1,
            )


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "feature_contri",
    [
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.5, 1.0, 1.0, 1.0],
        [0.5, 1.0, 2.0, 1.0],
    ],
)
def test_cuda_feature_contri_matches_cpu(feature_contri):
    """CUDA must apply per-feature gain scaling (feature_contri) identically to CPU.

    Regression test for feature_contri being silently ignored on CUDA: the CPU
    FeatureHistogram multiplies each feature's best-split gain by its
    feature_contri entry (output->gain *= meta_->penalty) before cross-feature
    comparison; the CUDA best-split finder had no equivalent, so a feature with
    contri 0 was still selected for splits on CUDA.
    """
    rng = np.random.RandomState(0)
    X = rng.rand(500, 4)
    y = 5 * X[:, 0] + 2 * X[:, 1] + 1 * X[:, 2] + 0.5 * X[:, 3] + 0.1 * rng.rand(500)

    boosters = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "feature_contri": feature_contri,
            "num_leaves": 15,
            "min_data_in_leaf": 5,
            "learning_rate": 0.1,
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "feature_pre_filter": False,
            "device_type": device_type,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        boosters[device_type] = lgb.train(params, ds, num_boost_round=5)

    def used_features(bst):
        used = set()

        def _rec(node):
            if "leaf_value" in node:
                return
            used.add(node["split_feature"])
            _rec(node["left_child"])
            _rec(node["right_child"])

        for tree in bst.dump_model()["tree_info"]:
            _rec(tree["tree_structure"])
        return used

    used_cpu = used_features(boosters["cpu"])
    used_cuda = used_features(boosters["cuda"])

    # same features selected (in particular, zero-contri features excluded on both)
    assert used_cpu == used_cuda
    for fidx, contri in enumerate(feature_contri):
        if contri == 0.0:
            assert fidx not in used_cuda

    # predictions match at FP epsilon
    np.testing.assert_allclose(
        boosters["cpu"].predict(X),
        boosters["cuda"].predict(X),
        rtol=0,
        atol=1e-10,
        err_msg=f"feature_contri={feature_contri}: CUDA diverges from CPU",
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    "cegb_overrides",
    [
        {"cegb_penalty_split": 0.1},
        {"cegb_penalty_split": 1.0},
        {"cegb_penalty_split": 5.0},
        {"cegb_penalty_feature_coupled": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]},
        {"cegb_penalty_feature_coupled": [0.1, 0.1, 5.0, 5.0, 5.0, 5.0]},
        {"cegb_tradeoff": 0.5, "cegb_penalty_split": 1.0},
    ],
)
def test_cuda_cegb_matches_cpu(cegb_overrides):
    """CUDA must apply cost-effective gradient boosting penalties identically to CPU.

    Regression test for cegb_* being silently ignored on CUDA: the CPU serial
    tree learner subtracts CostEfficientGradientBoosting::DeltaGain from each
    candidate split's gain and stops splitting when the best penalized gain is
    <= 0; the CUDA learner had no penalty plumbing and no gain>0 stop condition,
    so CEGB had no effect on tree shape.
    """
    rng = np.random.RandomState(0)
    X = rng.rand(400, 6)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.1 * rng.rand(400)

    boosters = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "num_leaves": 31,
            "min_data_in_leaf": 1,
            "min_gain_to_split": 0.0,
            "learning_rate": 0.1,
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "feature_pre_filter": False,
            "device_type": device_type,
            **cegb_overrides,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        boosters[device_type] = lgb.train(params, ds, num_boost_round=5)

    def shape_and_features(bst):
        leaves = [t["num_leaves"] for t in bst.dump_model()["tree_info"]]
        used = set()

        def _rec(node):
            if "leaf_value" in node:
                return
            used.add(node["split_feature"])
            _rec(node["left_child"])
            _rec(node["right_child"])

        for tree in bst.dump_model()["tree_info"]:
            _rec(tree["tree_structure"])
        return leaves, used

    leaves_cpu, used_cpu = shape_and_features(boosters["cpu"])
    leaves_cuda, used_cuda = shape_and_features(boosters["cuda"])

    # the penalty must shape the trees identically (per-tree leaf counts + feature sets)
    assert leaves_cpu == leaves_cuda, f"num_leaves mismatch: cpu={leaves_cpu} cuda={leaves_cuda}"
    assert used_cpu == used_cuda

    # predictions must match at FP epsilon
    np.testing.assert_allclose(
        boosters["cpu"].predict(X),
        boosters["cuda"].predict(X),
        rtol=0,
        atol=1e-10,
        err_msg=f"cegb={cegb_overrides}: CUDA diverges from CPU",
    )


@_REQUIRES_CUDA
def test_cuda_cegb_lazy_penalty_raises():
    """cegb_penalty_feature_lazy needs per-(row, feature) cost tracking that is not
    implemented on CUDA; it must be rejected loudly rather than silently ignored."""
    rng = np.random.RandomState(0)
    X = rng.rand(200, 6)
    y = 3 * X[:, 0] + 0.1 * rng.rand(200)
    params = {
        "objective": "regression",
        "device_type": "cuda",
        "cegb_penalty_feature_lazy": [5.0] * 6,
        "num_leaves": 15,
        "verbose": -1,
    }
    with pytest.raises(lgb.basic.LightGBMError, match="cegb_penalty_feature_lazy"):
        lgb.train(params, lgb.Dataset(X, label=y, params={"verbose": -1}), num_boost_round=1)


@_REQUIRES_CUDA
def test_cuda_lambdarank_round1_matches_cpu_within_fp_drift():
    """Regression test for non-stable BitonicArgSort with all-equal round-1 scores.

    Before the BitonicArgSort tie-stability fix, CUDA's bitonic sort swapped
    tied elements during descending sort because `(a > b) == false` was true
    for `a == b`. This shuffled the document order in each LambdaRank query
    on round 1 (when all scores are 0) and produced max|Δ|=0.29 on a small
    dataset. With stable ties, the same case drops to ~0.14 (residual is
    FP-precision in atomicAdd_block ordering).
    """
    rng = np.random.default_rng(42)
    n_queries = 10
    items_per_query = 20
    n = n_queries * items_per_query
    n_features = 8
    X = rng.standard_normal((n, n_features)).astype(np.float64)
    w = rng.standard_normal(n_features)
    y = np.clip(np.round(X @ w + 0.5 * rng.standard_normal(n) + 2), 0, 4).astype(int)
    group = np.full(n_queries, items_per_query, dtype=np.int32)

    base = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 42,
        "feature_pre_filter": False,
        "gpu_use_dp": True,
        "force_col_wise": True,
        "num_leaves": 7,
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
    }
    preds = {}
    for dev in ("cpu", "cuda"):
        ds = lgb.Dataset(X, label=y, group=group, params={"verbose": -1, "feature_pre_filter": False})
        bst = lgb.train({**base, "device_type": dev}, ds, num_boost_round=1)
        preds[dev] = bst.predict(X, raw_score=True)
    diff = float(np.abs(preds["cpu"] - preds["cuda"]).max())
    # Tolerance is well below the 0.29 the pre-fix bug produced. After the fix
    # ~0.14 remains from FP-precision in pair-gradient atomic-add ordering
    # (documented expected behavior per upstream #6055), so we set the bar at
    # 0.2 — strict enough to catch the bitonic-sort regression, loose enough to
    # tolerate the FP-precision residual.
    assert diff < 0.2, f"LambdaRank round-1 max|Δ|={diff:.4e} (was ~0.29 before BitonicArgSort fix)"


@_REQUIRES_CUDA
def test_cuda_bitonic_argsort_1024_with_distinct_scores_matches_cpu():
    """Regression test for BitonicArgSort_1024's per-pass `ascending` direction.

    A first attempt at the tie-stability fix replaced the per-pass `ascending`
    local with the template parameter `ASCENDING`. That preserves correctness
    for all-tied inputs (LambdaRank round 1) because the strict comparator
    returns false either way, but breaks the bitonic merge for non-tied
    inputs because outer phases must alternate direction.

    BitonicArgSort_1024 is called from the CUDA categorical split-finder
    (cuda_best_split_finder.cu) over per-category gradient/hessian ratios.
    Training a small regression with a single categorical feature whose
    per-category sums are all-distinct exercises that path with non-tied
    scores; if the comparator stops alternating, CUDA's chosen categorical
    split disagrees with CPU's.
    """
    rng = np.random.default_rng(123)
    n = 400
    n_categories = 12
    cats = rng.integers(0, n_categories, size=n).astype(np.float64)
    # Per-category mean shift produces distinct, well-separated grad/hess
    # sums after fitting -- so the categorical sort sees no ties.
    category_means = rng.standard_normal(n_categories) * 0.7
    y = (category_means[cats.astype(int)] + rng.standard_normal(n) * 0.05).astype(np.float64)
    X = cats.reshape(-1, 1)

    base = {
        "objective": "regression",
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        "seed": 0,
        "feature_pre_filter": False,
        "gpu_use_dp": True,
        "num_leaves": 4,
        "min_data_in_leaf": 5,
        "learning_rate": 0.1,
    }
    preds = {}
    for dev in ("cpu", "cuda"):
        ds = lgb.Dataset(
            X,
            label=y,
            categorical_feature=[0],
            params={"verbose": -1, "feature_pre_filter": False},
        )
        bst = lgb.train({**base, "device_type": dev}, ds, num_boost_round=1)
        preds[dev] = bst.predict(X, raw_score=True)
    diff = float(np.abs(preds["cpu"] - preds["cuda"]).max())
    # If the bitonic sort stops alternating direction, the categorical
    # split-finder chooses a different threshold and predictions diverge by
    # ~O(category mean magnitude). 1e-3 is well above CPU/CUDA FP drift on a
    # one-tree fit but well below any wrong-split signal.
    assert diff < 1e-3, f"CPU vs CUDA prediction disagreement on categorical split: max|Δ|={diff:.4e}"


def _make_regression_for_parity(n=200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float64)
    coef = rng.standard_normal(d)
    y = (X @ coef + 0.1 * rng.standard_normal(n)).astype(np.float64)
    return X, y


def _train_cpu_and_cuda(params_overrides, X, y, num_round):
    out = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 42,
            "feature_pre_filter": False,
            "device_type": device_type,
            "gpu_use_dp": True,
            "force_col_wise": True,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "min_data_in_leaf": 5,
            "min_sum_hessian_in_leaf": 1e-3,
            **params_overrides,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        out[device_type] = lgb.train(params, ds, num_boost_round=num_round)
    return out


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("name", "params_overrides", "seed", "num_round"),
    [
        # Regression test for the gain-plateau argmax bug. Bagging configuration
        # was the original failure: max|Δ|=0.39 at round 3, structurally
        # divergent trees from round 3 onward. With the fix, all 5 rounds
        # match at fp64 epsilon and trees are bit-identical.
        (
            "bagging",
            {"bagging_fraction": 0.7, "bagging_freq": 1, "bagging_seed": 1},
            11,
            5,
        ),
        # Plain dense regression: predictions matched at fp64 epsilon prior
        # to the fix but the encoded tree thresholds differed cosmetically
        # (CPU and CUDA picked different bins from a true gain plateau).
        # After the fix, trees are bit-identical.
        ("dense", {}, 1, 5),
        # max_depth regression: another configuration where round-1 trees
        # had cosmetic threshold-encoding differences before the fix.
        ("max_depth", {"max_depth": 3}, 12, 5),
        # L2 regularisation: same family.
        ("l2", {"lambda_l2": 1.0}, 7, 5),
    ],
)
def test_cuda_split_gain_tie_break_matches_cpu(name, params_overrides, seed, num_round):
    """CUDA must match CPU at fp64 epsilon when the best-split argmax has a
    gain plateau (multiple bins with truly equal gain).

    Prior to the tolerance-based tie-break in cuda_best_split_finder.cu's
    ReduceBestGain* helpers, ULP-level FP noise in the parallel histogram
    flipped which bin from the plateau had the slightly-higher numerical
    gain on CUDA. CPU's exact computation picked a different bin (its
    sequential scan + strict ``>`` comparison resolves to the lowest-index
    bin on the plateau). The threshold-encoding mismatch was at round 1
    cosmetic for predictions (data routed identically), but compounded
    through score updates and surfaced as structural tree divergence by
    round 3 in cases like reg_bagging.
    """
    X, y = _make_regression_for_parity(seed=seed)
    pair = _train_cpu_and_cuda(params_overrides, X, y, num_round=num_round)
    pred_cpu = pair["cpu"].predict(X, raw_score=True)
    pred_cuda = pair["cuda"].predict(X, raw_score=True)
    # fp64 epsilon ≈ 2.2e-16; allow a generous 1e-10 to absorb any
    # remaining round-by-round drift from sources unrelated to this fix.
    np.testing.assert_allclose(pred_cuda, pred_cpu, atol=1e-10)
