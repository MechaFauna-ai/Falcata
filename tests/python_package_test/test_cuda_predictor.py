"""The native CUDA predictor against the CPU predictor on every model kind and input layout.

Both predictors route a float32 input with thresholds rounded down to float (exact for every float) and see values
with |x| <= 1e-35 as zero, so leaf indices and feature contributions are bit-identical. Raw scores:
- ``deterministic=True`` adds each output's leaf values in tree order in double precision, as the CPU predictor does:
  bit-identical, and normal predictions (converted by the same host code, Boosting::ConvertRawToPrediction) too;
- the default mode adds them in the canonical pairwise order (groups of 8 iterations in order from 0.0, then the
  complete binary tree over the groups). ``_pairwise_reference`` builds that sum from the CPU predictor's per-iteration
  scores, and the default mode must equal it bit for bit, in every batch, layout and kernel shape.
Every comparison below is exact unless it says otherwise.
"""

import itertools
import json
import subprocess
import sys

import numpy as np
import pytest

import falcata as lgb
from falcata import basic

CUDA_PREDICTOR = basic._cuda_predictor_usable()
requires_cuda = pytest.mark.skipif(not CUDA_PREDICTOR, reason="the CUDA predictor needs a CUDA build and a GPU")


def test_cpu_build_refuses_cuda_device():
    """Without a usable CUDA predictor, device='cuda' is an error and device=None predicts on the CPU."""
    if CUDA_PREDICTOR:
        pytest.skip("this build and machine can run the CUDA predictor")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    booster = lgb.train(
        {"objective": "regression", "device_type": "cpu", "verbose": -1}, lgb.Dataset(X, X[:, 0]), num_boost_round=3
    )
    booster.params["device_type"] = "cuda"
    np.testing.assert_array_equal(booster.predict(X), booster.predict(X, device="cpu"))
    with pytest.raises(lgb.basic.FalcataError, match="CUDA"):
        booster.predict(X, device="cuda")


def test_device_argument_is_checked():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    booster = lgb.train({"objective": "regression", "verbose": -1, "device_type": "cpu"}, lgb.Dataset(X, X[:, 0]), 2)
    with pytest.raises(ValueError, match="device must be"):
        booster.predict(X, device="tpu")


def test_use_fil_was_removed():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    booster = lgb.train({"objective": "regression", "verbose": -1, "device_type": "cpu"}, lgb.Dataset(X, X[:, 0]), 2)
    with pytest.raises(TypeError, match="use_fil"):
        booster.predict(X, use_fil=True)


# --------------------------------------------------------------------------- model zoo


def _features(rng, n, d, *, nan_frac=0.1, zero_frac=0.1, categorical=()):
    X = rng.normal(size=(n, d))
    X[rng.random((n, d)) < zero_frac] = 0.0
    X[rng.random((n, d)) < nan_frac] = np.nan
    for c in categorical:
        X[:, c] = rng.integers(0, 40, size=n)
        X[rng.random(n) < 0.05, c] = np.nan
        X[rng.random(n) < 0.02, c] = -3  # negative categories go right
    return X


def _base(device="cpu", **extra):
    params = {
        "verbose": -1,
        "device_type": device,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "learning_rate": 0.3,
        "seed": 7,
        "deterministic": True,
        "num_threads": 4,
    }
    params.update(extra)
    return params


def _zoo():
    rng = np.random.default_rng(20261117)
    n = 2000
    X = _features(rng, n, 12)
    y = np.nansum(X[:, :3], axis=1) + 0.1 * rng.normal(size=n)
    yb = (y > 0).astype(float)
    yk = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75])).astype(float)
    Xc = _features(rng, n, 8, categorical=(1, 4))
    yc = (Xc[:, 1] % 3) + np.nan_to_num(Xc[:, 0])
    models = {
        "regression": (_base(objective="regression"), X, y, {}),
        "regression_zero_missing": (_base(objective="regression", zero_as_missing=True), X, y, {}),
        "regression_no_missing": (_base(objective="regression", use_missing=False), X, y, {}),
        "binary": (_base(objective="binary"), X, yb, {}),
        "multiclass": (_base(objective="multiclass", num_class=4), X, yk, {}),
        "multiclassova": (_base(objective="multiclassova", num_class=4), X, yk, {}),
        "xentropy": (_base(objective="cross_entropy"), X, yb, {}),
        "poisson": (_base(objective="poisson"), X, np.abs(y), {}),
        "rf": (_base(objective="regression", boosting="rf", bagging_fraction=0.5, bagging_freq=1), X, y, {}),
        "dart": (_base(objective="regression", boosting="dart"), X, y, {}),
        "linear_tree": (_base(objective="regression", linear_tree=True), np.nan_to_num(X), y, {}),
        "linear_tree_nan": (_base(objective="regression", linear_tree=True), X, y, {}),
        "categorical": (_base(objective="regression", max_cat_to_onehot=1), Xc, yc, {"categorical_feature": [1, 4]}),
        "categorical_onehot": (_base(objective="regression"), Xc, yc, {"categorical_feature": [1, 4]}),
        "deep": (_base(objective="regression", num_leaves=255, min_data_in_leaf=1, max_depth=-1), X, y, {}),
        "stumps": (_base(objective="regression", num_leaves=2), X, y, {}),
    }
    out = {}
    for name, (params, data, label, ds_kwargs) in models.items():
        booster = lgb.train(params, lgb.Dataset(data, label, **ds_kwargs), num_boost_round=25)
        out[name] = (booster, data)
    return out


@pytest.fixture(scope="module")
def zoo():
    if not CUDA_PREDICTOR:
        pytest.skip("the CUDA predictor needs a CUDA build and a GPU")
    return _zoo()


def _probe_rows(booster, X, rng):
    """Rows that sit exactly on, just below and just above the model's thresholds, plus special values."""
    dump = booster.dump_model()
    thresholds = []

    def walk(node):
        if "split_index" in node:
            if node.get("decision_type") == "<=":
                thresholds.append((node["split_feature"], float(node["threshold"])))
            walk(node["left_child"])
            walk(node["right_child"])

    for tree in dump["tree_info"]:
        walk(tree["tree_structure"])
    rows = [X[rng.integers(0, len(X))].copy() for _ in range(64)]
    for feature, t in thresholds[:400]:
        base = X[rng.integers(0, len(X))].copy()
        for v in (t, np.nextafter(t, -np.inf), np.nextafter(t, np.inf), float(np.float32(t))):
            r = base.copy()
            r[feature] = v
            rows.append(r)
    special = X[0].copy()
    for v in (0.0, -0.0, 1e-36, -1e-36, 1e-35, 2e-35, np.inf, -np.inf, np.nan, 1e300, -1e300):
        r = special.copy()
        r[:] = v
        rows.append(r)
    return np.array(rows)


def _assert_same(booster, X, **kwargs):
    """The deterministic mode equals the CPU predictor bit for bit."""
    cpu = booster.predict(X, device="cpu", **kwargs)
    gpu = booster.predict(X, device="cuda", deterministic=True, **kwargs)
    assert type(gpu) is type(cpu)
    assert gpu.shape == cpu.shape, (gpu.shape, cpu.shape)
    assert gpu.dtype == cpu.dtype, (gpu.dtype, cpu.dtype)
    np.testing.assert_array_equal(gpu, cpu)


def _pairwise_reference(booster, X, start_iteration=0, num_iteration=-1):
    """The default mode's raw scores, from the CPU predictor's score of each iteration alone (0.0 + v = v exactly).

    Each output: groups of 8 consecutive iterations of the range added in order from 0.0, then pairs of group sums
    (0, 1), (2, 3), ..., pairs of those, and so on; an odd last element is paired with 0.0, which changes nothing.
    """
    total = booster.current_iteration()
    start = min(max(start_iteration, 0), total)
    end = total if num_iteration is None or num_iteration <= 0 else min(total, start + num_iteration)
    n = X.shape[0]
    k = booster.num_model_per_iteration()
    per_iteration = [
        np.asarray(booster.predict(X, device="cpu", raw_score=True, start_iteration=it, num_iteration=1)).reshape(n, k)
        for it in range(start, end)
    ]
    sums = []
    for g in range(0, len(per_iteration), 8):
        acc = np.zeros((n, k))
        for v in per_iteration[g : g + 8]:
            acc = acc + v
        sums.append(acc)
    if not sums:
        sums = [np.zeros((n, k))]
    while len(sums) > 1:
        if len(sums) % 2:
            sums.append(np.zeros((n, k)))
        sums = [sums[i] + sums[i + 1] for i in range(0, len(sums), 2)]
    return sums[0].reshape(n) if k == 1 else sums[0]


def _assert_pairwise(booster, X, start_iteration=0, num_iteration=-1, **kwargs):
    """The default mode equals the canonical pairwise sum bit for bit, and the in-order sum up to rounding."""
    gpu = booster.predict(
        X, device="cuda", raw_score=True, start_iteration=start_iteration, num_iteration=num_iteration, **kwargs
    )
    ref = _pairwise_reference(booster, X, start_iteration, num_iteration)
    assert gpu.shape == ref.shape, (gpu.shape, ref.shape)
    np.testing.assert_array_equal(gpu, ref)
    cpu = booster.predict(
        X, device="cpu", raw_score=True, start_iteration=start_iteration, num_iteration=num_iteration, **kwargs
    )
    np.testing.assert_allclose(gpu, cpu, rtol=1e-12, atol=1e-12 * (1.0 + float(np.max(np.abs(cpu), initial=0.0))))
    return gpu


@requires_cuda
@pytest.mark.parametrize(
    "name",
    [
        "regression",
        "regression_zero_missing",
        "regression_no_missing",
        "binary",
        "multiclass",
        "multiclassova",
        "xentropy",
        "poisson",
        "rf",
        "dart",
        "linear_tree",
        "linear_tree_nan",
        "categorical",
        "categorical_onehot",
        "deep",
        "stumps",
    ],
)
def test_bit_identical_to_cpu(zoo, name):
    booster, X = zoo[name]
    rng = np.random.default_rng(len(name))
    rows = np.vstack([X[:500], _probe_rows(booster, X, rng)])
    for dtype, order in itertools.product((np.float64, np.float32), ("C", "F")):
        data = np.asarray(rows, dtype=dtype, order=order)
        _assert_same(booster, data, raw_score=True)
        _assert_same(booster, data)
        _assert_same(booster, data, pred_leaf=True)
        if not name.startswith("linear_tree"):
            _assert_same(booster, data[:300], pred_contrib=True)
            _assert_same(booster, data[:300], pred_contrib=True, start_iteration=5, num_iteration=11)
        for start, num in ((0, 1), (3, 7), (10, -1), (24, 5), (30, 2)):
            _assert_same(booster, data, raw_score=True, start_iteration=start, num_iteration=num)
            _assert_same(booster, data, pred_leaf=True, start_iteration=start, num_iteration=num)
    # every other dtype is read as float32, as the CPU predictor reads it
    _assert_same(booster, np.nan_to_num(rows[:200], posinf=100, neginf=-100).astype(np.int16), raw_score=True)


@requires_cuda
@pytest.mark.parametrize(
    "name",
    [
        "regression",
        "regression_zero_missing",
        "binary",
        "multiclass",
        "multiclassova",
        "rf",
        "dart",
        "linear_tree_nan",
        "categorical",
        "deep",
        "stumps",
    ],
)
def test_default_mode_is_the_pairwise_sum(zoo, name):
    """Default mode: the canonical pairwise sum, for every dtype, layout and iteration slice; normal predictions are
    converted from it (close to the CPU's); leaf indices and contributions do not depend on the mode."""
    booster, X = zoo[name]
    rng = np.random.default_rng(len(name) + 100)
    rows = np.vstack([X[:400], _probe_rows(booster, X, rng)])
    for dtype, order in itertools.product((np.float64, np.float32), ("C", "F")):
        data = np.asarray(rows, dtype=dtype, order=order)
        _assert_pairwise(booster, data)
        for start, num in ((0, 1), (3, 7), (10, -1), (24, 5), (30, 2), (0, 9), (1, 17)):
            _assert_pairwise(booster, data, start_iteration=start, num_iteration=num)
        normal = booster.predict(data, device="cuda")
        np.testing.assert_allclose(normal, booster.predict(data, device="cpu"), rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(
            booster.predict(data, device="cuda", pred_leaf=True), booster.predict(data, device="cpu", pred_leaf=True)
        )
        if not name.startswith("linear_tree"):
            np.testing.assert_array_equal(
                booster.predict(data[:100], device="cuda", pred_contrib=True),
                booster.predict(data[:100], device="cpu", pred_contrib=True),
            )


@requires_cuda
def test_default_mode_does_not_depend_on_the_batch():
    """One row gets the same bits alone, in any slice of a batch, from host or device memory, and in any layout."""
    rng = np.random.default_rng(77)
    X = _features(rng, 3000, 30)
    y = np.nansum(X[:, :4], axis=1)
    for objective, label, extra, rounds in (
        ("regression", y, {}, 300),
        ("multiclass", np.digitize(y, [-1, 0, 1]).astype(float), {"num_class": 4}, 120),
    ):
        booster = lgb.train(_base(objective=objective, num_leaves=31, **extra), lgb.Dataset(X, label), rounds)
        test = _features(rng, 2500, 30).astype(np.float32)
        full = _assert_pairwise(booster, test)
        for lo, hi in ((0, 1), (7, 8), (0, 3), (100, 233), (1999, 2500), (5, 1205)):
            np.testing.assert_array_equal(booster.predict(test[lo:hi], device="cuda", raw_score=True), full[lo:hi])
        np.testing.assert_array_equal(booster.predict(np.asfortranarray(test), device="cuda", raw_score=True), full)
        np.testing.assert_array_equal(booster.predict(test[::7], device="cuda", raw_score=True), full[::7])


@requires_cuda
def test_shape_check_and_disabled_check(zoo):
    booster, X = zoo["regression"]
    with pytest.raises(lgb.basic.FalcataError, match="number of features"):
        booster.predict(X[:, :-1], device="cuda")
    for cols in (X[:, :-3], np.hstack([X, X[:, :2]])):
        _assert_same(booster, np.ascontiguousarray(cols), raw_score=True, predict_disable_shape_check=True)


@requires_cuda
def test_non_contiguous_and_empty_inputs(zoo):
    booster, X = zoo["multiclass"]
    _assert_same(booster, X[::3, :], raw_score=True)
    _assert_same(booster, X[:0], raw_score=True)
    _assert_same(booster, X[:0], pred_leaf=True)
    _assert_same(booster, X[:1])


@requires_cuda
def test_early_stop_is_an_error_on_the_device(zoo):
    booster, X = zoo["regression"]
    with pytest.raises(lgb.basic.FalcataError, match="pred_early_stop"):
        booster.predict(X, device="cuda", pred_early_stop=True)


@requires_cuda
def test_contributions_refused_as_on_the_cpu(zoo):
    """Linear trees have no TreeSHAP on either predictor; vector leaves neither (test_vector_leaf_models)."""
    booster, X = zoo["linear_tree"]
    for device in ("cpu", "cuda"):
        with pytest.raises(lgb.basic.FalcataError, match="linear trees"):
            booster.predict(X[:10], device=device, pred_contrib=True)


# --------------------------------------------------------------------------- device=None on host arrays


def _estimate(**overrides):
    kwargs = {
        "nrow": 100_000,
        "ncol": 50,
        "in_itemsize": 4,
        "out_bytes_per_row": 8,
        "splits_per_row": 1000.0,
        "free_device_bytes": float(16 << 30),
        "num_trees": 200,
        "packed": True,
        "loaded": True,
        "threads": 16,
    }
    kwargs.update(overrides)
    return basic._auto_cuda_estimate(**kwargs)


def test_auto_estimate_both_sides_of_the_threshold():
    """More work per row tips a request to the GPU; less (or more bytes per unit of work) keeps it on the CPU."""
    assert _estimate(splits_per_row=20_000.0)["device"] == "cuda"
    assert _estimate(nrow=100, splits_per_row=50.0)["device"] == "cpu"
    # find the boundary in rows x splits for one shape and check both of its sides
    lo, hi = 1.0, 1e7
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if _estimate(splits_per_row=mid)["device"] == "cuda":
            hi = mid
        else:
            lo = mid
    assert _estimate(splits_per_row=hi * 1.01)["device"] == "cuda"
    assert _estimate(splits_per_row=lo * 0.99)["device"] == "cpu"
    # the same work over wide rows: the copy outweighs it
    assert _estimate(splits_per_row=hi * 1.01, ncol=50_000, nrow=1000)["device"] == "cpu"


def test_auto_estimate_first_prediction_pays_the_kernel_load():
    """A request the GPU wins once its kernels are loaded can lose while they are not (the first CUDA prediction of
    a process), and packing the model counts until it is packed."""
    lo, hi = 1.0, 1e7
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if _estimate(nrow=20_000, splits_per_row=mid)["device"] == "cuda":
            hi = mid
        else:
            lo = mid
    warm = _estimate(nrow=20_000, splits_per_row=hi * 1.01)
    cold = _estimate(nrow=20_000, splits_per_row=hi * 1.01, loaded=False)
    unpacked = _estimate(nrow=20_000, splits_per_row=hi * 1.01, packed=False, num_trees=100_000)
    assert warm["device"] == "cuda"
    assert cold["device"] == "cpu"
    assert cold["cuda_s"] > warm["cuda_s"]
    assert unpacked["device"] == "cpu"
    assert unpacked["cuda_s"] > warm["cuda_s"]


def test_auto_estimate_device_memory_limits():
    """The GPU is never chosen when the call's buffers exceed the fixed cap or the share of the free memory."""
    c = basic._AUTO_CUDA_COST
    heavy = {"splits_per_row": 1e6}
    assert _estimate(**heavy)["device"] == "cuda"
    rows_over_cap = int(c["max_buffer_bytes"] // (50 * 4 + 8)) + 1
    over = _estimate(nrow=rows_over_cap, **heavy)
    assert over["device"] == "cpu"
    assert not over["fits"]
    tight = _estimate(free_device_bytes=float(100_000 * (50 * 4 + 8)) / c["max_free_fraction"] * 0.99, **heavy)
    assert tight["device"] == "cpu"
    assert not tight["fits"]
    assert _estimate(free_device_bytes=-1.0, **heavy)["device"] == "cpu"


def test_auto_estimate_more_cpu_threads_favour_the_cpu():
    lo, hi = 1.0, 1e7
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if _estimate(splits_per_row=mid, threads=16)["device"] == "cuda":
            hi = mid
        else:
            lo = mid
    assert _estimate(splits_per_row=hi * 1.01, threads=16)["device"] == "cuda"
    assert _estimate(splits_per_row=hi * 1.01, threads=64)["device"] == "cpu"


def test_cpu_predict_threads(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert basic._cpu_predict_threads({"num_threads": 3}) == 3
    assert basic._cpu_predict_threads({"n_jobs": 5}) == 5
    assert basic._cpu_predict_threads({}) >= 1
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    assert basic._cpu_predict_threads({}) == 7
    assert basic._cpu_predict_threads({"num_threads": 2}) == 2


def _split_count_reference(booster, start_iteration=0, num_iteration=-1):
    """Sum over the trees of the range: each split node's training rows over the rows at the root."""
    k = booster.num_model_per_iteration()
    trees = booster.dump_model()["tree_info"]
    total = len(trees) // k
    end = total if num_iteration <= 0 else min(total, start_iteration + num_iteration)
    out = 0.0

    def walk(node, root):
        if "split_index" not in node:
            return 0.0
        return node["internal_count"] / root + walk(node["left_child"], root) + walk(node["right_child"], root)

    for tree in trees[start_iteration * k : end * k]:
        s = tree["tree_structure"]
        if "split_index" in s:
            out += walk(s, s["internal_count"])
    return out


def _cost_inputs(booster, start=0, num=-1, what=3, params=""):
    values = (basic.ctypes.c_double * 3)()
    basic._safe_call(
        basic._LIB.FLC_BoosterCUDAPredictCostInputs(
            booster._handle,
            basic.ctypes.c_int(start),
            basic.ctypes.c_int(num),
            basic._c_str(params) if params else basic._EMPTY_C_STR,
            basic.ctypes.c_int(what),
            values,
        )
    )
    return list(values)


def _leaf_depth_reference(booster):
    """Sum over the trees of the mean depth of their leaves (each leaf counted once)."""

    def leaves(node, depth):
        if "split_index" not in node:
            return [depth]
        return leaves(node["left_child"], depth + 1) + leaves(node["right_child"], depth + 1)

    return sum(float(np.mean(leaves(t["tree_structure"], 0))) for t in booster.dump_model()["tree_info"])


def test_cost_inputs_report_the_split_nodes_per_row(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 6))
    y = X[:, 0] + np.sin(3 * X[:, 1]) + 0.1 * rng.normal(size=3000)
    for leaves in (2, 7, 31):
        booster = lgb.train(_base(objective="regression", num_leaves=leaves), lgb.Dataset(X, y), num_boost_round=12)
        for start, num in ((0, -1), (3, 4), (0, 1)):
            values = _cost_inputs(booster, start, num)
            np.testing.assert_allclose(values[0], _split_count_reference(booster, start, num), rtol=1e-12)
            if leaves == 2:
                assert values[0] == (12 - start if num <= 0 else num)
            if basic._cuda_predictor_usable():
                assert values[1] > 0
            else:
                assert values[1] == -1.0
        # only what is asked for
        assert _cost_inputs(booster, what=1)[1:] == [-1.0, 0.0]
        assert _cost_inputs(booster, what=2)[0] == -1.0
        # a model read back from text keeps its row counts, so the same number
        again = lgb.Booster(model_str=booster.model_to_string())
        np.testing.assert_allclose(_cost_inputs(again)[0], _split_count_reference(booster), rtol=1e-12)
        # a binary model saved without row counts: the mean leaf depth of each tree
        path = tmp_path / f"model_{leaves}.falb"
        path.write_bytes(booster.model_to_binary(with_stats=False))
        no_counts = lgb.Booster(model_file=str(path))
        np.testing.assert_allclose(_cost_inputs(no_counts)[0], _leaf_depth_reference(booster), rtol=1e-12)
    # multiclass: every class's trees count
    yk = np.digitize(y, np.quantile(y, [0.33, 0.66])).astype(float)
    multi = lgb.train(_base(objective="multiclass", num_class=3), lgb.Dataset(X, yk), num_boost_round=5)
    assert multi.num_trees() == 15
    np.testing.assert_allclose(_cost_inputs(multi)[0], _split_count_reference(multi), rtol=1e-12)
    np.testing.assert_allclose(_cost_inputs(multi, 1, 2)[0], _split_count_reference(multi, 1, 2), rtol=1e-12)
    # single-leaf trees pass no split node
    flat = lgb.train(_base(objective="regression", min_data_in_leaf=10_000), lgb.Dataset(X, y), num_boost_round=3)
    assert _cost_inputs(flat)[0] == 0.0


def test_auto_rule_touches_the_device_last(monkeypatch):
    """A request over the buffer cap makes no cost-input call; one the CPU wins even against a warm GPU asks only for
    the split nodes (no CUDA call); only a request the GPU could win reads the device."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(600, 5)).astype(np.float32)
    booster = lgb.train(_base(objective="regression"), lgb.Dataset(X, X[:, 0]), num_boost_round=4)
    calls = []
    original = basic._LIB.FLC_BoosterCUDAPredictCostInputs

    def recording(*args):
        calls.append(args[4].value)
        return original(*args)

    monkeypatch.setattr(basic._LIB, "FLC_BoosterCUDAPredictCostInputs", recording)
    monkeypatch.setitem(basic._AUTO_CUDA_COST, "max_buffer_bytes", 1000)
    assert not booster._auto_predict_on_cuda(X, 0, -1, basic._C_API_PREDICT_NORMAL, {})
    assert calls == []
    assert not booster._last_auto_predict["fits"]
    monkeypatch.setitem(basic._AUTO_CUDA_COST, "max_buffer_bytes", 256 << 20)
    monkeypatch.setitem(basic._AUTO_CUDA_COST, "margin", float("inf"))
    assert not booster._auto_predict_on_cuda(X, 0, -1, basic._C_API_PREDICT_NORMAL, {})
    assert calls == [1]
    calls.clear()
    monkeypatch.setitem(basic._AUTO_CUDA_COST, "margin", 0.0)
    on_cuda = booster._auto_predict_on_cuda(X, 0, -1, basic._C_API_PREDICT_NORMAL, {})
    usable = basic._cuda_predictor_usable()
    assert calls == ([1, 2] if usable else [1])  # without a usable CUDA predictor, no device query
    assert on_cuda == usable


_CPU_ROUTED_PREDICT = r"""
import json
import os
import sys

import numpy as np

import falcata as lgb
from falcata import basic

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 5))
trained = lgb.train(
    {"objective": "regression", "device_type": "cpu", "verbose": -1, "num_threads": 2},
    lgb.Dataset(X, np.ascontiguousarray(X[:, 0])),
    5,
)
text = trained.model_to_string().replace("[device_type: cpu]", "[device_type: cuda]")
assert "[device_type: cuda]" in text
loaded = lgb.Booster(model_str=text)
assert loaded._trained_on_cuda()
ref = trained.predict(X, device="cpu")
ok = {}
# a small request: the CPU wins even against a warm GPU with the model packed
ok["small"] = bool(np.array_equal(loaded.predict(X), ref)) and loaded._last_auto_predict["device"] == "cpu"
# over the buffer cap, with a margin that would otherwise always choose the GPU
basic._AUTO_CUDA_COST["margin"] = 0.0
basic._AUTO_CUDA_COST["max_buffer_bytes"] = 1000
ok["cap"] = bool(np.array_equal(loaded.predict(X), ref)) and not loaded._last_auto_predict["fits"]
basic._AUTO_CUDA_COST["max_buffer_bytes"] = 256 << 20
# contributions, early stopping and non-numpy inputs never reach the estimate
loaded.predict(X[:20], pred_contrib=True)
loaded.predict(X, pred_early_stop=True)
loaded.predict(X[:20].tolist())
# the CPU predictor itself, explicitly and for a CPU-configured model: no GPU probe of an unset device_type
ok["cpu"] = bool(np.array_equal(loaded.predict(X, device="cpu"), ref)) and bool(np.array_equal(trained.predict(X), ref))
state = {"ok": ok, "probed": basic._CUDA_PREDICTOR_USABLE is not None, "libcuda": [], "nvidia_fds": []}
if sys.platform.startswith("linux"):
    with open("/proc/self/maps") as maps:
        state["libcuda"] = sorted({line.split()[-1] for line in maps if "libcuda.so" in line})
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith("/dev/nvidia"):
            state["nvidia_fds"].append(target)
print("PREDICT_STATE " + json.dumps(state))
"""


def test_cpu_routed_auto_predict_does_not_initialize_cuda():
    """predict(device=None) of a CUDA-configured model that ends on the CPU never initializes CUDA, and neither does
    the CPU predictor itself (device="cpu", a CPU-configured model): a process that forks children which use the GPU
    must stay CUDA-free (a fresh interpreter; on Linux libcuda must not be mapped and no /dev/nvidia* device open)."""
    result = subprocess.run(
        [sys.executable, "-c", _CPU_ROUTED_PREDICT],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("PREDICT_STATE ")]
    assert lines, result.stdout[-4000:] + result.stderr[-4000:]
    state = json.loads(lines[-1].removeprefix("PREDICT_STATE "))
    assert state["ok"] == {"small": True, "cap": True, "cpu": True}
    assert not state["probed"], "a CPU-routed predict(device=None) probed the CUDA predictor"
    assert not state["libcuda"], f"a CPU-routed predict(device=None) loaded the CUDA driver: {state['libcuda']}"
    assert not state["nvidia_fds"], f"a CPU-routed predict(device=None) opened NVIDIA devices: {state['nvidia_fds']}"


def test_auto_rule_unreadable_device_means_cpu(monkeypatch):
    """Free memory that cannot be read (-1: an invalid gpu_device_id, a CPU build) never picks the GPU."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(600, 5)).astype(np.float32)
    booster = lgb.train(_base(objective="regression"), lgb.Dataset(X, X[:, 0]), num_boost_round=4)
    monkeypatch.setitem(basic._AUTO_CUDA_COST, "margin", 0.0)
    assert not booster._auto_predict_on_cuda(X, 0, -1, basic._C_API_PREDICT_NORMAL, {"gpu_device_id": 4096})
    assert booster._last_auto_predict["free_device_bytes"] == -1.0
    assert not booster._last_auto_predict["fits"]


@requires_cuda
def test_auto_routing_follows_the_rules(zoo, monkeypatch):
    """device=None on a host array: the cost rule decides for a CUDA model; the result is the CPU predictor's bit
    for bit either way (the GPU adds in tree order unless deterministic=False is passed)."""
    booster, X = zoo["deep"]
    X = X.astype(np.float32)
    cpu = booster.predict(X, device="cpu", raw_score=True)
    booster.cuda_predict_stats = None
    booster.predict(X)  # trained with device_type=cpu: the CPU predictor, whatever the cost
    assert booster.cuda_predict_stats is None
    booster.params["device_type"] = "cuda"
    try:
        # the rule says CPU (the GPU estimate can never win against an infinite margin)
        monkeypatch.setitem(basic._AUTO_CUDA_COST, "margin", float("inf"))
        booster.cuda_predict_stats = None
        np.testing.assert_array_equal(booster.predict(X, raw_score=True), cpu)
        assert booster.cuda_predict_stats is None
        assert booster._last_auto_predict["device"] == "cpu"
        # the rule says GPU
        monkeypatch.setitem(basic._AUTO_CUDA_COST, "margin", 0.0)
        np.testing.assert_array_equal(booster.predict(X, raw_score=True), cpu)
        assert booster.cuda_predict_stats is not None
        assert booster.cuda_predict_stats["model_device_bytes"] > 0
        assert booster._last_auto_predict["device"] == "cuda"
        for kwargs in ({}, {"pred_leaf": True}):
            np.testing.assert_array_equal(
                booster.predict(X[:300], **kwargs), booster.predict(X[:300], device="cpu", **kwargs)
            )
        # feature contributions stay on the CPU under device=None (TreeSHAP's cost and scratch are outside the rule)
        booster.cuda_predict_stats = None
        np.testing.assert_array_equal(
            booster.predict(X[:300], pred_contrib=True), booster.predict(X[:300], device="cpu", pred_contrib=True)
        )
        assert booster.cuda_predict_stats is None
        # deterministic=False is the fast pairwise sum, on request
        np.testing.assert_array_equal(
            booster.predict(X, raw_score=True, deterministic=False), _pairwise_reference(booster, X)
        )
        # buffers over the cap: the CPU, however fast the GPU would be
        monkeypatch.setitem(basic._AUTO_CUDA_COST, "max_buffer_bytes", 1024)
        booster.cuda_predict_stats = None
        np.testing.assert_array_equal(booster.predict(X, raw_score=True), cpu)
        assert booster.cuda_predict_stats is None
        assert not booster._last_auto_predict["fits"]
        monkeypatch.setitem(basic._AUTO_CUDA_COST, "max_buffer_bytes", 256 << 20)
        # early stopping stays on the CPU predictor
        booster.cuda_predict_stats = None
        booster.predict(X, pred_early_stop=True)
        assert booster.cuda_predict_stats is None
        # other inputs (lists, 1-D, sparse) stay on the CPU predictor
        booster.cuda_predict_stats = None
        booster.predict(X[:5].tolist())
        assert booster.cuda_predict_stats is None
    finally:
        booster.params["device_type"] = "cpu"


@requires_cuda
def test_explicit_device_cuda_ignores_the_cost_rule(zoo, monkeypatch):
    """device='cuda' runs on the GPU for any size, in the default (pairwise) mode unless deterministic=True."""
    booster, X = zoo["regression"]
    monkeypatch.setitem(basic._AUTO_CUDA_COST, "margin", float("inf"))
    monkeypatch.setitem(basic._AUTO_CUDA_COST, "max_buffer_bytes", 0)
    for rows in (1, 7, len(X)):
        booster.cuda_predict_stats = None
        out = booster.predict(X[:rows], device="cuda", raw_score=True)
        assert booster.cuda_predict_stats is not None
        np.testing.assert_array_equal(out, _pairwise_reference(booster, X[:rows]))
        np.testing.assert_array_equal(
            booster.predict(X[:rows], device="cuda", raw_score=True, deterministic=True),
            booster.predict(X[:rows], device="cpu", raw_score=True),
        )


@requires_cuda
def test_auto_routing_of_device_arrays_ignores_the_cost_rule(zoo, monkeypatch):
    """A device array is predicted on its GPU whatever the cost rule says, in the default mode."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch without CUDA")
    booster, X = zoo["regression"]
    monkeypatch.setitem(basic._AUTO_CUDA_COST, "margin", float("inf"))
    t = torch.as_tensor(X[:3].astype(np.float32), device="cuda")
    out = booster.predict(t, raw_score=True)
    assert isinstance(out, torch.Tensor)
    np.testing.assert_array_equal(out.cpu().numpy(), _pairwise_reference(booster, X[:3].astype(np.float32)))


@requires_cuda
def test_auto_routing_cost_inputs_track_packing():
    """The packed-model flag the rule reads: set by a CUDA prediction, cleared when the model changes."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 4))
    booster = lgb.Booster(_base(objective="regression"), lgb.Dataset(X, X[:, 0], free_raw_data=False))
    for _ in range(3):
        booster.update()

    def packed():
        return _cost_inputs(booster)[2]

    assert packed() == 0.0
    booster.predict(X, device="cuda")
    assert packed() == 1.0
    booster.update()
    assert packed() == 0.0


@requires_cuda
def test_unreadable_device_falls_back_and_leaves_no_cuda_error(monkeypatch):
    """predict(device=None) with a gpu_device_id that does not exist predicts on the CPU; device='cuda' raises; and
    neither leaves a CUDA error behind for the next CUDA call of the process."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(800, 5)).astype(np.float32)
    y = X[:, 0] - X[:, 2]
    booster = lgb.train(_base("cuda", objective="regression"), lgb.Dataset(X, y), num_boost_round=5)
    cpu = booster.predict(X, device="cpu")
    monkeypatch.setitem(basic._AUTO_CUDA_COST, "margin", 0.0)
    booster.cuda_predict_stats = None
    np.testing.assert_array_equal(booster.predict(X, gpu_device_id=4096), cpu)
    assert booster.cuda_predict_stats is None
    assert booster._last_auto_predict["free_device_bytes"] == -1.0
    with pytest.raises(lgb.basic.FalcataError, match="CUDA"):
        booster.predict(X, device="cuda", gpu_device_id=4096)
    # the process's next CUDA work is unaffected: training, and the device chosen by the rule
    again = lgb.train(_base("cuda", objective="regression"), lgb.Dataset(X, y), num_boost_round=5)
    np.testing.assert_array_equal(again.predict(X), again.predict(X, device="cpu"))
    assert again.cuda_predict_stats is not None
    np.testing.assert_array_equal(booster.predict(X, device="cuda", deterministic=True), cpu)


@requires_cuda
def test_auto_estimate_matches_the_allocation(monkeypatch):
    """The rule's buffer bytes are the CUDA predictor's (one chunk: rows and result, plus one byte each) for
    multi-target, multiclass and leaf-index requests, and its output width is the result's for contributions."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(2000, 8)).astype(np.float32)
    targets = np.stack([X[:, k % 8] + 0.1 * k for k in range(16)], axis=1)
    vector = lgb.train(
        _base("cuda", objective="multi_regression", metric="multi_rmse", num_class=16, tree_mode="vector_leaf"),
        lgb.Dataset(X, label=targets),
        num_boost_round=6,
    )
    yk = np.digitize(X[:, 0], [-1.0, 0.0, 1.0]).astype(float)
    multi = lgb.train(_base("cuda", objective="multiclass", num_class=4), lgb.Dataset(X, yk), num_boost_round=6)
    monkeypatch.setitem(basic._AUTO_CUDA_COST, "margin", 0.0)
    for booster, kwargs in ((vector, {}), (multi, {}), (multi, {"raw_score": True}), (multi, {"pred_leaf": True})):
        booster.cuda_predict_stats = None
        out = booster.predict(X, **kwargs)
        assert booster.cuda_predict_stats is not None
        assert booster._last_auto_predict["buffer_bytes"] + 2 == booster.cuda_predict_stats["buffer_device_bytes"]
        assert booster._last_auto_predict["buffer_bytes"] == X.nbytes + out.nbytes
    assert vector._predict_output_width(basic._C_API_PREDICT_NORMAL, 0, -1) == 16
    contrib = multi.predict(X[:50], device="cuda", pred_contrib=True)
    assert multi._predict_output_width(basic._C_API_PREDICT_CONTRIB, 0, -1) * 50 == contrib.size


@requires_cuda
def test_model_changes_reach_the_device_copy():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 5))
    y = X[:, 0] - X[:, 1]
    ds = lgb.Dataset(X, y, free_raw_data=False)
    booster = lgb.Booster(_base(objective="regression"), ds)
    for _ in range(5):
        booster.update()
    _assert_same(booster, X, raw_score=True)
    booster.update()  # one more tree
    _assert_same(booster, X, raw_score=True)
    booster.set_leaf_output(0, 0, 123.25)
    _assert_same(booster, X, raw_score=True)
    booster.rollback_one_iter()
    _assert_same(booster, X, raw_score=True)
    booster.model_from_string(booster.model_to_string(num_iteration=2))
    _assert_same(booster, X, raw_score=True)


@requires_cuda
def test_cuda_trained_models():
    rng = np.random.default_rng(11)
    X = _features(rng, 3000, 20).astype(np.float32)
    y = np.nansum(X[:, :4], axis=1)
    for objective, label, extra in (
        ("regression", y, {}),
        ("multiclass", np.digitize(y, [-1, 0, 1]).astype(float), {"num_class": 4}),
    ):
        booster = lgb.train(
            _base("cuda", objective=objective, num_leaves=63, **extra), lgb.Dataset(X, label), num_boost_round=30
        )
        _assert_same(booster, X, raw_score=True)
        _assert_same(booster, X)
        _assert_same(booster, X, pred_leaf=True)
        _assert_same(booster, X[:500], pred_contrib=True)


@requires_cuda
def test_vector_leaf_models():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(2000, 6)).astype(np.float32)
    labels = np.stack([X[:, 0] + X[:, 1], X[:, 2] - X[:, 0], X[:, 3]], axis=1)
    params = _base("cuda", objective="multi_regression", metric="multi_rmse", num_class=3, tree_mode="vector_leaf")
    booster = lgb.train(params, lgb.Dataset(X, label=labels), num_boost_round=15)
    _assert_same(booster, X, raw_score=True)
    _assert_same(booster, X)
    _assert_same(booster, X, pred_leaf=True)
    for device in ("cpu", "cuda"):
        with pytest.raises(lgb.basic.FalcataError, match="vector-leaf"):
            booster.predict(X[:10], device=device, pred_contrib=True)


@requires_cuda
def test_fuzz_shapes():
    """Random model shapes (trees, leaves, depth, features, classes, NaN density) against random inputs."""
    rng = np.random.default_rng(424242)
    for case in range(12):
        n_features = int(rng.choice([1, 3, 17, 64, 300]))
        n_class = int(rng.choice([1, 1, 3]))
        params = _base(
            objective="multiclass" if n_class > 1 else "regression",
            num_leaves=int(rng.choice([2, 7, 31, 127])),
            max_depth=int(rng.choice([-1, 3, 8])),
            learning_rate=0.2,
            zero_as_missing=bool(rng.random() < 0.3),
            seed=case,
        )
        if n_class > 1:
            params["num_class"] = n_class
        n = 1500
        X = _features(rng, n, n_features, nan_frac=float(rng.choice([0.0, 0.05, 0.6])))
        signal = np.nansum(X[:, : min(3, n_features)], axis=1)
        label = signal if n_class == 1 else np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3])).astype(float)
        booster = lgb.train(params, lgb.Dataset(X, label), num_boost_round=int(rng.integers(1, 40)))
        test = _features(rng, 700, n_features, nan_frac=0.2)
        test = np.vstack([test, _probe_rows(booster, X, rng)])
        for dtype in (np.float32, np.float64):
            data = test.astype(dtype)
            _assert_same(booster, data, raw_score=True)
            _assert_same(booster, data, pred_leaf=True)
            _assert_same(booster, data[:200], pred_contrib=True)


@requires_cuda
def test_device_arrays_stay_on_the_device(zoo):
    cupy = pytest.importorskip("cupy")
    for name in ("regression", "multiclass", "binary"):
        booster, X = zoo[name]
        for dtype in (np.float32, np.float64):
            host = X.astype(dtype)
            dev = cupy.asarray(host)
            for kwargs in ({}, {"raw_score": True}, {"pred_leaf": True}, {"pred_contrib": True}):
                out = booster.predict(dev, deterministic=True, **kwargs)
                assert isinstance(out, cupy.ndarray)
                np.testing.assert_array_equal(cupy.asnumpy(out), booster.predict(host, device="cpu", **kwargs))
            ref = _pairwise_reference(booster, host)
            out = booster.predict(dev, raw_score=True)
            assert isinstance(out, cupy.ndarray)
            np.testing.assert_array_equal(cupy.asnumpy(out), ref)
            # a strided view (every other row, a column slice) is read in place
            view = dev[::2, :]
            np.testing.assert_array_equal(
                cupy.asnumpy(booster.predict(view, raw_score=True, deterministic=True)),
                booster.predict(np.ascontiguousarray(host[::2, :]), device="cpu", raw_score=True),
            )
            np.testing.assert_array_equal(cupy.asnumpy(booster.predict(view, raw_score=True)), ref[::2])
            fortran = cupy.asfortranarray(dev)
            np.testing.assert_array_equal(
                cupy.asnumpy(booster.predict(fortran, raw_score=True, deterministic=True)),
                booster.predict(host, device="cpu", raw_score=True),
            )
            np.testing.assert_array_equal(cupy.asnumpy(booster.predict(fortran, raw_score=True)), ref)
        with pytest.raises(ValueError, match="device='cpu'"):
            booster.predict(cupy.asarray(X), device="cpu")


@requires_cuda
def test_torch_tensors_stay_torch(zoo):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch without CUDA")
    booster, X = zoo["multiclass"]
    t = torch.as_tensor(X.astype(np.float32), device="cuda")
    out = booster.predict(t, raw_score=True, deterministic=True)
    assert isinstance(out, torch.Tensor)
    assert out.is_cuda
    assert out.dtype == torch.float64
    np.testing.assert_array_equal(
        out.cpu().numpy(), booster.predict(X.astype(np.float32), device="cpu", raw_score=True)
    )
    out = booster.predict(t, raw_score=True)
    assert isinstance(out, torch.Tensor)
    np.testing.assert_array_equal(out.cpu().numpy(), _pairwise_reference(booster, X.astype(np.float32)))


@requires_cuda
def test_predict_many_equals_one_by_one(zoo):
    """A batch of jobs (several models, shared and own inputs, iteration slices, linear and categorical models that
    take other paths) gives each job exactly its own prediction, in both modes, from host and device memory."""
    names = ["regression", "multiclass", "categorical", "linear_tree", "binary", "deep", "stumps", "rf"]
    boosters, inputs, starts, nums = [], [], [], []
    rng = np.random.default_rng(9)
    for name in names:
        booster, X = zoo[name]
        own = np.ascontiguousarray(X[: int(rng.integers(1, 600))], dtype=np.float32)
        for start, num in ((0, None), (3, 7), (0, 1), (2, -1)):
            boosters.append(booster)
            inputs.append(own)
            starts.append(start)
            nums.append(num)
    for raw in (True, False):
        for deterministic in (False, True):
            got = lgb.predict_many(boosters, inputs, starts, nums, raw_score=raw, deterministic=deterministic)
            assert len(got) == len(boosters)
            for b, x, st, nm, g in zip(boosters, inputs, starts, nums, got, strict=True):
                one = b.predict(
                    x, device="cuda", start_iteration=st, num_iteration=nm, raw_score=raw, deterministic=deterministic
                )
                assert g.shape == one.shape
                np.testing.assert_array_equal(g, one)
                if deterministic:
                    np.testing.assert_array_equal(
                        g, b.predict(x, device="cpu", start_iteration=st, num_iteration=nm, raw_score=raw)
                    )
    # one input for every job, Fortran order, float64
    booster, X = zoo["regression"]
    shared = np.asfortranarray(X[:333])
    got = lgb.predict_many(
        [booster, zoo["regression_zero_missing"][0], booster], shared, num_iteration=[5, None, -1], raw_score=True
    )
    np.testing.assert_array_equal(got[0], booster.predict(shared, device="cuda", num_iteration=5, raw_score=True))
    np.testing.assert_array_equal(got[2], booster.predict(shared, device="cuda", raw_score=True))
    cupy = pytest.importorskip("cupy")
    dev = [cupy.asarray(x) for x in inputs]
    got = lgb.predict_many(boosters, dev, starts, nums, raw_score=True)
    for b, x, st, nm, g in zip(boosters, inputs, starts, nums, got, strict=True):
        assert isinstance(g, cupy.ndarray)
        np.testing.assert_array_equal(
            cupy.asnumpy(g), b.predict(x, device="cuda", start_iteration=st, num_iteration=nm, raw_score=True)
        )


@requires_cuda
def test_predict_many_refuses_mixed_inputs(zoo):
    booster, X = zoo["regression"]
    assert lgb.predict_many([], X) == []
    with pytest.raises(TypeError, match="all float32 or all float64"):
        lgb.predict_many([booster, booster], [X.astype(np.float32), X.astype(np.float64)])
    with pytest.raises(ValueError, match="2 boosters, 3 inputs"):
        lgb.predict_many([booster, booster], [X, X, X])
    cupy = pytest.importorskip("cupy")
    with pytest.raises(TypeError, match="all numpy arrays or all device arrays"):
        lgb.predict_many([booster, booster], [X, cupy.asarray(X)])


_KERNEL_SCRIPT = r"""
import sys
import numpy as np
import falcata as lgb
sys.path.insert(0, {test_dir!r})
from test_cuda_predictor import _zoo, _pairwise_reference
for name, (booster, X) in _zoo().items():
    for dtype in (np.float32, np.float64):
        for order in ("C", "F"):
            data = np.asarray(X, dtype=dtype, order=order)
            requests = [{{"raw_score": True}}, {{}}, {{"pred_leaf": True}},
                        {{"raw_score": True, "start_iteration": 4, "num_iteration": 9}}]
            if not name.startswith("linear_tree"):
                requests.append({{"pred_contrib": True}})
            for kwargs in requests:
                cpu = booster.predict(data, device="cpu", **kwargs)
                gpu = booster.predict(data, device="cuda", deterministic=True, **kwargs)
                if not np.array_equal(cpu, gpu, equal_nan=True):
                    print("MISMATCH", name, dtype.__name__, order, kwargs, float(np.nanmax(np.abs(cpu - gpu))))
                    sys.exit(1)
            for start, num in ((0, -1), (4, 9), (1, 17)):
                ref = _pairwise_reference(booster, data, start, num)
                gpu = booster.predict(data, device="cuda", raw_score=True, start_iteration=start, num_iteration=num)
                if not np.array_equal(ref, gpu, equal_nan=True):
                    print("PAIRWISE MISMATCH", name, dtype.__name__, order, start, num,
                          float(np.nanmax(np.abs(ref - gpu))))
                    sys.exit(1)
print("OK")
"""


@requires_cuda
@pytest.mark.parametrize(
    "debug",
    [
        "predict_kernel=row",
        "predict_kernel=row,predict_ilp=1",
        "predict_kernel=row,predict_ilp=8",
        "predict_kernel=tile",
        "predict_kernel=tile,predict_ilp=1,predict_tile_rows=5,predict_tile_trees=7",
        "predict_kernel=tile,predict_ilp=4,predict_tile_rows=40",
        "predict_kernel=tile,predict_tile_rows=3,predict_tile_trees=1",
        "predict_chunk_rows=37",
        "predict_tile_rows=1",
        "predict_tile_rows=3,predict_wave_groups=1,predict_ilp=1",
        "predict_tile_rows=32,predict_wave_groups=8,predict_ilp=4",
        "predict_spans=1",
        "predict_spans=5,predict_wave_groups=2",
        "predict_target_blocks=100000,predict_ilp=2",
        "predict_stage=0",
        "predict_stage=1,predict_tile_rows=2",
        "predict_chunk_rows=37,predict_spans=3",
        "predict_compact=0",
        "predict_compact=0,predict_stage=0,predict_spans=4",
        "predict_group_lanes=1",
        "predict_group_lanes=2,predict_compact=0",
        "predict_group_lanes=2,predict_stage=1,predict_tile_rows=2,predict_out_chunk=3",
        "predict_group_lanes=4,predict_stage=0,predict_spans=3",
        "predict_group_lanes=4,predict_stage=1,predict_tile_rows=1",
        "predict_group_lanes=8,predict_tile_rows=3",
        "predict_wave_groups=64,predict_threads=64",
        "predict_group_lanes=1,predict_task_groups=2",
        "predict_group_lanes=1,predict_task_groups=8,predict_spans=3",
        "predict_group_lanes=1,predict_task_groups=4,predict_out_chunk=1,predict_compact=0",
        "predict_out_chunk=3",
    ],
)
def test_every_kernel_is_exact(debug, tmp_path):
    """Each raw-score kernel and shape, forced through FALCATA_DEBUG (read once per process): the deterministic mode
    equals the CPU predictor and the default mode the pairwise reference, bit for bit."""
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    script = tmp_path / "kernels.py"
    script.write_text(_KERNEL_SCRIPT.format(test_dir=os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, FALCATA_DEBUG=debug)
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout, result.stdout + result.stderr
