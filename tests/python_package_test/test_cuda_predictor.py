"""The native CUDA predictor equals the CPU predictor on every model kind and input layout.

Raw scores and leaf indices must be bit-identical: both predictors add the leaf values of each output in tree order
in double precision, route a float32 input with thresholds rounded down to float (exact for every float), and see
values with |x| <= 1e-35 as zero. Normal predictions are converted from the raw scores by the same host code
(Boosting::ConvertRawToPrediction), so they are bit-identical too. Every comparison below is exact.
"""

import itertools

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
    cpu = booster.predict(X, device="cpu", **kwargs)
    gpu = booster.predict(X, device="cuda", **kwargs)
    assert type(gpu) is type(cpu)
    assert gpu.shape == cpu.shape, (gpu.shape, cpu.shape)
    assert gpu.dtype == cpu.dtype, (gpu.dtype, cpu.dtype)
    np.testing.assert_array_equal(gpu, cpu)


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


@requires_cuda
def test_auto_routing_follows_the_rules(zoo):
    booster, X = zoo["regression"]
    booster.cuda_predict_stats = None
    booster.predict(X)  # trained with device_type=cpu: the CPU predictor
    assert booster.cuda_predict_stats is None
    booster.params["device_type"] = "cuda"
    try:
        booster.predict(X)
        assert booster.cuda_predict_stats is not None
        stats = booster.cuda_predict_stats
        assert stats["model_device_bytes"] > 0
        assert stats["num_chunks"] >= 1
        booster.cuda_predict_stats = None
        booster.predict(X, pred_early_stop=True)  # early stopping stays on the CPU predictor
        assert booster.cuda_predict_stats is None
    finally:
        booster.params["device_type"] = "cpu"


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
                out = booster.predict(dev, **kwargs)
                assert isinstance(out, cupy.ndarray)
                np.testing.assert_array_equal(cupy.asnumpy(out), booster.predict(host, device="cpu", **kwargs))
            # a strided view (every other row, a column slice) is read in place
            view = dev[::2, :]
            np.testing.assert_array_equal(
                cupy.asnumpy(booster.predict(view, raw_score=True)),
                booster.predict(np.ascontiguousarray(host[::2, :]), device="cpu", raw_score=True),
            )
            fortran = cupy.asfortranarray(dev)
            np.testing.assert_array_equal(
                cupy.asnumpy(booster.predict(fortran, raw_score=True)),
                booster.predict(host, device="cpu", raw_score=True),
            )
        with pytest.raises(ValueError, match="device='cpu'"):
            booster.predict(cupy.asarray(X), device="cpu")


@requires_cuda
def test_torch_tensors_stay_torch(zoo):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch without CUDA")
    booster, X = zoo["multiclass"]
    t = torch.as_tensor(X.astype(np.float32), device="cuda")
    out = booster.predict(t, raw_score=True)
    assert isinstance(out, torch.Tensor)
    assert out.is_cuda
    assert out.dtype == torch.float64
    np.testing.assert_array_equal(
        out.cpu().numpy(), booster.predict(X.astype(np.float32), device="cpu", raw_score=True)
    )


_KERNEL_SCRIPT = r"""
import sys
import numpy as np
import falcata as lgb
sys.path.insert(0, {test_dir!r})
from test_cuda_predictor import _zoo
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
                gpu = booster.predict(data, device="cuda", **kwargs)
                if not np.array_equal(cpu, gpu, equal_nan=True):
                    print("MISMATCH", name, dtype.__name__, order, kwargs, float(np.nanmax(np.abs(cpu - gpu))))
                    sys.exit(1)
print("OK")
"""


@requires_cuda
@pytest.mark.parametrize(
    "debug",
    [
        "predict_kernel=row",
        "predict_kernel=ilp,predict_ilp=2",
        "predict_kernel=ilp,predict_ilp=8",
        "predict_kernel=tile",
        "predict_kernel=tile,predict_tile_rows=3",
        "predict_kernel=tile,predict_tile_rows=64",
        "predict_chunk_rows=37",
    ],
)
def test_every_kernel_is_exact(debug, tmp_path):
    """Each raw-score kernel, forced through FALCATA_DEBUG (read once per process), equals the CPU predictor."""
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
