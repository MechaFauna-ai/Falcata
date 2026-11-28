"""Device-side ConvertOutput kernels, spill-free span combine, and power-of-2 shift indexing."""

import ctypes
import ctypes.util
import functools
import os

import numpy as np
import pytest

import falcata as lgb
from falcata import basic

_NO_CUDA_PREDICTOR = "the CUDA predictor needs a CUDA build and a GPU"


@functools.lru_cache(maxsize=1)
def _cudart():
    candidates = []
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    candidates.extend(["libcudart.so", "libcudart.so.13", "libcudart.so.12"])
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        base = os.environ.get(env_var)
        if base:
            candidates.append(os.path.join(base, "lib64", "libcudart.so"))
    candidates.append("/usr/local/cuda/lib64/libcudart.so")
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            lib.cudaMalloc.restype = ctypes.c_int
            lib.cudaFree.argtypes = [ctypes.c_void_p]
            lib.cudaFree.restype = ctypes.c_int
            lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            lib.cudaMemcpy.restype = ctypes.c_int
            lib.cudaDeviceSynchronize.argtypes = []
            lib.cudaDeviceSynchronize.restype = ctypes.c_int
            return lib
        except OSError:
            continue
    pytest.skip("libcudart not found")


class _DeviceBuffer:
    """Minimal RAII device buffer via libcudart for testing data_on_device=1 without CuPy/Torch."""

    def __init__(self, nbytes: int):
        self._lib = _cudart()
        self.ptr = ctypes.c_void_p(0)
        rc = self._lib.cudaMalloc(ctypes.byref(self.ptr), ctypes.c_size_t(max(nbytes, 1)))
        assert rc == 0, f"cudaMalloc failed: {rc}"

    def copy_from_host(self, arr: np.ndarray) -> None:
        rc = self._lib.cudaMemcpy(self.ptr, ctypes.c_void_p(arr.ctypes.data), ctypes.c_size_t(arr.nbytes), 1)
        assert rc == 0, f"cudaMemcpy H2D failed: {rc}"

    def copy_to_host(self, arr: np.ndarray) -> None:
        rc = self._lib.cudaMemcpy(ctypes.c_void_p(arr.ctypes.data), self.ptr, ctypes.c_size_t(arr.nbytes), 2)
        assert rc == 0, f"cudaMemcpy D2H failed: {rc}"

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        if self.ptr.value:
            self._lib.cudaFree(self.ptr)
            self.ptr = ctypes.c_void_p(0)


def _predict_on_device_capi(
    booster: lgb.Booster,
    arr: np.ndarray,
    *,
    raw_score: bool = False,
    deterministic: bool = False,
    start_iteration: int = 0,
    num_iteration: int = -1,
) -> np.ndarray:
    """Call FLC_BoosterPredictForMatCUDA with data_on_device=1 using device pointers."""
    arr = np.ascontiguousarray(arr)
    nrow, ncol = arr.shape
    dtype_code = basic._C_API_DTYPE_FLOAT32 if arr.dtype == np.float32 else basic._C_API_DTYPE_FLOAT64
    predict_type = basic._C_API_PREDICT_RAW_SCORE if raw_score else basic._C_API_PREDICT_NORMAL
    width = booster._predict_output_width(predict_type, start_iteration, num_iteration)
    out_shape = (nrow, width) if width != 1 else (nrow,)
    out_host = np.empty(out_shape, dtype=np.float64)
    out_len = ctypes.c_int64(0)
    stats = (ctypes.c_int64 * 4)()
    with _DeviceBuffer(arr.nbytes) as d_in, _DeviceBuffer(out_host.nbytes) as d_out:
        d_in.copy_from_host(arr)
        basic._safe_call(
            basic._LIB.FLC_BoosterPredictForMatCUDA(
                booster._handle,
                d_in.ptr,
                ctypes.c_int(dtype_code),
                ctypes.c_int64(nrow),
                ctypes.c_int32(ncol),
                ctypes.c_int64(ncol),
                ctypes.c_int64(1),
                ctypes.c_int(1),
                ctypes.c_void_p(0),
                ctypes.c_int(predict_type),
                ctypes.c_int(start_iteration),
                ctypes.c_int(num_iteration),
                basic._EMPTY_C_STR,
                ctypes.c_int(1 if deterministic else 0),
                ctypes.byref(out_len),
                d_out.ptr,
                stats,
            )
        )
        assert out_len.value == nrow * width
        d_out.copy_to_host(out_host)
    return out_host


def _predict_many_on_device_capi(
    boosters: list[lgb.Booster],
    arrays: list[np.ndarray],
    *,
    raw_score: bool = False,
    deterministic: bool = False,
) -> list[np.ndarray]:
    """Call FLC_BoostersPredictForMatCUDA with data_on_device=1 using device pointers."""
    n = len(boosters)
    arrays = [np.ascontiguousarray(a) for a in arrays]
    dtype_code = basic._C_API_DTYPE_FLOAT32 if arrays[0].dtype == np.float32 else basic._C_API_DTYPE_FLOAT64
    predict_type = basic._C_API_PREDICT_RAW_SCORE if raw_score else basic._C_API_PREDICT_NORMAL
    d_ins = [_DeviceBuffer(a.nbytes) for a in arrays]
    outs_host = []
    d_outs = []
    try:
        for d_in, a in zip(d_ins, arrays, strict=True):
            d_in.copy_from_host(a)
        for b, a in zip(boosters, arrays, strict=True):
            w = b._predict_output_width(predict_type, 0, -1)
            shape = (a.shape[0], w) if w != 1 else (a.shape[0],)
            h = np.empty(shape, dtype=np.float64)
            outs_host.append(h)
            d_outs.append(_DeviceBuffer(h.nbytes))
        c_handles = (ctypes.c_void_p * n)(*[b._handle.value for b in boosters])
        c_data = (ctypes.c_void_p * n)(*[d.ptr.value for d in d_ins])
        c_nrow = (ctypes.c_int32 * n)(*[a.shape[0] for a in arrays])
        c_ncol = (ctypes.c_int32 * n)(*[a.shape[1] for a in arrays])
        c_rs = (ctypes.c_int32 * n)(*[a.shape[1] for a in arrays])
        c_cs = (ctypes.c_int32 * n)(*([1] * n))
        c_start = (ctypes.c_int * n)(*([0] * n))
        c_num = (ctypes.c_int * n)(*([-1] * n))
        c_out = (ctypes.c_void_p * n)(*[d.ptr.value for d in d_outs])
        c_len = (ctypes.c_int64 * n)()
        basic._safe_call(
            basic._LIB.FLC_BoostersPredictForMatCUDA(
                ctypes.c_int(n),
                c_handles,
                c_data,
                ctypes.c_int(dtype_code),
                c_nrow,
                c_ncol,
                c_rs,
                c_cs,
                ctypes.c_int(1),
                ctypes.c_void_p(0),
                ctypes.c_int(predict_type),
                c_start,
                c_num,
                basic._EMPTY_C_STR,
                ctypes.c_int(1 if deterministic else 0),
                c_out,
                c_len,
            )
        )
        for d_out, h in zip(d_outs, outs_host, strict=True):
            d_out.copy_to_host(h)
        return outs_host
    finally:
        for buf in d_ins + d_outs:
            buf.__exit__(None, None, None)


@pytest.fixture
def cuda_predictor():
    if not basic._cuda_predictor_usable():
        pytest.skip(_NO_CUDA_PREDICTOR)


requires_cuda = pytest.mark.usefixtures("cuda_predictor")


def _base(**extra):
    params = {
        "verbose": -1,
        "device_type": "cpu",
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "learning_rate": 0.25,
        "seed": 42,
        "deterministic": True,
        "num_threads": 4,
    }
    params.update(extra)
    return params


def _ref_convert(raw, kind, *, sigmoid=1.0, avg_div=1):
    x = np.array(raw, dtype=np.float64, copy=True)
    if avg_div > 1:
        x = x / float(avg_div)
    if kind == "none":
        return x
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-sigmoid * x))
    if kind == "softmax":
        m = np.max(x, axis=1, keepdims=True)
        e = np.exp(x - m)
        return e / np.sum(e, axis=1, keepdims=True)
    if kind == "exp":
        return np.exp(x)
    if kind == "reg_sqrt":
        return np.sign(x) * x * x
    if kind == "softplus":
        return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))
    raise ValueError(kind)


@requires_cuda
@pytest.mark.parametrize(
    ("objective", "extra", "kind"),
    [
        ("binary", {}, "sigmoid"),
        ("binary", {"sigmoid": 2.5}, "sigmoid"),
        ("multiclass", {"num_class": 3}, "softmax"),
        ("multiclass", {"num_class": 7}, "softmax"),
        ("multiclass", {"num_class": 12}, "softmax"),
        ("multiclass", {"num_class": 24}, "softmax"),
        ("multiclass", {"num_class": 40}, "softmax"),
        ("multiclassova", {"num_class": 7, "sigmoid": 1.5}, "sigmoid"),
        ("poisson", {}, "exp"),
        ("gamma", {}, "exp"),
        ("tweedie", {"tweedie_variance_power": 1.5}, "exp"),
        ("regression", {}, "none"),
        ("regression", {"reg_sqrt": True}, "reg_sqrt"),
        ("cross_entropy", {}, "sigmoid"),
        ("cross_entropy_lambda", {}, "softplus"),
    ],
)
def test_device_convert_accuracy_host_and_device(objective, extra, kind):
    """Device ConvertOutput matches CPU predict (<= 1e-14) on host and device pointers and in predict_many."""
    rng = np.random.default_rng(1234 + extra.get("num_class", 0))
    n_class = extra.get("num_class", 1)
    n_train = max(400, n_class * 25)
    X = rng.normal(size=(n_train, 10))
    X[rng.random(X.shape) < 0.05] = np.nan
    signal = np.nan_to_num(X[:, 0]) - 0.7 * np.nan_to_num(X[:, 1]) + 0.5 * np.nan_to_num(X[:, 2])
    if objective in ("binary", "cross_entropy", "cross_entropy_lambda"):
        y = (signal > 0).astype(float)
    elif objective in ("multiclass", "multiclassova"):
        # ensure every class appears in the training labels
        y = np.arange(n_train, dtype=float) % n_class
        y = (y + (signal > 0).astype(int)) % n_class
    elif objective in ("poisson", "gamma", "tweedie"):
        y = np.abs(signal) + 0.5
    else:
        y = signal

    booster = lgb.train(_base(objective=objective, **extra), lgb.Dataset(X, y), num_boost_round=12)
    test = rng.normal(size=(257, 10)).astype(np.float32)
    test[rng.random(test.shape) < 0.05] = np.nan

    cpu = booster.predict(test, device="cpu")
    gpu_host = booster.predict(test, device="cuda")
    gpu_raw = booster.predict(test, device="cuda", raw_score=True)
    ref_from_raw = _ref_convert(gpu_raw, kind, sigmoid=extra.get("sigmoid", 1.0))

    # Conversion accuracy itself: <= 1e-14 relative & absolute
    np.testing.assert_allclose(gpu_host, ref_from_raw, rtol=1e-14, atol=1e-14)
    # Agreement with CPU predict (raw pairwise rounding + conversion): <= 1e-14
    scale = 1.0 + float(np.max(np.abs(cpu)))
    np.testing.assert_allclose(gpu_host, cpu, rtol=1e-14, atol=1e-14 * scale)

    # Deterministic mode stays bit-identical to CPU
    gpu_det = booster.predict(test, device="cuda", deterministic=True)
    np.testing.assert_array_equal(gpu_det, cpu)

    # predict_many on host agrees bit-for-bit with single predict(device="cuda")
    many_host = lgb.predict_many([booster, booster], [test, test[:17]])
    np.testing.assert_array_equal(many_host[0], gpu_host)
    np.testing.assert_array_equal(many_host[1], booster.predict(test[:17], device="cuda"))

    # Small odd batch sizes (non-power-of-2 row counts: 1, 3, 7)
    for nr in (1, 3, 7):
        sub = booster.predict(test[:nr], device="cuda")
        np.testing.assert_allclose(sub, cpu[:nr], rtol=1e-14, atol=1e-14 * scale)

    # Device input (data_on_device=1 via C API device pointers) matches host CUDA predict bit-for-bit
    gpu_dev = _predict_on_device_capi(booster, test)
    np.testing.assert_array_equal(gpu_dev, gpu_host)
    gpu_dev_det = _predict_on_device_capi(booster, test, deterministic=True)
    np.testing.assert_array_equal(gpu_dev_det, cpu)

    many_dev = _predict_many_on_device_capi([booster, booster], [test, test[:17]])
    np.testing.assert_array_equal(many_dev[0], gpu_host)
    np.testing.assert_array_equal(many_dev[1], many_host[1])


@requires_cuda
@pytest.mark.parametrize(
    ("objective", "extra"),
    [
        ("regression", {}),
        ("binary", {}),
        ("multiclass", {"num_class": 5}),
    ],
)
def test_device_convert_rf_average_output(objective, extra):
    """Random forest average_output divides by num_iteration before objective conversion on device."""
    rng = np.random.default_rng(99)
    n_class = extra.get("num_class", 1)
    X = rng.normal(size=(500, 8))
    if objective == "binary":
        y = (X[:, 0] > 0).astype(float)
    elif objective == "multiclass":
        y = (np.arange(500) % n_class).astype(float)
    else:
        y = X[:, 0] - X[:, 1]
    params = _base(
        objective=objective,
        boosting="rf",
        bagging_fraction=0.6,
        bagging_freq=1,
        **extra,
    )
    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=10)
    test = X[:65].astype(np.float32)

    for start, num in ((0, -1), (2, 5), (0, 1)):
        cpu = booster.predict(test, device="cpu", start_iteration=start, num_iteration=num)
        gpu = booster.predict(test, device="cuda", start_iteration=start, num_iteration=num)
        gpu_det = booster.predict(test, device="cuda", deterministic=True, start_iteration=start, num_iteration=num)
        np.testing.assert_array_equal(gpu_det, cpu)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-14, atol=1e-14)
        dev_out = _predict_on_device_capi(booster, test, start_iteration=start, num_iteration=num)
        np.testing.assert_array_equal(dev_out, gpu)


@requires_cuda
def test_pairwise_combine_multispans_bit_identical():
    """Multi-span PairwiseCombineKernel and PairwiseJobsCombineKernel (shared-memory stack) are bit-identical."""
    rng = np.random.default_rng(777)
    X = rng.normal(size=(1000, 12)).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] - X[:, 2]
    # 400 rounds on small batch (1..9 rows) forces many spans -> exercises PairwiseCombineKernel
    booster = lgb.train(_base(objective="regression", num_leaves=15), lgb.Dataset(X, y), num_boost_round=400)
    sub = X[:9]
    per_it = [
        booster.predict(sub, device="cpu", raw_score=True, start_iteration=it, num_iteration=1) for it in range(400)
    ]
    groups = []
    for g in range(0, 400, 8):
        acc = np.zeros(len(sub), dtype=np.float64)
        for v in per_it[g : g + 8]:
            acc = acc + v
        groups.append(acc)
    while len(groups) > 1:
        if len(groups) % 2:
            groups.append(np.zeros(len(sub), dtype=np.float64))
        groups = [groups[i] + groups[i + 1] for i in range(0, len(groups), 2)]
    ref = groups[0]

    for nr in (1, 3, 7, 9):
        gpu = booster.predict(sub[:nr], device="cuda", raw_score=True)
        np.testing.assert_array_equal(gpu, ref[:nr])
        dev = _predict_on_device_capi(booster, sub[:nr], raw_score=True)
        np.testing.assert_array_equal(dev, ref[:nr])

    many = lgb.predict_many([booster, booster, booster], [sub[:1], sub[:5], sub], raw_score=True)
    np.testing.assert_array_equal(many[0], ref[:1])
    np.testing.assert_array_equal(many[1], ref[:5])
    np.testing.assert_array_equal(many[2], ref)

    many_dev = _predict_many_on_device_capi([booster, booster, booster], [sub[:1], sub[:5], sub], raw_score=True)
    np.testing.assert_array_equal(many_dev[0], ref[:1])
    np.testing.assert_array_equal(many_dev[1], ref[:5])
    np.testing.assert_array_equal(many_dev[2], ref)


@requires_cuda
@pytest.mark.parametrize("num_class", [2, 4, 5, 8, 9, 16, 17, 32, 33])
def test_device_softmax_boundary_class_counts(num_class):
    """Softmax subwarp boundaries (W=4/8/16/32) and >32 streaming kernel match CPU <= 1e-14."""
    rng = np.random.default_rng(5000 + num_class)
    n_train = max(300, num_class * 15)
    X = rng.normal(size=(n_train, 8)).astype(np.float32)
    y = (np.arange(n_train) % num_class).astype(float)
    booster = lgb.train(_base(objective="multiclass", num_class=num_class), lgb.Dataset(X, y), num_boost_round=6)
    test = rng.normal(size=(97, 8)).astype(np.float32)

    cpu = booster.predict(test, device="cpu")
    gpu_host = booster.predict(test, device="cuda")
    gpu_dev = _predict_on_device_capi(booster, test)
    gpu_raw = booster.predict(test, device="cuda", raw_score=True)
    ref = _ref_convert(gpu_raw, "softmax")

    np.testing.assert_allclose(gpu_host, ref, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(gpu_host, cpu, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.sum(gpu_host, axis=1), 1.0, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(gpu_dev, gpu_host)


@requires_cuda
def test_predict_many_mixed_objectives_host_and_device():
    """Single predict_many call mixing binary, multiclass (small & large), regression, reg_sqrt, and poisson."""
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(600, 10)).astype(np.float32)
    sig = X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2]

    b_bin = lgb.train(_base(objective="binary"), lgb.Dataset(X, (sig > 0).astype(float)), 10)
    b_mc7 = lgb.train(
        _base(objective="multiclass", num_class=7),
        lgb.Dataset(X, (np.arange(len(X)) % 7).astype(float)),
        8,
    )
    b_mc35 = lgb.train(
        _base(objective="multiclass", num_class=35),
        lgb.Dataset(X, (np.arange(len(X)) % 35).astype(float)),
        5,
    )
    b_reg = lgb.train(_base(objective="regression"), lgb.Dataset(X, sig), 10)
    b_sqrt = lgb.train(_base(objective="regression", reg_sqrt=True), lgb.Dataset(X, sig), 10)
    b_pois = lgb.train(_base(objective="poisson"), lgb.Dataset(X, np.abs(sig) + 0.5), 10)

    boosters = [b_bin, b_mc7, b_mc35, b_reg, b_sqrt, b_pois]
    test_a = rng.normal(size=(131, 10)).astype(np.float32)
    test_b = rng.normal(size=(43, 10)).astype(np.float32)
    inputs = [test_a, test_b, test_a, test_b, test_a, test_b]

    expected = [b.predict(x, device="cuda") for b, x in zip(boosters, inputs, strict=True)]
    many_host = lgb.predict_many(boosters, inputs)
    many_dev = _predict_many_on_device_capi(boosters, inputs)

    for b, x, exp, got_h, got_d in zip(boosters, inputs, expected, many_host, many_dev, strict=True):
        np.testing.assert_array_equal(got_h, exp)
        np.testing.assert_array_equal(got_d, exp)
        cpu = b.predict(x, device="cpu")
        scale = 1.0 + float(np.max(np.abs(cpu)))
        np.testing.assert_allclose(got_h, cpu, rtol=1e-14, atol=1e-14 * scale)


@requires_cuda
def test_device_convert_extreme_logits():
    """Extreme raw scores (±800) stay finite and match CPU for binary, multiclass, softplus, and reg_sqrt."""
    rng = np.random.default_rng(800)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    y_bin = (X[:, 0] > 0).astype(float)
    y_mc = (np.arange(200) % 5).astype(float)
    test = rng.normal(size=(64, 4)).astype(np.float32)

    # Binary, cross_entropy_lambda, and reg_sqrt stumps with leaf outputs forced to ±800
    for objective, extra in (
        ("binary", {}),
        ("cross_entropy_lambda", {}),
        ("regression", {"reg_sqrt": True}),
    ):
        y = y_bin if objective != "regression" else X[:, 0]
        b = lgb.train(_base(objective=objective, num_leaves=2, **extra), lgb.Dataset(X, y), num_boost_round=1)
        b.set_leaf_output(0, 0, 800.0)
        b.set_leaf_output(0, 1, -800.0)
        cpu = b.predict(test, device="cpu")
        gpu = b.predict(test, device="cuda")
        gpu_dev = _predict_on_device_capi(b, test)
        assert np.all(np.isfinite(gpu))
        np.testing.assert_allclose(gpu, cpu, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(gpu_dev, gpu)

    # Multiclass (5-class) with extreme shifted logits (+800, +799, -800, ...) testing max-shift stability
    b_mc = lgb.train(
        _base(objective="multiclass", num_class=5, num_leaves=2),
        lgb.Dataset(X, y_mc),
        num_boost_round=1,
    )
    vals_a = [800.0, 799.0, 798.5, -800.0, -800.0]
    vals_b = [-800.0, -799.0, -798.0, -797.5, -800.0]
    for k in range(5):
        b_mc.set_leaf_output(k, 0, vals_a[k])
        b_mc.set_leaf_output(k, 1, vals_b[k])
    cpu_mc = b_mc.predict(test, device="cpu")
    gpu_mc = b_mc.predict(test, device="cuda")
    gpu_mc_dev = _predict_on_device_capi(b_mc, test)
    assert np.all(np.isfinite(gpu_mc))
    np.testing.assert_allclose(gpu_mc, cpu_mc, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.sum(gpu_mc, axis=1), 1.0, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(gpu_mc_dev, gpu_mc)


@requires_cuda
def test_device_convert_fortran_order():
    """Fortran-contiguous inputs match C-contiguous CUDA predictions bit-for-bit across objectives."""
    rng = np.random.default_rng(314)
    X = rng.normal(size=(400, 8))
    sig = X[:, 0] - 0.5 * X[:, 1]
    models = [
        lgb.train(_base(objective="binary"), lgb.Dataset(X, (sig > 0).astype(float)), 10),
        lgb.train(
            _base(objective="multiclass", num_class=6),
            lgb.Dataset(X, (np.arange(len(X)) % 6).astype(float)),
            8,
        ),
        lgb.train(_base(objective="regression", reg_sqrt=True), lgb.Dataset(X, sig), 10),
        lgb.train(_base(objective="poisson"), lgb.Dataset(X, np.abs(sig) + 0.5), 10),
    ]
    for dtype in (np.float32, np.float64):
        xc = np.ascontiguousarray(X[:113], dtype=dtype)
        xf = np.asfortranarray(xc)
        for b in models:
            np.testing.assert_array_equal(b.predict(xf, device="cuda"), b.predict(xc, device="cuda"))


@requires_cuda
def test_device_convert_lambdarank():
    """lambdarank uses identity (kNone) conversion with zero host round-trip."""
    rng = np.random.default_rng(2718)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    y = np.clip(np.digitize(X[:, 0] - X[:, 1], [-1.0, 0.0, 1.0]), 0, 3).astype(float)
    booster = lgb.train(_base(objective="lambdarank"), lgb.Dataset(X, y, group=[100] * 4), num_boost_round=12)
    test = rng.normal(size=(120, 8)).astype(np.float32)

    cpu = booster.predict(test, device="cpu")
    gpu_host = booster.predict(test, device="cuda")
    gpu_det = booster.predict(test, device="cuda", deterministic=True)
    gpu_dev = _predict_on_device_capi(booster, test)
    many_dev = _predict_many_on_device_capi([booster, booster], [test, test[:19]])

    np.testing.assert_array_equal(gpu_det, cpu)
    np.testing.assert_allclose(gpu_host, cpu, rtol=1e-14, atol=1e-14 * (1.0 + float(np.max(np.abs(cpu)))))
    np.testing.assert_array_equal(gpu_dev, gpu_host)
    np.testing.assert_array_equal(many_dev[0], gpu_host)
    np.testing.assert_array_equal(many_dev[1], booster.predict(test[:19], device="cuda"))
