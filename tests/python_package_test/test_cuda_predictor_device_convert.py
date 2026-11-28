"""Device-side ConvertOutput kernels, spill-free span combine, and power-of-2 shift indexing."""

import ctypes
import functools

import numpy as np
import pytest

import falcata as lgb
from falcata import basic

_NO_CUDA_PREDICTOR = "the CUDA predictor needs a CUDA build and a GPU"


@functools.lru_cache(maxsize=1)
def _cudart():
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12", "/usr/local/cuda-13.2/lib64/libcudart.so"):
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
