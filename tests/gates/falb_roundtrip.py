"""FALB binary-format gate: round-trip fidelity + untrusted-loader robustness.

Drives the C API directly through ctypes against the freshly built CPU
lib_falcata.so, so M1 is verifiable before M2's Python plumbing exists.
"""
import ctypes
import struct
import numpy as np
import sys

LIB = ctypes.cdll.LoadLibrary("/home/felixjk/Documents/lightgbm-fork/lib_falcata.so")
LIB.FLC_GetLastError.restype = ctypes.c_char_p


def check(ret):
    if ret != 0:
        raise RuntimeError(LIB.FLC_GetLastError().decode())


def make_data(n=4000, m=12, cat_cols=(), seed=0, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m))
    for c in cat_cols:
        X[:, c] = rng.integers(0, 40, size=n)
    if nan_frac:
        X[rng.random((n, m)) < nan_frac] = np.nan
    w = rng.standard_normal(m)
    y = (X[:, [i for i in range(m) if i not in cat_cols]] @
         w[[i for i in range(m) if i not in cat_cols]])
    y = np.nan_to_num(y) + 0.3 * rng.standard_normal(n)
    return np.ascontiguousarray(X), y


def train(X, y, params):
    n, m = X.shape
    Xf = np.ascontiguousarray(X, dtype=np.float64)
    ds = ctypes.c_void_p()
    check(LIB.FLC_DatasetCreateFromMat(
        Xf.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1),
        ctypes.c_int32(n), ctypes.c_int32(m), ctypes.c_int(1),
        params.encode(), ctypes.c_void_p(), ctypes.byref(ds)))
    yf = np.ascontiguousarray(y, dtype=np.float32)
    check(LIB.FLC_DatasetSetField(ds, b"label", yf.ctypes.data_as(ctypes.c_void_p),
                                  ctypes.c_int(n), ctypes.c_int(0)))
    booster = ctypes.c_void_p()
    check(LIB.FLC_BoosterCreate(ds, params.encode(), ctypes.byref(booster)))
    fin = ctypes.c_int(0)
    for _ in range(30):
        check(LIB.FLC_BoosterUpdateOneIter(booster, ctypes.byref(fin)))
        if fin.value:
            break
    return booster, ds


def save_text(booster):
    n = ctypes.c_int64(0)
    LIB.FLC_BoosterSaveModelToString(booster, 0, -1, 0, ctypes.c_int64(0),
                                     ctypes.byref(n), ctypes.c_char_p(None))
    buf = ctypes.create_string_buffer(n.value)
    check(LIB.FLC_BoosterSaveModelToString(booster, 0, -1, 0, n, ctypes.byref(n), buf))
    return buf.value.decode()


def save_falb(booster, stats=1, diag=1, f32=0, level=6):
    args = (ctypes.c_int(stats), ctypes.c_int(diag), ctypes.c_int(f32),
            ctypes.c_int(level))
    n = ctypes.c_int64(0)
    LIB.FLC_BoosterSaveModelToBinary(booster, 0, -1, 0, *args, ctypes.c_int64(0),
                                     ctypes.byref(n), ctypes.c_char_p(None))
    buf = ctypes.create_string_buffer(n.value)
    check(LIB.FLC_BoosterSaveModelToBinary(booster, 0, -1, 0, *args, n,
                                           ctypes.byref(n), buf))
    return buf.raw[:n.value]


def load_falb(blob):
    out = ctypes.c_void_p()
    it = ctypes.c_int(0)
    check(LIB.FLC_BoosterCreateFromBinary(blob, ctypes.c_int64(len(blob)),
                                          ctypes.byref(it), ctypes.byref(out)))
    return out


def load_text(text):
    out = ctypes.c_void_p()
    it = ctypes.c_int(0)
    check(LIB.FLC_BoosterLoadModelFromString(text.encode(), ctypes.byref(it),
                                             ctypes.byref(out)))
    return out


def predict(booster, X):
    n, m = X.shape
    Xf = np.ascontiguousarray(X, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    out_len = ctypes.c_int64(0)
    check(LIB.FLC_BoosterPredictForMat(
        booster, Xf.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1),
        ctypes.c_int32(n), ctypes.c_int32(m), ctypes.c_int(1),
        ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(-1), b"",
        ctypes.byref(out_len), out.ctypes.data_as(ctypes.c_void_p)))
    return out


CASES = [
    ("plain", dict(), "objective=regression num_leaves=31 max_bin=255 verbose=-1"),
    ("missing", dict(nan_frac=0.15),
     "objective=regression num_leaves=31 max_bin=255 verbose=-1"),
    ("categorical", dict(cat_cols=(2, 5)),
     "objective=regression num_leaves=31 max_bin=255 categorical_feature=2,5 verbose=-1"),
    ("wide-bins", dict(),
     "objective=regression num_leaves=63 max_bin=400 verbose=-1"),
    ("deep", dict(),
     "objective=regression num_leaves=255 max_bin=255 min_data_in_leaf=1 verbose=-1"),
    ("binary", dict(), "objective=binary num_leaves=31 max_bin=255 verbose=-1"),
]

fails = 0
for name, kw, params in CASES:
    X, y = make_data(seed=abs(hash(name)) % 1000, **kw)
    if "binary" in params:
        y = (y > np.median(y)).astype(np.float64)
    booster, ds = train(X, y, params)
    text = save_text(booster)
    blob = save_falb(booster)
    b2 = load_falb(blob)
    text2 = save_text(b2)
    p1, p2 = predict(booster, X), predict(b2, X)

    # raw (uncompressed, mmap-able) must round-trip identically too
    raw = save_falb(booster, level=0)
    text_raw = save_text(load_falb(raw))
    ok_raw = text_raw == text
    # f32 leaves are opt-in lossy: structure exact, predictions close
    f32blob = save_falb(booster, 0, 0, f32=1)
    p_f32 = predict(load_falb(f32blob), X)
    f32_rel = float(np.max(np.abs(p_f32 - p1)) / max(1e-12, np.max(np.abs(p1))))

    ok_text = text == text2 and ok_raw
    maxdiff = float(np.max(np.abs(p1 - p2)))
    ok_pred = maxdiff == 0.0
    ratio = len(text) / len(blob)
    status = "PASS" if (ok_text and ok_pred) else "FAIL"
    print(f"{name:12s} text={len(text):>9,}B falb={len(blob):>9,}B ({ratio:4.1f}x) "
          f"raw={len(raw):>9,}B f32core={len(f32blob):>8,}B  "
          f"text_roundtrip={'exact' if ok_text else 'DIFFERS'}  "
          f"pred_maxdiff={maxdiff:.1e} f32rel={f32_rel:.1e}  {status}")
    if not (ok_text and ok_pred):
        fails += 1
        if not ok_text:
            a, b = text.splitlines(), text2.splitlines()
            for i, (x, yy) in enumerate(zip(a, b)):
                if x != yy:
                    print(f"    first text diff line {i}:\n      orig: {x[:150]}\n      rt  : {yy[:150]}")
                    break
            if len(a) != len(b):
                print(f"    line counts differ: {len(a)} vs {len(b)}")


# ---------------------------------------------------------------------------
# Loader robustness: a model file is untrusted input. Truncated / corrupt /
# future-version files must fail cleanly, never crash and never load silently.
# ---------------------------------------------------------------------------
def contrib_suite():
    """pred_contrib is the one consumer that needs the stats section.

    It is also where a FALB-loaded tree is most fragile: TreeSHAP sizes its
    path buffer from max_depth_, which is DERIVED lazily and only recomputed
    when leaf_depth_ is empty. A loader that leaves leaf_depth_ pre-sized (as
    Tree's max_leaves constructor does) yields max_depth_ == 0, a 1-entry
    buffer, and nondeterministic heap corruption on the first contrib call.
    """
    X, y = make_data(seed=21)
    booster, _ = train(X, y, "objective=regression num_leaves=31 max_bin=255 verbose=-1")
    n_feat = X.shape[1]

    def contrib(handle, rows=8):
        need = ctypes.c_int64(0)
        check(LIB.FLC_BoosterCalcNumPredict(handle, ctypes.c_int(rows), ctypes.c_int(3),
                                            ctypes.c_int(0), ctypes.c_int(-1),
                                            ctypes.byref(need)))
        out = np.zeros(need.value)
        ol = ctypes.c_int64(0)
        check(LIB.FLC_BoosterPredictForMat(
            handle, np.ascontiguousarray(X, dtype=np.float64).ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(1), ctypes.c_int32(rows), ctypes.c_int32(n_feat), ctypes.c_int(1),
            ctypes.c_int(3), ctypes.c_int(0), ctypes.c_int(-1), b"",
            ctypes.byref(ol), out.ctypes.data_as(ctypes.c_void_p)))
        return out[: ol.value]

    ref = contrib(booster)
    bad = 0
    with_stats = load_falb(save_falb(booster, 1, 1))
    got = contrib(with_stats)
    ok = np.isfinite(got).all() and np.allclose(got, ref, atol=1e-9)
    print(f"  {'PASS' if ok else 'FAIL'}  pred_contrib with stats: finite and matches original")
    bad += 0 if ok else 1
    # repeated cycles surface a heap smash that a single call can hide
    for _ in range(3):
        contrib(load_falb(save_falb(booster, 1, 1)))
        _junk = [bytearray(1 << 20) for _ in range(16)]
        del _junk
    print("  PASS  pred_contrib repeated save/load/contrib cycles leave the heap intact")
    # without the stats section it must REFUSE, not return NaN/Inf
    core = load_falb(save_falb(booster, 0, 0))
    try:
        contrib(core)
        print("  FAIL  core-only pred_contrib returned instead of refusing")
        bad += 1
    except RuntimeError as exc:
        good = "with_stats" in str(exc)
        print(f"  {'PASS' if good else 'FAIL'}  core-only pred_contrib refuses with a clear message")
        bad += 0 if good else 1
    return bad


def robustness_suite():
    X, y = make_data(seed=3)
    booster, _ = train(X, y, "objective=regression num_leaves=31 max_bin=255 verbose=-1")
    blob = save_falb(booster)
    cases = [(f"truncated@{f}", blob[:int(len(blob) * f)])
             for f in (0.01, 0.1, 0.5, 0.9, 0.999)]
    cases.append(("bad magic", b"XXXX" + blob[4:]))
    cases.append(("bad version", blob[:4] + struct.pack("<I", 999) + blob[8:]))
    cases.append(("unknown required flag", blob[:8] + struct.pack("<Q", 1 << 40) + blob[16:]))
    cases.append(("huge num_trees", blob[:16] + struct.pack("<Q", 2 ** 40) + blob[24:]))
    for label, off in (("wild section offset", 16), ("wild section length", 24)):
        t = bytearray(blob)
        struct.pack_into("<Q", t, 32 + off, 2 ** 40)
        cases.append((label, bytes(t)))
    accepted = 0
    for label, payload in cases:
        try:
            load_falb(payload)
            print(f"  ROBUSTNESS FAIL {label}: loaded corrupt input without error")
            accepted += 1
        except RuntimeError:
            pass
    print(f"  loader robustness: {len(cases) - accepted}/{len(cases)} rejected cleanly")
    return accepted


bad = robustness_suite() + contrib_suite()
if bad:
    fails += bad

print("FALB GATE PASS" if fails == 0 else f"FALB GATE FAIL ({fails})")
sys.exit(1 if fails else 0)
