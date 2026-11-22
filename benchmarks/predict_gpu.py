"""Benchmark matrix for GPU prediction: Falcata's native CUDA predictor against the CPU predictor and NVIDIA FIL.

Each case is a model shape (trees, leaves, depth profile, features, classes, categorical splits, NaN density) and a
row count. Synthetic models are written directly as model text, so every shape is exact and cheap to build: trees
grow by splitting leaves (``balanced``: the shallowest leaf, ``leafwise``: a random leaf, ``deep``: mostly the
deepest leaf), and each split's threshold cuts the probability mass that reaches the node, so every leaf is
reachable by the N(0, 1) inputs. ``--model-file`` adds real models (Falcata/LightGBM text).

The native predictor is timed in both summation modes: the default (pairwise) mode and ``deterministic=True`` (tree
order, bit-identical to the CPU predictor, which every run checks). FIL runs in its default configuration (no
``optimize()``) in float32 and float64, each the best of three timings; ``--fil-optimize`` adds its ``optimize()``d
layout. Timings are medians of repeated calls (at least 5, until ``--min-time`` seconds) after one warm-up call; the
warm-up (which packs the model for the device, or converts it for FIL) is reported separately as ``first_s``.
Throughput is rows per second. The default mode's difference from the in-order sum is reported as the largest
absolute difference, the largest difference relative to the in-order value (over |value| > 1e-300), and the largest
difference over the spread (standard deviation) of the scores.

    python benchmarks/predict_gpu.py --list
    python benchmarks/predict_gpu.py --cases t1000_l63_f500 --rows 1,1000,100000 --fil --out results.jsonl
"""

import argparse
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- synthetic models


def _ndtri(p):
    """Inverse of the standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > 1 - 0.02425:
        return -_ndtri(1 - p)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _tree_text(rng, index, n_leaves, n_features, profile, cat_frac, nan_mode):
    # leaves carry their quantile box per feature so a split always cuts mass that reaches it
    leaves = [{"depth": 0, "box": {}, "parent": -1, "side": 0}]
    split_feature, threshold, decision, left, right, cat_bounds, cat_bits = [], [], [], [], [], [0], []
    fifo = [0]  # balanced: leaves in creation order, so the front is a shallowest one
    fifo_head = 0
    deepest = 0
    while len(leaves) < n_leaves:
        if profile == "balanced":
            choice = fifo[fifo_head]
            fifo_head += 1
        elif profile == "deep" and rng.random() < 0.7:
            choice = deepest
        else:
            choice = int(rng.integers(len(leaves)))
        leaf = leaves[choice]
        node = len(split_feature)
        feature = int(rng.integers(n_features))
        lo, hi = leaf["box"].get(feature, (0.0, 1.0))
        is_cat = rng.random() < cat_frac
        if is_cat:
            # a categorical split on a value range: the bitset holds a random subset of 0..63
            words = [int(rng.integers(0, 2**32)) for _ in range(2)]
            cat_bits.extend(words)
            cat_bounds.append(len(cat_bits))
            threshold.append(float(len(cat_bounds) - 2))
            decision.append(1)
            left_box = right_box = (lo, hi)
        else:
            q = lo + (hi - lo) * rng.uniform(0.3, 0.7)
            threshold.append(_ndtri(q))
            missing = {"none": 0, "nan": 2, "zero": 1}[nan_mode]
            default_left = int(rng.random() < 0.5)
            decision.append((default_left << 1) | (missing << 2))
            left_box, right_box = (lo, q), (q, hi)
        split_feature.append(feature)
        left.append(~choice)
        right.append(~len(leaves))
        parent, side = leaf["parent"], leaf["side"]
        if parent >= 0:
            (left if side == 0 else right)[parent] = node
        new_depth = leaf["depth"] + 1
        lbox = dict(leaf["box"])
        rbox = dict(leaf["box"])
        lbox[feature], rbox[feature] = left_box, right_box
        leaves[choice] = {"depth": new_depth, "box": lbox, "parent": node, "side": 0}
        leaves.append({"depth": new_depth, "box": rbox, "parent": node, "side": 1})
        fifo += [choice, len(leaves) - 1]
        if new_depth > leaves[deepest]["depth"] or deepest == choice:
            deepest = choice
    values = rng.normal(scale=0.01, size=len(leaves))
    n_int = len(split_feature)
    lines = [
        f"Tree={index}",
        f"num_leaves={len(leaves)}",
        f"num_cat={len(cat_bounds) - 1}",
        "split_feature=" + " ".join(map(str, split_feature)),
        "split_gain=" + " ".join(["1"] * n_int),
        "threshold=" + " ".join(repr(t) for t in threshold),
        "decision_type=" + " ".join(map(str, decision)),
        "left_child=" + " ".join(map(str, left)),
        "right_child=" + " ".join(map(str, right)),
        "leaf_value=" + " ".join(repr(float(v)) for v in values),
        "leaf_weight=" + " ".join(["1"] * len(leaves)),
        "leaf_count=" + " ".join(["1"] * len(leaves)),
        "internal_value=" + " ".join(["0"] * n_int),
        "internal_weight=" + " ".join(["1"] * n_int),
        "internal_count=" + " ".join(["1"] * n_int),
    ]
    if len(cat_bounds) > 1:
        lines.append("cat_boundaries=" + " ".join(map(str, cat_bounds)))
        lines.append("cat_threshold=" + " ".join(map(str, cat_bits)))
    if len(leaves) == 1:
        lines = [
            lines[0],
            "num_leaves=1",
            "num_cat=0",
            "split_feature=",
            "split_gain=",
            "threshold=",
            "decision_type=",
            "left_child=",
            "right_child=",
            "leaf_value=" + repr(float(values[0])),
            "leaf_weight=1",
            "leaf_count=1",
            "internal_value=",
            "internal_weight=",
            "internal_count=",
        ]
    lines += ["is_linear=0", "shrinkage=1", "", ""]
    return "\n".join(lines)


def synthetic_model(trees, leaves, features, profile="leafwise", classes=1, cat_frac=0.0, nan_mode="nan", seed=0):
    """Model text with `trees` boosting iterations (x `classes` trees each) of exactly `leaves` leaves."""
    rng = np.random.default_rng(seed)
    objective = "regression" if classes == 1 else f"multiclass num_class:{classes}"
    head = [
        "tree",
        "version=v4",
        f"num_class={classes}",
        f"num_tree_per_iteration={classes}",
        "label_index=0",
        f"max_feature_idx={features - 1}",
        f"objective={objective}",
        "feature_names=" + " ".join(f"f{i}" for i in range(features)),
        "feature_infos=" + " ".join(["[-10:10]"] * features),
        "",
    ]
    body = [_tree_text(rng, i, leaves, features, profile, cat_frac, nan_mode) for i in range(trees * classes)]
    return (
        "\n".join(head) + "\n" + "".join(body) + "end of trees\n\nparameters:\n[device_type: cuda]\nend of parameters\n"
    )


# name: (trees, leaves, features, profile, classes, cat_frac); rows are chosen per run
CASES = {
    "t100_l15_f50": (100, 15, 50, "leafwise", 1, 0.0),
    "t1000_l15_f50": (1000, 15, 50, "leafwise", 1, 0.0),
    "t1000_l63_f500": (1000, 63, 500, "leafwise", 1, 0.0),
    "t1000_l255_f500": (1000, 255, 500, "leafwise", 1, 0.0),
    "t1000_l255_f500_bal": (1000, 255, 500, "balanced", 1, 0.0),
    "t1000_l255_f500_deep": (1000, 255, 500, "deep", 1, 0.0),
    "t300_l1023_f500": (300, 1023, 500, "leafwise", 1, 0.0),
    "t2000_l1023_f500_deep": (2000, 1023, 500, "deep", 1, 0.0),
    "t5000_l31_f5000": (5000, 31, 5000, "leafwise", 1, 0.0),
    "t1000_l63_f5000": (1000, 63, 5000, "leafwise", 1, 0.0),
    "t20000_l15_f250": (20000, 15, 250, "leafwise", 1, 0.0),
    "t20000_l63_f250": (20000, 63, 250, "leafwise", 1, 0.0),
    "t20000_l255_f250": (20000, 255, 250, "leafwise", 1, 0.0),
    "t2000_l63_f500_k5": (2000, 63, 500, "leafwise", 5, 0.0),
    "t5000_l7_f500_k13": (5000, 7, 500, "leafwise", 13, 0.0),
    "t1000_l63_f200_cat": (1000, 63, 200, "leafwise", 1, 0.2),
}


# --------------------------------------------------------------------------- inputs and timing


def make_rows(n, features, nan_frac, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, features), dtype=np.float32)
    if nan_frac > 0:
        X[rng.random((n, features), dtype=np.float32) < nan_frac] = np.nan
    return X


def threshold_ranges(model_text, features):
    """Per feature, the range of the model's numerical thresholds (NaN where a feature is never split on)."""
    lo = np.full(features, np.inf)
    hi = np.full(features, -np.inf)
    split = None
    for line in model_text.splitlines():
        if line.startswith("split_feature="):
            split = np.array(line[14:].split(), dtype=np.int64)
        elif line.startswith("threshold=") and split is not None and len(split):
            t = np.array(line[10:].split(), dtype=np.float64)
            np.minimum.at(lo, split, t)
            np.maximum.at(hi, split, t)
            split = None
        elif line.startswith("decision_type=") and split is not None:
            split = None
    lo[~np.isfinite(lo)] = np.nan
    hi[~np.isfinite(hi)] = np.nan
    return lo, hi


def rows_in_ranges(n, lo, hi, nan_frac, seed=1):
    """Rows whose features fall where the model splits: uniform over each feature's threshold range plus 10%."""
    rng = np.random.default_rng(seed)
    width = np.where(np.isfinite(hi - lo), hi - lo, 0.0)
    base = np.where(np.isfinite(lo), lo - 0.1 * width, 0.0)
    X = (base + rng.random((n, len(lo)), dtype=np.float32) * (1.2 * width)).astype(np.float32)
    if nan_frac > 0:
        X[rng.random(X.shape, dtype=np.float32) < nan_frac] = np.nan
    return X


def timed(fn, sync=None, min_time=0.5, max_reps=2000, min_reps=5):
    """(first call seconds, median of further calls, last output): at least min_reps calls, until min_time seconds."""
    t0 = time.perf_counter()
    out = fn()
    if sync:
        sync()
    first = time.perf_counter() - t0
    times = []
    total = 0.0
    while len(times) < max_reps and (len(times) < min_reps or total < min_time):
        t0 = time.perf_counter()
        out = fn()
        if sync:
            sync()
        dt = time.perf_counter() - t0
        times.append(dt)
        total += dt
    return first, statistics.median(times), out


def differences(a, b):
    """Largest |a - b|, largest |a - b| / |b| (|b| > 1e-300) and largest |a - b| / std(b)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = np.abs(a - b)
    nz = np.abs(b) > 1e-300
    spread = float(np.std(b)) if b.size > 1 else 0.0
    return {
        "max_abs": float(np.nanmax(d)) if d.size else 0.0,
        "max_rel": float(np.nanmax(d[nz] / np.abs(b[nz]))) if nz.any() else 0.0,
        "max_over_spread": float(np.nanmax(d) / spread) if spread > 0 else None,
        "frac_differing": float(np.mean(a != b)) if a.size else 0.0,
    }


def treelite_model(model_text, cache=None):
    """Load a model text into Treelite; `cache` (a path) keeps its checkpoint, so the parse is paid once."""
    import treelite  # noqa: PLC0415
    import treelite.frontend  # noqa: PLC0415

    if cache is not None and Path(cache).exists():
        return treelite.Model.deserialize(str(cache))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(model_text)
        path = f.name
    try:
        tl = treelite.frontend.load_lightgbm_model(path)
    finally:
        os.remove(path)
    if cache is not None:
        tl.serialize(str(cache))
    return tl


def fil_raw_predict(model):
    """Return FIL's call for raw outputs, which is what the native timings return.

    A classifier's predict() gives class labels (an argmax or a threshold on top of the outputs); predict_proba()
    gives the outputs themselves, raw scores under the identity postprocessor.
    """
    return getattr(model, "predict_proba", model.predict)


def fil_model(model_text, raw_score, precision, optimize_rows=None, cache=None):
    import nvforest  # noqa: PLC0415

    tl = treelite_model(model_text, cache)
    if raw_score:
        tl.get_header_accessor().set_field("postprocessor", "identity")
    model = nvforest.load_from_treelite_model(tl, device="gpu", precision=precision)
    if optimize_rows is not None:
        model = model.optimize(data=optimize_rows)
    return model


def run_case(name, model_text, n_features, rows_list, nan_frac, args, out, ranges=None, tl_cache=None):
    import falcata as flc  # noqa: PLC0415

    t0 = time.perf_counter()
    booster = flc.Booster(model_str=model_text)
    load_s = time.perf_counter() - t0
    info = {
        "case": name,
        "trees": booster.num_trees(),
        "classes": booster.num_model_per_iteration(),
        "features": n_features,
        "nan_frac": nan_frac,
        "load_s": load_s,
    }
    import cupy  # noqa: PLC0415

    sync = cupy.cuda.get_current_stream().synchronize
    fil_cache = {}
    for n in rows_list:
        X = make_rows(n, n_features, nan_frac) if ranges is None else rows_in_ranges(n, *ranges, nan_frac)
        rec = dict(info, rows=n, variant=os.environ.get("FALCATA_DEBUG", ""))
        # CPU predictor on a slice sized to take about args.cpu_budget seconds (the exactness reference)
        probe = X[: min(n, 1000)]
        t0 = time.perf_counter()
        booster.predict(probe, device="cpu", raw_score=True, num_threads=args.cpu_threads)
        per_row = (time.perf_counter() - t0) / len(probe)
        cpu_rows = int(min(n, max(len(probe), args.cpu_budget / max(per_row, 1e-12))))
        Xc = X[:cpu_rows]
        if args.cpu_budget > 0:
            _, cpu_s, ref = timed(
                lambda Xc=Xc: booster.predict(Xc, device="cpu", raw_score=True, num_threads=args.cpu_threads),
                min_time=args.min_time,
                max_reps=3,
                min_reps=1,
            )
            rec["cpu_rows_per_s"] = cpu_rows / cpu_s
        else:
            ref = booster.predict(Xc, device="cpu", raw_score=True, num_threads=args.cpu_threads)
        rec["cpu_rows"] = cpu_rows
        Xd = cupy.asarray(X)
        try:
            # default mode, device input (the headline) and host input
            first, s_dev, got = timed(lambda Xd=Xd: booster.predict(Xd, raw_score=True), sync, args.min_time)
            rec["native_dev_first_s"] = first
            rec["native_dev_rows_per_s"] = n / s_dev
            rec["native_stats"] = booster.cuda_predict_stats
            fast = cupy.asnumpy(got)
            del got
            _, s_host, _ = timed(lambda X=X: booster.predict(X, device="cuda", raw_score=True), None, args.min_time)
            rec["native_host_rows_per_s"] = n / s_host
            # deterministic mode: bit-identical to the CPU predictor
            _, s_det, got = timed(
                lambda Xd=Xd: booster.predict(Xd, raw_score=True, deterministic=True), sync, args.min_time
            )
            rec["native_det_dev_rows_per_s"] = n / s_det
            exact = cupy.asnumpy(got)
            del got
            rec["native_det_bit_identical"] = bool(np.array_equal(exact[:cpu_rows], ref, equal_nan=True))
            rec["native_det_max_abs_diff"] = float(np.nanmax(np.abs(exact[:cpu_rows] - ref))) if cpu_rows else 0.0
            rec["default_vs_det"] = differences(fast, exact)
        except Exception as exc:  # recorded, so one failing configuration does not end the run
            rec["native_error"] = f"{type(exc).__name__}: {exc}"
        if args.fil:
            variants = [(p, False) for p in args.fil_precisions]
            if args.fil_optimize:
                variants += [(p, True) for p in args.fil_precisions]
            for precision, optimized in variants:
                key = (precision, optimized)
                tag = f"fil_{precision}{'_opt' if optimized else ''}"
                try:
                    if key not in fil_cache or optimized:
                        t0 = time.perf_counter()
                        fil_cache[key] = fil_model(
                            model_text, True, precision, Xd[: min(n, 100000)] if optimized else None, tl_cache
                        )
                        rec[f"{tag}_build_s"] = time.perf_counter() - t0
                    fm = fil_raw_predict(fil_cache[key])
                    # FIL's call time varies from one timing to the next on some shapes (an order of magnitude at
                    # small batches): its best of three timings is the reference
                    fil_runs = [timed(lambda fm=fm, Xd=Xd: fm(Xd), sync, args.min_time) for _ in range(3)]
                    fil_s = min(t[1] for t in fil_runs)
                    fp = fil_runs[-1][2]
                    rec[f"{tag}_dev_rows_per_s"] = n / fil_s
                    rec[f"{tag}_dev_rows_per_s_runs"] = [n / t[1] for t in fil_runs]
                    fp = cupy.asnumpy(fp).reshape(n, -1)[:cpu_rows].reshape(ref.shape)
                    rec[f"{tag}_max_abs_diff"] = float(np.nanmax(np.abs(fp - ref)))
                    if not optimized and args.fil_host:
                        _, fil_h, _ = timed(lambda fm=fm, X=X: fm(X), None, args.min_time)
                        rec[f"{tag}_host_rows_per_s"] = n / fil_h
                    rec[f"{tag}_layout"] = getattr(fil_cache[key], "layout", None)
                    rec[f"{tag}_chunk"] = getattr(fil_cache[key], "default_chunk_size", None)
                except Exception as exc:  # report and keep going: one failing FIL config is a result
                    rec[f"{tag}_error"] = f"{type(exc).__name__}: {exc}"
            cupy.get_default_memory_pool().free_all_blocks()
        del Xd
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(json.dumps(rec), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true")
    p.add_argument("--cases", default="", help="comma-separated case names (default: none)")
    p.add_argument("--model-file", action="append", default=[], help="name=path of a model text file")
    p.add_argument("--rows", default="1,100,1000,100000,1000000")
    p.add_argument("--nan-frac", type=float, default=0.1)
    p.add_argument(
        "--cpu-budget",
        type=float,
        default=2.0,
        help="seconds of CPU prediction per timing (0: no CPU timing, only the reference rows)",
    )
    p.add_argument("--cpu-threads", type=int, default=16)
    p.add_argument("--max-cells", type=float, default=2e9, help="skip row counts above this many matrix cells")
    p.add_argument("--fil", action="store_true")
    p.add_argument("--fil-precisions", default="single,double")
    p.add_argument("--fil-optimize", action="store_true", help="also time FIL's optimize()d layout")
    p.add_argument("--fil-host", action="store_true", help="also time FIL on host (numpy) input")
    p.add_argument("--min-time", type=float, default=0.3, help="seconds of repeated calls per timing")
    p.add_argument("--out", default="-")
    p.add_argument("--cache-dir", default="", help="keep generated model texts here and reuse them")
    p.add_argument("--treelite-cache", default="", help="keep Treelite checkpoints of the models here (FIL's parse)")
    p.add_argument("--treelite-only", action="store_true", help="only write the Treelite checkpoints (CPU) and stop")
    args = p.parse_args()
    args.fil_precisions = [x for x in args.fil_precisions.split(",") if x]
    if args.list:
        for name, spec in CASES.items():
            print(name, spec)
        return
    rows_list = [int(r) for r in args.rows.split(",") if r]
    out = sys.stdout if args.out == "-" else open(args.out, "a")
    meta = {"host": platform.node(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        import cupy  # noqa: PLC0415

        meta["gpu"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:  # noqa: BLE001 (the meta line is informational)
        pass
    out.write(json.dumps({"meta": meta}) + "\n")
    for name in [c for c in args.cases.split(",") if c]:
        trees, leaves, features, profile, classes, cat = CASES[name]
        cached = Path(args.cache_dir) / f"{name}.txt" if args.cache_dir else None
        if cached is not None and cached.exists():
            text = cached.read_text()
        else:
            text = synthetic_model(trees, leaves, features, profile, classes, cat)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(text)
        rows = [r for r in rows_list if r * features <= args.max_cells]
        tl_cache = Path(args.treelite_cache) / f"{name}.tl" if args.treelite_cache else None
        if args.treelite_only:
            treelite_model(text, tl_cache)
            continue
        run_case(name, text, features, rows, args.nan_frac, args, out, tl_cache=tl_cache)
    for spec in args.model_file:
        name, path = spec.split("=", 1)
        text = Path(path).read_text()
        features = (
            int(next(line for line in text.splitlines() if line.startswith("max_feature_idx=")).split("=")[1]) + 1
        )
        rows = [r for r in rows_list if r * features <= args.max_cells]
        tl_cache = Path(args.treelite_cache) / f"{name}.tl" if args.treelite_cache else None
        ranges_cache = Path(args.treelite_cache) / f"{name}.ranges.npz" if args.treelite_cache else None
        if ranges_cache is not None and ranges_cache.exists():
            z = np.load(ranges_cache)
            ranges = (z["lo"], z["hi"])
        else:
            ranges = threshold_ranges(text, features)
            if ranges_cache is not None:
                np.savez(ranges_cache, lo=ranges[0], hi=ranges[1])
        if args.treelite_only:
            treelite_model(text, tl_cache)
            continue
        run_case(name, text, features, rows, args.nan_frac, args, out, ranges=ranges, tl_cache=tl_cache)


if __name__ == "__main__":
    main()
