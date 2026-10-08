"""Run exactly one benchmark cell: (library, dataset, regime, kind).

Appends one JSON line to ``<workspace>/results/runs.jsonl`` and exits. Meant
to be launched by orchestrate.py in a fresh subprocess using the venv python
that owns the requested library, so GPU state and library versions are
isolated per run.

Timing semantics (reported separately, because they differ per library):

- ``construct_s``: building the library's training data structure —
  ``lgb.Dataset`` (binning happens here), ``xgb.QuantileDMatrix`` (quantile
  sketch), ``catboost.Pool`` (thin wrapper; CatBoost quantizes inside fit).
- ``train_s``: the boosting loop, including any remaining device transfer.
- ``curve`` runs additionally evaluate the held-out set every
  ``eval_every`` iterations; their timings are NOT comparable to plain runs
  (CatBoost's curve time axis is linearly approximated — no per-iteration
  wall clock is exposed — and is flagged with ``curve_time_approx``).
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
from common import ALL_LIBRARIES, CACHE_DIR, DATASETS, REGIMES, RUNS_JSONL, SEED

NUM_THREADS = int(os.environ.get("FALCATA_BENCH_THREADS", "0")) or os.cpu_count()

# Anonymous-memory cap: competitors that materialize full float copies
# (catboost/xgboost peaked >100GB anonymous on numerai) die with a clean
# MemoryError instead of inviting the host OOM killer. RLIMIT_DATA leaves
# file-backed memmaps (the cache reads) uncounted. Override with
# FALCATA_BENCH_MEM_GB; the default suits a 128GB host with swap. The guard's
# job is stopping UNBOUNDED growth, not excluding known-hungry-but-finite
# competitors, so it is set well above their measured peak.
try:
    import resource

    _cap = int(float(os.environ.get("FALCATA_BENCH_MEM_GB", "120")) * 1024**3)
    resource.setrlimit(resource.RLIMIT_DATA, (_cap, _cap))
except Exception:
    pass


class ResourceMonitor:
    """Polls peak GPU memory (nvidia-smi) and host RSS in a thread."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.gpu_peak_mb = 0
        self.rss_peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import psutil

        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                self.gpu_peak_mb = max(self.gpu_peak_mb, int(out.splitlines()[0]))
            except Exception:
                pass
            try:
                rss = proc.memory_info().rss
                for c in proc.children(recursive=True):
                    try:
                        rss += c.memory_info().rss
                    except Exception:
                        pass
                self.rss_peak_mb = max(self.rss_peak_mb, rss // (1024 * 1024))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._thread.join(timeout=5)


def load_data(name, input_dtype="f32"):
    """Memory-map the preprocessed cache of ``name``.

    ``input_dtype`` picks the numerai matrix: ``f32`` is the float32 twin every
    engine reads, ``i8`` the int8 twin (same rows and values, a quarter of the
    bytes) that Falcata bins natively. Other datasets have one dtype only.
    """
    d = os.path.join(CACHE_DIR, name)
    if name == "numerai":
        with open(os.path.join(d, "meta.json")) as fh:
            meta = json.load(fh)
        fname, dtype = {"f32": ("X.f32.mem", np.float32), "i8": ("X.i8.mem", np.int8)}[input_dtype]
        x = np.memmap(
            os.path.join(d, fname),
            dtype=dtype,
            mode="r",
            shape=(meta["n_rows"], meta["n_features"]),
        )
        y = np.load(os.path.join(d, "y.npy"))
        era = np.load(os.path.join(d, "era_codes.npy"))
        tr_end, te_start = meta["train_end"], meta["test_start"]
        return (
            x[:tr_end],
            y[:tr_end],
            x[te_start:],
            y[te_start:],
            {"era_test": era[te_start:]},
        )
    x_tr = np.load(os.path.join(d, "X_train.npy"), mmap_mode="r")
    y_tr = np.load(os.path.join(d, "y_train.npy"))
    x_te = np.load(os.path.join(d, "X_test.npy"), mmap_mode="r")
    y_te = np.load(os.path.join(d, "y_test.npy"))
    return x_tr, y_tr, x_te, y_te, {}


def numerai_corr_np(preds, targets):
    """Official Numerai correlation (rank -> gaussianize -> pow 1.5 -> Pearson)."""
    from scipy.stats import norm, rankdata

    n = len(preds)
    ranked = (rankdata(preds, method="average") - 0.5) / n
    gauss = norm.ppf(ranked)
    preds_p15 = np.sign(gauss) * np.abs(gauss) ** 1.5
    centered = targets - targets.mean()
    targets_p15 = np.sign(centered) * np.abs(centered) ** 1.5
    return float(np.corrcoef(preds_p15, targets_p15)[0, 1])


def numerai_metrics(preds, y, era):
    corrs = np.array([numerai_corr_np(preds[era == e], y[era == e]) for e in np.unique(era)])
    mean, std = corrs.mean(), corrs.std(ddof=0)
    cumulative = np.cumprod(1 + corrs)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = ((rolling_max - cumulative) / rolling_max).max()
    return {
        "corr_mean": float(mean),
        "corr_std": float(std),
        "corr_sharpe": float(mean / std) if std > 0 else None,
        "max_drawdown": float(-drawdown),
        "n_eras": len(corrs),
    }


def quality_metrics(task, preds, y, extra):
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    if task == "binary":
        auc = roc_auc_score(y, preds)
        return {"auc": float(auc), "sane": bool(auc > 0.55)}
    if task == "multiclass":
        acc = accuracy_score(y, preds.argmax(axis=1))
        ll = log_loss(y, preds, labels=list(range(preds.shape[1])))
        return {"accuracy": float(acc), "mlogloss": float(ll), "sane": bool(acc > 0.5)}
    if task == "regression":
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        return {"rmse": rmse, "sane": bool(rmse < float(np.std(y)))}
    if task == "numerai":
        m = numerai_metrics(preds, y, extra["era_test"])
        m["rmse"] = float(np.sqrt(np.mean((preds - y) ** 2)))
        m["sane"] = bool(m["corr_mean"] > 0)
        return m
    raise ValueError(task)


def curve_metric_ok(task, curve_pts, metrics):
    """Cross-check a curve's final point against the predict()-based metric.

    The curve's last evaluation and the end-of-run quality are computed on the
    same model and test set, so they must agree closely. A large gap means the
    curve recorded the WRONG SERIES -- exactly what happened when catboost's
    evals_result was read by position and yielded Logloss on an AUC axis.
    Binary only: that is where every engine reports the same metric (AUC);
    elsewhere the curve metric legitimately differs from the recorded one
    (lightgbm regression curves carry l2 = MSE, metrics carry RMSE).
    """
    if task != "binary" or not curve_pts:
        return True
    final = curve_pts[-1][2]
    auc = (metrics or {}).get("auc")
    if final is None or auc is None:
        return True
    return abs(final - auc) < 0.02


def run_lightgbm(task, x_tr, y_tr, x_te, y_te, reg, library, curve, cat_cols=None, shared=None):
    """One LightGBM-API cell.

    ``shared`` is a per-regime dict the caller keeps across the kinds of one
    regime: the constructed ``lgb.Dataset`` (binning + the CUDA column store)
    is built once, timed once, and reused by every kind, so the five kinds of
    a regime cost one construct instead of five. Training never mutates a
    constructed Dataset, so the reuse leaves ``train_s`` untouched.
    """
    # falcata* variants import falcata, lightgbm* import upstream lightgbm --
    # the two install under different venvs (both own the ``lgb`` API surface).
    if library.startswith("falcata"):
        import falcata as lgb
    else:
        import lightgbm as lgb

    opencl = library == "lightgbm-ocl"
    params = {
        "objective": {
            "binary": "binary",
            "multiclass": "multiclass",
            "regression": "regression",
            "numerai": "regression",
        }[task],
        "learning_rate": reg["lr"],
        "num_leaves": reg["leaves"],
        "max_depth": reg["depth"],
        "max_bin": 255,
        # upstream's legacy OpenCL backend is the working GPU path on hardware
        # where its CUDA backend fails; max_bin stays at the suite-wide 255 for
        # config parity (OpenCL's recommended 63 would favor it)
        "device_type": "gpu" if opencl else "cuda",
        "num_threads": NUM_THREADS,
        "seed": SEED,
        "verbose": -1,
        # speed cells never evaluate; curve runs set a real metric below
        "metric": "None",
    }
    if task == "multiclass":
        params["num_class"] = DATASETS["covtype"]["num_class"]
    if "colsample" in reg:
        params["feature_fraction"] = reg["colsample"]
    if "l2" in reg:
        params["lambda_l2"] = reg["l2"]
    if "min_data" in reg:
        params["min_data_in_leaf"] = reg["min_data"]
    if library == "falcata-stoch":
        params["quant_mode"] = "stochastic"
    elif library == "falcata-stoch64":
        # fixedpoint's bin budget with stochastic's rounding scheme: isolates
        # the rounding scheme from the bin count in the mode comparison
        params["quant_mode"] = "stochastic"
        params["quant_bins"] = 64
    elif library == "falcata-fixed":
        params["quant_mode"] = "fixedpoint"
    elif library == "falcata-noquant":
        params["quant_mode"] = "none"
    elif library == "lightgbm-quant":
        params["use_quantized_grad"] = True

    shared = {} if shared is None else shared
    construct_shared = "dtrain" in shared
    if construct_shared:
        dtrain, construct_s = shared["dtrain"], shared["construct_s"]
    else:
        t0 = time.perf_counter()
        # the Dataset must be built with the final params (incl. device_type);
        # non-memmap inputs are made contiguous so binning does not pay for strides
        dtrain = lgb.Dataset(
            x_tr if isinstance(x_tr, np.memmap) else np.ascontiguousarray(x_tr),
            label=y_tr,
            params=params,
            categorical_feature=cat_cols if cat_cols else "auto",
        )
        dtrain.construct()
        construct_s = time.perf_counter() - t0
        shared["dtrain"], shared["construct_s"] = dtrain, construct_s

    curve_pts = []
    t0 = time.perf_counter()
    if curve:
        # without a real metric eval_valid() returns nothing and the curve
        # records null quality
        params["metric"] = {
            "binary": "auc",
            "multiclass": "multi_logloss",
            "regression": "l2",
            "numerai": "l2",
        }[task]
        dvalid = lgb.Dataset(x_te, label=y_te, reference=dtrain)
        bst = lgb.Booster(params=params, train_set=dtrain)
        bst.add_valid(dvalid, "test")
        for i in range(reg["rounds"]):
            bst.update()
            if (i + 1) % reg["eval_every"] == 0 or i + 1 == reg["rounds"]:
                t_now = time.perf_counter() - t0
                res = bst.eval_valid()
                curve_pts.append([i + 1, t_now, res[0][2] if res else None])
    else:
        bst = lgb.train(params, dtrain, num_boost_round=reg["rounds"])
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = bst.predict(x_te)
    predict_s = time.perf_counter() - t0
    return {
        "construct_s": construct_s,
        "construct_shared": construct_shared,
        "train_s": train_s,
        "predict_s": predict_s,
        "preds": preds,
        "version": lgb.__version__,
        "curve": curve_pts,
    }


def run_xgboost(task, x_tr, y_tr, x_te, y_te, reg, library, curve, cat_cols=None):
    import xgboost as xgb

    params = {
        "objective": {
            "binary": "binary:logistic",
            "multiclass": "multi:softprob",
            "regression": "reg:squarederror",
            "numerai": "reg:squarederror",
        }[task],
        "eta": reg["lr"],
        "max_depth": reg["depth"],
        "max_bin": 255,
        "device": "cuda",
        "tree_method": "hist",
        "nthread": NUM_THREADS,
        "seed": SEED,
    }
    if reg["depth"] == -1:
        # unbounded-depth leaf-limited regime -> xgboost's leaf-wise mode
        params.update(grow_policy="lossguide", max_leaves=reg["leaves"], max_depth=0)
    elif library == "xgboost-lossguide":
        # leaf-wise apples-to-apples: the same (num_leaves, max_depth) pair the
        # LightGBM-family libraries get, instead of depth-wise max_depth alone
        params.update(grow_policy="lossguide", max_leaves=reg["leaves"])
    if task == "multiclass":
        params["num_class"] = DATASETS["covtype"]["num_class"]
    if "colsample" in reg:
        params["colsample_bytree"] = reg["colsample"]
    if "l2" in reg:
        params["lambda"] = reg["l2"]  # xgboost default is already 1
    if "min_data" in reg:
        # hessian-weighted, but the hessian is 1 per row for squared error, so
        # this is exactly a row count -- equivalent to min_data_in_leaf
        params["min_child_weight"] = reg["min_data"]
    eval_metric = {
        "binary": "auc",
        "multiclass": "mlogloss",
        "regression": "rmse",
        "numerai": "rmse",
    }[task]

    ft = None
    if cat_cols:
        ft = ["c" if i in set(cat_cols) else "q" for i in range(x_tr.shape[1])]
    t0 = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(
        np.asarray(x_tr),
        label=y_tr,
        max_bin=255,
        feature_types=ft,
        enable_categorical=bool(cat_cols),
    )
    construct_s = time.perf_counter() - t0

    curve_pts = []
    t0 = time.perf_counter()
    if curve:
        dtest = xgb.DMatrix(
            np.asarray(x_te),
            label=y_te,
            feature_types=ft,
            enable_categorical=bool(cat_cols),
        )
        params["eval_metric"] = eval_metric
        bst = xgb.Booster(params, [dtrain])
        for i in range(reg["rounds"]):
            bst.update(dtrain, i)
            if (i + 1) % reg["eval_every"] == 0 or i + 1 == reg["rounds"]:
                t_now = time.perf_counter() - t0
                res = bst.eval_set([(dtest, "test")], i)
                curve_pts.append([i + 1, t_now, float(res.split(":")[-1])])
    else:
        bst = xgb.train(params, dtrain, num_boost_round=reg["rounds"])
    train_s = time.perf_counter() - t0

    # chunked GPU predict: a single DMatrix over the full test set asks for
    # more VRAM than the card has on numerai-scale data (observed 34GB ask)
    bst.set_param({"device": "cuda"})
    step = 200_000
    preds = np.concatenate(
        [
            bst.predict(
                xgb.DMatrix(
                    np.asarray(x_te[i : i + step]),
                    feature_types=ft,
                    enable_categorical=bool(cat_cols),
                )
            )
            for i in range(0, x_te.shape[0], step)
        ]
    )
    return {
        "construct_s": construct_s,
        "train_s": train_s,
        "preds": preds,
        "version": xgb.__version__,
        "curve": curve_pts,
    }


def run_catboost(task, x_tr, y_tr, x_te, y_te, reg, curve, cat_cols=None):
    import catboost as cb

    loss = {
        "binary": "Logloss",
        "multiclass": "MultiClass",
        "regression": "RMSE",
        "numerai": "RMSE",
    }[task]
    kw = {
        "iterations": reg["rounds"],
        "learning_rate": reg["lr"],
        "depth": reg["depth"],
        "border_count": 254,
        "task_type": "GPU",
        "devices": "0",
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False,
        "loss_function": loss,
    }
    if reg["depth"] == -1:
        # leaf-limited regime -> catboost's leaf-wise mode; 16 is its hard
        # maximum depth (it has no unlimited setting), noted in the report
        kw.update(grow_policy="Lossguide", max_leaves=reg["leaves"], depth=16)
    rsm_dropped = False
    if "colsample" in reg:
        kw["rsm"] = reg["colsample"]
    if "l2" in reg:
        kw["l2_leaf_reg"] = reg["l2"]  # catboost default is 3
    if "min_data" in reg:
        # ignored by the default SymmetricTree grow policy; noted in the report
        kw["min_data_in_leaf"] = reg["min_data"]

    def to_pool_matrix(x):
        # catboost requires integer/string categorical values; the airline
        # columns are all integral, so an int32 cast is lossless
        return np.asarray(x).astype(np.int32) if cat_cols else np.asarray(x)

    t0 = time.perf_counter()
    train_pool = cb.Pool(to_pool_matrix(x_tr), label=y_tr, cat_features=cat_cols)
    construct_s = time.perf_counter() - t0

    cls = cb.CatBoostClassifier if task in ("binary", "multiclass") else cb.CatBoostRegressor

    def fit(kw):
        model = cls(**kw)
        t0 = time.perf_counter()
        if curve:
            if task == "binary":
                kw["eval_metric"] = "AUC"
                model = cls(**kw)
            test_pool = cb.Pool(to_pool_matrix(x_te), label=y_te, cat_features=cat_cols)
            model.fit(train_pool, eval_set=test_pool, metric_period=reg["eval_every"])
        else:
            model.fit(train_pool)
        return model, time.perf_counter() - t0

    try:
        model, train_s = fit(kw)
    except Exception as e:
        # rsm is not supported for every GPU loss; retry without and flag it
        if "rsm" in kw and "rsm" in str(e).lower():
            kw.pop("rsm")
            rsm_dropped = True
            model, train_s = fit(kw)
        else:
            raise

    curve_pts = []
    if curve:
        vals = model.get_evals_result().get("validation", {})
        if vals:
            # catboost records the LOSS first and any extra eval_metric after
            # it, so taking the first key yields Logloss on binary tasks --
            # where every other engine here records AUC. That silently puts a
            # loss curve on an AUC axis, so pick the series by name.
            want = {
                "binary": "AUC",
                "multiclass": "MultiClass",
                "regression": "RMSE",
                "numerai": "RMSE",
            }[task]
            series = vals.get(want, vals[next(iter(vals))])
            n = len(series)
            # metric_period=eval_every already makes catboost record ONLY every
            # eval_every-th iteration, so `series` IS the subsampled curve --
            # 20 entries for 500 rounds at eval_every 25. Striding by eval_every
            # again is a second subsample: range(0, 20, 25) yields one point,
            # which is what silently dropped catboost from the time-to-quality
            # plot. Walk every recorded entry and map index i back to its real
            # iteration. catboost exposes no per-iteration wall clock, so the
            # time axis is approximated linearly (flagged in the record).
            for i in range(n):
                iteration = reg["rounds"] if i == n - 1 else min(i * reg["eval_every"] + 1, reg["rounds"])
                curve_pts.append([iteration, train_s * iteration / reg["rounds"], series[i]])

    if task == "binary":
        preds = model.predict_proba(to_pool_matrix(x_te))[:, 1]
    elif task == "multiclass":
        preds = model.predict_proba(to_pool_matrix(x_te))
    else:
        preds = model.predict(to_pool_matrix(x_te))
    return {
        "construct_s": construct_s,
        "train_s": train_s,
        "preds": preds,
        "version": cb.__version__,
        "curve": curve_pts,
        "rsm_dropped": rsm_dropped,
        "curve_time_approx": bool(curve),
    }


def parse_cells(args):
    """Return the (regime, kind) cells this process runs, in matrix order."""
    if args.cells:
        cells = []
        for item in args.cells.split(","):
            reg, kind = item.split(":")
            if reg not in REGIMES:
                raise SystemExit(f"unknown regime {reg!r}; known: {list(REGIMES)}")
            cells.append((reg, kind))
        return cells
    if not (args.regime and args.kind):
        raise SystemExit("pass --cells REGIME:KIND,... or both --regime and --kind")
    return [(args.regime, args.kind)]


def recorded_cells(path, library, dataset):
    """(regime, kind) -> status of every cell of this library/dataset already in ``path``."""
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("library") == library and r.get("dataset") == dataset:
                    done[(r["regime"], r["kind"])] = r["status"]
    return done


def write_record(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    shown = {k: v for k, v in rec.items() if k not in ("curve", "error")}
    print(json.dumps(shown)[:2000], flush=True)


def run_cell(args, task, reg_name, kind, data, shared):
    """Run one (regime, kind) cell on already-loaded ``data``; returns its record."""
    x_tr, y_tr, x_te, y_te, extra = data
    reg = dict(REGIMES[reg_name])
    if args.align_l2:
        reg["l2"] = 1.0
    overrides = {}
    for kv in args.overrides:
        k, v = kv.split("=", 1)
        x = float(v)
        overrides[k] = int(x) if x.is_integer() else x
    reg.update(overrides)
    curve = kind == "curve"

    rec = {
        "library": args.library,
        "dataset": args.dataset,
        "regime": reg_name,
        "kind": kind,
        "status": "ok",
        "input_dtype": str(np.asarray(x_tr[:1]).dtype) if hasattr(x_tr, "dtype") else None,
        "n_train": int(x_tr.shape[0]),
        "n_features": int(x_tr.shape[1]),
    }
    if overrides:
        rec["overrides"] = overrides
    try:
        cat_cols = DATASETS[args.dataset].get("cat_cols")
        with ResourceMonitor() as mon:
            t_total = time.perf_counter()
            if args.library.startswith(("falcata", "lightgbm")):
                r = run_lightgbm(
                    task,
                    x_tr,
                    y_tr,
                    x_te,
                    y_te,
                    reg,
                    args.library,
                    curve,
                    cat_cols=cat_cols,
                    shared=shared,
                )
            elif args.library.startswith("xgboost"):
                r = run_xgboost(
                    task,
                    x_tr,
                    y_tr,
                    x_te,
                    y_te,
                    reg,
                    args.library,
                    curve,
                    cat_cols=cat_cols,
                )
            else:
                r = run_catboost(task, x_tr, y_tr, x_te, y_te, reg, curve, cat_cols=cat_cols)
            total_s = time.perf_counter() - t_total
        preds = r.pop("preds")
        rec.update(r)
        rec["total_s"] = total_s
        rec["trees_per_s"] = reg["rounds"] / r["train_s"]
        rec["gpu_mem_peak_mb"] = mon.gpu_peak_mb
        rec["rss_peak_mb"] = mon.rss_peak_mb
        t0 = time.perf_counter()
        rec["metrics"] = quality_metrics(task, np.asarray(preds), y_te, extra)
        rec["eval_s"] = time.perf_counter() - t0
        # a run that trains but produces a garbage model is a FAILURE, not a
        # fast run -- its timings must never enter the report tables
        if not rec["metrics"].get("sane", True):
            rec["status"] = "insane"
            rec["error"] = f"quality sanity check failed: {rec['metrics']}"
        elif curve and not curve_metric_ok(task, rec.get("curve"), rec["metrics"]):
            # wrong-series curves must never reach the time-to-quality plot
            rec["status"] = "bad_curve"
            rec["error"] = (
                f"curve final point {rec['curve'][-1][2]} disagrees with "
                f"recomputed auc {rec['metrics']['auc']} -- the curve recorded "
                "a different metric than the one it claims"
            )
    except Exception:
        rec["status"] = "failed"
        rec["error"] = traceback.format_exc()[-3000:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True, choices=ALL_LIBRARIES)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--regime", choices=list(REGIMES), help="one cell: its regime (with --kind)")
    ap.add_argument("--kind", help="one cell: warmup | timed1..3 | curve (with --regime)")
    ap.add_argument(
        "--cells",
        default=None,
        metavar="REGIME:KIND,...",
        help="several cells of this library/dataset in one process: the data is "
        "loaded once and each regime's Dataset is constructed once; cells already "
        "recorded in --out are skipped",
    )
    ap.add_argument(
        "--input-dtype",
        choices=["auto", "f32", "i8"],
        default="auto",
        help="numerai matrix twin to read; auto = i8 for the falcata arms (native "
        "int8 binning, a quarter of the bytes), f32 for every other engine",
    )
    ap.add_argument(
        "--align-l2",
        action="store_true",
        help="set lambda_l2/lambda/l2_leaf_reg to 1.0 on every engine. Off by "
        "default because the engines' own defaults differ (0/1/3) and the "
        "published numbers were measured that way; on, the comparison is "
        "stricter but will NOT match docs/performance.md.",
    )
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="override a regime knob (l2, min_data, colsample, rounds, ...) for "
        "one-off experiments; repeatable. Overridden runs are stamped with an "
        "'overrides' field so they can never be mistaken for published cells.",
    )
    ap.add_argument("--out", default=RUNS_JSONL)
    args = ap.parse_args()

    task = DATASETS[args.dataset]["task"]
    cells = parse_cells(args)
    done = recorded_cells(args.out, args.library, args.dataset) if args.cells else {}
    todo = [c for c in cells if c not in done]
    if not todo:
        print("nothing to run: every requested cell is recorded", flush=True)
        return

    input_dtype = args.input_dtype
    if input_dtype == "auto":
        input_dtype = "i8" if args.library.startswith("falcata") else "f32"
    if args.dataset != "numerai":
        input_dtype = "f32"
    t0 = time.perf_counter()
    data = load_data(args.dataset, input_dtype)
    load_s = time.perf_counter() - t0

    all_ok = True
    shared_by_regime = {}
    for reg_name, kind in todo:
        # a regime whose warmup failed in this process is not worth its timed kinds
        if kind != "warmup" and done.get((reg_name, "warmup")) == "failed":
            rec = {
                "library": args.library,
                "dataset": args.dataset,
                "regime": reg_name,
                "kind": kind,
                "status": "skipped_warmup_failed",
            }
        else:
            t0 = time.perf_counter()
            shared = shared_by_regime.setdefault(reg_name, {})
            rec = run_cell(args, task, reg_name, kind, data, shared)
            rec["load_s"] = load_s
            rec["cell_s"] = time.perf_counter() - t0
        done[(reg_name, kind)] = rec["status"]
        all_ok = all_ok and rec["status"] == "ok"
        write_record(args.out, rec)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
