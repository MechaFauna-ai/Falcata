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

from common import CACHE_DIR, DATASETS, REGIMES, RUNS_JSONL, SEED


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


def load_data(name):
    d = os.path.join(CACHE_DIR, name)
    if name == "numerai":
        with open(os.path.join(d, "meta.json")) as fh:
            meta = json.load(fh)
        x = np.memmap(
            os.path.join(d, "X.f32.mem"),
            dtype=np.float32,
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
    corrs = np.array(
        [numerai_corr_np(preds[era == e], y[era == e]) for e in np.unique(era)]
    )
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


def run_lightgbm(task, x_tr, y_tr, x_te, y_te, reg, quantized, curve, opencl=False):
    import lightgbm as lgb

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
        # OpenCL backend ("gpu"): platform 0 is the only ICD on the bench box
        # (NVIDIA); used only where the upstream CUDA build crashes
        "device_type": "gpu" if opencl else "cuda",
        "num_threads": os.cpu_count(),
        "seed": SEED,
        "verbose": -1,
        # plain runs never evaluate, so metric stays "None" there; curve runs
        # need a real eval metric or eval_valid() returns no results and the
        # curve records null quality (matching the metrics xgboost curves use)
        "metric": {
            "binary": "auc",
            "multiclass": "multi_logloss",
            "regression": "rmse",
            "numerai": "rmse",
        }[task]
        if curve
        else "None",
    }
    if opencl:
        params["gpu_platform_id"] = 0
        params["gpu_device_id"] = 0
    if task == "multiclass":
        params["num_class"] = DATASETS["covtype"]["num_class"]
    if "colsample" in reg:
        params["feature_fraction"] = reg["colsample"]
    if "l2" in reg:
        params["lambda_l2"] = reg["l2"]
    if "min_data" in reg:
        params["min_data_in_leaf"] = reg["min_data"]
    if quantized:
        params["use_quantized_grad"] = True

    t0 = time.perf_counter()
    # the Dataset must be created with the final params (incl. device_type)
    dtrain = lgb.Dataset(x_tr, label=y_tr, params=params)
    dtrain.construct()
    construct_s = time.perf_counter() - t0

    curve_pts = []
    t0 = time.perf_counter()
    if curve:
        dvalid = lgb.Dataset(x_te, label=y_te, reference=dtrain)
        bst = lgb.Booster(params=params, train_set=dtrain)
        bst.add_valid(dvalid, "test")
        for i in range(reg["rounds"]):
            bst.update()
            if (i + 1) % reg["eval_every"] == 0 or i + 1 == reg["rounds"]:
                res = bst.eval_valid()
                curve_pts.append(
                    [i + 1, time.perf_counter() - t0, res[0][2] if res else None]
                )
    else:
        bst = lgb.train(params, dtrain, num_boost_round=reg["rounds"])
    train_s = time.perf_counter() - t0

    preds = bst.predict(x_te)
    return {
        "construct_s": construct_s,
        "train_s": train_s,
        "preds": preds,
        "version": lgb.__version__,
        "curve": curve_pts,
    }


def run_xgboost(task, x_tr, y_tr, x_te, y_te, reg, curve):
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
        "nthread": os.cpu_count(),
        "seed": SEED,
    }
    if task == "multiclass":
        params["num_class"] = DATASETS["covtype"]["num_class"]
    if "colsample" in reg:
        params["colsample_bytree"] = reg["colsample"]
    if "l2" in reg:
        params["lambda"] = reg["l2"]  # xgboost default is already 1
    if "min_data" in reg:
        # for squared error the hessian is 1 per row, so min_child_weight
        # is exactly a row count -- equivalent to min_data_in_leaf
        params["min_child_weight"] = reg["min_data"]

    t0 = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(np.asarray(x_tr), label=y_tr, max_bin=255)
    construct_s = time.perf_counter() - t0

    curve_pts = []
    t0 = time.perf_counter()
    if curve:
        dtest = xgb.DMatrix(np.asarray(x_te), label=y_te)
        params["eval_metric"] = {
            "binary": "auc",
            "multiclass": "mlogloss",
            "regression": "rmse",
            "numerai": "rmse",
        }[task]
        bst = xgb.Booster(params, [dtrain])
        for i in range(reg["rounds"]):
            bst.update(dtrain, i)
            if (i + 1) % reg["eval_every"] == 0 or i + 1 == reg["rounds"]:
                res = bst.eval_set([(dtest, "test")], i)
                curve_pts.append(
                    [i + 1, time.perf_counter() - t0, float(res.split(":")[-1])]
                )
    else:
        bst = xgb.train(params, dtrain, num_boost_round=reg["rounds"])
    train_s = time.perf_counter() - t0

    # chunked inplace_predict: a full GPU DMatrix of a wide test set can OOM
    # the device while the trained booster is still resident
    step = 200_000
    preds = np.concatenate(
        [
            bst.inplace_predict(np.ascontiguousarray(x_te[i : i + step]))
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


def run_catboost(task, x_tr, y_tr, x_te, y_te, reg, curve):
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
    rsm_dropped = False
    min_data_dropped = "min_data" in reg  # symmetric trees don't support it
    if "colsample" in reg:
        kw["rsm"] = reg["colsample"]
    if "l2" in reg:
        kw["l2_leaf_reg"] = reg["l2"]  # catboost default is 3

    t0 = time.perf_counter()
    train_pool = cb.Pool(np.asarray(x_tr), label=y_tr)
    construct_s = time.perf_counter() - t0

    cls = (
        cb.CatBoostClassifier
        if task in ("binary", "multiclass")
        else cb.CatBoostRegressor
    )

    def fit(kw):
        model = cls(**kw)
        t0 = time.perf_counter()
        if curve:
            model.fit(
                train_pool,
                eval_set=cb.Pool(np.asarray(x_te), label=y_te),
                metric_period=reg["eval_every"],
            )
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
            series = vals[next(iter(vals))]
            n = len(series)
            # no per-iteration wall clock exposed; approximate linearly
            curve_pts = [
                [i + 1, train_s * (i + 1) / n, series[i]]
                for i in range(0, n, reg["eval_every"])
            ]

    if task == "binary":
        preds = model.predict_proba(np.asarray(x_te))[:, 1]
    elif task == "multiclass":
        preds = model.predict_proba(np.asarray(x_te))
    else:
        preds = model.predict(np.asarray(x_te))
    return {
        "construct_s": construct_s,
        "train_s": train_s,
        "preds": preds,
        "version": cb.__version__,
        "curve": curve_pts,
        "rsm_dropped": rsm_dropped,
        "min_data_dropped": min_data_dropped,
        "curve_time_approx": bool(curve),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--library",
        required=True,
        choices=[
            "exaboost",
            "exaboost-quant",
            "lightgbm",
            "lightgbm-quant",
            "lightgbm-ocl",
            "xgboost",
            "catboost",
        ],
    )
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--regime", required=True, choices=list(REGIMES))
    ap.add_argument("--kind", required=True)  # warmup | timed1..3 | curve
    ap.add_argument("--out", default=RUNS_JSONL)
    args = ap.parse_args()

    task = DATASETS[args.dataset]["task"]
    reg = REGIMES[args.regime]
    curve = args.kind == "curve"

    rec = {
        "library": args.library,
        "dataset": args.dataset,
        "regime": args.regime,
        "kind": args.kind,
        "status": "ok",
    }
    try:
        x_tr, y_tr, x_te, y_te, extra = load_data(args.dataset)
        rec["n_train"], rec["n_features"] = int(x_tr.shape[0]), int(x_tr.shape[1])
        with ResourceMonitor() as mon:
            t_total = time.perf_counter()
            if args.library.startswith(("exaboost", "lightgbm")):
                r = run_lightgbm(
                    task,
                    x_tr,
                    y_tr,
                    x_te,
                    y_te,
                    reg,
                    quantized=args.library.endswith("-quant"),
                    curve=curve,
                    opencl=args.library == "lightgbm-ocl",
                )
            elif args.library == "xgboost":
                r = run_xgboost(task, x_tr, y_tr, x_te, y_te, reg, curve)
            else:
                r = run_catboost(task, x_tr, y_tr, x_te, y_te, reg, curve)
            total_s = time.perf_counter() - t_total
        preds = r.pop("preds")
        rec.update(r)
        rec["total_s"] = total_s
        rec["trees_per_s"] = reg["rounds"] / r["train_s"]
        rec["gpu_mem_peak_mb"] = mon.gpu_peak_mb
        rec["rss_peak_mb"] = mon.rss_peak_mb
        rec["metrics"] = quality_metrics(task, np.asarray(preds), y_te, extra)
    except Exception:
        rec["status"] = "failed"
        rec["error"] = traceback.format_exc()[-3000:]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "curve"})[:2000])
    sys.exit(0 if rec["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
