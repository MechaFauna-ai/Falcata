"""Nightly full-scale perf tier: real-dataset timed cells.

The per-commit lattice cannot resolve single-digit-percent perf regressions
(1-second cells, +/-10% noise). This tier trains REAL bench-cache datasets at
full benchmark scale, 3 repeats, median construct/train per cell -- enough
resolution for the ~2%-class regressions that so far only same-session A/B
caught (e.g. the colcap non-quant regression, fixed in 35a7e683, whose shape
is the numerai-nonquant-fp32 cell here).

Emits the lattice_results.json schema so perf_gate.py consumes it unchanged:
  python tests/gates/bench_tier.py --out bench_results.json
  python tests/gates/perf_gate.py --results bench_results.json \
      --baseline-file ~/.cache/falcata-gates/bench_baseline.json \
      --warn-pct 4 --fail-pct 8 --record
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

# Machine-local bench data. The directory keeps its pre-rename name because it
# holds ~105GB of cached datasets; override to point elsewhere.
CACHE = Path(os.environ.get("FALCATA_BENCH_CACHE", "/home/felixjk/Documents/exaboost-bench/data/cache"))
REPEATS = 3

BASE = {
    "device_type": "cuda",
    "seed": 42,
    "verbose": -1,
    "metric": "None",
    "num_threads": 32,
}

# (cell id, dataset, params). numerai-nonquant-fp32 is the colcap-regression
# shape (low-bin many-column, ff<1, fp32, no quant); covtype-deep-quant is the
# historical headline cell; numerai-example-quant is the compact-quant winner.
CELLS = [
    (
        "bench/covtype-deep-quant",
        "covtype",
        {
            "objective": "multiclass",
            "num_class": 7,
            "learning_rate": 0.1,
            "num_leaves": 1023,
            "max_depth": 10,
            "max_bin": 255,
            "lambda_l2": 1.0,
            "use_quantized_grad": True,
        },
        100,
    ),
    (
        "bench/covtype-shallow-nonquant",
        "covtype",
        {
            "objective": "multiclass",
            "num_class": 7,
            "learning_rate": 0.1,
            "num_leaves": 63,
            "max_depth": 6,
            "max_bin": 255,
            "lambda_l2": 1.0,
        },
        100,
    ),
    (
        "bench/year-shallow-fp32",
        "year",
        {
            "objective": "regression",
            "learning_rate": 0.1,
            "num_leaves": 63,
            "max_depth": 6,
            "max_bin": 255,
            "cuda_precision": "fp32",
        },
        100,
    ),
    (
        "bench/numerai-example-quant",
        "numerai",
        {
            "objective": "regression",
            "learning_rate": 0.01,
            "num_leaves": 32,
            "max_depth": 5,
            "max_bin": 255,
            "feature_fraction": 0.1,
            "use_quantized_grad": True,
        },
        100,
    ),
    (
        "bench/numerai-nonquant-fp32",
        "numerai",
        {
            "objective": "regression",
            "learning_rate": 0.01,
            "num_leaves": 32,
            "max_depth": 5,
            "max_bin": 255,
            "feature_fraction": 0.1,
            "cuda_precision": "fp32",
        },
        100,
    ),
]


def load(dataset):
    import numpy as np

    d = CACHE / dataset
    if dataset == "numerai":
        meta = json.load(open(d / "meta.json"))
        n, m, tr = meta["n_rows"], meta["n_features"], meta["train_end"]
        X = np.memmap(d / "X.i8.mem", dtype=np.int8, mode="r", shape=(n, m))[:tr]
        y = np.load(d / "y.npy")[:tr]
    else:
        X, y = np.load(d / "X_train.npy"), np.load(d / "y_train.npy")
    return X, y


def run_cell(cid, dataset, params, rounds):
    import falcata as lgb

    X, y = load(dataset)
    p = {**BASE, **params}
    cs, ts = [], []
    for _ in range(REPEATS):
        t0 = time.monotonic()
        ds = lgb.Dataset(X, label=y, params=p)
        ds.construct()
        t1 = time.monotonic()
        bst = lgb.train(p, ds, num_boost_round=rounds)
        t2 = time.monotonic()
        assert bst.num_trees() == rounds * p.get("num_class", 1)
        cs.append(t1 - t0)
        ts.append(t2 - t1)
        del bst, ds
    return {
        "id": cid,
        "ok": True,
        "construct_sec": round(statistics.median(cs), 4),
        "train_sec": round(statistics.median(ts), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "bench_results.json"))
    args = ap.parse_args()

    results = {}
    t0 = time.monotonic()
    for cid, dataset, params, rounds in CELLS:
        if not (CACHE / dataset).exists():
            print(f"skip {cid}: cache missing {CACHE / dataset}")
            continue
        try:
            r = run_cell(cid, dataset, params, rounds)
        except Exception as e:  # noqa: BLE001 - report and continue; gate decides
            r = {"id": cid, "ok": False, "error": f"{type(e).__name__}: {e}"}
        results[cid] = r
        print(r)
    Path(args.out).write_text(
        json.dumps(
            {"results": results, "elapsed_sec": round(time.monotonic() - t0, 1)},
            indent=1,
        )
    )
    print(f"-> {args.out}")
    return 0 if all(r.get("ok") for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
