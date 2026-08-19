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
CACHE = Path(os.environ.get("FALCATA_BENCH_CACHE", "bench-cache"))
REPEATS = 3
# Fast cells get more repeats. The tier's shortest cell (year-shallow, ~0.39s)
# has historically ranged -4% to +8% around its own median at three repeats --
# the same width as the perf gate's fail threshold, so it fired on its own
# upper tail and taught readers to ignore the gate. Every cell's tail reaches
# ~8%; the short ones just get there often. Median over more samples narrows
# that, and nine repeats of a 0.4s cell costs about four seconds.
MEASURE_BUDGET_SEC = 3.0
MAX_REPEATS = 15

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
    # The production h60 sweep shape: sparse-encoded columns concentrated in
    # one EFB bundle + feature_fraction < 1 + deep wide trees. This is the
    # shape where the packed split read silently un-shipped itself (21358be8,
    # -27% trees/s, fixed 12701bf5) and NO bench cell showed it -- the cache's
    # numerai sample does not reproduce the sparse-bundle structure, so this
    # cell trains the real binary dataset (newest round on disk). It is the
    # absolute-throughput trend complement to sparse_column_view.py's
    # packed-vs-gather ratio assertion.
    (
        "bench/numerai-h60-sparse",
        "numerai-h60",
        {
            "objective": "regression",
            "learning_rate": 0.003375,
            "num_leaves": 250,
            "max_depth": 12,
            "min_data_in_leaf": 40_000,
            "feature_fraction": 0.15,
            "max_bin": 5,
        },
        60,
    ),
]

NUMERAI_DATA_DIR = Path.home() / "Documents/numerai/data"


def latest_numerai_dataset():
    """The newest round's binary dataset (<round>_int8nan.dataset); the h60
    cell trains whatever the sweep currently trains."""
    rounds = []
    for p in NUMERAI_DATA_DIR.glob("*_int8nan.dataset"):
        prefix = p.name.split("_", 1)[0]
        if prefix.isdigit():
            rounds.append((int(prefix), p))
    return max(rounds)[1] if rounds else None


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


def run_h60_cell(cid, params, rounds):
    """Like run_cell, but on the real numerai binary dataset: construct is a
    12GB binary load + device ingest, and the label is a fixed seeded vector
    (the file's label may be a placeholder; timing does not care which)."""
    import numpy as np

    import falcata as lgb

    path = latest_numerai_dataset()
    p = {**BASE, **params}
    cs, ts = [], []
    label = None
    for _ in range(REPEATS):
        t0 = time.monotonic()
        ds = lgb.Dataset(str(path), params=p)
        ds.construct()
        if label is None:
            label = np.random.default_rng(42).standard_normal(ds.num_data()).astype(np.float32)
        ds.set_label(label)
        t1 = time.monotonic()
        bst = lgb.train(p, ds, num_boost_round=rounds)
        t2 = time.monotonic()
        assert bst.num_trees() == rounds
        cs.append(t1 - t0)
        ts.append(t2 - t1)
        del bst, ds
    return {
        "id": cid,
        "ok": True,
        "dataset": path.name,
        "construct_sec": round(statistics.median(cs), 4),
        "train_sec": round(statistics.median(ts), 4),
    }


def run_cell(cid, dataset, params, rounds):
    import falcata as lgb

    if dataset == "numerai-h60":
        return run_h60_cell(cid, params, rounds)
    X, y = load(dataset)
    p = {**BASE, **params}
    cs, ts = [], []
    repeats = REPEATS
    for attempt in range(MAX_REPEATS):
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
        if attempt == 0:
            # Budget the repeats off the first one: long cells stay at three.
            per_run = max(t2 - t0, 1e-6)
            repeats = max(REPEATS, min(MAX_REPEATS, int(MEASURE_BUDGET_SEC / per_run) + 1))
        if attempt + 1 >= repeats:
            break
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
        if dataset == "numerai-h60":
            if latest_numerai_dataset() is None:
                # Hard-fail, not skip: on this box the sweep data is always
                # present, and a silent skip is how this shape went unwatched.
                results[cid] = {
                    "id": cid,
                    "ok": False,
                    "error": f"no *_int8nan.dataset under {NUMERAI_DATA_DIR}",
                }
                print(results[cid])
                continue
        elif not (CACHE / dataset).exists():
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
