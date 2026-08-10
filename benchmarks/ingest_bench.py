"""Native int8 ingestion benchmark (Falcata-only, separate from the matrix).

The cross-library matrix feeds every library the same float32 bits for
fairness, so Falcata's native small-int path (int8/int16 zero-copy through
the C API + LUT binning) is measured here instead: Dataset construct time,
Booster create time, first-trees time, and peak host RSS on numerai, fed from
the float32 memmap vs its int8 twin (``datasets.py numerai-int8``). The two
flows produce identical models (bins are byte-identical by construction).

Run inside the Falcata venv::

    python benchmarks/ingest_bench.py               # 3 repeats each, medians
    python benchmarks/ingest_bench.py --repeats 1

Each measurement runs in a fresh process (clean GPU + page-cache-warm parity
with the matrix); records append to ``<workspace>/results/ingest.jsonl``.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

from common import CACHE_DIR, REGIMES, RESULTS_DIR, SEED

INGEST_JSONL = os.path.join(RESULTS_DIR, "ingest.jsonl")
WARMUP_TREES = 5


def measure_one(kind):
    """Run in a fresh process: construct/create/first-trees timings."""
    import resource
    import time

    import lightgbm as lgb
    import numpy as np

    d = os.path.join(CACHE_DIR, "numerai")
    meta = json.load(open(os.path.join(d, "meta.json")))
    shape = (meta["n_rows"], meta["n_features"])
    n = meta["train_end"]
    if kind == "f32":
        x = np.memmap(os.path.join(d, "X.f32.mem"), np.float32, "r", shape=shape)
    else:
        path = os.path.join(d, "X.i8.mem")
        if not os.path.exists(path):
            sys.exit("ingest_bench: run `datasets.py numerai-int8` first")
        x = np.memmap(path, np.int8, "r", shape=shape)
    y = np.load(os.path.join(d, "y.npy"))[:n]

    reg = REGIMES["numerai"]
    params = {
        "objective": "regression",
        "learning_rate": reg["lr"],
        "num_leaves": reg["leaves"],
        "max_depth": reg["depth"],
        "feature_fraction": reg["colsample"],
        "max_bin": 255,
        "device_type": "cuda",
        "num_threads": os.cpu_count(),
        "seed": SEED,
        "verbose": -1,
        "metric": "None",
    }

    t0 = time.perf_counter()
    dtrain = lgb.Dataset(x[:n], label=y, params=params)
    dtrain.construct()
    construct_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    bst = lgb.Booster(params=params, train_set=dtrain)
    create_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(WARMUP_TREES):
        bst.update()
    trees_s = time.perf_counter() - t0

    print(
        json.dumps(
            {
                "kind": kind,
                "construct_s": round(construct_s, 2),
                "create_s": round(create_s, 2),
                f"trees{WARMUP_TREES}_s": round(trees_s, 2),
                "peak_rss_gb": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1
                ),
                "version": lgb.__version__,
            }
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--one", choices=["f32", "i8"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        measure_one(args.one)
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {"f32": [], "i8": []}
    for kind in ("f32", "i8"):
        for rep in range(args.repeats):
            out = (
                subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--one", kind],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                .stdout.strip()
                .splitlines()[-1]
            )
            rec = json.loads(out)
            rec["repeat"] = rep
            results[kind].append(rec)
            with open(INGEST_JSONL, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{kind} repeat {rep}: {out}", flush=True)

    print(f"\nmedians over {args.repeats} repeats (numerai, Falcata cuda):")
    keys = ["construct_s", "create_s", f"trees{WARMUP_TREES}_s", "peak_rss_gb"]
    header = "flow      " + "".join(f"{k:>14}" for k in keys)
    print(header)
    for kind in ("f32", "i8"):
        med = [statistics.median(r[k] for r in results[kind]) for k in keys]
        print(f"{kind:<10}" + "".join(f"{v:>14.2f}" for v in med))


if __name__ == "__main__":
    main()
