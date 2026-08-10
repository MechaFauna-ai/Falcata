"""Leave-one-out feature ablation across the benchmark datasets.

For each (dataset, regime) cell: run the full plan (baseline), then flip each
``cuda_plan`` key away from its default, one at a time, and report what the
feature buys -- throughput delta AND quality delta.

Two properties make this more than a perf table:

- Mechanical plan keys are bit-identical BY CONTRACT (same contract the gate
  lattice enforces): any LOO flip of an IDENTITY key whose tree-md5 differs
  from baseline is a CORRECTNESS BUG, flagged loudly, not a trade-off.
  GROWTH keys (hybrid/selective/one_sync) legitimately change the tree -- see
  the separate covtype-classic canonical lock -- so those report the quality
  delta instead.
- Reduced round counts (per-cell) keep the sweep inside a small GPU budget;
  rates were verified stable in rounds for these regimes, so per-tree
  throughput generalizes.

Each cell runs one discarded warmup (cold CUDA context skews the first run by
~10%), then REPEATS timed runs per configuration with the median reported.

Usage:
  benchmarks/ablation.py --list
  benchmarks/ablation.py [--cells covtype-deep numerai-example] [--out FILE]
Emits JSONL records (one per run) and prints a per-cell markdown table.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

CACHE = Path(os.environ.get("FALCATA_BENCH_CACHE", "bench-cache"))
OUT_DEFAULT = Path(__file__).resolve().parent / "ablation_results.jsonl"

#: every cuda_plan key with its default state; LOO flips it to the opposite
ALL_KEYS = {
    "hybrid": True,
    "selective": True,
    "one_sync": True,
    "batch_kernels": True,
    "batch_apply": True,
    "graph_loop": True,
    "graph_quant": False,
    "compact_quant": True,
    "construct_jit": False,
    "fast_rowdata": True,
    "rowdata_4bit": True,
    "gpu_construct": True,
    "efb_precheck": True,
    "batch_reghist": True,
    "batch_wide": True,
    "gh_interleave": True,
    "split_packed_read": True,
    "small_leaf_construct": True,
    "compact_prefill": False,
}

#: growth-strategy keys: changing them changes WHICH tree is built (legit),
#: so they are judged on quality delta, not bit-identity
GROWTH_KEYS = {"hybrid", "selective", "one_sync"}

#: keys whose fallback path breaks exact-gain ties in a different order at
#: scale (verified benign on numerai-deep: different tree md5, holdout corr
#: identical to 5 decimals). Judged like growth keys.
TIEBREAK_KEYS = {"batch_kernels"}

#: keys whose fallback assigns child-leaf indices in a different ORDER
#: (batched apply numbers new leaves level-wise, the per-split fallback in
#: split order): equivalent trees, different node numbering, different file
#: md5. Verified 2026-08-09 on covtype-deep-quant: predictions bit-identical
#: (max |diff| = 0.0) between batch_apply on/off. Judged on quality delta.
RENUMBER_KEYS = {"batch_apply"}

#: keys worth paying for on the expensive numerai-deep cell
DEEP_KEYS = [
    "hybrid",
    "selective",
    "one_sync",
    "batch_kernels",
    "batch_apply",
    "graph_loop",
    "compact_quant",
    "construct_jit",
    "gh_interleave",
    "split_packed_read",
]

BASE = {"device_type": "cuda", "seed": 42, "verbose": -1, "metric": "None", "num_threads": 32}

# cell id -> (dataset, params, rounds, keys). Quantized cells get md5
# verification; the numerai deep cell runs a restricted key list at 300
# rounds (rate-stable; full runs are minutes each).
CELLS = {
    "covtype-deep-quant": (
        "covtype",
        {
            "objective": "multiclass",
            "num_class": 7,
            "learning_rate": 0.1,
            "num_leaves": 1023,
            "max_depth": 10,
            "max_bin": 255,
            "lambda_l2": 1.0,
            "quant_mode": "stochastic",
        },
        100,
        list(ALL_KEYS),
    ),
    "covtype-shallow-quant": (
        "covtype",
        {
            "objective": "multiclass",
            "num_class": 7,
            "learning_rate": 0.1,
            "num_leaves": 63,
            "max_depth": 6,
            "max_bin": 255,
            "lambda_l2": 1.0,
            "quant_mode": "stochastic",
        },
        100,
        list(ALL_KEYS),
    ),
    "year-shallow-quant": (
        "year",
        {
            "objective": "regression",
            "learning_rate": 0.1,
            "num_leaves": 63,
            "max_depth": 6,
            "max_bin": 255,
            "quant_mode": "stochastic",
        },
        100,
        list(ALL_KEYS),
    ),
    "fraud-deep-quant": (
        "fraud",
        {
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 1023,
            "max_depth": 10,
            "max_bin": 255,
            "quant_mode": "stochastic",
        },
        100,
        list(ALL_KEYS),
    ),
    "higgs-shallow-quant": (
        "higgs",
        {
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 63,
            "max_depth": 6,
            "max_bin": 255,
            "quant_mode": "stochastic",
        },
        100,
        list(ALL_KEYS),
    ),
    "epsilon-shallow-quant": (
        "epsilon",
        {
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 63,
            "max_depth": 6,
            "max_bin": 255,
            "quant_mode": "stochastic",
        },
        100,
        list(ALL_KEYS),
    ),
    "numerai-example-quant": (
        "numerai",
        {
            "objective": "regression",
            "learning_rate": 0.01,
            "num_leaves": 32,
            "max_depth": 5,
            "max_bin": 255,
            "feature_fraction": 0.1,
            "quant_mode": "stochastic",
        },
        200,
        list(ALL_KEYS),
    ),
    "numerai-deep-quant": (
        "numerai",
        {
            "objective": "regression",
            "learning_rate": 0.001,
            "num_leaves": 1024,
            "max_depth": 10,
            "max_bin": 255,
            "feature_fraction": 0.1,
            "min_data_in_leaf": 10000,
            "quant_mode": "stochastic",
        },
        300,
        DEEP_KEYS,
    ),
}


def load(dataset):
    import numpy as np

    d = CACHE / dataset
    if dataset == "numerai":
        meta = json.load(open(d / "meta.json"))
        n, m, tr = meta["n_rows"], meta["n_features"], meta["train_end"]
        X = np.memmap(d / "X.i8.mem", dtype=np.int8, mode="r", shape=(n, m))[:tr]
        y = np.load(d / "y.npy")[:tr]
        return X, y, None, None
    X, y = np.load(d / "X_train.npy"), np.load(d / "y_train.npy")
    xt = d / "X_test.npy"
    if xt.exists():
        return X, y, np.load(xt), np.load(d / "y_test.npy")
    return X, y, None, None


def quality(params, bst, X_te, y_te):
    import numpy as np

    if X_te is None:
        return None, None
    step = 500_000
    preds = np.concatenate([bst.predict(np.ascontiguousarray(X_te[i : i + step])) for i in range(0, len(X_te), step)])
    obj = params.get("objective")
    if obj == "multiclass":
        return float((preds.argmax(axis=1) == y_te).mean()), "acc"
    if obj == "binary":
        order = np.argsort(preds)
        ranks = np.empty(len(preds))
        ranks[order] = np.arange(len(preds))
        pos = y_te == 1
        auc = (ranks[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / max(1, pos.sum() * (~pos).sum())
        return float(auc), "auc"
    return float(np.sqrt(np.mean((preds - y_te) ** 2))), "rmse"


REPEATS = 3


def run_one(cell_id, data, params, rounds, plan):
    import statistics

    import falcata as flc

    X, y, X_te, y_te = data
    p = {**BASE, **params, "cuda_plan": plan}
    cs, ts = [], []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        ds = flc.Dataset(X, label=y, params=p)
        ds.construct()
        t1 = time.perf_counter()
        bst = flc.train(p, ds, num_boost_round=rounds)
        t2 = time.perf_counter()
        cs.append(t1 - t0)
        ts.append(t2 - t1)
    md5 = hashlib.md5(bst.model_to_string().split("\nparameters:")[0].encode()).hexdigest()[:12]
    metric, metric_name = quality(p, bst, X_te, y_te)
    train_s = statistics.median(ts)
    return {
        "cell": cell_id,
        "plan": plan,
        "construct_s": round(statistics.median(cs), 3),
        "train_s": round(train_s, 3),
        "trees_per_s": round(rounds / train_s, 2),
        "tree_md5": md5,
        "metric": metric,
        "metric_name": metric_name,
    }


def flip_plan(key):
    return f"auto,{key}:{'off' if ALL_KEYS[key] else 'on'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for cid, (ds, _p, rounds, keys) in CELLS.items():
            print(f"{cid:26s} {ds:10s} rounds={rounds:<4d} keys={len(keys)}")
        return 0

    out = Path(args.out)
    bugs = []
    for cid in args.cells:
        dataset, params, rounds, keys = CELLS[cid]
        if not (CACHE / dataset).exists():
            print(f"skip {cid}: cache missing")
            continue
        data = load(dataset)
        run_one(cid, data, params, min(rounds, 25), "auto")  # discarded warmup
        base = run_one(cid, data, params, rounds, "auto")
        with out.open("a") as f:
            f.write(json.dumps({**base, "key": "BASELINE"}) + "\n")
        print(f"\n## {cid}  (baseline {base['trees_per_s']} trees/s, {base['metric_name']}={base['metric']})")
        print(f"{'key':22s} {'Δ throughput':>13s} {'Δ construct':>12s}  verdict")
        for key in keys:
            r = run_one(cid, data, params, rounds, flip_plan(key))
            with out.open("a") as f:
                f.write(json.dumps({**r, "key": key}) + "\n")
            dtp = (base["trees_per_s"] / r["trees_per_s"] - 1) * 100 if r["trees_per_s"] else 0
            dcon = r["construct_s"] - base["construct_s"]
            same = r["tree_md5"] == base["tree_md5"]
            if key in GROWTH_KEYS or key in TIEBREAK_KEYS or key in RENUMBER_KEYS:
                # different tree is legitimate; judge on quality
                kind = "growth" if key in GROWTH_KEYS else "renumber" if key in RENUMBER_KEYS else "tiebreak"
                if base["metric"] is None or same:
                    note = "same tree" if same else f"{kind} key (no holdout metric)"
                else:
                    dm = r["metric"] - base["metric"]
                    note = f"{kind} key, Δ{base['metric_name']}={dm:+.5f}"
            elif same:
                note = "bit-identical"
            else:
                note = "*** MD5 DIFFERS — CORRECTNESS BUG ***"
                bugs.append((cid, key, base["tree_md5"], r["tree_md5"]))
            print(f"{key:22s} {dtp:+12.1f}% {dcon:+11.2f}s  {note}")
    if bugs:
        print(f"\n{len(bugs)} NON-BIT-IDENTICAL IDENTITY-KEY FLIPS (bugs):")
        for b in bugs:
            print("  ", b)
        return 1
    print(f"\nresults -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
