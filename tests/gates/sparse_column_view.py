"""Per-tree feature sampling must not change which rows a split routes.

Falcata keeps two device representations of the same bins: a row-major matrix
the histograms are built from, and per-column buffers the split-apply kernels
read. They coincide on dense columns and NOT on a column whose bins came from a
sparse bin iterator -- that one spells the feature's most-frequent bin as 0,
where the row matrix carries the bin's real index. The per-tree compact view
(built whenever feature_fraction < 1) is gathered from the ROW matrix and
published as the per-column pointers, so serving a sparse column from it made
the apply route that feature's rows by the wrong rule.

The gate is two invariants that hold for ANY dataset and cost one short run:

  1. Under an L2 objective the hessian is exactly 1 per row, so a leaf's
     recorded sum of hessians IS its row count. leaf_weight != leaf_count means
     the split finder and the data partition disagree about which rows the leaf
     holds -- which is what a mis-served column produces, long before the leaf
     values look wrong.
  2. min_data_in_leaf is a hard floor. A tree that records a leaf below it has
     already routed rows by a rule the finder did not score.

Both failed on the numerai h60 shape (6,824,864 x 3,555, feature_fraction=0.15,
15 sparse columns from one 30-feature bundle): 64 of 1,993 leaves disagreed and
54 sat below a 40,000 floor, some holding zero rows, and training reached inf
leaf values by tree ~1500.

Datasets are taken in this order and every one that exists is checked:
  --dataset PATH             (repeatable)
  $FALCATA_SPARSE_GATE_DATA  (colon-separated paths)
  the numerai binary dataset, if this machine has it
plus a synthetic dense case that always runs, so the gate still exercises the
sampled path where no such dataset exists. A dataset carrying sparse columns is
what makes this gate a regression test for THIS bug rather than a smoke test;
`--require-sparse-dataset` fails instead of skipping when none is present.

Usage:
  python tests/gates/sparse_column_view.py
  python tests/gates/sparse_column_view.py --dataset /path/to/x.dataset
  FALCATA_SPARSE_GATE_DATA=/a.dataset:/b.dataset python tests/gates/sparse_column_view.py
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

NUMERAI_DATASET = Path.home() / "Documents/numerai/data/1227_int8nan.dataset"

# feature_fraction well under 1 so the per-tree compact view is built; a leaf
# floor big enough that a violation cannot be a rounding artefact
PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "device_type": "cuda",
    "learning_rate": 0.01,
    "num_leaves": 512,
    "max_depth": 10,
    "feature_fraction": 0.15,
    "max_bin": 5,
    "verbosity": -1,
    "num_threads": 16,
    "seed": 0,
}
ROUNDS = 20
# five levels mapped to N(0,1) quantiles: a label independent of the features,
# which is the harshest case for a split scored on the wrong rows
GAUSS_LEVELS = np.array([-1.6448536269514729, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722])


def leaf_stats(booster):
    """(leaves, leaves whose weight != count, smallest leaf, worst relative gap)."""
    leaves = mismatched = 0
    smallest = np.inf
    worst = 0.0
    for block in booster.model_to_string().split("\nTree=")[1:]:
        counts = re.search(r"\nleaf_count=([^\n]*)", block)
        weights = re.search(r"\nleaf_weight=([^\n]*)", block)
        if not counts or not weights:
            continue
        c = np.array([float(v) for v in counts.group(1).split()])
        w = np.array([float(v) for v in weights.group(1).split()])
        gap = np.abs(w - c) / np.maximum(c, 1.0)
        leaves += c.size
        mismatched += int((gap > 1e-6).sum())
        smallest = min(smallest, c.min())
        worst = max(worst, gap.max())
    return leaves, mismatched, smallest, worst


def check(name, booster, min_data_in_leaf):
    leaves, mismatched, smallest, worst = leaf_stats(booster)
    ok = leaves > 0 and mismatched == 0 and smallest >= min_data_in_leaf
    print(
        f"{'PASS' if ok else 'FAIL'} {name}: leaves={leaves} weight!=count={mismatched} "
        f"smallest_leaf={smallest:,.0f} (floor {min_data_in_leaf:,}) worst_rel={worst:.4g}"
    )
    if leaves == 0:
        print(f"  {name}: trained no splittable tree -- the run proves nothing, tighten the data or params")
    return ok


def run_dataset(path, lgb):
    """Train on an existing binary dataset. Its label may be a placeholder, so
    a target is supplied here; the invariants do not depend on which one."""
    params = dict(PARAMS)
    ds = lgb.Dataset(str(path), params=params)
    ds.construct()
    num_data = ds.num_data()
    params["min_data_in_leaf"] = max(1000, num_data // 170)
    ds.set_label(GAUSS_LEVELS[np.random.default_rng(0).integers(0, 5, size=num_data)])
    booster = lgb.train(params, ds, num_boost_round=ROUNDS)
    ok = check(f"{Path(path).name} ({num_data:,} rows)", booster, params["min_data_in_leaf"])
    del booster
    return ok and check_packed_read_speedup(path, ds, lgb)


# The packed split read's win must survive sparse columns: a per-TREE fallback
# (instead of the per-COLUMN escape SetCompactPackedColumnView implements)
# fires on nearly every tree when a dataset's sparse columns share one EFB
# bundle -- at feature_fraction 0.15 a 30-feature bundle is sampled by
# 1 - 0.85^30 ~ 99.2% of trees -- and silently un-ships the read on exactly the
# dataset class it was built for (measured -27% trees/s on the numerai h60
# shape). Guard the mechanism as a throughput ratio: default plan vs
# split_packed_read:off must beat this floor, at the production h60 config
# (250 leaves / depth 12 / 40k floor) where the gather dominates per-tree
# cost. The measured gap there is 1.10-1.35x depending on window and clocks; a
# ratio near 1.0 means every tree is paying the column-major gather again, so
# the floor sits between the two regimes rather than close to either.
SPEED_ROUNDS = 60
SPEED_WARMUP = 10  # trees before the timer starts (allocations, graph capture)
SPEED_MIN_RATIO = 1.05
SPEED_PARAMS = {**PARAMS, "num_leaves": 250, "max_depth": 12, "min_data_in_leaf": 40_000}


def timed_train(params, ds, lgb):
    booster = lgb.Booster(params=params, train_set=ds)
    for _ in range(SPEED_WARMUP):
        booster.update()
    start = time.perf_counter()
    for _ in range(SPEED_ROUNDS - SPEED_WARMUP):
        booster.update()
    elapsed = time.perf_counter() - start
    del booster
    return (SPEED_ROUNDS - SPEED_WARMUP) / elapsed


def check_packed_read_speedup(path, ds, lgb):
    params = dict(SPEED_PARAMS)
    packed = timed_train(params, ds, lgb)
    gather = timed_train({**params, "cuda_plan": "auto,split_packed_read:off"}, ds, lgb)
    ratio = packed / gather
    ok = ratio >= SPEED_MIN_RATIO
    print(
        f"{'PASS' if ok else 'FAIL'} {Path(path).name} packed split read: "
        f"{packed:.2f} vs {gather:.2f} trees/s (ratio {ratio:.3f}, floor {SPEED_MIN_RATIO})"
    )
    return ok


def run_synthetic(lgb):
    """Always-available case: no sparse columns, so it guards the sampled path
    generally rather than this bug specifically."""
    rows, feats = 200_000, 120
    rng = np.random.default_rng(0)
    X = rng.integers(0, 5, size=(rows, feats)).astype(np.float32)
    for j in range(0, feats, 8):  # a bin-dominated column every eighth feature
        col = X[:, j]
        col[rng.random(rows) < 0.9] = 3.0
        X[:, j] = col
    y = rng.normal(size=rows)
    params = {**PARAMS, "min_data_in_leaf": 2_000, "num_leaves": 64, "max_depth": 8}
    booster = lgb.train(params, lgb.Dataset(X, label=y, params=params), num_boost_round=ROUNDS)
    return check(f"synthetic ({rows:,} rows)", booster, params["min_data_in_leaf"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", default=[], help="binary dataset to check (repeatable)")
    parser.add_argument(
        "--require-sparse-dataset",
        action="store_true",
        help="fail instead of skipping when no real dataset is available",
    )
    args = parser.parse_args()

    import falcata as lgb

    candidates = [Path(p) for p in args.dataset]
    candidates += [Path(p) for p in os.environ.get("FALCATA_SPARSE_GATE_DATA", "").split(":") if p]
    candidates.append(NUMERAI_DATASET)

    present = [p for p in candidates if p.exists()]
    if not present:
        message = (
            "no binary dataset available; only the synthetic case ran. A dataset carrying "
            "sparse-encoded columns is what makes this gate cover the compact-view bug"
        )
        if args.require_sparse_dataset:
            print(f"FAIL {message}")
            return 1
        print(f"SKIP {message}")

    results = [run_dataset(p, lgb) for p in present]
    results.append(run_synthetic(lgb))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
