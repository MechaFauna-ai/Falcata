"""Serialized models must be internally consistent and route-faithful.

Guards the stale-candidate resurrection class of bug: the CUDA best-split
sync reductions used to default their nothing-found read index to the
smaller role's task-0 output slot. Whenever a leaf's fresh search found no
valid split while its sibling was skipped (a sibling parked at exactly
min_data_in_leaf skips the find, leaving its output slots stale), the stale
slot's surviving is_valid flag resurrected an EARLIER level's candidate --
tagged with the lowest real feature index -- as the leaf's best split. The
partition then applied that split to a leaf it was never scored on:
recorded leaf_count below the min_data floor, leaf_weight disagreeing with
leaf_count, and descendant leaves reachable by (almost) no rows.

The gate trains in the regime that fires the bug -- budget-limited configs
(selective grow-then-prune), tiny late-boosting gains, 5-valued features
with NaN, a binding min_data floor -- and asserts, per parameterized case:

  1. RECORDED COUNTS: under L2 the hessian is 1 per row, so every leaf's
     serialized leaf_weight must equal its leaf_count, and no leaf may sit
     below min_data_in_leaf.
  2. ROUND-TRIP FIDELITY: in-process predictions == text-save/reload
     predictions == falb-binary (pickle) reload predictions, bit-exact.
  3. REACHABILITY: routing the training matrix through the reloaded model
     must reproduce every leaf's recorded leaf_count exactly -- zero dead
     leaves, zero occupancy drift.

Cases cover quant_mode=fixedpoint and quant_mode=none (the vulnerable sync
kernels are shared), both on CUDA. Runtime is a few minutes.

A third case appends 0/1 and three-valued columns that exclusive feature
bundling merges into shared groups (one-hot blocks, mostly-one columns,
columns that collide on a few rows, columns whose most frequent value is not
zero). The 5-valued columns above are never sparse enough to bundle, so the
first two cases cannot see a row that bundling puts in the wrong bin; that is
how rows of bundled 0/1 columns crossing their splits (2,501 row-visits in
300 iterations of one production fit) went unnoticed. The case asserts that
the dataset really bundles, so it cannot stop covering EFB silently.

Usage:
  python tests/gates/leaf_integrity.py
  python tests/gates/leaf_integrity.py --trees 2000   # quicker smoke
"""

import argparse
import pickle
import re
import sys

import numpy as np


def synthetic_data(rows, feats, seed=0, nan_frac=0.15):
    """Numerai-shaped: ~5 distinct small-integer values per feature plus NaN,
    a mostly-noise target so boosting reaches the tiny-gain regime fast."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 5, size=(rows, feats)).astype(np.float32)
    # skew some columns so floor-pinned children appear often
    for j in range(0, feats, 7):
        col = x[:, j]
        col[rng.random(rows) < 0.75] = 2.0
        x[:, j] = col
    x[rng.random((rows, feats)) < nan_frac] = np.nan
    signal = np.where(np.isnan(x[:, 0]), 0.0, x[:, 0]) - np.where(np.isnan(x[:, 1]), 0.0, x[:, 1])
    y = (0.05 * signal + rng.normal(size=rows)).astype(np.float32)
    return x, y


def bundled_columns(rows, seed=0):
    """0/1 and three-valued columns for exclusive feature bundling: one-hot
    blocks (exclusive, must bundle), mostly-one columns (most frequent bin is
    not the zero bin), pairs of rare columns that share a few rows (fewer than
    the rows / 10000 LightGBM let a bundle keep), and 0 / 0.5 / 1 columns with
    a partner that is 1 on some of their early ones (bundled wrongly when the
    search pairs a value with another row's id)."""
    rng = np.random.default_rng(seed + 1000)
    cols = []
    for _ in range(3):  # one-hot blocks of 10 levels
        level = rng.integers(0, 10, size=rows)
        cols.extend((level == k).astype(np.float32) for k in range(10))
    for p in (0.002, 0.01):  # mostly ones
        cols.append((rng.random(rows) >= p).astype(np.float32))
    shared = max(1, rows // 40000)
    for _ in range(6):  # rare pairs that share `shared` rows
        a = rng.choice(rows, rows // 50, replace=False)
        rest = np.setdiff1d(np.arange(rows), a)
        b = np.concatenate([rng.choice(rest, rows // 50 - shared, replace=False), a[:shared]])
        for idx in (a, b):
            c = np.zeros(rows, np.float32)
            c[idx] = 1
            cols.append(c)
    for _ in range(3):  # most frequent value 0.5, a partner on early ones
        v = rng.choice([0.0, 0.5, 1.0], size=rows, p=[0.05, 0.9, 0.05]).astype(np.float32)
        ones = np.where(v == 1.0)[0]
        early = ones[ones < rows // 2]
        partner = np.zeros(rows, np.float32)
        partner[early[: len(early) // 4]] = 1
        cols.extend([v, partner])
    return np.stack(cols, axis=1)


def num_groups(dataset):
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        dataset._dump_text(path)
        with open(path) as f:
            for line in f:
                if line.startswith("num_groups:"):
                    return int(line.split(":")[1])
    finally:
        os.remove(path)
    raise RuntimeError("no num_groups line in the dataset dump")


def parse_model(text):
    floor = int(re.search(r"\[min_data_in_leaf:\s*(\d+)\]", text).group(1))
    counts, weights = [], []
    for block in text.split("\nTree=")[1:]:
        c = re.search(r"\nleaf_count=([^\n]*)", block)
        w = re.search(r"\nleaf_weight=([^\n]*)", block)
        counts.append(np.array([int(v) for v in c.group(1).split()]) if c else np.array([], dtype=np.int64))
        weights.append(np.array([float(v) for v in w.group(1).split()]) if w else np.array([]))
    return floor, counts, weights


def check_recorded_counts(name, floor, counts, weights):
    mismatched = subfloor = total = 0
    for c, w in zip(counts, weights):
        total += c.size
        if c.size != w.size:
            mismatched += max(c.size, w.size)
            continue
        mismatched += int((np.abs(w - c) > 0.5).sum())
        subfloor += int((c < floor).sum())
    ok = total > 0 and mismatched == 0 and subfloor == 0
    print(
        f"{'PASS' if ok else 'FAIL'} {name} recorded counts: leaves={total} "
        f"weight!=count={mismatched} below_floor={subfloor} (floor {floor})"
    )
    if total == 0:
        print(f"  {name}: no leaves recorded -- the run proves nothing")
    return ok


def check_round_trip(name, booster, text, x_probe):
    p_live = booster.predict(x_probe)
    import falcata

    b_text = falcata.Booster(model_str=text)
    p_text = b_text.predict(x_probe)
    b_bin = pickle.loads(pickle.dumps(booster))  # falb binary payload
    p_bin = b_bin.predict(x_probe)
    text_ok = np.array_equal(p_live, p_text)
    bin_ok = np.array_equal(p_live, p_bin)
    ok = text_ok and bin_ok
    print(f"{'PASS' if ok else 'FAIL'} {name} round trip: text bit-identical={text_ok} binary bit-identical={bin_ok}")
    if not text_ok:
        print(f"  max text diff {np.max(np.abs(p_live - p_text)):.3e}")
    if not bin_ok:
        print(f"  max binary diff {np.max(np.abs(p_live - p_bin)):.3e}")
    return ok, b_text


def check_reachability(name, b_text, x, counts, chunk=100_000):
    occ = [np.zeros(c.size, dtype=np.int64) for c in counts]
    for s in range(0, len(x), chunk):
        leaves = np.asarray(b_text.predict(x[s : s + chunk], pred_leaf=True))
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        for t in range(leaves.shape[1]):
            np.add.at(occ[t], leaves[:, t].astype(np.int64), 1)
    mismatch = dead = 0
    for c, o in zip(counts, occ):
        if c.size != o.size:
            mismatch += max(c.size, o.size)
            continue
        mismatch += int((c != o).sum())
        dead += int((o == 0).sum())
    ok = mismatch == 0 and dead == 0
    print(f"{'PASS' if ok else 'FAIL'} {name} reachability: occupancy!=recorded={mismatch} dead_leaves={dead}")
    return ok


# Budget-limited shape (num_leaves < 2^max_depth) so the selective
# grow-then-prune flow -- the production numerai path -- builds the trees;
# feature_fraction < 1 so task compaction and per-tree masks are live; the
# floor and the mostly-noise target put late trees in the tiny-gain regime
# where searches come back empty next to floor-pinned siblings.
def case_params(quant_mode, floor):
    return {
        "objective": "regression",
        "device": "cuda",
        "quant_mode": quant_mode,
        "cuda_precision": "fp32",
        "learning_rate": 0.02,
        "num_leaves": 16384,
        "max_depth": 15,
        "min_data_in_leaf": floor,
        "feature_fraction": 0.15,
        "max_bin": 255,
        "seed": 7,
        "verbosity": -1,
        "num_threads": 16,
    }


def run_case(lgb, name, quant_mode, rows, feats, trees, seed, bundled=False):
    x, y = synthetic_data(rows, feats, seed=seed)
    if bundled:
        b = bundled_columns(rows, seed=seed)
        x = np.ascontiguousarray(np.hstack([x[:, : feats // 4], b]))
        y = (y + 0.3 * (b[:, :30].argmax(1) % 3) + 0.5 * (b[:, 30:] @ np.linspace(1.0, 2.0, b.shape[1] - 30))).astype(
            np.float32
        )
    floor = max(1, rows // 56)  # production min_data-to-rows ratio
    params = case_params(quant_mode, floor)
    dataset = lgb.Dataset(x, label=y, params=params, free_raw_data=False)
    booster = lgb.train(params, dataset, num_boost_round=trees)
    text = booster.model_to_string()
    floor_read, counts, weights = parse_model(text)
    ok = check_recorded_counts(name, floor_read, counts, weights)
    if bundled:
        groups = num_groups(dataset)
        bundles = groups < x.shape[1]
        print(f"{'PASS' if bundles else 'FAIL'} {name} bundling: {x.shape[1]} features in {groups} groups")
        ok = ok and bundles
    rt_ok, b_text = check_round_trip(name, booster, text, x[:50_000])
    reach_ok = check_reachability(name, b_text, x, counts)
    del booster
    return ok and rt_ok and reach_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--features", type=int, default=2000)
    parser.add_argument("--trees", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import falcata as lgb

    results = [
        run_case(lgb, "fixedpoint", "fixedpoint", args.rows, args.features, args.trees, args.seed),
        run_case(lgb, "quant-none", "none", args.rows, args.features, args.trees, args.seed + 1),
        run_case(lgb, "bundled", "fixedpoint", args.rows, args.features, args.trees, args.seed + 2, bundled=True),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
