"""The canonical Falcata md5 locks, as a permanent script.

These are the two full-scale bit-identity gates the project's verification
discipline is built on (see also tests/gates/lattice.py for the per-commit
small-cell lattice). They need the bench data cache (machine-local; override
with FALCATA_BENCH_CACHE) and an idle GPU.

Locks are md5[:12] over the TREE section of model_to_string() only (up to the
"parameters:" block): the parameters dump moves whenever a config param is
added/renamed -- it did in the 2026-07 planner refactor with zero behavior
change (all 700 covtype trees bit-identical, full-string md5 moved) -- so tree
bytes are the behavior signature.

The LOCKS dict below is the single source of truth for the expected values.
If a value differs: FIRST distrust the build/invocation, not the lock --
rebuild from the exact commit and rerun this exact script. Only re-baseline
with an understood, approved behavior change. A lock of None means
"not yet baselined": the script prints BASELINE <md5> for it and fails.

Usage:
  python tests/gates/canonical.py covtype [--classic]
  python tests/gates/canonical.py numerai
  python tests/gates/canonical.py all
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

CACHE = Path(os.environ.get("FALCATA_BENCH_CACHE", "bench-cache"))

LOCKS = {
    # re-baselined 2026-07-31 (2nd): stochastic rounding is Philox-generated
    # in-kernel from (seed, tree, row) -- machine/thread/GPU-independent by
    # construction (no tables). Same-day 1st re-baseline was the fixed-chunk
    # table fix; Philox replaced the tables entirely.
    # Re-baselined 2026-08-15 for RECORDED COUNTS, not a training change. A CUDA
    # histogram bin carries gradient and hessian but no row count, so the split
    # finder estimated one as round(hessian * num_data / sum_hessians) -- exact
    # only when every hessian is 1. covtype is multiclass, so its leaf_count /
    # internal_count fields were a few rows off and a node disagreed with its own
    # children. The tree now records the data partition's exact counts. Proof it
    # is metadata and nothing else: md5 over the tree text with leaf_count,
    # internal_count and tree_sizes lines removed is 19dfb0a3b2c5 for both this
    # build and 1.0.3 (97401efd76b6 for the classic cell) -- every split,
    # threshold and leaf value across all 700 trees is bit-identical, and test
    # accuracy is 0.91841 either way.
    "covtype": "fad8aaa500f7",
    "covtype-classic": "f2c90e1bfc66",
    # Re-baselined 2026-08-10 for the DATA, not a code change. The bench cache's
    # numerai build was regenerated on 2026-08-07 (the 1224 -> 1226 source bump,
    # which is explicitly not comparable to older archives); the old lock was set
    # 2026-07-31 against the previous build. Proof it is data and not drift: the
    # lock's own commit (07257bd8) run against today's cache produces this same
    # 795599fb164c, and its covtype cell still reproduces cf08d1a953b2 exactly.
    "numerai": "795599fb164c",
}


def _tree_md5(model_str):
    return hashlib.md5(model_str.split("\nparameters:")[0].encode()).hexdigest()[:12]


def run_covtype(classic=False):
    import numpy as np

    import falcata as lgb

    d = CACHE / "covtype"
    X_tr, y_tr = np.load(d / "X_train.npy"), np.load(d / "y_train.npy")
    X_te, y_te = np.load(d / "X_test.npy"), np.load(d / "y_test.npy")
    p = {
        "objective": "multiclass",
        "num_class": 7,
        "learning_rate": 0.1,
        "num_leaves": 1023,
        "max_depth": 10,
        "max_bin": 255,
        "lambda_l2": 1.0,
        "device_type": "cuda",
        "seed": 42,
        "verbose": -1,
        "metric": "None",
        "num_threads": 32,
        "use_quantized_grad": True,
    }
    if classic:
        p["cuda_plan"] = "auto,hybrid:off"
    ds = lgb.Dataset(X_tr, label=y_tr, params=p)
    ds.construct()
    t0 = time.time()
    bst = lgb.train(p, ds, num_boost_round=100)
    t = time.time() - t0
    md5 = _tree_md5(bst.model_to_string())
    pred = bst.predict(X_te).argmax(axis=1)
    quality = float((pred == y_te).mean())
    name = "covtype-classic" if classic else "covtype"
    return name, md5, f"train={t:.2f}s quality={quality:.5f}"


def run_numerai():
    import numpy as np

    import falcata as lgb

    d = CACHE / "numerai"
    meta = json.load(open(d / "meta.json"))
    n, m, tr = meta["n_rows"], meta["n_features"], meta["train_end"]
    X = np.memmap(d / "X.i8.mem", dtype=np.int8, mode="r", shape=(n, m))[:tr]
    y = np.load(d / "y.npy")[:tr]
    p = {
        "objective": "regression",
        "learning_rate": 0.01,
        "num_leaves": 32,
        "max_depth": 5,
        "max_bin": 255,
        "feature_fraction": 0.1,
        "device_type": "cuda",
        "seed": 42,
        "verbose": -1,
        "metric": "None",
        "num_threads": 32,
        "use_quantized_grad": True,
    }
    ds = lgb.Dataset(X, label=y, params=p)
    t0 = time.time()
    bst = lgb.train(p, ds, num_boost_round=20)
    t = time.time() - t0
    md5 = _tree_md5(bst.model_to_string())
    return "numerai", md5, f"train={t:.2f}s"


NUMERAI_V53_DATASET = Path(os.environ.get("FALCATA_NUMERAI_V53", "numerai-v53.dataset"))


def run_numerai_treecount():
    """Full-scale fixedpoint tree-emission gate (nightly; needs ~20GB free VRAM).

    Encodes the 2026-07 production bug: quant_mode=fixedpoint on the real v5.3
    dataset (6.8M x 3555, max_bin=5, h60 hyperparameters) stopped emitting
    trees after ~44 of 40000 and produced a near-constant model. The trigger is
    scale/data-dependent -- probes up to 550k rows with identical
    hyperparameters do NOT reproduce it, so only this full-scale cell catches
    the class. Detectors: exact tree count + non-collapsed late-tree leaves.
    """
    import numpy as np

    import falcata as lgb

    if not NUMERAI_V53_DATASET.exists():
        return "numerai-treecount", "SKIP", f"dataset missing: {NUMERAI_V53_DATASET}"
    parquet = NUMERAI_V53_DATASET.with_suffix(".parquet")
    if not parquet.exists():
        return "numerai-treecount", "SKIP", f"parquet (for labels) missing: {parquet}"
    rounds = 200
    p = {
        "objective": "regression",
        "learning_rate": 0.0015,
        "max_depth": 11,
        "num_leaves": 8192,
        "min_data_in_leaf": 40000,
        "feature_fraction": 0.1,
        "max_bin": 5,
        "quant_mode": "fixedpoint",
        "cuda_precision": "fp32",
        "device_type": "cuda",
        "seed": 42,
        "verbose": -1,
        "metric": "None",
        "num_threads": 16,
    }
    ds = lgb.Dataset(str(NUMERAI_V53_DATASET), params=p)
    # the binary dataset is label-less by design (production swaps labels per
    # target via set_label); train the h60 benchmark target
    import pyarrow.parquet as pq

    y = pq.read_table(parquet, columns=["target_ender_60"]).column(0).to_numpy()
    ds.construct()
    ds.set_label(np.nan_to_num(y.astype(np.float64), nan=0.5))
    t0 = time.time()
    bst = lgb.train(p, ds, num_boost_round=rounds)
    t = time.time() - t0
    n = bst.num_trees()
    if n != rounds:
        return (
            "numerai-treecount",
            "FAIL",
            f"num_trees={n} != {rounds} (early emission stop) train={t:.1f}s",
        )
    late_leaf_max = 0.0
    for tree in range(rounds - 20, rounds):
        txt = bst.model_to_string(start_iteration=tree, num_iteration=1)
        for ln in txt.splitlines():
            if ln.startswith("leaf_value="):
                late_leaf_max = max(late_leaf_max, *(abs(float(v)) for v in ln.split("=", 1)[1].split()))
                break
    if late_leaf_max == 0.0:
        return (
            "numerai-treecount",
            "FAIL",
            f"late trees have all-zero leaves (collapsed model) train={t:.1f}s",
        )
    return (
        "numerai-treecount",
        "PASS",
        f"{rounds} trees, late max|leaf|={late_leaf_max:.3e}, train={t:.1f}s",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["covtype", "numerai", "numerai-treecount", "all"])
    ap.add_argument("--classic", action="store_true", help="covtype with cuda_plan=auto,hybrid:off")
    args = ap.parse_args()

    runs = []
    if args.gate in ("covtype", "all"):
        runs.append(run_covtype(classic=args.classic))
        if args.gate == "all" and not args.classic:
            runs.append(run_covtype(classic=True))
    if args.gate in ("numerai", "all"):
        runs.append(run_numerai())

    failed = False
    for name, md5, info in runs:
        want = LOCKS[name]
        if want is None:
            print(f"BASELINE {name}: tree_md5={md5} {info} -- record in LOCKS after cross-build verification")
            failed = True
            continue
        ok = md5 == want
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: tree_md5={md5} (lock {want}) {info}")
    if args.gate == "numerai-treecount":
        name, verdict, info = run_numerai_treecount()
        print(f"{verdict} {name}: {info}")
        failed |= verdict == "FAIL"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
