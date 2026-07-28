"""The canonical ExaBoost md5 locks, as a permanent script.

These are the two full-scale bit-identity gates the project's verification
discipline is built on (see also tests/gates/lattice.py for the per-commit
small-cell lattice). They need the bench data cache (machine-local; override
with EXABOOST_BENCH_CACHE) and an idle GPU.

Locks are md5[:12] over the TREE section of model_to_string() only (up to the
"parameters:" block): the parameters dump moves whenever a config param is
added/renamed -- it did in the 2026-07 planner refactor with zero behavior
change (all 700 covtype trees bit-identical, full-string md5 moved) -- so tree
bytes are the behavior signature. Historical full-string locks: covtype
1bfd2d7aed5f / classic 26852449fbac (pre-refactor builds only).

Expected values (post-#13 tie-break baseline; tree-only since 2026-07-28):
  covtype 1023/10 quant hybrid      -> 20cb576f6758   (quality 0.91800)
  covtype 1023/10 quant hybrid:off  -> cross-build-verified then recorded
  numerai int8 quant example-shape  -> cross-build-verified then recorded

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

CACHE = Path(
    os.environ.get(
        "EXABOOST_BENCH_CACHE", "/home/felixjk/Documents/exaboost-bench/data/cache"
    )
)

LOCKS = {
    "covtype": "20cb576f6758",
    "covtype-classic": None,  # pending cross-build verification (task #44)
    "numerai": None,  # pending cross-build verification (task #44)
}


def _tree_md5(model_str):
    return hashlib.md5(model_str.split("\nparameters:")[0].encode()).hexdigest()[:12]


def run_covtype(classic=False):
    import numpy as np
    import lightgbm as lgb

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
    import lightgbm as lgb

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


NUMERAI_V53_DATASET = Path(
    os.environ.get(
        "EXABOOST_NUMERAI_V53",
        "/home/felixjk/Documents/numerai/data/1224_int8nan.dataset",
    )
)


def run_numerai_treecount():
    """Full-scale fixedpoint tree-emission gate (nightly; needs ~20GB free VRAM).

    Encodes the 2026-07 production bug: quant_mode=fixedpoint on the real v5.3
    dataset (6.8M x 3555, max_bin=5, h60 hyperparameters) stopped emitting
    trees after ~44 of 40000 and produced a near-constant model. The trigger is
    scale/data-dependent -- probes up to 550k rows with identical
    hyperparameters do NOT reproduce it, so only this full-scale cell catches
    the class. Detectors: exact tree count + non-collapsed late-tree leaves.
    """
    import lightgbm as lgb

    if not NUMERAI_V53_DATASET.exists():
        return "numerai-treecount", "SKIP", f"dataset missing: {NUMERAI_V53_DATASET}"
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
                late_leaf_max = max(
                    late_leaf_max,
                    max(abs(float(v)) for v in ln.split("=", 1)[1].split()),
                )
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
    ap.add_argument(
        "--classic", action="store_true", help="covtype with cuda_plan=auto,hybrid:off"
    )
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
            print(
                f"BASELINE {name}: tree_md5={md5} {info} -- record in LOCKS after cross-build verification"
            )
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
