"""The canonical ExaBoost md5 locks, as a permanent script.

These are the two full-scale bit-identity gates the project's verification
discipline is built on (see also tests/gates/lattice.py for the per-commit
small-cell lattice). They need the bench data cache (machine-local; override
with EXABOOST_BENCH_CACHE) and an idle GPU.

Expected values (post-#13 tie-break baseline, re-baselined 2026-07-16):
  covtype 1023/10 quant hybrid      -> 1bfd2d7aed5f   (quality 0.91800)
  covtype 1023/10 quant hybrid:off  -> 26852449fbac
  numerai int8 quant example-shape  -> 763c75c0d9cb

If a value differs: FIRST distrust the build/invocation, not the lock --
rebuild from the exact commit and rerun this exact script. Only re-baseline
with an understood, approved behavior change.

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
    "covtype": "1bfd2d7aed5f",
    "covtype-classic": "26852449fbac",
    "numerai": "763c75c0d9cb",
}


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
    md5 = hashlib.md5(bst.model_to_string().encode()).hexdigest()[:12]
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
    md5 = hashlib.md5(bst.model_to_string().encode()).hexdigest()[:12]
    return "numerai", md5, f"train={t:.2f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["covtype", "numerai", "all"])
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
        ok = md5 == want
        failed |= not ok
        print(
            f"{'PASS' if ok else 'FAIL'} {name}: model_md5={md5} (lock {want}) {info}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
