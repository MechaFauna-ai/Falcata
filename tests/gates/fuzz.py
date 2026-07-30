"""Nightly config x shape fuzzer.

Samples random (dataset shape, dtype, NaN fraction, sparsity) x (tree shape,
quant mode, cuda_plan flips, objective) combinations -- exactly the "some
combination of certain configs and certain dataset dimensions" class of bug --
and checks each one for:

  1. validity: training completes, model round-trips, predictions finite,
     tree count correct (via the same worker asserts as the lattice);
  2. CUDA determinism (quant modes): the identical spec trained twice must be
     bit-identical;
  3. CPU parity (tolerant): the same spec trained on device_type=cpu must
     reach a held-out metric within tolerance of the CUDA model (trees are not
     required to be identical; a large metric gap means one side built
     garbage).

Every failing spec prints a self-contained JSON repro. Promote it into
fuzz_corpus.json (a list of spec objects) and it reruns first on every future
fuzz run, forever -- the corpus grows from real bugs.

Usage:
  python tests/gates/fuzz.py --minutes 30 [--seed N]
Seed defaults to the UTC date, so each nightly explores new ground while any
failure reproduces with --seed.
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

GATES_DIR = Path(__file__).resolve().parent
CORPUS_FILE = GATES_DIR / "fuzz_corpus.json"
CPU_METRIC_TOLERANCE = 0.10  # relative; cpu-vs-cuda held-out metric gap


def sample_spec(rng):
    n = int(rng.choice([2000, 5000, 12000, 40000]))
    m = int(rng.choice([5, 20, 80, 300, 800]))
    objective = str(rng.choice(["regression", "binary", "multiclass"]))
    quant_mode = str(rng.choice(["stochastic", "stochastic", "fixedpoint", "none"]))
    plan = ["auto"]
    for key in [
        "graph_loop",
        "compact_quant",
        "fast_rowdata",
        "rowdata_4bit",
        "gpu_construct",
        "efb_precheck",
    ]:
        if rng.random() < 0.15:
            plan.append(f"{key}:off")
    if rng.random() < 0.15:
        plan.append("construct_jit:on")
    if rng.random() < 0.15:
        plan.append("compact_prefill:on")
    if rng.random() < 0.10:
        plan.append("hybrid:off")
    if quant_mode == "fixedpoint" and rng.random() < 0.20:
        plan.append("robust_scale:off")
    spec = {
        "n": n,
        "m": m,
        "learning_rate": float(rng.choice([0.1, 0.1, 0.01, 0.0015])),
        "dtype": str(rng.choice(["float64", "int8", "int16"])),
        "nan_frac": float(rng.choice([0.0, 0.0, 0.1, 0.4])),
        "sparsity": float(rng.choice([0.0, 0.0, 0.8])),
        "int_lo": -6,
        "int_hi": 7,
        "objective": objective,
        "num_class": int(rng.integers(3, 8)) if objective == "multiclass" else None,
        "quant_mode": quant_mode,
        "quant_bins": int(rng.choice([0, 0, 16, 256])) if quant_mode != "none" else 0,
        "num_leaves": int(rng.choice([7, 31, 63, 255, 1023])),
        "max_depth": int(rng.choice([-1, 4, 6, 10, 12])),
        "min_data_in_leaf": int(rng.choice([1, 5, 20, 100])),
        "feature_fraction": float(rng.choice([1.0, 1.0, 0.2, 0.6])),
        "bagging_fraction": float(rng.choice([1.0, 1.0, 0.7])),
        "max_bin": int(rng.choice([5, 15, 63, 255])),
        "cuda_plan": ",".join(plan),
        "rounds": int(rng.choice([10, 25, 25, 150])),
        "data_seed": int(rng.integers(0, 2**31)),
    }
    if spec["bagging_fraction"] < 1.0:
        spec["bagging_freq"] = 1
    return spec


_WORKER_SOURCE = r"""
import hashlib, json, sys
import numpy as np
import falcata as lgb

spec = json.loads(sys.argv[1])
device = sys.argv[2]

rng = np.random.default_rng(spec["data_seed"])
n, m = spec["n"], spec["m"]
if spec["dtype"] == "float64":
    X = rng.standard_normal((n, m))
else:
    X = rng.integers(spec["int_lo"], spec["int_hi"], size=(n, m)).astype(spec["dtype"])
Xf = X.astype(np.float64)
if spec["sparsity"] > 0:
    mask = rng.random((n, m)) < spec["sparsity"]
    Xf[mask] = 0.0
    X = Xf if spec["dtype"] == "float64" else Xf.astype(spec["dtype"])
w = rng.standard_normal(m)
score = Xf @ w
if spec["objective"] == "regression":
    y = score + 0.3 * rng.standard_normal(n)
elif spec["objective"] == "binary":
    y = (score > np.median(score)).astype(np.float64)
else:
    y = (Xf @ rng.standard_normal((m, spec["num_class"]))).argmax(axis=1).astype(np.float64)
if spec["dtype"] == "float64" and spec["nan_frac"] > 0:
    X = X.copy(); X[rng.random((n, m)) < spec["nan_frac"]] = np.nan

p = {"objective": spec["objective"], "num_leaves": spec["num_leaves"],
     "max_depth": spec["max_depth"], "min_data_in_leaf": spec["min_data_in_leaf"],
     "feature_fraction": spec["feature_fraction"], "bagging_fraction": spec["bagging_fraction"],
     "max_bin": spec["max_bin"], "learning_rate": spec.get("learning_rate", 0.1),
     "quant_mode": spec["quant_mode"], "quant_bins": spec["quant_bins"],
     "cuda_plan": spec["cuda_plan"],
     "device_type": device, "seed": 42, "verbose": -1, "metric": "None", "num_threads": 8}
# fixedpoint is CUDA-only; the CPU reference run maps it to stochastic quant
# (same quant family, well-defined on CPU) -- the parity check is tolerant.
if device == "cpu" and p["quant_mode"] == "fixedpoint":
    p["quant_mode"] = "stochastic"
    p["quant_bins"] = 0
if spec.get("bagging_freq"):
    p["bagging_freq"] = spec["bagging_freq"]
if spec["objective"] == "multiclass":
    p["num_class"] = spec["num_class"]

k = int(n * 0.8)
ds = lgb.Dataset(X[:k], label=y[:k], params=p)
bst = lgb.train(p, ds, num_boost_round=spec["rounds"])
model_str = bst.model_to_string()
expected = spec["rounds"] * (spec["num_class"] or 1)
assert bst.num_trees() == expected, f"tree count {bst.num_trees()} != {expected}"
pred = bst.predict(X[k:])
assert np.all(np.isfinite(pred)), "non-finite predictions"
assert float(np.std(pred)) > 0.0, "constant predictions (garbage model)"
pred2 = lgb.Booster(model_str=model_str).predict(X[k:])
assert np.array_equal(pred, pred2), "reloaded model predicts differently"
yt = y[k:]
if spec["objective"] == "regression":
    metric = float(np.sqrt(np.mean((pred - yt) ** 2))); higher = False
elif spec["objective"] == "binary":
    order = np.argsort(pred); ranks = np.empty(len(pred)); ranks[order] = np.arange(len(pred))
    pos = yt == 1
    metric = float((ranks[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / max(1, pos.sum() * (~pos).sum()))
    higher = True
else:
    eps = 1e-12
    metric = float(-np.log(np.clip(pred[np.arange(len(yt)), yt.astype(int)], eps, 1)).mean()); higher = False
tree_str = model_str.split("\nparameters:")[0]
print("OUT " + json.dumps({"md5": hashlib.md5(tree_str.encode()).hexdigest(),
                           "metric": metric, "higher_better": higher}))
"""


def run(spec, device, timeout=300):
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_SOURCE, json.dumps(spec), device],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[-800:] or f"exit {proc.returncode}"}
    for line in proc.stdout.splitlines():
        if line.startswith("OUT "):
            return json.loads(line[4:])
    return {"error": f"no output: {proc.stdout[-300:]}"}


def check_spec(spec):
    """Returns a list of failure strings for this spec."""
    fails = []
    a = run(spec, "cuda")
    if "error" in a:
        return [f"cuda run failed: {a['error']}"]
    if spec["quant_mode"] != "none":
        b = run(spec, "cuda")
        if "error" in b:
            fails.append(f"cuda rerun failed: {b['error']}")
        elif a["md5"] != b["md5"]:
            fails.append(f"quant nondeterminism: {a['md5']} != {b['md5']}")
    c = run(spec, "cpu")
    if "error" in c:
        fails.append(f"cpu run failed: {c['error']}")
    else:
        ref, cur = c["metric"], a["metric"]
        tol = abs(ref) * CPU_METRIC_TOLERANCE + 1e-9
        worse = cur < ref - tol if a["higher_better"] else cur > ref + tol
        if worse:
            fails.append(f"cuda metric {cur:.6f} much worse than cpu {ref:.6f}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--spec", type=str, default=None, help="run one JSON spec (repro mode)"
    )
    args = ap.parse_args()

    import numpy as np

    if args.spec:
        fails = check_spec(json.loads(args.spec))
        for f in fails:
            print(f"FAIL {f}")
        return 1 if fails else 0

    seed = (
        args.seed
        if args.seed is not None
        else int(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d"))
    )
    rng = np.random.default_rng(seed)
    print(f"fuzz: seed={seed} budget={args.minutes}min")

    corpus = json.loads(CORPUS_FILE.read_text()) if CORPUS_FILE.exists() else []
    failures = []
    deadline = time.monotonic() + args.minutes * 60
    tried = 0

    for i, spec in enumerate(corpus):
        fails = check_spec(spec)
        tried += 1
        for f in fails:
            failures.append((f"corpus[{i}]", f, spec))

    while time.monotonic() < deadline:
        spec = sample_spec(rng)
        fails = check_spec(spec)
        tried += 1
        for f in fails:
            failures.append((f"seed{seed}#{tried}", f, spec))

    print(f"fuzz: {tried} specs tried, {len(failures)} failure(s)")
    for name, f, spec in failures:
        print(
            f"\nFAIL [{name}] {f}\nrepro: python tests/gates/fuzz.py --spec '{json.dumps(spec)}'"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
