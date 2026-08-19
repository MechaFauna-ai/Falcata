"""Nightly config x shape fuzzer.

Samples random (dataset shape, dtype, NaN fraction, sparsity) x (tree shape,
quant mode, cuda_plan flips, objective) combinations -- exactly the "some
combination of certain configs and certain dataset dimensions" class of bug --
and checks each one for:

  1. validity: training completes, model round-trips, predictions finite,
     tree count matches the emission contract (via the same worker asserts
     as the lattice): every requested iteration, unless boosting saturated
     and stopped early -- which truncates WHOLE iterations only;
  2. CUDA determinism (quant modes): the identical spec trained twice must be
     bit-identical;
  3. CPU quality reference (tolerant): the same data and model shape trained
     full-precision on device_type=cpu must reach a held-out metric within
     tolerance of the CUDA model (trees are not required to be identical; a
     large metric gap means one side built garbage).

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
import os
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
    # ~25% of specs get categorical features (the hybrid+cat paths were fenced
    # until 2026-08-11 and therefore under-fuzzed; the null-bitset restore bug
    # shipped precisely because nothing here generated cat+bagging)
    num_cat = int(rng.choice([0, 0, 0, 2, 4]))
    spec = {
        "n": n,
        "m": m,
        "num_cat_features": num_cat,
        # cardinalities <= 250 keep quant modes eligible (256-bin CUDA guard);
        # int8 storage caps codes at 127 -- larger cards would wrap NEGATIVE,
        # and negative categorical codes mean "missing", not a category
        "cat_cards": [int(c) for c in rng.choice([12, 31, 64, 120, 250], size=num_cat)] if num_cat else [],
        "max_cat_threshold": int(rng.choice([8, 32, 32, 64])),
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

    # CPU-reference cost guard: the parity arm runs on 8 CPU threads under a
    # bounded timeout, and an unlucky joint draw (40k rows x 800 cols x 150
    # rounds x 7 classes) is hours of CPU work, not signal -- it was the
    # standing cause of nightly timeouts. Downscale rounds, then rows, until
    # the cost estimate is bounded; every dimension still gets fuzzed, just
    # not all maxed in the same spec.
    def est(s):
        return s["n"] * s["m"] * s["rounds"] * (s["num_class"] or 1)

    while est(spec) > 6e8 and spec["rounds"] > 10:
        spec["rounds"] = max(10, spec["rounds"] // 2)
    while est(spec) > 6e8 and spec["n"] > 2000:
        spec["n"] = max(2000, spec["n"] // 2)
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
num_cat = spec.get("num_cat_features", 0)
if num_cat:
    # codes must FIT the storage dtype (int8 tops out at 127; wrapping would
    # turn high codes into negative = missing)
    if spec["dtype"] == "int8":
        spec["cat_cards"] = [min(c, 120) for c in spec["cat_cards"]]
    # first num_cat columns become skewed categorical codes
    X = X.astype(np.float64) if spec["dtype"] != "float64" else X
    for j, card in enumerate(spec["cat_cards"]):
        pj = 1.0 / np.arange(1, card + 1) ** 1.2
        X[:, j] = rng.choice(card, size=n, p=pj / pj.sum())
    if spec["dtype"] != "float64":
        X = X.astype(spec["dtype"])
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
     "device_type": device, "seed": 42, "verbose": -1, "metric": "None", "num_threads": 8,
     "max_cat_threshold": spec.get("max_cat_threshold", 32)}
cat_features = list(range(spec.get("num_cat_features", 0)))
# Quantized training is CUDA-only in Falcata. The CPU arm is a full-precision
# quality reference for both quantized modes, not an implementation-parity arm:
# quantized CPU histograms cannot recover exact row counts from their hessians.
if device == "cpu" and p["quant_mode"] != "none":
    p["quant_mode"] = "none"
    p["quant_bins"] = 0
if device == "cpu":
    # Pin the reference basin: OpenMP's unspecified reduction combine order
    # jitters the gradient/hessian sums at the last ulp per run, which flips
    # plateau-tie split picks on degenerate data and lands DIFFERENT models
    # (and metrics) across identical invocations. deterministic serializes
    # exactly those reductions; verified bit-stable.
    p["deterministic"] = True
if spec.get("bagging_freq"):
    p["bagging_freq"] = spec["bagging_freq"]
if spec["objective"] == "multiclass":
    p["num_class"] = spec["num_class"]

k = int(n * 0.8)
ds = lgb.Dataset(X[:k], label=y[:k], params=p,
                 categorical_feature=cat_features if cat_features else "auto")
bst = lgb.train(p, ds, num_boost_round=spec["rounds"])
model_str = bst.model_to_string()
# Emission contract. Boosting stops early when no candidate anywhere clears
# the split requirements (saturated data: few features, few bins) and
# truncates the model by WHOLE iterations; every suppressed round would have
# emitted a zero-output constant tree, so the stopped model predicts exactly
# what a full run would. A short but whole-iteration count is therefore
# legitimate on any device; a partial iteration, zero trees, or more trees
# than requested is a real emission defect.
per_iter = spec["num_class"] or 1
expected = spec["rounds"] * per_iter
got = bst.num_trees()
assert got <= expected, f"tree count {got} > {expected}"
assert got > 0 and got % per_iter == 0, (
    f"tree count {got} is not a whole number of iterations (expected {expected})")
stopped_early = got < expected
pred = bst.predict(X[k:])
assert np.all(np.isfinite(pred)), "non-finite predictions"
if float(np.std(pred)) <= 0.0:
    # A model that saturated at iteration 1 (every split rejected) holds only
    # zero-output constant trees; a FULL-length model predicting one value is
    # a different, hard failure -- the harness classifies only the former as
    # the known cpu-quant defect.
    if stopped_early:
        raise AssertionError("constant predictions after early stop (all-zero model)")
    raise AssertionError("constant predictions (garbage model)")
assert float(np.std(pred)) > 0.0
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
                           "metric": metric, "higher_better": higher,
                           "trees": got, "stopped_early": stopped_early}))
"""


def run(spec, device, timeout=300):
    # Engine predictor only: FIL predicts in single precision, which collapses
    # legitimate near-zero models (whole-iteration early stops) to constant
    # and fails the std(pred) > 0 assert on a healthy model.
    env = dict(os.environ, FALCATA_FIL="0")
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_SOURCE, json.dumps(spec), device],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[-800:] or f"exit {proc.returncode}"}
    for line in proc.stdout.splitlines():
        if line.startswith("OUT "):
            return json.loads(line[4:])
    return {"error": f"no output: {proc.stdout[-300:]}"}


def is_lowbin_quant_bias(spec):
    """Aggressively quantized modes at very low bin budgets with single-row
    leaves: the cpu cross-check compares against UNQUANTIZED cpu (quantized
    training is CUDA-only), so in this corner the 10% tolerance measures
    quantization noise, not implementation parity. Fixedpoint with error
    feedback is symmetric seed variance (roughly -30%..+15% across seeds);
    stochastic at its 4-bin default trades accuracy for speed by design and
    single knife-edge specs cross the tolerance in either direction. All
    conditions are required; everything outside them still fails hard, the
    35% blow-out bound below still catches genuinely broken kernels, and the
    md5 determinism check above still applies here.
    """
    lowbin = (spec["quant_mode"] == "fixedpoint" and spec.get("quant_bins", 0) <= 16) or (
        spec["quant_mode"] == "stochastic" and spec.get("quant_bins", 0) <= 8
    )
    return lowbin and spec.get("min_data_in_leaf", 20) <= 2


def check_spec(spec):
    """Returns (failures, known_issues) for this spec."""
    fails = []
    known = []
    a = run(spec, "cuda")
    if "error" in a:
        return [f"cuda run failed: {a['error']}"], known
    if a.get("stopped_early"):
        known.append(
            f"saturation stop: cuda emitted {a['trees'] // (spec['num_class'] or 1)}/{spec['rounds']} iterations"
        )
    # Determinism rerun. Quantized training is bit-deterministic everywhere
    # (integer atomics); non-quantized training is bit-deterministic on every
    # shape the deterministic float constructs cover -- which is everything
    # EXCEPT the graph-captured level loop (still atomic by design). A
    # non-quant spec is md5-checked when its shape cannot take the graph loop:
    # the plan disables it (or the whole hybrid flow), or the spec is not
    # depth-limited (the hybrid prefix requires 2^max_depth <= num_leaves + 1).
    md5_checked = spec["quant_mode"] != "none"
    if not md5_checked:
        plan = spec.get("cuda_plan", "")
        depth_limited = spec.get("max_depth", -1) > 0 and (2 ** spec["max_depth"] <= spec["num_leaves"] + 1)
        batched_possible = depth_limited and "hybrid:off" not in plan
        graph_possible = batched_possible and "graph_loop:off" not in plan
        # the BATCHED flow keeps the atomic kernel for compact views
        # (feature sampling); per-leaf covers them
        compact_hole = batched_possible and spec.get("feature_fraction", 1.0) < 1.0
        md5_checked = not graph_possible and not compact_hole
    if md5_checked:
        b = run(spec, "cuda")
        if "error" in b:
            fails.append(f"cuda rerun failed: {b['error']}")
        elif a["md5"] != b["md5"]:
            fails.append(f"cuda nondeterminism: {a['md5']} != {b['md5']}")
    # the CPU arm is legitimately ~10-50x slower than CUDA on big specs; give
    # it headroom beyond the cost guard rather than fail on machine noise
    c = run(spec, "cpu", timeout=900)
    if "error" in c:
        fails.append(f"cpu run failed: {c['error']}")
    else:
        if c.get("stopped_early"):
            known.append(
                f"saturation stop: cpu emitted {c['trees'] // (spec['num_class'] or 1)}/{spec['rounds']} iterations"
            )
        ref, cur = c["metric"], a["metric"]
        tol = abs(ref) * CPU_METRIC_TOLERANCE + 1e-9
        worse = cur < ref - tol if a["higher_better"] else cur > ref + tol
        if worse and spec.get("bagging_fraction", 1.0) < 1.0:
            # the two arms draw DIFFERENT bags by design (device Philox vs
            # host RNG), so a single-run gap can be sampler variance rather
            # than a defect: re-draw twice and fail only if cuda loses every
            # round (a systematic bug does; bag luck does not)
            confirm = 0
            for retry_seed in (spec["data_seed"] ^ 0x5EED, spec["data_seed"] ^ 0xBA66):
                s2 = dict(spec, data_seed=retry_seed)
                a2, c2 = run(s2, "cuda"), run(s2, "cpu", timeout=900)
                if "error" in a2 or "error" in c2:
                    confirm += 1  # a retry that cannot run does not exonerate
                    continue
                r2, u2 = c2["metric"], a2["metric"]
                t2 = abs(r2) * CPU_METRIC_TOLERANCE + 1e-9
                confirm += (u2 < r2 - t2) if a2["higher_better"] else (u2 > r2 + t2)
            if confirm == 0:
                known.append(
                    f"bagged single-run variance: cuda {cur:.6f} vs cpu {ref:.6f}, 2/2 reseeded runs within tolerance"
                )
                worse = False
        if worse and is_lowbin_quant_bias(spec):
            # still bounded: the post-error-feedback variance envelope is
            # +-30%; a corrupted kernel blows past it (historically 2-10x)
            blown = cur < ref - abs(ref) * 0.35 if a["higher_better"] else cur > ref + abs(ref) * 0.35
            if blown:
                fails.append(
                    f"cuda metric {cur:.6f} vs cpu {ref:.6f}: beyond the "
                    "low-bin quant variance envelope -- treat as real"
                )
            else:
                known.append(f"low-bin quant variance ({spec['quant_mode']}): cuda {cur:.6f} vs cpu-noquant {ref:.6f}")
        elif worse:
            fails.append(f"cuda metric {cur:.6f} much worse than cpu {ref:.6f}")
    return fails, known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--spec", type=str, default=None, help="run one JSON spec (repro mode)")
    args = ap.parse_args()

    import numpy as np

    if args.spec:
        fails, known = check_spec(json.loads(args.spec))
        for k in known:
            print(f"KNOWN {k}")
        for f in fails:
            print(f"FAIL {f}")
        return 1 if fails else 0

    seed = args.seed if args.seed is not None else int(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)
    print(f"fuzz: seed={seed} budget={args.minutes}min")

    corpus = json.loads(CORPUS_FILE.read_text()) if CORPUS_FILE.exists() else []
    failures = []
    num_known = 0
    deadline = time.monotonic() + args.minutes * 60
    tried = 0

    for i, spec in enumerate(corpus):
        fails, known = check_spec(spec)
        tried += 1
        num_known += len(known)
        for f in fails:
            failures.append((f"corpus[{i}]", f, spec))

    while time.monotonic() < deadline:
        spec = sample_spec(rng)
        fails, known = check_spec(spec)
        tried += 1
        num_known += len(known)
        for f in fails:
            failures.append((f"seed{seed}#{tried}", f, spec))

    print(f"fuzz: {tried} specs tried, {len(failures)} failure(s), {num_known} known variance note(s)")
    for name, f, spec in failures:
        print(f"\nFAIL [{name}] {f}\nrepro: python tests/gates/fuzz.py --spec '{json.dumps(spec)}'")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
