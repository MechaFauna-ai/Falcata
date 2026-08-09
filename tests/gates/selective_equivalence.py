"""Selective-growth equivalence gate.

Selective growth explores past the leaf budget and collapses the splits a
fresh greedy selection no longer picks. The property this rests on: a
displaced split can NEVER re-enter the selection, so eagerly killing its
subtree can never cost the tree a split it would otherwise have taken. The
worry it answers (raised 2026-08-02): once a subtree is collapsed the tree is
back under budget, so could the collapsed PARENT have been the best next
split? No -- everything that displaced it stays in the candidate pool forever,
and new candidates only ever insert ahead of it in the pick order.

Rather than trust the argument, this gate checks it: on adversarial gain
distributions (heavy-tailed feature strengths plus interaction terms, so deep
candidates really do outrank their ancestors and displacement happens on most
seeds), selective growth must produce EXACTLY the tree classic
one-split-at-a-time growth produces -- identical structure and bit-identical
predictions.

Runs on the quantized path: integer histograms make CUDA training
bit-deterministic, so "identical" means identical. (Under quant_mode=none the
fp32 histogram atomics make both paths nondeterministic run-to-run at the
1e-8 level, which can flip an exact-gain tie -- real, accepted, and NOT what
this gate is about.)

Usage:
  python tests/gates/selective_equivalence.py [--seeds 16] [--rounds 20]
"""

import argparse
import os
import subprocess
import sys
import textwrap

import numpy as np

# the +-kZeroThreshold pair are the zero-bin sandwich boundaries; with an empty
# zero bin the two sides are an exact-gain tie that the batched and per-pair
# finders may break differently. Same data partition, same predictions.
ZERO_THRESHOLD = 1.0000000180025095e-35

STRUCTURAL_KEYS = ("split_feature=", "threshold=", "decision_type=", "left_child=", "right_child=", "num_leaves=")


def tree_structure(model_str):
    """Structural projection of a model: splits only, no leaf values."""
    keep = []
    for line in model_str.splitlines():
        if not line.startswith(STRUCTURAL_KEYS):
            continue
        if line.startswith("threshold="):
            vals = ["ZERO" if abs(float(v)) == ZERO_THRESHOLD else v for v in line.split("=", 1)[1].split()]
            line = "threshold=" + " ".join(vals)
        keep.append(line)
    return "\n".join(keep)


def make_data(rng, n=200_000, m=30):
    """Adversarial gains: a few dominant features, long Pareto tail, and
    interactions so candidates deep in the tree can outrank shallow ones."""
    w = rng.pareto(1.2, m) * rng.choice([-1, 1], m)
    X = rng.standard_normal((n, m)).astype(np.float32)
    y = X @ (w / np.abs(w).sum())
    y += 0.7 * np.tanh(X[:, 0] * X[:, 1]) + 0.5 * (X[:, 2] > 1.1) * X[:, 3]
    y += 0.3 * rng.standard_normal(n)
    return X, y.astype(np.float32)


def params_for(leaves, extra):
    # max_depth=-1 is what engages selective growth: the leaf budget, not the
    # depth cap, is the binding constraint (depth-capped configs never run it)
    p = {
        "objective": "regression",
        "learning_rate": 0.1,
        "num_leaves": leaves,
        "max_depth": -1,
        "max_bin": 255,
        "min_data_in_leaf": 50,
        "device_type": "cuda",
        "seed": 42,
        "verbose": -1,
        "num_threads": 16,
        "quant_mode": "stochastic",
    }
    p.update(extra)
    return p


def assert_selective_engaged():
    """A vacuous gate is worse than no gate: confirm the path actually runs."""
    probe = textwrap.dedent("""
        import numpy as np, falcata as flc
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200000, 30)).astype(np.float32)
        y = (X[:, 0] + 0.7 * np.tanh(X[:, 1] * X[:, 2])).astype(np.float32)
        p = {"objective": "regression", "learning_rate": 0.1, "num_leaves": 127,
             "max_depth": -1, "max_bin": 255, "min_data_in_leaf": 50,
             "device_type": "cuda", "seed": 42, "verbose": -1,
             "num_threads": 16, "quant_mode": "stochastic"}
        flc.train(p, flc.Dataset(X, label=y, params=p), num_boost_round=5)
    """)
    env = dict(os.environ, FALCATA_DEBUG="debug")
    out = subprocess.run([sys.executable, "-c", probe], check=False, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr[-2000:])
        raise SystemExit("selective_equivalence: probe run failed")
    if "[selective]" not in out.stderr:
        raise SystemExit(
            "selective_equivalence: selective growth never engaged -- the gate "
            "would pass vacuously. Did the plan stop selecting it for "
            "unbounded-depth leaf-budget configs?"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=20)
    args = ap.parse_args()

    import falcata as flc

    assert_selective_engaged()

    fails = 0
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        X, y = make_data(rng)
        leaves = int(rng.choice([15, 31, 63, 127]))
        structs, preds = {}, {}
        for mode, extra in (("selective", {}), ("classic", {"cuda_plan": "hybrid:off"})):
            p = params_for(leaves, extra)
            ds = flc.Dataset(X, label=y, params=p)
            bst = flc.train(p, ds, num_boost_round=args.rounds)
            structs[mode] = tree_structure(bst.model_to_string())
            preds[mode] = bst.predict(X[:20_000])
        struct_ok = structs["selective"] == structs["classic"]
        pdiff = float(np.max(np.abs(preds["selective"] - preds["classic"])))
        ok = struct_ok and pdiff == 0.0
        print(
            f"seed={seed} leaves={leaves}: "
            f"struct={'identical' if struct_ok else 'DIVERGED'} "
            f"pred_maxdiff={pdiff:.2e} -> {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            fails += 1
            for i, (a, b) in enumerate(zip(structs["selective"].splitlines(), structs["classic"].splitlines())):
                if a != b:
                    print(f"  first differing structural line {i}:")
                    print(f"    selective: {a[:200]}")
                    print(f"    classic  : {b[:200]}")
                    break

    print("SELECTIVE EQUIVALENCE PASS" if fails == 0 else f"SELECTIVE EQUIVALENCE FAIL ({fails}/{args.seeds} seeds)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
