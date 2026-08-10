#!/usr/bin/env python3
"""Validation-scoring gate: the metric the training loop reports must equal the
metric recomputed from predict().

Nothing else in the suite exercised validation sets on CUDA. That blind spot hid
two bugs at once:

  - an ODR violation (a Config member declared inside `#ifndef __NVCC__`) gave
    .cu and .cpp objects different layouts, so the metric read its label pointer
    from the wrong offset and died with an illegal memory access;
  - the CUDA tree-traversal kernel compared a feature-local bin against a
    raw-column max_bin, so every NaN row was routed by value instead of down the
    split's default direction -- silently wrong validation metrics, and silently
    worse models whenever bagging made that kernel score out-of-bag rows.

Both are invisible to a lattice cell (no valid set, no bagging) and to a plain
train/predict test (predict() traverses on the host). Reporting one number while
predict() computes another is exactly the discrepancy that catches them.

Exit 0 if every case agrees, 1 otherwise.
"""

import sys

import numpy as np

import falcata as flc

# The reported metric is a device-side reduction and the reference is numpy, so
# the last couple of ULPs may differ; anything real shows up orders of magnitude
# above this.
TOL = 1e-6


def make(rng, n, m, nan_frac):
    x = rng.normal(size=(n, m))
    if nan_frac:
        x[rng.random((n, m)) < nan_frac] = np.nan
    return x


def case(name, nan_frac, extra, weighted=False, rounds=30):
    rng = np.random.default_rng(17)
    x = make(rng, 8000, 6, nan_frac)
    y = np.nan_to_num(x[:, 0]) + 0.5 * np.nan_to_num(x[:, 1]) + rng.normal(size=8000) * 0.2
    xv = make(rng, 2000, 6, nan_frac)
    yv = np.nan_to_num(xv[:, 0]) + 0.5 * np.nan_to_num(xv[:, 1]) + rng.normal(size=2000) * 0.2
    wt = rng.random(8000) + 0.5 if weighted else None
    wv = rng.random(2000) + 0.5 if weighted else None

    params = {
        "objective": "regression",
        "device_type": "cuda",
        "metric": "l2",
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "verbose": -1,
        "seed": 1,
        **extra,
    }
    train = flc.Dataset(x, label=y, weight=wt)
    valid = flc.Dataset(xv, label=yv, weight=wv, reference=train)
    history = {}
    booster = flc.train(
        params,
        train,
        num_boost_round=rounds,
        valid_sets=[valid],
        callbacks=[flc.record_evaluation(history)],
    )
    reported = history["valid_0"]["l2"][-1]
    pred = booster.predict(xv)
    recomputed = float(np.average((pred - yv) ** 2, weights=wv) if weighted else np.mean((pred - yv) ** 2))
    ok = abs(reported - recomputed) <= TOL * max(1.0, abs(recomputed))
    print(
        f"{'ok  ' if ok else 'FAIL'} {name:<28} reported={reported:.8f} "
        f"recomputed={recomputed:.8f} diff={abs(reported - recomputed):.2e}"
    )
    return ok


def main():
    cases = [
        ("clean", 0.0, {}, False),
        ("40% NaN", 0.4, {}, False),
        ("40% NaN + weights", 0.4, {}, True),
        ("bagging", 0.0, {"bagging_fraction": 0.7, "bagging_freq": 1}, False),
        ("bagging + 40% NaN", 0.4, {"bagging_fraction": 0.7, "bagging_freq": 1}, False),
        ("bagging 0.3 + 40% NaN", 0.4, {"bagging_fraction": 0.3, "bagging_freq": 1}, False),
        ("feature_fraction + NaN", 0.4, {"feature_fraction": 0.5}, False),
        ("quantized + NaN", 0.4, {"quant_mode": "stochastic"}, False),
    ]
    results = [case(*c) for c in cases]
    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} validation-scoring cases agree")
    if failed:
        print("VALID METRIC GATE FAIL")
        return 1
    print("VALID METRIC GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
