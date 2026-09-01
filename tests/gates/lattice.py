"""Falcata regression-gate lattice.

A matrix of small, deterministic (config x dataset-shape) training cells that
runs after every commit (~minutes on an idle GPU) and catches the three
regression classes this project keeps hitting:

  1. Invalid trees / silent behavior changes -> md5 fingerprints. Quantized
     CUDA training is bit-deterministic, so every fingerprinted cell's model
     md5 is a stable signature; ANY unintended behavior change flips one.
     Intended changes re-baseline explicitly (--update / --baseline), and the
     md5_lattice.json diff in the commit shows exactly which cells moved.
  2. Plan decisions that stop being behavior-preserving -> equality cells:
     every cuda_plan key is flipped against its base cell and the two models
     must be bit-identical (pairwise coverage instead of 2^N).
  3. Quality / perf regressions -> metric floors on cells (metric recorded in
     the fingerprint file; checked with tolerance) and per-cell wall times
     (consumed by perf_gate.py against a machine-local rolling baseline).

Every cell additionally asserts validity: the model round-trips through
model_to_string -> Booster(model_str) with identical predictions, predictions
are finite, and the tree count matches num_boost_round.

Usage:
  python tests/gates/lattice.py --check              # CI: diff vs fingerprints
  python tests/gates/lattice.py --baseline           # (re)write all fingerprints
  python tests/gates/lattice.py --update ID [ID..]   # re-baseline specific cells
  python tests/gates/lattice.py --list               # print the cell matrix
Cells run sequentially in chunked worker subprocesses (crash isolation: a
crashing cell fails alone and the runner respawns for the rest).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GATES_DIR = Path(__file__).resolve().parent
FINGERPRINT_FILE = GATES_DIR / "md5_lattice.json"
RESULTS_FILE = Path(os.environ.get("FALCATA_GATES_RESULTS", GATES_DIR / "lattice_results.json"))
WORKER_CHUNK = 30
METRIC_TOLERANCE = 0.02  # relative; only applied to non-fingerprinted (nondeterministic) cells

# --------------------------------------------------------------------------- #
# Cell matrix. Each cell:
#   id       unique name (stable across time -- it keys the fingerprint file)
#   profile  dataset builder in _worker_source below
#   params   overrides merged onto the profile's base params
#   rounds   num_boost_round
#   equal_to base-cell id whose md5 this cell must reproduce (plan-flip cells);
#            such cells are NOT in the fingerprint file
#   fingerprint  False for nondeterministic (non-quant) cells: validity+metric only
#   perf     True -> wall time tracked by perf_gate.py
# --------------------------------------------------------------------------- #


def build_cells():
    cells = []

    def cell(
        cid,
        profile,
        params=None,
        rounds=25,
        equal_to=None,
        fingerprint=True,
        perf=False,
    ):
        cells.append(
            {
                "id": cid,
                "profile": profile,
                "params": params or {},
                "rounds": rounds,
                "equal_to": equal_to,
                "fingerprint": fingerprint and equal_to is None,
                "perf": perf,
            }
        )

    # --- base fingerprints: every profile x stochastic quant, shallow ------- #
    for profile in [
        "dense",
        "fewbin",
        "sampled",
        "graph",
        "int8wide",
        "efb",
        "imbalanced",
        "missing",
        "missing-mfb0",
    ]:
        cell(f"{profile}/quant", profile, perf=profile in ("dense", "int8wide"))

    # deep + classic twins for the missing-bin-0 shape: the synthesised bin 0
    # feeds the level-batched and the classic split search through the same
    # kernel, and only the deep cell exercises enough leaves to catch a
    # threshold that is right at the root and wrong further down.
    cell("missing-mfb0/quant-deep", "missing-mfb0", {"num_leaves": 255, "max_depth": 12, "min_data_in_leaf": 5})
    cell("missing-mfb0/quant-classic", "missing-mfb0", {"cuda_plan": "auto,hybrid:off"})
    cell("missing-mfb0/fixedpoint", "missing-mfb0", {"quant_mode": "fixedpoint"})

    # --- deep trees: where leaf-wise/level-batched divergence bugs live ----- #
    for profile in ["dense", "sampled", "int8wide"]:
        cell(
            f"{profile}/quant-deep",
            profile,
            {"num_leaves": 255, "max_depth": 12, "min_data_in_leaf": 5},
            perf=profile == "dense",
        )

    # --- classic (non-hybrid) growth path: separate fingerprint ------------- #
    # hybrid growth is NOT bit-identical to the classic loop (different split
    # ordering on gain plateaus), so this is its own fingerprint, guarding the
    # fallback path that cuda_plan=auto,hybrid:off users get.
    cell("dense/quant-classic", "dense", {"cuda_plan": "auto,hybrid:off"})
    cell(
        "dense/quant-deep-classic",
        "dense",
        {
            "cuda_plan": "auto,hybrid:off",
            "num_leaves": 255,
            "max_depth": 12,
            "min_data_in_leaf": 5,
        },
    )

    # --- fixedpoint quant mode ---------------------------------------------- #
    # imbalanced exercises the gap-gated robust scale; dense/fewbin pin the
    # balanced (gap-free, exact global-max) behavior.
    for profile in ["dense", "fewbin", "imbalanced"]:
        cell(f"{profile}/fixedpoint", profile, {"quant_mode": "fixedpoint"})
    # weighted CUDA binary gradients: is_unbalance indexes the 2-entry label
    # weight table -- the 2026-08-04 fix (negative labels read one double
    # BEFORE the allocation) lives exactly here
    cell("imbalanced/unbalance", "imbalanced", {"is_unbalance": True})
    cell("imbalanced/posweight", "imbalanced", {"scale_pos_weight": 10.0})
    cell(
        "dense/fixedpoint-256bins",
        "dense",
        {"quant_mode": "fixedpoint", "quant_bins": 256},
    )

    # --- plan-flip equality cells: each decision must be behavior-preserving - #
    flips = [
        (
            "graph",
            "graph_loop",
            {"cuda_plan": "auto,graph_quant:on"},
            {"cuda_plan": "auto,graph_quant:on,graph_loop:off"},
        ),
        ("fewbin", "rowdata_4bit", {}, {"cuda_plan": "auto,rowdata_4bit:off"}),
        ("dense", "fast_rowdata", {}, {"cuda_plan": "auto,fast_rowdata:off"}),
        ("dense", "gpu_construct", {}, {"cuda_plan": "auto,gpu_construct:off"}),
        ("efb", "efb_precheck", {}, {"cuda_plan": "auto,efb_precheck:off"}),
        ("sampled", "compact_quant", {}, {"cuda_plan": "auto,compact_quant:off"}),
        ("sampled", "construct_jit", {}, {"cuda_plan": "auto,construct_jit:on"}),
        ("sampled", "compact_prefill", {}, {"cuda_plan": "auto,compact_prefill:on"}),
        ("int8wide", "rowdata_4bit", {}, {"cuda_plan": "auto,rowdata_4bit:off"}),
        ("int8wide", "fast_rowdata", {}, {"cuda_plan": "auto,fast_rowdata:off"}),
    ]
    for profile, key, base_params, flip_params in flips:
        base_id = f"{profile}/quant" if not base_params else f"{profile}/flipbase-{key}"
        if base_params:
            cell(base_id, profile, base_params)
        cell(f"{profile}/flip-{key}", profile, flip_params, equal_to=base_id)
    # 8-bit compact JIT kernel (4-bit disabled): its own base + flip pair
    cell("sampled/flipbase-jit8", "sampled", {"cuda_plan": "auto,rowdata_4bit:off"})
    cell(
        "sampled/flip-construct_jit8",
        "sampled",
        {"cuda_plan": "auto,construct_jit:on,rowdata_4bit:off"},
        equal_to="sampled/flipbase-jit8",
    )

    # --- tiny-gradient regime: leaf-collapse / quant-underflow class -------- #
    # 150 rounds gives gradient decay room; tree-count + nonconstant-pred
    # asserts are the detectors. NOTE: the 2026-07 fixedpoint 44-trees bug does
    # NOT reproduce at this scale (verified empirically, probes up to 550k
    # rows) -- the full-scale nightly twin in canonical.py is the gate that
    # catches it; these cells guard the class at per-commit cost.
    cell("tinygrad/quant", "tinygrad", rounds=150)

    # --- large-row overflow class: 2M rows x 64 bins, compact view ----------- #
    # fingerprint + compact-flip equality at a scale where the packed-int16
    # partials actually overflow if unguarded (~15s/cell; the priciest cells
    # in the lattice, and worth every second)
    cell("bigrow/quant64", "bigrow", rounds=40, perf=True)
    cell(
        "bigrow/flip-compact_quant",
        "bigrow",
        {"cuda_plan": "auto,compact_quant:off"},
        rounds=40,
        equal_to="bigrow/quant64",
    )
    cell("tinygrad/fixedpoint", "tinygrad", {"quant_mode": "fixedpoint"}, rounds=150)

    # --- nondeterministic tiers: validity + metric floor only --------------- #
    # non-quant float-atomic path and the fp32 opt-in (cuda_precision) --
    # exactly the paths md5 cannot cover.
    cell(
        "dense/nonquant",
        "dense",
        {"quant_mode": "none"},
        rounds=50,
        fingerprint=False,
        perf=True,
    )
    cell(
        "dense/fp32",
        "dense",
        {"quant_mode": "none", "cuda_precision": "fp32"},
        rounds=50,
        fingerprint=False,
        perf=True,
    )
    # Non-quant DETERMINISTIC cells: on these shapes every construct runs the
    # deterministic kernels (per-leaf, batched-level, compact view, sparse
    # fallback), so their models fingerprint like quant cells do. The graph
    # loop now captures the det construct+merge node pair too (per-level
    # extents come from the controller and the kernels' device layout
    # replica), so DEFAULT-plan cells on graph-eligible shapes fingerprint as
    # well -- graph/nonquant-det and imbalanced/nonquant-det below pin
    # exactly that path; the graph_loop:off det cells pin the host loop.
    cell("dense/nonquant-det", "dense", {"quant_mode": "none", "cuda_plan": "auto,graph_loop:off"}, rounds=50)
    # hybrid:off -- the BATCHED level flow keeps the atomic kernel for compact
    # views (per-leaf det covers them; batched does not yet), so the compact
    # det lock pins the per-leaf flow.
    cell("sampled/nonquant-det", "sampled", {"quant_mode": "none", "cuda_plan": "auto,hybrid:off"}, rounds=50)
    cell("missing/nonquant-det", "missing", {"quant_mode": "none", "cuda_plan": "auto,graph_loop:off"}, rounds=50)
    cell(
        "categorical/nonquant-det", "categorical", {"quant_mode": "none", "cuda_plan": "auto,graph_loop:off"}, rounds=50
    )
    # graph_det:on -- the captured det construct+merge nodes inside the graph
    # loop (opt-in; the DEFAULT graph plan keeps the atomic construct for
    # speed). graph/nonquant-det replaces the old unfingerprinted
    # graph/nonquant cell; imbalanced is the shape whose atomic/det mix raced
    # in the 2026-08-19 lattice, so its graph-det model is pinned too.
    cell("graph/nonquant-det", "graph", {"quant_mode": "none", "cuda_plan": "auto,graph_det:on"}, rounds=50)
    cell("imbalanced/nonquant-det", "imbalanced", {"quant_mode": "none", "cuda_plan": "auto,graph_det:on"}, rounds=50)
    # the DEFAULT plan stays atomic in the graph: metric-only coverage
    cell("graph/nonquant", "graph", {"quant_mode": "none"}, rounds=50, fingerprint=False)
    # ... and the graph det loop must be BIT-IDENTICAL to the host det loop:
    # the controller replays the host loop's decisions and the det
    # construct+merge is row-grouping invariant, so swapping graph_det:on for
    # graph_loop:off may not change a single tree byte.
    cell(
        "graph/flip-graph_loop-det",
        "graph",
        {"quant_mode": "none", "cuda_plan": "auto,graph_loop:off"},
        rounds=50,
        equal_to="graph/nonquant-det",
    )
    # categorical, non-quant: hybrid-vs-classic equivalence guard
    cell(
        "categorical/nonquant",
        "categorical",
        {"quant_mode": "none"},
        rounds=50,
        fingerprint=False,
    )
    # categorical, quantized (supported since b235de83): fingerprint-locks the
    # quant-categorical finder (reader-unpacked integer histograms + packed
    # child totals) and the batched-level categorical kernels
    cell("categorical/quant", "categorical", rounds=50)
    # deep variant: selective (grow-then-prune) flow with categorical replay
    cell(
        "categorical/quant-deep",
        "categorical",
        {"num_leaves": 63, "max_depth": -1},
        rounds=25,
    )
    # quantized categorical x BAGGING x multiclass x tiny leaves: pins the
    # bagged-quant discretized find path, where the hessian-quantum l2 ridge
    # is active (no other fingerprinted cell reaches it). Deep trees + small
    # min_data are load-bearing: they produce the dither-dominated tiny-leaf
    # hessian sums whose 1/H the ridge bounds.
    cell(
        "categorical-mc/quant-bagged",
        "categorical-mc",
        {
            "quant_mode": "fixedpoint",
            "quant_bins": 16,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "num_leaves": 255,
            "max_depth": 10,
            "min_data_in_leaf": 5,
        },
        rounds=50,
    )
    # the same ridge WITHOUT bagging: sparse few-bin columns against a
    # separable multiclass label, grown deep over tiny leaves for long enough
    # that a mis-sized leaf output feeds back into the next round's gradients.
    # Bagging is not what compounds the dither's tiny-hessian outputs -- a
    # geometry that keeps re-splitting the same few regions does it alone --
    # and no bagged cell can show that. rounds is load-bearing: the collapse
    # needs tens of rounds of feedback to leave the quantization envelope.
    cell(
        "sparse-mc/quant-deep",
        "sparse-mc",
        {
            "quant_mode": "fixedpoint",
            "quant_bins": 16,
            "num_leaves": 63,
            "max_depth": 10,
            "min_data_in_leaf": 5,
        },
        rounds=80,
    )

    # --- wide features: the GLOBAL-MEMORY split finder ---------------------- #
    # Once a feature's histogram exceeds one block
    # (NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER == 256 bins) CUDA switches to a
    # separate kernel family that stages histograms in global memory and scans
    # them GRID-STRIDE -- several bin positions per thread instead of one.
    # Every other cell here caps at max_bin=255 and therefore never launches
    # it, which is how two whole defect clusters survived: the numeric scan
    # kept the LAST qualifying bin rather than the best one and broke exact
    # gain ties on threadIdx_x, and the NaN head scored the missing-mass
    # position CPU never considers (plus a reduce shape, an epsilon sign and a
    # missing-count rounding that all disagreed with CPU's forward scan).
    # max_bin=511 is the smallest round budget past the boundary; deep trees
    # are what expose it, because the divergence needs the small leaves where
    # plateau gains and exact ties appear.
    #
    # These cells fingerprint despite being non-quant: quantized gradients are
    # REFUSED on this path (the discretized finder has no global-memory
    # variant), and the tree learner also disables the whole hybrid family and
    # the fp32 histogram layout there, so a wide shape always runs the classic
    # loop over the deterministic per-leaf constructs -- bit-stable, no
    # graph_loop:off needed.
    #
    # Three cells because the wide kernel has three distinct entries and one
    # shape covers only one of them (task counts confirmed against the tasks
    # the finder builds): `dense` is numeric-only (20 wide tasks, no NaN
    # tasks) and pins the running-maximum scan; `missing` adds the reverse
    # na_as_missing scan (20 of its 40 tasks) whose gate had gone dead;
    # `missing-mfb0-wide` is the only one that reaches the FORWARD
    # na_as_missing head (20 forward-head + 20 reverse tasks), where the
    # reduce shape, the epsilon sign and the missing-count rounding had to
    # match CPU bit for bit.
    cell(
        "dense/nonquant-widebin",
        "dense",
        {"quant_mode": "none", "max_bin": 511, "num_leaves": 255, "max_depth": 12, "min_data_in_leaf": 5},
    )
    cell(
        "missing/nonquant-widebin",
        "missing",
        {"quant_mode": "none", "max_bin": 511, "num_leaves": 255, "max_depth": 12, "min_data_in_leaf": 5},
    )
    cell(
        "missing-mfb0-wide/nonquant",
        "missing-mfb0-wide",
        {"quant_mode": "none", "num_leaves": 255, "max_depth": 12, "min_data_in_leaf": 5},
    )

    ids = [c["id"] for c in cells]
    assert len(ids) == len(set(ids)), "duplicate cell ids"
    by_id = {c["id"]: c for c in cells}
    for c in cells:
        if c["equal_to"] is not None:
            assert c["equal_to"] in by_id, f"{c['id']}: unknown equal_to {c['equal_to']}"
    return cells


# --------------------------------------------------------------------------- #
# Worker: runs in a fresh interpreter; builds datasets, trains cells, emits
# one JSON line per cell. Receives the cell list as JSON on argv[1].
# --------------------------------------------------------------------------- #

_WORKER_SOURCE = r"""
import hashlib, json, sys, time
import numpy as np
import falcata as lgb

def _split(X, y, frac=0.2):
    n = len(y); k = int(n * (1 - frac))
    return X[:k], y[:k], X[k:], y[k:]

def build_profile(name):
    # stable seed: builtin hash() is per-process randomized (PYTHONHASHSEED)
    rng = np.random.default_rng(int(hashlib.md5(name.encode()).hexdigest()[:8], 16))
    base = {"objective": "regression", "num_leaves": 63, "max_depth": 6,
            "learning_rate": 0.1, "max_bin": 255, "use_quantized_grad": True,
            "device_type": "cuda", "seed": 42, "verbose": -1, "metric": "None",
            "num_threads": 8}
    n = 8000
    if name == "dense":
        m = 20
        X = rng.standard_normal((n, m))
        y = X @ rng.standard_normal(m) + 0.3 * rng.standard_normal(n)
    elif name == "fewbin":
        m = 12
        X = rng.integers(0, 12, size=(n, m)).astype(np.float64)
        y = X @ rng.standard_normal(m) + rng.standard_normal(n)
        base["max_bin"] = 15
    elif name == "sampled":
        m = 600
        X = rng.integers(0, 6, size=(n, m)).astype(np.float64)
        y = X @ rng.standard_normal(m) + rng.standard_normal(n)
        base.update({"max_bin": 15, "feature_fraction": 0.2})
    elif name == "graph":
        m = 20
        X = rng.standard_normal((n, m))
        y = (X @ rng.standard_normal((m, 5))).argmax(axis=1).astype(np.float64)
        base.update({"objective": "multiclass", "num_class": 5, "metric": "multi_logloss"})
    elif name == "int8wide":
        n, m = 6000, 400
        X = rng.integers(-6, 7, size=(n, m)).astype(np.int8)
        y = X.astype(np.float64) @ rng.standard_normal(m) + rng.standard_normal(n)
        base["max_bin"] = 15
    elif name == "efb":
        m = 60
        X = np.zeros((n, m))
        hot = rng.integers(0, m // 3, size=n)
        X[np.arange(n), hot] = 1.0
        X[:, m // 3:] = rng.standard_normal((n, m - m // 3)) * (rng.random((n, m - m // 3)) < 0.05)
        y = X @ rng.standard_normal(m) + 0.1 * rng.standard_normal(n)
    elif name == "imbalanced":
        n, m = 12000, 20
        X = rng.standard_normal((n, m))
        score = X @ rng.standard_normal(m)
        y = (score > np.quantile(score, 0.99)).astype(np.float64)
        base.update({"objective": "binary", "metric": "auc"})
    elif name == "missing":
        m = 20
        X = rng.standard_normal((n, m))
        y = X @ rng.standard_normal(m) + 0.3 * rng.standard_normal(n)
        X[rng.random((n, m)) < 0.3] = np.nan
        y[np.isnan(y)] = 0.0
    elif name == "missing-mfb0":
        # NaN features whose MOST FREQUENT bin is bin 0 -- the one shape that
        # sets na_as_missing && mfb_offset==1, where the histogram omits bin 0
        # and the forward split scan has to synthesise it from the leaf total.
        # (2026-08 numerai bug: the quantized CUDA kernel skipped that entirely,
        # so quantized runs on NaN data silently trained on wrong left sums and
        # off-by-one thresholds. The plain `missing` profile misses this: its
        # Gaussian features spread mass evenly, so mfb is never bin 0.)
        m = 20
        X = rng.choice([0.0, 1.0, 2.0, 3.0, 4.0], size=(n, m),
                       p=[0.4, 0.15, 0.15, 0.15, 0.15]).astype(np.float64)
        w = rng.standard_normal(m)
        y = (X - 2.0) @ w + 0.3 * rng.standard_normal(n)
        X[rng.random((n, m)) < 0.2] = np.nan
        base["max_bin"] = 5
    elif name == "missing-mfb0-wide":
        # NaN features, MORE than 256 histogram bins, and a most-frequent bin
        # of 0 -- the only shape that reaches the global-memory finder's
        # FORWARD na_as_missing head (na_as_missing && mfb_offset == 1), where
        # slot 0 is a missing pseudo-bin synthesised from the leaf totals
        # minus every real bin. Non-negative values are what buy mfb 0: the
        # binner keeps an arg-max bin only above the 0.7 sparse rate and
        # otherwise falls back to ValueToBin(0), which is bin 0 here.
        # `missing` at the same width only reaches the reverse NaN scan (its
        # Gaussian features straddle zero, so ValueToBin(0) is mid-range).
        m = 20
        X = rng.random((n, m))
        w = rng.standard_normal(m)
        y = (X - 0.5) @ w + 0.3 * rng.standard_normal(n)
        X = X.copy()
        X[rng.random((n, m)) < 0.2] = np.nan
        base["max_bin"] = 511
    elif name == "bigrow":
        # 2M rows x 64 quant bins x compact view: the packed-int16 shared-hist
        # overflow class (2026-07 production corruption: the compact-view grid
        # override dropped the 65534/bins rows-per-block guard; corruption
        # grows with leaf size, invisible below ~1M rows)
        n, m = 2_000_000, 20
        X = rng.integers(0, 5, size=(n, m)).astype(np.int8)
        w = rng.standard_normal(m) * (rng.random(m) < 0.5)
        signal = (X.astype(np.float64) - 2.0) @ w
        qs = np.quantile(signal, [0.05, 0.25, 0.75, 0.95])
        y = np.select([signal < qs[0], signal < qs[1], signal < qs[2], signal < qs[3]],
                      [0.0, 0.25, 0.5, 0.75], 1.0)
        base.update({"max_bin": 5, "learning_rate": 0.0015, "feature_fraction": 0.5,
                     "min_data_in_leaf": 40000, "num_leaves": 8192, "max_depth": 11,
                     "num_grad_quant_bins": 64})
    elif name == "tinygrad":
        # numerai-deep example regime shrunk to lattice scale: near-constant
        # [0,1] target, tiny learning rate, max_bin=5 -- the leaf-collapse /
        # quant-scale-underflow class (2026-07 fixedpoint 44-trees bug lived
        # here at FULL scale; the full-scale twin is canonical.py numerai-treecount)
        m = 100
        X = rng.integers(0, 5, size=(n, m)).astype(np.float64)
        w = rng.standard_normal(m) * (rng.random(m) < 0.1)
        y = 0.5 + 0.02 * ((X - 2.0) @ w) / max(1.0, np.abs(w).sum() ** 0.5) + 0.05 * rng.standard_normal(n)
        base.update({"max_bin": 5, "learning_rate": 0.0015, "feature_fraction": 0.1,
                     "min_data_in_leaf": 500})
    elif name == "categorical":
        m = 10
        X = rng.standard_normal((n, m))
        X[:, :3] = rng.integers(0, 30, size=(n, 3))
        y = X @ rng.standard_normal(m) + np.sin(X[:, 0]) + 0.3 * rng.standard_normal(n)
        base["categorical_feature"] = [0, 1, 2]
    elif name == "categorical-mc":
        # multiclass with skew-frequency categorical columns: the shape whose
        # non-constant hessians shrink toward zero as training fits, which is
        # what the bagged-quant hessian-quantum ridge exists for
        m = 10
        X = rng.standard_normal((n, m))
        pj = 1.0 / np.arange(1, 31) ** 1.2
        for j in range(3):
            X[:, j] = rng.choice(30, size=n, p=pj / pj.sum())
        y = (X @ rng.standard_normal((m, 4))).argmax(axis=1).astype(np.float64)
        base.update({"objective": "multiclass", "num_class": 4,
                     "categorical_feature": [0, 1, 2]})
    elif name == "sparse-mc":
        # five 80%-zero integer columns binned to 5 bins, against a SEPARABLE
        # 6-class argmax label. Few features x few bins means the tree keeps
        # carving the same handful of regions, and a separable label lets the
        # model fit them to near-certainty -- so late leaves hold almost no
        # gradient or hessian mass and their whole content is dither. The
        # sparsity is what makes the leaves tiny: 80% of every column sits in
        # one bin, so a split can only ever peel off a small group.
        m = 5
        X = rng.integers(-6, 7, size=(n, m)).astype(np.float64)
        X[rng.random((n, m)) < 0.8] = 0.0
        y = (X @ rng.standard_normal((m, 6))).argmax(axis=1).astype(np.float64)
        base.update({"objective": "multiclass", "num_class": 6,
                     "metric": "multi_logloss", "max_bin": 5})
    else:
        raise ValueError(f"unknown profile {name}")
    return X, np.asarray(y, dtype=np.float64), base

def eval_metric(params, model, X_test, y_test):
    pred = model.predict(X_test)
    if params.get("objective") == "binary":
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_test, pred)), "auc", True
    if params.get("objective") == "multiclass":
        from sklearn.metrics import log_loss
        return float(log_loss(y_test, pred, labels=list(range(params["num_class"])))), "mlogloss", False
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    return rmse, "rmse", False

def run_cell(cell):
    X, y, params = build_profile(cell["profile"])
    params.update(cell["params"])
    X_tr, y_tr, X_te, y_te = _split(X, y)
    t0 = time.monotonic()
    ds = lgb.Dataset(X_tr, label=y_tr, params=params)
    ds.construct()
    t_construct = time.monotonic() - t0
    t1 = time.monotonic()
    bst = lgb.train(params, ds, num_boost_round=cell["rounds"])
    t_train = time.monotonic() - t1
    model_str = bst.model_to_string()
    # fingerprint ONLY the trees: the trailing "parameters:" dump moves whenever
    # a config param is added/renamed (it did in the 2026-07 planner refactor
    # with zero behavior change) -- tree bytes are the actual behavior signature
    md5 = hashlib.md5(model_str.split("\nparameters:")[0].encode()).hexdigest()
    # validity: round-trip + finite predictions + tree count
    expected_trees = cell["rounds"] * params.get("num_class", 1)
    assert bst.num_trees() == expected_trees, f"tree count {bst.num_trees()} != {expected_trees}"
    pred = bst.predict(X_te)
    assert np.all(np.isfinite(pred)), "non-finite predictions"
    assert float(np.std(pred)) > 0.0, "constant predictions (garbage model)"
    reloaded = lgb.Booster(model_str=model_str)
    pred2 = reloaded.predict(X_te)
    assert np.array_equal(pred, pred2), "reloaded model predicts differently"
    metric, metric_name, higher_better = eval_metric(params, bst, X_te, y_te)
    return {"id": cell["id"], "ok": True, "md5": md5, "metric": metric,
            "metric_name": metric_name, "higher_better": higher_better,
            "construct_sec": round(t_construct, 4), "train_sec": round(t_train, 4)}

for cell in json.loads(sys.argv[1]):
    try:
        out = run_cell(cell)
    except Exception as e:  # noqa: BLE001 - report and continue; runner decides
        out = {"id": cell["id"], "ok": False, "error": f"{type(e).__name__}: {e}"}
    print("CELL " + json.dumps(out), flush=True)
"""


def run_cells(cells):
    """Run cells in chunked subprocesses; crash-isolate and collect results."""
    results = {}
    pending = list(cells)
    while pending:
        chunk, pending = pending[:WORKER_CHUNK], pending[WORKER_CHUNK:]
        proc = subprocess.run(
            [sys.executable, "-c", _WORKER_SOURCE, json.dumps(chunk)],
            capture_output=True,
            text=True,
            check=False,
        )
        seen = set()
        for line in proc.stdout.splitlines():
            if line.startswith("CELL "):
                rec = json.loads(line[5:])
                results[rec["id"]] = rec
                seen.add(rec["id"])
        if proc.returncode != 0:
            # the first unreported cell of the chunk crashed the worker;
            # record it and continue with the rest in a fresh worker
            unreported = [c for c in chunk if c["id"] not in seen]
            if unreported:
                crashed = unreported[0]
                results[crashed["id"]] = {
                    "id": crashed["id"],
                    "ok": False,
                    "error": f"worker crashed (exit {proc.returncode}): {proc.stderr[-800:]}",
                }
                pending = unreported[1:] + pending
            else:
                print(f"worker exited {proc.returncode} after all cells reported:\n{proc.stderr[-800:]}")
    return results


def check(cells, results, fingerprints, update_ids=None, write_all=False):
    """Compare results against fingerprints/equalities. Returns (failures, updates)."""
    failures = []
    new_fp = dict(fingerprints)
    for c in cells:
        r = results.get(c["id"])
        if r is None or not r.get("ok"):
            failures.append(f"{c['id']}: {'missing result' if r is None else r['error']}")
            continue
        if c["equal_to"] is not None:
            base = results.get(c["equal_to"])
            if not base or not base.get("ok"):
                failures.append(f"{c['id']}: base cell {c['equal_to']} failed; equality unchecked")
            elif r["md5"] != base["md5"]:
                failures.append(
                    f"{c['id']}: NOT bit-identical to {c['equal_to']} "
                    f"({r['md5']} != {base['md5']}) -- this plan decision changes the model"
                )
            continue
        if c["fingerprint"]:
            known = fingerprints.get(c["id"])
            if write_all or (update_ids and c["id"] in update_ids) or known is None:
                new_fp[c["id"]] = {
                    "md5": r["md5"],
                    "metric": round(r["metric"], 6),
                    "metric_name": r["metric_name"],
                }
                if known is None and not write_all and not (update_ids and c["id"] in update_ids):
                    failures.append(f"{c['id']}: no fingerprint on record -- run --update '{c['id']}' to baseline it")
            elif r["md5"] != known["md5"]:
                failures.append(
                    f"{c['id']}: md5 {r['md5']} != baseline {known['md5']} "
                    f"(metric {r['metric']:.6f} vs baseline {known['metric']:.6f} {known['metric_name']}) "
                    f"-- unintended behavior change, or re-baseline explicitly with --update"
                )
        else:
            # nondeterministic cell: metric floor vs recorded baseline
            known = fingerprints.get(c["id"])
            if write_all or (update_ids and c["id"] in update_ids) or known is None:
                new_fp[c["id"]] = {
                    "metric": round(r["metric"], 6),
                    "metric_name": r["metric_name"],
                }
                if known is None and not write_all and not (update_ids and c["id"] in update_ids):
                    failures.append(f"{c['id']}: no metric baseline -- run --update '{c['id']}'")
            else:
                base_m, cur = known["metric"], r["metric"]
                tol = abs(base_m) * METRIC_TOLERANCE
                worse = cur < base_m - tol if r["higher_better"] else cur > base_m + tol
                if worse:
                    failures.append(
                        f"{c['id']}: {r['metric_name']} {cur:.6f} regressed past baseline "
                        f"{base_m:.6f} (tolerance {METRIC_TOLERANCE:.0%})"
                    )
    return failures, new_fp


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--update", nargs="+", metavar="CELL_ID")
    mode.add_argument("--list", action="store_true")
    ap.add_argument(
        "--only",
        nargs="+",
        metavar="CELL_ID",
        help="restrict the run to these cells (+ their bases)",
    )
    args = ap.parse_args()

    cells = build_cells()
    if args.list:
        for c in cells:
            kind = "equal_to:" + c["equal_to"] if c["equal_to"] else ("md5" if c["fingerprint"] else "metric")
            print(f"{c['id']:40s} {c['profile']:12s} rounds={c['rounds']:<3d} {kind}")
        print(f"{len(cells)} cells")
        return 0

    if args.only:
        keep = set(args.only)
        for c in cells:
            if c["id"] in keep and c["equal_to"]:
                keep.add(c["equal_to"])
        cells = [c for c in cells if c["id"] in keep]

    fingerprints = {}
    if FINGERPRINT_FILE.exists():
        fingerprints = json.loads(FINGERPRINT_FILE.read_text())["cells"]

    t0 = time.monotonic()
    results = run_cells(cells)
    elapsed = time.monotonic() - t0

    failures, new_fp = check(
        cells,
        results,
        fingerprints,
        update_ids=set(args.update or []),
        write_all=args.baseline,
    )

    RESULTS_FILE.write_text(json.dumps({"results": results, "elapsed_sec": round(elapsed, 2)}, indent=1))
    n_ok = sum(1 for r in results.values() if r.get("ok"))
    print(f"lattice: {n_ok}/{len(cells)} cells ok in {elapsed:.1f}s; results -> {RESULTS_FILE}")

    if args.baseline or args.update:
        FINGERPRINT_FILE.write_text(json.dumps({"cells": dict(sorted(new_fp.items()))}, indent=1) + "\n")
        print(f"fingerprints written -> {FINGERPRINT_FILE}")
        failures = [f for f in failures if "no fingerprint" not in f and "no metric baseline" not in f]

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("all gates green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
