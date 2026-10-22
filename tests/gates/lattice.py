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
  python tests/gates/lattice.py --update 'route/*' --only 'route/*'   # ids are fnmatch patterns
  python tests/gates/lattice.py --list               # print the cell matrix
Cells run sequentially in chunked worker subprocesses (crash isolation: a
crashing cell fails alone and the runner respawns for the rest).
"""

import argparse
import fnmatch
import json
import os
import re
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
#   finder_tier  the quantized level split finder every tree must take, as the
#            FALCATA_DEBUG=diag line names it ("warp (tier: ...)" or "block"),
#            or a list: the set of finders the run's trees must log, exactly;
#            such a cell runs alone in a worker with FALCATA_DEBUG=diag
#   diag_set     (regex with one group, [values]): under FALCATA_DEBUG=diag the
#            set of the group's captures over the run must be exactly these
#            values (e.g. every tree's compact fill logs "pair_code5 on"); such a
#            cell runs alone in a worker like a finder_tier cell
#   debug    extra FALCATA_DEBUG tokens (e.g. ["vramfree=1"]); FALCATA_DEBUG is
#            read once per process, so such a cell runs alone in a worker
#   forced_splits  a forced-split JSON object: the worker writes it to a temp
#            file and passes it as forcedsplits_filename
#   fingerprint_strip  model-text line prefixes left out of the md5 (e.g. the
#            linear leaf fit's leaf_const= / leaf_coeff= and tree_sizes=)
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
        leaf_bound=None,
        finder_tier=None,
        diag_set=None,
        debug=None,
        forced_splits=None,
        fingerprint_strip=None,
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
                "leaf_bound": leaf_bound,
                "finder_tier": finder_tier,
                "diag_set": diag_set,
                "debug": debug,
                "forced_splits": forced_splits,
                "fingerprint_strip": fingerprint_strip,
            }
        )

    # --- base fingerprints: every profile x stochastic quant, shallow ------- #
    for profile in [
        "dense",
        "fewbin",
        "sampled",
        "sampledwide",
        "graph",
        "int8wide",
        "efb",
        "imbalanced",
        "missing",
        "missing-mfb0",
        "missing-mfb0-dense",
    ]:
        cell(f"{profile}/quant", profile, perf=profile in ("dense", "int8wide"))

    # deep + classic twins for the missing-bin-0 shape: the synthesised bin 0
    # feeds the level-batched and the classic split search through the same
    # kernel, and only the deep cell exercises enough leaves to catch a
    # threshold that is right at the root and wrong further down.
    cell("missing-mfb0/quant-deep", "missing-mfb0", {"num_leaves": 255, "max_depth": 12, "min_data_in_leaf": 5})
    cell("missing-mfb0/quant-classic", "missing-mfb0", {"cuda_plan": "auto,hybrid:off"})
    cell("missing-mfb0/fixedpoint", "missing-mfb0", {"quant_mode": "fixedpoint"})
    cell(
        "missing-mfb0/flip-dense_rowdata-deep",
        "missing-mfb0",
        {"num_leaves": 255, "max_depth": 12, "min_data_in_leaf": 5, "cuda_plan": "auto,dense_rowdata:off"},
        equal_to="missing-mfb0/quant-deep",
    )
    cell(
        "missing-mfb0/flip-dense_rowdata-fixedpoint",
        "missing-mfb0",
        {"quant_mode": "fixedpoint", "cuda_plan": "auto,dense_rowdata:off"},
        equal_to="missing-mfb0/fixedpoint",
    )
    # missing-mfb0's rows are sparse (bin 0 holds over a quarter of the values), so its quantized levels never
    # reach the batched level finder. The dense twin does, and every task of it is a NaN scan with an unstored
    # bin 0: warp_find's forward scan has to rebuild bin 0 from the leaf total there. Shallow and deep, each
    # pinned and flipped against the block kernel (the shallow flip is in the plan-flip list below).
    deep = {"num_leaves": 255, "max_depth": 12, "min_data_in_leaf": 5}
    cell("missing-mfb0-dense/quant-deep", "missing-mfb0-dense", deep)
    cell(
        "missing-mfb0-dense/flip-warp_find-deep",
        "missing-mfb0-dense",
        {**deep, "cuda_plan": "auto,warp_find:off"},
        equal_to="missing-mfb0-dense/quant-deep",
    )

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
    # (the quantized graph prefix's graph_loop flip, graph/flip*-graph_loop, went with graph_quant in 1.2, and the
    # non-quantized graph loop with graph_loop / graph_det in P3: device_tree:off flips the nonquant cells below)
    flips = [
        ("fewbin", "rowdata_4bit", {}, {"cuda_plan": "auto,rowdata_4bit:off"}),
        ("dense", "fast_rowdata", {}, {"cuda_plan": "auto,fast_rowdata:off"}),
        ("dense", "gpu_construct", {}, {"cuda_plan": "auto,gpu_construct:off"}),
        ("efb", "efb_precheck", {}, {"cuda_plan": "auto,efb_precheck:off"}),
        ("sampled", "compact_quant", {}, {"cuda_plan": "auto,compact_quant:off"}),
        ("sampled", "compact_prefill", {}, {"cuda_plan": "auto,compact_prefill:on"}),
        # the column-major copy is made before the second tree; off, every fill reads the row-major matrix
        ("sampled", "colmajor_fill", {}, {"cuda_plan": "auto,colmajor_fill:off"}),
        ("sampledwide", "colmajor_fill", {}, {"cuda_plan": "auto,colmajor_fill:off"}),
        # colmajor_direct: the store (compact regime) or the full view (mask regime) built from the Dataset's columns
        # against the row-major matrix built at Init; fewbin is feature_fraction 1, i.e. the mask regime
        ("sampled", "colmajor_direct", {}, {"cuda_plan": "auto,colmajor_direct:off"}),
        ("sampledwide", "colmajor_direct", {}, {"cuda_plan": "auto,colmajor_direct:off"}),
        ("fewbin", "colmajor_direct", {}, {"cuda_plan": "auto,colmajor_direct:off"}),
        # the two regimes against each other: sampled trees under column masks, every column through a compact view
        ("sampled", "view_mask", {}, {"cuda_plan": "auto,view_mode:mask"}),
        ("sampledwide", "view_mask", {}, {"cuda_plan": "auto,view_mode:mask"}),
        ("fewbin", "view_compact", {}, {"cuda_plan": "auto,view_mode:compact"}),
        ("sampled", "tiled_fill", {}, {"cuda_plan": "auto,tiled_fill:off"}),
        ("sampledwide", "tiled_fill", {}, {"cuda_plan": "auto,tiled_fill:off"}),
        ("dense", "warp_find", {}, {"cuda_plan": "auto,warp_find:off"}),
        ("fewbin", "warp_find", {}, {"cuda_plan": "auto,warp_find:off"}),
        ("missing", "warp_find", {}, {"cuda_plan": "auto,warp_find:off"}),
        # NaN with an unstored bin 0: the forward scan rebuilds bin 0 from the leaf total
        ("missing-mfb0-dense", "warp_find", {}, {"cuda_plan": "auto,warp_find:off"}),
        ("sampled", "warp_find", {}, {"cuda_plan": "auto,warp_find:off"}),
        ("dense", "row_batch", {}, {"cuda_plan": "auto,row_batch:off"}),
        ("sampled", "row_batch", {}, {"cuda_plan": "auto,row_batch:off"}),
        ("int8wide", "row_batch", {}, {"cuda_plan": "auto,row_batch:off"}),
        ("fewbin", "row_batch", {}, {"cuda_plan": "auto,row_batch:off"}),
        ("int8wide", "rowdata_4bit", {}, {"cuda_plan": "auto,rowdata_4bit:off"}),
        ("int8wide", "fast_rowdata", {}, {"cuda_plan": "auto,fast_rowdata:off"}),
        # construct_tile_bin: the int8 LUT binning of one-column groups through the shared-memory tile kernel (400
        # columns: three full 128-column tiles and a partial one)
        ("int8wide", "construct_tile_bin", {}, {"cuda_plan": "auto,construct_tile_bin:off"}),
        # async_tree_start: the host-launched quantized flow with and without a compact view
        ("dense", "async_tree_start", {}, {"cuda_plan": "auto,async_tree_start:off"}),
        ("sampled", "async_tree_start", {}, {"cuda_plan": "auto,async_tree_start:off"}),
        ("sampledwide", "async_tree_start", {}, {"cuda_plan": "auto,async_tree_start:off"}),
        ("int8wide", "async_tree_start", {}, {"cuda_plan": "auto,async_tree_start:off"}),
        ("dense", "leaf_map_small_blocks", {}, {"cuda_plan": "auto,leaf_map_small_blocks:off"}),
        # dense_rowdata: quantized sparse-ish data (average sparse rate >= 0.25: efb 0.93, missing-mfb0 0.32) on the
        # dense row-wise bin and the batched level kernels; off, the sparse bin and the per-leaf kernels
        ("efb", "dense_rowdata", {}, {"cuda_plan": "auto,dense_rowdata:off"}),
        ("missing-mfb0", "dense_rowdata", {}, {"cuda_plan": "auto,dense_rowdata:off"}),
    ]
    for profile, key, base_params, flip_params in flips:
        base_id = f"{profile}/quant" if not base_params else f"{profile}/flipbase-{key}"
        if base_params:
            cell(base_id, profile, base_params)
        cell(f"{profile}/flip-{key}", profile, flip_params, equal_to=base_id)
    # the 8-bit compact construct (4-bit disabled), fingerprint-locked; it was the base of the construct JIT's
    # 8-bit flip (the JIT was removed 2026-10-15) and keeps its id so the fingerprint carries over
    cell("sampled/flipbase-jit8", "sampled", {"cuda_plan": "auto,rowdata_4bit:off"})

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

    # --- non-quant default plan: deterministic since cuda_plan key nonquant_deterministic (default on) --- #
    # the levels the float atomics used to build (graph loop, compact views, levels wider than the det slab)
    # accumulate in 64-bit fixed point; the fp32 opt-in (cuda_precision) keeps its fp32 gain math on hist_t
    # histograms. Fingerprinted since 2026-10-16 (they were metric-only before).
    cell(
        "dense/nonquant",
        "dense",
        {"quant_mode": "none"},
        rounds=50,
        perf=True,
    )
    cell(
        "dense/fp32",
        "dense",
        {"quant_mode": "none", "cuda_precision": "fp32"},
        rounds=50,
        perf=True,
    )
    # --- nondeterministic tier: validity + metric floor only ---------------- #
    # nonquant_deterministic:off keeps the float-atomic kernels: the only mode
    # whose models vary run to run, exactly the path md5 cannot cover.
    cell(
        "dense/nonquant-atomic",
        "dense",
        {"quant_mode": "none", "cuda_plan": "auto,nonquant_deterministic:off"},
        rounds=50,
        fingerprint=False,
    )
    # Non-quant DETERMINISTIC cells: on these shapes every construct runs the
    # deterministic kernels (per-leaf, batched-level, compact view, sparse
    # fallback), so their models fingerprint like quant cells do. The
    # device_tree:off det cells pin the one-sync host loop with the double-slot
    # deterministic batched construct (graph_loop:off's flow before P3); the
    # DEFAULT-plan cells pin the complete device tree's fixed-point levels.
    cell("dense/nonquant-det", "dense", {"quant_mode": "none", "cuda_plan": "auto,device_tree:off"}, rounds=50)
    # hybrid:off -- the BATCHED level flow keeps the atomic kernel for compact
    # views (per-leaf det covers them; batched does not yet), so the compact
    # det lock pins the per-leaf flow.
    cell("sampled/nonquant-det", "sampled", {"quant_mode": "none", "cuda_plan": "auto,hybrid:off"}, rounds=50)
    cell("missing/nonquant-det", "missing", {"quant_mode": "none", "cuda_plan": "auto,device_tree:off"}, rounds=50)
    cell(
        "categorical/nonquant-det",
        "categorical",
        {"quant_mode": "none", "cuda_plan": "auto,device_tree:off"},
        rounds=50,
    )
    # the double-slot deterministic models of the graph and imbalanced shapes (graph_det:on's before P3, which the
    # graph loop's det nodes trained bit-identically to the host det loop). imbalanced is the shape whose atomic/det
    # mix raced in the 2026-08-19 lattice.
    cell("graph/nonquant-det", "graph", {"quant_mode": "none", "cuda_plan": "auto,device_tree:off"}, rounds=50)
    cell(
        "imbalanced/nonquant-det", "imbalanced", {"quant_mode": "none", "cuda_plan": "auto,device_tree:off"}, rounds=50
    )
    # the DEFAULT plan: the complete device tree's fixed-point construct + convert levels (fingerprinted since
    # nonquant_deterministic; the graph loop's model before P3)
    cell("graph/nonquant", "graph", {"quant_mode": "none"}, rounds=50)
    # ... equal, on this shape, to the double-slot deterministic host loop (empirical: both sums are exact here)
    cell(
        "graph/flip-device_tree-nonquant",
        "graph",
        {"quant_mode": "none", "cuda_plan": "auto,device_tree:off"},
        rounds=50,
        equal_to="graph/nonquant",
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

    # --- vector-leaf multi-target training --------------------------------- #
    # The fixedpoint cell is a permanent fingerprint for the quantized vector
    # finder, its per-target gradient scales, and the selective level prefix.
    # The fp64 twin fingerprints the deterministic per-leaf dense construct and
    # vector finder. Both cells also gate worst-target normalized RMSE.
    cell(
        "vector-leaf/fixedpoint",
        "vector-leaf",
        {"quant_mode": "fixedpoint", "quant_bins": 16},
        rounds=30,
    )
    cell(
        "vector-leaf/nonquant",
        "vector-leaf",
        {"quant_mode": "none"},
        rounds=30,
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
    # dense_rowdata off: the 81%-zero columns on the sparse row-wise bin (per-leaf kernels)
    cell(
        "sparse-mc/flip-dense_rowdata-deep",
        "sparse-mc",
        {
            "quant_mode": "fixedpoint",
            "quant_bins": 16,
            "num_leaves": 63,
            "max_depth": 10,
            "min_data_in_leaf": 5,
            "cuda_plan": "auto,dense_rowdata:off",
        },
        rounds=80,
        equal_to="sparse-mc/quant-deep",
    )

    # unbagged fixedpoint multiclass at the benchmark's own defaults (lambda_l2 0,
    # min_data_in_leaf 20, 64 quant bins) on covtype's class mix, rare classes
    # included. The objective starts every row at the log class prior, so the
    # first trees take large Newton steps on the rare classes; rows pushed past
    # their class saturate (|g| near 1, h near 0), and a child of such rows plus
    # well-fit rows records a few quanta of hessian against many quanta of true
    # hessian mass. Without the hessian-quantum ridge the gain argmax harvests
    # those children and the leaf outputs run away by orders of magnitude (the
    # covtype fixedpoint collapse). The md5 pins the model; leaf_bound pins the
    # property, so re-baselining the md5 cannot wave a collapse through. Two
    # draws of the one shape (the profile name seeds the data): each reaches
    # the collapse without the ridge, through a different first leaf.
    for profile in ["rare-mc-b", "rare-mc-c"]:
        cell(
            f"{profile}/fixedpoint-deep",
            profile,
            {"quant_mode": "fixedpoint", "num_leaves": 255, "max_depth": 10},
            rounds=80,
            leaf_bound=500.0,
        )
        # dense_rowdata off: the six 80%-zero binary columns put the average sparse rate at 0.40 (sparse row-wise bin)
        cell(
            f"{profile}/flip-dense_rowdata-fixedpoint-deep",
            profile,
            {"quant_mode": "fixedpoint", "num_leaves": 255, "max_depth": 10, "cuda_plan": "auto,dense_rowdata:off"},
            rounds=80,
            leaf_bound=500.0,
            equal_to=f"{profile}/fixedpoint-deep",
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
    # device_tree:off needed.
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

    # --- tree boundary, level pruning, level apply bookkeeping --------------- #
    # A full tree: num_leaves = 2^max_depth, so the leaf budget never binds and the level prefix ends on a complete
    # final level (skip_empty_tail skips the tail search, early_leaf_map writes the residual leaf map with that
    # level); min_data_in_leaf 40 leaves 40-80-row leaves that cannot split before it (skip_unsplittable) and
    # terminal index ranges that later levels used to copy again (gap_copy_once). The sampled profile's compact view
    # takes the pair-joint construct (one partition: not the whole-row launch the pair_* keys reshape, see ff15
    # below). Each key off, and all eight off, must train the same model.
    full = {"num_leaves": 64, "max_depth": 6, "min_data_in_leaf": 40}
    night2_keys = [
        "pair_block_rows",
        "pair_capped_rows",
        "async_tree_start",
        "skip_unsplittable",
        "skip_empty_tail",
        "early_leaf_map",
        "gap_copy_once",
        "leaf_map_small_blocks",
    ]
    for profile in ["sampled", "dense"]:
        cell(f"{profile}/quant-fulltree", profile, full)
        for key in night2_keys:
            cell(
                f"{profile}/flip-{key}-fulltree",
                profile,
                {**full, "cuda_plan": f"auto,{key}:off"},
                equal_to=f"{profile}/quant-fulltree",
            )
        cell(
            f"{profile}/flip-night2-fulltree",
            profile,
            {**full, "cuda_plan": "auto," + ",".join(f"{k}:off" for k in night2_keys)},
            equal_to=f"{profile}/quant-fulltree",
        )
    # whole-row pair-joint construct: sampledwide at feature_fraction 0.15 samples ~180 nibble columns over two
    # partitions, a 90-byte interleaved row. The default build launches 3-row blocks; pair_block_rows takes 11 rows
    # (more resident warps at 64 registers) and pair_capped_rows the 48-register build in 7-row blocks (nsys).
    # (At the profiles' own feature fractions neither key changes the launch: those blocks already fill the SM.)
    ff15 = {"feature_fraction": 0.15}
    cell("sampledwide/quant-ff15", "sampledwide", ff15)
    for name, plan in [
        ("pair_capped_rows", "pair_capped_rows:off"),
        ("pair_block_rows", "pair_block_rows:off"),
        ("pair_both", "pair_block_rows:off,pair_capped_rows:off"),
    ]:
        cell(
            f"sampledwide/flip-{name}-ff15",
            "sampledwide",
            {**ff15, "cuda_plan": f"auto,{plan}"},
            equal_to="sampledwide/quant-ff15",
        )
    # deep trees over tiny leaves, the tiny-gradient regime's 500-row leaves (150 rounds), bagged multiclass, and the
    # quantized graph prefix, which reads the root sums async_tree_start defers (the search's lattice caught it)
    for key in ["skip_unsplittable", "gap_copy_once", "early_leaf_map"]:
        cell(
            f"dense/flip-{key}-deep",
            "dense",
            {"num_leaves": 255, "max_depth": 12, "min_data_in_leaf": 5, "cuda_plan": f"auto,{key}:off"},
            equal_to="dense/quant-deep",
        )
    for key in ["skip_unsplittable", "gap_copy_once", "async_tree_start"]:
        cell(
            f"tinygrad/flip-{key}", "tinygrad", {"cuda_plan": f"auto,{key}:off"}, rounds=150, equal_to="tinygrad/quant"
        )
    cell(
        "categorical-mc/flip-async_tree_start-bagged",
        "categorical-mc",
        {
            "quant_mode": "fixedpoint",
            "quant_bins": 16,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "num_leaves": 255,
            "max_depth": 10,
            "min_data_in_leaf": 5,
            "cuda_plan": "auto,async_tree_start:off",
        },
        rounds=50,
        equal_to="categorical-mc/quant-bagged",
    )

    # --- fused fill, level find/sync/fix/apply kernels, host round trips ------ #
    # Every key below is default on and exact by contract; each off (and all of them off) must train the base cell's
    # model. On the full trees (nsys, key on vs off): the sampled profile's 4-bit compact view takes the fused root
    # fill (root_hist_short_blocks; root_hist_block_rounds:0 is 16-tile blocks; the L2 lead runs on its word path),
    # the strided 4-lane level finder (5 thresholds per task), the used-task sync, the fused fix + subtract over the
    # sampled features and the pair-joint construct of multi-pair levels (pair_block_map); dense takes the
    # per-1024-task sync; both run the batched apply with gaps (min_data_in_leaf 40), the level readbacks, the tree
    # boundary and the tree-completing final level (final_map_only, final_readback_first). The fill, finder and
    # construct keys are inert on dense (no compact view, 255 bins), colmajor_align on both (an aligned pitch) and
    # shape_memo on both (no whole-row construct): sampled-oddrows below reaches the first, sampledwide-ff15 the
    # second (gdb: 30 carveout calls with the memo, 52 without). nsys does not see the L2 lead (an argument of the
    # same fill kernel); ncu does: the fill's instructions per launch drop by 2.5% with root_hist_l2_prefetch off.
    night3_keys = [
        "root_hist_l2_prefetch",
        "colmajor_align",
        "root_hist_short_blocks",
        "sync_copy_batched",
        "fix_subtract_fused",
        "readback_kernel",
        "invalidate_async",
        "level_apply_first",
        "gradients_no_sync",
        "tree_meta_batch",
        "tree_end_prealloc",
        "pair_block_map",
        "final_map_only",
        "final_readback_first",
        "shape_memo",
    ]
    night3_off = "auto," + ",".join(f"{k}:off" for k in night3_keys) + ",root_hist_block_rounds:0"
    for profile in ["sampled", "dense"]:
        for key in night3_keys:
            cell(
                f"{profile}/flip-{key}-fulltree",
                profile,
                {**full, "cuda_plan": f"auto,{key}:off"},
                equal_to=f"{profile}/quant-fulltree",
            )
        cell(
            f"{profile}/flip-night3-fulltree",
            profile,
            {**full, "cuda_plan": night3_off},
            equal_to=f"{profile}/quant-fulltree",
        )
    cell(
        "sampled/flip-root_hist_block_rounds0-fulltree",
        "sampled",
        {**full, "cuda_plan": "auto,root_hist_block_rounds:0"},
        equal_to="sampled/quant-fulltree",
    )
    # an odd column pitch: master's fill reads bytes there, colmajor_align's aligned store engages the word staging
    # and with it the L2 lead (the other profiles' 6400 training rows already give an aligned pitch)
    cell("sampled-oddrows/quant", "sampled-oddrows")
    for key in ["colmajor_align", "root_hist_l2_prefetch", "root_hist_short_blocks"]:
        cell(
            f"sampled-oddrows/flip-{key}",
            "sampled-oddrows",
            {"cuda_plan": f"auto,{key}:off"},
            equal_to="sampled-oddrows/quant",
        )
    cell("sampled-oddrows/flip-night3", "sampled-oddrows", {"cuda_plan": night3_off}, equal_to="sampled-oddrows/quant")
    # the whole-row pair-joint construct on two partitions: the capped build's per-shape memo (shape_memo; every
    # tree's column sample is a new joint-table size) and the block map of the interleaved row's launches
    for key in ["pair_block_map", "shape_memo", "fix_subtract_fused", "tree_meta_batch", "colmajor_align"]:
        cell(
            f"sampledwide/flip-{key}-ff15",
            "sampledwide",
            {**ff15, "cuda_plan": f"auto,{key}:off"},
            equal_to="sampledwide/quant-ff15",
        )
    cell(
        "sampledwide/flip-night3-ff15",
        "sampledwide",
        {**ff15, "cuda_plan": night3_off},
        equal_to="sampledwide/quant-ff15",
    )
    # features of 3 to 6 values: the whole-row construct sees another joint-table size nearly every tree, so the
    # capped build's memo holds many shapes (whose saving shows where shapes repeat, sampledwide-ff15 above)
    cell("sampledwide-mixbins/quant-ff15", "sampledwide-mixbins", ff15)
    for key in ["shape_memo", "pair_block_map"]:
        cell(
            f"sampledwide-mixbins/flip-{key}-ff15",
            "sampledwide-mixbins",
            {**ff15, "cuda_plan": f"auto,{key}:off"},
            equal_to="sampledwide-mixbins/quant-ff15",
        )
    cell(
        "sampledwide-mixbins/flip-night3-ff15",
        "sampledwide-mixbins",
        {**ff15, "cuda_plan": night3_off},
        equal_to="sampledwide-mixbins/quant-ff15",
    )
    # the tiny-gradient regime (4-bit, 5 bins, feature_fraction 0.1, 500-row leaves) over 150 rounds
    for key in ["fix_subtract_fused", "sync_copy_batched"]:
        cell(
            f"tinygrad/flip-{key}", "tinygrad", {"cuda_plan": f"auto,{key}:off"}, rounds=150, equal_to="tinygrad/quant"
        )
    cell("tinygrad/flip-night3", "tinygrad", {"cuda_plan": night3_off}, rounds=150, equal_to="tinygrad/quant")
    # --- the generic warp split finder on mixed and wide scans ------------------------------------------------ #
    # warp_find_mid_ppl runs a wide task at 2, 4 or 8 positions per lane (up to 64, 128, 256) instead of 8;
    # find_prune_fp32 prunes from fp32 bounds of the integer prefix (one exact verify per lane, an exact screen, a
    # per-lane work list); find_prune_sure takes its lower bound from every position passing the gates with a sure
    # fp32 margin instead of the exact verify; find_compact_survivors hands the warp's surviving positions out 32 at
    # a time;
    # find_select_int reduces the best split in integer compares of the gain bits (also on the 2-per-lane tier:
    # fewbin's 12-bin tasks). Each key off and all five off must train the base cell's model on mixwidth (every
    # tier, its narrow tasks as 4-lane groups in the same launch, NaN scans; its base fingerprints were recorded with
    # master's build), dense (255-bin scans) and missing (NaN scans).
    finder_keys = [
        "warp_find_mid_ppl",
        "find_prune_fp32",
        "find_prune_sure",
        "find_select_int",
        "find_compact_survivors",
    ]
    finder_flips = [
        *[(key, f"{key}:off") for key in finder_keys],
        ("jit1-finder", ",".join(f"{k}:off" for k in finder_keys)),
    ]
    cell("mixwidth/quant", "mixwidth")
    cell("mixwidth/quant-deep", "mixwidth", deep)
    for suffix, base_params, base in [("", {}, "mixwidth/quant"), ("-deep", deep, "mixwidth/quant-deep")]:
        for name, plan in finder_flips:
            cell(
                f"mixwidth/flip-{name}{suffix}", "mixwidth", {**base_params, "cuda_plan": f"auto,{plan}"}, equal_to=base
            )
    for name, plan in finder_flips:
        cell(f"dense/flip-{name}-deep", "dense", {**deep, "cuda_plan": f"auto,{plan}"}, equal_to="dense/quant-deep")
        cell(f"missing/flip-{name}", "missing", {"cuda_plan": f"auto,{plan}"}, equal_to="missing/quant")
    cell("fewbin/flip-find_select_int", "fewbin", {"cuda_plan": "auto,find_select_int:off"}, equal_to="fewbin/quant")
    # the warp finder's tiers by scan width (no keys): tasks of at most 8 positions as 4-lane groups (sampled's
    # 6-value columns), wider ones a warp each (dense's 255 bins, fewbin's 12 values), both in one launch (mixwidth);
    # FALCATA_DEBUG=diag names the tier on every tree's single-pair levels, and the model is the base cell's
    for profile, tier in [
        ("sampled", "warp (tier: few-position)"),
        ("dense", "warp (tier: wide)"),
        ("fewbin", "warp (tier: wide)"),
        ("mixwidth", "warp (tier: few-position + wide)"),
    ]:
        cell(f"{profile}/finder-tier", profile, {"verbose": 1}, equal_to=f"{profile}/quant", finder_tier=tier)
    # the tier boundaries: boundwidth's columns have 8-11, 64-67, 128-131 and 255 bins (k values as 0..k-1, 1..k and
    # 1..k with NaN, for k = 8, 9, 64, 65, 128, 129), so tasks sit at 8 and 9, 64 and 65, 128 and 129 scan positions:
    # the few-position tier's last width, the wide tier's first, and warp_find_mid_ppl's 2/4 and 4/8 per-lane steps.
    # Its base fingerprints were recorded with this branch's build; the block kernel (warp_find off) and the five
    # finder keys off must train the same model. narrowplus (57 columns of at most 6 values and one 33-value column at
    # feature_fraction 0.2) has trees that draw only narrow tasks of a mixed dataset: those take the launch without
    # wide units over the narrow-first order list (no key forces the wide launch there, so the block kernel is the
    # equality), and both tiers appear in one run.
    for profile, base_params, suffix in [("boundwidth", {}, ""), ("boundwidth", deep, "-deep"), ("narrowplus", {}, "")]:
        base = f"{profile}/quant{suffix}"
        cell(base, profile, base_params)
        for name, plan in [("warp_find", "warp_find:off"), finder_flips[-1]]:
            cell(f"{profile}/flip-{name}{suffix}", profile, {**base_params, "cuda_plan": f"auto,{plan}"}, equal_to=base)
    cell(
        "boundwidth/finder-tier",
        "boundwidth",
        {"verbose": 1},
        equal_to="boundwidth/quant",
        finder_tier="warp (tier: few-position + wide)",
    )
    cell(
        "narrowplus/finder-tier",
        "narrowplus",
        {"verbose": 1},
        equal_to="narrowplus/quant",
        finder_tier=["warp (tier: few-position)", "warp (tier: few-position + wide)"],
    )

    # deep trees with a binding leaf budget (no tree-completing level), the bagged multiclass categorical cell
    # (level_apply_first keeps the record first for categorical splits; final_map_only is off with bagging)
    for key in ["readback_kernel"]:
        cell(
            f"dense/flip-{key}-deep",
            "dense",
            {**deep, "cuda_plan": f"auto,{key}:off"},
            equal_to="dense/quant-deep",
        )
    cell("sampled/flip-night3-deep", "sampled", {**deep, "cuda_plan": night3_off}, equal_to="sampled/quant-deep")
    cell(
        "categorical-mc/flip-night3-bagged",
        "categorical-mc",
        {
            "quant_mode": "fixedpoint",
            "quant_bins": 16,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "num_leaves": 255,
            "max_depth": 10,
            "min_data_in_leaf": 5,
            "cuda_plan": night3_off,
        },
        rounds=50,
        equal_to="categorical-mc/quant-bagged",
    )

    # --- the 5-bit pair-code view (pair_code5) ------------------------------- #
    # sampledwide5 is sampledwide with 5 values per feature: every byte's joint table is 5 x 5 = 25 cells, within the
    # 5-bit code, so the fused fill writes the code view and the whole-row construct reads it (sampledwide's 6-value
    # columns give 36-cell tables, which decline it, as does every other profile). Its base cells' fingerprints were
    # recorded with master's build, which writes and reads the nibble view throughout. At feature_fraction 0.15 a row
    # is ~90 bytes, 16 code words: the one-word pack's block covers whole rows (pair_code5_pack2:off); at the profile's
    # own 0.2 it is ~120 bytes, 24 words, where pair_code5_pack2:off takes the per-byte pack. pair_code5_words:off
    # takes the per-byte code kernel (capped, or uncapped with pair_capped_rows:off); compact_prefill turns
    # the view off at the fill. Deep trees (a binding leaf budget) run the code-word kernel on
    # multi-pair levels, with and without the block map and level_row_blocks' shared rows per thread. Each key off,
    # and all four off, must train the base cell's model. (FALCATA_DEBUG=diag: every tree of the three bases writes
    # the code view and every gathering level reads it; pair_code5_pack2 and pair_code5_l2_store are an argument of
    # the same fill launch, invisible to nsys, which ncu and the kernel's SASS see.)
    night4_keys = ["pair_code5", "pair_code5_words", "pair_code5_pack2", "pair_code5_l2_store"]
    night4_off = "auto," + ",".join(f"{k}:off" for k in night4_keys)
    cell("sampledwide5/quant-ff15", "sampledwide5", ff15)
    for name, plan in [
        *[(key, f"{key}:off") for key in night4_keys],
        ("night4", night4_off.removeprefix("auto,")),
        ("code5_bytes_uncapped", "pair_code5_words:off,pair_capped_rows:off"),
        ("compact_prefill", "compact_prefill:on"),
    ]:
        cell(
            f"sampledwide5/flip-{name}-ff15",
            "sampledwide5",
            {**ff15, "cuda_plan": f"auto,{plan}"},
            equal_to="sampledwide5/quant-ff15",
        )
    cell("sampledwide5/quant", "sampledwide5")
    for key in ["pair_code5", "pair_code5_pack2"]:
        cell(
            f"sampledwide5/flip-{key}", "sampledwide5", {"cuda_plan": f"auto,{key}:off"}, equal_to="sampledwide5/quant"
        )
    cell("sampledwide5/flip-night4", "sampledwide5", {"cuda_plan": night4_off}, equal_to="sampledwide5/quant")
    cell("sampledwide5/quant-deep", "sampledwide5", deep)
    for name, plan in [
        ("pair_code5", "pair_code5:off"),
        ("pair_code5_words", "pair_code5_words:off"),
        ("code5_plain_grid", "pair_block_map:off,level_row_blocks:off,per_pair_rows:off"),
        ("night4", night4_off.removeprefix("auto,")),
    ]:
        cell(
            f"sampledwide5/flip-{name}-deep",
            "sampledwide5",
            {**deep, "cuda_plan": f"auto,{plan}"},
            equal_to="sampledwide5/quant-deep",
        )

    # --- the code-word construct's grid, block and adds ------------------------ #
    # pair_code5_slots sizes a code-word level's grid to the kernel's resident blocks (one rows-per-thread for all
    # its pairs; the block map and the kernel's per-pair grid follow it), pair_code5_tall takes the tallest
    # launchable block where the grid is slot-sized, pair_code5_flat the branch-free adds (a slot without columns
    # adds into dummy cells past the tables; 5 rows in flight, a thread's last rows as one batch). Each key off and
    # all three off must train the base cell's model; the deep base runs multi-pair levels, also on the plain grid
    # (slot-sized, without the block map and level_row_blocks' or per_pair_rows' sizing). The partword base
    # (feature_fraction 0.1375, 165 columns: 83 byte slots in 14 code words) ends every tree's rows inside a code
    # word, so the flat adds' dummy cells take the slots without columns on every tree (the other bases' rows end
    # inside a word only on some trees, at 91 or 121 byte slots; a whole padding word takes no rows), and has a byte
    # with a low column only; its fingerprint was recorded with master's build.
    night5_keys = ["pair_code5_slots", "pair_code5_tall", "pair_code5_flat"]
    night5_flips = [
        *[(key, f"{key}:off") for key in night5_keys],
        ("night5", ",".join(f"{k}:off" for k in night5_keys)),
    ]
    partword = {"feature_fraction": 0.1375}
    cell("sampledwide5/quant-partword", "sampledwide5", partword)
    for suffix, base_params, base in [
        ("-ff15", ff15, "sampledwide5/quant-ff15"),
        ("", {}, "sampledwide5/quant"),
        ("-partword", partword, "sampledwide5/quant-partword"),
    ]:
        for name, plan in night5_flips:
            cell(
                f"sampledwide5/flip-{name}{suffix}",
                "sampledwide5",
                {**base_params, "cuda_plan": f"auto,{plan}"},
                equal_to=base,
            )
    for name, plan in [
        *night5_flips,
        ("code5_slots_plain_grid", "pair_block_map:off,level_row_blocks:off,per_pair_rows:off"),
    ]:
        cell(
            f"sampledwide5/flip-{name}-deep",
            "sampledwide5",
            {**deep, "cuda_plan": f"auto,{plan}"},
            equal_to="sampledwide5/quant-deep",
        )

    # --- construction's host buffers ------------------------------------------ #
    # pin_bins, construct_staging_pool and construct_h2d_overlap only decide how construction from host arrays
    # allocates and copies; each must build the base cell's Dataset and so train its model. The lattice's 6,400
    # training rows are far below pin_bins:auto's bound, where the bins are page-locked as before: pin_bins:never
    # makes them pageable (and the per-column copies read pageable memory). The cells run 30 to a worker process, so
    # pooled staging blocks are reused across cells of other shapes. gpu_construct:off on the 4-bit profiles fills
    # the calloc'd odd-nibble buffer on the host (dense's flip above covers 8-bit bins). Profiles: dense (float64,
    # 8-bit bins), int8wide (the int8 LUT kernel, 4-bit), sampled (float64, 4-bit, a compact view).
    construct_off = "auto,pin_bins:always,construct_staging_pool:off,construct_h2d_overlap:off"
    for profile in ["dense", "int8wide", "sampled"]:
        for name, plan in [
            ("pin_bins_never", "auto,pin_bins:never"),
            ("construct_staging_pool", "auto,construct_staging_pool:off"),
            ("construct_h2d_overlap", "auto,construct_h2d_overlap:off"),
            ("construct_buffers", construct_off),
        ]:
            cell(f"{profile}/flip-{name}", profile, {"cuda_plan": plan}, equal_to=f"{profile}/quant")
    for profile in ["int8wide", "sampled", "fewbin"]:
        cell(
            f"{profile}/flip-gpu_construct",
            profile,
            {"cuda_plan": "auto,gpu_construct:off"},
            equal_to=f"{profile}/quant",
        )

    # --- the level readbacks, the tree start, the level apply's wide builds, the code-word flush ------------ #
    # readback_fused: the classic host level loop's two readbacks written into their mapped staging by the kernels
    # that produce them (the graph prefix keeps the copy); const_hess_reads: the tree start's min/max and discretize
    # take row 0's hessian (regression: constant hessian, no weights); dirty_final_slots: the next tree zeroes only
    # the histogram slots a tree ending on a complete final level dirtied (the full trees); hist_zero_quant_half: the
    # quantized half of each dirty slot; apply_inner_rows / apply_genbit_rows: the 8-rows-per-thread builds of the
    # batched apply's fused partition and gen-bit kernels (every batched level, where the occupancy API gives them
    # more rows in flight); pair_code5_epilogue: the flat (slot, marginal) flush of the code-word construct (the
    # sampledwide5 bases; off by default, so its cells turn it on). Each key flipped from its default, and all seven
    # flipped together, must train the base cell's model.
    night6_keys = [
        "readback_fused",
        "const_hess_reads",
        "dirty_final_slots",
        "hist_zero_quant_half",
        "apply_inner_rows",
        "apply_genbit_rows",
        "pair_code5_epilogue",
    ]
    night6_flip = "auto," + ",".join(f"{k}:off" for k in night6_keys[:6]) + ",pair_code5_epilogue:on"
    for profile in ["sampled", "dense"]:
        for key in night6_keys[:6]:
            cell(
                f"{profile}/flip-{key}-fulltree",
                profile,
                {**full, "cuda_plan": f"auto,{key}:off"},
                equal_to=f"{profile}/quant-fulltree",
            )
        cell(
            f"{profile}/flip-night6-fulltree",
            profile,
            {**full, "cuda_plan": night6_flip},
            equal_to=f"{profile}/quant-fulltree",
        )
    # deep trees with a binding leaf budget (no tree-completing level), fixedpoint gradients, 150 small-leaf rounds,
    # the bagged multiclass categorical cell and the quantized graph prefix
    for key in ["readback_fused", "apply_inner_rows", "apply_genbit_rows"]:
        cell(
            f"dense/flip-{key}-deep",
            "dense",
            {**deep, "cuda_plan": f"auto,{key}:off"},
            equal_to="dense/quant-deep",
        )
    cell("sampled/flip-night6-deep", "sampled", {**deep, "cuda_plan": night6_flip}, equal_to="sampled/quant-deep")
    for key in ["const_hess_reads", "hist_zero_quant_half"]:
        cell(
            f"dense/flip-{key}-fixedpoint",
            "dense",
            {"quant_mode": "fixedpoint", "cuda_plan": f"auto,{key}:off"},
            equal_to="dense/fixedpoint",
        )
    cell("tinygrad/flip-night6", "tinygrad", {"cuda_plan": night6_flip}, rounds=150, equal_to="tinygrad/quant")
    cell(
        "categorical-mc/flip-night6-bagged",
        "categorical-mc",
        {
            "quant_mode": "fixedpoint",
            "quant_bins": 16,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "num_leaves": 255,
            "max_depth": 10,
            "min_data_in_leaf": 5,
            "cuda_plan": night6_flip,
        },
        rounds=50,
        equal_to="categorical-mc/quant-bagged",
    )
    for suffix, base_params, base in [
        ("-ff15", ff15, "sampledwide5/quant-ff15"),
        ("", {}, "sampledwide5/quant"),
        ("-partword", partword, "sampledwide5/quant-partword"),
        ("-deep", deep, "sampledwide5/quant-deep"),
    ]:
        for name, plan in [("pair_code5_epilogue_on", "auto,pair_code5_epilogue:on"), ("night6", night6_flip)]:
            cell(
                f"sampledwide5/flip-{name}{suffix}",
                "sampledwide5",
                {**base_params, "cuda_plan": plan},
                equal_to=base,
            )

    # --- the integration of night6's partition keys onto master ------------------------------------------- #
    # aggregate_load_first: the batched level apply's load-first per-split block-offset prefix; split_counts_early:
    # a non-final classic level's split-info readback written by the aggregate kernel and waited on by its event
    # (the next level's search orders the histogram stream behind the partition and the finder's stream behind the
    # tree record). Each key off, and both off, must train the base cell's model: full trees, a binding leaf budget,
    # the bagged multiclass categorical cell (categorical levels: the tree record also runs SplitCategorical).
    integ_keys = ["aggregate_load_first", "split_counts_early"]
    integ_flip = "auto," + ",".join(f"{k}:off" for k in integ_keys)
    for key, plan in [*((k, f"auto,{k}:off") for k in integ_keys), ("integ", integ_flip)]:
        for profile in ["sampled", "dense"]:
            cell(
                f"{profile}/flip-{key}-fulltree",
                profile,
                {**full, "cuda_plan": plan},
                equal_to=f"{profile}/quant-fulltree",
            )
        cell(f"dense/flip-{key}-deep", "dense", {**deep, "cuda_plan": plan}, equal_to="dense/quant-deep")
    cell(
        "categorical-mc/flip-integ-bagged",
        "categorical-mc",
        {
            "quant_mode": "fixedpoint",
            "quant_bins": 16,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "num_leaves": 255,
            "max_depth": 10,
            "min_data_in_leaf": 5,
            "cuda_plan": integ_flip,
        },
        rounds=50,
        equal_to="categorical-mc/quant-bagged",
    )
    # split_counts_early on the non-quantized classic loop (one_sync off; non-quantized training takes the one-sync
    # prefix by default): the deterministic host loop over 255-bin columns, and the categorical twin (level_apply_first
    # keeps the record first for categorical splits, and the tree record runs SplitCategorical). Bases fingerprinted
    # with this branch's build; the key off, and both integration keys off, must train the base cell's model.
    for profile in ["dense", "categorical"]:
        base = f"{profile}/nonquant-det-twosync"
        cell(base, profile, {"quant_mode": "none", "cuda_plan": "auto,device_tree:off,one_sync:off"}, rounds=50)
        for name, plan in [
            ("split_counts_early", "split_counts_early:off"),
            ("integ", integ_flip.removeprefix("auto,")),
        ]:
            cell(
                f"{profile}/flip-{name}-nonquant-twosync",
                profile,
                {"quant_mode": "none", "cuda_plan": f"auto,device_tree:off,one_sync:off,{plan}"},
                rounds=50,
                equal_to=base,
            )

    # --- the int8 tile binning's knobs (construct_tile_rows, construct_tile_threads) ------------------------- #
    # Values other than the default 64 rows take the run-time-rows instantiation, and each block size its own build.
    # int8wide (6000 x 400: three full 128-column tiles and a 16-column one) at 2 rows and 32 threads; int8odd (6002
    # x 131, 4801 training rows: an odd row count, a 3-column partial tile) at 384 rows and 1024 threads (the 48 KB
    # tile) and at 6 rows and 64 threads, and with the tile kernel off. Each must bin the base cell's bytes and so
    # train its model; int8odd's base fingerprint was recorded with this branch's build.
    cell("int8odd/quant", "int8odd")
    for profile, name, plan in [
        ("int8wide", "construct_tile_2_32", "construct_tile_rows:2,construct_tile_threads:32"),
        ("int8odd", "construct_tile_384_1024", "construct_tile_rows:384,construct_tile_threads:1024"),
        ("int8odd", "construct_tile_6_64", "construct_tile_rows:6,construct_tile_threads:64"),
        ("int8odd", "construct_tile_bin", "construct_tile_bin:off"),
    ]:
        cell(f"{profile}/flip-{name}", profile, {"cuda_plan": f"auto,{plan}"}, equal_to=f"{profile}/quant")

    # --- the 6-bit pair-code view (pair_code_wide) -------------------------------------------------------- #
    # sampledwide's 6 x 6 = 36-cell joint tables do not fit the 5-bit code. At feature_fraction 0.3 a row is ~180
    # bytes, and the 6-bit code view (five codes per word, 40 words) is smaller than the nibble view, so the fused fill
    # writes it and the code-word construct reads it (FALCATA_DEBUG=diag: every tree; at the ff15 and 0.2 rows the
    # 6-bit row's 32-byte padding makes it larger and the nibble view stays). sampledwide-mixbins' 3-6 value columns
    # give it trees of either width. The base cells' fingerprints were recorded with master's build (a65d34e8), which
    # reads the nibble view. pair_code_wide off (the nibble view again), the branchy adds (pair_code5_flat off) and
    # the per-slot flush (pair_code5_epilogue on) over the 6-bit view must train the base cell's model; deep trees run
    # it on multi-pair levels.
    ff30 = {"feature_fraction": 0.3}
    code6_flips = [
        ("pair_code_wide", "pair_code_wide:off"),
        ("code6_branchy", "pair_code5_flat:off"),
        ("code6_epilogue", "pair_code5_epilogue:on"),
    ]
    for base_profile, suffix, base_params in [
        ("sampledwide", "ff30", ff30),
        ("sampledwide-mixbins", "ff30", ff30),
        ("sampledwide", "ff30-deep", {**ff30, **deep}),
    ]:
        cell(f"{base_profile}/quant-{suffix}", base_profile, base_params)
        for name, plan in code6_flips:
            cell(
                f"{base_profile}/flip-{name}-{suffix}",
                base_profile,
                {**base_params, "cuda_plan": f"auto,{plan}"},
                equal_to=f"{base_profile}/quant-{suffix}",
            )

    # --- the open1 lineage's nine keys (PR #99) ------------------------------------------------------------ #
    # Each key off must train the base cell's model. Bases added here (quant-deep2047, flipbase-pair_hist,
    # flipbase-fix_subtract_fused, sigmoid03, quant-ova, the graph<k> and weighted profiles, the bagged bases) had
    # their fingerprints recorded with master's build (d919b841).
    #
    # dense_level_row_blocks / dense_block_map: the quantized per-column dense batched construct. dense's 255-bin deep
    # trees; 2047 leaves at depth 12 with 1-row leaves give levels of more than 256 pairs (the chunked block map, one
    # mapped grid per 256 pairs); fewbin with pair_hist off puts 4-bit data on the dense kernel.
    dense_keys = ["dense_level_row_blocks", "dense_block_map"]
    dense_flips = [*((k, f"{k}:off") for k in dense_keys), ("dense_both", ",".join(f"{k}:off" for k in dense_keys))]
    for name, plan in dense_flips:
        cell(f"dense/flip-{name}-deep", "dense", {**deep, "cuda_plan": f"auto,{plan}"}, equal_to="dense/quant-deep")
    deep2047 = {"num_leaves": 2047, "max_depth": 12, "min_data_in_leaf": 1}
    cell("dense/quant-deep2047", "dense", deep2047)
    for name, plan in dense_flips:
        cell(
            f"dense/flip-{name}-deep2047",
            "dense",
            {**deep2047, "cuda_plan": f"auto,{plan}"},
            equal_to="dense/quant-deep2047",
        )
    cell("fewbin/flipbase-pair_hist", "fewbin", {"cuda_plan": "auto,pair_hist:off"})
    for name, plan in dense_flips:
        cell(
            f"fewbin/flip-{name}-nopair",
            "fewbin",
            {"cuda_plan": f"auto,pair_hist:off,{plan}"},
            equal_to="fewbin/flipbase-pair_hist",
        )
    # fix_subtract_masked: need-fix features (missing-mfb0's synthesised bin 0, efb's bundles), the classic flow's
    # per-leaf launch (hybrid off), 32-bit leaf bins (bigrow, sub_case 3), and the masked kernel's bin_used /
    # feature_used skip with the fused fix + subtract off (sampled's feature-sampled trees)
    for base, profile, base_params, rounds in [
        ("missing-mfb0/quant", "missing-mfb0", {}, 25),
        ("efb/quant", "efb", {}, 25),
        ("missing-mfb0/quant-classic", "missing-mfb0", {"cuda_plan": "auto,hybrid:off"}, 25),
        ("bigrow/quant64", "bigrow", {}, 40),
        ("sampled/quant", "sampled", {}, 25),
        ("dense/quant-fulltree", "dense", full, 25),
    ]:
        plan = base_params.get("cuda_plan", "auto")
        suffix = base.split("/", 1)[1].removeprefix("quant").removeprefix("-")
        cell(
            f"{profile}/flip-fix_subtract_masked" + (f"-{suffix}" if suffix else ""),
            profile,
            {**base_params, "cuda_plan": f"{plan},fix_subtract_masked:off"},
            rounds=rounds,
            equal_to=base,
        )
    cell("sampled/flipbase-fix_subtract_fused", "sampled", {"cuda_plan": "auto,fix_subtract_fused:off"})
    cell(
        "sampled/flip-fix_subtract_masked-nofused",
        "sampled",
        {"cuda_plan": "auto,fix_subtract_fused:off,fix_subtract_masked:off"},
        equal_to="sampled/flipbase-fix_subtract_fused",
    )
    # gradient_fp64_lean: the binary logloss gradients, plain, deterministic non-quantized (double-slot), with label
    # weights (is_unbalance, scale_pos_weight), mild sample weights (uniform 0.5-2: wide ones make master's model
    # nondeterministic, see the init score), another sigmoid, fixedpoint, and one-vs-all multiclass (the binary
    # kernel per class)
    sig03 = {"sigmoid": 0.3}
    cell("imbalanced/sigmoid03", "imbalanced", sig03)
    cell("imbalanced-w/quant", "imbalanced-w")
    ova = {"objective": "multiclassova", "metric": "None"}
    cell("graph/quant-ova", "graph", ova)
    for base, profile, base_params, rounds in [
        ("imbalanced/quant", "imbalanced", {}, 25),
        ("imbalanced/nonquant-det", "imbalanced", {"quant_mode": "none", "cuda_plan": "auto,device_tree:off"}, 50),
        ("imbalanced/unbalance", "imbalanced", {"is_unbalance": True}, 25),
        ("imbalanced/posweight", "imbalanced", {"scale_pos_weight": 10.0}, 25),
        ("imbalanced/fixedpoint", "imbalanced", {"quant_mode": "fixedpoint"}, 25),
        ("imbalanced/sigmoid03", "imbalanced", sig03, 25),
        ("imbalanced-w/quant", "imbalanced-w", {}, 25),
        ("graph/quant-ova", "graph", ova, 25),
    ]:
        plan = base_params.get("cuda_plan", "auto")
        cell(
            f"{profile}/flip-gradient_fp64_lean-{base.split('/', 1)[1]}",
            profile,
            {**base_params, "cuda_plan": f"{plan},gradient_fp64_lean:off"},
            rounds=rounds,
            equal_to=base,
        )
    # softmax_registers: class capacities 8 (graph's 5 classes, sparse-mc's 6, categorical-mc's 4 bagged), 16 (12
    # classes), 32 (24 classes) and the scratch fallback (40 classes), and a weighted multiclass
    for k in (12, 24, 40):
        cell(f"graph{k}/quant", f"graph{k}")
    cell("graph-w/quant", "graph-w")
    mc_bagged = {
        "quant_mode": "fixedpoint",
        "quant_bins": 16,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "num_leaves": 255,
        "max_depth": 10,
        "min_data_in_leaf": 5,
    }
    sparse_deep = {
        "quant_mode": "fixedpoint",
        "quant_bins": 16,
        "num_leaves": 63,
        "max_depth": 10,
        "min_data_in_leaf": 5,
    }
    for base, profile, base_params, rounds in [
        ("graph/quant", "graph", {}, 25),
        ("sparse-mc/quant-deep", "sparse-mc", sparse_deep, 80),
        ("categorical-mc/quant-bagged", "categorical-mc", mc_bagged, 50),
        ("graph12/quant", "graph12", {}, 25),
        ("graph24/quant", "graph24", {}, 25),
        ("graph40/quant", "graph40", {}, 25),
        ("graph-w/quant", "graph-w", {}, 25),
    ]:
        cell(
            f"{profile}/flip-softmax_registers-{base.split('/', 1)[1]}",
            profile,
            {**base_params, "cuda_plan": "auto,softmax_registers:off"},
            rounds=rounds,
            equal_to=base,
        )
    # sync_used_narrow: the full tree (readback_fused's mirror engages), dense, missing, and sampled (a no-op control:
    # feature-sampled trees already take the used-task kernel)
    for base, profile, base_params in [
        ("dense/quant-fulltree", "dense", full),
        ("dense/quant", "dense", {}),
        ("missing/quant", "missing", {}),
        ("sampled/quant", "sampled", {}),
    ]:
        suffix = base.split("/", 1)[1].removeprefix("quant").removeprefix("-")
        cell(
            f"{profile}/flip-sync_used_narrow" + (f"-{suffix}" if suffix else ""),
            profile,
            {**base_params, "cuda_plan": "auto,sync_used_narrow:off"},
            equal_to=base,
        )
    # bagged_code_view: sampledwide5 bagged takes the code view on every tree (the diag cell asserts it), with a bag
    # drawn every round and one reused for five; sampled's columns decline it on every tree, where the bagged tree
    # takes the plain tiled fill (no fused fill for a view it does not write); sampledwide's trees do either (the
    # 6-bit view on some 120-byte rows, not on 121-byte ones). Each key off, and all three bagging keys off, must
    # train the base cell's model.
    bag_keys = ["bagged_code_view", "oob_node_table", "bagging_fused_scan"]
    bag_off = ",".join(f"{k}:off" for k in bag_keys)
    bag = {"bagging_fraction": 0.7, "bagging_freq": 1}
    plain = "bagged tree without the code view: plain tiled fill"
    for name, profile, bag_params, expect in [
        ("quant-bagged", "sampledwide5", {"bagging_fraction": 0.6, "bagging_freq": 1}, ["pair_code5 on"]),
        ("quant-bagged5", "sampledwide5", {"bagging_fraction": 0.6, "bagging_freq": 5}, ["pair_code5 on"]),
        ("quant-bagged", "sampled", bag, [plain]),
        ("quant-bagged", "sampledwide", {"bagging_fraction": 0.6, "bagging_freq": 1}, ["pair_code5 on", plain]),
    ]:
        base = f"{profile}/{name}"
        cell(base, profile, bag_params)
        for key, plan in [("bagged_code_view", "bagged_code_view:off"), ("bagging", bag_off)]:
            cell(f"{profile}/flip-{key}-{name}", profile, {**bag_params, "cuda_plan": f"auto,{plan}"}, equal_to=base)
        cell(
            f"{profile}/codeview-{name}",
            profile,
            {**bag_params, "verbose": 1},
            equal_to=base,
            diag_set=(r"compact fill: (pair_code5 on|bagged tree without the code view: plain tiled fill)", expect),
        )
    # oob_node_table: the out-of-bag pass on the per-column view (dense), the packed view (sampled), NaN default-left
    # (missing), bundled columns (efb), synthesised bin 0 (missing-mfb0); categorical-mc and vector-leaf bagged are
    # controls (they keep the re-upload path), dart re-scores after ToHost
    vl_bagged = {"quant_mode": "fixedpoint", "quant_bins": 16, **bag}
    for profile in ["dense", "missing", "efb", "missing-mfb0"]:
        cell(f"{profile}/quant-bagged", profile, bag)
    cell("vector-leaf/fixedpoint-bagged", "vector-leaf", vl_bagged, rounds=30)
    dart = {"boosting": "dart", **bag}
    cell("dense/quant-dart-bagged", "dense", dart)
    for base, profile, base_params, rounds in [
        *((f"{p}/quant-bagged", p, bag, 25) for p in ["dense", "sampled", "missing", "efb", "missing-mfb0"]),
        ("categorical-mc/quant-bagged", "categorical-mc", mc_bagged, 50),
        ("vector-leaf/fixedpoint-bagged", "vector-leaf", vl_bagged, 30),
        ("dense/quant-dart-bagged", "dense", dart, 25),
    ]:
        cell(
            f"{profile}/flip-oob_node_table-{base.split('/', 1)[1]}",
            profile,
            {**base_params, "cuda_plan": "auto,oob_node_table:off"},
            rounds=rounds,
            equal_to=base,
        )
    # bagging_fused_scan: small and large bags, a bag reused for three rounds, and bigrow's 1.6M training rows: 1563
    # blocks of 1024, more than one block of the base kernel's scan, so its carry runs across passes
    for name, bag_params in [
        ("quant-bagged", bag),
        ("quant-bag05", {"bagging_fraction": 0.05, "bagging_freq": 1}),
        ("quant-bag95", {"bagging_fraction": 0.95, "bagging_freq": 1}),
        ("quant-bagfreq3", {"bagging_fraction": 0.7, "bagging_freq": 3}),
    ]:
        if name != "quant-bagged":
            cell(f"dense/{name}", "dense", bag_params)
        cell(
            f"dense/flip-bagging_fused_scan-{name}",
            "dense",
            {**bag_params, "cuda_plan": "auto,bagging_fused_scan:off"},
            equal_to=f"dense/{name}",
        )
    bigbag = {"bagging_fraction": 0.5, "bagging_freq": 1}
    cell("bigrow/quant64-bagged", "bigrow", bigbag, rounds=20)
    cell(
        "bigrow/flip-bagging_fused_scan-bagged",
        "bigrow",
        {**bigbag, "cuda_plan": "auto,bagging_fused_scan:off"},
        rounds=20,
        equal_to="bigrow/quant64-bagged",
    )

    # random forest (boosting=rf): the device objective (rf_device_objective: constant gradients computed once on the
    # device, no host score mirror, the CUDA metrics) and the deferred train score (rf_lazy_train_score) against the
    # host objective path, on binary stochastic, 5-class fixedpoint and L1 (leaf renewal against the init score
    # column) bagged forests. The bases are fingerprinted on the build before the keys (the host objective path).
    rf_bag = {
        "boosting": "rf",
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "num_leaves": 127,
        "max_depth": 8,
        "min_data_in_leaf": 5,
    }
    for base, profile, base_params in [
        ("imbalanced/rf-bagged", "imbalanced", rf_bag),
        ("graph/rf-fixedpoint-bagged", "graph", {**rf_bag, "quant_mode": "fixedpoint"}),
        ("dense/rf-l1-bagged", "dense", {**rf_bag, "objective": "regression_l1"}),
    ]:
        cell(base, profile, base_params)
        for key in ["rf_device_objective", "rf_lazy_train_score"]:
            cell(
                f"{profile}/flip-{key}-{base.split('/', 1)[1]}",
                profile,
                {**base_params, "cuda_plan": f"auto,{key}:off"},
                equal_to=base,
            )

    # --- route goldens (FULL_DEVICE_PLAN P0) --------------------------------------------------------------------- #
    # One cell per refusal route of the CUDA learner on master 5bec2574 (runs/device_default/FULL_DEVICE_PLAN.md
    # section 1, rows rNN; a/b/c are variants): each configuration the complete device tree refuses today, recorded on
    # the route that serves it now (the device_levels prefix + host two-sync loop, the selective device prefix + host
    # selective loop, the non-quantized L1.5 graph (on the device tree since P3) / one-sync loops, forest slots, the
    # per-pair search, the classic per-split loop). The full-device-controller PRs port these routes onto the device controller and delete host
    # paths; each must keep every md5 below unless it is marked as a model change (CEGB: level-wise on the device, a
    # documented change; re-baseline with --update route/r18-cegb). Not here: multi-GPU (rows 10-11, the vast.ai
    # 2-GPU check), low shared memory (row 16, not reproducible on this GPU), the empty task list (row 23, a unit
    # test in P6), debug hooks and routing keys (rows 25-26, deleted). Bagging by query (r27c) needs a ranking
    # profile; the lattice has none.
    route_ic = {"interaction_constraints": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], list(range(10, 20))]}
    # monotone constraints are refused with quantized gradients on CUDA (Fatal): non-quantized, the classic loop
    route_mono = {"monotone_constraints": [1, -1, *([0] * 17), 1], "quant_mode": "none"}
    route_rf = {"boosting": "rf", "bagging_fraction": 0.632, "bagging_freq": 1}
    for cid, profile, params, rounds, extra in [
        ("r01-depth13", "dense", {"num_leaves": 8191, "max_depth": 13, "min_data_in_leaf": 1}, 5, {}),
        # vramfree=1 caps every free-memory check, the histogram constructor's (view regime, column-major fill
        # reserves) as well as the device tree's pair-chunk plan (its level searches run in pair chunks)
        (
            "r02-memgate",
            "dense",
            {"num_leaves": 1023, "max_depth": 10, "min_data_in_leaf": 2},
            5,
            {"debug": ["vramfree=1"]},
        ),
        ("r03a-budget", "dense", {"num_leaves": 151, "max_depth": 10}, 25, {}),
        ("r03b-unbounded", "dense", {"num_leaves": 255, "max_depth": -1, "min_data_in_leaf": 5}, 25, {}),
        ("r04a-nq-l15", "dense", {"quant_mode": "none"}, 25, {}),
        (
            "r04b-nq-onesync",
            "dense",
            {"quant_mode": "none", "num_leaves": 4095, "max_depth": 12, "min_data_in_leaf": 1},
            5,
            {},
        ),
        ("r05-nq-selective", "dense", {"quant_mode": "none", "num_leaves": 31, "max_depth": -1}, 25, {}),
        ("r06a-cat", "categorical", {"num_leaves": 255, "max_depth": 8}, 25, {}),
        ("r06b-cat-mc", "categorical-mc", {}, 25, {}),
        ("r06c-cat-unbounded", "categorical", {"num_leaves": 63, "max_depth": -1}, 25, {}),
        ("r07a-bynode-sel", "dense", {"feature_fraction_bynode": 0.5, "num_leaves": 31, "max_depth": -1}, 25, {}),
        (
            "r07b-bynode-d13",
            "dense",
            {"feature_fraction_bynode": 0.5, "num_leaves": 8191, "max_depth": 13, "min_data_in_leaf": 1},
            5,
            {},
        ),
        ("r08a-ic", "dense", route_ic, 25, {}),
        ("r08b-ic-bynode", "dense", {**route_ic, "feature_fraction_bynode": 0.6}, 25, {}),
        ("r09a-rf-g4", "imbalanced", {**route_rf, "cuda_plan": "auto,forest_group:4"}, 25, {}),
        ("r09b-rf-mc", "rare-mc-b", route_rf, 10, {}),
        ("r12-vector", "vector-leaf", {"num_leaves": 31, "max_depth": -1}, 25, {}),
        ("r13a-l1", "dense", {"lambda_l1": 1.0}, 25, {}),
        ("r13b-smooth", "dense", {"path_smooth": 5.0}, 25, {}),
        ("r13c-extra", "dense", {"extra_trees": True}, 25, {}),
        ("r13d-catrand", "categorical", {"cat_random_search": 4}, 25, {}),
        # sparse row data: quantized sparse-mc trains on the dense row-wise bin (dense_rowdata) and takes the complete
        # device tree; non-quantized it keeps the sparse (CSR) row data, which the batched level construct refuses
        ("r14a-sparse", "sparse-mc", {"num_leaves": 31, "max_depth": 5, "quant_mode": "none"}, 25, {}),
        # r14b and r15 pin the non-quantized route only: on 5bec2574 the non-quantized flows print no histogram or
        # finder gate line, so diag confirms neither the 16-bit compact read nor the global-memory finder (job 7042)
        ("r14b-compact16", "dense", {"max_bin": 511, "feature_fraction": 0.5, "quant_mode": "none"}, 25, {}),
        ("r15-widebin", "missing-mfb0-wide", {"quant_mode": "none", "max_bin": 511}, 25, {}),
        ("r17-monotone", "dense", route_mono, 25, {}),
        ("r17b-monotone-sel", "dense", {**route_mono, "num_leaves": 31, "max_depth": -1}, 25, {}),
        # planned model change: CEGB becomes level-wise on the device controller (Felix 2026-10-23); that PR
        # re-baselines this cell with --update route/r18-cegb
        (
            "r18-cegb",
            "dense",
            {"cegb_penalty_split": 0.05, "cegb_penalty_feature_coupled": [1.0] * 20, "cegb_tradeoff": 0.5},
            25,
            {},
        ),
        (
            "r19-forced",
            "dense",
            {},
            25,
            {"forced_splits": {"feature": 0, "threshold": 0.0, "left": {"feature": 1, "threshold": 0.0}}},
        ),
        # the linear leaf fit is not run-to-run deterministic on 5bec2574 (job 7021: leaf_const / leaf_coeff of every
        # tree differ between runs, at num_threads 8 and 1; structure and leaf values repeat): the md5 pins the tree
        # text without those lines (and tree_sizes, which follows their lengths)
        (
            "r20-linear-sel",
            "dense",
            {"linear_tree": True, "num_leaves": 31, "max_depth": -1},
            25,
            {"fingerprint_strip": ["leaf_const=", "leaf_coeff=", "tree_sizes="]},
        ),
        ("r21-stump", "dense", {"num_leaves": 2, "max_depth": -1}, 25, {}),
        # the device-tree root tail needs the batched level root to find no valid split while the classic root search
        # finds a positive-gain one; no known shape reaches it on 5bec2574 (host_tail=0 on every tree at
        # feature_fraction 0.05 / 0.03 / 0.02 / 0.01 and seeds 42, 1, 2, 3, job 7007: the compact-view feature-mask fix
        # made both root searches agree). The cell pins the plan's sampled-compact complete tree.
        ("r22-roottail", "sampledwide", {"feature_fraction": 0.05, "num_leaves": 63, "max_depth": 6}, 25, {}),
        # sampledwide-mixbins: each tree's column sample gives another joint-table size (part of the graph key), so the
        # selective prefix misses its graph cache every tree: diag "device_level_graph: 9 misses, 0 hits: graphs off",
        # then "host fallback (changing graph layouts)" (job 7042; `sampled` hits the cache every tree)
        ("r24-churn", "sampledwide-mixbins", {"num_leaves": 31, "max_depth": -1}, 40, {}),
        ("r27a-goss", "dense", {"data_sample_strategy": "goss"}, 25, {}),
        ("r27b-renew", "dense", {"objective": "l1", "quant_train_renew_leaf": True}, 25, {}),
    ]:
        cell(f"route/{cid}", profile, params, rounds=rounds, **extra)

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
import hashlib, json, os, sys, tempfile, time
import numpy as np
import falcata as lgb

def _split(X, y, frac=0.2):
    n = len(y); k = int(n * (1 - frac))
    return X[:k], y[:k], X[k:], y[k:]

def build_profile(name):
    # stable seed: builtin hash() is per-process randomized (PYTHONHASHSEED)
    rng = np.random.default_rng(int(hashlib.md5(name.encode()).hexdigest()[:8], 16))
    sample_w = None  # sample weights (weighted profiles only)
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
    elif name == "sampledwide":
        # ~120 sampled byte slots over several feature partitions: the multi-partition tiled compact fill
        m = 1200
        X = rng.integers(0, 6, size=(n, m)).astype(np.float64)
        y = X[:, :40] @ rng.standard_normal(40) + rng.standard_normal(n)
        base.update({"max_bin": 15, "feature_fraction": 0.2})
    elif name == "sampled-oddrows":
        # sampled with 8002 rows, 6401 of them trained: a column of the column-major store is 3201 bytes, an odd pitch,
        # so the fused root fill's word staging runs only with colmajor_align (6400 rows, 3200 bytes, are
        # sector-aligned already)
        n, m = 8002, 600
        X = rng.integers(0, 6, size=(n, m)).astype(np.float64)
        y = X @ rng.standard_normal(m) + rng.standard_normal(n)
        base.update({"max_bin": 15, "feature_fraction": 0.2})
    elif name == "sampledwide5":
        # sampledwide with 5 values per feature: every byte's joint table is 5 x 5 = 25 cells, so the fused fill
        # writes the 5-bit pair-code view (pair_code5), which sampledwide's 6 x 6 = 36-cell tables decline
        m = 1200
        X = rng.integers(0, 5, size=(n, m)).astype(np.float64)
        y = X[:, :40] @ rng.standard_normal(40) + rng.standard_normal(n)
        base.update({"max_bin": 15, "feature_fraction": 0.2})
    elif name == "sampledwide-mixbins":
        # sampledwide with 3 to 6 distinct values per feature: every tree's column sample gives the whole-row
        # pair-joint construct another joint-table size (sampledwide's equal features always give the same one)
        m = 1200
        X = np.floor(rng.random((n, m)) * rng.integers(3, 7, size=m)).astype(np.float64)
        y = X[:, :40] @ rng.standard_normal(40) + rng.standard_normal(n)
        base.update({"max_bin": 15, "feature_fraction": 0.2})
    elif name == "mixwidth":
        # one dataset with both tiers of the warp split finder: 2-bin indicators and 6-value columns (at most 8 scan
        # positions: the few-position tier, 4-lane groups, eight per warp), columns of 40, 70, 100, 130, 160 and 200
        # values (the wide tier, a warp per task: warp_find_mid_ppl's 2, 4, 4, 8, 8 and 8 positions per lane) and
        # continuous columns (255 bins, 8 per lane), NaN in 15% of the rows of four wide columns (the NaN scans). Each
        # tree's used tasks go out in one launch, narrow first, then wide; the wide tasks run find_prune_fp32,
        # find_prune_sure, find_compact_survivors and find_select_int.
        cols = [rng.integers(0, 2, size=n) for _ in range(10)]
        cols += [rng.integers(0, 6, size=n) for _ in range(6)]
        cols += [rng.integers(0, k, size=n) for k in (40, 70, 100, 130, 160, 200)]
        cols += [rng.standard_normal(n) for _ in range(4)]
        X = np.column_stack(cols).astype(np.float64)
        m = X.shape[1]
        Z = (X - X.mean(axis=0)) / X.std(axis=0)
        y = Z @ rng.standard_normal(m) + 0.5 * np.sin(2 * Z[:, 18]) + 0.3 * rng.standard_normal(n)
        for j in (17, 19, 21, 23):
            X[rng.random(n) < 0.15, j] = np.nan
    elif name == "boundwidth":
        # columns at the warp finder's tier boundaries: k values as 0..k-1, 1..k and 1..k with NaN in 15% of the rows,
        # for k = 8, 9, 64, 65, 128, 129 (8-11, 64-67, 128-131 bins: scan widths on both sides of 8, 64 and 128), and
        # two continuous columns (255 bins, one with NaN)
        cols, nan_cols = [], []
        for k in (8, 9, 64, 65, 128, 129):
            cols.append(rng.integers(0, k, size=n).astype(np.float64))
            cols.append(rng.integers(1, k + 1, size=n).astype(np.float64))
            nan_cols.append(len(cols))
            cols.append(rng.integers(1, k + 1, size=n).astype(np.float64))
        cols.append(rng.standard_normal(n))
        cols.append(rng.standard_normal(n))
        nan_cols.append(len(cols) - 1)
        X = np.column_stack(cols)
        m = X.shape[1]
        Z = (X - X.mean(axis=0)) / X.std(axis=0)
        y = Z @ rng.standard_normal(m) + 0.5 * np.sin(2 * Z[:, 4]) + 0.3 * rng.standard_normal(n)
        for j in nan_cols:
            X[rng.random(n) < 0.15, j] = np.nan
    elif name == "narrowplus":
        # 19 columns each of 2, 4 and 6 values (1..v: no zeros, so the row data stays dense) and one 33-value column;
        # at feature_fraction 0.2 most trees draw only the narrow columns (the warp finder's launch without wide
        # units, over a mixed dataset's narrow-first order list), the rest the 33-value one too
        cols = [rng.integers(1, v + 1, size=n) for v in (2, 4, 6) for _ in range(19)]
        cols.append(rng.integers(1, 34, size=n))
        X = np.column_stack(cols).astype(np.float64)
        m = X.shape[1]
        Z = (X - X.mean(axis=0)) / X.std(axis=0)
        y = Z @ rng.standard_normal(m) + 1.5 * Z[:, -1] + 0.3 * rng.standard_normal(n)
        base["feature_fraction"] = 0.2
    elif name in ("graph", "graph12", "graph24", "graph40", "graph-w"):
        # graph<k>: the same shape with k classes (the register softmax's class capacities 16 and 32, and the
        # scratch fallback past 32); graph-w: 5 classes with uniform(0.5, 2) sample weights
        k = int(name[5:]) if name[5:].isdigit() else 5
        m = 20
        X = rng.standard_normal((n, m))
        y = (X @ rng.standard_normal((m, k))).argmax(axis=1).astype(np.float64)
        base.update({"objective": "multiclass", "num_class": k, "metric": "multi_logloss"})
        if name == "graph-w":
            sample_w = rng.uniform(0.5, 2.0, size=n)
    elif name == "int8wide":
        n, m = 6000, 400
        X = rng.integers(-6, 7, size=(n, m)).astype(np.int8)
        y = X.astype(np.float64) @ rng.standard_normal(m) + rng.standard_normal(n)
        base["max_bin"] = 15
    elif name == "int8odd":
        # int8wide's values at 6002 x 131: 4801 training rows (an odd count: a half row pair) and a partial tile of
        # 3 columns after one full 128-column tile
        n, m = 6002, 131
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
    elif name in ("imbalanced", "imbalanced-w"):
        # imbalanced-w: uniform(0.5, 2) sample weights (mild: their sums are not exact in double, but they are far
        # from the wide weights that make the weighted init score order-dependent)
        n, m = 12000, 20
        X = rng.standard_normal((n, m))
        score = X @ rng.standard_normal(m)
        y = (score > np.quantile(score, 0.99)).astype(np.float64)
        base.update({"objective": "binary", "metric": "auc"})
        if name == "imbalanced-w":
            sample_w = rng.uniform(0.5, 2.0, size=n)
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
    elif name == "missing-mfb0-dense":
        # missing-mfb0 at a density that keeps the row data dense: zero (bin 0, the default and unstored bin)
        # about 22% of the values and NaN 10%, under the 25% multi-val sparse threshold
        m = 20
        X = rng.choice([0.0, 1.0, 2.0, 3.0, 4.0], size=(n, m),
                       p=[0.24, 0.19, 0.19, 0.19, 0.19]).astype(np.float64)
        w = rng.standard_normal(m)
        y = (X - 2.0) @ w + 0.3 * rng.standard_normal(n)
        X[rng.random((n, m)) < 0.1] = np.nan
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
    elif name == "vector-leaf":
        # Four targets share their strongest drivers, which makes a shared
        # structure appropriate, but each has a distinct secondary feature.
        # Their 3^t scale spread makes the per-target quantization scales
        # observable: one scale shared by all targets would collapse the
        # smallest target. Five-valued features exercise the production-like
        # low-cardinality regime where vector-leaf is intended to be useful.
        n, m, num_targets = 8000, 20, 4
        X = rng.integers(0, 5, size=(n, m)).astype(np.float64)
        centered = X - 2.0
        shared = 1.7 * centered[:, 0] - 1.2 * centered[:, 3]
        columns = []
        for target in range(num_targets):
            signal = shared + 0.4 * centered[:, target + 5]
            noise = 0.05 * rng.standard_normal(n)
            columns.append((signal + noise) * (3.0**target))
        y = np.column_stack(columns)
        base.update({"objective": "multi_regression",
                     "num_class": num_targets, "tree_mode": "vector_leaf",
                     "boost_from_average": False,
                     "max_bin": 5, "num_leaves": 31, "max_depth": 10,
                     "min_data_in_leaf": 20})
    elif name in ("rare-mc-b", "rare-mc-c"):
        # covtype's class frequencies, its rarest class (0.47%) living in a
        # corner of feature space it shares with the common classes: six
        # common classes from a linear 12-feature score (six of the features
        # binary) with Gumbel label noise and per-class biases solved to the
        # frequencies, then 85% of the rows past feature 0's 99.5th percentile
        # relabelled to the rare class
        n, m = 40000, 12
        common_priors = np.array([0.365, 0.488, 0.0615, 0.0163, 0.030, 0.035])
        X = rng.standard_normal((n, m))
        X[:, -6:] = (rng.random((n, 6)) < 0.2).astype(np.float64)
        z = 3.0 * (X @ rng.standard_normal((m, 6))) / np.sqrt(m) + 0.5 * rng.gumbel(size=(n, 6))
        bias = np.zeros(6)
        for _ in range(200):
            freq = np.bincount((z + bias).argmax(axis=1), minlength=6) / n
            bias += 0.5 * (np.log(common_priors) - np.log(np.maximum(freq, 1e-6)))
        y = np.array([0, 1, 2, 4, 5, 6])[(z + bias).argmax(axis=1)]
        corner = X[:, 0] > np.quantile(X[:, 0], 0.995)
        y[corner & (rng.random(n) < 0.85)] = 3
        base.update({"objective": "multiclass", "num_class": 7, "metric": "multi_logloss"})
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
    return X, np.asarray(y, dtype=np.float64), base, sample_w

def eval_metric(params, pred, y_test):
    if params.get("objective") == "multi_regression":
        target_variance = np.var(y_test, axis=0)
        nrmse = np.sqrt(np.mean((pred - y_test) ** 2, axis=0) / target_variance)
        # A mean can hide a collapsed small-magnitude target. The worst target
        # is the conservative single-number quality signal the lattice needs.
        return float(np.max(nrmse)), "max_nrmse", False
    if params.get("objective") == "binary":
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_test, pred)), "auc", True
    if params.get("objective") == "multiclassova":
        # one-vs-all scores do not sum to 1: the mean negative log of the true class's score
        p_true = pred[np.arange(len(y_test)), y_test.astype(int)]
        return float(-np.mean(np.log(np.clip(p_true, 1e-15, 1.0)))), "ova_logloss", False
    if params.get("objective") == "multiclass":
        from sklearn.metrics import log_loss
        return float(log_loss(y_test, pred, labels=list(range(params["num_class"])))), "mlogloss", False
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    return rmse, "rmse", False

def run_cell(cell):
    X, y, params, w = build_profile(cell["profile"])
    params.update(cell["params"])
    X_tr, y_tr, X_te, y_te = _split(X, y)
    w_tr = None if w is None else w[:len(y_tr)]
    with tempfile.TemporaryDirectory() as tmp:
        if cell.get("forced_splits") is not None:
            # the path lands in the parameters block only, which the fingerprint excludes
            params["forcedsplits_filename"] = os.path.join(tmp, "forced_splits.json")
            with open(params["forcedsplits_filename"], "w") as f:
                json.dump(cell["forced_splits"], f)
        t0 = time.monotonic()
        ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, params=params)
        ds.construct()
        t_construct = time.monotonic() - t0
        t1 = time.monotonic()
        bst = lgb.train(params, ds, num_boost_round=cell["rounds"])
        t_train = time.monotonic() - t1
    model_str = bst.model_to_string()
    # fingerprint ONLY the trees: the trailing "parameters:" dump moves whenever
    # a config param is added/renamed (it did in the 2026-07 planner refactor
    # with zero behavior change) -- tree bytes are the actual behavior signature
    trees_str = model_str.split("\nparameters:")[0]
    if cell.get("fingerprint_strip"):
        strip = tuple(cell["fingerprint_strip"])
        trees_str = "\n".join(line for line in trees_str.splitlines() if not line.startswith(strip))
    md5 = hashlib.md5(trees_str.encode()).hexdigest()
    # validity: round-trip + finite predictions + tree count
    is_vector_leaf = str(params.get("tree_mode", "scalar")).strip().lower() == "vector_leaf"
    expected_trees = cell["rounds"] if is_vector_leaf else cell["rounds"] * params.get("num_class", 1)
    assert bst.num_trees() == expected_trees, f"tree count {bst.num_trees()} != {expected_trees}"
    pred = bst.predict(X_te)
    assert np.all(np.isfinite(pred)), "non-finite predictions"
    if y_te.ndim == 2:
        assert pred.shape == y_te.shape, (
            f"multi-target prediction shape {pred.shape} != label shape {y_te.shape}"
        )
    assert np.all(np.std(pred, axis=0) > 0.0), "constant predictions (garbage model)"
    reloaded = lgb.Booster(model_str=model_str)
    pred2 = reloaded.predict(X_te)
    assert np.array_equal(pred, pred2), "reloaded model predicts differently"
    if cell.get("leaf_bound") is not None:
        leaf_abs_max = max(
            abs(float(v))
            for line in model_str.split("\nparameters:")[0].splitlines()
            if line.startswith("leaf_value=")
            for v in line[len("leaf_value="):].split()
        )
        assert leaf_abs_max <= cell["leaf_bound"], (
            f"max |leaf value| {leaf_abs_max:.4g} exceeds the cell's bound {cell['leaf_bound']}"
        )
    metric, metric_name, higher_better = eval_metric(params, pred, y_te)
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


def _runs_alone(cell):
    return bool(cell.get("finder_tier") or cell.get("diag_set") or cell.get("debug"))


def run_tier_cell(cell):
    """A finder_tier, diag_set or debug cell, alone in a worker with FALCATA_DEBUG=diag (finder_tier, diag_set) plus
    the cell's debug tokens: its result, failed unless every logged level split finder is the expected one
    (finder_tier), or the set of the diag pattern's captures is exactly the expected one (diag_set)."""
    tokens = (["diag"] if cell.get("finder_tier") or cell.get("diag_set") else []) + list(cell.get("debug") or [])
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_SOURCE, json.dumps([cell])],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "FALCATA_DEBUG": ",".join(tokens)},
    )
    recs = [json.loads(line[5:]) for line in proc.stdout.splitlines() if line.startswith("CELL ")]
    if not recs:
        return {"id": cell["id"], "ok": False, "error": f"worker exited {proc.returncode}: {proc.stderr[-800:]}"}
    rec = recs[0]
    if cell.get("diag_set"):
        pattern, values = cell["diag_set"]
        seen = sorted(set(re.findall(pattern, proc.stdout)))
        if rec.get("ok") and seen != sorted(values):
            rec = {
                "id": cell["id"],
                "ok": False,
                "error": f"diag {pattern!r}: logged {seen}, expected exactly {values}",
            }
        return rec
    if not cell.get("finder_tier"):
        return rec
    finders = re.findall(r"split finder: (.*) kernel for this tree's quantized levels(.*)", proc.stdout)
    finders = sorted({f"{kernel}{rest}".strip() for kernel, rest in finders})
    expected = sorted(cell["finder_tier"]) if isinstance(cell["finder_tier"], list) else [cell["finder_tier"]]
    if rec.get("ok") and finders != expected:
        rec = {"id": cell["id"], "ok": False, "error": f"split finder {finders} logged, expected exactly {expected!r}"}
    return rec


def run_cells(cells):
    """Run cells in chunked subprocesses; crash-isolate and collect results."""
    results = {c["id"]: run_tier_cell(c) for c in cells if _runs_alone(c)}
    pending = [c for c in cells if not _runs_alone(c)]
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

    def expand(patterns):
        # cell ids or fnmatch patterns ('route/*'); a pattern that matches nothing is an error
        ids = set()
        for pattern in patterns:
            hits = {c["id"] for c in cells if fnmatch.fnmatchcase(c["id"], pattern)}
            if not hits:
                ap.error(f"no cell matches {pattern!r}")
            ids |= hits
        return ids

    if args.update:
        args.update = sorted(expand(args.update))
    if args.only:
        args.only = sorted(expand(args.only))
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
