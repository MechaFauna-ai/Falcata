/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_PLAN_H_
#define FALCATA_PLAN_H_

#include <Falcata/config.h>
#include <Falcata/utils/common.h>
#include <Falcata/utils/log.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace Falcata {

/*!
 * \brief CUDA execution plan: every shape-conditional kernel/pipeline choice,
 * resolved once from Config (the ``cuda_plan`` parameter) at the two entry
 * points that precede any consumer -- DatasetLoader construction (ingestion
 * decisions) and CUDASingleGPUTreeLearner::Init (training decisions).
 *
 * All decisions here are perf-only and bit-identical by contract: they change
 * how fast the model is produced, never the model. Anything that changes
 * results (quant_mode, cuda_precision) is a first-class Config parameter, not
 * a plan key.
 *
 * The plan is process-global, mirroring the env-var switches it replaced:
 * concurrent in-process boosters with different plans are unsupported.
 *
 * ``cuda_plan`` grammar: ``auto`` (default) optionally followed by
 * comma-separated expert overrides, e.g. ``auto,graph_loop:off,tuner:on``.
 * Values: ``on``/``off`` (also ``1``/``0``, ``true``/``false``). The separator
 * is ``:`` (not ``=``) because the surrounding parameter string already uses
 * ``=`` -- a nested ``=`` would be split by the parameter tokenizer.
 *
 * Fields WITHOUT an override key are measured-invariant winners: they won on
 * every benchmarked shape and are baked; the bool remains only as a one-line
 * debugging lever for developers.
 */
struct FalcataPlan {
  static constexpr int kViewModeAuto = 0;
  static constexpr int kViewModeCompact = 1;
  static constexpr int kViewModeMask = 2;
  static constexpr int kPinBinsAuto = 0;
  static constexpr int kPinBinsAlways = 1;
  static constexpr int kPinBinsNever = 2;
  // --- shape-conditional decisions (cuda_plan override keys) ---------------
  // hybrid level-batched growth (off = classic one-split-at-a-time leaf-wise)
  bool hybrid = true;               // key: hybrid
  // CUDA-graph level loop for the hybrid prefix (helps shallow trees; the
  // fixed controller latency turns net-negative on deep configs)
  bool graph_loop = true;           // key: graph_loop
  // graph level loop also for quantized training (opt-in; less measured)
  bool graph_quant = false;         // key: graph_quant
  // deterministic construct+merge nodes inside the graph loop (non-quant):
  // bit-identical to the host det loop (graph_loop:off), so opting in buys
  // run-to-run md5 identity AND the graph's launch-overhead win. Off by
  // default: the det merge costs ~4x on shallow non-quant shapes
  // (covtype-shallow-class), and the default trades that determinism for
  // speed like the fp32 opt-ins do.
  bool graph_det = false;           // key: graph_det
  // per-tree compact column view for quantized construct at any
  // 0 < feature_fraction < 1 (win scales with the excluded fraction:
  // ~3.4x at ff=0.1, ~1.1x at ff=0.6)
  bool compact_quant = true;        // key: compact_quant
  // NVRTC runtime-JIT construct kernels (self-test-then-promote; AOT
  // fallback). The JIT body is the unbatched row loop, so under auto it is
  // resolved off while row_batch is on; with row_batch:off it engages for quant
  // runs >= 300 rounds (the ~230ms one-time compile amortizes). Measured against
  // the unbatched AOT loop: +4.0% numerai-deep, +2.4% covtype-deep, +2.2% year,
  // +0.7% higgs, bit-identical.
  bool construct_jit = true;        // key: construct_jit
  // batched quantized construct: with one column per thread, each thread issues
  // the index, gradient and bin loads of 8 rows before their shared atomics (the
  // row loop is latency-bound on that dependent chain), and the flush skips empty
  // bins. Wide partitions keep the two-column loop. An explicit construct_jit:on
  // runs the JIT in its place where the JIT applies (with a warning).
  // Bit-identical. Measured: +26.1% numerai-deep, +7.7% higgs-deep against the
  // JIT (300 rounds).
  bool row_batch = true;            // key: row_batch
  // root histogram fused into the tiled 4-bit compact fill (quantized, no
  // bagging, so the root leaf is every row the fill stages): the fill adds
  // each staged row's packed gradient into per-block column histograms and the
  // root level adds that scratch instead of re-reading every row in the
  // construct. Integer sums under the construct's per-block bound: bit-identical.
  bool fused_root_hist = true;      // key: fused_root_hist
  // fused root histogram's per-block table laid out [nibble][byte slot] with
  // a 32-multiple slot stride (instead of one cell per local bin), so the
  // warp's atomics over 32 consecutive slots never share a bank. Same cells,
  // same integer sums: bit-identical. Falls back to the per-bin table when the
  // slot-major table does not fit 48 KB of shared memory.
  bool root_hist_slot_major = true;   // key: root_hist_slot_major
  // fused root histogram fill: block b stages tiles b, b + grid, b + 2*grid,
  // ... instead of tiles_per_block consecutive tiles, so at any time the
  // resident blocks read adjacent 64-byte runs of each source column (DRAM
  // page locality, like the one-tile-per-block plain fill) rather than runs
  // tiles_per_block tiles apart. Same tile count per block (the packed row
  // bound), integer sums: bit-identical.
  bool root_hist_tile_stride = true;  // key: root_hist_tile_stride
  // fused root histogram fill with a single partition spanning every slot:
  // the tile is staged at row stride = width and offset to the output run's
  // 16-byte phase, so the write is aligned 16-byte shared->global copies
  // instead of 16 byte gathers per vector. Same bytes: bit-identical.
  bool root_hist_run_copy = true;    // key: root_hist_run_copy
  // fused root histogram fill from the column-major store whose columns are
  // whole 4-byte words: each thread holds its share of the next tile's source
  // as aligned 32-bit words in registers, loaded right after the current tile
  // is staged, so those loads overlap the tile's write and histogram phases.
  // Same tile bytes: bit-identical.
  bool root_hist_prefetch = true;    // key: root_hist_prefetch
  // root_hist_prefetch with an L2 lead: right after a block issues the register
  // loads of its next tile, it also prefetches into L2 the source sectors (rows
  // below num_data only) and gradients of the tile after that, so a second tile
  // per block is in flight from DRAM and the next register loads hit L2 instead
  // of stalling the unpack. A cache hint only: bit-identical.
  bool root_hist_l2_prefetch = true;  // key: root_hist_l2_prefetch
  // colmajor_direct's column-major store with each column's start rounded up
  // to a 32-byte sector (instead of (num_data + 1) / 2 bytes apart, an odd
  // pitch for odd half row counts): every column then starts on a 4-byte word,
  // which the fused fill's word staging (root_hist_prefetch) requires, and its
  // 64-byte tile chunks are whole sectors. Pad bytes are never read as rows:
  // same column bytes, bit-identical.
  bool colmajor_align = true;        // key: colmajor_align
  // fused_root_hist's blocks sized from the shape instead of always 16 tiles: the
  // tiles over root_hist_block_rounds rounds of the blocks resident on the device
  // (occupancy API x SMs), 2 to 16 tiles each (the packed row bound still caps
  // it). Blocks of equal work end up to a block's lifetime apart, so shorter
  // blocks shorten the kernel's tail. Integer sums regrouped: bit-identical.
  bool root_hist_short_blocks = true;  // key: root_hist_short_blocks
  // true when the user wrote construct_jit:on/off -- bypasses the >=300
  // rounds auto-gate (mirrors tuner_explicit)
  bool construct_jit_explicit = false;
  // dense row-data build from column bins (skips host multi-val bin)
  bool fast_rowdata = true;         // key: fast_rowdata
  // 4-bit packed row data for <=16-bin features
  bool rowdata_4bit = true;         // key: rowdata_4bit
  // GPU dense-matrix binning during dataset construction
  bool gpu_construct = true;        // key: gpu_construct
  // GPU binning: the page-locked staging blocks stay in a process-wide pool for
  // the next construct (off: page-locked and freed by every construct). Holds
  // the ring (~1.5 GB on the Numerai split) after the first construct.
  bool construct_staging_pool = true;  // key: construct_staging_pool
  // GPU binning: the chunk upload runs on its own stream, overlapping the bin kernel
  bool construct_h2d_overlap = true;  // key: construct_h2d_overlap
  // page-locking of the Dataset's host bin storage on CUDA (key pin_bins):
  //  - auto: only where CUDAColumnData copies it per column, i.e. when
  //    num_groups x num_data <= kCUDAPerColumnMaxBytes. Above that bound only
  //    the column-major store upload of the first train() reads the bins
  //    (slower from pageable memory), while page-locking thousands of group
  //    buffers costs seconds of construct time;
  //  - always: at every size (the behaviour before the key);
  //  - never: pageable at every size (per-column copies from pageable memory;
  //    reaches the pageable path at test scale).
  int pin_bins = kPinBinsAuto;
  // cheap host precheck that skips EFB bundling on provably-unbundlable data
  bool efb_precheck = true;         // key: efb_precheck

  // --- measured-invariant winners: default ON, but keyed so each one can be
  // ablated (leave-one-out) to attribute its share of the speedup ----------
  bool split_packed_read = true;    // key: split_packed_read
  bool batch_kernels = true;        // key: batch_kernels -- one find/sync launch per level
  bool batch_apply = true;          // key: batch_apply -- batched per-level apply phase
  bool one_sync = true;             // key: one_sync -- SPECULATIVE single-sync level pipeline
  bool selective = true;            // key: selective -- SPECULATIVE grow-then-prune
  bool batch_reghist = true;        // key: batch_reghist -- register-tiled construct
  bool batch_wide = true;           // key: batch_wide -- wide leaf-splits init batching
  bool gh_interleave = true;        // key: gh_interleave -- packed grad/hess layout
  bool small_leaf_construct = true;  // key: small_leaf_construct
  // gap-gated outlier-robust gradient scale (fixedpoint quant only; no-op in
  // other modes). Changes the model when it fires -- NOT an equality key.
  bool robust_scale = true;         // key: robust_scale
  // error-feedback accumulation for fixedpoint rounding: rows carry their
  // rounding residual into the next tree, so round-to-nearest bias telescopes
  // away. Deterministic; changes the model -- NOT an equality key.
  bool quant_ef = true;             // key: quant_ef
  // cat_hybrid:off routes categorical data to the classic training loop.
  bool cat_hybrid = true;           // key: cat_hybrid
  // experimental compact-view packing codecs (uniform per tree; eligible only
  // when every sampled feature's bin count fits the codec). Bit-identical by
  // construction (lossless packing). Priority radix5 > radix6 > bit3.
  bool pack_bit3 = false;           // key: pack_bit3 -- 2.5 values/byte, <=8 bins
  bool pack_radix5 = false;         // key: pack_radix5 -- 3.25 values/byte, <=5 bins
  bool pack_radix6 = false;         // key: pack_radix6 -- 3.0 values/byte, <=6 bins
  bool pack_radix7 = false;         // key: pack_radix7 -- 2.75 values/byte, <=7 bins
  // L2 persistence window on the per-row scattered-reread buffers (grad/hess,
  // leaf data indices): each level's bin-matrix stream otherwise evicts them.
  // Cache hint only -- bit-identical by construction. Measured:
  // +2.8% numerai-deep/covtype-deep, neutral on year at real run lengths.
  bool l2_policy = true;            // key: l2_policy
  // one-time column-major copy of the packed bin matrix as the compact-fill
  // gather source: fill reads contiguous columns (~10x less fill traffic on
  // ff<<1 data) at the cost of duplicating the matrix in VRAM. Made before the
  // second tree, and only if free memory then still covers what training can
  // allocate later (CUDASingleGPUTreeLearner::ColMajorFillReserveBytes);
  // otherwise declined with a warning. Bit-identical (same bytes, different
  // source layout). Measured: +2.2% numerai-deep; inert without feature
  // sampling.
  bool colmajor_fill = true;        // key: colmajor_fill
  // with colmajor_fill, for data whose columns are all 4-bit (or 8-bit) dense
  // columns: build the device copy of the bin matrix straight from the
  // Dataset's column buffers and keep ONE copy, in the layout the training
  // regime reads (view_mode below). No row-major matrix is built at Init, no
  // transpose, no memory rule. Bit-identical (same bytes). off: the row-major
  // matrix is built at Init and colmajor_fill's copy decided before the second
  // tree, as before.
  bool colmajor_direct = true;      // key: colmajor_direct
  // colmajor_direct's training regime, decided per train() (and again on
  // ResetTrainingData / a feature_fraction change):
  //  - compact: the column-major store (the Dataset's 4-bit columns end to end)
  //    stays resident and every tree fills its compact view of the sampled
  //    columns from it;
  //  - mask: one full row-major matrix, filled from the columns once, no store,
  //    no per-tree copy; the per-tree sample reaches the kernels as the column
  //    masks they already take.
  // key view_mode: auto (mask iff feature_fraction >= view_mask_ff, or the
  // compact view is off for this training) | compact | mask.
  int view_mode = kViewModeAuto;
  // key view_mask_ff: the auto threshold, a number >= 0 (above 1 = never mask). Measured on numerai53-deep
  // (6.79M x 3555, 200 rounds): compact 189 vs mask 224 ms per round at ff 0.80, 246 vs 224 at 0.85; md5
  // identical at every ff. The mask regime also holds one matrix where compact holds the store plus an ff-sized
  // view (13.0 vs 22.2 GB peak at 0.80).
  double view_mask_ff = 0.85;
  // keys view_probe_lo / view_probe_hi: with the tuner active (cuda_plan tuner, quantized training) and view_mode
  // auto, a feature_fraction in [view_probe_lo, view_probe_hi] is decided by measurement instead: the first trees
  // run in each regime (both train the same model) and the faster one is kept and cached in the tuner's wisdom.
  // Outside the band, the static rule above. The numerai53-deep sweep's margins: compact ahead by 39% at 0.5 and
  // 16% at 0.8, mask ahead by 9-15% from 0.85 to 0.99.
  double view_probe_lo = 0.5;
  double view_probe_hi = 0.95;
  // root_hist_short_blocks: rounds of the resident blocks the fused fill's grid
  // is sized for (cuda_plan key root_hist_block_rounds:<number>; below 1: off)
  double root_hist_block_rounds = 48.0;
  // 4-bit compact fill through a shared-memory [row][slot] tile: coalesced
  // column reads, 16-byte streaming writes of each partition's contiguous
  // destination run. The host takes it for column-major sources with at most
  // 256 byte slots; anything else keeps the per-cell kernel. Bit-identical
  // (same bytes). Measured: +10.2% numerai-deep; inert without feature
  // sampling.
  bool tiled_fill = true;           // key: tiled_fill
  // quantized per-level split finder with one warp per (task, leaf) instead of
  // a 256-thread block: register prefix sums and shuffle scans/reductions, and
  // fp64 gains evaluated only for thresholds an fp32 bound cannot rule out.
  // Used for host-launched quantized levels with fp64 gains when every task of
  // the dataset is numerical with at most 256 bins and none is a zero-as-missing
  // scan (NaN handling with or without a stored most-frequent bin); otherwise
  // the block kernel. CUDA only. Bit-identical. Measured:
  // +6.3% numerai-deep, +22.7% year-deep.
  bool warp_find = true;            // key: warp_find
  // warp_find on tasks with at most 8 bins: one scan position per lane instead
  // of eight in lane 0, so the threshold unpack and fp64 gain of all positions
  // issue once per warp, not once per position. Bit-identical (same positions,
  // same expressions, same first-maximum selection as lane 0's strict scan).
  bool warp_find_narrow = true;     // key: warp_find_narrow
  // warp_find for tasks of at most 32 scan positions (few-bin features): one
  // position per lane instead of 8, so the fp64 unpack and gain math of a
  // feature with a handful of bins runs across lanes rather than serially in
  // lane 0; when every task fits 8 positions, a warp instead serves eight
  // (task, leaf) items with 4 lanes x 2 positions each. The selection
  // reproduces the 8-per-lane winner exactly (exact first maximum within each
  // 8 positions, then the same tolerance tie-break between 8-position groups
  // in the same order). Bit-identical.
  bool warp_find_spread = true;     // key: warp_find_spread
  // warp_find_spread's 4-lane groups (every task at most 8 scan positions)
  // with the positions strided over the lanes (lane l holds positions l and
  // l + 4) instead of blocked (2l and 2l + 1): a task with at most 4
  // candidate thresholds (a 5-bin feature) then has all of them in the first
  // of the two per-lane slots, so a warp whose eight items all fit skips the
  // second slot's fp64 unpack and gain math instead of issuing it with half
  // its lanes masked off. The gain loop reuses the prune's unpack of each
  // slot, and the winner's left and right child outputs are computed side by
  // side in two lanes. Prefixes are the same wrapping integer sums, every
  // value comes from the same expressions on the same inputs, and the winner
  // is the same exact first maximum (gain, then lower threshold).
  // Bit-identical.
  bool warp_find_strided = true;    // key: warp_find_strided
  // warp_find_strided's level kernel with each (task, leaf) item's loads
  // issued in three dependency levels (pair descriptor + used-task index +
  // gradient/hessian scales; then the task and the valid leaf's sums and
  // histogram pointer; then the feature-used flag) instead of one global
  // round trip per load behind each early exit. Same values, same exits and
  // writes: bit-identical.
  bool find_loads_batched = true;   // key: find_loads_batched
  // 4-bit compact quantized construct with one thread per packed byte: the two
  // nibbles of a byte index one cell of a joint (lo, hi) shared histogram, so a
  // row costs one shared atomic per byte instead of one per column; each
  // column's histogram is the joint table summed over the partner's bins before
  // the usual flush. Used for host-launched levels on the nibble compact view
  // when every partition's joint table fits 48 KB of shared memory; otherwise
  // the per-column kernel. Bit-identical (integer sums of the same rows).
  bool pair_hist = true;            // key: pair_hist
  // pair_hist row grouping per pair: each pair of a level splits its rows into
  // as many y-blocks as the batched sizing formula gives at that pair's own
  // smaller-leaf size, not at the level's largest one, so the small leaves of
  // deep levels do not pay a joint-table zero and flush per handful of rows.
  // The formula keeps the packed-cell rows-per-block cap. Bit-identical
  // (integer sums are row-grouping invariant within that cap).
  bool per_pair_rows = true;        // key: per_pair_rows
  // nibble compact view with each row's bytes of all feature partitions
  // stored back to back (partition p at its packed byte offset of the row)
  // instead of one row-major block per partition. Below the first levels a
  // leaf's rows are sparse in the matrix, so every gathered row costs whole
  // DRAM sectors: one contiguous run of the full row touches fewer of them
  // than a partial run per partition. Used on the 4-bit nibble view whenever
  // the source has more than one partition. Bit-identical (same bytes, only
  // their addresses change).
  bool compact_row_interleave = true;  // key: compact_row_interleave
  // pair_hist on the row-interleaved view: one block covers whole rows (the
  // bytes of every partition) instead of one block per partition, so a row's
  // bytes are read by one block as one contiguous run and no lanes idle on
  // narrower partitions. Used when all partitions' joint tables fit 48 KB
  // together and the row fits a block. Under per_pair_rows its rows-per-thread
  // floor scales by the partitions a block covers (a block carries all their
  // tables' zeroing and flush). Bit-identical (integer sums).
  bool pair_hist_rows = true;       // key: pair_hist_rows
  // pair_hist_rows block height: as many rows as keep the block's warps within
  // what one SM's register file holds for the pair-joint kernel (it runs one
  // block per SM), instead of the per-partition kernel's 504-thread rows
  // (Numerai v5.3's 178-byte row: 5 rows, 28 warps, instead of 4 rows, 23
  // warps). Ties keep the fewer rows. The grid is re-derived for that height
  // by the same formula and packed-cell guard. Bit-identical (integer sums).
  bool pair_block_rows = true;      // key: pair_block_rows
  // pair_hist_rows blocks on a register-capped build of the pair-joint kernel
  // (fewer rows in flight per thread), at the whole-row block height that
  // keeps strictly more warps resident per SM than the default build by the
  // occupancy API (registers and joint-table shared memory): two or three
  // shorter blocks per SM instead of one, with the kernel's shared-memory
  // carveout fitted to the resident blocks' tables (only where they need at
  // most 64% of the SM's shared memory, so the gathers keep their L1). The
  // grid is re-derived for that height by the same formula and packed-cell
  // guard. Needs CUDA 12.4 (__maxnreg__). Bit-identical (same rows, integer
  // sums).
  bool pair_capped_rows = true;     // key: pair_capped_rows
  // pair-joint construct (pair_hist) of a host-launched level with several
  // pairs: the grid is sized for the level's largest smaller leaf, and
  // with level_row_blocks every leaf runs at that leaf's rows per thread, so a
  // pair of n rows has rows in only about grid_y x n / n_max of its grid_y
  // blocks; the others are dispatched, read the pair's descriptor and leaf
  // struct, and exit. Instead the launch holds only the blocks that have rows:
  // the host counts each pair's by the device's own per-pair formula (host leaf
  // counts are the device's, read back by the level's apply) and passes the
  // pairs' first-block prefix as a kernel parameter; a block finds its pair and
  // its block row there. Same blocks with the same rows: bit-identical.
  bool pair_block_map = true;       // key: pair_block_map
  // tree boundary without host waits the computation does not need: the
  // tree start's KB-scale metadata (live compact fill, split slot tables,
  // feature masks, used tasks, hist pool pointer) is uploaded with
  // cudaMemcpyAsync on the legacy default stream the synchronous cudaMemcpy
  // used (stream order unchanged; a source local to its function is copied
  // and kept until the tree end, so it outlives the copy however the driver
  // stages it); the host blocks neither on the live fill nor on the leaf-map
  // pass or the train-score update; the gradient discretizer does not
  // synchronize the device between its default-stream kernels; the quantized
  // root-sum readback moves from InitValues to the level prefix's
  // EnsureRootSumsReadBack (every other quantized flow reads them back before
  // it starts); and at the tree end the tree's host copy and exact leaf
  // counts are read back before the leaf-map pass. The host samples columns
  // and launches the fill while the GPU still discretizes, prepares the root
  // level while the fill runs, and finalizes the tree while the leaf map is
  // written. Off with compact_prefill (its non-blocking stream shares the
  // fill metadata). Same kernels, same GPU order and inputs: bit-identical.
  bool async_tree_start = true;     // key: async_tree_start
  // with async_tree_start: the tree start's KB-scale metadata uploads between the split finder's BeforeTrain and
  // the compact fill (feature masks, used tasks, bin mask, compact-view layout, fill slot tables, fused-root meta;
  // ~14 copies of 20 B .. 18 KB) are staged into one host buffer and moved by one H2D copy into a device arena, from
  // which one kernel scatters each segment to its destination; both on the legacy default stream the copies used,
  // flushed before any GPU operation issued while the batch collects. ~14 copy-engine operations, each ~1.5 us on
  // a GPU-bound tree start, become two. Same bytes in the same buffers before the same readers: bit-identical.
  bool tree_meta_batch = true;      // key: tree_meta_batch
  // pair_hist joint tables laid out with odd per-byte strides (an even span
  // product gets one pad cell), so the same cell of neighbouring threads' tables
  // falls in distinct shared-memory banks. Bit-identical (layout only).
  bool pair_pad = true;             // key: pair_pad
  // pair_hist_rows (whole-row pair-joint construct) reading a code view of
  // the compact rows that the fused fill (fused_root_hist, one-run
  // row-interleaved layout) writes INSTEAD of the 4-bit view, in the same
  // buffer: each byte slot's joint-table cell lo * span(hi) + hi as a 5-bit
  // code, six codes per 32-bit word, a row padded to whole 32-byte sectors
  // (178 byte slots: 30 words -> 128 B instead of 178 B, 4 sectors per
  // gathered row instead of ~6.6; the fill writes 28% fewer bytes). Taken only
  // when every byte's span(lo) * span(hi) <= 32, the split reads use the
  // column-major store (colmajor_split), without compact_prefill or
  // graph_quant. Any other reader of the 4-bit view (other construct kernels,
  // the row->column gathers) first writes it with the plain tiled fill from
  // the code fill's own copy of the byte-slot tables, over the code view. The
  // kernel adds the same packed gradient to the same cell of the same table
  // per row: bit-identical.
  bool pair_code5 = true;           // key: pair_code5
  // pair_code5: the code-view construct with one thread per 32-bit code
  // word (six byte slots) instead of one per byte, so one warp covers a
  // row of up to 32 words and issues one index, gradient and coalesced code
  // load per row (the per-byte build: six warps, one of each per warp).
  // Block rows by the occupancy API. Same rows, same cells, same flush
  // rule: bit-identical.
  bool pair_code5_words = true;     // key: pair_code5_words
  // pair_code5: the fused fill packs two adjacent code words per thread
  // (four aligned shared loads for their twelve bytes, one 8-byte store)
  // instead of one word per thread (three loads, one 4-byte store). Same
  // words: bit-identical.
  bool pair_code5_pack2 = true;     // key: pair_code5_pack2
  // pair_code5: the fused fill writes the code view with plain stores, so
  // its last written rows stay in L2 for the first level's construct (off:
  // evict-first streaming stores). Same words: bit-identical.
  bool pair_code5_l2_store = true;  // key: pair_code5_l2_store
  // pair_code5_words: the code-word construct's level grid sized to the kernel's resident block slots (blocks per
  // SM by the occupancy API at its block and joint tables, times the device's SMs) instead of the generic formula
  // (whose per-pair cap predates two such blocks per SM): one rows-per-thread for all the level's pairs, the
  // smallest whose blocks fit the slots, at least the min-rows-per-thread knob and within the packed-cell row cap.
  // Same rows, other block grouping; integer sums: bit-identical.
  bool pair_code5_slots = true;     // key: pair_code5_slots
  // pair_code5_slots: the code-word blocks are the tallest the kernel can launch (fewest blocks per level, each
  // zeroing and flushing all of the row's joint tables once; on a 48-register kernel one block of 32 warps per SM)
  // instead of the height with the most resident warps. Same rows, other block grouping; integer sums:
  // bit-identical.
  bool pair_code5_tall = true;      // key: pair_code5_tall
  // pair_code5_words: the six adds of a code word without a branch per byte slot: a slot without columns adds into
  // 32 dummy cells past the joint tables (never flushed; 128 more bytes of shared memory, taken only where the block
  // still fits the device's default), and the cells are addressed by precomputed byte offsets. The build needs fewer
  // registers (40 instead of 48 here), so the occupancy API takes taller blocks; 5 rows in flight per thread, the
  // thread's last rows as one batch (unused positions add a zero gradient). Same cells of the flushed tables, same
  // nonzero adds: bit-identical.
  bool pair_code5_flat = true;      // key: pair_code5_flat
  // pair_code5_words: the code-word construct's flush maps every (slot, marginal) item of a thread's 6 byte slots
  // onto one flat threadIdx.y-strided index instead of 6 serial per-slot passes (most of blockDim.y's warps have no
  // marginal to add for a given slot with few bins per feature). Composed only with pair_code5_flat (the branch-free
  // adds this was measured on); a template choice, not a runtime branch (keeps the unchosen path's registers out of
  // the allocation). Same cells, same per-cell wrapping sum, same visitation order: bit-identical. Off by default:
  // with pair_code5_slots / pair_code5_tall's tall blocks the flat flush made a Numerai round 0.5% slower (RTX 5090).
  bool pair_code5_epilogue = false;  // key: pair_code5_epilogue
  // pair_hist on levels with several leaf pairs: every leaf takes the largest
  // leaf's rows per thread, so small leaves fill a few whole blocks instead of
  // spreading a few rows over every block row (each block zeroes and flushes
  // a whole table). Blocks never exceed the largest leaf's row count, which
  // the overflow guard bounds. Bit-identical (integer sums).
  bool level_row_blocks = true;     // key: level_row_blocks
  // pair_hist on a leaf that holds every row (the root without bagging): its
  // index list is a permutation of all rows, so each position is read as that
  // row number without the index gather. Same rows per block, same rows in
  // total. Bit-identical (integer sums).
  bool all_rows_direct = true;      // key: all_rows_direct
  // host-launched batched apply (gen-bit-vector and split-inner kernels):
  // each 1024-row chunk is handled by 256 threads of 4 rows each, all loads of
  // a thread's rows issued before their use, instead of 1024 threads of one
  // row. A 1024-thread block fills an SM alone and stalls on its own index ->
  // bin load chain; quarter-size blocks keep several chunks in flight per SM.
  // Same chunks, same ballot words, same block totals and output positions.
  // Bit-identical (only the thread -> row mapping changes).
  bool apply_row_batch = true;      // key: apply_row_batch
  // batched level best-split sync over the tree's feature sample (or over a
  // task list wider than one 1024-task block): one block per leaf reads only
  // the used tasks' slots and folds them in task order, instead of one block
  // per 1024 tasks over every task plus a cross-block merge kernel. With
  // finite gains the comparison (higher gain, then lower task) is a strict
  // total order, so any fold order picks the same winner; if a found gain is
  // not finite, the block replays the original per-1024-task reductions and
  // merge exactly. Bit-identical.
  bool sync_used_tasks = true;      // key: sync_used_tasks
  // host-launched level best-split sync (the used-task kernel of
  // sync_used_tasks and the per-1024-task kernel): the winner copy (thread 0
  // copies the winning task's split info into the leaf's slot) with every
  // field loaded before the first store, instead of CUDASplitInfo::operator='s
  // alternating load / store per field (a store may alias the next source
  // field: one global round trip per field in a row); the used-task fold's
  // slot loads batched per thread and skipped for a leaf the descriptor marks
  // invalid; and, when every found gain of the block is finite, the block
  // reduction on integer keys of the gains (same order as the fp64 tie-break,
  // whose zero tolerance makes it gain-then-lower-index) instead of a chain of
  // fp64 compares. Same winner, same values in the same fields: bit-identical.
  bool sync_copy_batched = true;    // key: sync_copy_batched
  // quantized host-launched level fix + subtract with a feature sample: one
  // thread per (pair, sampled feature) fixes the smaller leaf's most-frequent
  // bin and subtracts that feature's bins into the larger leaf (or the pair's
  // bit-change buffer), instead of a fix kernel with a block per (pair,
  // feature needing a fix) and a subtract kernel with a thread per (pair,
  // histogram bin) whose unsampled features' blocks and threads exit at once.
  // Taken when every sampled feature spans at most 8 histogram bins (and at
  // most 512 features are sampled). The same
  // integer expressions per bin in the same bit-width cases, the fixed bin's
  // value used by its own subtract: bit-identical.
  bool fix_subtract_fused = true;   // key: fix_subtract_fused
  // host-launched batched level apply: each split's tree-structure update (child
  // leaf outputs, split info slots, hist pool pointers, smaller/larger leaf
  // structs) is written by warp 0 of one extra block of the partition kernel's
  // flat grid (the first num_splits flat ids, so its short dependent chain
  // overlaps the partition) instead of by a separate 32-thread-per-split kernel
  // after the partition. Its inputs (the aggregate's child counts and starts, the
  // level's split infos) are final before the partition kernel, and the
  // partition neither reads nor writes what it writes. Same expressions, same
  // values, same stream: bit-identical.
  bool apply_struct_fused = true;   // key: apply_struct_fused
  // host-launched batched level apply: the gap copy (terminal leaves' index
  // ranges carried from the old main array into the out buffer) runs as extra
  // 1024-row chunks at the end of the partition kernel's flat grid instead of a
  // separate (largest gap x gaps) kernel. The ranges are disjoint from every
  // split window the partition reads or writes. Same copies: bit-identical.
  bool gap_copy_fused = true;       // key: gap_copy_fused
  // the batched level apply's fused partition kernel (apply_struct_fused / gap_copy_fused) with apply_row_batch at
  // twice its rows per thread (8, 128-thread blocks for each 1024-row chunk) under that block's launch bound, so all
  // of a thread's rows stay in flight in registers: the default build's bound (a 1024-thread block) caps it at 64
  // registers, at which fewer 256-thread blocks fit an SM than its thread limit allows. Taken only where the
  // occupancy API gives the wide build strictly more rows in flight per SM. Same chunk rows, ballot bits, positions
  // and stores: bit-identical.
  bool apply_inner_rows = true;     // key: apply_inner_rows
  // with apply_row_batch, the batched level apply's host-launched gen-bit kernel at twice its rows per thread (8,
  // 128-thread blocks for each 1024-row chunk) with the block size a compile-time constant and that block's launch
  // bound. Taken only where the occupancy API gives it strictly more rows in flight per SM than the default build.
  // Same chunk rows, ballot words and chunk totals: bit-identical.
  bool apply_genbit_rows = true;    // key: apply_genbit_rows
  // quantized root sums (gradient/hessian totals of the tree's rows): one warp
  // per 1024-row chunk sums 32 rows per lane, instead of one 1024-thread block
  // per chunk with one row per thread and two block reductions. Each chunk's
  // integer sums, and so every per-chunk partial the final reduction reads,
  // are the same. Bit-identical.
  bool root_sums_warp = true;       // key: root_sums_warp
  // quantized gradient discretizer's per-chunk min/max: one warp per chunk of
  // 1024 rows runs the block reduction's shuffle trees itself (the 32 per-warp
  // trees in turn, then the cross-warp tree over their results), instead of a
  // 1024-thread block with one row per thread and four block reductions. Same
  // trees and operand order, so the same partials, NaN and signed zero
  // included. Bit-identical.
  bool minmax_warp = true;          // key: minmax_warp
  // packed split read from the column-major nibble store (colmajor_direct's
  // compact regime, or colmajor_fill's copy) instead of the row-major compact
  // matrix: the partition reads a row's split bin from a contiguous
  // two-rows-per-byte column, not one sector of the row matrix per row. Used
  // whenever the store exists. Bit-identical (same nibbles).
  bool colmajor_split = true;       // key: colmajor_split
  // host-launched level flow: a leaf whose row count n cannot give both children min_data_in_leaf rows
  // (n + 2 + n / 2^20 < 2 * min_data_in_leaf) is not split-searched, and a pair whose two leaves are both such leaves
  // (or fail the existing min_data / min_sum_hessian gates) is not constructed. Every finder count gate needs left
  // and right counts of at least min_data_in_leaf, and the two counts of a threshold sum to n (one is n minus the
  // other) or, rounded separately from hessian sums, to at most n + 1 plus their floating-point error: no threshold
  // of such a leaf passes, the finder would report no split, and its histogram has no other reader. Off with forced
  // splits. Bit-identical.
  bool skip_unsplittable = true;    // key: skip_unsplittable
  // the leaf-wise tail's first best-of-all-leaves search (two kernels, two device syncs and a readback at every
  // tree end) is not run when the level prefix ended on a final level that split every candidate leaf with the
  // leaf budget not binding and every child at max_depth: the children's cached candidates were invalidated and
  // every other leaf's candidate was already invalid (otherwise it would have been a candidate of that level), so
  // the search can only report no split. Same tree: bit-identical.
  bool skip_empty_tail = true;      // key: skip_empty_tail
  // batched level apply: the out index buffer becomes the main one after each level, so every leaf not split at a
  // level (a gap of the split regions) has its index range carried from the old main into the out buffer. After
  // that copy both buffers hold the level's gap ranges, and at the next level the out buffer is that old main,
  // untouched there: only the parts of the gaps that were not gaps of the previous batched level (ranges that
  // became terminal now) are copied; the rest already holds the same indices. Tracked on the host, valid only
  // between consecutive batched levels of one tree (every other writer of either buffer resets it). Same values in
  // the same buffers: bit-identical. (Port of c043.)
  bool gap_copy_once = true;        // key: gap_copy_once
  // the host-launched tree-end row -> leaf map pass (MaterializeLeafMap*, 64 blocks per leaf as before) and the
  // batched level apply's gap copy (blocks per gap from its rows) use 256-thread blocks instead of 1024-thread ones:
  // a deep tree's leaves and gaps hold a few thousand rows each, so most of a 1024-thread block had no row while it
  // held a whole SM. Same rows, each written once with the same value: bit-identical. (Port of c043.)
  bool leaf_map_small_blocks = true;  // key: leaf_map_small_blocks
  // final batched level (every child at max_depth): the row -> leaf map pass over the leaves NOT split at that level
  // is launched right after the level's apply kernels instead of after the tree end's readbacks and leaf-wise tail
  // check: its inputs (those leaves' index windows and the leaf list) are final then, and nothing between writes the
  // map or those windows, so the GPU writes the map while the host finalizes the tree. If the leaf-wise tail still
  // splits after it, the tree end writes the whole map again as before. Same kernel, same values: bit-identical.
  bool early_leaf_map = true;       // key: early_leaf_map
  // the two synchronous readbacks of every host-launched level (the level's best splits, SyncAllLeafBestSplitsToHost;
  // the applied splits' child counts and sums, FinishSplitBatch) and the tree's own readbacks (the deferred root sums,
  // the tree's pooled slab in CUDATree::ToHost, the exact leaf counts) are copied by a small kernel on the default
  // stream into pinned staging buffers allocated mapped, followed by a synchronize of that stream, instead of a
  // cudaMemcpy D2H on that stream: same stream, so the copy waits for the same preceding work, and the host blocks
  // until the bytes are in host memory as before, but the copy starts about as soon as the producing kernel ends
  // instead of 2-6 us later when the copy engine picks it up. Falls back to cudaMemcpy where the device cannot map
  // the buffer. Same bytes: bit-identical.
  bool readback_kernel = true;      // key: readback_kernel
  // with readback_kernel, the classic host-launched quantized/plain level loop (single GPU, no feature-parallel
  // merge, scalar leaves, no compact_prefill side stream): the level's two readbacks are written into their mapped
  // pinned staging by the kernels that produce them instead of by a copy kernel on the default stream after them.
  // SyncBestSplitForLevelUsedTasksKernel copies each leaf entry it has just written (the whole struct, as the device
  // holds it) into the best-split staging; the split tree-structure body copies its split's 18 ints into the
  // split-info staging. The host then waits with one device synchronize (every stream, a superset of what the
  // default-stream copy waited for) and reads the staging as before. Within the loop every leaf the level readback
  // returns was written by a mirrored sync kernel of this tree (each child is in exactly one pair of the next level)
  // and nothing else writes those entries before the readback, so the staging holds the same bytes the copy kernel
  // would have copied; a level whose sync kernel did not mirror falls back to the full copy. Removes the copy kernel
  // and its wait on the producer's stream from both per-level round trips. Same bytes: bit-identical.
  bool readback_fused = true;       // key: readback_fused
  // the final batched level's invalidation of its children's cached split candidates (InvalidateLeafCandidates):
  // the leaf list goes up with cudaMemcpyAsync on the default stream the synchronous cudaMemcpy used (from a member
  // copy, kept until the next tree) and the kernel is not followed by a device synchronize. The host
  // reads nothing the kernel writes; every reader of those candidates (the leaf-wise tail's search, the next tree's
  // level syncs and readbacks) is ordered after it on the GPU. The host no longer waits for the final level's apply
  // to drain before it launches the residual-leaf map pass and the tree-end readbacks. Off with compact_prefill (as
  // async_tree_start). Same kernel, same GPU order: bit-identical.
  bool invalidate_async = true;     // key: invalidate_async
  // host-launched batched level without categorical splits or vector leaves: the data partition's apply (descriptor
  // upload and the gen-bit / aggregate / split-inner / tree-structure kernels on its stream) is issued before the
  // tree's record of the same splits (CUDATree::SplitBatch: upload and SplitBatchKernel on the tree's stream). The two
  // touch disjoint buffers (the tree's arrays vs the partition's; both only read the level's cached split infos),
  // there is no event between those streams, and every later reader waits for both (the level's readback on the
  // default stream), so the level's long kernel chain no longer starts behind the record's upload, launch and host
  // bookkeeping. The record's entries (real feature index and threshold, missing type: per-feature lookups only the
  // record reads) are built after the apply is launched, in the same order with the same values. Same kernels, same
  // streams, same inputs: bit-identical.
  bool level_apply_first = true;    // key: level_apply_first
  // the CUDA objective's GetGradients (the shared CUDAObjectiveInterface path) does not end with a device
  // synchronize when the gradients are device memory: its kernel runs on the legacy default stream, every reader of
  // the gradients is a kernel or a cudaMemcpy on that stream or on a blocking stream (ordered after it), and the
  // host reads nothing the kernel writes. The host then prepares the tree start (bagging, discretizer launches,
  // metadata uploads) while the gradients are computed instead of after. Host-memory gradients keep the synchronize.
  // Off with compact_prefill (its non-blocking stream). Same kernels, same GPU order: bit-identical.
  bool gradients_no_sync = true;    // key: gradients_no_sync
  // a pooled CUDA tree's retained device leaf values (the one device array ToHost keeps for shrinkage and score
  // updates) go into a buffer allocated when the host-launched flow issues a final batched level (the tree's leaf
  // count after it is known then, and the host would otherwise wait for that level's readback), and the slab's leaf
  // values are copied into it at the tree end with cudaMemcpyAsync on the legacy default stream the synchronous
  // cudaMemcpy used (its readers, shrinkage and score updates, are kernels and copies on that stream; the slab is
  // rewritten only by later work of the next tree). The tree-end allocation and the host wait on the copy leave the
  // path to the shrinkage kernel. Used only if the tree ends with exactly that leaf count, otherwise freed and the
  // tree end allocates as before. Same bytes: bit-identical.
  bool tree_end_prealloc = true;    // key: tree_end_prealloc
  // quantized gradient discretizer with constant hessians (the objective reports IsConstantHessian, no GOSS, and the
  // dataset has no weights, so every row's hessian is the value the objective writes for all rows): the per-chunk
  // min/max (minmax_warp) and the discretize kernel take row 0's hessian for every row instead of reading the whole
  // hessian array (5.5 M floats, twice per tree on numerai). Same value in the same expressions: bit-identical.
  bool const_hess_reads = true;     // key: const_hess_reads
  // histogram pool zeroing at the tree start: after a tree whose level prefix ended on a final level that completed
  // it (skip_empty_tail ran no tail), only the slots of the leaves that existed before that level can hold sums --
  // the slots the final level hands its new children are never written (no histogram is built after it) -- so the
  // next tree zeroes that prefix of slots instead of the tree's num_leaves. The pool is all zero at every tree start
  // as before: bit-identical.
  bool dirty_final_slots = true;    // key: dirty_final_slots
  // quantized training: the tree start's histogram pool zeroing covers only the first half of each dirty slot (a
  // quantized bin is an int32 or int64 at its bin index: num_total_bin * 8 of the slot's num_total_bin * 16 bytes),
  // with one 2D memset on the same stream; the other half, zeroed by the first full zeroing after the pool was set
  // up or reset, is never written by a quantized kernel and stays zero. Same zero pool: bit-identical.
  bool hist_zero_quant_half = true;  // key: hist_zero_quant_half
  // final batched level of a level prefix that completes the tree (every child at max_depth, every candidate split,
  // budget not binding: the skip_empty_tail condition, which must be on): the children's index windows have no
  // reader -- the tail is known empty, the tree end reads exact counts and the map, and the residual leaves'
  // windows are gaps that stay in the main index array -- so the split-inner pass writes only the row -> leaf map
  // (same rows, same values), after a full-sector clear of the map that keeps its scatter in L2 (every entry is
  // rewritten by this pass or the residual-leaf map pass), and the gap copy and index buffer swap are not run. A
  // reader of the windows that turns up after all (an objective's leaf renewal, refit, any later apply) completes
  // the partition first from the level's untouched descriptors and direction bits. Off with bagging (the next tree
  // reuses the index array), linear trees, quantized leaf renewal, multi-GPU, vector leaves and selective growth.
  // Same map, same counts, same tree: bit-identical.
  bool final_map_only = true;       // key: final_map_only
  // tree end without host waits on work the readbacks do not need. With early_leaf_map, the final batched level's
  // residual-leaf map pass is not launched with the level (there the level's split batch readback and the tree
  // end's readbacks queued behind it on the default stream, the host waiting for the whole pass each time) but at
  // the tree end, after the exact leaf counts are read back (with the slab's copy under readback_kernel), right
  // after ToHost's copy of the tree: it runs while the host scatters the copy (its inputs, the residual leaves'
  // windows, are final then; it reads none of the tree's arrays). A pooled tree's ToHost does not synchronize the
  // device (its arrays are views of the learner's slab: nothing is freed; the retained leaf values are copied out on
  // the default stream). Same kernels, same values: bit-identical.
  bool final_readback_first = true;  // key: final_readback_first
  // the register-capped pair-joint construct's block height (pair_capped_rows) memoised per launch shape (row bytes,
  // joint-table bytes, default height, gradient bins) instead of for the last shape only: the joint-table bytes
  // change with every tree's column sample, so every tree re-ran the occupancy queries and re-set the kernels'
  // shared-memory carveout before its first construct. The carveout is re-set only when the chosen value changes.
  // Same block height per shape: bit-identical.
  bool shape_memo = true;           // key: shape_memo
  // runtime tier-1 tuner: bandit over the batched-construct saturation floor,
  // timed per tree; quantized training only (integer hists keep results
  // schedule-invariant, so retuning cannot change the model). The probe phase
  // costs ~60 trees, so the learner additionally gates on num_iterations >=
  // 300 under auto (explicit tuner:on bypasses that gate). Measured: +2.1%
  // numerai-deep, +2.7% year @500r.
  bool tuner = true;                // key: tuner
  // true when the user wrote tuner:on/off themselves -- the learner's
  // num_iterations >= 300 auto-gate only applies when this is false
  bool tuner_explicit = false;
  // wide partitions: let few-bin partitions hold up to 2x504 columns (each
  // construct thread handles 2 columns), halving partition count and its
  // per-partition zero/merge overhead on wide low-bin data. Measured: +8.8%
  // numerai-deep; engages only above 504 columns.
  bool wide_partitions = true;      // key: wide_partitions
  // prefill the next tree's compact column view on a side stream during the
  // current tree's training (bit-identical). Default OFF: measured no wall
  // win -- steady-state training is 97% device-busy and the ~41 synchronous
  // D2H readbacks per tree impose legacy-stream barriers that serialize the
  // fill anyway -- while the alt buffer costs VRAM (620MB on numerai). Becomes
  // worthwhile only if the barrier structure is reduced; kept keyed and
  // gate-covered so the machinery stays correct.
  bool compact_prefill = false;     // key: compact_prefill

  // --- baked tuning constants (no keys) ------------------------------------
  int batch_construct_min_rows_per_thread = 64;
  int batch_construct_saturation_floor = 160;
  // quant small-leaf direct-kernel row threshold (0 = off). Behavior-
  // preserving at any value (order-invariant integer atomics), so the
  // runtime tuner may retune it mid-training like the saturation floor.
  int quant_small_leaf_rows = 1024;
  int construct_column_cap = -1;    // -1 = auto sizing

  // --- resolved facts from Config (no keys; inputs to auto decisions) -------
  // Quantized training active. Shape auto-decisions that only win under the
  // quantized construct kernel (e.g. the low-bin column-cap lowering) gate on
  // this: the 252-column auto-cap measured +7% construct on quant numerai but
  // -2% wall on the non-quant numerai config (4-bit packed + compact view).
  bool quant_training = false;

  /*!
   * \brief Address of the flag a ``cuda_plan`` key controls, or nullptr if the
   * key is unknown. Every ablatable decision is reachable here, so the
   * benchmark suite can leave-one-out each feature without new plumbing.
   */
  bool* KeySlot(const std::string& key) {
    if (key == "hybrid") return &hybrid;
    if (key == "graph_loop") return &graph_loop;
    if (key == "graph_quant") return &graph_quant;
    if (key == "graph_det") return &graph_det;
    if (key == "compact_quant") return &compact_quant;
    if (key == "construct_jit") return &construct_jit;
    if (key == "row_batch") return &row_batch;
    if (key == "fused_root_hist") return &fused_root_hist;
    if (key == "root_hist_slot_major") return &root_hist_slot_major;
    if (key == "root_hist_tile_stride") return &root_hist_tile_stride;
    if (key == "root_hist_run_copy") return &root_hist_run_copy;
    if (key == "root_hist_prefetch") return &root_hist_prefetch;
    if (key == "root_hist_l2_prefetch") return &root_hist_l2_prefetch;
    if (key == "colmajor_align") return &colmajor_align;
    if (key == "root_hist_short_blocks") return &root_hist_short_blocks;
    if (key == "fast_rowdata") return &fast_rowdata;
    if (key == "rowdata_4bit") return &rowdata_4bit;
    if (key == "gpu_construct") return &gpu_construct;
    if (key == "construct_staging_pool") return &construct_staging_pool;
    if (key == "construct_h2d_overlap") return &construct_h2d_overlap;
    if (key == "efb_precheck") return &efb_precheck;
    if (key == "split_packed_read") return &split_packed_read;
    if (key == "batch_kernels") return &batch_kernels;
    if (key == "batch_apply") return &batch_apply;
    if (key == "one_sync") return &one_sync;
    if (key == "selective") return &selective;
    if (key == "batch_reghist") return &batch_reghist;
    if (key == "batch_wide") return &batch_wide;
    if (key == "gh_interleave") return &gh_interleave;
    if (key == "small_leaf_construct") return &small_leaf_construct;
    if (key == "robust_scale") return &robust_scale;
    if (key == "quant_ef") return &quant_ef;
    if (key == "cat_hybrid") return &cat_hybrid;
    if (key == "compact_prefill") return &compact_prefill;
    if (key == "pack_bit3") return &pack_bit3;
    if (key == "pack_radix5") return &pack_radix5;
    if (key == "pack_radix6") return &pack_radix6;
    if (key == "pack_radix7") return &pack_radix7;
    if (key == "l2_policy") return &l2_policy;
    if (key == "colmajor_fill") return &colmajor_fill;
    if (key == "colmajor_direct") return &colmajor_direct;
    if (key == "tiled_fill") return &tiled_fill;
    if (key == "warp_find") return &warp_find;
    if (key == "warp_find_narrow") return &warp_find_narrow;
    if (key == "warp_find_spread") return &warp_find_spread;
    if (key == "warp_find_strided") return &warp_find_strided;
    if (key == "find_loads_batched") return &find_loads_batched;
    if (key == "pair_hist") return &pair_hist;
    if (key == "per_pair_rows") return &per_pair_rows;
    if (key == "compact_row_interleave") return &compact_row_interleave;
    if (key == "pair_hist_rows") return &pair_hist_rows;
    if (key == "pair_block_rows") return &pair_block_rows;
    if (key == "pair_capped_rows") return &pair_capped_rows;
    if (key == "pair_block_map") return &pair_block_map;
    if (key == "async_tree_start") return &async_tree_start;
    if (key == "tree_meta_batch") return &tree_meta_batch;
    if (key == "pair_pad") return &pair_pad;
    if (key == "pair_code5") return &pair_code5;
    if (key == "pair_code5_words") return &pair_code5_words;
    if (key == "pair_code5_pack2") return &pair_code5_pack2;
    if (key == "pair_code5_l2_store") return &pair_code5_l2_store;
    if (key == "pair_code5_slots") return &pair_code5_slots;
    if (key == "pair_code5_tall") return &pair_code5_tall;
    if (key == "pair_code5_flat") return &pair_code5_flat;
    if (key == "pair_code5_epilogue") return &pair_code5_epilogue;
    if (key == "level_row_blocks") return &level_row_blocks;
    if (key == "all_rows_direct") return &all_rows_direct;
    if (key == "apply_row_batch") return &apply_row_batch;
    if (key == "sync_used_tasks") return &sync_used_tasks;
    if (key == "sync_copy_batched") return &sync_copy_batched;
    if (key == "fix_subtract_fused") return &fix_subtract_fused;
    if (key == "apply_struct_fused") return &apply_struct_fused;
    if (key == "gap_copy_fused") return &gap_copy_fused;
    if (key == "apply_inner_rows") return &apply_inner_rows;
    if (key == "apply_genbit_rows") return &apply_genbit_rows;
    if (key == "root_sums_warp") return &root_sums_warp;
    if (key == "minmax_warp") return &minmax_warp;
    if (key == "colmajor_split") return &colmajor_split;
    if (key == "skip_unsplittable") return &skip_unsplittable;
    if (key == "skip_empty_tail") return &skip_empty_tail;
    if (key == "gap_copy_once") return &gap_copy_once;
    if (key == "leaf_map_small_blocks") return &leaf_map_small_blocks;
    if (key == "early_leaf_map") return &early_leaf_map;
    if (key == "readback_kernel") return &readback_kernel;
    if (key == "readback_fused") return &readback_fused;
    if (key == "invalidate_async") return &invalidate_async;
    if (key == "level_apply_first") return &level_apply_first;
    if (key == "gradients_no_sync") return &gradients_no_sync;
    if (key == "tree_end_prealloc") return &tree_end_prealloc;
    if (key == "const_hess_reads") return &const_hess_reads;
    if (key == "dirty_final_slots") return &dirty_final_slots;
    if (key == "hist_zero_quant_half") return &hist_zero_quant_half;
    if (key == "final_map_only") return &final_map_only;
    if (key == "final_readback_first") return &final_readback_first;
    if (key == "shape_memo") return &shape_memo;
    if (key == "tuner") return &tuner;
    if (key == "wide_partitions") return &wide_partitions;
    return nullptr;
  }

  /*! \brief The process-global plan (mutable form, for the resolve points). */
  static FalcataPlan& Mutable() {
    static FalcataPlan plan;
    return plan;
  }
  /*! \brief The process-global plan, as consumers read it. */
  static const FalcataPlan& Get() { return Mutable(); }

  /*!
   * \brief Resolve the global plan from a parsed Config. Called from
   * DatasetLoader (ingestion) and the CUDA tree learner Init (training);
   * both parse the same string so the plan is consistent across phases.
   */
  static void ResolveFromConfig(const Config& config) {
    FalcataPlan plan;  // defaults = the auto plan
    plan.quant_training = config.ResolvedQuantMode() != QuantMode::kNone;
    std::string spec = Common::Trim(config.cuda_plan);
    std::transform(spec.begin(), spec.end(), spec.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    bool overridden = false;
    for (const std::string& raw : Common::Split(spec.c_str(), ',')) {
      const std::string token = Common::Trim(raw);
      if (token.empty() || token == std::string("auto")) continue;
      const std::vector<std::string> kv = Common::Split(token.c_str(), ':');
      if (kv.size() != 2) {
        Log::Fatal("cuda_plan: bad token \"%s\" (expected key:on|off)", token.c_str());
      }
      // the non-boolean keys: view_mode, pin_bins and the view_* numbers
      if (kv[0] == std::string("pin_bins")) {
        if (kv[1] == std::string("auto")) {
          plan.pin_bins = kPinBinsAuto;
        } else if (kv[1] == std::string("always")) {
          plan.pin_bins = kPinBinsAlways;
        } else if (kv[1] == std::string("never")) {
          plan.pin_bins = kPinBinsNever;
        } else {
          Log::Fatal("cuda_plan: bad value \"%s\" for key \"pin_bins\" (expected auto|always|never)", kv[1].c_str());
        }
        overridden = true;
        continue;
      }
      if (kv[0] == std::string("view_mode")) {
        if (kv[1] == std::string("auto")) {
          plan.view_mode = kViewModeAuto;
        } else if (kv[1] == std::string("compact")) {
          plan.view_mode = kViewModeCompact;
        } else if (kv[1] == std::string("mask")) {
          plan.view_mode = kViewModeMask;
        } else {
          Log::Fatal("cuda_plan: bad value \"%s\" for key \"view_mode\" (expected auto|compact|mask)", kv[1].c_str());
        }
        overridden = true;
        continue;
      }
      double* number = kv[0] == std::string("view_mask_ff") ? &plan.view_mask_ff :
                       kv[0] == std::string("view_probe_lo") ? &plan.view_probe_lo :
                       kv[0] == std::string("view_probe_hi") ? &plan.view_probe_hi :
                       kv[0] == std::string("root_hist_block_rounds") ? &plan.root_hist_block_rounds : nullptr;
      if (number != nullptr) {
        char* end = nullptr;
        const double value = std::strtod(kv[1].c_str(), &end);
        if (end == kv[1].c_str() || *end != '\0' || !(value >= 0.0)) {
          Log::Fatal("cuda_plan: bad value \"%s\" for key \"%s\" (expected a number >= 0)", kv[1].c_str(),
                     kv[0].c_str());
        }
        *number = value;
        overridden = true;
        continue;
      }
      bool value;
      if (kv[1] == std::string("on") || kv[1] == std::string("1") || kv[1] == std::string("true")) {
        value = true;
      } else if (kv[1] == std::string("off") || kv[1] == std::string("0") || kv[1] == std::string("false")) {
        value = false;
      } else {
        Log::Fatal("cuda_plan: bad value \"%s\" for key \"%s\" (expected on|off)",
                   kv[1].c_str(), kv[0].c_str());
        return;
      }
      bool* slot = plan.KeySlot(kv[0]);
      if (slot == nullptr) {
        Log::Fatal("cuda_plan: unknown key \"%s\"", kv[0].c_str());
      }
      *slot = value;
      if (kv[0] == std::string("tuner")) plan.tuner_explicit = true;
      if (kv[0] == std::string("construct_jit")) plan.construct_jit_explicit = true;
      overridden = true;
    }
    if (plan.row_batch && plan.construct_jit) {
      if (plan.construct_jit_explicit) {
        Log::Warning("cuda_plan: construct_jit:on replaces the row_batch construct with the unbatched JIT kernel "
                     "wherever the JIT applies");
      } else {
        plan.construct_jit = false;
      }
    }
    Mutable() = plan;
    if (overridden) {
      Log::Info("cuda_plan: hybrid=%d selective=%d one_sync=%d graph_loop=%d graph_quant=%d "
                "compact_quant=%d construct_jit=%d row_batch=%d fast_rowdata=%d rowdata_4bit=%d "
                "gpu_construct=%d efb_precheck=%d batch_kernels=%d batch_apply=%d "
                "batch_reghist=%d batch_wide=%d gh_interleave=%d split_packed_read=%d "
                "small_leaf_construct=%d",
                plan.hybrid, plan.selective, plan.one_sync, plan.graph_loop, plan.graph_quant,
                plan.compact_quant, plan.construct_jit, plan.row_batch, plan.fast_rowdata, plan.rowdata_4bit,
                plan.gpu_construct, plan.efb_precheck, plan.batch_kernels, plan.batch_apply,
                plan.batch_reghist, plan.batch_wide, plan.gh_interleave, plan.split_packed_read,
                plan.small_leaf_construct);
    }
  }
};

/*!
 * \brief FALCATA_VERIFY=1: verify every enabled fast path against its
 * reference implementation (byte/bit comparison) during the run. Developer
 * gate for the check-then-drop workflow; never affects results, only speed.
 */
inline bool FalcataVerifyEnabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("FALCATA_VERIFY");
    return env != nullptr && std::string(env) == std::string("1");
  }();
  return enabled;
}

/*!
 * \brief FALCATA_DEBUG: comma-separated developer diagnostics, e.g.
 * ``FALCATA_DEBUG=diag,dump,maxsplits=8``. Tokens:
 *  - ``diag``       per-phase timing/counter diagnostics
 *  - ``debug``      verbose hybrid-growth debug checks/logging
 *  - ``dump``       dump partition snapshots to files
 *  - ``syncpairs``  force per-pair synchronization (isolates batching)
 *  - ``aggressive`` experimental aggressive hybrid batching
 *  - ``maxsplits=N`` cap splits per level (isolates multi-pair interactions)
 *  - ``vramfree=N`` treat at most N MiB of device memory as free when
 *    colmajor_fill decides whether its copy fits (tests the decline path
 *    without filling the GPU); does not change any result
 * The growth tokens (debug, syncpairs, maxsplits) also disable the CUDA-graph
 * controller path (the device controller does not replicate these hooks).
 */
struct FalcataDebugOptions {
  bool diag = false;
  bool debug = false;
  bool dump = false;
  bool syncpairs = false;
  bool aggressive = false;
  int maxsplits = -1;  // -1 = uncapped
  int64_t vramfree_mib = -1;  // -1 = the real free memory
  bool any_growth_hook() const { return debug || syncpairs || maxsplits >= 0; }
};

inline const FalcataDebugOptions& FalcataDebug() {
  static const FalcataDebugOptions opts = []() {
    FalcataDebugOptions o;
    const char* env = std::getenv("FALCATA_DEBUG");
    if (env == nullptr) return o;
    for (const std::string& raw : Common::Split(env, ',')) {
      const std::string token = Common::Trim(raw);
      if (token == std::string("diag")) {
        o.diag = true;
      } else if (token == std::string("debug")) {
        o.debug = true;
      } else if (token == std::string("dump")) {
        o.dump = true;
      } else if (token == std::string("syncpairs")) {
        o.syncpairs = true;
      } else if (token == std::string("aggressive")) {
        o.aggressive = true;
      } else if (token.rfind("maxsplits=", 0) == 0) {
        o.maxsplits = std::atoi(token.c_str() + 10);
      } else if (token.rfind("vramfree=", 0) == 0) {
        o.vramfree_mib = std::min<int64_t>(std::max<int64_t>(0, std::atoll(token.c_str() + 9)), int64_t{1} << 40);
      } else if (!token.empty()) {
        Log::Warning("FALCATA_DEBUG: unknown token \"%s\" ignored", token.c_str());
      }
    }
    return o;
  }();
  return opts;
}

}  // namespace Falcata

#endif  // FALCATA_PLAN_H_
