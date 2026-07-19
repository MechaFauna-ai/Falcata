# NVRTC JIT construct-kernel specialization — design (phase 2)

Status: DESIGN (phase 1 characterization + a compile-time-templated prototype are
landed; this doc specifies the runtime NVRTC JIT that phase 2 builds).

## 1. Motivation & phase-1 findings

The quantized dense **construct** kernel
(`CUDAConstructDiscretizedHistogramDenseBatchedKernel`) is the single largest GPU
cost in quantized CUDA training. Phase 1 characterized it per benchmark to decide
where a shape-specialized (eventually JIT-compiled) construct can win.

### Per-benchmark characterization (RTX 5090, 1.8 TB/s roofline)

`block_dim_x = max_num_column_per_partition`, `block_dim_y = 504 / block_dim_x`
(NUM_THREADS_PER_BLOCK = 504). Achieved bandwidth is an **upper bound** (assumes
every tree level re-reads all rows; real is lower, so real % is lower still).

| bench   | feat | bins/feat | binding cap        | #part | block_dim_y | construct ms/tree | UB %roofline | bound type |
|---------|------|-----------|--------------------|-------|-------------|-------------------|--------------|------------|
| numerai | 2748 | ~6        | **504-COLUMN**     | 6     | **1**       | 51.0 (ex.)        | 9.9%         | scattered-read LATENCY |
| higgs   | 28   | 255       | bin (6144)         | 1     | 18          | 2.13              | 52.5%        | ~bandwidth |
| epsilon | 2000 | ~4-256    | bin (6144)         | 84    | 21          | 6.23              | 42.9%        | ~bandwidth |
| year    | 90   | ~120-255  | bin (6144)         | 4     | 21          | 0.29              | 50.0%        | ~bandwidth |
| fraud   | 29   | ~var      | bin (6144)         | 2     | 21          | 0.09              | 27.4%        | launch/latency (tiny) |
| covtype | 53   | ~40-255   | neither (12 grp)   | 1     | 42          | 1.14              | 13.2%        | small / other-stage |

Key structural facts:
- **numerai is the only column-capped benchmark.** ~6 bins/feature means 504
  columns fit in one partition well under the 6144-bin shared cap, so the
  504-column cap binds and forces `block_dim_y = 1`: every column-thread issues
  one **uncoalesced** bin read (column-major storage → consecutive column threads
  are `num_data` bytes apart) then a dependent shared atomic, with no row-level ILP
  to hide the ~hundreds-of-cycle gather latency. Achieved DRAM throughput is ~10%
  of roofline (UB) → **not bandwidth-bound; latency-bound.**
- The ≤8-bin **register-histogram path** (`USE_REG_BINS`, `kRegHistMaxBins=8`,
  `construct_reg_bins_`) exists **only in the non-quantized** construct body and is
  quality-parity (float order differs), not bit-parity. The **quantized** discretized
  body has no register path — but integer atomics are order-invariant, so a
  bit-identical quant register/regroup specialization is possible. A/B on non-quant
  numerai showed the register path is **not** faster there (8.94 vs 8.14 ms),
  confirming the bottleneck is the scattered read, not atomic contention.
- Construct time is **flat** across feature_fraction 0.05→0.2 (~51 ms): the cost is
  fixed per-row / per-block structure, not the active-feature reads. The compact
  view (materializes only sampled columns, shrinking blocks) is **disabled for
  quantized training** (`BuildCompactView` returns false on `use_quantized_grad_`),
  so numerai quant pays for all 504 columns per block even at ff=0.1.
- higgs / epsilon / year are bin-cap bound with healthy `block_dim_y` (18–21) and
  40–52% UB bandwidth → **little JIT headroom** (they are near their effective
  roofline once the real <UB row counts are accounted for).

### Go / no-go per benchmark for phase-2 JIT

- **numerai: GO.** The dominant, latency-bound, column-capped shape. Phase-1
  column-cap regroup already lands a bit-identical ~7% construct win; the JIT can
  push further with a coalesced low-bin layout + baked offsets (below).
- **higgs / epsilon / year: NO-GO for a bandwidth win** — already ~bandwidth-bound.
  A JIT could still shave a few % of address arithmetic / flush-loop bounds, but
  the roofline says the ceiling is low. Deprioritize.
- **fraud / covtype: NO-GO** — too small; launch/other-stage dominated.

## 2. Phase-1 prototype (landed, no NVRTC)

`EXABOOST_CONSTRUCT_COLCAP` + auto-trigger in `DivideCUDAFeatureGroups`
(`cuda_row_data.cpp`): for low-bin (`max_bin_per_col ≤ 32`) many-feature data
where the 504-column cap would bind, cap partitions at **252 columns**
(`block_dim_y = 2`), regrouping columns into more, narrower partitions so each
thread walks 2 rows and overlaps the two scattered reads. Integer-atomic sums are
order-invariant → **bit-identical** (numerai `763c75c0d9cb`, covtype
`5f4e7bdfff1e` locks unchanged). Auto-trigger is a no-op on all bin-cap-bound
benchmarks. Kill switch: `EXABOOST_CONSTRUCT_COLCAP=0`. Measured ~7% construct
kernel / ~2–5% wall on the numerai example. This is the compile-time-templated
proof; the JIT generalizes it.

## 3. What the NVRTC JIT bakes

The JIT compiles a construct kernel specialized to the dataset's exact shape,
turning today's runtime lookups and loop bounds into compile-time constants:

1. **Per-feature bin counts & column→hist offsets** — baked as a `constexpr`
   array (or, for uniform low-bin data like numerai, a single `constexpr BINS`).
   Removes the per-column `column_hist_offsets[column_index]` load and lets the
   compiler size register-histogram arrays and unroll the flush loop exactly.
2. **Feature-group / partition layout** — the chosen `column_cap`, `#partitions`,
   `block_dim_x`, `block_dim_y`, and per-partition column counts baked as launch
   constants and `constexpr` loop bounds (`num_columns_in_partition`,
   `num_items_in_partition`), enabling full unroll of the zero/merge loops.
3. **Coalesced low-bin data layout (the real phase-2 win for numerai)** — for
   ≤4-bit features, JIT a kernel that reads a **row-major packed** tile (all of a
   row's active bins contiguous) so the gather coalesces, instead of today's
   column-major `data + col*num_data` stride. This attacks the 10%-roofline
   scattered-read directly. The packed width is a compile-time constant.
4. **Register-accumulation body for quant** — with baked `constexpr BINS ≤ 8`,
   emit the order-invariant integer-atomic register histogram (bit-identical,
   unlike the float non-quant path) flushed once per thread.
5. **hist bit-width (16 vs 32)** and `IS_4BIT` — baked, dropping the runtime
   branches at 1220–1234 and the `ReadDenseBin` indirection.

## 4. Compile trigger, cache key, fallback

- **Trigger:** first `ConstructHistogramsForLevel` of the first tree (Dataset is
  constructed and partition layout is known). Compile happens once, off the
  training hot path; the ~0.5 s one-time NVRTC compile amortizes over the
  thousands of trees in a run (numerai example: >250 ms/tree, so break-even in ~2
  trees).
- **Cache key (shape signature):** hash of `{bit_type, is_4bit_packed,
  num_feature_partitions, per-partition column counts, column_hist_offsets,
  hist bit width, use_16bit_hist, num_grad_quant_bins, SHARED_HIST_SIZE,
  sm_arch}`. Cache CUmodule/CUfunction in-process keyed by this hash; an on-disk
  cache (keyed by the same hash + driver/NVRTC version) can skip recompiles across
  runs.
- **AOT fallback:** the current AOT-compiled general kernel
  (`CUDAConstructDiscretizedHistogramDenseBatchedKernel`). Used when NVRTC is
  unavailable, the shape is unsupported (sparse, large-bin partitions,
  categorical), the compile fails, or `EXABOOST_CONSTRUCT_JIT=0`. The JIT is a
  **perf-only fast path**; its histogram output must be bit-identical to the AOT
  kernel (same gate locks).

## 5. Integration with the existing dispatch

`LaunchConstructHistogramBatchedKernel` (cuda_histogram_constructor.cu ~2378) gains
a pre-check: if a JIT `CUfunction` is cached for the current shape signature and
enabled, `cuLaunchKernel` it with the same argument pack and the JIT-baked launch
dims; otherwise fall through to the existing templated dispatch unchanged. The
argument ABI matches the current kernel's parameter list so the launch site is a
thin branch. The phase-1 `column_cap` selection stays in `DivideCUDAFeatureGroups`
and feeds the JIT's baked partition plan.

## 6. Gates (unchanged from phase 1)

Every JIT variant must keep the md5 locks: covtype 1023/10 quant `5f4e7bdfff1e`,
numerai int8 `763c75c0d9cb`, plus the RMSE/bagging/graph locks in the session gate
suite. A JIT that changes any md5 is a **bug**, not a feature — the specialization
is perf-only. Kill switch: `EXABOOST_CONSTRUCT_JIT=0` → AOT.
