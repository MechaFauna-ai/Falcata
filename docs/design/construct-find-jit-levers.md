# Bin-cap JIT levers (lean construct / JIT find / train A/B) — investigated, negative

## Question

The NVRTC construct JIT wins ~2.5x on numerai (the *compact* column-view path,
feature_fraction 0.1). The five bin-cap benchmarks (higgs, covtype, year, fraud,
epsilon) run the **full-column** quantized dense construct — which the compact
JIT never fires on (`use_compact_view_` is false when all features are used, so
`TryLaunchConstructJITBatchedCompactQuant` is not even reached). Phase 4 already
showed shared-mem **partition fusion** loses. This phase tests the three
remaining JIT levers on the bin-cap benches with real train-time measurement.

**Result: all three negative.** No shippable bit-identical JIT train win exists
on the bin-cap benches. The construct is memory-bound at ~94% of the hardware's
*achievable* occupancy, and the find kernel is not occupancy-bound. Measured
below.

The one code change this investigation ships is unrelated to a perf win: a
**stream-capture guard** on the JIT self-test (it crashed
`cuda_plan=auto,construct_jit:on` + the multiclass graph loop). See the last section.

## Where train time actually goes (nsys, RTX 5090, per-kernel %)

Quantized dense path, `construct_jit:off` (default AOT):

| bench (regime) | construct | find | apply* | note |
|---|---|---|---|---|
| higgs deep (1023/10) | **38.5%** | 4.4% | 43.5% | construct-heavy |
| epsilon shallow (63/6) | **52.3%** | 21.7% | ~1% | construct-heavy |
| year deep (1023/10) | 22.6% | **50.8%** | 5.7% | find-heavy |
| covtype deep (1023/10) | 16.1% | **24.4%** | 35.5% | find-heavy |

*apply = `HybridGenBitVectorUpdateLeafIndexBatchKernel` +
`HybridSplitInnerBatchKernel` (data partitioning, not a JIT construct/find lever).

Takeaway: construct dominates only higgs/epsilon; **find** dominates year and
covtype. So the levers were measured against the kernel that actually matters
per bench.

## Occupancy is already near the hardware ceiling (not register/shared-bound)

`cuobjdump -res-usage` on the shipped AOT construct kernel:
`CUDAConstructDiscretizedHistogramDenseBatchedKernel<uint8,12288,*>` →
**REG:40, SHARED:25600 B**. Real launch: `block_dim_x =
max_num_column_per_partition` (24–53), `block_dim_y = 504 / block_dim_x`, so
**tpb ≈ 477–504**.

The decisive hardware fact (from `cudaGetDeviceProperties`): RTX 5090 sm_120 has
**maxThreadsPerMultiProcessor = 1536** (48 warps/SM), *not* 2048. With
tpb ≈ 504, the occupancy limiter is **threads per SM**:
`1536 / 504 = 3 blocks/SM`. `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
confirms 3 blocks/SM for the real config. Registers (40 → allows more) and
shared (24.6 KB → 4 fit in 100 KB by shared) are *not* the binding constraint.

So achieved occupancy = 3 × (504/32) = **45 of 48 warps = 94%** (covtype 42/48 =
88%, its tpb=477 doesn't tile 1536 evenly). Raising a dynamic-shared carveout,
cutting registers, or baking constants **cannot** add a 4th block — the thread
cap already binds at 3. This is the first-principles reason Lever 1 has almost no
headroom, and it is *not* what phase 4 tested (phase 4 tested partition fusion).

## Lever 1 — lean JIT construct (bake BINS/COLS/row_stride): NEGATIVE

Standalone A/B (`benchmarks/construct_find_jit_probe/lean_construct_ab.cu`):
runtime-offset kernel (mirrors the AOT body: int16 packed shared hist,
`atomicAdd_block`, 32-bit flush) vs a **lean** kernel with `BINS`, `COLS`, and
`row_stride` baked as compile-time constants, uniform `col*BINS` offsets, and
known loop bounds. Real root-level shapes, median of 3, compute-sanitizer clean:

| shape | parts | cols/part | runtime ms | lean ms | speedup |
|---|---|---|---|---|---|
| higgs   | 1  | 28 | 0.0369 | 0.0375 | **0.98x** |
| covtype | 1  | 53 | 0.0350 | 0.0348 | **1.00x** |
| epsilon | 83 | 24 | 0.484  | 0.561  | **0.86x** |

Baking constants is **neutral-to-slower**. The lean kernel compiles to 43
registers vs the runtime kernel's 40 (removing offset-array loads does not
shrink the register file, and the extra `constexpr` index math / any unrolling
raises pressure). The inner accumulation loop is pure DRAM traffic — one
`grad_and_hess` read + one bin-matrix byte per (row, column) — which baking does
not reduce. At fixed 94% occupancy there is nothing to recover. epsilon's 84
tiny partitions make the (unchanged-traffic) lean kernel measurably worse.

Combined with phase 4 (partition fusion monotonically slower; construct at 71%
of the 1652 GB/s empirical read roofline at the root, occupancy-bound below):
**construct is exhausted** on the bin-cap benches.

## Lever 2 — JIT / specialized find: NEGATIVE

`FindBestSplitsDiscretizedForLevelKernel` → **REG:60, SHARED:9004 B**, launched
`<<<(num_features, num_pairs, 2), 256>>>`. 60 registers limit it to **4
blocks/SM** (65536 / (60×256 rounded) ). The JIT-find hypothesis is that baked
`num_bin` + fewer registers → more blocks/SM → faster. Tested directly by
forcing the compiler to fit more blocks with `__launch_bounds__`:

| variant | find regs | year deep find | covtype deep find |
|---|---|---|---|
| baseline | 60 | 146.5 µs avg | 22.4 µs avg |
| `__launch_bounds__(256,5)` | 48 (+48 B stack) | 143.7 µs | 21.9 µs |
| `__launch_bounds__(256,6)` | 40 (+64–88 B stack) | 145.5 µs | 25.5 µs |

Forcing 5–6 blocks/SM (up from 4) leaves find time **flat, or worse** — the
register cut spills to local memory, and the added local traffic cancels the
occupancy gain. **Find is not occupancy-bound.** It is bound by its intrinsic
serial work per block: the 256-wide `ShufflePrefixSum` over the histogram and
the double-precision split-output block, which run at warp granularity
regardless of resident-block count. Baking `num_bin` cannot remove the prefix
sum or the fp64 output; the block is already right-sized (256 ≥ max_bin 255), so
there are no wasted threads to reclaim. covtype's find is fed by 2310 tiny
launches; year's biggest launches are `(90, 242, 2)` = 43 K blocks already
saturating the device — neither is helped by per-shape ISA.

## Lever 3 — full train A/B (JIT all-on vs off): no-op on bin-cap benches

Because the JIT only ever launches on the compact path, `construct_jit:on`
vs `construct_jit:off` is a **bit-identical no-op** on the bin-cap benches (same md5, same
train time within noise): higgs 63/6 `ad3a467e47a7`, covtype 1023/10
`c5107e1c48de`, epsilon 63/6 `191a53630c11` under both settings. There is no
specialized bin-cap kernel to A/B because Levers 1 and 2 produced none worth
shipping (both measured neutral-to-negative). The compact JIT win on numerai is
unchanged (`763c75c0d9cb`, JIT on == off).

## Verdict

| bench | construct lever | find lever | net |
|---|---|---|---|
| higgs   | 0.98x (neg) | find only 4.4% of train | exhausted |
| covtype | 1.00x (flat) | find not occ-bound (flat) | exhausted |
| year    | flat (phase 4) | find not occ-bound (flat) | exhausted |
| fraud   | +37% slower fused (phase 4) | sub-0.05 s, in noise | exhausted |
| epsilon | 0.86x (neg) | find not occ-bound (flat) | exhausted |

**No bit-identical JIT train win exists on the bin-cap benches.** Construct is
memory-bound at ~94% achievable occupancy / 71% read roofline; find is
serial-work-bound, not occupancy-bound. Both are measurably exhausted across the
two kernels that matter. The goal's bin-cap speed-up is physically bounded by
DRAM bandwidth (construct) and per-block serial reduction latency (find), which
no shape-specialized codegen changes.

## The one shipped change: JIT self-test stream-capture guard

`cuda_plan=auto,construct_jit:on` combined with the multiclass CUDA graph loop aborted
with `operation not permitted when stream is capturing`. Cause: the one-time JIT
self-test (NVRTC compile + `cuLaunchKernel` + `cudaMemcpy`) ran off the *first*
construct, and when that first construct is captured into the graph, those
stream ops are illegal mid-capture. Fix
(`cuda_histogram_constructor.cu`): skip the self-test while
`cudaStreamIsCapturing` reports active capture and defer it to the first
non-capturing construct. The JIT declines under graph capture anyway
(`hybrid_graph_capture_gstate_ != nullptr` → AOT), so deferring never loses a
live launch. With the guard, all 23 non-`matches_cpu` dual tests pass with
`cuda_plan=auto,construct_jit:on`, and every lock is unchanged (covtype
`5f4e7bdfff1e` / `fcb9f6c2ab87`, numerai int8 `763c75c0d9cb`).
