# Bin-cap construct: shape-specialized shared-mem partition reduction — investigated, negative

## Question

The quantized dense construct kernel splits a dataset's features into
shared-histogram *partitions*. The per-partition bin capacity is
`max_num_bin_per_partition = shared_hist_size_ / 2`
(`src/io/cuda/cuda_row_data.cpp`, `DivideCUDAFeatureGroups`). With the current
`SP_SHARED_HIST_SIZE = 12288` (int16 slots → **24 KB** shared, 6144 bins/part),
a 255-bins-per-feature dataset fits only ~24 features per partition. Wide
datasets therefore span **many** partitions, and each partition's construct
block re-reads the gradient/hessian array for every row it processes.

Hypothesis (this investigation): a *shape-specialized* construct that uses **more
shared memory** to fit **more features per partition** would cut the partition
count → fewer full-row scatter passes and fewer per-partition gradient/hessian
re-reads → faster construct on the bin-cap benchmarks (higgs / epsilon / year /
covtype / fraud), even under a "bandwidth-bound" reading.

**Result: negative.** Built and measured. Partition reduction is *monotonically
slower* on every benchmark where it applies; the two widest benches
(higgs, covtype) are already single-partition (nothing to reduce). The base
kernel is already occupancy/latency-bound near the memory wall; enlarging shared
memory to fuse partitions collapses occupancy far faster than it saves traffic.

## Characterization (per benchmark, `EXABOOST_DUMP_PARTITIONS=1`)

Shape-driven, identical across depth. `max_col_per_part = 24` on the
multi-partition benches confirms they are **bin-capped** (24 × 255 ≈ 6144 cap),
not column-capped.

| bench | rows | feat | partitions | max cols/part | binding | lever |
|---|---|---|---|---|---|---|
| higgs | 10.5M | 28 | **1** | 28 | fits in one part | N/A |
| covtype | 465K | 53 | **1** | 12 | fits in one part | N/A |
| fraud | 228K | 29 | **2** | 24 | bin cap | reduce 2→1 |
| year | 464K | 90 | **4** | 24 | bin cap | reduce 4→2 |
| epsilon | 400K | 2000 | **84** | 24 | bin cap | reduce 84→42→21 |

Only epsilon, year, fraud can even change partition count. higgs and covtype
are already one partition.

## Empirical DRAM roofline (RTX 5090, 170 SMs; `benchmarks/construct_partition_probe/roofline_micro.cu`)

Measured with a streaming copy/read microbench (ncu is blocked on this host):

- **read-only sustained: 1652 GB/s**
- copy (r+w) sustained: 1400 GB/s

(vs the ~1800 GB/s nominal spec). The construct kernel is read-dominated (bin
matrix + scattered gradient reads; the shared→global histogram flush write is
small), so the relevant ceiling is ~**1650 GB/s**.

## Where the traffic is

Per construct launch over P partitions, the DRAM reads are:

- **bin matrix**: each row's feature slice is read **once total** across all
  partitions (partition p reads only its own 24 columns) → `rows × num_feature`
  bytes. **Independent of partition count.**
- **gradient/hessian**: re-read **once per partition** → `rows × P × 4` bytes.
  This is the *only* term that scales with P.

For epsilon (2000 feat, 84 parts) the gh term is ~14 % of traffic; the
partition-independent bin term is ~86 %. So even eliminating *all* gh
amplification caps the possible saving at ~14 % of traffic — and only if the
kernel were bandwidth-bound.

## Built-kernel A/B (`benchmarks/construct_partition_probe/construct_ab.cu`)

A standalone kernel that replicates the discretized construct accumulation body
(int16 packed shared hist, `atomicAdd_block`, 32-bit flush), run on the exact
epsilon root-level shape (400K rows, all in the leaf), sweeping cols-per-partition:

| variant | cols/part | partitions | shared | kernel time |
|---|---|---|---|---|
| **SP (base)** | 24 | 84 | 24 KB static | **0.79 ms** |
| BIG48 | 48 | 42 | 48 KB static | 1.05 ms (**+32 %**) |
| DYN96 | 96 | 21 | 96 KB dynamic | 1.71 ms (**+2.2×**) |

The standalone base time (0.79 ms) matches the real in-training construct
(0.79–0.95 ms/level from nsys), so the reproduction is faithful. Partition
reduction is **monotonically slower**: bigger shared mem → fewer blocks per SM
(occupancy: ~4 → 2 → 1 block/SM at 24/48/96 KB) → the scattered gh-read + atomic
latency can no longer be hidden. The ~14 %-of-traffic gh saving is dwarfed by
the occupancy loss.

Same trend on the narrower benches (fuse to the minimum partition count):

| bench | base | fused | Δ |
|---|---|---|---|
| year (90 feat) | 4 parts, 0.047 ms | 2 parts, 0.048 ms | flat / slightly slower |
| fraud (29 feat) | 2 parts, 0.016 ms | 1 part, 0.022 ms | **+37 %** |

## Roofline evidence (base kernel is already near the wall)

At the well-provisioned root level (all 400K rows, full occupancy) the base
construct moves `rows × feat` bin bytes + `rows × 84 × 4` gh bytes ≈ 0.94 GB in
0.79 ms = **1179 GB/s = 71 % of the 1652 GB/s empirical read roofline**. That is
the *best-case* level; deeper, smaller leaves run at lower utilization
(whole-tree average ~27–54 % of roofline). There is only ~29 % headroom at the
root, and partition reduction spends it on occupancy loss rather than recovering
it — the A/B confirms this directly.

## Build note (why not ship an SP2 static variant)

A 48 KB *static* `__shared__ int16_t[24576]` construct kernel does not build in
this toolchain: the whole-program device link (`nvcc -dlink` with static NCCL)
fails with `nvlink fatal: Internal FNLZR error`, independent of the float
non-quant kernels (which are 96 KB at that size and separately over the 48 KB
static ceiling). The only viable route past 24 KB is **dynamic** shared memory
(`extern __shared__` + `cudaFuncSetAttribute`), which the A/B above uses — and
which is *slower*. No production change is warranted.

## Verdict

| bench | verdict |
|---|---|
| higgs | already 1 partition — lever N/A |
| covtype | already 1 partition — lever N/A |
| fraud | partition reduction measured **+37 %** slower |
| year | partition reduction measured flat/slower |
| epsilon | partition reduction measured **+32 %** (48 KB) to **+2.2×** (96 KB) slower; base at 71 % of empirical roofline at root |

The shared-mem partition-reduction lever is a measured net loss across all
bin-cap benchmarks. The base construct is occupancy/latency-bound near the
memory wall; the win does not exist. Kept as AOT; no kernel shipped.
