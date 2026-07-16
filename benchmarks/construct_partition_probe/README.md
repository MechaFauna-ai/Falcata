# Construct partition-reduction probe

Standalone microbenchmarks backing
`docs/design/construct-partition-reduction.md` (bin-cap construct: shared-mem
partition reduction — investigated, negative).

## `roofline_micro.cu`

Empirical DRAM bandwidth of the GPU (streaming copy + read), used to establish
the real roofline the construct kernel is compared against (ncu is unavailable
on the target host).

```
nvcc -O3 -arch=sm_120 roofline_micro.cu -o roofline_micro && ./roofline_micro
```

On an RTX 5090 (170 SMs): read-only ~1652 GB/s, copy ~1400 GB/s.

## `construct_ab.cu`

Replicates the quantized discretized construct accumulation body (int16 packed
shared histogram, `atomicAdd_block`, 32-bit flush) and sweeps
cols-per-partition (24 / 48 / 96 / 128 → 84 / 42 / 21 / 16 partitions) on the
epsilon root-level shape, timing the kernel each way. Optional args
`<num_data> <num_feature>` reproduce other bench shapes (e.g. `463715 90` for
year, `227845 29` for fraud).

```
nvcc -O3 -arch=sm_120 construct_ab.cu -o construct_ab && ./construct_ab
```

Result: partition reduction is monotonically slower (bigger shared mem → fewer
blocks/SM → occupancy collapse outweighs the ~14 %-of-traffic gradient re-read
saving). See the design doc for the full table and verdict.
