# construct/find JIT-lever probes

Standalone reproducers for `docs/design/construct-find-jit-levers.md` (the
negative result for the three JIT levers on the bin-cap benchmarks). Not built
into the library; excluded from cpplint (compact benchmark style).

- `occ_sweep.cu` — queries `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and
  the device limits for the real construct launch config. Shows the sm_120
  occupancy limiter is **threads/SM (1536)**, not shared/registers: 3 blocks/SM
  (~94% of achievable) regardless of shared-hist size.

  ```
  nvcc -arch=sm_120 -O3 occ_sweep.cu -o occ_sweep && ./occ_sweep
  ```

- `lean_construct_ab.cu` — A/B of the quantized discretized construct: a
  runtime-offset kernel (mirrors the AOT body) vs a **lean** kernel with
  BINS/COLS/row_stride baked as compile-time constants, on real root-level bench
  shapes. Lean is neutral-to-slower (higgs 0.98x, covtype 1.00x, epsilon 0.86x).

  ```
  nvcc -arch=sm_120 -O3 --std=c++14 lean_construct_ab.cu -o lean_ab && ./lean_ab
  ```

Find-lever evidence (`__launch_bounds__` A/B on the real
`FindBestSplitsDiscretizedForLevelKernel`) is in the design doc; it is an
in-tree edit + nsys measurement rather than a standalone kernel.
