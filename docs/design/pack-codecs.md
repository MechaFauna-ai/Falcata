# Packing codecs for the compact column view: design, measurements, conclusion

Status: implemented (plan keys `pack_bit3` / `pack_radix5` / `pack_radix6` /
`pack_radix7`, all default OFF); measured on numerai-deep. This document is
the record of the design, the bugs the gates caught, and the measured verdict.

## The idea

The per-tree compact column view stores bins as nibbles (2 values/byte) even
though numerai-class features hold only 5-7 distinct values (log2(6) ~ 2.6
bits). Packing tighter cuts the bytes the histogram-construct kernels read on
every level pass:

| codec | layout | values/byte | bins cap | extraction |
|---|---|---|---|---|
| nibble4 (baseline) | 2 x 4-bit / byte | 2.00 | 16 | shift+mask |
| bit3x32 | 10 x 3-bit / uint32 | 2.50 | 8 | shift+mask |
| radix7x32 | 11 base-7 digits / uint32 (7^11 < 2^32) | 2.75 | 7 | magic division |
| radix6x32 | 12 base-6 digits / uint32 (6^12 < 2^32) | 3.00 | 6 | magic division |
| radix5x32 | 13 base-5 digits / uint32 (5^13 < 2^32) | 3.25 | 5 | magic division |

## The abstraction

`include/Falcata/cuda/pack_codecs.hpp`: each codec is a compile-time policy
struct (`RowBytes`, `Extract`, `Prepare`/`ExtractAt`, `PackRow`, magic tables
as constexpr). Kernels template on the codec (`class PACK`), dispatched once
per launch — the same pattern as the existing `BIN_TYPE`/`SHARED_HIST_SIZE`
parameters. Scope of this cut: the QUANTIZED batched compact chain (fill
transcode + discretized batched construct); the split readers fall back to
the original nibble row data when a codec view is active; the NVRTC JIT
declines codec views. Codec choice is UNIFORM per tree: eligible only when
every sampled column's value bound fits the codec.

Magic division: divisor d = R^i with shift = 32 + ceil(log2 d) is exact for
every 32-bit dividend unconditionally (error < d <= 2^L); the 33-bit
multiplier is evaluated with a split multiply. Roundtrips are proven at
compile time (static_assert over every digit position with worst-case
neighbor digits) and at runtime by 200k random-row host tests per codec.

## Three bugs the measurement phase caught (in order)

1. **Eligibility must bound COLUMN values, not feature bins.** With EFB, one
   column can carry a multi-feature bundle whose stored values exceed any
   single feature's bin count. The correct bound is the column's hist-offset
   delta — the same quantity the 4-bit row-data gate uses. The original
   per-feature check let a bundled column's value >= R into a radix word,
   carry-corrupting every neighboring digit (caught as an md5 mismatch on
   real data at 1M rows; invisible on synthetics with clean columns).
2. **Runtime-indexed constexpr tables live in thread-local memory.** The
   first `Extract(row_ptr, col)` looked up magic constants per row per value;
   nvcc materializes a dynamically-indexed constexpr array in local memory,
   and the reread made engaged radix codecs ~19x SLOWER at 5.4M-row scale.
   Fix: `Prepare(col)` resolves the per-column constants once per thread
   (the digit index is loop-invariant) into a register-resident Cursor;
   the fill kernel replaced its table with a running-multiplier accumulator.
3. **Silent disengagement masquerading as a win.** Real numerai columns have
   spans {<=5, 6, 7} and every ff=0.1 tree samples at least one span-7 column,
   so radix5/radix6 (caps 5/6) never engage at the real workload — their
   early "+11%" was run-to-run noise on the identical nibble path. The
   `FALCATA_DEBUG=diag` engagement line (`[pack-codec] tree codec=...`)
   exists so this cannot be misread again. radix7 (cap 7) is the codec that
   actually engages on numerai.

## Measurements (numerai-deep canonical config, 1224-era cache: 5.43M rows x
3555 features, ff=0.1 => ~355 sampled columns/tree, 300 rounds, median of 3)

All engaged codecs are BIT-IDENTICAL to the nibble baseline (same tree md5),
as the lossless-packing contract requires.

| variant | engaged? | trees/s | vs matched baseline |
|---|---|---|---|
| nibble4 (full default) | — | 17.5–19.0 | +9% (packed split reads) |
| matched baseline (split_packed_read:off) | — | 16.0–17.5 | 1.00x |
| bit3x32 (1.25x fewer bytes) | yes | 15.6 | −2.5% |
| radix5x32 | NO (span-7 columns) | = nibble | n/a |
| radix6x32 | NO (span-7 columns) | = nibble | n/a |
| radix7x32 (1.375x fewer bytes) | yes | 12.7 | **−27%** |

Synthetic control (200k x 60, clean 6-value columns, single partition,
shallow trees): radix6 engaged, bit-identical, 2.2x FASTER than nibble —
the codec machinery works and can win when construct cost dominates and the
shape is small; the question is the real deep workload.

## Conclusion

**Honest negative at the target workload.** At numerai-deep scale the
discretized construct kernel is ISSUE/ALU-sensitive, not read-bound: nsys
attributes the entire radix7 loss to the hist kernel itself (~2x its nibble
time, ~41ms vs ~20ms per tree), while the codec fill is cost-neutral. Two
magic divisions per value double the kernel; even bit3's near-free two extra
ops cost 2.5% against a 1.25x byte saving. Extraction cost scales the kernel
linearly; byte savings buy almost nothing — consistent with the earlier
finding that compact-view reads at nibble density are already cheap relative
to the shared-memory atomic accumulation.

Decisions:
- All four pack keys stay DEFAULT OFF. The abstraction and codecs remain in
  the tree, gate-covered (39/39 lattice green, bit-identity enforced), as
  infrastructure and as a documented negative.
- Mixed-radix partitioning and codec-aware split readers are NOT worth
  building for this workload class and should not be revisited unless the
  shared-atomic bottleneck is first removed (at which point reads could
  become marginal again).
- The synthetic result (2.2x on small single-partition shapes) suggests a
  possible niche for construct-dominated small-model regimes; nothing
  currently exercises that niche.

## Productionization notes (if ever pursued)

- Mixed radix per PARTITION is the real design: group columns by span at
  partition build (5-value columns → radix5, 6/7 → radix6/7, bundles → nibble)
  and launch each partition group with its codec instantiation. Removes the
  all-or-nothing per-tree eligibility that disengages radix5/6 on numerai.
- Split readers would need codec-aware packed column views to avoid the ~9%
  fallback penalty.
- The JIT could bake magic constants as literals per shape.
