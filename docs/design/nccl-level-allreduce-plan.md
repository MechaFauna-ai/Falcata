# NCCL multi-GPU: hybrid level flow + one grouped all-reduce per level

**Status:** design for implementation during the phase-1 rental · **Date:** 2026-08-02
**Prereq:** task #68 phase 1 (verify the existing classic NCCL path on a rented
2×GPU pair) — the runbook lives in `scripts/nccl-rental/` in the bench workspace.

## Current state

NCCL training uses the CLASSIC per-split loop only (`HybridGrowthUsable()`
excludes `nccl_communicator_`): one `ncclAllReduce` of the smaller leaf's
histogram per sibling pair (`NCCLReduceHistogram`, learner:3514), i.e. two
network round-trips per split plus the per-split host syncs. On PCIe-staged
consumer pairs the latency of ~2·num_leaves round-trips per tree dominates;
on NVLink it wastes less but still serializes.

## Target

Enable the hybrid two-sync level flow under NCCL with ONE grouped
communication per level:

1. **Grouped histogram all-reduce.** After `ConstructHistogramsForLevel`
   (before `FindBestSplitsForLevel`): `ncclGroupStart()`, one
   `ncclAllReduce` per pair's smaller-leaf histogram region (regions are
   per-leaf slots in the hist pool — not contiguous, so a group of N reduces
   rather than one big one; NCCL fuses the group into one launch/network
   phase), `ncclGroupEnd()`, all on `nccl_stream_`, with a stream-event fence
   into the finder's stream. Quantized: per-leaf bit width selects
   int32/int64 region + dtype exactly as the classic branch does.
2. **Global leaf counts per level.** The finder's validity gates and the
   discretizer's bit-width choices must use GLOBAL counts. Today the level
   descs carry local `leaf_num_data_`. Add: after `FinishSplitBatch`, one
   small host `Network::GlobalSum` (or nccl all-reduce of a 2·num_splits int
   buffer) of the level's child counts, feeding `global_num_data_in_leaf_`;
   `EnqueueLevelBestSplitSearch` then fills `desc.num_data_in_*` and the
   validity gates from the global values (mirroring what
   `EnqueuePairBestSplitSearch` does with `global_num_data_in_leaf_`).
3. **Root init.** Root sums already all-reduce in leaf-splits init (classic
   path machinery, reused).
4. **Fences kept:** selective flow, one-sync, graph, categoricals stay
   excluded under NCCL for v1 (plain two-sync level prefix + classic tail
   only). Lift later as measured.

## Validation plan (phase-1 rental, consumer pair)

- 1-rank smoke locally first: NCCL with nranks=1 must be a bit-exact no-op
  against the single-GPU run (cheap regression harness before any rental).
- 2-rank correctness: covtype + numerai-probe configs, quality parity vs
  single-GPU (NOT bit-parity: row partitioning differs), sane-model gates.
- Measure: per-level comm time from nsys (nccl kernels), classic-vs-level
  reduce count and wall time. The LATENCY win (fewer round trips) shows
  stronger on PCIe than NVLink; the BANDWIDTH story needs the phase-2 NVLink
  session (2×A100 SXM) for numerai-deep 2-GPU vs 1-GPU scaling.

## Cost/infra

- Fat wheel: add `compute_80` (A100) to the current 89+120 set;
  `build-python.sh bdist_wheel --cuda` with CMAKE_CUDA_ARCHITECTURES
  "80;89;120" (see runbook).
- Phase 1: 2×RTX 4070S Ti / 4090 pair, ~$0.40–0.70/hr, budget ~3h.
- Phase 2: 2×A100 SXM (NVLink), ~$1.5–2.5/hr, budget ~1h.
