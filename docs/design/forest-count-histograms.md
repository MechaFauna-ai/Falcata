# Count histograms for a random forest (forest stage 3)

Status: design note and implementation (cuda_plan key `forest_count_hist`, default on), 2026-10-16.
Plan: kernel-evo `runs/forest/PLAN_batched_forest.md` §3 stage 3. Numbers: one RTX 5090, relaxed (desktop on),
master 065ac1eb (after #115/#116 `forest_batched`) unless a line says otherwise.

## 1. Where a quant-none forest spends its GPU time

After stage 2 (forest slots) the covtype forests are host/launch-bound, but the non-quantized deep forests are
GPU-bound in one kernel. nsys over 12 rounds of each cell shape on 065ac1eb (kernel-evo
`runs/forest/stage3/nsys/*_none_base`, `cuda_gpu_kern_sum`):

| shape (depth 12, 4095 leaves, min_data 1, bagging 0.632, √F per node, quant none) | kernel time / tree | the deterministic batched construct | its merge | the level finder | everything else |
|---|---:|---:|---:|---:|---:|
| higgs 3.2M × 28, binary | 83 ms | 77.1 ms (92.5 %) | 3.1 ms (3.7 %) | 1.0 ms (1.2 %) | 2.6 % |
| numerai 400k × 3555 (5 bins, 4-bit rows), regression | 91 ms | 72.4 ms (79.4 %) | 3.8 ms (4.2 %) | 3.7 ms (4.0 %) | 12 % (one-off setup: the bin chunk copy, cat vectors) |

Same shapes, 30 rounds, ms per round in the second half (`runs/forest/stage3/time_t0.jsonl`): higgs d12 quant none
63.4, stochastic 3.0, fixedpoint 3.3; numerai 94.1 / 24.6 / 24.9; covtype d12 (7 trees per round) 29.8 / 8.2. The
quantized forests are 3.6-21x faster per round, and what separates them is the histogram construct.

`CUDAConstructHistogramDenseGMDeterministicBatchedKernel` is the non-quantized construct of a hybrid level when
`DetDenseBatchedEligible`: each (pair, partition, row tile) block owns `dy` private "slot rows" of doubles in global
memory, zeroes all of them, and every (row, column) does a non-atomic read-modify-write of two doubles there; a second
kernel sums the slot rows in a fixed order into the leaf histogram. It exists to make fp64 histograms independent of
the order of the float atomics: "float32 gradients summed in double are exact, which makes the histogram bit-equal to
the CPU reference". The price is global-memory traffic per (row, column) instead of shared-memory atomics.

## 2. What a forest's gradients are

`RF::Boosting` computes the gradients once, from the init score (`BoostFromAverage`), with the objective's kernel
(`rf_device_objective`, #103); every tree of the forest reads the same arrays (`gradients_pointer_ + class * num_data`),
and the learner gets a device pointer that does not change during training. Every row has the same score, so:

- binary logloss: two gradient values (one per label), one hessian value (`|r| (σ - |r|)` of the common score);
- multiclass softmax / one-vs-rest: per class tree, two gradient values (`p_k - [y = k]`, scaled) and one hessian;
- regression L2: one gradient value per distinct label (`s0 - y`), hessian 1. numerai's 5 label values give 5
  gradient values; higgs and covtype labels give 2 per tree.

Sample weights (or GOSS, which rescales gradients per round) multiply rows by different factors: many values,
non-constant hessian.

## 3. The mechanism

**Per-row codes, built once per gradient array.** On the first tree that trains on a gradient array the learner reads
the array and its hessians back (one copy each, once per class for the whole training) and checks, on the host:

1. every hessian is the same finite positive float `h`;
2. the gradients take at most 256 distinct finite values `v_0 < ... < v_m`.

It then writes every value as an integer multiple of one power of two: `u = 2^L` with `L` the lowest set mantissa bit
over the nonzero values (`v_j = M_j u`, `M_j` an integer), takes `a = v_0`, `g = gcd(M_j - M_0)`, `b = g u`, and the
value table `t_j = (M_j - M_0) / g` (so `v_j = a + b t_j` exactly). Each row gets the one-byte index `j` of its value;
the codes (one byte per row) and the table go to the device and are kept for the training (a cache keyed by the
gradient pointer, shared with the forest slots, cleared when the training data, config or learner is reset).

**The construct.** Where the learner would launch the deterministic construct (per leaf, compact view included, or
batched per level), it launches the count construct instead: one block per (pair, partition, row tile) accumulates,
in shared memory with integer atomics, per bin the row count `n` and the code sum `S = Σ t_j` of its rows, and adds
its nonzero bins to a per-pair global integer scratch (integer atomics: any order, same sums). Two layouts:
`packed` (one 64-bit word `S << 32 | n`, one atomic per (row, column); used when `max t · num_data < 2^32`: binary and
multiclass have `t ∈ {0, 1}`) and `wide` (64-bit `S`, 32-bit `n`; numerai's table spans 2^33). A convert kernel then
writes each bin of the pair's leaf histogram as the fp64 pair the deterministic construct would have written,
`G = a·n + b·S` and `H = h·n`, and zeroes the scratch it read (so the next level starts from zero without a memset).
The per-row read is one code byte instead of two floats.

**Everything downstream is unchanged.** The leaf histogram is the same fp64 (grad, hess) pair array, so the fix of the
most-frequent bin, the parent − child subtraction, the level finder, the leaf sums and the leaf outputs run the code
they run today on the same bits. The finder does not compute gains from counts; with constant hessian it could, but
the profile gives the finder 1-4 % of kernel time, and keeping its input bit-identical is what makes the model
identical (§4) for free. Downstream histogram bytes are not halved (the scratch is 8 B per bin packed, 12 B wide,
against the det construct's 16 B per bin per slot row and `dy × tiles` slot rows per pair).

## 4. Exactness: the same histogram bits, hence the same model

Let `B = max_j |M_j| · num_data` (an upper bound, in units of `u`, of any partial sum of any bin of any leaf).

- **The deterministic construct's histogram is the exact sum when `B < 2^53`.** Every addend `v_j` and every partial
  sum is an integer multiple of `u` of magnitude below `2^53 u`, so every double addition is exact; the merged value
  is the true `Σ v`, in any order. Same for `Σ h = h n` (`h` has a 24-bit significand, `n < 2^29`).
- **The count construct gives the same double when `2B < 2^53`.** `a·n = M_0 u n`: |M_0 n| ≤ B, exact. `S ≤ (max_j
  (M_j − M_0) / g) · n`, so `S < 2^53` is an exact double and `b·S = u · Σ (M_j − M_0)` has magnitude ≤ 2B u: exact.
  Their sum is `u Σ M`, of magnitude ≤ B u, exactly representable, so the rounded sum is exact. `h·n` exact as above.
  A zero sum is +0.0 in both (the merge starts from +0.0; the convert adds +0.0 to normalize a −0.0).
- The learner engages the count construct only when the check passes (`2B < 2^53`, `num_data < 2^29`, at most 256
  values, constant hessian); otherwise the tree keeps the deterministic construct, with a diag line saying why.

So the count construct is an exact re-evaluation of the same sums: every histogram bit equal, every gain, tie-break,
split, count and leaf value equal, and the model is md5-identical to the key-off model (not "the same gain ordering
up to ties": the same gains). On the three forest datasets the bound holds with room: higgs (`v` ≈ ±0.5, `L` = −26,
2.0M bagged rows) needs 2B ≈ 2^48; covtype's rarest class (`p` ≈ 0.005, `L` ≈ −31, 294k rows) ≈ 2^51; numerai's five
values (`L` = −34 from `s0 − 0.5` ≈ −2.5e-4) ≈ 2^52. Rows of data outside the bound (a gradient value much smaller
than the largest, many rows) keep the deterministic construct, which is then not exact either.

**Determinism.** Integer sums do not depend on the order of the atomics, and the convert is a function of them: the
count construct is run-to-run deterministic by construction, with no slot rows and no fixed-order merge.

## 5. Which trees qualify (the learner's gate)

`forest_count_hist` on (default), and:

- `boosting=rf`, objective `binary`, `multiclass`, `multiclassova` or `regression` (L2), the device objective
  (`boosting_on_cuda_`: the gradient pointer is the objective's array, constant for the training), no GOSS, no
  dataset weights, one GPU, scalar leaves;
- not quantized (`use_quantized_grad` keeps its own integer pipeline; the quantized forests are not touched);
- the deterministic construct is the one in use: dense row data, no large-bin partition, no fp32 histogram mode,
  not under graph capture (the graph loop keeps its own construct nodes);
- the data check of §3/§4 passes for the tree's gradient array.

Every other tree, gbdt always, runs exactly as before; with the key off (`cuda_plan=auto,forest_count_hist:off`) every
forest runs the stage-2 construct.

## 6. Memory and launches

Codes: one byte per row per trained class (higgs 3.2 MB; covtype 7 × 465k = 3.3 MB). Scratch: one row of
`num_total_bin` words per pair, reused per level, sized for the level's pairs up to 64 MiB (higgs packed 57 KB per
pair: ~1170 pairs per chunk; numerai wide ~213 KB: ~300 pairs per chunk; a level with more pairs runs in chunks), plus
one row per histogram pipeline for the per-leaf construct. The det slot slab (`kDetDenseSlotBudget`, 96 MiB per
learner) stays allocated (the fallback). Launches per level: construct + convert (as construct + merge before).

## 7. The init-score column (plan: "drop the per-tree init-score column handling rf does not need")

After #103 (`rf_lazy_train_score`) no score column is updated per tree during training. What remains is setup:
`RF::Boosting` fills a host vector of `num_class × num_data` doubles with the init scores and uploads it, once for the
gradients and once more for the objectives that renew leaf outputs (L1, quantile: `cuda_init_score_`). On the device
path both columns are now filled on the device (a fill kernel per class), without the host vector and the uploads.
Same values, same gradients.

## 8. Not done here

- Gains from counts in the finder (binary: from `(n, n_pos)` per bin) and histograms of 8 B per bin downstream: not
  needed for the construct-bound cells (§1); it would change the gain arithmetic and so the tie-breaks.
- Quantized forests: an exact count histogram would replace their rounding noise (and change their models); not this
  stage.
- Other objectives with few gradient values and a constant hessian (L1, quantile, huber at a constant score,
  cross-entropy): the gate lists the objectives the plan names; the check itself is objective-agnostic.
