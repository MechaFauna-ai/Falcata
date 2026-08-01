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
 * comma-separated expert overrides, e.g. ``auto,graph_loop:off,construct_jit:on``.
 * Values: ``on``/``off`` (also ``1``/``0``, ``true``/``false``). The separator
 * is ``:`` (not ``=``) because the surrounding parameter string already uses
 * ``=`` -- a nested ``=`` would be split by the parameter tokenizer.
 *
 * Fields WITHOUT an override key are measured-invariant winners: they won on
 * every benchmarked shape and are baked; the bool remains only as a one-line
 * debugging lever for developers.
 */
struct FalcataPlan {
  // --- shape-conditional decisions (cuda_plan override keys) ---------------
  // hybrid level-batched growth (off = classic one-split-at-a-time leaf-wise)
  bool hybrid = true;               // key: hybrid
  // CUDA-graph level loop for the hybrid prefix (helps shallow trees; the
  // fixed controller latency turns net-negative on deep configs)
  bool graph_loop = true;           // key: graph_loop
  // graph level loop also for quantized training (opt-in; less measured)
  bool graph_quant = false;         // key: graph_quant
  // per-tree compact column view for quantized construct at any
  // 0 < feature_fraction < 1 (win scales with the excluded fraction:
  // ~3.4x at ff=0.1, ~1.1x at ff=0.6)
  bool compact_quant = true;        // key: compact_quant
  // NVRTC runtime-JIT construct kernels (self-test-then-promote; AOT
  // fallback). Default ON for quant runs >= 300 rounds (the ~230ms one-time
  // compile amortizes): measured +4.0% numerai-deep, +2.4% covtype-deep,
  // +2.2% year, +0.7% higgs, bit-identical (2026-08-01, post-ldcs sync).
  bool construct_jit = true;        // key: construct_jit
  // true when the user wrote construct_jit:on/off -- bypasses the >=300
  // rounds auto-gate (mirrors tuner_explicit)
  bool construct_jit_explicit = false;
  // dense row-data build from column bins (skips host multi-val bin)
  bool fast_rowdata = true;         // key: fast_rowdata
  // 4-bit packed row data for <=16-bin features
  bool rowdata_4bit = true;         // key: rowdata_4bit
  // GPU dense-matrix binning during dataset construction
  bool gpu_construct = true;        // key: gpu_construct
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
  // experimental compact-view packing codecs (uniform per tree; eligible only
  // when every sampled feature's bin count fits the codec). Bit-identical by
  // construction (lossless packing). Priority radix5 > radix6 > bit3.
  bool pack_bit3 = false;           // key: pack_bit3 -- 2.5 values/byte, <=8 bins
  bool pack_radix5 = false;         // key: pack_radix5 -- 3.25 values/byte, <=5 bins
  bool pack_radix6 = false;         // key: pack_radix6 -- 3.0 values/byte, <=6 bins
  bool pack_radix7 = false;         // key: pack_radix7 -- 2.75 values/byte, <=7 bins
  // L2 persistence window on the per-row scattered-reread buffers (grad/hess,
  // leaf data indices): each level's bin-matrix stream otherwise evicts them.
  // Cache hint only -- bit-identical by construction. Measured 2026-07-31:
  // +2.8% numerai-deep/covtype-deep, neutral on year at real run lengths.
  bool l2_policy = true;            // key: l2_policy
  // one-time column-major copy of the packed bin matrix as the compact-fill
  // gather source: fill reads contiguous columns (~10x less fill traffic on
  // ff<<1 data) at the cost of duplicating the matrix in VRAM (planner-gated
  // on free memory). Bit-identical (same bytes, different source layout).
  // Measured: +2.2% numerai-deep; inert without feature sampling.
  bool colmajor_fill = true;        // key: colmajor_fill
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
    if (key == "compact_quant") return &compact_quant;
    if (key == "construct_jit") return &construct_jit;
    if (key == "fast_rowdata") return &fast_rowdata;
    if (key == "rowdata_4bit") return &rowdata_4bit;
    if (key == "gpu_construct") return &gpu_construct;
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
    if (key == "compact_prefill") return &compact_prefill;
    if (key == "pack_bit3") return &pack_bit3;
    if (key == "pack_radix5") return &pack_radix5;
    if (key == "pack_radix6") return &pack_radix6;
    if (key == "pack_radix7") return &pack_radix7;
    if (key == "l2_policy") return &l2_policy;
    if (key == "colmajor_fill") return &colmajor_fill;
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
    Mutable() = plan;
    if (overridden) {
      Log::Info("cuda_plan: hybrid=%d selective=%d one_sync=%d graph_loop=%d graph_quant=%d "
                "compact_quant=%d construct_jit=%d fast_rowdata=%d rowdata_4bit=%d "
                "gpu_construct=%d efb_precheck=%d batch_kernels=%d batch_apply=%d "
                "batch_reghist=%d batch_wide=%d gh_interleave=%d split_packed_read=%d "
                "small_leaf_construct=%d",
                plan.hybrid, plan.selective, plan.one_sync, plan.graph_loop, plan.graph_quant,
                plan.compact_quant, plan.construct_jit, plan.fast_rowdata, plan.rowdata_4bit,
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
 * Any debug token also disables the CUDA-graph controller path (the device
 * controller does not replicate these hooks).
 */
struct FalcataDebugOptions {
  bool diag = false;
  bool debug = false;
  bool dump = false;
  bool syncpairs = false;
  bool aggressive = false;
  int maxsplits = -1;  // -1 = uncapped
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
