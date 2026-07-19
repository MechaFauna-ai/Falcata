/*!
 * Copyright (c) 2026 ExaBoost contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_EXABOOST_PLAN_H_
#define LIGHTGBM_EXABOOST_PLAN_H_

#include <LightGBM/config.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

namespace LightGBM {

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
struct ExaBoostPlan {
  // --- shape-conditional decisions (cuda_plan override keys) ---------------
  // hybrid level-batched growth (off = classic one-split-at-a-time leaf-wise)
  bool hybrid = true;               // key: hybrid
  // CUDA-graph level loop for the hybrid prefix (helps shallow trees; the
  // fixed controller latency turns net-negative on deep configs)
  bool graph_loop = true;           // key: graph_loop
  // graph level loop also for quantized training (opt-in; less measured)
  bool graph_quant = false;         // key: graph_quant
  // per-tree compact column view for feature_fraction<1 quantized construct
  bool compact_quant = true;        // key: compact_quant
  // NVRTC runtime-JIT construct kernels (self-test-then-promote; AOT fallback)
  bool construct_jit = false;       // key: construct_jit
  // dense row-data build from column bins (skips host multi-val bin)
  bool fast_rowdata = true;         // key: fast_rowdata
  // 4-bit packed row data for <=16-bin features
  bool rowdata_4bit = true;         // key: rowdata_4bit
  // GPU dense-matrix binning during dataset construction
  bool gpu_construct = true;        // key: gpu_construct
  // cheap host precheck that skips EFB bundling on provably-unbundlable data
  bool efb_precheck = true;         // key: efb_precheck

  // --- baked invariant winners (no keys) -----------------------------------
  bool split_packed_read = true;    // packed split reads in apply kernels
  bool batch_kernels = true;        // one find/sync launch per level
  bool batch_apply = true;          // batched per-level apply phase
  bool one_sync = true;             // speculative single-sync level pipeline
  bool selective = true;            // grow-then-prune for budget-limited configs
  bool batch_reghist = true;        // register-tiled batched construct kernel
  bool batch_wide = true;           // wide leaf-splits init batching
  bool gh_interleave = true;        // packed grad/hess interleaved layout
  bool small_leaf_construct = true;  // small-leaf construct specialization

  // --- baked tuning constants (no keys) ------------------------------------
  int batch_construct_min_rows_per_thread = 64;
  int batch_construct_saturation_floor = 160;
  int construct_column_cap = -1;    // -1 = auto sizing

  /*! \brief The process-global plan (mutable form, for the resolve points). */
  static ExaBoostPlan& Mutable() {
    static ExaBoostPlan plan;
    return plan;
  }
  /*! \brief The process-global plan, as consumers read it. */
  static const ExaBoostPlan& Get() { return Mutable(); }

  /*!
   * \brief Resolve the global plan from a parsed Config. Called from
   * DatasetLoader (ingestion) and the CUDA tree learner Init (training);
   * both parse the same string so the plan is consistent across phases.
   */
  static void ResolveFromConfig(const Config& config) {
    ExaBoostPlan plan;  // defaults = the auto plan
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
      if (kv[0] == std::string("hybrid")) {
        plan.hybrid = value;
      } else if (kv[0] == std::string("graph_loop")) {
        plan.graph_loop = value;
      } else if (kv[0] == std::string("graph_quant")) {
        plan.graph_quant = value;
      } else if (kv[0] == std::string("compact_quant")) {
        plan.compact_quant = value;
      } else if (kv[0] == std::string("construct_jit")) {
        plan.construct_jit = value;
      } else if (kv[0] == std::string("fast_rowdata")) {
        plan.fast_rowdata = value;
      } else if (kv[0] == std::string("rowdata_4bit")) {
        plan.rowdata_4bit = value;
      } else if (kv[0] == std::string("gpu_construct")) {
        plan.gpu_construct = value;
      } else if (kv[0] == std::string("efb_precheck")) {
        plan.efb_precheck = value;
      } else {
        Log::Fatal("cuda_plan: unknown key \"%s\"", kv[0].c_str());
      }
      overridden = true;
    }
    Mutable() = plan;
    if (overridden) {
      Log::Info("cuda_plan: hybrid=%d graph_loop=%d graph_quant=%d compact_quant=%d "
                "construct_jit=%d fast_rowdata=%d rowdata_4bit=%d gpu_construct=%d efb_precheck=%d",
                plan.hybrid, plan.graph_loop, plan.graph_quant, plan.compact_quant,
                plan.construct_jit, plan.fast_rowdata, plan.rowdata_4bit,
                plan.gpu_construct, plan.efb_precheck);
    }
  }
};

/*!
 * \brief EXABOOST_VERIFY=1: verify every enabled fast path against its
 * reference implementation (byte/bit comparison) during the run. Developer
 * gate for the check-then-drop workflow; never affects results, only speed.
 */
inline bool ExaboostVerifyEnabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("EXABOOST_VERIFY");
    return env != nullptr && std::string(env) == std::string("1");
  }();
  return enabled;
}

/*!
 * \brief EXABOOST_DEBUG: comma-separated developer diagnostics, e.g.
 * ``EXABOOST_DEBUG=diag,dump,maxsplits=8``. Tokens:
 *  - ``diag``       per-phase timing/counter diagnostics
 *  - ``debug``      verbose hybrid-growth debug checks/logging
 *  - ``dump``       dump partition snapshots to files
 *  - ``syncpairs``  force per-pair synchronization (isolates batching)
 *  - ``aggressive`` experimental aggressive hybrid batching
 *  - ``maxsplits=N`` cap splits per level (isolates multi-pair interactions)
 * Any debug token also disables the CUDA-graph controller path (the device
 * controller does not replicate these hooks).
 */
struct ExaboostDebugOptions {
  bool diag = false;
  bool debug = false;
  bool dump = false;
  bool syncpairs = false;
  bool aggressive = false;
  int maxsplits = -1;  // -1 = uncapped
  bool any_growth_hook() const { return debug || syncpairs || maxsplits >= 0; }
};

inline const ExaboostDebugOptions& ExaboostDebug() {
  static const ExaboostDebugOptions opts = []() {
    ExaboostDebugOptions o;
    const char* env = std::getenv("EXABOOST_DEBUG");
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
        Log::Warning("EXABOOST_DEBUG: unknown token \"%s\" ignored", token.c_str());
      }
    }
    return o;
  }();
  return opts;
}

}  // namespace LightGBM

#endif  // LIGHTGBM_EXABOOST_PLAN_H_
