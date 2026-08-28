/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <sys/stat.h>

#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/falcata_plan.h>
#include <Falcata/feature_group.h>
#include <Falcata/network.h>
#include <Falcata/objective_function.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Falcata/cuda/cuda_tree.hpp>

#include "cuda_single_gpu_tree_learner.hpp"
#include "../cost_effective_gradient_boosting.hpp"
#include "../linear_leaf_solver.h"

namespace Falcata {

CUDASingleGPUTreeLearner::CUDASingleGPUTreeLearner(const Config* config, const bool boosting_on_cuda): SerialTreeLearner(config), boosting_on_cuda_(boosting_on_cuda) {}

CUDASingleGPUTreeLearner::~CUDASingleGPUTreeLearner() {
  if (nccl_communicator_ != nullptr) {
    CUDAStreamDestroy(nccl_stream_);
  }
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  hybrid_graph_cache_.clear();
  if (pinned_hybrid_graph_state_ != nullptr) {
    cudaFreeHost(pinned_hybrid_graph_state_);
  }
  if (pinned_hybrid_graph_journal_ != nullptr) {
    cudaFreeHost(pinned_hybrid_graph_journal_);
  }
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
}

void CUDASingleGPUTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  // Resolve the CUDA execution plan for the training phase before any
  // component reads it (ingestion resolved the same string earlier).
  FalcataPlan::ResolveFromConfig(*config_);
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  num_threads_ = OMP_NUM_THREADS();
  // use the first gpu by default
  if (nccl_communicator_ == nullptr) {
    gpu_device_id_ = config_->gpu_device_id >= 0 ? config_->gpu_device_id : 0;
    SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
  }
  // quant_mode=fixedpoint: xgboost-inspired near-lossless deterministic
  // quantization. Non-stochastic round-to-nearest over a high bin count fills
  // the int16 gradient range, and the existing packed-histogram path already
  // auto-promotes to exact int32/int64 accumulation as num_data*bins grows --
  // so no rounding noise accumulates and results are run-to-run identical.
  // The same bin count (config quant_bins, auto=64 for fixedpoint) must reach
  // BOTH the histogram constructor (its 65534/bins per-block row cap guards
  // packed-hist overflow) and the discretizer (dequant scale), so it is
  // resolved once in Config::ResolveFalcataParams.
  //
  // The outlier-robust gradient scale is an unconditional part of fixedpoint
  // mode: it is gap-gated (see ComputeRobustGradScaleKernel), so balanced
  // gradient distributions keep the exact global-max scale bit-for-bit, and
  // only a genuinely imbalanced distribution (rare-class outliers separated
  // from the bulk by an empty magnitude gap) gets the bulk re-anchored while
  // the outliers clamp to +/-(bins/2) -- exactly the magnitude bound the
  // global-max path guarantees, so SetNumBitsInHistogramBin's num_data*bins
  // promotion stays valid. Users unhappy with fixedpoint quality should step
  // down to quant_mode=stochastic or none, not tune the scale.
  const QuantMode quant_mode = config_->ResolvedQuantMode();
  fixedpoint_quant_ = (quant_mode == QuantMode::kFixedPoint);
  effective_quant_bins_ = config_->num_grad_quant_bins;
  if (config_->use_quantized_grad) {
    // The packed histogram accumulates quantized gradients in (at most) 32-bit
    // halves; the per-bin worst case is num_data * (bins/2). Beyond int32 that
    // sum is unrepresentable and training silently emits garbage trees (found
    // empirically: quant_bins=1024 at 5M rows -> constant predictions), so
    // refuse loudly instead. Ceiling: bins < 2^32 / num_data.
    // HESSIANS bound the range, not gradients: hessians quantize to the FULL
    // bin count (gradients only to +/-bins/2), so the worst per-bin sum is
    // num_data * bins. The old bins/2 bound let e.g. 515k rows x 4096 bins
    // through and training emitted inf predictions (hessian sums wrapped).
    const uint64_t worst_per_bin = static_cast<uint64_t>(num_data_) *
        static_cast<uint64_t>(effective_quant_bins_);
    if (worst_per_bin >= (1ULL << 31)) {
      const int max_bins = static_cast<int>((1ULL << 31) / static_cast<uint64_t>(num_data_));
      if (config_->quant_bins_from_auto && max_bins >= 2) {
        // the bin count was the mode's auto default, not a user choice: pick
        // the finest dataset-safe count instead of refusing (the guard exists
        // to prevent silent int32 wrap, not to make big data unusable)
        Log::Warning(
            "quant_bins auto default %d exceeds the 32-bit histogram range for "
            "%d rows; using the dataset-safe quant_bins=%d instead (set "
            "quant_bins explicitly to override).",
            effective_quant_bins_, num_data_, max_bins);
        effective_quant_bins_ = max_bins;
      } else {
        Log::Fatal(
            "quant_bins=%d is too large for %d rows: the per-bin gradient sum can "
            "exceed the 32-bit histogram range. Use quant_bins <= %d for this dataset.",
            effective_quant_bins_, num_data_, max_bins);
      }
    }
  }
  // Stochastic small-leaf auto-raise: with the historical 4-bin
  // auto default, deep regimes whose leaves get small fit rounding noise and
  // test quality DECLINES mid-training (measured: epsilon-deep peaks at iter
  // ~300 then loses 0.005 AUC; year-deep peaks at ~75 then loses +4.9 MSE;
  // higgs at ~10k rows/leaf is immune). The rounding scheme is innocent --
  // stochastic@64 exactly matches fixedpoint@64 quality -- the resolution is
  // just too coarse for small per-leaf sums, and renew_leaf recovers only
  // ~1/3 of the damage (split-selection noise dominates). So when the bin
  // count is the mode's auto default and the expected leaf is small, use 64
  // bins (fixedpoint's count, capped by the int32 guard ceiling). Costs +28%
  // on epsilon-deep (= fixedpoint's cost) and nothing on big-leaf shapes,
  // which keep the 4-bin speed.
  if (config_->use_quantized_grad && !fixedpoint_quant_ &&
      config_->quant_bins_from_auto) {
    const int64_t depth_leaf_cap = (config_->max_depth > 0 && config_->max_depth < 31) ?
        (1LL << config_->max_depth) : std::numeric_limits<int64_t>::max();
    const int64_t effective_leaves = std::min<int64_t>(config_->num_leaves, depth_leaf_cap);
    const data_size_t rows_per_leaf = static_cast<data_size_t>(
        num_data_ / std::max<int64_t>(1, effective_leaves));
    constexpr data_size_t kSmallLeafRows = 4096;
    // the raise needs BOTH conditions: on small datasets (gate-profile scale)
    // 4-bin stochastic noise acts as regularization and finer bins measurably
    // HURT (lattice: dense rmse +5%, unbalance AUC -0.08 even at 16); the
    // fit-the-noise drift only emerges when a big dataset drives many small
    // leaves (year/epsilon deep, >=400k rows)
    constexpr data_size_t kMinRowsForRaise = 100000;
    // extreme binary imbalance defeats finer stochastic bins outright
    // (fraud-deep 577:1 measured AUC .956@4 -> .870@16: rare-class outlier
    // gradients pin the global-max scale, the same no-robust-scale failure
    // mode fixedpoint guards against) -- keep the 4-bin default there. 50:1
    // matches the objective's imbalance-warning threshold.
    bool extreme_imbalance = false;
    if (config_->objective == std::string("binary")) {
      const label_t* labels = train_data_->metadata().label();
      data_size_t pos = 0;
      for (data_size_t i = 0; i < num_data_; ++i) pos += (labels[i] > 0);
      const data_size_t minority = std::min(pos, num_data_ - pos);
      extreme_imbalance = (minority == 0) ||
          (static_cast<int64_t>(num_data_ - minority) >= 50LL * minority);
    }
    // Multiclass is excluded for the same reason as extreme binary imbalance,
    // which it structurally resembles: each per-class tree sees only ~1/k
    // positives, so the rare-class gradients pin the global-max scale and the
    // finer bins buy resolution the bulk cannot use. Measured on covtype
    // (7 classes, 464k rows, ~454 rows/leaf -- squarely inside the raise
    // window): accuracy .91841 at the 4-bin default vs .91510 raised to 16.
    // The shapes the raise was built for (year-deep regression, epsilon-deep
    // binary) are untouched.
    const bool multiclass = config_->num_class > 1;
    if (num_data_ >= kMinRowsForRaise && rows_per_leaf < kSmallLeafRows &&
        !extreme_imbalance && !multiclass) {
      // constant-hessian objectives take the full 64 (fixedpoint's count,
      // drift-free on year); non-constant ones take 16 -- enough to kill the
      // drift (epsilon: zero at 16) without the imbalanced-binary quality
      // loss the gate lattice showed at 64 (unbalance AUC .94 -> .81: coarse
      // global-max scale + rare-class outliers, the failure mode fixedpoint's
      // robust scale exists for; stochastic has no robust scale)
      const int target = is_constant_hessian ? 64 : 16;
      const uint64_t guard_cap = (1ULL << 31) / static_cast<uint64_t>(num_data_) - 1;
      const int raised = static_cast<int>(std::min<uint64_t>(target, guard_cap));
      if (raised > effective_quant_bins_) {
        effective_quant_bins_ = raised;
        Log::Info("stochastic quant_bins auto-raised to %d: ~%d rows/leaf is too "
                  "small for the 4-bin default (test quality would degrade "
                  "mid-training); set quant_bins explicitly to override.",
                  effective_quant_bins_, rows_per_leaf);
      }
    }
  }
  // The outlier-robust scale protects COARSE quantization (its measured win
  // is fraud AUC .9825-vs-.8001 at the 64-bin default): re-anchoring to the
  // bulk buys resolution at the cost of clamping outliers. At fine bin
  // counts the global-max scale already resolves the bulk, and the clamped
  // outlier mass turns into measured quality LOSS that grows with bins
  // (covtype-deep acc .945/.838/.807 at 1024/2048/4096 vs .963 without).
  // 256 keeps every historical behavior (default 64 and the 256-bin gate
  // cell inclusive) and frees fine-grained fixedpoint to behave like the
  // near-lossless mode it is.
  constexpr int kRobustScaleMaxBins = 256;
  // Multiclass is excluded. The guard looks for a SMALL cluster of outlier
  // gradients sitting above a magnitude gap, and re-anchors the quant scale to
  // the bulk below it -- the shape produced by extreme binary imbalance, which
  // is what it was built for. Softmax gradients are bounded and bimodal by
  // construction (p-1 for the true class, p for the rest), so the "outliers"
  // it finds are the most informative gradients in the problem, and clipping
  // them costs real accuracy: measured multi_logloss 1.2219 with the guard vs
  // 1.0956 without, against a 1.0954 full-precision reference (7 classes,
  // max_bin 5). It does not fire at all for binary or regression, so nothing
  // it was designed for is lost here.
  fixedpoint_robust_scale_ = fixedpoint_quant_ && FalcataPlan::Get().robust_scale &&
      effective_quant_bins_ <= kRobustScaleMaxBins && config_->num_class <= 1;
  // INTERIM CLAMP (tracked): with NON-constant hessians (binary/multiclass),
  // high quant_bins corrupts training; the threshold is data-dependent
  // (covtype breaks at 2048, imbalanced fraud already at 1024 -- auc .4996;
  // constant-hessian regression is immune at any guard-passing bins).
  // 256 is the verified-safe line across every measured case; clamp there
  // until the corruption is root-caused.
  if (effective_quant_bins_ > 256 && !is_constant_hessian) {
    Log::Warning("quant_bins=%d clamped to 256 for this objective: larger "
                 "values with non-constant hessians corrupt training on "
                 "some data (under investigation); constant-hessian "
                 "regression objectives are unaffected.", effective_quant_bins_);
    effective_quant_bins_ = 256;
  }
  // deterministic=true with auto bin count (quant_bins=0): pick the finest
  // SAFE resolution instead of the historical default -- near-lossless
  // deterministic training is the stated intent. Constant-hessian
  // objectives take the finest guard-passing count up to 2048 (verified
  // lossless-equivalent on year); others stay at the 256 safety line.
  // Test quant_bins_from_auto, not quant_bins == 0: ResolveFalcataParams
  // rewrites the 0 sentinel to the mode default at parse time, so it never
  // reaches here.
  if (config_->deterministic && fixedpoint_quant_ && config_->quant_bins_from_auto) {
    const uint64_t guard_cap = (1ULL << 31) / static_cast<uint64_t>(num_data_) - 1;
    const int safe_cap = static_cast<int>(std::min<uint64_t>(guard_cap, is_constant_hessian ? 2048 : 256));
    if (safe_cap > effective_quant_bins_) {
      effective_quant_bins_ = safe_cap;
      Log::Info("deterministic=true: quant_bins auto-raised to %d (near-lossless "
                "fixedpoint resolution for this dataset)", effective_quant_bins_);
    }
  }
  if (fixedpoint_quant_) {
    Log::Info("quant_mode=fixedpoint: non-stochastic quant with %d bins", effective_quant_bins_);
  }
  if (TunerActive()) {
    TunerInitSeeds();
  }
  // cuda_precision=fp32 engages both fp32 switches (histogram storage + gain
  // math). They were only ever measured -- and only ever win -- together, so
  // one knob. Must be set before the histogram constructor / best split finder
  // below read the process-global flags.
  const bool fp32 = (config_->ResolvedCudaPrecision() == CudaPrecision::kFP32);
  FalcataFP32HistRequestedFlag() = fp32;
  FalcataFP32GainEnabledFlag() = fp32;
  // vector-leaf (multi-target) mode: T = num_class when tree_mode=vector_leaf
  // (the config layer guarantees objective=multi_regression for T > 1)
  vec_num_targets_ = (config_->tree_mode == std::string("vector_leaf") &&
                      config_->num_class > 1) ? config_->num_class : 0;
  cuda_smaller_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  cuda_smaller_leaf_splits_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, global_num_data_);
  cuda_smaller_leaf_splits_->Init(config_->use_quantized_grad);
  cuda_larger_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  cuda_larger_leaf_splits_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, global_num_data_);
  cuda_larger_leaf_splits_->Init(config_->use_quantized_grad);

  cuda_histogram_constructor_.reset(new CUDAHistogramConstructor(train_data_, config_->num_leaves, num_threads_,
    share_state_->feature_hist_offsets(),
    config_->min_data_in_leaf, config_->min_sum_hessian_in_leaf, gpu_device_id_, config_->gpu_use_dp,
    config_->use_quantized_grad, effective_quant_bins_, config_->feature_fraction));
  if (vec_num_targets_ > 1) {
    // one (grad, hess)-pair histogram plane per target within each leaf slot
    cuda_histogram_constructor_->SetNumHistPlanes(vec_num_targets_);
  }
  cuda_histogram_constructor_->Init(train_data_, share_state_.get());

  // construct-JIT auto-gate (mirrors the tuner's): the one-time NVRTC
  // compile + self-test (~230ms) amortizes only on real runs. Vector mode
  // launches the construct kernels once per gradient plane through swapped
  // pointers, which the JIT capture does not model -- keep it off there.
  cuda_histogram_constructor_->SetConstructJITAllowed(
      vec_num_targets_ <= 1 &&
      FalcataPlan::Get().construct_jit &&
      (FalcataPlan::Get().construct_jit_explicit || config_->num_iterations >= 300));

  const auto& feature_hist_offsets = share_state_->feature_hist_offsets();
  num_total_bin_ = feature_hist_offsets.empty() ? 0 : static_cast<int>(feature_hist_offsets.back());
  // the partition's bin count is ONLY used for leaf histogram slot arithmetic
  // (cuda_hist + 2 * leaf * num_total_bin); vector mode widens each slot to T
  // contiguous planes, so hand it the per-slot width
  cuda_data_partition_.reset(new CUDADataPartition(
    train_data_, num_total_bin_ * std::max(1, vec_num_targets_), config_->num_leaves,
    num_threads_, config_->use_quantized_grad,
    cuda_histogram_constructor_->cuda_hist_pointer()));
  cuda_data_partition_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, global_num_data_);
  cuda_data_partition_->Init();

  select_features_by_node_ = !config_->interaction_constraints_vector.empty() || config_->feature_fraction_bynode < 1.0;
  // hybrid level-batched growth is on by default; cuda_plan=auto,hybrid:off
  // falls back to the classic one-split-at-a-time leaf-wise loop everywhere
  use_hybrid_growth_ = FalcataPlan::Get().hybrid;
  // Monotone constraints are inherited down the tree: a leaf's [min, max] bounds
  // come from the splits already applied above it. Level-batched growth scores a
  // whole level before applying any of it, so the children of a level's splits
  // would be searched against their parents' stale bounds. Fall back to the
  // classic leaf-wise loop, the same way the batched path is gated off for
  // categorical features.
  if (!config_->monotone_constraints.empty()) {
    use_hybrid_growth_ = false;
  }
  // CEGB is order-dependent: cegb_penalty_feature_coupled is charged the first
  // time a feature is used, so a split's gain depends on which splits were
  // already applied. Level-batched growth scores a whole level before applying
  // any of it, so the coupled penalty would be charged to every leaf in the
  // level instead of just the first -- CPU (leaf-wise, best-first) and CUDA then
  // grow different trees. Fall back to the classic leaf-wise loop, the same way
  // the batched path is gated off for categorical features.
  if (CostEfficientGradientBoosting::IsEnable(config_)) {
    use_hybrid_growth_ = false;
  }
  // batched per-level kernels for the hybrid prefix (one construct/fix/subtract/
  // find/sync launch per level instead of per pair); "0" keeps the per-pair path
  use_hybrid_batch_kernels_ = FalcataPlan::Get().batch_kernels;
  // batched per-level APPLY phase for the hybrid prefix (one launch per kernel
  // family per level and zero per-split host syncs, instead of ~7 launches and
  // 2 device syncs per split); "0" keeps the per-split deferred loop
  use_hybrid_batch_apply_ = FalcataPlan::Get().batch_apply;
  // single-sync (speculative) level pipeline for the hybrid prefix: each level's
  // apply and the children's search are enqueued back to back and read back with
  // ONE device synchronization per level; "0" keeps the classic two-sync flow
  use_hybrid_one_sync_ = FalcataPlan::Get().one_sync;
  // selective (grow-then-prune) growth for budget-limited configs
  // (num_leaves << 2^max_depth): exactly leaf-wise-equivalent level batching
  // with end-of-selection pruning; "0" falls back to the classic loop there
  use_hybrid_selective_ = FalcataPlan::Get().selective;
  // vector-leaf training rides the hybrid TWO-SYNC level flow: the speculative
  // (one-sync) flow derives its child gates from device structs the plane
  // fan-out has not refreshed yet, and selective growth recycles leaf indices
  // through a host tree rebuild that carries target-0 leaf values only. Both
  // fall back to the classic per-split loop here, as does the graph loop
  // (HybridGraphPrefixUsable).
  if (vec_num_targets_ > 1) {
    use_hybrid_one_sync_ = false;
    use_hybrid_selective_ = false;
  }
  cuda_best_split_finder_.reset(new CUDABestSplitFinder(cuda_histogram_constructor_->cuda_hist(),
    train_data_, this->share_state_->feature_hist_offsets(), select_features_by_node_, config_));
  cuda_best_split_finder_->Init();
  if (config_->use_quantized_grad && cuda_best_split_finder_->use_global_memory()) {
    // The discretized best-split finder has no global-memory variant (upstream
    // left the branch as a TODO that launches NOTHING, silently training on
    // stale split info). Refuse loudly instead of producing a garbage model.
    Log::Fatal("use_quantized_grad is not supported on CUDA when a feature has more than %d histogram bins "
               "(max_bin=%d requested; a categorical feature with more than %d categories also exceeds the "
               "limit). Reduce max_bin / group rare categories, or disable quantized gradients.",
               NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, config_->max_bin,
               NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER);
  }
  // Wire the histogram constructor's completion events into the best split finder so
  // per-leaf FindBestSplits kernels are ordered after histogram construction/subtraction
  // via cudaStreamWaitEvent instead of per-split device syncs.
  cuda_best_split_finder_->SetHistogramEvents(
    cuda_histogram_constructor_->construct_done_events(),
    cuda_histogram_constructor_->subtract_done_events());
  SyncHistFP32();
  cuda_best_split_finder_->SetCEGB(config_->cegb_penalty_feature_coupled,
                                   config_->cegb_tradeoff,
                                   config_->cegb_penalty_split,
                                   train_data_);

  leaf_best_split_feature_.resize(config_->num_leaves, -1);
  leaf_best_split_threshold_.resize(config_->num_leaves, 0);
  leaf_best_split_default_left_.resize(config_->num_leaves, 0);
  leaf_num_data_.resize(config_->num_leaves, 0);
  leaf_data_start_.resize(config_->num_leaves, 0);
  leaf_sum_gradients_.resize(config_->num_leaves, 0.0f);
  leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);

  use_monotone_constraints_ = !config_->monotone_constraints.empty();
  if (use_monotone_constraints_) {
    leaf_constraint_min_.resize(config_->num_leaves, -std::numeric_limits<double>::max());
    leaf_constraint_max_.resize(config_->num_leaves, std::numeric_limits<double>::max());
    // config_->monotone_constraints is real-indexed; map to inner feature index.
    monotone_constraints_.resize(train_data_->num_features(), 0);
    for (int inner_feature_index = 0; inner_feature_index < train_data_->num_features(); ++inner_feature_index) {
      const int real_feature_index = train_data_->RealFeatureIndex(inner_feature_index);
      monotone_constraints_[inner_feature_index] =
        config_->monotone_constraints[real_feature_index];
    }
  }

  if (!boosting_on_cuda_) {
    const size_t grad_planes = static_cast<size_t>(std::max(1, vec_num_targets_));
    cuda_gradients_.Resize(static_cast<size_t>(num_data_) * grad_planes);
    cuda_hessians_.Resize(static_cast<size_t>(num_data_) * grad_planes);
  }
  AllocateBitset();

  // vector-leaf gates read has_categorical_feature_, which AllocateBitset
  // just derived from the dataset
  if (vec_num_targets_ > 1) {
    CheckVectorLeafSupported();
    cuda_best_split_finder_->InitVectorMode(vec_num_targets_);
    vec_plane_structs_.Resize(2 * static_cast<size_t>(vec_num_targets_));
    vec_root_sum_gradients_.resize(vec_num_targets_, 0.0);
    vec_root_sum_hessians_.resize(vec_num_targets_, 0.0);
  }

  leaf_stat_buffer_size_ = 0;
  num_cat_threshold_ = 0;

  if (config_->use_quantized_grad) {
    cuda_leaf_gradient_stat_buffer_.Resize(config_->num_leaves);
    cuda_leaf_hessian_stat_buffer_.Resize(config_->num_leaves);
    // effective_quant_bins_ / fixedpoint_quant_ were resolved at the top of Init
    // so the histogram constructor and the discretizer agree on the bin count.
    const bool fp_stochastic = fixedpoint_quant_ ? false : config_->stochastic_rounding;
    // one random-offset slot per TREE: multiclass trains num_class trees per
    // iteration, and the discretizer's iter_ counter advances once per tree
    cuda_gradient_discretizer_.reset(new CUDAGradientDiscretizer(
      effective_quant_bins_, config_->num_iterations * std::max(config_->num_class, 1), config_->seed, is_constant_hessian, fp_stochastic));
    cuda_gradient_discretizer_->SetRobustScale(fixedpoint_robust_scale_);
    // error feedback debiases fixedpoint's deterministic rounding; stochastic
    // rounding is already unbiased and never gets it
    cuda_gradient_discretizer_->SetErrorFeedback(
        fixedpoint_quant_ && FalcataPlan::Get().quant_ef,
        std::max(config_->num_class, 1));
    cuda_gradient_discretizer_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, global_num_data_);
    cuda_gradient_discretizer_->Init(num_data_, config_->num_leaves, train_data_->num_features(), train_data_);
  } else {
    cuda_gradient_discretizer_.reset(nullptr);
  }

  #ifdef DEBUG
  host_gradients_.resize(num_data_, 0.0f);
  host_hessians_.resize(num_data_, 0.0f);
  #endif  // DEBUG

  if (config_->linear_tree) {
    InitLinearTreeCUDA(train_data_);
  }
}

// The fp32 histogram layout does not cover the large-bin global-memory find
// path, forced splits (their gather kernel reads hist_t) or NCCL histogram
// reduction: fall back to fp64 there, then hand the resolved mode to the best
// split finder (whose kernels read the entries).
void CUDASingleGPUTreeLearner::SyncHistFP32() {
  if (cuda_histogram_constructor_->hist_fp32() &&
      (cuda_best_split_finder_->use_global_memory() ||
       !config_->forcedsplits_filename.empty() ||
       nccl_communicator_ != nullptr)) {
    cuda_histogram_constructor_->DisableHistFP32();
  }
  cuda_best_split_finder_->SetHistFP32(cuda_histogram_constructor_->hist_fp32());
}

void CUDASingleGPUTreeLearner::BeforeTrain() {
  const data_size_t root_num_data = cuda_data_partition_->root_num_data();
  if (!boosting_on_cuda_) {
    const size_t grad_count = static_cast<size_t>(num_data_) *
      static_cast<size_t>(std::max(1, vec_num_targets_));
    CopyFromHostToCUDADevice<score_t>(cuda_gradients_.RawData(), gradients_, grad_count, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<score_t>(cuda_hessians_.RawData(), hessians_, grad_count, __FILE__, __LINE__);
    gradients_ = cuda_gradients_.RawData();
    hessians_ = cuda_hessians_.RawData();
  }

  #ifdef DEBUG
  CopyFromCUDADeviceToHost<score_t>(host_gradients_.data(), gradients_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<score_t>(host_hessians_.data(), hessians_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  #endif  // DEBUG

  const data_size_t* leaf_splits_init_indices =
    cuda_data_partition_->use_bagging() ? cuda_data_partition_->cuda_data_indices() : nullptr;
  cuda_data_partition_->BeforeTrain();
  if (config_->use_quantized_grad) {
    if (cuda_data_partition_->use_bagging()) {
      cuda_gradient_discretizer_->SetBagForThisTree(
          cuda_data_partition_->cuda_data_indices(), cuda_data_partition_->root_num_data());
    } else {
      cuda_gradient_discretizer_->SetBagForThisTree(nullptr, 0);
    }
    cuda_gradient_discretizer_->DiscretizeGradients(num_data_, gradients_, hessians_);
    cuda_histogram_constructor_->BeforeTrain(
      reinterpret_cast<const score_t*>(cuda_gradient_discretizer_->discretized_gradients_and_hessians()), nullptr);
    cuda_smaller_leaf_splits_->InitValues(
      config_->lambda_l1,
      config_->lambda_l2,
      config_->max_delta_step,
      reinterpret_cast<const int16_t*>(cuda_gradient_discretizer_->discretized_gradients_and_hessians()),
      leaf_splits_init_indices,
      cuda_data_partition_->cuda_data_indices(),
      root_num_data,
      cuda_histogram_constructor_->cuda_hist_pointer(),
      &leaf_sum_gradients_[0],
      &leaf_sum_hessians_[0],
      cuda_gradient_discretizer_->grad_scale_ptr(),
      cuda_gradient_discretizer_->hess_scale_ptr());
      cuda_gradient_discretizer_->SetNumBitsInHistogramBin<false>(0, -1, root_num_data, 0);
      if (nccl_communicator_ != nullptr) {
        cuda_gradient_discretizer_->SetNumBitsInHistogramBin<true>(0, -1, global_num_data_, 0);
      }
  } else {
    cuda_histogram_constructor_->BeforeTrain(gradients_, hessians_);
    // The single-sync hybrid prefix derives every root gate on-device, so the
    // synchronous root-sum readback (which would block on the histogram-buffer
    // zeroing above) is deferred; EnsureRootSumsReadBack performs it lazily on
    // the paths that do need host root sums. Optimistic: SupportsBatchedLevel
    // may flip once BuildCompactView below runs, which the classic prefix
    // handles by reading the sums back at its start.
    root_sums_deferred_ = HybridGrowthUsable() && UseOneSyncPrefix();
    // One ownership model of the histogram storage per tree (see
    // DetDenseBatchedEligible): host batched det may run alongside the graph
    // prefix only when the graph itself captured the det construct+merge
    // nodes; a run whose graph stays atomic stays atomic end to end.
    bool det_batched_allowed = !HybridGraphPrefixUsable();
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
    if (!det_batched_allowed) {
      const int widest = std::min(
        1 << std::max(0, std::min(config_->max_depth - 1, 10)),
        kHybridGraphMaxSplitsPerLevel);
      det_batched_allowed = cuda_histogram_constructor_->DetDenseGraphEligible(widest);
    }
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
    cuda_histogram_constructor_->SetDetBatchedAllowed(det_batched_allowed);
    cuda_smaller_leaf_splits_->InitValues(
      config_->lambda_l1,
      config_->lambda_l2,
      config_->max_delta_step,
      gradients_,
      hessians_,
      leaf_splits_init_indices,
      cuda_data_partition_->cuda_data_indices(),
      root_num_data,
      cuda_histogram_constructor_->cuda_hist_pointer(),
      &leaf_sum_gradients_[0],
      &leaf_sum_hessians_[0],
      root_sums_deferred_);
  }
  leaf_num_data_[0] = root_num_data;
  cuda_larger_leaf_splits_->InitValues();
  if (vec_num_targets_ > 1) {
    VectorInitRootPlanes(leaf_splits_init_indices, root_num_data);
  }
  if (next_tree_col_sample_ready_) {
    // the previous Train() already drew this tree's column sample (prefill)
    next_tree_col_sample_ready_ = false;
  } else {
    col_sampler_.ResetByTree();
  }
  if (fp_merge_state_ != nullptr) {
    // feature-parallel: every rank draws the IDENTICAL per-tree sample (same
    // seed, same RNG call sequence) and keeps its stripe of it. The union of
    // the ranks' masks is exactly the sample, so the merged search space
    // matches single-GPU training.
    const std::vector<int8_t>& sample = col_sampler_.is_feature_used_bytree();
    fp_feature_mask_.assign(sample.begin(), sample.end());
    for (int i = 0; i < static_cast<int>(fp_feature_mask_.size()); ++i) {
      if (i % fp_num_ranks_ != fp_rank_) {
        fp_feature_mask_[i] = 0;
      }
    }
    cuda_best_split_finder_->BeforeTrain(fp_feature_mask_);
    cuda_histogram_constructor_->SetFeatureUsedBytree(fp_feature_mask_);
    cuda_histogram_constructor_->BuildCompactView(fp_feature_mask_);
  } else {
    cuda_best_split_finder_->BeforeTrain(col_sampler_.is_feature_used_bytree());
    cuda_histogram_constructor_->SetFeatureUsedBytree(col_sampler_.is_feature_used_bytree());
    cuda_histogram_constructor_->BuildCompactView(col_sampler_.is_feature_used_bytree());
  }
  if (nccl_communicator_ != nullptr) {
    // this tree's feature sample decides which histogram bins the all-reduce
    // has to carry; everything else is dead storage no kernel of this tree
    // reads (see BuildNCCLReduceBinIndex)
    BuildNCCLReduceBinIndex();
  }
  // Build per-tree compact column data for the partition kernels.
  BuildCompactColumnView();
  leaf_data_start_[0] = 0;
  smaller_leaf_index_ = 0;
  larger_leaf_index_ = -1;

  if (use_monotone_constraints_) {
    std::fill(leaf_constraint_min_.begin(), leaf_constraint_min_.end(),
              -std::numeric_limits<double>::max());
    std::fill(leaf_constraint_max_.begin(), leaf_constraint_max_.end(),
              std::numeric_limits<double>::max());
  }

  if (nccl_communicator_ != nullptr) {
    leaf_to_hist_index_map_.resize(config_->num_leaves, -1);
    leaf_to_hist_index_map_[0] = 0;
    global_num_data_in_leaf_[0] = global_num_data_;
  }

  // Compact-view prefill (cuda_plan key compact_prefill): draw the NEXT tree's
  // per-tree column sample now -- the same single Sample() call the next
  // BeforeTrain would make, so the RNG sequence is unchanged -- and fill its
  // compact view into the alt buffer on the side stream. Launched HERE, before
  // this tree's training begins, so the ~5ms fill hides inside this tree's
  // launch gaps rather than sitting on the critical path. Every consumer of
  // THIS tree's mask above has already taken its copy. Gated off under by-node
  // feature sampling (GetByNode draws from the same RNG per node; an early
  // by-tree draw would reorder the stream and change models).
  if (FalcataPlan::Get().compact_prefill && !select_features_by_node_ &&
      nccl_communicator_ == nullptr && fp_merge_state_ == nullptr &&
      cuda_histogram_constructor_->compact_view_active()) {
    col_sampler_.ResetByTree();
    next_tree_col_sample_ready_ = true;
    cuda_histogram_constructor_->PrefillNextCompactView(col_sampler_.is_feature_used_bytree());
  }
}

namespace {

// Split kernels read the 4-bit packed compact matrix directly (bit_type 4
// descriptors) instead of materializing the per-tree column-major buffer
// (~1.5GB write per tree on numerai-shaped data); the scattered packed reads
// cost one 32B sector per row but only the split columns are ever touched.
// FalcataPlan::split_packed_read=false restores the column-major gather.
bool SplitPackedReadEnabled() {
  return FalcataPlan::Get().split_packed_read;
}

}  // anonymous namespace

// Forward decl: defined in cuda_histogram_constructor.cu.
extern void LaunchRowToColCompactKernel(
    cudaStream_t stream,
    const uint8_t* src_data,
    uint8_t* compact_col_buf,
    const size_t* slot_p_byte,
    const int* slot_p_stride,
    const int* slot_col_in_p,
    int num_compact_cols,
    data_size_t num_data,
    bool src_is_4bit);

void CUDASingleGPUTreeLearner::BuildCompactColumnView() {
  const auto* row_data = cuda_histogram_constructor_->cuda_row_data_internal();
  if (row_data == nullptr) return;
  const uint8_t* src = row_data->GetBin<uint8_t>();
  if (src == nullptr) return;  // host-mapped path; not supported here
  if (row_data->bit_type() != 8 || row_data->is_sparse()) return;  // only dense uint8 path supported

  const auto& bytree_mask = col_sampler_.is_feature_used_bytree();
  uint64_t sig = 0;
  for (size_t i = 0; i < bytree_mask.size(); ++i) {
    if (bytree_mask[i]) sig = sig * 1099511628211ULL ^ static_cast<uint64_t>(i + 1);
  }
  // The published view is made of DEVICE POINTERS into the histogram
  // constructor's compact buffer (SetCompactPackedColumnView bakes
  // packed_buf + per-column offset, which the split kernels dereference), so
  // an unchanged column set does NOT by itself mean the cached pointers are
  // still valid: compact_prefill swaps the compact buffer between trees, and
  // a stale base silently routes splits through the previous tree's bytes.
  // Fold the source addresses into the key so any swap forces a rebuild.
  sig = sig * 1099511628211ULL ^
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            cuda_histogram_constructor_->compact_data_device()));
  sig = sig * 1099511628211ULL ^
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            cuda_histogram_constructor_->compact_col_major_device()));
  // Someone else may have replaced the published table since we installed ours
  // -- tree traversal restores the original per-column view (see
  // CUDATree::LaunchAddPredictionToScoreKernel) -- and then an unchanged layout
  // no longer means our view is the one in force.
  CUDAColumnData* col_data = const_cast<CUDAColumnData*>(train_data_->cuda_column_data());
  if (sig == compact_col_signature_ && !compact_column_to_orig_.empty() &&
      col_view_generation_seen_ == col_data->column_view_generation()) {
    // Cache hit: layout, source pointers AND published view unchanged.
    return;
  }

  // Build column->slot mapping. CUDAColumnData uses column index, not feature index.
  // For dense single-feature groups (Numerai), feature_to_column[f] returns the column index.
  const int num_features = static_cast<int>(bytree_mask.size());
  const int num_total_cols = static_cast<int>(row_data->host_feature_partition_column_index_offsets().back());

  orig_column_to_compact_slot_.assign(num_total_cols, -1);
  compact_column_to_orig_.clear();
  for (int f = 0; f < num_features; ++f) {
    if (!bytree_mask[f]) continue;
    int c = col_data->feature_to_column(f);
    if (c < 0 || c >= num_total_cols) continue;
    if (orig_column_to_compact_slot_[c] < 0) {
      orig_column_to_compact_slot_[c] = static_cast<int>(compact_column_to_orig_.size());
      compact_column_to_orig_.push_back(c);
    }
  }
  const int num_compact_cols = static_cast<int>(compact_column_to_orig_.size());
  if (num_compact_cols == 0) return;

  const data_size_t num_data = row_data->num_data();
  const size_t needed_bytes = static_cast<size_t>(num_compact_cols) * static_cast<size_t>(num_data);
  EnsureCompactColumnBuffer(needed_bytes, num_compact_cols, num_data);

  // Build per-slot source-frame metadata host-side (one entry per compact slot).
  // When the histogram constructor already gathered this tree's sampled columns
  // into its compact matrix (same column set, row-major-in-partition), gather
  // from THAT instead of the full bin matrix: the transpose then reads
  // ~fraction of the bytes (the full-matrix gather effectively streams the
  // whole dataset through L2 once per tree).
  const auto& src_part_col_offsets_h = row_data->host_feature_partition_column_index_offsets();
  const bool compact_src = cuda_histogram_constructor_->CompactViewSourceAvailable() &&
    // codec-packed compact views (pack_bit3/pack_radix*) are unreadable by the
    // split gather/packed readers; fall back to the original row data
    cuda_histogram_constructor_->compact_codec() == PackCodecId::kNibble4 &&
    cuda_histogram_constructor_->compact_src_cols() == compact_column_to_orig_;
  if (compact_src && cuda_histogram_constructor_->CompactColMajorFilled()) {
    // the histogram constructor's compact fill already produced the column-major
    // view as a fused second output (same slot order, verified above): point the
    // column data at it directly, no gather kernel at all
    col_data->SetCompactColumnView(orig_column_to_compact_slot_,
                                   const_cast<uint8_t*>(cuda_histogram_constructor_->compact_col_major_device()),
                                   static_cast<size_t>(num_data));
    compact_col_signature_ = sig;
    col_view_generation_seen_ = col_data->column_view_generation();
    return;
  }
  const uint8_t* gather_src = compact_src ? cuda_histogram_constructor_->compact_data_device() : src;
  const bool gather_src_is_4bit = compact_src ?
    cuda_histogram_constructor_->compact_src_is_4bit() : row_data->is_4bit_packed();
  std::vector<size_t> slot_p_byte_h(num_compact_cols);
  std::vector<int> slot_p_stride_h(num_compact_cols);
  std::vector<int> slot_col_in_p_h(num_compact_cols);
  if (compact_src) {
    // 8-bit: slot byte offsets already include the column-in-partition offset
    // (col entries are 0); 4-bit: byte offsets are the packed partition bases
    // and the logical column-in-partition rides in compact_slot_cols()
    slot_p_byte_h = cuda_histogram_constructor_->compact_slot_bytes();
    slot_p_stride_h = cuda_histogram_constructor_->compact_slot_strides();
    slot_col_in_p_h = cuda_histogram_constructor_->compact_slot_cols();
  } else {
    const std::vector<int>& packed_offsets_h = row_data->host_packed_partition_byte_offsets();
    for (int s = 0; s < num_compact_cols; ++s) {
      const int orig_col = compact_column_to_orig_[s];
      int p = 0;
      while (p + 1 < static_cast<int>(src_part_col_offsets_h.size()) &&
             orig_col >= src_part_col_offsets_h[p + 1]) ++p;
      const int p_start = src_part_col_offsets_h[p];
      const int p_end = src_part_col_offsets_h[p + 1];
      if (gather_src_is_4bit) {
        slot_p_byte_h[s] = static_cast<size_t>(packed_offsets_h[p]) * static_cast<size_t>(num_data);
        slot_p_stride_h[s] = packed_offsets_h[p + 1] - packed_offsets_h[p];
      } else {
        slot_p_byte_h[s] = static_cast<size_t>(p_start) * static_cast<size_t>(num_data);
        slot_p_stride_h[s] = p_end - p_start;
      }
      slot_col_in_p_h[s] = orig_col - p_start;
    }
  }
  if (cuda_slot_p_byte_.Size() < static_cast<size_t>(num_compact_cols)) {
    cuda_slot_p_byte_.Resize(num_compact_cols);
    cuda_slot_p_stride_.Resize(num_compact_cols);
    cuda_slot_col_in_p_.Resize(num_compact_cols);
  }
  CopyFromHostToCUDADevice<size_t>(cuda_slot_p_byte_.RawData(), slot_p_byte_h.data(),
                                   num_compact_cols, __FILE__, __LINE__);
  CopyFromHostToCUDADevice<int>(cuda_slot_p_stride_.RawData(), slot_p_stride_h.data(),
                                num_compact_cols, __FILE__, __LINE__);
  CopyFromHostToCUDADevice<int>(cuda_slot_col_in_p_.RawData(), slot_col_in_p_h.data(),
                                num_compact_cols, __FILE__, __LINE__);

  compact_packed_view_active_ = false;
  if (SplitPackedReadEnabled() && gather_src_is_4bit && HybridGrowthUsable() &&
      !has_categorical_feature_) {
    // categorical batched-apply descriptors need the plain per-column view
    // (SplitLevelBatched CHECK-enforces it); numerical-only datasets keep the packed read
    // packed split read: skip the column-major gather entirely; the batched
    // apply descriptors address the packed source per column (byte of row 0,
    // per-row byte stride, nibble shift). Sparse-encoded columns are the
    // exception: SetCompactPackedColumnView points them at their own
    // materialized buffer at their real width, because their encoding (the
    // most-frequent bin spelled as 0) never matches the row matrix. The slot
    // metadata just uploaded stays valid for the lazy classic fallback
    // (EnsureClassicColumnView), which any classic per-split Split() triggers
    // via ApplySplit.
    std::vector<size_t> col_base_byte_h(num_compact_cols);
    std::vector<uint8_t> col_shift_h(num_compact_cols);
    for (int s = 0; s < num_compact_cols; ++s) {
      col_base_byte_h[s] = slot_p_byte_h[s] + static_cast<size_t>(slot_col_in_p_h[s] >> 1);
      col_shift_h[s] = static_cast<uint8_t>((slot_col_in_p_h[s] & 1) << 2);
    }
    col_data->SetCompactPackedColumnView(orig_column_to_compact_slot_, gather_src,
                                         col_base_byte_h, slot_p_stride_h, col_shift_h);
    compact_packed_view_active_ = true;
    compact_gather_src_ = gather_src;
    compact_gather_src_is_4bit_ = gather_src_is_4bit;
    compact_col_signature_ = sig;
    col_view_generation_seen_ = col_data->column_view_generation();
    return;
  }

  LaunchRowToColCompactKernel(
      0,
      gather_src,
      compact_column_buffer_.RawData(),
      cuda_slot_p_byte_.RawData(),
      cuda_slot_p_stride_.RawData(),
      cuda_slot_col_in_p_.RawData(),
      num_compact_cols,
      num_data,
      gather_src_is_4bit);
  CUDASUCCESS_OR_FATAL(cudaDeviceSynchronize());

  // Update CUDAColumnData's data_by_column_ pointers to slots in compact buffer.
  col_data->SetCompactColumnView(orig_column_to_compact_slot_,
                                 compact_column_buffer_.RawData(),
                                 static_cast<size_t>(num_data));
  compact_col_signature_ = sig;
  col_view_generation_seen_ = col_data->column_view_generation();
}

void CUDASingleGPUTreeLearner::EnsureCompactColumnBuffer(
    const size_t needed_bytes, const int num_compact_cols, const data_size_t num_data) {
  if (compact_column_buffer_.Size() >= needed_bytes) {
    return;
  }
  // Resize allocates the new block before releasing the old one, so the peak is
  // the sum. Check before asking: this buffer is the largest single allocation
  // training makes on a wide dataset, and cudaMalloc failing here reports only
  // "[CUDA] out of memory" with no indication of what wanted the memory or how
  // much. Note the one byte per value: a 4-bit dataset is NOT packed here, so
  // this view costs twice what the dataset itself does.
  size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  const size_t peak_bytes = needed_bytes + compact_column_buffer_.Size();
  const size_t headroom = 256ULL << 20;
  if (free_bytes < peak_bytes + headroom) {
    Log::Fatal(
      "Not enough GPU memory for the per-tree column view: %.2f GiB needed "
      "(%d features x %d rows, one byte per value), %.2f GiB free of %.2f GiB. "
      "This view is not bit-packed, so it costs one byte per value even when the "
      "dataset itself is 4-bit. Lower feature_fraction to shrink it, train on "
      "fewer rows, or use a card with more memory.",
      peak_bytes / static_cast<double>(1ULL << 30), num_compact_cols,
      static_cast<int>(num_data), free_bytes / static_cast<double>(1ULL << 30),
      total_bytes / static_cast<double>(1ULL << 30));
  }
  compact_column_buffer_.Resize(needed_bytes);
}

void CUDASingleGPUTreeLearner::EnsureClassicColumnView() {
  if (!compact_packed_view_active_) {
    return;
  }
  // run the gather that BuildCompactColumnView skipped for the packed split
  // read; once per tree at most (the classic view stays registered after)
  const data_size_t num_data = cuda_histogram_constructor_->cuda_row_data_internal()->num_data();
  const int num_compact_cols = static_cast<int>(compact_column_to_orig_.size());
  const size_t needed_bytes = static_cast<size_t>(num_compact_cols) * static_cast<size_t>(num_data);
  EnsureCompactColumnBuffer(needed_bytes, num_compact_cols, num_data);
  LaunchRowToColCompactKernel(
      0,
      compact_gather_src_,
      compact_column_buffer_.RawData(),
      cuda_slot_p_byte_.RawData(),
      cuda_slot_p_stride_.RawData(),
      cuda_slot_col_in_p_.RawData(),
      num_compact_cols,
      num_data,
      compact_gather_src_is_4bit_);
  CUDASUCCESS_OR_FATAL(cudaDeviceSynchronize());
  CUDAColumnData* col_data = const_cast<CUDAColumnData*>(train_data_->cuda_column_data());
  col_data->SetCompactColumnView(orig_column_to_compact_slot_,
                                 compact_column_buffer_.RawData(),
                                 static_cast<size_t>(num_data));
  compact_packed_view_active_ = false;
}

void CUDASingleGPUTreeLearner::AddPredictionToScore(const Tree* tree, double* out_score) const {
  if (config_->linear_tree && last_tree_is_linear_ && tree->is_linear()) {
    // Linear tree: score += leaf_const + sum(coeff * raw_feat), NaN -> leaf_output.
    // Build the device model from the tree's current (post-shrinkage) coeffs, then
    // reuse the just-trained tree's per-data leaf assignment held by the data partition.
    BuildAndUploadLinearScoreModel(tree);
    LaunchLinearAddScoreKernel(
      cuda_data_partition_->cuda_data_index_to_leaf_index(),
      cuda_linear_leaf_const_.RawData(),
      cuda_linear_leaf_coeff_.RawData(),
      cuda_linear_leaf_coeff_offset_.RawData(),
      cuda_linear_leaf_coeff_col_.RawData(),
      cuda_linear_leaf_num_coeff_.RawData(),
      cuda_linear_leaf_output_.RawData(),
      out_score);
    return;
  }
  cuda_data_partition_->UpdateTrainScore(tree, out_score);
}

namespace {

// FALCATA_DEBUG=aggressive opts into plain level batching in budget-limited
// configs, accepting the approximate (breadth-biased) growth policy in exchange
// for hybrid speed. Off by default; exact for depth-limited configs either way.
bool HybridAggressiveEnv() {
  return FalcataDebug().aggressive;
}

}  // anonymous namespace

bool CUDASingleGPUTreeLearner::HybridGrowthUsable() const {
  // Plain level batching is exactly leaf-wise-equivalent only in the
  // depth-limited regime (2^max_depth <= num_leaves + 1): there the budget can
  // only bind at the final (max-depth) level, where the batched top-K-by-gain
  // selection is optimal because no deeper candidates can exist. In
  // budget-limited configs (num_leaves << 2^max_depth) the SELECTIVE
  // grow-then-prune mode (UseSelectiveGrowth) provides exact leaf-wise
  // equivalence; where it does not apply, FALCATA_HYBRID_AGGRESSIVE=1 opts
  // into the approximate (breadth-biased) plain batching, and otherwise the
  // classic one-split-at-a-time loop runs.
  const bool depth_limited = config_->max_depth > 0 && config_->max_depth < 31 &&
      (1LL << config_->max_depth) <= static_cast<int64_t>(config_->num_leaves) + 1;
  // Categorical datasets are allowed on the (two-sync) hybrid prefix: the
  // batched apply routes categorical splits through the
  // per-level bitset arena, and tree recording interleaves the per-split
  // categorical path with numerical SplitBatch chunks. The selective and
  // one-sync/speculative flows keep their categorical exclusions for now.
  // The global-memory finder stages per-feature scratch in buffers shared by
  // every in-flight launch; the classic loop serializes those launches with a
  // device sync per split, but the hybrid per-pair fallback overlaps pairs and
  // trampled the staging (>256-bin features trained garbage). The batched
  // level kernels are excluded for global memory anyway, so hybrid offers no
  // upside on these shapes: route them to the classic flow.
  // FENCE: hybrid growth is disabled for every categorical dataset. It emits
  // categorical split thresholds containing invalid bins, and on sm_86 the
  // level bitset arena hits an illegal access consuming a valid-flagged
  // candidate whose num_cat_threshold vanished -- at the DEFAULT
  // max_cat_threshold, with both the batched-level and per-pair finders, one-
  // sync and two-sync alike. Root cause is still open (see ROADMAP); the
  // classic loop is verified correct here, so categorical data takes it and
  // gives up hybrid's ~3.7x on those shapes until this is fixed.
  const bool cat_threshold_fenced =
      has_categorical_feature_ && !FalcataPlan::Get().cat_hybrid;
  if (cat_threshold_fenced) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      Log::Warning("cat_hybrid:off -- categorical dataset routed to the "
                   "classic CUDA training loop.");
    }
  }
  // Multi-GPU rides the hybrid TWO-SYNC batched flow only: that is the flow
  // whose per-level structure allows ONE grouped all-reduce per level instead
  // of one per split, which is the dominant multi-GPU cost. The one-sync,
  // graph and selective flows derive leaf counts on-device or reorder the
  // readback, and their rank-visible decisions have not been made global, so
  // they stay single-GPU (see UseOneSyncPrefix / HybridGraphPrefixUsable /
  // UseSelectiveGrowth).
  const bool nccl_ok = nccl_communicator_ == nullptr ||
      (cuda_histogram_constructor_ != nullptr && cuda_best_split_finder_ != nullptr &&
       use_hybrid_batch_kernels_ && use_hybrid_batch_apply_ &&
       cuda_histogram_constructor_->SupportsBatchedLevel() &&
       cuda_best_split_finder_->SupportsBatchedLevel());
  // Vector-leaf mode has one level flow: the two-sync prefix with the batched
  // level kernels (per-plane construct/fix/subtract + the batched vector find)
  // and the batched apply, which is where the per-level plane fan-out lives.
  const bool vec_ok = vec_num_targets_ <= 1 ||
      (use_hybrid_batch_kernels_ && use_hybrid_batch_apply_ &&
       cuda_histogram_constructor_ != nullptr && cuda_best_split_finder_ != nullptr &&
       cuda_histogram_constructor_->SupportsBatchedLevel() &&
       cuda_best_split_finder_->SupportsBatchedLevel());
  const bool base = use_hybrid_growth_ &&
         nccl_ok &&
         vec_ok &&
         !select_features_by_node_ &&
         !cat_threshold_fenced &&
         (cuda_best_split_finder_ == nullptr || !cuda_best_split_finder_->use_global_memory()) &&
         (forced_split_json_ == nullptr || forced_split_json_->is_null()) &&
         config_->num_leaves > 2;
  if (!base) {
    return false;
  }
  return depth_limited || HybridAggressiveEnv() || UseSelectiveGrowth();
}

bool CUDASingleGPUTreeLearner::UseSelectiveGrowth() const {
  // multi-GPU: selective grow-then-prune keeps its own host-side leaf bookkeeping, none of which is global
  if (nccl_communicator_ != nullptr) {
    return false;
  }
  // feature-parallel: the per-level winner merge lives in the two-sync loop
  if (fp_merge_state_ != nullptr) {
    return false;
  }
  const bool depth_limited = config_->max_depth > 0 && config_->max_depth < 31 &&
      (1LL << config_->max_depth) <= static_cast<int64_t>(config_->num_leaves) + 1;
  if (depth_limited || HybridAggressiveEnv()) {
    // depth-limited configs keep the (bit-exact-locked) exact-fit prefix;
    // AGGRESSIVE explicitly selects the approximate plain batching
    return false;
  }
  // Selective growth needs the batched apply phase (host-assigned, recyclable
  // right-child leaf indices) and is only the default when both batching env
  // switches are untouched: any explicit fallback env reproduces the previous
  // behavior (classic loop for budget-limited configs). Linear trees renumber
  // leaves through their own device paths, so they keep the classic loop.
  return use_hybrid_selective_ &&
         use_hybrid_batch_apply_ &&
         use_hybrid_batch_kernels_ &&
         !config_->linear_tree &&
         nccl_communicator_ == nullptr &&
         !select_features_by_node_ &&
         (forced_split_json_ == nullptr || forced_split_json_->is_null()) &&
         config_->num_leaves > 2;
}

bool CUDASingleGPUTreeLearner::UseOneSyncPrefix() const {
  // multi-GPU: the one-sync flow derives leaf counts on-device, so ranks cannot agree host-side
  if (nccl_communicator_ != nullptr) {
    return false;
  }
  // feature-parallel: the host merge needs the two-sync readback each level
  if (fp_merge_state_ != nullptr) {
    return false;
  }
  // categorical datasets ride the one-sync flow: the apply
  // runs AFTER the level's single readback (host split infos, including
  // num_cat_threshold, are available -- the phase-1b operator= fix keeps the
  // count through host copies), and the speculative children's search uses the
  // batched level finder, categorical-capable since phase 2b. The graph loop
  // keeps its categorical exclusion (arena build is host-driven).
  return use_hybrid_one_sync_ &&
         !config_->use_quantized_grad &&
         use_hybrid_batch_kernels_ &&
         use_hybrid_batch_apply_ &&
         cuda_histogram_constructor_->SupportsBatchedLevel() &&
         cuda_best_split_finder_->SupportsBatchedLevel();
}

void CUDASingleGPUTreeLearner::EnsureRootSumsReadBack(CUDATree* tree) {
  if (!root_sums_deferred_) {
    return;
  }
  root_sums_deferred_ = false;
  cuda_smaller_leaf_splits_->CopyRootSumsToHost(&leaf_sum_gradients_[0], &leaf_sum_hessians_[0]);
  // the deferred half of Train()'s root initialization (see there)
  tree->SetLeafOutput(0, CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(
    leaf_sum_gradients_[0], leaf_sum_hessians_[0],
    config_->lambda_l1, config_->lambda_l2, config_->path_smooth, config_->max_delta_step,
    static_cast<data_size_t>(num_data_), 0.0));
  tree->SyncLeafOutputFromHostToCUDA();
}

void CUDASingleGPUTreeLearner::EnqueuePairBestSplitSearch(const CUDATree* tree,
    const CUDALeafSplitsStruct* smaller_struct, const CUDALeafSplitsStruct* larger_struct,
    const int smaller_leaf_index, const int larger_leaf_index, const bool synchronize,
    const int pipeline, const int parent_leaf_index_arg) {
  cuda_histogram_constructor_->SetActivePipeline(pipeline);
  cuda_best_split_finder_->SetActiveHistPipeline(pipeline);
  global_timer.Start("CUDASingleGPUTreeLearner::ConstructHistogramForLeaf");
  const data_size_t global_num_data_in_smaller_leaf = nccl_communicator_ != nullptr ?
    global_num_data_in_leaf_[smaller_leaf_index] :
    leaf_num_data_[smaller_leaf_index];
  const data_size_t global_num_data_in_larger_leaf = nccl_communicator_ != nullptr ?
    (larger_leaf_index < 0 ? 0 : global_num_data_in_leaf_[larger_leaf_index]) :
    (larger_leaf_index < 0 ? 0 : leaf_num_data_[larger_leaf_index]);
  const data_size_t num_data_in_smaller_leaf = leaf_num_data_[smaller_leaf_index];
  const data_size_t num_data_in_larger_leaf = larger_leaf_index < 0 ? 0 : leaf_num_data_[larger_leaf_index];
  const double sum_hessians_in_smaller_leaf = leaf_sum_hessians_[smaller_leaf_index];
  const double sum_hessians_in_larger_leaf = larger_leaf_index < 0 ? 0 : leaf_sum_hessians_[larger_leaf_index];
  const uint8_t num_bits_in_histogram_bins = config_->use_quantized_grad ? (nccl_communicator_ != nullptr ?
      cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(smaller_leaf_index) :
      cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(smaller_leaf_index)) : 0;
  cuda_histogram_constructor_->ConstructHistogramForLeaf(
    smaller_struct,
    larger_struct,
    global_num_data_in_smaller_leaf,
    global_num_data_in_larger_leaf,
    num_data_in_smaller_leaf,
    num_data_in_larger_leaf,
    sum_hessians_in_smaller_leaf,
    sum_hessians_in_larger_leaf,
    num_bits_in_histogram_bins);
  global_timer.Stop("CUDASingleGPUTreeLearner::ConstructHistogramForLeaf");

  global_timer.Start("CUDASingleGPUTreeLearner::NCCLReduceHistogram");
  if (nccl_communicator_ != nullptr) {
    NCCLReduceHistogram(pipeline);
  }
  global_timer.Stop("CUDASingleGPUTreeLearner::NCCLReduceHistogram");

  global_timer.Start("CUDASingleGPUTreeLearner::FindBestSplitsForLeaf");
  uint8_t parent_num_bits_bin = 0;
  uint8_t smaller_num_bits_bin = 0;
  uint8_t larger_num_bits_bin = 0;
  if (config_->use_quantized_grad) {
    if (larger_leaf_index != -1) {
      const int parent_leaf_index = parent_leaf_index_arg >= 0 ?
        parent_leaf_index_arg : std::min(smaller_leaf_index, larger_leaf_index);
      if (nccl_communicator_ != nullptr) {
        parent_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInNode<true>(parent_leaf_index);
        smaller_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(smaller_leaf_index);
        larger_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(larger_leaf_index);
      } else {
        parent_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInNode<false>(parent_leaf_index);
        smaller_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(smaller_leaf_index);
        larger_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(larger_leaf_index);
      }
    } else {
      if (nccl_communicator_ != nullptr) {
        parent_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(0);
        smaller_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(0);
        larger_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(0);
      } else {
        parent_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(0);
        smaller_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(0);
        larger_num_bits_bin = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(0);
      }
    }
  }
  cuda_histogram_constructor_->SubtractHistogramForLeaf(
    smaller_struct,
    larger_struct,
    config_->use_quantized_grad,
    parent_num_bits_bin,
    smaller_num_bits_bin,
    larger_num_bits_bin);

  SelectFeatureByNode(tree);
  // monotone constraints for this pair (identity bounds when unused)
  const double smaller_leaf_constraint_min = use_monotone_constraints_ ?
    leaf_constraint_min_[smaller_leaf_index] : -std::numeric_limits<double>::max();
  const double smaller_leaf_constraint_max = use_monotone_constraints_ ?
    leaf_constraint_max_[smaller_leaf_index] : std::numeric_limits<double>::max();
  const double larger_leaf_constraint_min = (use_monotone_constraints_ && larger_leaf_index >= 0) ?
    leaf_constraint_min_[larger_leaf_index] : -std::numeric_limits<double>::max();
  const double larger_leaf_constraint_max = (use_monotone_constraints_ && larger_leaf_index >= 0) ?
    leaf_constraint_max_[larger_leaf_index] : std::numeric_limits<double>::max();

  if (config_->use_quantized_grad) {
    const uint8_t smaller_leaf_num_bits_bin = nccl_communicator_ == nullptr ?
      cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(smaller_leaf_index) :
      cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(smaller_leaf_index);
    const uint8_t larger_leaf_num_bits_bin = larger_leaf_index < 0 ? 32 : (nccl_communicator_ == nullptr ?
      cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(larger_leaf_index) :
      cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(larger_leaf_index));
    cuda_best_split_finder_->FindBestSplitsForLeaf(
      smaller_struct,
      larger_struct,
      smaller_leaf_index, larger_leaf_index,
      global_num_data_in_smaller_leaf, global_num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf,
      cuda_gradient_discretizer_->grad_scale_ptr(),
      cuda_gradient_discretizer_->hess_scale_ptr(),
      smaller_leaf_num_bits_bin,
      larger_leaf_num_bits_bin,
      config_->max_depth <= 0 || GrowthLeafDepth(tree, smaller_leaf_index) < config_->max_depth,
      larger_leaf_index < 0 || config_->max_depth <= 0 ||
        GrowthLeafDepth(tree, larger_leaf_index) < config_->max_depth,
      smaller_leaf_constraint_min, smaller_leaf_constraint_max,
      larger_leaf_constraint_min, larger_leaf_constraint_max,
      synchronize);
  } else {
    cuda_best_split_finder_->FindBestSplitsForLeaf(
      smaller_struct,
      larger_struct,
      smaller_leaf_index, larger_leaf_index,
      global_num_data_in_smaller_leaf, global_num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf,
      nullptr, nullptr, 0, 0,
      config_->max_depth <= 0 || GrowthLeafDepth(tree, smaller_leaf_index) < config_->max_depth,
      larger_leaf_index < 0 || config_->max_depth <= 0 ||
        GrowthLeafDepth(tree, larger_leaf_index) < config_->max_depth,
      smaller_leaf_constraint_min, smaller_leaf_constraint_max,
      larger_leaf_constraint_min, larger_leaf_constraint_max,
      synchronize);
  }
  global_timer.Stop("CUDASingleGPUTreeLearner::FindBestSplitsForLeaf");
  cuda_histogram_constructor_->SetActivePipeline(0);
  cuda_best_split_finder_->SetActiveHistPipeline(0);
}

void CUDASingleGPUTreeLearner::CheckVectorLeafSupported() const {
  if (vec_num_targets_ > 16) {
    Log::Fatal("tree_mode=vector_leaf supports at most 16 targets "
               "(num_class=%d requested).", vec_num_targets_);
  }
  if (has_categorical_feature_ && config_->cat_random_search > 0) {
    Log::Fatal("tree_mode=vector_leaf multi-target training does not support "
               "cat_random_search yet.");
  }
  if (cuda_best_split_finder_->use_global_memory()) {
    Log::Fatal("tree_mode=vector_leaf multi-target training requires every "
               "feature to fit %d histogram bins (reduce max_bin).",
               NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER);
  }
  if (select_features_by_node_) {
    Log::Fatal("tree_mode=vector_leaf multi-target training does not support "
               "interaction constraints or feature_fraction_bynode.");
  }
  if (nccl_communicator_ != nullptr || fp_merge_state_ != nullptr) {
    Log::Fatal("tree_mode=vector_leaf multi-target training is single-GPU only.");
  }
  if (cuda_histogram_constructor_->hist_fp32()) {
    Log::Fatal("tree_mode=vector_leaf multi-target training requires "
               "cuda_precision=fp64.");
  }
}

void CUDASingleGPUTreeLearner::VectorInitRootPlanes(
    const data_size_t* leaf_splits_init_indices, const data_size_t root_num_data) {
  const int num_targets = vec_num_targets_;
  CUDALeafSplitsStruct* planes = vec_plane_structs_.RawData();
  // plane 0 is exactly the primary init BeforeTrain just ran
  CopyFromCUDADeviceToCUDADevice<CUDALeafSplitsStruct>(
    planes, cuda_smaller_leaf_splits_->GetCUDAStruct(), 1, __FILE__, __LINE__);
  vec_root_sum_gradients_[0] = leaf_sum_gradients_[0];
  vec_root_sum_hessians_[0] = leaf_sum_hessians_[0];
  // planes 1..T-1: rerun the root reduction on each target's gradient slice
  // into the primary struct, snapshot it into the plane slab, then restore
  // plane 0 as the primary contents
  for (int t = 1; t < num_targets; ++t) {
    cuda_smaller_leaf_splits_->InitValues(
      config_->lambda_l1,
      config_->lambda_l2,
      config_->max_delta_step,
      gradients_ + static_cast<size_t>(t) * num_data_,
      hessians_ + static_cast<size_t>(t) * num_data_,
      leaf_splits_init_indices,
      cuda_data_partition_->cuda_data_indices(),
      root_num_data,
      cuda_histogram_constructor_->cuda_hist_pointer() +
        static_cast<size_t>(t) * 2 * num_total_bin_,
      &vec_root_sum_gradients_[t],
      &vec_root_sum_hessians_[t]);
    CopyFromCUDADeviceToCUDADevice<CUDALeafSplitsStruct>(
      planes + t, cuda_smaller_leaf_splits_->GetCUDAStruct(), 1, __FILE__, __LINE__);
  }
  if (num_targets > 1) {
    CopyFromCUDADeviceToCUDADevice<CUDALeafSplitsStruct>(
      cuda_smaller_leaf_splits_->GetCUDAStructRef(), planes, 1, __FILE__, __LINE__);
  }
  // larger-leaf planes start empty, like the primary larger struct
  for (int t = 0; t < num_targets; ++t) {
    CopyFromCUDADeviceToCUDADevice<CUDALeafSplitsStruct>(
      planes + num_targets + t, cuda_larger_leaf_splits_->GetCUDAStruct(), 1, __FILE__, __LINE__);
  }
}

void CUDASingleGPUTreeLearner::EnqueuePairBestSplitSearchVector(const CUDATree* tree,
    const int smaller_leaf_index, const int larger_leaf_index) {
  const int num_targets = vec_num_targets_;
  CUDALeafSplitsStruct* planes = vec_plane_structs_.RawData();
  const data_size_t num_data_in_smaller_leaf = leaf_num_data_[smaller_leaf_index];
  const data_size_t num_data_in_larger_leaf =
    larger_leaf_index < 0 ? 0 : leaf_num_data_[larger_leaf_index];
  const double sum_hessians_in_smaller_leaf = leaf_sum_hessians_[smaller_leaf_index];
  const double sum_hessians_in_larger_leaf =
    larger_leaf_index < 0 ? 0 : leaf_sum_hessians_[larger_leaf_index];
  global_timer.Start("CUDASingleGPUTreeLearner::ConstructHistogramForLeaf");
  for (int t = 0; t < num_targets; ++t) {
    cuda_histogram_constructor_->SelectGradientPlane(
      gradients_ + static_cast<size_t>(t) * num_data_,
      hessians_ + static_cast<size_t>(t) * num_data_);
    cuda_histogram_constructor_->ConstructHistogramForLeaf(
      planes + t,
      planes + num_targets + t,
      num_data_in_smaller_leaf,
      num_data_in_larger_leaf,
      num_data_in_smaller_leaf,
      num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf,
      sum_hessians_in_larger_leaf,
      0);
    cuda_histogram_constructor_->SubtractHistogramForLeaf(
      planes + t,
      planes + num_targets + t,
      false, 0, 0, 0);
  }
  cuda_histogram_constructor_->SelectGradientPlane(gradients_, hessians_);
  global_timer.Stop("CUDASingleGPUTreeLearner::ConstructHistogramForLeaf");
  global_timer.Start("CUDASingleGPUTreeLearner::FindBestSplitsForLeaf");
  cuda_best_split_finder_->FindBestSplitsForLeafVector(
    planes,
    planes + num_targets,
    smaller_leaf_index, larger_leaf_index,
    num_data_in_smaller_leaf, num_data_in_larger_leaf,
    sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf,
    config_->max_depth <= 0 || tree->leaf_depth(smaller_leaf_index) < config_->max_depth,
    larger_leaf_index < 0 || config_->max_depth <= 0 ||
      tree->leaf_depth(larger_leaf_index) < config_->max_depth,
    /*synchronize=*/true);
  global_timer.Stop("CUDASingleGPUTreeLearner::FindBestSplitsForLeaf");
}

void CUDASingleGPUTreeLearner::VectorLevelPlaneFanOut(
    const std::vector<HybridAppliedSplit>& applied) {
  const int num_pairs = static_cast<int>(applied.size());
  if (num_pairs <= 0) {
    return;
  }
  host_vec_fanout_infos_.resize(static_cast<size_t>(num_pairs));
  host_vec_fanout_left_leaves_.resize(static_cast<size_t>(num_pairs));
  for (int k = 0; k < num_pairs; ++k) {
    // the fan-out kernel addresses the pair's two primary slots as
    // hybrid_pair_slots_[2k, 2k + 1], the layout ApplyLevelBatched assigns
    CHECK_EQ(applied[k].smaller_slot, hybrid_pair_slots_.RawData() + 2 * k);
    host_vec_fanout_infos_[k] =
      cuda_best_split_finder_->leaf_best_split_info_ptr(applied[k].left);
    host_vec_fanout_left_leaves_[k] = applied[k].left;
  }
  if (cuda_vec_fanout_infos_.Size() < static_cast<size_t>(num_pairs)) {
    const size_t capacity = static_cast<size_t>(
      std::max(num_pairs, config_->num_leaves / 2 + 2));
    cuda_vec_fanout_infos_.Resize(capacity);
    cuda_vec_fanout_left_leaves_.Resize(capacity);
  }
  CopyFromHostToCUDADevice<const CUDASplitInfo*>(cuda_vec_fanout_infos_.RawData(),
    host_vec_fanout_infos_.data(), static_cast<size_t>(num_pairs), __FILE__, __LINE__);
  CopyFromHostToCUDADevice<int>(cuda_vec_fanout_left_leaves_.RawData(),
    host_vec_fanout_left_leaves_.data(), static_cast<size_t>(num_pairs), __FILE__, __LINE__);
  LaunchVectorLevelPlaneFanOutKernel(num_pairs);
}

void CUDASingleGPUTreeLearner::EnqueueLevelHistogramsAndFindVector(
    const int num_pairs,
    const data_size_t max_num_data_in_smaller_leaf,
    CUDALeafSplitsStruct* plane_slab) {
  const int num_targets = vec_num_targets_;
  (void)max_num_data_in_smaller_leaf;
  // Histogram phase: PER PAIR, all T planes of a pair back to back -- the
  // per-leaf construct sizing and the T launches' shared row working set. The
  // batched level construct is what makes plain level batching pay for scalar
  // training (one launch, one shared saturation floor over the level's pairs),
  // but with T gradient planes it measures 110.8 ms/tree against this loop's
  // 68.1 on 200k x 200 / T=5 / 63 leaves, which is the whole of the level
  // flow's cost there. What the level flow contributes for vector mode is the
  // ONE find + ONE sync + ONE apply below (find 3.1 ms/tree against the
  // per-pair 4.7, plus the per-split device syncs the classic loop pays).
  for (int i = 0; i < num_pairs; ++i) {
    const CUDAHybridPairDescriptor& desc = host_hybrid_pair_descs_[i];
    if (desc.construct_valid == 0) {
      continue;
    }
    const double sum_hessians_in_smaller_leaf =
      leaf_sum_hessians_[desc.smaller_leaf_index];
    const double sum_hessians_in_larger_leaf = desc.larger_leaf_index < 0 ?
      0.0 : leaf_sum_hessians_[desc.larger_leaf_index];
    CUDALeafSplitsStruct* smaller_planes =
      plane_slab + static_cast<size_t>(2 * i) * num_targets;
    CUDALeafSplitsStruct* larger_planes =
      plane_slab + (static_cast<size_t>(2 * i) + 1) * num_targets;
    for (int t = 0; t < num_targets; ++t) {
      cuda_histogram_constructor_->SelectGradientPlane(
        gradients_ + static_cast<size_t>(t) * num_data_,
        hessians_ + static_cast<size_t>(t) * num_data_);
      cuda_histogram_constructor_->ConstructHistogramForLeaf(
        smaller_planes + t, larger_planes + t,
        desc.num_data_in_smaller_leaf, desc.num_data_in_larger_leaf,
        desc.num_data_in_smaller_leaf, desc.num_data_in_larger_leaf,
        sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf, 0);
      cuda_histogram_constructor_->SubtractHistogramForLeaf(
        smaller_planes + t, larger_planes + t, false, 0, 0, 0);
    }
  }
  cuda_histogram_constructor_->SelectGradientPlane(gradients_, hessians_);
  cuda_best_split_finder_->FindBestSplitsForLevelVector(
    cuda_hybrid_pair_descs_.RawDataReadOnly(), num_pairs, plane_slab);
}

void CUDASingleGPUTreeLearner::EnqueueLevelBestSplitSearch(const CUDATree* tree,
    const std::vector<HybridPendingPair>& pairs,
    CUDALeafSplitsStruct* vec_plane_slab) {
  const int num_pairs = static_cast<int>(pairs.size());
  if (num_pairs <= 0) {
    return;
  }
  global_timer.Start("CUDASingleGPUTreeLearner::EnqueueLevelBestSplitSearch");
  host_hybrid_pair_descs_.resize(static_cast<size_t>(num_pairs));
  data_size_t max_num_data_in_smaller_leaf = 0;
  bool any_bit_change_copy = false;
  for (int i = 0; i < num_pairs; ++i) {
    const HybridPendingPair& pair = pairs[i];
    CUDAHybridPairDescriptor& desc = host_hybrid_pair_descs_[i];
    desc.smaller_struct = pair.smaller_struct;
    desc.larger_struct = pair.larger_struct;
    desc.smaller_leaf_index = pair.smaller;
    desc.larger_leaf_index = pair.larger;
    // Under NCCL every rank must reach the SAME validity verdict, so these
    // gates read the global counts; the local ones would differ per rank and
    // desynchronise the collectives (that is how the classic path deadlocked).
    const data_size_t num_data_in_smaller_leaf = nccl_communicator_ != nullptr ?
        global_num_data_in_leaf_[pair.smaller] : leaf_num_data_[pair.smaller];
    const data_size_t num_data_in_larger_leaf = pair.larger < 0 ? 0 :
        (nccl_communicator_ != nullptr ? global_num_data_in_leaf_[pair.larger]
                                       : leaf_num_data_[pair.larger]);
    const double sum_hessians_in_smaller_leaf = leaf_sum_hessians_[pair.smaller];
    const double sum_hessians_in_larger_leaf = pair.larger < 0 ? 0.0 : leaf_sum_hessians_[pair.larger];
    desc.num_data_in_smaller_leaf = num_data_in_smaller_leaf;
    desc.num_data_in_larger_leaf = num_data_in_larger_leaf;
    if (nccl_communicator_ != nullptr && FalcataDebug().dump) {
      Log::Warning("[enq rank%d] pair s=%d l=%d desc_n=(%d,%d) local=(%d,%d) gN=%d",
                   nccl_gpu_rank_, pair.smaller, pair.larger,
                   num_data_in_smaller_leaf, num_data_in_larger_leaf,
                   leaf_num_data_[pair.smaller],
                   pair.larger < 0 ? 0 : leaf_num_data_[pair.larger],
                   global_num_data_);
    }
    // mirror of ConstructHistogramForLeaf's min_data/min_hessian early return
    desc.construct_valid =
      ((num_data_in_smaller_leaf <= config_->min_data_in_leaf ||
        sum_hessians_in_smaller_leaf <= config_->min_sum_hessian_in_leaf) &&
       (num_data_in_larger_leaf <= config_->min_data_in_leaf ||
        sum_hessians_in_larger_leaf <= config_->min_sum_hessian_in_leaf)) ? 0 : 1;
    // mirror of CUDABestSplitFinder::FindBestSplitsForLeaf's leaf validity checks
    const bool smaller_below_max_depth =
      config_->max_depth <= 0 || GrowthLeafDepth(tree, pair.smaller) < config_->max_depth;
    const bool larger_below_max_depth = pair.larger < 0 ||
      config_->max_depth <= 0 || GrowthLeafDepth(tree, pair.larger) < config_->max_depth;
    desc.smaller_valid = (num_data_in_smaller_leaf > config_->min_data_in_leaf &&
                          sum_hessians_in_smaller_leaf > config_->min_sum_hessian_in_leaf &&
                          smaller_below_max_depth) ? 1 : 0;
    desc.larger_valid = (pair.larger >= 0 &&
                         num_data_in_larger_leaf > config_->min_data_in_leaf &&
                         sum_hessians_in_larger_leaf > config_->min_sum_hessian_in_leaf &&
                         larger_below_max_depth) ? 1 : 0;
    if (config_->use_quantized_grad) {
      if (pair.larger >= 0) {
        // pair.parent == the split's left leaf == min(smaller, larger) unless
        // selective growth recycled the right-child index
        const int parent_leaf_index = pair.parent >= 0 ?
          pair.parent : std::min(pair.smaller, pair.larger);
        // Under NCCL the bit widths must come from the GLOBAL counts: they
        // select the all-reduce element type AND must guarantee the summed
        // (global) histogram fits the chosen width -- the local table would
        // pick 16-bit where the global sum needs 32 and differ per rank.
        if (nccl_communicator_ != nullptr) {
          desc.parent_num_bits = cuda_gradient_discretizer_->GetHistBitsInNode<true>(parent_leaf_index);
          desc.smaller_num_bits = cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(pair.smaller);
          desc.larger_num_bits = cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(pair.larger);
        } else {
          desc.parent_num_bits = cuda_gradient_discretizer_->GetHistBitsInNode<false>(parent_leaf_index);
          desc.smaller_num_bits = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(pair.smaller);
          desc.larger_num_bits = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(pair.larger);
        }
        if (desc.parent_num_bits > 16 && desc.larger_num_bits <= 16) {
          any_bit_change_copy = true;
        }
      } else {
        desc.parent_num_bits = nccl_communicator_ != nullptr ?
          cuda_gradient_discretizer_->GetHistBitsInLeaf<true>(0) :
          cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(0);
        desc.smaller_num_bits = desc.parent_num_bits;
        desc.larger_num_bits = desc.parent_num_bits;
      }
    } else {
      desc.parent_num_bits = 0;
      desc.smaller_num_bits = 0;
      desc.larger_num_bits = 0;
    }
    if (num_data_in_smaller_leaf > max_num_data_in_smaller_leaf) {
      max_num_data_in_smaller_leaf = num_data_in_smaller_leaf;
    }
  }
  if (cuda_hybrid_pair_descs_.Size() < static_cast<size_t>(num_pairs)) {
    // preallocate for the deepest possible level so this resizes at most once
    cuda_hybrid_pair_descs_.Resize(static_cast<size_t>(
      std::max(num_pairs, config_->num_leaves / 2 + 2)));
  }
  // single async H2D per level on the histogram stream: ordered before the
  // construct kernel on the same stream, and before the find/sync kernels via
  // subtract_done_events_[0]. The host staging buffer is only rewritten after
  // the next FinishSplitBatch/SyncAllLeafBestSplitsToHost full sync, and a
  // pageable async H2D returns only once the data is staged, so reuse is safe.
  CopyFromHostToCUDADeviceAsync<CUDAHybridPairDescriptor>(
    cuda_hybrid_pair_descs_.RawData(), host_hybrid_pair_descs_.data(),
    static_cast<size_t>(num_pairs), cuda_histogram_constructor_->hist_stream(),
    __FILE__, __LINE__);
  if (vec_num_targets_ > 1) {
    CHECK(vec_plane_slab != nullptr);
    EnqueueLevelHistogramsAndFindVector(num_pairs, max_num_data_in_smaller_leaf,
                                        vec_plane_slab);
    global_timer.Stop("CUDASingleGPUTreeLearner::EnqueueLevelBestSplitSearch");
    return;
  }
  // Multi-GPU defers the fix+subtract tail: the mfb fix writes
  // leaf_total - sum(other bins) with leaf-struct totals that are GLOBAL under
  // NCCL, and the subtract derives the larger child from the (global) parent,
  // so both must run on globally reduced smaller histograms. Order mirrors the
  // classic flow: construct -> reduce -> fix/subtract -> find.
  cuda_histogram_constructor_->ConstructHistogramsForLevel(
    cuda_hybrid_pair_descs_.RawDataReadOnly(), num_pairs,
    max_num_data_in_smaller_leaf, any_bit_change_copy,
    /*level_smaller_num_data=*/nullptr,
    /*defer_subtract=*/nccl_communicator_ != nullptr);
  if (nccl_communicator_ != nullptr) {
    // ONE collective for the level instead of one per split
    NCCLReduceLevelHistograms(num_pairs);
    cuda_histogram_constructor_->SubtractHistogramsForLevel(
      cuda_hybrid_pair_descs_.RawDataReadOnly(), num_pairs, any_bit_change_copy);
    if (FalcataDebug().dump) {
      // larger-side diagnostics: after the deferred subtract, each pair's
      // larger hist must total (parent - smaller) with GLOBAL sums
      SynchronizeCUDADevice(__FILE__, __LINE__);
      std::vector<double> hist_host(static_cast<size_t>(num_total_bin_) * 2);
      for (int i = 0; i < num_pairs; ++i) {
        const CUDAHybridPairDescriptor& d = host_hybrid_pair_descs_[i];
        if (d.larger_leaf_index < 0) continue;
        CUDALeafSplitsStruct s{};
        CopyFromCUDADeviceToHost<CUDALeafSplitsStruct>(&s, d.larger_struct, 1, __FILE__, __LINE__);
        double hess_total = 0.0;
        if (s.hist_in_leaf != nullptr) {
          CopyFromCUDADeviceToHost<double>(hist_host.data(), reinterpret_cast<const double*>(s.hist_in_leaf),
                                           hist_host.size(), __FILE__, __LINE__);
          for (int b = 0; b < num_total_bin_; ++b) hess_total += hist_host[2 * b + 1];
        }
        Log::Warning("[nccl-lvl-L rank%d] pair%d Lleaf=%d desc_n=%d struct_n=%d struct_hess=%.2f hist_hess=%.2f valid=%d",
                     nccl_gpu_rank_, i, d.larger_leaf_index, d.num_data_in_larger_leaf,
                     s.num_data_in_leaf, s.sum_of_hessians, hess_total,
                     static_cast<int>(d.larger_valid));
      }
    }
  }
  cuda_best_split_finder_->WaitOnCatSlabConsumers(cat_consumed_default_event_, cat_consumed_tree_event_);
  cuda_best_split_finder_->FindBestSplitsForLevel(
    cuda_hybrid_pair_descs_.RawDataReadOnly(), num_pairs,
    config_->use_quantized_grad ? cuda_gradient_discretizer_->grad_scale_ptr() : nullptr,
    config_->use_quantized_grad ? cuda_gradient_discretizer_->hess_scale_ptr() : nullptr,
    /*gate_on_desc_counts=*/nccl_communicator_ != nullptr);
      if (FalcataDebug().dump) {
        SynchronizeCUDADevice(__FILE__, __LINE__);
        Log::Warning("[stage] find-sync clean");
      }
  global_timer.Stop("CUDASingleGPUTreeLearner::EnqueueLevelBestSplitSearch");
}

void CUDASingleGPUTreeLearner::EnqueueLevelBestSplitSearchSpeculative(const CUDATree* tree,
    const std::vector<HybridAppliedSplit>& applied) {
  const int num_pairs = static_cast<int>(applied.size());
  if (num_pairs <= 0) {
    return;
  }
  global_timer.Start("CUDASingleGPUTreeLearner::EnqueueLevelBestSplitSearch");
  host_hybrid_pair_descs_.resize(static_cast<size_t>(num_pairs));
  // the level's exact smaller-child sizes are not host-known yet (the apply's
  // readback is deferred); bound the construct grid by the parents' sizes
  // (smaller child <= floor(parent / 2)) -- the kernel restores the exact row
  // grouping from the device-side actual sizes written by the apply kernels
  data_size_t max_smaller_leaf_bound = 0;
  for (int i = 0; i < num_pairs; ++i) {
    const HybridAppliedSplit& a = applied[i];
    CUDAHybridPairDescriptor& desc = host_hybrid_pair_descs_[i];
    desc.smaller_struct = a.smaller_slot;
    desc.larger_struct = a.larger_slot;
    // informational only: the batched kernels read the leaf indices from the
    // structs (the smaller/larger designation is decided on-device)
    desc.smaller_leaf_index = a.left;
    desc.larger_leaf_index = a.right;
    // derived on-device from the child structs
    desc.num_data_in_smaller_leaf = 0;
    desc.num_data_in_larger_leaf = 0;
    desc.construct_valid = 1;
    // both children of a split are siblings at the same (host-known) depth; the
    // min_data/min_hessian gates are evaluated on-device
    const uint8_t below_max_depth = (config_->max_depth <= 0 ||
      GrowthLeafDepth(tree, a.left) < config_->max_depth) ? 1 : 0;
    desc.smaller_valid = below_max_depth;
    desc.larger_valid = below_max_depth;
    // single-sync flow is non-quantized only
    desc.parent_num_bits = 0;
    desc.smaller_num_bits = 0;
    desc.larger_num_bits = 0;
    // leaf_num_data_[a.left] still holds the PARENT's size (its readback-based
    // update is deferred to the next level's combined readback)
    const data_size_t smaller_child_bound = leaf_num_data_[a.left] / 2;
    if (smaller_child_bound > max_smaller_leaf_bound) {
      max_smaller_leaf_bound = smaller_child_bound;
    }
  }
  if (cuda_hybrid_pair_descs_.Size() < static_cast<size_t>(num_pairs)) {
    cuda_hybrid_pair_descs_.Resize(static_cast<size_t>(
      std::max(num_pairs, config_->num_leaves / 2 + 2)));
  }
  CopyFromHostToCUDADeviceAsync<CUDAHybridPairDescriptor>(
    cuda_hybrid_pair_descs_.RawData(), host_hybrid_pair_descs_.data(),
    static_cast<size_t>(num_pairs), cuda_histogram_constructor_->hist_stream(),
    __FILE__, __LINE__);
  // order the whole speculative search phase after this level's batched apply
  // kernels: they write the child structs (and the smaller-count array) the
  // construct/fix/subtract/find/sync kernels read. Everything enqueued before
  // the previous level's combined readback has already completed.
  CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(
    cuda_histogram_constructor_->hist_stream(),
    cuda_data_partition_->apply_done_event(), 0));
  cuda_histogram_constructor_->ConstructHistogramsForLevel(
    cuda_hybrid_pair_descs_.RawDataReadOnly(), num_pairs,
    max_smaller_leaf_bound, /*any_pair_needs_bit_change_copy=*/false,
    cuda_data_partition_->level_smaller_leaf_counts());
  cuda_best_split_finder_->WaitOnCatSlabConsumers(cat_consumed_default_event_, cat_consumed_tree_event_);
  cuda_best_split_finder_->FindBestSplitsForLevel(
    cuda_hybrid_pair_descs_.RawDataReadOnly(), num_pairs, nullptr, nullptr);
  global_timer.Stop("CUDASingleGPUTreeLearner::EnqueueLevelBestSplitSearch");
}

void CUDASingleGPUTreeLearner::CollectSplittableLeaves(const CUDATree* tree,
    std::vector<int>* splittable) {
  splittable->clear();
  for (int leaf = 0; leaf < tree->num_leaves(); ++leaf) {
    const CUDASplitInfo& info = host_leaf_best_splits_[leaf];
    if (info.is_valid) {
      // keep the host-side split arrays consistent with the device cache for
      // every candidate leaf: the leaf-wise tail may apply any of them later
      leaf_best_split_feature_[leaf] = info.inner_feature_index;
      leaf_best_split_threshold_[leaf] = info.threshold;
      leaf_best_split_default_left_[leaf] = static_cast<uint8_t>(info.default_left);
      splittable->push_back(leaf);
    }
  }
  const bool hybrid_debug = FalcataDebug().debug;
  if (hybrid_debug) {
    double max_gain = -1e300, min_gain = 1e300;
    for (const int leaf : *splittable) {
      max_gain = std::max(max_gain, host_leaf_best_splits_[leaf].gain);
      min_gain = std::min(min_gain, host_leaf_best_splits_[leaf].gain);
    }
    Log::Warning("[hybrid] leaves=%d splittable=%d gain=[%g, %g]",
                 tree->num_leaves(), static_cast<int>(splittable->size()), min_gain, max_gain);
  }
  if (FalcataDebug().dump) {
    for (int leaf = 0; leaf < tree->num_leaves(); ++leaf) {
      const CUDASplitInfo& info = host_leaf_best_splits_[leaf];
      Log::Warning("[leafcache rank%d] leaf=%d valid=%d gain=%.4f feat=%d thr=%u lc=%d rc=%d",
                   nccl_gpu_rank_, leaf, static_cast<int>(info.is_valid), info.gain,
                   info.inner_feature_index, info.threshold,
                   info.left_count, info.right_count);
    }
    // categorical candidate validation: every cached cat threshold must be a
    // REAL bin of its feature (inner bins live in [mfb_offset, num_bin));
    // dumps the first corruption the moment it enters the cache
    for (int leaf = 0; leaf < tree->num_leaves(); ++leaf) {
      const CUDASplitInfo& info = host_leaf_best_splits_[leaf];
      if (!info.is_valid || info.num_cat_threshold <= 0) continue;
      const int feat = info.inner_feature_index;
      if (feat < 0 || feat >= train_data_->num_features()) {
        Log::Warning("[catcheck] BAD-FEAT leaf=%d feat=%d nct=%d gain=%.4f",
                     leaf, feat, info.num_cat_threshold, info.gain);
        continue;
      }
      if (train_data_->FeatureBinMapper(feat)->bin_type() != BinType::CategoricalBin) {
        Log::Warning("[catcheck] NUMFEAT-WITH-CATCOUNT leaf=%d feat=%d nct=%d thr=%u gain=%.4f",
                     leaf, feat, info.num_cat_threshold, info.threshold, info.gain);
        continue;
      }
      const int nb = static_cast<int>(train_data_->FeatureBinMapper(feat)->num_bin());
      std::vector<uint32_t> bins;
      cuda_best_split_finder_->CopyLeafCatThresholdToHost(leaf, info.num_cat_threshold, &bins);
      for (int i = 0; i < info.num_cat_threshold; ++i) {
        if (static_cast<int>(bins[i]) >= nb) {
          Log::Warning("[catcheck] CORRUPT leaf=%d feat=%d nct=%d idx=%d bin=%u (num_bin=%d) gain=%.4f",
                       leaf, feat, info.num_cat_threshold, i, bins[i], nb, info.gain);
        }
      }
    }
  }
}

bool CUDASingleGPUTreeLearner::ArbitrateLevelBudget(const CUDATree* tree,
    std::vector<int>* splittable, bool* final_partial_level) {
  // batch the level whenever it fits in the leaf budget. If the finished tree
  // never exhausts num_leaves the result is exactly the leaf-wise tree (split
  // decisions are node-local); only when the budget binds does the final level
  // defer to the leaf-wise tail, which selects among the cached candidates in
  // exact best-gain order.
  *final_partial_level = false;
  bool all_children_depth_capped = config_->max_depth > 0 && !splittable->empty();
  if (all_children_depth_capped) {
    for (const int leaf : *splittable) {
      if (tree->leaf_depth(leaf) + 1 < config_->max_depth) {
        all_children_depth_capped = false;
        break;
      }
    }
  }
  if (tree->num_leaves() + static_cast<int>(splittable->size()) <= config_->num_leaves) {
    // Budget does not bind. When every child this level creates sits at
    // max_depth the level is still FINAL: no child can become a candidate, so
    // flagging it skips a whole wasted next-level search AND lets the apply
    // write the leaf map inline (see ApplyLevelBatched).
    *final_partial_level = all_children_depth_capped;
    if (FalcataDebug().maxsplits >= 0) {
      const size_t cap = static_cast<size_t>(FalcataDebug().maxsplits);
      if (splittable->size() > cap) splittable->resize(cap);
    }
    return true;
  }
  {
    // The budget binds. In general the leaf-wise tail must arbitrate between
    // this level's candidates and their future children, so defer to it. But
    // when every child this level could create would sit at max_depth, no
    // child can ever become a candidate itself (FindBestSplitsForLeaf marks
    // leaves at max_depth invalid) and splitting one frontier leaf leaves the
    // cached best splits of the other candidates untouched: the tail would
    // simply apply the top-(remaining budget) candidates in descending gain
    // order, one full search-partition-sync round trip per split. Reproduce
    // exactly that as one final batched partial level.
    const int remaining_budget = config_->num_leaves - tree->num_leaves();
    if (!all_children_depth_capped || remaining_budget <= 0) {
      return false;
    }
    // descending gain; ties go to the lower leaf index (splittable is leaf-
    // ascending and the sort is stable), matching FindBestFromAllSplitsKernel's
    // strict-greater selection. Applying in this order reassigns right-child
    // leaf indices exactly as the tail's sequential splits would.
    std::stable_sort(splittable->begin(), splittable->end(),
      [this](const int a, const int b) {
        return host_leaf_best_splits_[a].gain > host_leaf_best_splits_[b].gain;
      });
    splittable->resize(static_cast<size_t>(remaining_budget));
    *final_partial_level = true;
  }
  // debug (FALCATA_DEBUG=maxsplits=N): cap splits per level to isolate
  // multi-pair interactions
  if (FalcataDebug().maxsplits >= 0) {
    const size_t cap = static_cast<size_t>(FalcataDebug().maxsplits);
    if (splittable->size() > cap) splittable->resize(cap);
  }
  return true;
}

void CUDASingleGPUTreeLearner::ApplyLevelBatched(CUDATree* tree,
    const std::vector<int>& splittable, std::vector<HybridAppliedSplit>* applied,
    const bool final_level) {
  // one launch per kernel family for the whole level: batched tree-structure
  // update (CUDATree::SplitBatch) + batched partition (SplitLevelBatched);
  // right children take consecutive leaf indices exactly as the per-split
  // loop would assign them
  applied->clear();
  applied->reserve(splittable.size());
  const int base_num_leaves = tree->num_leaves();
  host_tree_batch_splits_.clear();
  host_apply_split_inputs_.clear();
  size_t slot_id = 0;
  for (const int leaf : splittable) {
    CUDALeafSplitsStruct* smaller_slot = hybrid_pair_slots_.RawData() + slot_id;
    CUDALeafSplitsStruct* larger_slot = hybrid_pair_slots_.RawData() + slot_id + 1;
    const int right_leaf_index = base_num_leaves + static_cast<int>(applied->size());
    const CUDASplitInfo* best_split_info = cuda_best_split_finder_->leaf_best_split_info_ptr(leaf);
    const int inner_feature_index = leaf_best_split_feature_[leaf];
    // The batched apply path does not go through ApplySplit(), so it marks the
    // feature itself -- otherwise CEGB would charge cegb_penalty_feature_coupled
    // again for a feature this tree already used. This is currently DEAD CODE:
    // CEGB forces use_hybrid_growth_ = false in Init(), so ApplyLevelBatched never
    // runs while CEGB is active. Kept as a defensive guard in case that gating is
    // ever relaxed. No-op unless CEGB is configured.
    cuda_best_split_finder_->MarkFeatureUsedInSplit(inner_feature_index);
    // Categorical splits: the entry is never consumed (the interleave below
    // routes them through SplitCategorical), and RealThreshold() would
    // OOB-index the bin-upper-bound array with the categorical threshold
    // payload -- do not call it.
    const bool is_cat_split = host_leaf_best_splits_[leaf].num_cat_threshold > 0;
    host_tree_batch_splits_.push_back({
      leaf,
      right_leaf_index,  // == tree num_leaves at the time of this split
      train_data_->RealFeatureIndex(inner_feature_index),
      is_cat_split ? 0.0 :
        train_data_->RealThreshold(inner_feature_index, leaf_best_split_threshold_[leaf]),
      static_cast<int>(train_data_->FeatureBinMapper(inner_feature_index)->missing_type()),
      best_split_info});
    host_apply_split_inputs_.push_back({
      best_split_info,
      leaf,
      right_leaf_index,
      inner_feature_index,
      leaf_best_split_threshold_[leaf],
      leaf_best_split_default_left_[leaf],
      host_leaf_best_splits_[leaf].num_cat_threshold,
      leaf_num_data_[leaf],
      leaf_data_start_[leaf],
      smaller_slot,
      larger_slot});
    applied->push_back({leaf, right_leaf_index, smaller_slot, larger_slot});
    slot_id += 2;
  }
  // ZeroHistForLeaf is a no-op (BeforeTrain zeroes the full histogram
  // buffer), so the batched path skips the per-split calls entirely.
  //
  // Tree recording: numerical runs go through SplitBatch chunks; the level's
  // categorical splits (rare) interleave through the per-split categorical
  // path IN LEVEL ORDER, preserving the consecutive right-child numbering.
  // The data partition still applies EVERYTHING through the batched arena
  // path below.
  {
    // Level-batched categorical bitset construction: build every categorical
    // split's bitsets + lengths with ONE kernel and ONE readback, then record
    // splits in level order (the per-split construction did ~6 launches and 2
    // blocking readbacks per categorical split -- hundreds per deep tree).
    host_batch_cat_infos_.clear();
    host_batch_cat_feats_.clear();
    for (size_t k = 0; k < host_apply_split_inputs_.size(); ++k) {
      const CUDAHybridApplySplitInput& in = host_apply_split_inputs_[k];
      if (in.num_cat_threshold > 0) {
        host_batch_cat_infos_.push_back(in.best_split_info);
        host_batch_cat_feats_.push_back(in.split_feature);
      }
    }
    const int num_cat_splits = static_cast<int>(host_batch_cat_infos_.size());
    if (num_cat_splits > 0) {
      const size_t needed_real = static_cast<size_t>(num_cat_splits) * batch_cat_bitset_stride_;
      const size_t needed_inner = static_cast<size_t>(num_cat_splits) * batch_cat_bitset_inner_stride_;
      if (cuda_batch_cat_bitset_.Size() < needed_real) {
        cuda_batch_cat_bitset_.Resize(needed_real);
        cuda_batch_cat_bitset_inner_.Resize(needed_inner);
      }
      if (cuda_batch_cat_infos_.Size() < static_cast<size_t>(num_cat_splits)) {
        cuda_batch_cat_infos_.Resize(num_cat_splits);
        cuda_batch_cat_feats_.Resize(num_cat_splits);
        cuda_batch_cat_lens_.Resize(2 * static_cast<size_t>(num_cat_splits));
      }
      CopyFromHostToCUDADevice<const CUDASplitInfo*>(cuda_batch_cat_infos_.RawData(),
        host_batch_cat_infos_.data(), num_cat_splits, __FILE__, __LINE__);
      CopyFromHostToCUDADevice<int>(cuda_batch_cat_feats_.RawData(),
        host_batch_cat_feats_.data(), num_cat_splits, __FILE__, __LINE__);
      if (FalcataDebug().dump) {
        // second checkpoint: validate the exact entries the bitset kernel is
        // about to consume (the sync-time check passed; a diff here pins the
        // corruption to the window between sync readback and apply)
        for (int c = 0; c < num_cat_splits; ++c) {
          // host_batch_cat_infos_ holds leaf_best_split_info_ptr(leaf); find leaf
          for (size_t k2 = 0; k2 < host_apply_split_inputs_.size(); ++k2) {
            const CUDAHybridApplySplitInput& in2 = host_apply_split_inputs_[k2];
            if (in2.best_split_info != host_batch_cat_infos_[c]) continue;
            const int leaf2 = in2.left_leaf_index;
            const int feat2 = in2.split_feature;
            const int nct2 = in2.num_cat_threshold;
            const bool is_cat2 = train_data_->FeatureBinMapper(feat2)->bin_type() == BinType::CategoricalBin;
            const int nb2 = static_cast<int>(train_data_->FeatureBinMapper(feat2)->num_bin());
            std::vector<uint32_t> bins2;
            if (nct2 > 0) {
              cuda_best_split_finder_->CopyLeafCatThresholdToHost(leaf2, nct2, &bins2);
            }
            uint32_t mx = 0;
            for (int i2 = 0; i2 < nct2; ++i2) mx = std::max(mx, bins2[i2]);
            // pointer-integrity probe: the cache entry's device cat_threshold
            // must aim at ITS per-leaf slab; anything else means a raw struct
            // copy clobbered it and the bitset kernel reads foreign memory
            // raw byte read: a real host CUDASplitInfo would run its dtor on
            // the DEVICE pointers it copies
            alignas(alignof(CUDASplitInfo)) unsigned char probe_bytes[sizeof(CUDASplitInfo)];
            CopyFromCUDADeviceToHost<unsigned char>(probe_bytes,
                reinterpret_cast<const unsigned char*>(in2.best_split_info),
                sizeof(CUDASplitInfo), __FILE__, __LINE__);
            const CUDASplitInfo& probe = *reinterpret_cast<const CUDASplitInfo*>(probe_bytes);
            const uint32_t* expect = cuda_best_split_finder_->ExpectedLeafCatThresholdPtr(leaf2);
            const int* expect_real = cuda_best_split_finder_->ExpectedLeafCatThresholdRealPtr(leaf2);
            if (probe.cat_threshold_real != expect_real) {
              Log::Warning("[applycheck] c=%d leaf=%d REAL-PTR-CLOBBERED real=%p expect=%p",
                           c, leaf2, static_cast<const void*>(probe.cat_threshold_real),
                           static_cast<const void*>(expect_real));
            }
            Log::Warning("[applycheck] c=%d leaf=%d feat=%d cat=%d nct=%d dev_nct=%d dev_feat=%d maxbin=%u num_bin=%d ptr=%p expect=%p%s%s%s",
                         c, leaf2, feat2, static_cast<int>(is_cat2), nct2,
                         probe.num_cat_threshold, probe.inner_feature_index, mx, nb2,
                         static_cast<const void*>(probe.cat_threshold),
                         static_cast<const void*>(expect),
                         probe.cat_threshold != expect ? " <-- PTR-CLOBBERED" : "",
                         probe.num_cat_threshold != nct2 ? " <-- NCT-MISMATCH" : "",
                         (!is_cat2 || static_cast<int>(mx) >= nb2) ? " <-- CORRUPT" : "");
          }
        }
      }
      LaunchBatchConstructCatBitsetsKernel(num_cat_splits);
      if (FalcataDebug().dump) {
        // surface the bitset kernel's own fault AT ITS LEVEL instead of one
        // level later through the sticky-error fog
        SynchronizeCUDADevice(__FILE__, __LINE__);
        Log::Warning("[applycheck] bitset kernel of this level completed clean");
      }
      host_batch_cat_lens_.resize(2 * static_cast<size_t>(num_cat_splits));
      CopyFromCUDADeviceToHost<uint64_t>(host_batch_cat_lens_.data(),
        cuda_batch_cat_lens_.RawData(), 2 * static_cast<size_t>(num_cat_splits),
        __FILE__, __LINE__);
      if (cat_consumed_default_event_ == nullptr) {
        CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&cat_consumed_default_event_, cudaEventDisableTiming));
        CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&cat_consumed_tree_event_, cudaEventDisableTiming));
      }
      // the bitset kernel (legacy stream) has consumed every candidate's
      // per-leaf thresholds once this event completes
      CUDASUCCESS_OR_FATAL(cudaEventRecord(cat_consumed_default_event_, nullptr));
    }
    std::vector<CUDATreeBatchSplit> chunk;
    chunk.reserve(host_tree_batch_splits_.size());
    int cat_index = 0;
    for (size_t k = 0; k < host_tree_batch_splits_.size(); ++k) {
      const CUDAHybridApplySplitInput& in = host_apply_split_inputs_[k];
      if (in.num_cat_threshold > 0) {
        if (!chunk.empty()) {
          tree->SplitBatch(chunk);
          if (vec_num_targets_ > 1) {
            tree->SetVectorLeafValuesFromSplitBatch(static_cast<int>(chunk.size()));
          }
          chunk.clear();
        }
        const int inner_feature = in.split_feature;
        CHECK_GE(inner_feature, 0);
        CHECK_LT(inner_feature, train_data_->num_features());
        const int right = tree->SplitCategorical(
            in.left_leaf_index,
            train_data_->RealFeatureIndex(inner_feature),
            train_data_->FeatureBinMapper(inner_feature)->missing_type(),
            in.best_split_info,
            cuda_batch_cat_bitset_.RawData() +
              static_cast<size_t>(cat_index) * batch_cat_bitset_stride_,
            static_cast<size_t>(host_batch_cat_lens_[2 * cat_index]),
            cuda_batch_cat_bitset_inner_.RawData() +
              static_cast<size_t>(cat_index) * batch_cat_bitset_inner_stride_,
            static_cast<size_t>(host_batch_cat_lens_[2 * cat_index + 1]));
        CHECK_EQ(right, in.right_leaf_index);
        if (vec_num_targets_ > 1) {
          tree->SetVectorLeafValuesFromSplit(in.left_leaf_index, right, in.best_split_info);
        }
        ++cat_index;
      } else {
        chunk.push_back(host_tree_batch_splits_[k]);
      }
    }
    if (!chunk.empty()) {
      tree->SplitBatch(chunk);
      if (vec_num_targets_ > 1) {
        // the split kernel wrote the target-0 (mirror) outputs; record every
        // target's child outputs from the winning splits' vector payloads
        tree->SetVectorLeafValuesFromSplitBatch(static_cast<int>(chunk.size()));
      }
    }
    if (cat_index > 0 && cat_consumed_tree_event_ != nullptr) {
      // SplitCategorical kernels (tree stream) also read the per-leaf
      // threshold slabs (cat_threshold_real for the tree's real-value bitset)
      CUDASUCCESS_OR_FATAL(cudaEventRecord(cat_consumed_tree_event_, tree->tree_stream()));
    }
      if (FalcataDebug().dump) {
        SynchronizeCUDADevice(__FILE__, __LINE__);
        Log::Warning("[stage] tree-record clean");
      }
  }
  cuda_data_partition_->SplitLevelBatched(host_apply_split_inputs_, final_level);
      if (FalcataDebug().dump) {
        SynchronizeCUDADevice(__FILE__, __LINE__);
        Log::Warning("[stage] partition-apply clean");
      }

  if (final_level) {
    // this level's children are never searched (the level is known final), so
    // their device leaf-cache slots still hold stale entries from earlier
    // occupants of the same indices; invalidate them so the leaf-wise tail's
    // FindBestFromAllSplits cannot resurrect a stale candidate
    std::vector<int> children;
    children.reserve(2 * host_apply_split_inputs_.size());
    for (const CUDAHybridApplySplitInput& in : host_apply_split_inputs_) {
      children.push_back(in.left_leaf_index);
      children.push_back(in.right_leaf_index);
    }
    cuda_best_split_finder_->InvalidateLeafCandidates(children);
    // the final level's split-inner wrote the map for every row it moved;
    // the tree-end materialize only needs the leaves NOT split at this level
    hybrid_map_final_written_ = true;
    hybrid_map_residual_leaves_.clear();
    std::vector<bool> covered(tree->num_leaves(), false);
    for (const CUDAHybridApplySplitInput& in : host_apply_split_inputs_) {
      covered[in.left_leaf_index] = true;
      covered[in.right_leaf_index] = true;
    }
    for (int leaf = 0; leaf < tree->num_leaves(); ++leaf) {
      if (!covered[leaf]) {
        hybrid_map_residual_leaves_.push_back(leaf);
      }
    }
  }
}

void CUDASingleGPUTreeLearner::FinishLevelBookkeeping(
    const std::vector<HybridAppliedSplit>& applied,
    const std::vector<int>& batch_info,
    std::vector<HybridPendingPair>* next_pairs,
    int* num_splits) {
  for (size_t k = 0; k < applied.size(); ++k) {
    const HybridAppliedSplit& a = applied[k];
    const int* info = batch_info.data() + 18 * k;
    const double* dinfo = reinterpret_cast<const double*>(info + 8);
    leaf_num_data_[a.left] = info[1];
    leaf_data_start_[a.left] = info[2];
    leaf_num_data_[a.right] = info[4];
    leaf_data_start_[a.right] = info[5];
    leaf_sum_hessians_[a.left] = dinfo[0];
    leaf_sum_hessians_[a.right] = dinfo[1];
    leaf_sum_gradients_[a.left] = dinfo[2];
    leaf_sum_gradients_[a.right] = dinfo[3];
    // Under NCCL the smaller/larger roles must match the device kernel's
    // choice AND be identical on every rank, so both read the GLOBAL counts
    // (slots 16/17, from the all-reduced histograms); the local partition
    // counts can order differently per rank on skewed leaves. These are also
    // the gates' counts: EnqueueLevelBestSplitSearch reads
    // global_num_data_in_leaf_ for the next level's validity decisions, and
    // without this update every child read stale garbage from earlier trees.
    int smaller, larger;
    if (nccl_communicator_ != nullptr) {
      global_num_data_in_leaf_[a.left] = info[16];
      global_num_data_in_leaf_[a.right] = info[17];
      smaller = info[16] < info[17] ? a.left : a.right;
      larger = smaller == a.left ? a.right : a.left;
      if (FalcataDebug().dump) {
        Log::Warning("[bookkeep rank%d] L=%d R=%d local(%d,%d) global(%d,%d)",
                     nccl_gpu_rank_, a.left, a.right,
                     leaf_num_data_[a.left], leaf_num_data_[a.right], info[16], info[17]);
      }
    } else {
      smaller = leaf_num_data_[a.left] < leaf_num_data_[a.right] ? a.left : a.right;
      larger = smaller == a.left ? a.right : a.left;
    }
    if (config_->use_quantized_grad) {
      cuda_gradient_discretizer_->SetNumBitsInHistogramBin<false>(
        a.left, a.right, leaf_num_data_[a.left], leaf_num_data_[a.right]);
      if (nccl_communicator_ != nullptr) {
        // global table: identical on every rank (slots 16/17 are the
        // all-reduce-derived counts) and the source of both the collective
        // element type and the construct's packing width under NCCL
        cuda_gradient_discretizer_->SetNumBitsInHistogramBin<true>(
          a.left, a.right, info[16], info[17]);
      }
    }
    if (next_pairs != nullptr) {
      next_pairs->push_back({smaller, larger, a.left, a.smaller_slot, a.larger_slot});
    }
    smaller_leaf_index_ = smaller;
    larger_leaf_index_ = larger;
    ++(*num_splits);
  }
}

void CUDASingleGPUTreeLearner::FeatureParallelMergeLevel(const int num_leaves) {
  global_timer.Start("CUDASingleGPUTreeLearner::FeatureParallelMergeLevel");
  FeatureParallelMergeState* st = fp_merge_state_;
  st->Publish(fp_rank_, &host_leaf_best_splits_);
  st->Arrive();  // every rank's host buffer is published and final
  // deterministic winner per leaf, computed identically on every rank
  fp_merge_tmp_.resize(static_cast<size_t>(num_leaves));
  for (int leaf = 0; leaf < num_leaves; ++leaf) {
    const CUDASplitInfo* best = &(*st->rank_buf(0))[leaf];
    for (int r = 1; r < st->num_ranks(); ++r) {
      const CUDASplitInfo* cand = &(*st->rank_buf(r))[leaf];
      if (FeatureParallelMergeState::FirstBeatsSecond(*cand, *best)) {
        best = cand;
      }
    }
    fp_merge_tmp_[leaf] = *best;
  }
  st->Arrive();  // all ranks done READING each other's buffers
  std::copy(fp_merge_tmp_.begin(), fp_merge_tmp_.end(), host_leaf_best_splits_.begin());
  // the batched apply reads split infos from the DEVICE cache: overwrite its
  // first num_leaves entries with the winners (numerical splits only; the
  // categorical side-band pointers in these entries are unused -- FP mode
  // fences categorical features)
  cuda_best_split_finder_->UploadLeafBestSplits(fp_merge_tmp_.data(), num_leaves);
  global_timer.Stop("CUDASingleGPUTreeLearner::FeatureParallelMergeLevel");
}

int CUDASingleGPUTreeLearner::TrainLevelWisePrefix(CUDATree* tree) {
  // two struct slots per split of a level, reused across levels; one device
  // slab, sized once (the kernels initialize every slot they use before any
  // read, exactly as with the previous individually-allocated structs)
  const size_t max_slots = static_cast<size_t>(config_->num_leaves) + 2;
  if (hybrid_pair_slots_.Size() < max_slots) {
    hybrid_pair_slots_.Resize(max_slots);
  }
  // vector mode: T plane structs per slot, in the slot order the fan-out and
  // the batched vector find share
  if (vec_num_targets_ > 1 &&
      vec_level_plane_slots_.Size() < max_slots * static_cast<size_t>(vec_num_targets_)) {
    vec_level_plane_slots_.Resize(max_slots * static_cast<size_t>(vec_num_targets_));
  }
  // batched per-level kernels: one construct/fix/subtract/find/sync launch covers
  // all pairs of a level. Falls back to the per-pair loop when the data layout or
  // config is outside the batched kernels' support, or when disabled via env.
  const bool use_batched_level_kernels = use_hybrid_batch_kernels_ &&
    cuda_histogram_constructor_->SupportsBatchedLevel() &&
    cuda_best_split_finder_->SupportsBatchedLevel();
  const bool hybrid_diag = FalcataDebug().diag;
  if (hybrid_diag) {
    static bool diag_logged = false;
    if (!diag_logged) {
      diag_logged = true;
      fprintf(stderr, "[hybrid-diag] prefix: batched_level_kernels=%d batch_kernels_env=%d one_sync=%d\n",
                   static_cast<int>(use_batched_level_kernels),
                   static_cast<int>(use_hybrid_batch_kernels_),
                   static_cast<int>(UseOneSyncPrefix()));
      fprintf(stderr, "[hybrid-diag] %s\n", cuda_histogram_constructor_->BatchedLevelGateDiag().c_str());
      fprintf(stderr, "[hybrid-diag] %s\n", cuda_best_split_finder_->BatchedLevelGateDiag().c_str());
    }
  }
  // batched apply: all tree-structure + partition updates of a level in one
  // launch per kernel family with zero per-split host syncs. The hybrid prefix
  // guarantees the preconditions (single GPU, no categorical splits; bagging
  // shares the same index array), so this only falls back when disabled by env.
  const bool use_batched_level_apply = use_hybrid_batch_apply_;
  // vector mode reaches the prefix only through the batched phases
  // (HybridGrowthUsable): the per-pair search fallback and the per-split apply
  // fallback below are scalar-only
  CHECK(vec_num_targets_ <= 1 || (use_batched_level_kernels && use_batched_level_apply));
  // single-sync (speculative) level pipeline: requires both batched phases and
  // non-quantized training (the quantized HOST path selects histogram kernels
  // host-side from per-leaf bit widths, which needs the classic readback
  // ordering).
  if (UseOneSyncPrefix()) {
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
    // graphs L1: run the whole depth-limited prefix as one device-driven
    // graph launch where supported (cuda_plan=auto,graph_loop:off disables)
    if (HybridGraphPrefixUsable()) {
      const int graph_splits = TrainLevelWisePrefixGraph(tree);
      if (graph_splits >= 0) {
        return graph_splits;
      }
    }
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
    return TrainLevelWisePrefixOneSync(tree);
  }
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  // quantized graph prefix: the graph body derives the per-leaf histogram bit
  // widths on-device from the exact leaf counts (bit-identical to the host's
  // SetNumBitsInHistogramBin thresholds), so the classic readback ordering is
  // not needed inside the graph. Any fallback (env off, unsupported driver,
  // capture failure, no usable feature) reverts to the classic (two-sync)
  // host loop below, bit-for-bit the previous behavior.
  if (config_->use_quantized_grad && use_hybrid_one_sync_ &&
      use_batched_level_kernels && use_batched_level_apply &&
      HybridGraphPrefixUsable()) {
    const int graph_splits = TrainLevelWisePrefixGraph(tree);
    if (graph_splits >= 0) {
      return graph_splits;
    }
  }
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  // the classic flow needs the host root sums up front (root descriptor
  // validity); a no-op unless BeforeTrain optimistically deferred the readback
  EnsureRootSumsReadBack(tree);
  // the root "pair" lives in the fixed pair structs, initialized by BeforeTrain
  std::vector<HybridPendingPair> pairs;
  pairs.push_back({smaller_leaf_index_, larger_leaf_index_, /*parent=*/-1,
                   cuda_smaller_leaf_splits_->GetCUDAStruct(),
                   cuda_larger_leaf_splits_->GetCUDAStruct()});
  // vector mode: the ROOT level's planes are the fixed slab BeforeTrain filled
  // (pair 0's smaller role at [0, T), larger at [T, 2T)); every later level
  // reads the per-slot slab the plane fan-out writes
  CUDALeafSplitsStruct* vec_plane_slab =
    vec_num_targets_ > 1 ? vec_plane_structs_.RawData() : nullptr;
  int num_splits = 0;
  while (true) {
    // enqueue histogram + best-split search for every pair of this level; device
    // work only, ordered per pair by the histogram-completion events
    if (use_batched_level_kernels) {
      EnqueueLevelBestSplitSearch(tree, pairs, vec_plane_slab);
    } else {
      int pair_counter = 0;
      for (const HybridPendingPair& pair : pairs) {
        const bool sync_pairs = FalcataDebug().syncpairs;
        EnqueuePairBestSplitSearch(tree, pair.smaller_struct, pair.larger_struct,
                                   pair.smaller, pair.larger, /*synchronize=*/sync_pairs,
                                   pair_counter % CUDAHistogramConstructor::kNumHistPipelines);
        ++pair_counter;
      }
    }
    // one device synchronization + one transfer for the whole level
    cuda_best_split_finder_->SyncAllLeafBestSplitsToHost(tree->num_leaves(), &host_leaf_best_splits_);
      if (fp_merge_state_ != nullptr) {
      FeatureParallelMergeLevel(tree->num_leaves());
    }
    std::vector<int> splittable;
    CollectSplittableLeaves(tree, &splittable);
    if (splittable.empty()) {
      break;
    }
    bool final_partial_level = false;
    if (!ArbitrateLevelBudget(tree, &splittable, &final_partial_level)) {
      break;
    }
    pairs.clear();
    // apply the whole level without any device synchronization: every launch
    // parameter is host-known from the previous level, and split info is read
    // back once for the whole level below (FinishSplitBatch synchronizes)
    std::vector<HybridAppliedSplit> applied;
    if (use_batched_level_apply) {
      ApplyLevelBatched(tree, splittable, &applied, final_partial_level);
    } else {
      applied.reserve(splittable.size());
      size_t slot_id = 0;
      for (const int leaf : splittable) {
        CUDALeafSplitsStruct* smaller_slot = hybrid_pair_slots_.RawData() + slot_id;
        CUDALeafSplitsStruct* larger_slot = hybrid_pair_slots_.RawData() + slot_id + 1;
        const int right_leaf_index = ApplySplit(
          tree, cuda_best_split_finder_->leaf_best_split_info_ptr(leaf), leaf,
          smaller_slot, larger_slot, static_cast<int>(applied.size()));
        applied.push_back({leaf, right_leaf_index, smaller_slot, larger_slot});
        slot_id += 2;
      }
    }
      std::vector<int> batch_info;
    cuda_data_partition_->FinishSplitBatch(static_cast<int>(applied.size()), &batch_info);
    FinishLevelBookkeeping(applied, batch_info, &pairs, &num_splits);
      if (vec_num_targets_ > 1) {
      // the partition seeded the primary child structs (target-0 sums, plane-0
      // histogram pointers); fan them out into this level's plane structs with
      // per-target sums/values from each parent split's vector payload
      VectorLevelPlaneFanOut(applied);
      vec_plane_slab = vec_level_plane_slots_.RawData();
        }
    if (final_partial_level) {
      // the tree is full and every child sits at max_depth: nothing is left
      // for the leaf-wise tail to search or split
      break;
    }
  }
  return num_splits;
}

void CUDASingleGPUTreeLearner::EnqueueRootLevelSearchOneSync() {
  // Root level of the single-sync flow: device-derived gating like every later
  // level, so the root-sum readback of InitValues stays deferred and the
  // level-0 search enqueue overlaps the per-tree histogram-buffer zeroing. The
  // construct grid is exact (the root size is host-known), matching the
  // classic sizing bit-for-bit.
  host_hybrid_pair_descs_.resize(1);
  CUDAHybridPairDescriptor& desc = host_hybrid_pair_descs_[0];
  desc.smaller_struct = cuda_smaller_leaf_splits_->GetCUDAStruct();
  desc.larger_struct = cuda_larger_leaf_splits_->GetCUDAStruct();
  desc.smaller_leaf_index = 0;
  desc.larger_leaf_index = -1;
  desc.num_data_in_smaller_leaf = leaf_num_data_[0];
  desc.num_data_in_larger_leaf = 0;
  desc.construct_valid = 1;  // min_data/min_hessian evaluated on-device
  // the root sits at depth 0
  desc.smaller_valid = (config_->max_depth <= 0 || 0 < config_->max_depth) ? 1 : 0;
  desc.larger_valid = 0;
  if (config_->use_quantized_grad) {
    // mirror of EnqueueLevelBestSplitSearch's root (larger < 0) branch: the
    // root's bit width is host-known (BeforeTrain's SetNumBitsInHistogramBin)
    const uint8_t root_num_bits = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(0);
    desc.parent_num_bits = root_num_bits;
    desc.smaller_num_bits = root_num_bits;
    desc.larger_num_bits = root_num_bits;
  } else {
    desc.parent_num_bits = 0;
    desc.smaller_num_bits = 0;
    desc.larger_num_bits = 0;
  }
  if (cuda_hybrid_pair_descs_.Size() < 1) {
    cuda_hybrid_pair_descs_.Resize(static_cast<size_t>(config_->num_leaves / 2 + 2));
  }
  CopyFromHostToCUDADeviceAsync<CUDAHybridPairDescriptor>(
    cuda_hybrid_pair_descs_.RawData(), host_hybrid_pair_descs_.data(), 1,
    cuda_histogram_constructor_->hist_stream(), __FILE__, __LINE__);
  cuda_histogram_constructor_->ConstructHistogramsForLevel(
    cuda_hybrid_pair_descs_.RawDataReadOnly(), 1, leaf_num_data_[0],
    /*any_pair_needs_bit_change_copy=*/false);
  cuda_best_split_finder_->WaitOnCatSlabConsumers(cat_consumed_default_event_, cat_consumed_tree_event_);
  cuda_best_split_finder_->FindBestSplitsForLevel(
    cuda_hybrid_pair_descs_.RawDataReadOnly(), 1,
    config_->use_quantized_grad ? cuda_gradient_discretizer_->grad_scale_ptr() : nullptr,
    config_->use_quantized_grad ? cuda_gradient_discretizer_->hess_scale_ptr() : nullptr);
}

int CUDASingleGPUTreeLearner::TrainLevelWisePrefixOneSync(CUDATree* tree) {
  // Single-sync level pipeline: the level's APPLY and the children's SEARCH are
  // enqueued back to back; the search kernels derive per-child validity, sizes
  // and hessian sums from the child structs the apply kernels write, so no host
  // value is needed in between. Each iteration then reads back the level's
  // deferred split info AND the children's best splits with ONE device
  // synchronization (the second, deferred-info copy finds the device idle).
  EnqueueRootLevelSearchOneSync();
  int num_splits = 0;
  std::vector<HybridAppliedSplit> pending;  // applied splits awaiting readback
  std::vector<HybridAppliedSplit> applied;
  std::vector<int> batch_info;
  std::vector<int> splittable;
  while (true) {
    // the ONE device synchronization of the level: the synchronous best-split
    // readback drains the previous apply and its children's search
    cuda_best_split_finder_->SyncAllLeafBestSplitsToHost(tree->num_leaves(), &host_leaf_best_splits_);
    if (!pending.empty()) {
      cuda_data_partition_->FinishSplitBatch(static_cast<int>(pending.size()), &batch_info);
      FinishLevelBookkeeping(pending, batch_info, nullptr, &num_splits);
      pending.clear();
    }
    CollectSplittableLeaves(tree, &splittable);
    if (splittable.empty()) {
      break;
    }
    bool final_partial_level = false;
    if (!ArbitrateLevelBudget(tree, &splittable, &final_partial_level)) {
      // leave the budget-bound level to the leaf-wise tail; every current leaf
      // already has a valid cached best-split candidate
      break;
    }
    ApplyLevelBatched(tree, splittable, &applied, final_partial_level);
    if (final_partial_level) {
      // the tree is full and every child sits at max_depth: nothing is left to
      // search, so read the apply info back immediately and stop
      cuda_data_partition_->FinishSplitBatch(static_cast<int>(applied.size()), &batch_info);
      FinishLevelBookkeeping(applied, batch_info, nullptr, &num_splits);
      break;
    }
    // speculative: the children's search goes out before their statistics are
    // host-known; the next iteration's readback collects both
    EnqueueLevelBestSplitSearchSpeculative(tree, applied);
    pending.swap(applied);
  }
  if (num_splits == 0) {
    // the root has no valid split: the leaf-wise tail (and a possible stump)
    // needs the host root sums after all
    EnsureRootSumsReadBack(tree);
  }
  return num_splits;
}

// ---- graphs L1: device-driven level loop (see cuda_hybrid_graph.hpp) ----
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED

namespace {

bool HybridGraphEnvEnabled() {
  return FalcataPlan::Get().graph_loop;
}

bool HybridGraphDriverSupported() {
  static const bool supported = []() {
    int driver_version = 0;
    return cudaDriverGetVersion(&driver_version) == cudaSuccess && driver_version >= 12040;
  }();
  return supported;
}

}  // anonymous namespace

CUDASingleGPUTreeLearner::HybridGraphInstance::~HybridGraphInstance() {
  if (exec != nullptr) {
    cudaGraphExecDestroy(exec);
  }
  if (graph != nullptr) {
    cudaGraphDestroy(graph);
  }
  if (state_dev != nullptr) {
    cudaFree(state_dev);
  }
}

bool CUDASingleGPUTreeLearner::HybridGraphPrefixUsable() const {
  // multi-GPU: the graph flow builds descriptors on-device; its rank-visible decisions are not global
  if (nccl_communicator_ != nullptr) {
    return false;
  }
  // feature-parallel: the device controller cannot host-merge winners per level
  if (fp_merge_state_ != nullptr) {
    return false;
  }
  // the per-level debug envs steer the host loop's decisions in ways the
  // device controller does not replicate
  const bool debug_envs = FalcataDebug().any_growth_hook();
  if (!HybridGraphEnvEnabled() || debug_envs || !HybridGraphDriverSupported() ||
      hybrid_graph_disabled_) {
    return false;
  }
  // categorical splits interleave a per-split host path (bitset construct +
  // SplitCategorical) through the level apply -- impossible inside a captured
  // graph body; categorical datasets keep the host-driven prefix loop
  if (has_categorical_feature_) {
    return false;
  }
  // vector-leaf training: the captured level body has one construct/find node
  // per level, while the vector flow issues one construct per gradient PLANE
  // and a plane fan-out per level, none of which the device controller models
  if (vec_num_targets_ > 1) {
    return false;
  }
  // quant graph support is bit-exact but a net loss on large/cheap-level
  // shapes (covtype 1023/10 -10.8%, year 63/6 -7%: quant levels are cheap, so the
  // controller's fixed serial latency dominates) -- opt-in until that frontier
  // shrinks. cuda_plan=auto,graph_quant:on enables.
  if (config_->use_quantized_grad && !FalcataPlan::Get().graph_quant) {
    return false;
  }
  // depth-limited exact regime only (mirrors HybridGrowthUsable): the
  // controller replays ArbitrateLevelBudget under the level == depth
  // invariant that only holds there
  const bool depth_limited = config_->max_depth > 0 && config_->max_depth < 31 &&
      (1LL << config_->max_depth) <= static_cast<int64_t>(config_->num_leaves) + 1;
  return depth_limited &&
         config_->num_leaves <= 2 * kHybridGraphMaxSplitsPerLevel - 1 &&
         config_->max_depth < kHybridGraphMaxLevels &&
         // unrolled graph: max_depth level bodies + the epilogue controller
         // (implied by the gates above, but keep the bound explicit)
         config_->max_depth < kHybridGraphMaxBodies;
}

bool CUDASingleGPUTreeLearner::SetupHybridGraphStatics() {
  // ---- static feature tables ----
  std::vector<CUDAHybridGraphFeatureMeta> meta;
  cuda_data_partition_->BuildHybridGraphFeatureMeta(&meta);
  CHECK_EQ(static_cast<int>(meta.size()), train_data_->num_features());
  std::vector<double> real_thresholds;
  uint32_t threshold_offset = 0;
  for (int feature_index = 0; feature_index < train_data_->num_features(); ++feature_index) {
    meta[feature_index].real_feature_index = train_data_->RealFeatureIndex(feature_index);
    meta[feature_index].missing_type =
      static_cast<int>(train_data_->FeatureBinMapper(feature_index)->missing_type());
    meta[feature_index].real_threshold_offset = threshold_offset;
    const int num_bin = train_data_->FeatureBinMapper(feature_index)->num_bin();
    for (int bin = 0; bin < num_bin; ++bin) {
      real_thresholds.push_back(train_data_->RealThreshold(feature_index, static_cast<uint32_t>(bin)));
    }
    threshold_offset += static_cast<uint32_t>(num_bin);
  }
  cuda_hybrid_graph_feature_meta_.InitFromHostVector(meta);
  cuda_hybrid_graph_real_thresholds_.InitFromHostVector(real_thresholds);
  cuda_hybrid_graph_feature_source_.Resize(static_cast<size_t>(train_data_->num_features()));
  hybrid_graph_journal_size_ = static_cast<size_t>(kHybridGraphJournalHeader) +
    2 * static_cast<size_t>(config_->num_leaves) + 4;
  cuda_hybrid_graph_journal_.Resize(hybrid_graph_journal_size_);
  if (pinned_hybrid_graph_state_ == nullptr) {
    if (cudaHostAlloc(reinterpret_cast<void**>(&pinned_hybrid_graph_state_),
                      sizeof(CUDAHybridGraphLoopState), cudaHostAllocDefault) != cudaSuccess) {
      return false;
    }
  }
  if (pinned_hybrid_graph_journal_ != nullptr) {
    cudaFreeHost(pinned_hybrid_graph_journal_);
    pinned_hybrid_graph_journal_ = nullptr;
  }
  if (cudaHostAlloc(reinterpret_cast<void**>(&pinned_hybrid_graph_journal_),
                    hybrid_graph_journal_size_ * sizeof(int), cudaHostAllocDefault) != cudaSuccess) {
    return false;
  }
  // ---- worst-case capacity of every captured device buffer (no captured
  // pointer may ever reallocate) ----
  const size_t max_slots = static_cast<size_t>(config_->num_leaves) + 2;
  if (hybrid_pair_slots_.Size() < max_slots) {
    hybrid_pair_slots_.Resize(max_slots);
  }
  const int max_pairs = config_->num_leaves / 2 + 2;
  if (cuda_hybrid_pair_descs_.Size() < static_cast<size_t>(max_pairs)) {
    cuda_hybrid_pair_descs_.Resize(static_cast<size_t>(max_pairs));
  }
  cuda_data_partition_->EnsureHybridGraphCapacity(num_data_);
  cuda_best_split_finder_->EnsureHybridGraphCapacity(max_pairs);
  // ---- the static half of the loop state ----
  CUDAHybridGraphLoopState& st = hybrid_graph_state_template_;
  std::memset(&st, 0, sizeof(st));
  st.max_depth = config_->max_depth;
  st.num_leaves_budget = config_->num_leaves;
  // construct grid x / block dims are filled per instance at build time (the
  // compact view's block shape follows the per-tree sampled column count and
  // is part of the graph key)
  st.construct_min_grid_dim_y = cuda_histogram_constructor_->hybrid_graph_min_grid_dim_y();
  st.construct_min_rows_per_thread = CUDAHistogramConstructor::BatchConstructMinRowsPerThread();
  st.construct_saturation_floor = CUDAHistogramConstructor::BatchConstructSaturationFloor();
  // quantized training: enables the construct overflow guard and the device
  // hist-bits derivation in the graph body kernels (0 = non-quantized)
  st.num_grad_quant_bins = cuda_histogram_constructor_->hybrid_graph_num_grad_quant_bins();
  // the 64->32-bit compaction scratch pointer is baked into the captured
  // subtract/copy kernels: presize it for the worst-case pair count
  cuda_histogram_constructor_->EnsureHybridGraphBitChangeCapacity(max_pairs);
  st.leaf_best_split_info = cuda_best_split_finder_->leaf_best_split_info_ptr(0);
  st.leaf_num_data = cuda_data_partition_->cuda_leaf_num_data();
  st.leaf_data_start = cuda_data_partition_->cuda_leaf_data_start();
  st.apply_descs = cuda_data_partition_->hybrid_graph_apply_descs();
  st.pair_descs = cuda_hybrid_pair_descs_.RawData();
  st.pair_slots = hybrid_pair_slots_.RawData();
  st.feature_meta = cuda_hybrid_graph_feature_meta_.RawDataReadOnly();
  st.feature_source = cuda_hybrid_graph_feature_source_.RawDataReadOnly();
  st.real_thresholds = cuda_hybrid_graph_real_thresholds_.RawDataReadOnly();
  st.journal = cuda_hybrid_graph_journal_.RawData();
  hybrid_graph_statics_ready_ = true;
  return true;
}

void CUDASingleGPUTreeLearner::ComputeHybridGraphKey(CUDATree* tree,
                                                     std::vector<const void*>* key) const {
  key->clear();
  cuda_histogram_constructor_->HybridGraphKeyPointers(key);
  cuda_best_split_finder_->HybridGraphKeyPointers(key);
  key->push_back(static_cast<const void*>(tree->hybrid_graph_batch_splits()));
  // the compact view's construct BLOCK shape follows the per-tree sampled
  // column count and block dims are not device-updatable: one instance per
  // shape, captured with the exact host dims (bit-identical results)
  key->push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(
    cuda_histogram_constructor_->HybridGraphCompactShapeKey())));
}

bool CUDASingleGPUTreeLearner::BuildHybridGraphInstance(CUDATree* tree,
                                                        HybridGraphInstance* instance) {
#define HYBRID_GRAPH_SOFT_CHECK(call) \
  do { \
    const cudaError_t soft_check_err_ = (call); \
    if (soft_check_err_ != cudaSuccess) { \
      Log::Warning("graphs L1: %s failed with %s; falling back to the host level loop", \
                   #call, cudaGetErrorString(soft_check_err_)); \
      return false; \
    } \
  } while (0)
  if (cudaMalloc(reinterpret_cast<void**>(&instance->state_dev),
                 sizeof(CUDAHybridGraphLoopState)) != cudaSuccess) {
    return false;
  }
  HYBRID_GRAPH_SOFT_CHECK(cudaGraphCreate(&instance->graph, 0));

  // ---- capture the UNROLLED level bodies (L1.5): max_depth straight-line
  // repetitions of controller -> apply chain -> search chain plus one final
  // epilogue controller, mirroring the host flow's stream/event wiring; each
  // record/wait pair becomes a graph edge, chaining the bodies ----
  cudaStream_t hist_stream = cuda_histogram_constructor_->hist_stream();
  cudaStream_t apply_stream = cuda_data_partition_->apply_stream();
  cudaStream_t find_stream = cuda_best_split_finder_->find_stream();
  cudaEvent_t fork_event = nullptr;
  cudaEvent_t apply_event = nullptr;
  cudaEvent_t join_event = nullptr;
  HYBRID_GRAPH_SOFT_CHECK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
  HYBRID_GRAPH_SOFT_CHECK(cudaEventCreateWithFlags(&apply_event, cudaEventDisableTiming));
  HYBRID_GRAPH_SOFT_CHECK(cudaEventCreateWithFlags(&join_event, cudaEventDisableTiming));
  std::vector<cudaGraphNode_t> nodes;
  std::vector<int> roles;
  std::vector<int> role_static_x;
  bool capture_ok = true;
  size_t nodes_per_level = 0;
  const int num_level_bodies = config_->max_depth;  // gated to < kHybridGraphMaxBodies
  HYBRID_GRAPH_SOFT_CHECK(cudaStreamBeginCaptureToGraph(
    hist_stream, instance->graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
  // graphs A2: device address of the loop state's live split count -- the
  // pow2-frozen tree-split grid's idle blocks guard on it (plain int pointer
  // so the tree's public header stays free of the loop-state type)
  const int* graph_live_split_count = &instance->state_dev->cur_num_splits;
  // deterministic construct inside the graph: decided ONCE for the whole
  // instance (node counts must be body-uniform) against the widest possible
  // level; each body's capture gets its own tighter pair bound so shallow
  // bodies keep a full block dy
  const int det_widest_level_pairs = std::min(
    1 << std::min(num_level_bodies - 1, 10), kHybridGraphMaxSplitsPerLevel);
  const bool graph_det =
    cuda_histogram_constructor_->DetDenseGraphEligible(det_widest_level_pairs);
  for (int body = 0; body < num_level_bodies; ++body) {
    const size_t body_node_start = nodes.size();
    CaptureHybridGraphControllerKernel(hist_stream, instance->state_dev, body);
    capture_ok = capture_ok && cudaEventRecord(fork_event, hist_stream) == cudaSuccess;
    capture_ok = capture_ok && cudaStreamWaitEvent(apply_stream, fork_event, 0) == cudaSuccess;
    tree->CaptureHybridGraphSplitBatchKernel(apply_stream, graph_live_split_count);
    capture_ok = capture_ok && AppendCapturedNode(apply_stream, &nodes);
    roles.push_back(kHybridGraphNodeTreeSplit);
    cuda_data_partition_->CaptureHybridGraphApplyKernels(instance->state_dev, &nodes, &roles);
    while (role_static_x.size() < roles.size()) {
      role_static_x.push_back(0);
    }
    capture_ok = capture_ok && cudaEventRecord(apply_event, apply_stream) == cudaSuccess;
    capture_ok = capture_ok && cudaStreamWaitEvent(hist_stream, apply_event, 0) == cudaSuccess;
    const int det_body_pairs = graph_det ?
      std::min(1 << std::min(body, 10), det_widest_level_pairs) : 0;
    cuda_histogram_constructor_->CaptureHybridGraphSearchKernels(
      cuda_hybrid_pair_descs_.RawDataReadOnly(),
      cuda_data_partition_->level_smaller_leaf_counts(),
      instance->state_dev,
      det_body_pairs,
      &nodes, &roles, &role_static_x);
    cuda_best_split_finder_->CaptureHybridGraphFindKernels(
      cuda_hybrid_pair_descs_.RawDataReadOnly(),
      config_->use_quantized_grad ? cuda_gradient_discretizer_->grad_scale_ptr() : nullptr,
      config_->use_quantized_grad ? cuda_gradient_discretizer_->hess_scale_ptr() : nullptr,
      instance->state_dev,
      &nodes, &roles, &role_static_x);
    capture_ok = capture_ok && cudaEventRecord(join_event, find_stream) == cudaSuccess;
    capture_ok = capture_ok && cudaStreamWaitEvent(hist_stream, join_event, 0) == cudaSuccess;
    const size_t body_nodes = nodes.size() - body_node_start;
    if (body == 0) {
      nodes_per_level = body_nodes;
    } else if (body_nodes != nodes_per_level) {
      capture_ok = false;  // non-uniform body capture; bail out below
    }
  }
  // epilogue controller: stops the loop (journal header + done flag) after
  // the last level body's speculative search
  CaptureHybridGraphControllerKernel(hist_stream, instance->state_dev, num_level_bodies);
  cudaGraph_t captured = nullptr;
  const cudaError_t end_err = cudaStreamEndCapture(hist_stream, &captured);
  cudaEventDestroy(fork_event);
  cudaEventDestroy(apply_event);
  cudaEventDestroy(join_event);
  if (end_err != cudaSuccess || !capture_ok ||
      nodes.size() != roles.size() || roles.size() != role_static_x.size() ||
      nodes_per_level > static_cast<size_t>(kHybridGraphMaxNodes) || nodes_per_level < 10) {
    Log::Warning("graphs L1: body capture failed (%s); falling back to the host level loop",
                 cudaGetErrorString(end_err));
    return false;
  }

  // ---- device-updatable node handles (per-body slices at stride
  // kHybridGraphMaxNodes, matching the controller's node_base) ----
  CUDAHybridGraphLoopState state = hybrid_graph_state_template_;
  state.tree_batch_splits = tree->hybrid_graph_batch_splits();
  // per-instance construct extents: block dims follow this tree's (compact)
  // shape and were baked into the capture above
  int construct_grid_x = 0;
  int construct_block_dim_y = 0;
  cuda_histogram_constructor_->HybridGraphConstructDims(&construct_grid_x, &construct_block_dim_y);
  state.construct_grid_x = construct_grid_x;
  state.construct_block_dim_y = construct_block_dim_y;
  state.det_total_slot_rows =
    graph_det ? cuda_histogram_constructor_->det_dense_total_slot_rows() : 0;
  state.det_rows_per_thread = CUDAHistogramConstructor::det_rows_per_thread();
  state.num_nodes = static_cast<int>(nodes_per_level);
  for (size_t i = 0; i < nodes.size(); ++i) {
    const size_t slot = (i / nodes_per_level) * static_cast<size_t>(kHybridGraphMaxNodes) +
      (i % nodes_per_level);
    cudaLaunchAttributeValue attr_value;
    std::memset(&attr_value, 0, sizeof(attr_value));
    attr_value.deviceUpdatableKernelNode.deviceUpdatable = 1;
    HYBRID_GRAPH_SOFT_CHECK(cudaGraphKernelNodeSetAttribute(
      nodes[i], cudaLaunchAttributeDeviceUpdatableKernelNode, &attr_value));
    state.nodes[slot].node = attr_value.deviceUpdatableKernelNode.devNode;
    state.nodes[slot].role = roles[i];
    state.nodes[slot].static_x = role_static_x[i];
    state.node_enabled[slot] = 1;  // captured nodes start enabled
    if (state.nodes[slot].node == nullptr) {
      Log::Warning("graphs L1: null device node handle; falling back to the host level loop");
      return false;
    }
  }
  HYBRID_GRAPH_SOFT_CHECK(cudaGraphInstantiate(&instance->exec, instance->graph, 0));
  HYBRID_GRAPH_SOFT_CHECK(cudaMemcpy(instance->state_dev, &state,
    sizeof(CUDAHybridGraphLoopState), cudaMemcpyHostToDevice));
  // a graph that updates its own device-updatable nodes from the device MUST
  // be uploaded before it is launched (CUDA launch-attribute docs); without
  // this the controller's first-launch updates race the lazy upload:
  // intermittent cudaErrorInvalidValue device-side updates or
  // cudaErrorIllegalAddress at the post-graph sync, ~20%/run on multiclass
  // (7 first launches per run) at machine-state-dependent rates
  HYBRID_GRAPH_SOFT_CHECK(cudaGraphUpload(instance->exec,
    cuda_best_split_finder_->find_stream()));
  ComputeHybridGraphKey(tree, &instance->key);
  const bool hybrid_diag = FalcataDebug().diag;
  if (hybrid_diag) {
    // fprintf like the other hybrid-diag lines: visible under verbose=-1
    fprintf(stderr, "[hybrid-diag] graphs L1.5: instantiated unrolled device level loop "
            "(%d bodies x %d nodes, %d cached)\n",
            num_level_bodies, state.num_nodes, static_cast<int>(hybrid_graph_cache_.size()) + 1);
  }
  return true;
#undef HYBRID_GRAPH_SOFT_CHECK
}

CUDASingleGPUTreeLearner::HybridGraphInstance* CUDASingleGPUTreeLearner::GetOrBuildHybridGraph(
    CUDATree* tree) {
  if (cuda_best_split_finder_->num_used_tasks() == 0) {
    return nullptr;  // no usable feature this tree; the host loop handles it
  }
  if (!hybrid_graph_statics_ready_) {
    if (!SetupHybridGraphStatics()) {
      hybrid_graph_disabled_ = true;
      return nullptr;
    }
  }
  std::vector<const void*> key;
  ComputeHybridGraphKey(tree, &key);
  for (size_t i = 0; i < hybrid_graph_cache_.size(); ++i) {
    if (hybrid_graph_cache_[i]->key == key) {
      // LRU: move the hit to the back (multiclass cycles one key per class,
      // the compact column view alternates two buffers -- the steady state
      // must keep every cycling key resident or each tree pays a rebuild)
      if (i + 1 != hybrid_graph_cache_.size()) {
        std::rotate(hybrid_graph_cache_.begin() + i,
                    hybrid_graph_cache_.begin() + i + 1,
                    hybrid_graph_cache_.end());
      }
      return hybrid_graph_cache_.back().get();
    }
  }
  // device-updatable graphs cannot be host-updated after instantiation, so a
  // new pointer key needs a fresh capture. Steady state: (#gradient buffers,
  // i.e. one per class) x (compact double buffer). The compact buffers also
  // grow to a running max early on, retiring old keys; evict LRU for those.
  // A key set larger than the cache would rebuild every tree -- give up on
  // the graph path instead (the host loop is bit-for-bit correct).
  const size_t cache_limit = 64;
  if (hybrid_graph_cache_.size() >= cache_limit) {
    ++hybrid_graph_build_failures_;
    if (hybrid_graph_build_failures_ >= 8) {
      hybrid_graph_disabled_ = true;
      Log::Warning("graphs L1: per-tree buffer pointers keep churning; using the host level loop");
      return nullptr;
    }
    hybrid_graph_cache_.erase(hybrid_graph_cache_.begin());
  }
  auto instance = std::unique_ptr<HybridGraphInstance>(new HybridGraphInstance());
  if (!BuildHybridGraphInstance(tree, instance.get())) {
    hybrid_graph_build_failures_ += 4;
    if (hybrid_graph_build_failures_ >= 8) {
      hybrid_graph_disabled_ = true;
    }
    return nullptr;
  }
  hybrid_graph_cache_.push_back(std::move(instance));
  return hybrid_graph_cache_.back().get();
}

int CUDASingleGPUTreeLearner::TrainLevelWisePrefixGraph(CUDATree* tree) {
  HybridGraphInstance* instance = GetOrBuildHybridGraph(tree);
  if (instance == nullptr) {
    return -1;  // host loop fallback
  }
  // root search exactly like the host one-sync flow (host-known exact grid)
  EnqueueRootLevelSearchOneSync();
  cudaStream_t find_stream = cuda_best_split_finder_->find_stream();
  // per-tree inputs: the column source table (per-tree compact/packed view)
  // and the mutable prefix of the loop state
  cuda_data_partition_->BuildHybridGraphFeatureSource(&host_hybrid_graph_feature_source_);
  CopyFromHostToCUDADeviceAsync<CUDAHybridGraphFeatureSource>(
    cuda_hybrid_graph_feature_source_.RawData(), host_hybrid_graph_feature_source_.data(),
    host_hybrid_graph_feature_source_.size(), find_stream, __FILE__, __LINE__);
  CUDAHybridGraphLoopState* staged = pinned_hybrid_graph_state_;
  staged->main_indices = cuda_data_partition_->hybrid_graph_main_indices();
  staged->out_indices = cuda_data_partition_->hybrid_graph_out_indices();
  staged->level = 0;
  staged->num_leaves = 1;
  staged->split_info_base = 0;
  staged->journal_split_cursor = 0;
  staged->root_num_data = cuda_data_partition_->root_num_data();
  staged->find_grid_x = cuda_best_split_finder_->hybrid_graph_find_grid_x();
  staged->done = 0;
  // live counts are controller-owned (written before each body's kernels run);
  // reset them with the prefix so a stale value can never leak across trees
  staged->cur_num_splits = 0;
  staged->cur_num_gaps = 0;
  CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(instance->state_dev, staged,
    offsetof(CUDAHybridGraphLoopState, max_depth), cudaMemcpyHostToDevice, find_stream));
  // ---- the ONE host launch of the whole prefix ----
  CUDASUCCESS_OR_FATAL(cudaGraphLaunch(instance->exec, find_stream));
  // ---- the ONE synchronization: EVERY post-graph readback (journal,
  // deferred split-info slab, per-leaf best splits for the tail's collect) is
  // prefetched stream-ordered behind the graph at a host-known upper-bound
  // size (journal[1] <= num_leaves - 1 splits; the collect reads
  // <= num_leaves leaves), then one stream sync covers all three. Nothing
  // mutates these device buffers between graph completion and the previous
  // in-place synchronous copies (FinishHybridGraphLevels is host-only
  // bookkeeping, FinishSplitBatch/SyncAllLeafBestSplitsToHost are pure
  // reads), so the prefetched values are bit-for-bit identical ----
  CopyFromCUDADeviceToHostAsync<int>(pinned_hybrid_graph_journal_,
    cuda_hybrid_graph_journal_.RawData(), hybrid_graph_journal_size_, find_stream,
    __FILE__, __LINE__);
  cuda_data_partition_->PrefetchSplitBatchAsync(config_->num_leaves - 1, find_stream);
  cuda_best_split_finder_->PrefetchLeafBestSplitsAsync(config_->num_leaves, find_stream);
  CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(find_stream));
  const int* journal = pinned_hybrid_graph_journal_;
  if (journal[2] != 0) {
    Log::Fatal("graphs L1: device-side graph update failed (flags 0x%x)", journal[2]);
  }
  const int num_levels = journal[0];
  const int total_splits = journal[1];
  CHECK_GE(num_levels, 0);
  CHECK_LE(num_levels, kHybridGraphMaxLevels);
  CHECK_GE(total_splits, 0);
  CHECK_LE(total_splits, config_->num_leaves - 1);
  int num_splits = 0;
  if (total_splits > 0) {
    cuda_data_partition_->FinishHybridGraphLevels(num_levels, total_splits);
    // deferred split info of the WHOLE prefix (cumulative slab slots),
    // already staged by the prefetch above
    cuda_data_partition_->ReadPrefetchedSplitBatch(total_splits, &hybrid_graph_batch_info_);
    std::vector<HybridAppliedSplit>& applied = hybrid_graph_applied_scratch_;
    applied.clear();
    int cursor = 0;
    for (int level = 0; level < num_levels; ++level) {
      const int level_splits = journal[3 + level];
      CHECK_GE(level_splits, 0);
      for (int k = 0; k < level_splits; ++k) {
        const int left_leaf = journal[kHybridGraphJournalHeader + 2 * (cursor + k)];
        const int inner_feature = journal[kHybridGraphJournalHeader + 2 * (cursor + k) + 1];
        const int right_leaf = tree->num_leaves();
        // host-mirror half of SplitBatch (device updates ran inside the graph)
        tree->ReplaySplitHostMirrors(left_leaf, train_data_->RealFeatureIndex(inner_feature));
        applied.push_back({left_leaf, right_leaf, nullptr, nullptr});
      }
      cursor += level_splits;
    }
    CHECK_EQ(cursor, total_splits);
    // batch_info slots are in global applied order, exactly what
    // FinishLevelBookkeeping's 18-ints-per-split walk expects
    FinishLevelBookkeeping(applied, hybrid_graph_batch_info_, nullptr, &num_splits);
  }
  if (tree->num_leaves() < config_->num_leaves) {
    // mirror the host loop's exit state: the final level's readback + collect
    // (skipped when the budget is exhausted -- the host loop breaks without
    // re-reading after a final partial level, and the tail never runs);
    // already staged by the prefetch above
    cuda_best_split_finder_->ReadPrefetchedLeafBestSplits(tree->num_leaves(), &host_leaf_best_splits_);
    CollectSplittableLeaves(tree, &hybrid_graph_splittable_scratch_);
  }
  if (num_splits == 0) {
    // the root has no valid split: the leaf-wise tail (and a possible stump)
    // needs the host root sums after all
    EnsureRootSumsReadBack(tree);
  }
  return num_splits;
}

void CUDASingleGPUTreeLearner::ReleaseHybridGraphs() {
  hybrid_graph_cache_.clear();
  hybrid_graph_statics_ready_ = false;
}

#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED

// ---- selective (grow-then-prune) hybrid growth ----

void CUDASingleGPUTreeLearner::SelectiveReset() {
  // hybrid leaf indices are recycled on collapse, so live leaves never exceed
  // num_leaves (the greedy selection holds <= num_leaves - 1 splits and fresh
  // indices are only allocated when the free list is empty, i.e. when every
  // allocated index is live); +2 is head room for the CHECK below
  const size_t cap = static_cast<size_t>(config_->num_leaves) + 2;
  sel_applied_.clear();
  sel_frontier_.clear();
  sel_leaf_parent_.assign(cap, -1);
  sel_leaf_depth_.assign(cap, 0);
  sel_leaf_classic_.assign(cap, 0);
  sel_leaf_frontier_.assign(cap, -1);
  sel_leaf_live_.assign(cap, 0);
  sel_free_leaves_.clear();
  sel_order_.clear();
  sel_order_classic_leaf_.clear();
  sel_level_apply_.clear();
  sel_num_allocated_ = 1;
  sel_num_splits_ = 0;
  sel_leaf_live_[0] = 1;
}

void CUDASingleGPUTreeLearner::SelectiveDiscoverFrontier(const std::vector<int>& child_leaves) {
  for (const SelectiveFrontier& f : sel_frontier_) {
    sel_leaf_frontier_[f.leaf] = -1;
  }
  sel_frontier_.clear();
  for (const int leaf : child_leaves) {
    const CUDASplitInfo& info = host_leaf_best_splits_[leaf];
    if (!info.is_valid) {
      continue;  // permanently dormant leaf (no positive gain / gates failed)
    }
    // diag: a candidate whose finder counts cannot reproduce the leaf's
    // actual row count was not computed from this leaf's histogram -- the
    // signature of the stale-slot resurrection bug the sync kernels guard
    // against (see SyncBestSplitForLevelKernel). Kept as a canary; counts
    // are hessian-derived estimates off-L2, so this only warns.
    if (FalcataDebug().diag &&
        info.left_count + info.right_count != leaf_num_data_[leaf]) {
      fprintf(stderr,
              "[selective-diag] stale candidate? tree=%" PRId64 " leaf=%d "
              "actual_n=%d finder_n=%d feat=%d thr=%u gain=%.17g\n",
              static_cast<int64_t>(sel_stat_trees_), leaf,
              static_cast<int>(leaf_num_data_[leaf]),
              static_cast<int>(info.left_count + info.right_count),
              info.inner_feature_index, info.threshold, info.gain);
    }
    sel_leaf_frontier_[leaf] = static_cast<int>(sel_frontier_.size());
    sel_frontier_.push_back({sel_leaf_parent_[leaf], leaf, info.gain});
  }
}

int CUDASingleGPUTreeLearner::SelectiveTieBreak(const std::vector<int>& tied_classic_leaves) const {
  // FindBestFromAllSplitsKernel runs with 256 threads: thread t scans leaves
  // t, t + 256, ... with a strict-greater update (the SMALLEST tied leaf wins
  // within a thread), then an 8-warp shuffle-down reduction and a final warp
  // reduction combine the threads, each with a strict-greater comparison that
  // KEEPS the receiving lane's value on ties. Candidates with smaller gains
  // can never displace a max-gain candidate and are always displaced by one,
  // so only the tied candidates' lane positions matter: simulate the exact
  // reduction over occupied/empty lanes.
  constexpr int kThreads = 256;  // NUM_THREADS_FIND_BEST_LEAF
  std::array<int, kThreads> thread_winner;
  thread_winner.fill(-1);
  for (const int leaf : tied_classic_leaves) {
    const int t = leaf % kThreads;
    if (thread_winner[t] < 0 || leaf < thread_winner[t]) {
      thread_winner[t] = leaf;
    }
  }
  // shuffle-down reduction over 32 lanes: lane i takes lane i + offset's value
  // only when lane i holds no tied candidate (out-of-warp reads return the
  // caller's own value); all lanes update simultaneously from the pre-step state
  const auto reduce32 = [](std::array<int, 32> lanes) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      std::array<int, 32> next = lanes;
      for (int i = 0; i + offset < 32; ++i) {
        if (lanes[i] < 0 && lanes[i + offset] >= 0) {
          next[i] = lanes[i + offset];
        }
      }
      lanes = next;
    }
    return lanes[0];
  };
  std::array<int, 32> lanes;
  std::array<int, 32> warp_winner_lanes;
  warp_winner_lanes.fill(-1);
  for (int w = 0; w < kThreads / 32; ++w) {
    for (int l = 0; l < 32; ++l) {
      lanes[l] = thread_winner[w * 32 + l];
    }
    warp_winner_lanes[w] = reduce32(lanes);
  }
  const int winner = reduce32(warp_winner_lanes);
  CHECK_GE(winner, 0);
  return winner;
}

int CUDASingleGPUTreeLearner::RunSelectiveLevel() {
  const auto sim_t0 = std::chrono::steady_clock::now();
  const int budget = config_->num_leaves - 1;
  // ---- 1. greedy simulation over live applied splits + the current frontier.
  // Exactly the classic leaf-wise selection: repeatedly take the max-gain
  // candidate whose parent split is already selected, until the budget binds or
  // no candidate remains. Ties go to the smaller CLASSIC leaf index, matching
  // FindBestFromAllSplitsKernel's reduction (exact for <= 256 concurrent
  // leaves; equal double gains across leaves are otherwise vanishingly rare).
  // Classic leaf indices are assigned along the way: the t-th selected split
  // keeps its leaf index for the left child and hands index t + 1 to the right
  // child, exactly as the classic loop numbers them.
  for (SelectiveApplied& node : sel_applied_) {
    node.selected = false;
  }
  sel_order_.clear();
  sel_order_classic_leaf_.clear();
  struct HeapEntry {
    double gain;
    int classic_leaf;
    int id;  // >= 0: applied record id; < 0: ~frontier index
  };
  const auto worse = [](const HeapEntry& a, const HeapEntry& b) {
    return a.gain < b.gain || (a.gain == b.gain && a.classic_leaf > b.classic_leaf);
  };
  std::priority_queue<HeapEntry, std::vector<HeapEntry>, decltype(worse)> heap(worse);
  sel_leaf_classic_[0] = 0;
  if (!sel_applied_.empty()) {
    // record 0 is the root split (first pop of the first simulation); the root
    // is always available first, so it can never be displaced
    heap.push({sel_applied_[0].gain, 0, 0});
  } else if (!sel_frontier_.empty()) {
    heap.push({sel_frontier_[0].gain, 0, ~0});  // the root candidate (leaf 0)
  }
  const auto push_child = [this, &heap](const SelectiveApplied& node, const bool left,
                                        const int classic_leaf) {
    const int kid = left ? node.kid_left : node.kid_right;
    if (kid >= 0) {
      heap.push({sel_applied_[kid].gain, classic_leaf, kid});
      return;
    }
    const int leaf = left ? node.left_leaf : node.right_leaf;
    const int fidx = sel_leaf_frontier_[leaf];
    if (fidx >= 0) {
      heap.push({sel_frontier_[fidx].gain, classic_leaf, ~fidx});
    }
  };
  std::vector<HeapEntry> tied;
  while (static_cast<int>(sel_order_.size()) < budget && !heap.empty()) {
    HeapEntry e = heap.top();
    heap.pop();
    if (!heap.empty() && heap.top().gain == e.gain) {
      // equal-gain candidates: resolve with the exact device tie-break
      tied.clear();
      tied.push_back(e);
      while (!heap.empty() && heap.top().gain == e.gain) {
        tied.push_back(heap.top());
        heap.pop();
      }
      std::vector<int> tied_leaves;
      tied_leaves.reserve(tied.size());
      for (const HeapEntry& c : tied) {
        tied_leaves.push_back(c.classic_leaf);
      }
      const int winner_leaf = SelectiveTieBreak(tied_leaves);
      for (const HeapEntry& c : tied) {
        if (c.classic_leaf == winner_leaf) {
          e = c;
        } else {
          heap.push(c);
        }
      }
    }
    const int t = static_cast<int>(sel_order_.size());
    sel_order_.push_back(e.id);
    sel_order_classic_leaf_.push_back(e.classic_leaf);
    if (e.id >= 0) {
      SelectiveApplied& node = sel_applied_[e.id];
      node.selected = true;
      sel_leaf_classic_[node.left_leaf] = e.classic_leaf;
      sel_leaf_classic_[node.right_leaf] = t + 1;
      push_child(node, true, e.classic_leaf);
      push_child(node, false, t + 1);
    }
    // selected frontier candidates have no discovered children yet
  }

  // ---- 2. eager collapse of displaced applied splits. By monotonicity a
  // displaced split can never re-enter the selection, and by availability
  // closure its whole subtree is displaced with it: revert the topmost
  // displaced node to a leaf (its left-child index), rewrite the window's
  // data-index-to-leaf-index entries to it, and recycle every right-child leaf
  // index (and histogram slot) of the subtree. Children ids are always larger
  // than their parent's, so the ascending scan visits topmost nodes first and
  // the dead flag skips their (already collapsed) descendants.
  sel_collapse_windows_.clear();
  sel_freed_hist_slots_.clear();
  std::vector<int> stack;
  int num_displaced = 0;
  for (int id = 0; id < static_cast<int>(sel_applied_.size()); ++id) {
    SelectiveApplied& node = sel_applied_[id];
    if (node.dead || node.selected) {
      continue;
    }
    // detach from the (still selected) parent's child slot
    if (node.parent >= 0) {
      SelectiveApplied& p = sel_applied_[node.parent];
      if (p.kid_left == id) {
        p.kid_left = -1;
      } else if (p.kid_right == id) {
        p.kid_right = -1;
      }
    }
    stack.clear();
    stack.push_back(id);
    while (!stack.empty()) {
      const int m_id = stack.back();
      stack.pop_back();
      SelectiveApplied& m = sel_applied_[m_id];
      m.dead = true;
      ++num_displaced;
      // the right-child leaf index (and its histogram slot, which is always
      // cuda_hist + 2 * right_leaf * num_total_bin) becomes reusable; the slot
      // must be re-zeroed (construct kernels accumulate into assumed-zero slots)
      sel_leaf_live_[m.right_leaf] = 0;
      sel_leaf_parent_[m.right_leaf] = -1;
      sel_free_leaves_.push_back(m.right_leaf);
      sel_freed_hist_slots_.push_back(m.right_leaf);
      if (m.kid_left >= 0) {
        stack.push_back(m.kid_left);
      }
      if (m.kid_right >= 0) {
        stack.push_back(m.kid_right);
      }
    }
    // the collapsed node reverts to a (permanently dormant) leaf at its
    // left-child index, with its pre-split window
    sel_leaf_parent_[node.left_leaf] = node.parent;
    leaf_num_data_[node.left_leaf] = node.num_data;
    leaf_data_start_[node.left_leaf] = node.data_start;
    sel_collapse_windows_.push_back({node.data_start, node.num_data, node.left_leaf});
  }
  if (!sel_collapse_windows_.empty()) {
    // device work is idle here (post-readback); the rewrite goes on the apply
    // stream ahead of this level's batched apply, the slot zeroing on the
    // legacy stream ahead of the next construct
    cuda_data_partition_->CollapseLeafWindows(sel_collapse_windows_);
    cuda_histogram_constructor_->ZeroHistSlots(sel_freed_hist_slots_);
  }

  // ---- 3. create the applied records for the selected frontier candidates (in
  // greedy order; right-child indices from the free list, else fresh) ----
  sel_level_apply_.clear();
  for (size_t t = 0; t < sel_order_.size(); ++t) {
    const int id = sel_order_[t];
    if (id >= 0) {
      continue;
    }
    const SelectiveFrontier& f = sel_frontier_[~id];
    int right = -1;
    if (!sel_free_leaves_.empty()) {
      right = sel_free_leaves_.back();
      sel_free_leaves_.pop_back();
    } else {
      right = sel_num_allocated_++;
    }
    // invariant: live leaves <= num_leaves, and fresh indices are allocated
    // only when every allocated index is live, so this can never fire (the
    // host leaf arrays are sized config_->num_leaves)
    CHECK_LT(right, config_->num_leaves);
    const int new_id = static_cast<int>(sel_applied_.size());
    SelectiveApplied node;
    node.parent = f.parent;
    node.left_leaf = f.leaf;
    node.right_leaf = right;
    node.kid_left = -1;
    node.kid_right = -1;
    node.dead = false;
    node.selected = true;
    node.gain = f.gain;
    node.data_start = leaf_data_start_[f.leaf];
    node.num_data = leaf_num_data_[f.leaf];
    node.info = host_leaf_best_splits_[f.leaf];
    if (node.info.num_cat_threshold > 0) {
      // snapshot the inner threshold bins now: the leaf's slab slot is reused
      // as soon as the (recycled) leaf index is searched again
      cuda_best_split_finder_->CopyLeafCatThresholdToHost(
        f.leaf, node.info.num_cat_threshold, &node.cat_inner);
    }
    sel_applied_.push_back(node);
    if (f.parent >= 0) {
      SelectiveApplied& p = sel_applied_[f.parent];
      if (p.left_leaf == f.leaf) {
        p.kid_left = new_id;
      } else {
        p.kid_right = new_id;
      }
    }
    sel_leaf_parent_[f.leaf] = new_id;
    sel_leaf_parent_[right] = new_id;
    sel_leaf_live_[right] = 1;
    sel_leaf_depth_[right] = sel_leaf_depth_[f.leaf] + 1;
    ++sel_leaf_depth_[f.leaf];
    sel_level_apply_.push_back(new_id);
  }
  sel_stat_applied_ += static_cast<int64_t>(sel_level_apply_.size());
  sel_stat_displaced_ += num_displaced;
  ++sel_stat_levels_;
  sel_stat_sim_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - sim_t0).count();
  return static_cast<int>(sel_level_apply_.size());
}

void CUDASingleGPUTreeLearner::ApplyLevelBatchedSelective(std::vector<HybridAppliedSplit>* applied) {
  applied->clear();
  applied->reserve(sel_level_apply_.size());
  host_apply_split_inputs_.clear();
  size_t slot_id = 0;
  for (const int id : sel_level_apply_) {
    const SelectiveApplied& node = sel_applied_[id];
    CUDALeafSplitsStruct* smaller_slot = hybrid_pair_slots_.RawData() + slot_id;
    CUDALeafSplitsStruct* larger_slot = hybrid_pair_slots_.RawData() + slot_id + 1;
    host_apply_split_inputs_.push_back({
      cuda_best_split_finder_->leaf_best_split_info_ptr(node.left_leaf),
      node.left_leaf,
      node.right_leaf,
      node.info.inner_feature_index,
      node.info.threshold,
      static_cast<uint8_t>(node.info.default_left),
      node.info.num_cat_threshold,
      node.num_data,
      node.data_start,
      smaller_slot,
      larger_slot});
    applied->push_back({node.left_leaf, node.right_leaf, smaller_slot, larger_slot});
    slot_id += 2;
  }
  cuda_data_partition_->SplitLevelBatched(host_apply_split_inputs_);
}

void CUDASingleGPUTreeLearner::TrainSelectiveOneSync(CUDATree* tree) {
  EnqueueRootLevelSearchOneSync();
  std::vector<HybridAppliedSplit> pending;  // applied splits awaiting readback
  std::vector<HybridAppliedSplit> applied;
  std::vector<int> batch_info;
  std::vector<int> child_leaves;
  while (true) {
    // the ONE device synchronization of the level
    cuda_best_split_finder_->SyncAllLeafBestSplitsToHost(sel_num_allocated_, &host_leaf_best_splits_);
    child_leaves.clear();
    if (!pending.empty()) {
      cuda_data_partition_->FinishSplitBatch(static_cast<int>(pending.size()), &batch_info);
      FinishLevelBookkeeping(pending, batch_info, nullptr, &sel_num_splits_);
      for (const HybridAppliedSplit& a : pending) {
        child_leaves.push_back(a.left);
        child_leaves.push_back(a.right);
      }
      pending.clear();
    } else if (sel_applied_.empty()) {
      child_leaves.push_back(0);  // the root candidate
    }
    SelectiveDiscoverFrontier(child_leaves);
    if (RunSelectiveLevel() == 0) {
      // no frontier candidate enters the greedy selection: by monotonicity the
      // live applied splits ARE the final leaf-wise selection
      break;
    }
    ApplyLevelBatchedSelective(&applied);
    // speculative: the children's search goes out before their statistics are
    // host-known; the next iteration's readback collects both
    EnqueueLevelBestSplitSearchSpeculative(tree, applied);
    pending.swap(applied);
  }
}

void CUDASingleGPUTreeLearner::TrainSelectiveTwoSync(CUDATree* tree) {
  // the two-sync flow builds host-known pair descriptors, which need the host
  // root sums (a no-op unless BeforeTrain optimistically deferred the readback)
  EnsureRootSumsReadBack(tree);
  std::vector<HybridPendingPair> pairs;
  pairs.push_back({smaller_leaf_index_, larger_leaf_index_, /*parent=*/-1,
                   cuda_smaller_leaf_splits_->GetCUDAStruct(),
                   cuda_larger_leaf_splits_->GetCUDAStruct()});
  const bool use_batched_level_kernels = use_hybrid_batch_kernels_ &&
    cuda_histogram_constructor_->SupportsBatchedLevel() &&
    cuda_best_split_finder_->SupportsBatchedLevel();
  const bool hybrid_diag = FalcataDebug().diag;
  if (hybrid_diag) {
    static bool diag_logged = false;
    if (!diag_logged) {
      diag_logged = true;
      fprintf(stderr, "[hybrid-diag] selective-two-sync: batched_level_kernels=%d batch_kernels_env=%d\n",
                   static_cast<int>(use_batched_level_kernels),
                   static_cast<int>(use_hybrid_batch_kernels_));
      fprintf(stderr, "[hybrid-diag] %s\n", cuda_histogram_constructor_->BatchedLevelGateDiag().c_str());
      fprintf(stderr, "[hybrid-diag] %s\n", cuda_best_split_finder_->BatchedLevelGateDiag().c_str());
    }
  }
  std::vector<HybridAppliedSplit> applied;
  std::vector<int> batch_info;
  std::vector<int> child_leaves;
  while (true) {
    if (use_batched_level_kernels) {
      EnqueueLevelBestSplitSearch(tree, pairs);
    } else {
      int pair_counter = 0;
      for (const HybridPendingPair& pair : pairs) {
        EnqueuePairBestSplitSearch(tree, pair.smaller_struct, pair.larger_struct,
                                   pair.smaller, pair.larger, /*synchronize=*/false,
                                   pair_counter % CUDAHistogramConstructor::kNumHistPipelines,
                                   pair.parent);
        ++pair_counter;
      }
    }
    cuda_best_split_finder_->SyncAllLeafBestSplitsToHost(sel_num_allocated_, &host_leaf_best_splits_);
    child_leaves.clear();
    for (const HybridPendingPair& pair : pairs) {
      child_leaves.push_back(pair.smaller);
      if (pair.larger >= 0) {
        child_leaves.push_back(pair.larger);
      }
    }
    SelectiveDiscoverFrontier(child_leaves);
    if (RunSelectiveLevel() == 0) {
      break;
    }
    ApplyLevelBatchedSelective(&applied);
    pairs.clear();
    cuda_data_partition_->FinishSplitBatch(static_cast<int>(applied.size()), &batch_info);
    FinishLevelBookkeeping(applied, batch_info, &pairs, &sel_num_splits_);
  }
}

void CUDASingleGPUTreeLearner::SelectiveFinalize(CUDATree* tree) {
  // the root output seeds the internal_value_ chain of the rebuilt tree (and is
  // the stump value); a no-op unless the one-sync flow deferred the readback
  EnsureRootSumsReadBack(tree);
  // the final greedy order is the last simulation's order: it selected no
  // frontier candidate, hence (by the eager collapse invariant) exactly the
  // live applied splits, in classic leaf-wise apply order with classic
  // node/leaf numbering
  std::vector<CUDATreeHostSplit> seq;
  seq.reserve(sel_order_.size());
  for (size_t t = 0; t < sel_order_.size(); ++t) {
    const int id = sel_order_[t];
    CHECK_GE(id, 0);
    const SelectiveApplied& node = sel_applied_[id];
    const int inner_feature = node.info.inner_feature_index;
    const bool is_cat_split = node.info.num_cat_threshold > 0;
    CUDATreeHostSplit s;
    s.leaf_index = sel_order_classic_leaf_[t];
    s.real_feature_index = train_data_->RealFeatureIndex(inner_feature);
    // categorical payloads carry no numerical threshold; RealThreshold would
    // index the bin-upper-bound array with garbage
    s.real_threshold = is_cat_split ? 0.0 :
      train_data_->RealThreshold(inner_feature, node.info.threshold);
    s.missing_type = static_cast<int>(train_data_->FeatureBinMapper(inner_feature)->missing_type());
    s.inner_feature_index = inner_feature;
    s.threshold_in_bin = is_cat_split ? 0 : node.info.threshold;
    if (is_cat_split) {
      // mirror LaunchConstructBitsetForCategoricalSplitKernel on host: the
      // inner bitset is over the snapshotted bin values, the real bitset over
      // categorical_bin_to_value_[offset + bin]
      CHECK_EQ(static_cast<int>(node.cat_inner.size()), node.info.num_cat_threshold);
      s.num_cat_threshold = node.info.num_cat_threshold;
      const int* bin_to_value = categorical_bin_to_value_.data() +
        categorical_bin_offsets_[inner_feature];
      uint32_t max_inner = 0;
      int max_real = 0;
      for (const uint32_t bin : node.cat_inner) {
        max_inner = std::max(max_inner, bin);
        max_real = std::max(max_real, bin_to_value[bin]);
      }
      s.cat_bitset_inner.assign(static_cast<size_t>(max_inner / 32 + 1), 0);
      s.cat_bitset.assign(static_cast<size_t>(max_real / 32 + 1), 0);
      for (const uint32_t bin : node.cat_inner) {
        s.cat_bitset_inner[bin / 32] |= (0x1u << (bin % 32));
        const uint32_t real = static_cast<uint32_t>(bin_to_value[bin]);
        s.cat_bitset[real / 32] |= (0x1u << (real % 32));
      }
    }
    s.gain = node.info.gain;
    s.default_left = node.info.default_left ? 1 : 0;
    s.left_sum_hessians = node.info.left_sum_hessians;
    s.right_sum_hessians = node.info.right_sum_hessians;
    s.left_count = node.info.left_count;
    s.right_count = node.info.right_count;
    s.left_value = node.info.left_value;
    s.right_value = node.info.right_value;
    seq.push_back(s);
  }
  tree->RebuildFromHostSplits(seq);
  const int final_num_leaves = tree->num_leaves();
  if (final_num_leaves < config_->num_leaves) {
    Log::Warning("No further splits with positive gain, training stopped with %d leaves.",
                 final_num_leaves);
  }
  // hybrid -> final leaf index map (pruned subtrees' rows were already rewritten
  // to their reverted dormant leaf at collapse time, so every row's entry names
  // a live hybrid leaf) + final per-leaf data layout, host and device
  std::vector<int> remap(sel_num_allocated_, 0);
  std::vector<data_size_t> final_num(final_num_leaves, 0);
  std::vector<data_size_t> final_start(final_num_leaves, 0);
  int live_count = 0;
  for (int leaf = 0; leaf < sel_num_allocated_; ++leaf) {
    if (!sel_leaf_live_[leaf]) {
      continue;
    }
    const int classic = sel_leaf_classic_[leaf];
    remap[leaf] = classic;
    final_num[classic] = leaf_num_data_[leaf];
    final_start[classic] = leaf_data_start_[leaf];
    ++live_count;
  }
  CHECK_EQ(live_count, final_num_leaves);
  // the batched apply defers the row -> leaf map; with the final classic
  // layout installed below, one materialize pass writes classic leaf ids
  // directly from the final windows (replacing the old remap of per-level
  // hybrid entries, which are stale under deferral)
  cuda_data_partition_->SetLeafDataLayout(final_num, final_start, final_num_leaves);
  for (int i = 0; i < final_num_leaves; ++i) {
    leaf_num_data_[i] = final_num[i];
    leaf_data_start_[i] = final_start[i];
  }
  cuda_data_partition_->MaterializeHybridLeafMap(final_num_leaves);
  sel_last_peak_ = sel_num_allocated_;
  ++sel_stat_trees_;
  const bool hybrid_debug = FalcataDebug().debug;
  if (hybrid_debug) {
    // stderr (not the logger): churn statistics must be visible under verbose=-1
    fprintf(stderr, "[selective] trees=%" PRId64 " leaves=%d cumulative: applied=%" PRId64 " displaced=%" PRId64 " levels=%" PRId64 " sim_ms=%.1f\n",
            static_cast<int64_t>(sel_stat_trees_), final_num_leaves,
            static_cast<int64_t>(sel_stat_applied_),
            static_cast<int64_t>(sel_stat_displaced_),
            static_cast<int64_t>(sel_stat_levels_),
            sel_stat_sim_ns_ * 1e-6);
  }
}

void CUDASingleGPUTreeLearner::TrainSelective(CUDATree* tree) {
  global_timer.Start("CUDASingleGPUTreeLearner::TrainSelective");
  selective_active_ = true;
  SelectiveReset();
  // two struct slots per split of a level (levels hold at most (num_leaves+1)/2
  // splits: a level's splits plus their ancestor chains all fit the selection)
  const size_t max_slots = static_cast<size_t>(config_->num_leaves) + 2;
  if (hybrid_pair_slots_.Size() < max_slots) {
    hybrid_pair_slots_.Resize(max_slots);
  }
  if (UseOneSyncPrefix()) {
    TrainSelectiveOneSync(tree);
  } else {
    TrainSelectiveTwoSync(tree);
  }
  SelectiveFinalize(tree);
  selective_active_ = false;
  global_timer.Stop("CUDASingleGPUTreeLearner::TrainSelective");
}

bool CUDASingleGPUTreeLearner::TunerActive() const {
  const auto& plan = FalcataPlan::Get();
  if (!plan.tuner || !config_->use_quantized_grad) return false;
  // the probe phase costs ~60 trees per knob; under auto, skip runs too
  // short to amortize it (measured -1.9% on year @100 rounds, +2.7% @500)
  if (!plan.tuner_explicit && config_->num_iterations < 300) return false;
  return true;
}

// ---- tier-0 seed + tier-3 wisdom -----------------------------------------
// Seed: scale the floor candidates by the device's SM count relative to the
// RTX 5090 they were tuned on (the elastic bracket would find the range
// anyway; seeding just starts the probe near the right octave).
// Wisdom: persist the chosen knob values keyed by (shape, device) so
// retrains of the same workload start from the known-best point instead of
// re-probing from scratch (FFTW-wisdom style; the periodic re-probe still
// verifies the cached choice against reality).
static const char* TunerWisdomPath() {
  static std::string path = []() {
    const char* home = std::getenv("HOME");
    return std::string(home ? home : "/tmp") + "/.cache/falcata/wisdom.txt";
  }();
  return path.c_str();
}

std::string CUDASingleGPUTreeLearner::TunerWisdomKey() const {
  char buf[160];
  snprintf(buf, sizeof(buf), "v1:%d:%d:%d:%d:%d:%d",
           num_data_, train_data_->num_features(), config_->num_leaves,
           config_->max_depth, effective_quant_bins_, tuner_device_sm_count_);
  return std::string(buf);
}

void CUDASingleGPUTreeLearner::TunerInitSeeds() {
  TierOneTuner& t = tuner_;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, gpu_device_id_ < 0 ? 0 : gpu_device_id_) == cudaSuccess) {
    tuner_device_sm_count_ = prop.multiProcessorCount;
    if (prop.multiProcessorCount > 0 && prop.multiProcessorCount != 170) {
      const double scale = prop.multiProcessorCount / 170.0;
      for (int& cand : t.candidates[0]) {
        cand = std::max(TierOneTuner::kFloorMin,
                        std::min(TierOneTuner::kFloorMax,
                                 static_cast<int>(cand * scale + 0.5)));
      }
    }
  }
  // wisdom lookup: exact shape+device hit -> adopt the cached values as the
  // exploited choice and defer the first probe to the periodic re-probe
  std::ifstream in(TunerWisdomPath());
  if (!in.good()) return;
  const std::string want = TunerWisdomKey();
  std::string key; int floor_v = 0, slr_v = 0;
  while (in >> key >> floor_v >> slr_v) {
    if (key != want) continue;
    auto adopt = [](std::vector<int>* cands, const int v) {
      auto it = std::find(cands->begin(), cands->end(), v);
      if (it == cands->end()) {
        cands->push_back(v);
        it = cands->end() - 1;
      }
      return static_cast<int>(it - cands->begin());
    };
    t.chosen[0] = adopt(&t.candidates[0], floor_v);
    t.chosen[1] = adopt(&t.candidates[1], slr_v);
    t.next_probe_at = TierOneTuner::kReprobeEvery;
    t.seeded_from_wisdom = true;
    Log::Debug("[tuner] wisdom hit: floor=%d small_leaf_rows=%d (re-verify at tree %d)",
               floor_v, slr_v, t.next_probe_at);
    return;
  }
}

void CUDASingleGPUTreeLearner::TunerSaveWisdom() {
  const TierOneTuner& t = tuner_;
  if (t.chosen[0] < 0 || t.chosen[1] < 0) return;
  const std::string key = TunerWisdomKey();
  const int floor_v = t.candidates[0][t.chosen[0]];
  const int slr_v = t.candidates[1][t.chosen[1]];
  // read-modify-write with rename for crash safety; best effort (races with
  // a concurrent trainer at worst lose one update)
  std::map<std::string, std::pair<int, int>> all;
  {
    std::ifstream in(TunerWisdomPath());
    std::string k; int a = 0, b = 0;
    while (in >> k >> a >> b) all[k] = {a, b};
  }
  all[key] = {floor_v, slr_v};
  const std::string path = TunerWisdomPath();
  const std::string dir = path.substr(0, path.find_last_of('/'));
  if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
    return;  // cache dir unavailable: wisdom is best-effort
  }
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp);
    for (const auto& kv : all) {
      out << kv.first << " " << kv.second.first << " " << kv.second.second << "\n";
    }
  }
  ::rename(tmp.c_str(), path.c_str());
}

void CUDASingleGPUTreeLearner::TunerBeforeTree() {
  if (!TunerActive()) return;
  TierOneTuner& t = tuner_;
  if (t.tree_index >= t.next_probe_at && t.probe_slot < 0) {
    // start a (re-)probe cycle at knob 0
    t.knob = 0;
    t.probe_slot = 0;
    t.probe_tree = 0;
    t.cur_best = 1e30;
    t.best_of_candidate.assign(t.candidates[0].size(), 1e30);
  }
  auto& plan = FalcataPlan::Mutable();
  // knob values: the currently-probed candidate for the active knob, the
  // exploited choice (or compile-time default) for the other
  const auto value_for = [&t](const int knob) {
    if (t.probe_slot >= 0 && t.knob == knob) return t.candidates[knob][t.probe_slot];
    if (t.chosen[knob] >= 0) return t.candidates[knob][t.chosen[knob]];
    return knob == 0 ? 160 : 1024;  // pre-first-probe defaults
  };
  plan.batch_construct_saturation_floor = value_for(0);
  plan.quant_small_leaf_rows = value_for(1);
}

void CUDASingleGPUTreeLearner::TunerAfterTree(double tree_seconds) {
  if (!TunerActive()) return;
  TierOneTuner& t = tuner_;
  ++t.tree_index;
  if (t.probe_slot < 0) return;
  // best-of (not mean): robust to interference spikes on a shared GPU
  t.cur_best = std::min(t.cur_best, tree_seconds);
  if (++t.probe_tree >= TierOneTuner::kTreesPerCandidate) {
    t.best_of_candidate[t.probe_slot] = t.cur_best;
    t.cur_best = 1e30;
    t.probe_tree = 0;
    std::vector<int>& cands = t.candidates[t.knob];
    if (++t.probe_slot >= static_cast<int>(cands.size())) {
      const int best = static_cast<int>(std::min_element(t.best_of_candidate.begin(),
        t.best_of_candidate.end()) - t.best_of_candidate.begin());
      // Elastic bracket (floor knob only): if the winner sits on an edge of
      // the set, the optimum for this silicon may lie beyond it. Extend
      // geometrically (bounded) and keep probing instead of settling.
      if (t.knob == 0) {
        if (best == static_cast<int>(cands.size()) - 1 &&
            cands.back() < TierOneTuner::kFloorMax) {
          cands.push_back(cands.back() * 2);
          t.best_of_candidate.push_back(1e30);
          return;  // probe_slot now indexes the appended candidate
        }
        if (best == 0 && cands.front() > TierOneTuner::kFloorMin) {
          cands.insert(cands.begin(), cands.front() / 2);
          t.best_of_candidate.insert(t.best_of_candidate.begin(), 1e30);
          t.probe_slot = 0;  // probe the prepended candidate
          return;
        }
      }
      t.chosen[t.knob] = best;
      if (t.knob + 1 < TierOneTuner::kNumKnobs) {
        // coordinate descent: next knob probes with this knob at its best
        ++t.knob;
        t.probe_slot = 0;
        t.best_of_candidate.assign(t.candidates[t.knob].size(), 1e30);
        return;
      }
      // full cycle complete: exploit and persist
      t.probe_slot = -1;
      t.next_probe_at = t.tree_index + TierOneTuner::kReprobeEvery;
      Log::Debug("[tuner] chosen floor=%d small_leaf_rows=%d (re-probe at tree %d)",
                 t.candidates[0][t.chosen[0]], t.candidates[1][t.chosen[1]], t.next_probe_at);
      TunerSaveWisdom();
    }
  }
}

Tree* CUDASingleGPUTreeLearner::Train(const score_t* gradients,
  const score_t* hessians, bool is_first_tree) {
  TunerBeforeTree();
  const auto tuner_t0 = std::chrono::steady_clock::now();
  gradients_ = gradients;
  hessians_ = hessians;
  global_timer.Start("CUDASingleGPUTreeLearner::BeforeTrain");
  BeforeTrain();
  global_timer.Stop("CUDASingleGPUTreeLearner::BeforeTrain");
  // linear trees need per-leaf branch features to know which features each leaf's
  // linear model uses (mirrors the CPU LinearTreeLearner using new Tree(.., true, true)).
  const bool track_branch_features = !(config_->interaction_constraints_vector.empty()) || config_->linear_tree;
  // reusable slab for the per-tree device arrays: one allocation for the whole
  // training run instead of ~17 cudaMallocs + ~16 cudaFrees per tree (trees are
  // trained one at a time and release the slab in ToHost()); lazily (re)sized so
  // ResetConfig num_leaves changes are honored
  const size_t tree_pool_bytes = CUDATree::PooledDeviceBufferSize(config_->num_leaves);
  if (cuda_tree_pool_buffer_.Size() < tree_pool_bytes) {
    cuda_tree_pool_buffer_.Resize(tree_pool_bytes);
  }
  std::unique_ptr<CUDATree> tree(new CUDATree(config_->num_leaves, track_branch_features,
    config_->linear_tree, gpu_device_id_, has_categorical_feature_,
    cuda_tree_pool_buffer_.RawData(), std::max(1, vec_num_targets_)));
  // set the root value by hand, as it is not handled by splits. When the root
  // sums were deferred (single-sync hybrid growth), this happens lazily in
  // EnsureRootSumsReadBack -- the root value only survives when the tree ends
  // up a stump (any split overwrites leaf 0's output on device).
  if (vec_num_targets_ > 1) {
    for (int t = 0; t < vec_num_targets_; ++t) {
      tree->SetLeafOutput(0, t, CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(
        vec_root_sum_gradients_[t], vec_root_sum_hessians_[t],
        config_->lambda_l1, config_->lambda_l2, config_->path_smooth, config_->max_delta_step,
        static_cast<data_size_t>(num_data_), 0.0));
    }
    tree->SyncLeafOutputFromHostToCUDA();
  } else if (!root_sums_deferred_) {
    tree->SetLeafOutput(0, CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(
      leaf_sum_gradients_[smaller_leaf_index_], leaf_sum_hessians_[smaller_leaf_index_],
      config_->lambda_l1, config_->lambda_l2,  config_->path_smooth, config_->max_delta_step,
      static_cast<data_size_t>(num_data_), 0.0));
    tree->SyncLeafOutputFromHostToCUDA();
  }
  int num_splits_done = 0;
  ForceSplitsCUDA(tree.get(), &num_splits_done);
  bool pair_search_cached = false;
  bool selective_handled = false;
  bool batched_apply_ran = false;
  if (fp_merge_state_ != nullptr &&
      (num_splits_done != 0 || !HybridGrowthUsable() || has_categorical_feature_)) {
    Log::Fatal(
        "feature-parallel multi-GPU (tree_learner=feature) currently requires "
        "the hybrid level flow: a depth-limited config "
        "(2^max_depth <= num_leaves + 1), no categorical features and no "
        "forced splits. Use tree_learner=serial (data-parallel) or num_gpu=1 "
        "for this configuration.");
  }
  if (nccl_communicator_ != nullptr && config_->use_quantized_grad &&
      (num_splits_done != 0 || !HybridGrowthUsable())) {
    // quantized multi-GPU is only wired through the hybrid level flow (integer
    // level all-reduce + global bit-width table); the classic per-split loop's
    // quantized reduce is unverified and historically hung. Refuse clearly.
    Log::Fatal(
        "multi-GPU + quantized gradients requires the hybrid level flow, "
        "which is unavailable for this configuration (needs a depth-limited "
        "config: 2^max_depth <= num_leaves + 1, no forced splits). "
        "Set quant_mode=none, adjust max_depth/num_leaves, or use num_gpu=1.");
  }
  if (num_splits_done == 0 && HybridGrowthUsable()) {
    if (UseSelectiveGrowth()) {
      // budget-limited exact grow-then-prune: builds the COMPLETE tree
      // (including the final pruned structure); no leaf-wise tail runs
      TrainSelective(tree.get());
      selective_handled = true;
    } else {
      global_timer.Start("CUDASingleGPUTreeLearner::TrainLevelWisePrefix");
      num_splits_done = TrainLevelWisePrefix(tree.get());
      global_timer.Stop("CUDASingleGPUTreeLearner::TrainLevelWisePrefix");
      batched_apply_ran = num_splits_done > 0;
      // every current leaf already has a cached best-split candidate; the first
      // tail iteration must not redo the pair search
      pair_search_cached = num_splits_done > 0;
    }
  }
  for (int i = num_splits_done; !selective_handled && i < config_->num_leaves - 1; ++i) {
    if (!pair_search_cached) {
      if (fp_merge_state_ != nullptr) {
        // The leaf-wise tail's fresh per-pair searches are not winner-merged
        // across ranks. The prefix covers the whole tree for depth-limited
        // configs unless the leaf budget binds mid-level; refuse rather than
        // silently searching only this rank's feature stripe.
        Log::Fatal(
            "feature-parallel multi-GPU reached the leaf-wise tail (the leaf "
            "budget bound before max_depth). Raise num_leaves to at least "
            "2^max_depth - 1 or use tree_learner=serial.");
      }
      if (vec_num_targets_ > 1) {
        EnqueuePairBestSplitSearchVector(tree.get(), smaller_leaf_index_, larger_leaf_index_);
      } else {
        EnqueuePairBestSplitSearch(tree.get(),
          cuda_smaller_leaf_splits_->GetCUDAStruct(),
          cuda_larger_leaf_splits_->GetCUDAStruct(),
          smaller_leaf_index_, larger_leaf_index_, /*synchronize=*/true);
      }
    }
    pair_search_cached = false;
    global_timer.Start("CUDASingleGPUTreeLearner::FindBestFromAllSplits");
    const CUDASplitInfo* best_split_info = nullptr;
    if (larger_leaf_index_ >= 0) {
      best_split_info = cuda_best_split_finder_->FindBestFromAllSplits(
        tree->num_leaves(),
        smaller_leaf_index_,
        larger_leaf_index_,
        &leaf_best_split_feature_[smaller_leaf_index_],
        &leaf_best_split_threshold_[smaller_leaf_index_],
        &leaf_best_split_default_left_[smaller_leaf_index_],
        &leaf_best_split_feature_[larger_leaf_index_],
        &leaf_best_split_threshold_[larger_leaf_index_],
        &leaf_best_split_default_left_[larger_leaf_index_],
        &best_leaf_index_,
        &num_cat_threshold_);
    } else {
      best_split_info = cuda_best_split_finder_->FindBestFromAllSplits(
        tree->num_leaves(),
        smaller_leaf_index_,
        larger_leaf_index_,
        &leaf_best_split_feature_[smaller_leaf_index_],
        &leaf_best_split_threshold_[smaller_leaf_index_],
        &leaf_best_split_default_left_[smaller_leaf_index_],
        nullptr,
        nullptr,
        nullptr,
        &best_leaf_index_,
        &num_cat_threshold_);
    }
    global_timer.Stop("CUDASingleGPUTreeLearner::FindBestFromAllSplits");

    if (best_leaf_index_ == -1) {
      if (std::getenv("FALCATA_DEBUG_SPLITS") != nullptr) {
        CUDALeafSplitsStruct sdbg;
        CopyFromCUDADeviceToHost<CUDALeafSplitsStruct>(&sdbg, cuda_smaller_leaf_splits_->GetCUDAStruct(), 1, __FILE__, __LINE__);
        double hist_head[16];
        CopyFromCUDADeviceToHost<double>(hist_head, cuda_histogram_constructor_->cuda_hist_pointer(), 16, __FILE__, __LINE__);
        fprintf(stderr, "CUDANOSPLIT iter=%d leaf=%d n=%d sg=%.17g sh=%.17g gain=%.17g hist0..7=", i,
                sdbg.leaf_index, sdbg.num_data_in_leaf, sdbg.sum_of_gradients, sdbg.sum_of_hessians, sdbg.gain);
        for (int hh = 0; hh < 16; ++hh) fprintf(stderr, "%.6g ", hist_head[hh]);
        fprintf(stderr, "\n");
      }
      Log::Warning("No further splits with positive gain, training stopped with %d leaves.", (i + 1));
      break;
    }
    if (std::getenv("FALCATA_DEBUG_SPLITS") != nullptr) {
      CUDASplitInfo dbg;
      CopyFromCUDADeviceToHost<CUDASplitInfo>(&dbg, best_split_info, 1, __FILE__, __LINE__);
      fprintf(stderr, "CUDASPLIT leaf=%d feat=%d thr=%u gain=%.17g slg=%.17g slh=%.17g lc=%d rc=%d\n",
              best_leaf_index_, dbg.inner_feature_index, dbg.threshold, dbg.gain,
              dbg.left_sum_gradients, dbg.left_sum_hessians, dbg.left_count, dbg.right_count);
    }

    global_timer.Start("CUDASingleGPUTreeLearner::Split");
    if (num_cat_threshold_ > 0) {
      ConstructBitsetForCategoricalSplit(best_split_info);
    }

    const int right_leaf_index = ApplySplit(tree.get(), best_split_info, best_leaf_index_);

    if (use_monotone_constraints_) {
      const int split_inner_feature = leaf_best_split_feature_[best_leaf_index_];
      const bool is_numerical_split =
        train_data_->FeatureBinMapper(split_inner_feature)->bin_type() == BinType::NumericalBin;
      double host_left_value = 0.0;
      double host_right_value = 0.0;
      CopyFromCUDADeviceToHost<double>(&host_left_value, &best_split_info->left_value, 1, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<double>(&host_right_value, &best_split_info->right_value, 1, __FILE__, __LINE__);
      UpdateLeafConstraints(best_leaf_index_, right_leaf_index, split_inner_feature,
                            host_left_value, host_right_value, is_numerical_split);
    }

    if (nccl_communicator_ != nullptr) {
      smaller_leaf_index_ = (global_num_data_in_leaf_[best_leaf_index_] < global_num_data_in_leaf_[right_leaf_index] ? best_leaf_index_ : right_leaf_index);
      larger_leaf_index_ = (smaller_leaf_index_ == best_leaf_index_ ? right_leaf_index : best_leaf_index_);
      const int best_leaf_hist_index = leaf_to_hist_index_map_[best_leaf_index_];
      leaf_to_hist_index_map_[smaller_leaf_index_] = right_leaf_index;
      leaf_to_hist_index_map_[larger_leaf_index_] = best_leaf_hist_index;
    } else {
      smaller_leaf_index_ = (leaf_num_data_[best_leaf_index_] < leaf_num_data_[right_leaf_index] ? best_leaf_index_ : right_leaf_index);
      larger_leaf_index_ = (smaller_leaf_index_ == best_leaf_index_ ? right_leaf_index : best_leaf_index_);
    }

    if (config_->use_quantized_grad) {
      cuda_gradient_discretizer_->SetNumBitsInHistogramBin<false>(
        best_leaf_index_, right_leaf_index, leaf_num_data_[best_leaf_index_], leaf_num_data_[right_leaf_index]);
      if (nccl_communicator_ != nullptr) {
        cuda_gradient_discretizer_->SetNumBitsInHistogramBin<true>(
          best_leaf_index_, right_leaf_index, global_num_data_in_leaf_[best_leaf_index_], global_num_data_in_leaf_[right_leaf_index]);
      }
    }
    global_timer.Stop("CUDASingleGPUTreeLearner::Split");
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  if (batched_apply_ran) {
    // the batched apply defers the per-level row -> leaf map scatter; write the
    // map once from the final leaf windows before any tree-end consumer
    // (AddPredictionToScore / refit stats / linear trees) reads it. The classic
    // tail splits above kept their inline updates, which this pass subsumes.
    // When the final level's split-inner already wrote the map inline
    // (final_partial_level), only earlier-finalized leaves remain.
    if (hybrid_map_final_written_) {
      cuda_data_partition_->MaterializeHybridLeafMapSubset(hybrid_map_residual_leaves_);
    } else {
      cuda_data_partition_->MaterializeHybridLeafMap(tree->num_leaves());
    }
  }
  hybrid_map_final_written_ = false;
  if (config_->use_quantized_grad && config_->quant_train_renew_leaf &&
      nccl_communicator_ != nullptr) {
    static bool warned_renew = false;
    if (!warned_renew) {
      warned_renew = true;
      Log::Warning("quant_train_renew_leaf is ignored under multi-GPU: the "
                   "renewed outputs would use rank-LOCAL gradient sums and "
                   "diverge across ranks.");
    }
  } else if (config_->use_quantized_grad && config_->quant_train_renew_leaf) {
    global_timer.Start("CUDASingleGPUTreeLearner::RenewDiscretizedTreeLeaves");
    RenewDiscretizedTreeLeaves(tree.get());
    if (selective_handled) {
      // the selective path already finalized the host tree (no ToHost below):
      // pull the renewed device leaf values back explicitly
      tree->SyncLeafOutputFromCUDAToHost();
    }
    global_timer.Stop("CUDASingleGPUTreeLearner::RenewDiscretizedTreeLeaves");
  }
  if (!selective_handled) {
    // the selective path rebuilt the host tree from captured split info and
    // released the device arrays in RebuildFromHostSplits already
    tree->ToHost();
  }
  // The counts the split finder recorded are estimates: a histogram bin holds
  // gradient and hessian but no row count, so it infers one from the hessian
  // mass. That is exact only when every hessian is 1 (l2); under binary or
  // poisson a node's recorded count disagreed with its own children's by a few
  // rows. The data partition knows the exact per-leaf counts -- take them.
  //
  // This is what the CPU learner does too: it keeps the estimate for the
  // min_data_in_leaf constraint (both devices do, by upstream design) and
  // overwrites it with data_partition_->leaf_count() before recording the
  // split, in serial_tree_learner.cpp. Same semantics, one tree later.
  //
  // Rank-local under multi-GPU, where the partition holds only this rank's
  // rows, so the estimate stays in force there.
  if (nccl_communicator_ == nullptr && tree->num_leaves() > 0) {
    std::vector<data_size_t> leaf_num_data(static_cast<size_t>(tree->num_leaves()));
    CopyFromCUDADeviceToHost<data_size_t>(leaf_num_data.data(),
      cuda_data_partition_->cuda_leaf_num_data(), leaf_num_data.size(), __FILE__, __LINE__);
    tree->SyncNodeCountsFromPartition(leaf_num_data);
  }
  // only the leaf histogram slots this tree used can be dirty; the next
  // BeforeTrain zeroes just that prefix (single-GPU only: the NCCL path keeps
  // the conservative full zeroing). Selective growth dirties every hybrid slot
  // it ever allocated (recycled slots are re-zeroed on the fly, but the last
  // occupants remain), so it reports its allocation peak instead.
  if (nccl_communicator_ == nullptr) {
    cuda_histogram_constructor_->SetNumDirtyLeaves(
      selective_handled ? sel_last_peak_ : tree->num_leaves());
  }
  last_tree_is_linear_ = false;
  if (config_->linear_tree) {
    // gradients_/hessians_ are device pointers after BeforeTrain (the Train args stay on host)
    CalculateLinearCUDA(tree.get(), gradients_, hessians_, is_first_tree);
    last_tree_is_linear_ = true;
  }
  TunerAfterTree(std::chrono::duration<double>(std::chrono::steady_clock::now() - tuner_t0).count());
  return tree.release();
}

int CUDASingleGPUTreeLearner::ApplySplit(CUDATree* tree, const CUDASplitInfo* best_split_info, const int leaf_index,
                                         CUDALeafSplitsStruct* smaller_slot, CUDALeafSplitsStruct* larger_slot,
                                         int deferred_slot) {
  // classic per-split path: needs the plain per-column view (only the batched
  // apply kernels understand the packed compact source)
  EnsureClassicColumnView();
  // CEGB charges cegb_penalty_feature_coupled the first time a feature is used in
  // the tree, so every path that commits a split has to mark it. No-op unless
  // CEGB is configured.
  cuda_best_split_finder_->MarkFeatureUsedInSplit(leaf_best_split_feature_[leaf_index]);
  int right_leaf_index = 0;
  if (train_data_->FeatureBinMapper(leaf_best_split_feature_[leaf_index])->bin_type() == BinType::CategoricalBin) {
    right_leaf_index = tree->SplitCategorical(leaf_index,
                                     train_data_->RealFeatureIndex(leaf_best_split_feature_[leaf_index]),
                                     train_data_->FeatureBinMapper(leaf_best_split_feature_[leaf_index])->missing_type(),
                                     best_split_info,
                                     cuda_bitset_,
                                     cuda_bitset_len_,
                                     cuda_bitset_inner_,
                                     cuda_bitset_inner_len_);
    if (vec_num_targets_ > 1) {
      // same as the numerical branch: the split kernel wrote the target-0
      // (mirror) outputs; record every target's child outputs from the
      // winning split's vector payload
      tree->SetVectorLeafValuesFromSplit(leaf_index, right_leaf_index, best_split_info);
    }
  } else {
    right_leaf_index = tree->Split(leaf_index,
                                     train_data_->RealFeatureIndex(leaf_best_split_feature_[leaf_index]),
                                     train_data_->RealThreshold(leaf_best_split_feature_[leaf_index],
                                      leaf_best_split_threshold_[leaf_index]),
                                     train_data_->FeatureBinMapper(leaf_best_split_feature_[leaf_index])->missing_type(),
                                     best_split_info);
    if (vec_num_targets_ > 1) {
      // the split kernel wrote the target-0 (mirror) outputs; record every
      // target's child outputs from the winning split's vector payload
      tree->SetVectorLeafValuesFromSplit(leaf_index, right_leaf_index, best_split_info);
    }
  }

  // The right leaf is a freshly allocated hist slot that hasn't been zeroed.
  // Zero it now so the next iter's ConstructHistogramForLeaf can atomicAdd into it
  // correctly. (left leaf keeps the parent slot, which already holds the parent hist;
  // SubtractHistogramKernel then computes left = parent - smaller.)
  cuda_histogram_constructor_->ZeroHistForLeaf(right_leaf_index);
  cuda_data_partition_->Split(best_split_info,
                              leaf_index,
                              right_leaf_index,
                              leaf_best_split_feature_[leaf_index],
                              leaf_best_split_threshold_[leaf_index],
                              cuda_bitset_inner_,
                              static_cast<int>(cuda_bitset_inner_len_),
                              leaf_best_split_default_left_[leaf_index],
                              leaf_num_data_[leaf_index],
                              leaf_data_start_[leaf_index],
                              smaller_slot != nullptr ? smaller_slot : cuda_smaller_leaf_splits_->GetCUDAStructRef(),
                              larger_slot != nullptr ? larger_slot : cuda_larger_leaf_splits_->GetCUDAStructRef(),
                              &leaf_num_data_[leaf_index],
                              &leaf_num_data_[right_leaf_index],
                              &leaf_data_start_[leaf_index],
                              &leaf_data_start_[right_leaf_index],
                              &leaf_sum_hessians_[leaf_index],
                              &leaf_sum_hessians_[right_leaf_index],
                              &leaf_sum_gradients_[leaf_index],
                              &leaf_sum_gradients_[right_leaf_index],
                              global_num_data_in_leaf_.data() + leaf_index,
                              global_num_data_in_leaf_.data() + right_leaf_index,
                              /*point_structs_at_main=*/smaller_slot != nullptr,
                              deferred_slot);
  if (vec_num_targets_ > 1) {
    // the partition seeded the primary smaller/larger structs (target-0 sums,
    // plane-0 histogram pointers); fan them out into the per-plane structs
    // with per-target sums/values from the split's vector payload
    LaunchVectorPlaneFanOutKernel(best_split_info, leaf_index);
  }
  if (config_->use_quantized_grad && nccl_communicator_ != nullptr) {
    // The GLOBAL per-leaf histogram bit widths were only ever set for the ROOT
    // (BeforeTrain). Every other leaf kept 0, so GetHistBitsInLeaf<true>()
    // returned 0 and NCCLReduceHistogram took its "<= 16 bits" branch and
    // all-reduced num_total_bin INT32s over a histogram that is actually
    // 32-bit-per-stat -- reducing the wrong element type and half the bytes.
    // The result was corrupt histograms, divergent gains and a hang. The CPU
    // data-parallel learner does this update per split
    // (data_parallel_tree_learner.cpp); the CUDA one simply never did.
    cuda_gradient_discretizer_->SetNumBitsInHistogramBin<true>(
        leaf_index, right_leaf_index,
        global_num_data_in_leaf_[leaf_index],
        global_num_data_in_leaf_[right_leaf_index]);
  }
  #ifdef DEBUG
  CheckSplitValid(leaf_index, right_leaf_index);
  #endif  // DEBUG
  return right_leaf_index;
}

int CUDASingleGPUTreeLearner::ForceSplitsCUDA(CUDATree* tree, int* num_splits_done) {
  *num_splits_done = 0;
  if (forced_split_json_ == nullptr || forced_split_json_->is_null()) {
    return 0;
  }
  if (config_->use_quantized_grad || nccl_communicator_ != nullptr) {
    Log::Warning("Forced splits on CUDA are not supported with quantized gradients or multi-GPU. "
                 "Ignoring forcedsplits_filename for this tree.");
    return 0;
  }
  // BFS over the forced-split JSON, mirroring SerialTreeLearner::ForceSplits.
  // Each iteration: construct histograms + run the regular best-split search for the
  // active (smaller, larger) pair (so every forced leaf gets a valid cached best split
  // for the main loop), sync the host-side best-split arrays with the device cache,
  // compute forced split infos for pending JSON nodes whose target leaves are active,
  // then pop one entry and apply its (already computed) forced split.
  int result_count = 0;
  std::queue<std::pair<Json, int>> q;
  q.push(std::make_pair(*forced_split_json_, 0));
  struct ForcedSplitEntry {
    const CUDASplitInfo* device_info;
    CUDASplitInfo host_info;
    int inner_feature_index;
    uint32_t threshold_bin;
  };
  std::unordered_map<int, ForcedSplitEntry> force_split_map;
  while (!q.empty()) {
    const data_size_t num_data_in_smaller_leaf = leaf_num_data_[smaller_leaf_index_];
    const data_size_t num_data_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_num_data_[larger_leaf_index_];
    const double sum_hessians_in_smaller_leaf = leaf_sum_hessians_[smaller_leaf_index_];
    const double sum_hessians_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_sum_hessians_[larger_leaf_index_];
    cuda_histogram_constructor_->ConstructHistogramForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      num_data_in_smaller_leaf,
      num_data_in_larger_leaf,
      num_data_in_smaller_leaf,
      num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf,
      sum_hessians_in_larger_leaf,
      0);
    cuda_histogram_constructor_->SubtractHistogramForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      false, 0, 0, 0);
    // regular best-split search for the active pair: populates the device per-leaf cache
    SelectFeatureByNode(tree);
    // Same monotone constraints the main search applies; without them the splits
    // cached here (and later picked by FindBestFromAllSplits) could violate them.
    const double forced_smaller_constraint_min = use_monotone_constraints_ ?
      leaf_constraint_min_[smaller_leaf_index_] : -std::numeric_limits<double>::max();
    const double forced_smaller_constraint_max = use_monotone_constraints_ ?
      leaf_constraint_max_[smaller_leaf_index_] : std::numeric_limits<double>::max();
    const double forced_larger_constraint_min = (use_monotone_constraints_ && larger_leaf_index_ >= 0) ?
      leaf_constraint_min_[larger_leaf_index_] : -std::numeric_limits<double>::max();
    const double forced_larger_constraint_max = (use_monotone_constraints_ && larger_leaf_index_ >= 0) ?
      leaf_constraint_max_[larger_leaf_index_] : std::numeric_limits<double>::max();
    cuda_best_split_finder_->FindBestSplitsForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      smaller_leaf_index_, larger_leaf_index_,
      num_data_in_smaller_leaf, num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf,
      nullptr, nullptr, 0, 0,
      config_->max_depth <= 0 || tree->leaf_depth(smaller_leaf_index_) < config_->max_depth,
      larger_leaf_index_ < 0 || config_->max_depth <= 0 ||
        tree->leaf_depth(larger_leaf_index_) < config_->max_depth,
      forced_smaller_constraint_min, forced_smaller_constraint_max,
      forced_larger_constraint_min, forced_larger_constraint_max);
    // sync host-side best-split arrays with the device cache for the searched pair, so
    // the main loop's FindBestFromAllSplits can pick these leaves later with consistent
    // host (feature, threshold) and device (gain, sums) information
    cuda_best_split_finder_->SyncLeafBestSplitToHost(
      smaller_leaf_index_, larger_leaf_index_,
      &leaf_best_split_feature_[smaller_leaf_index_],
      &leaf_best_split_threshold_[smaller_leaf_index_],
      &leaf_best_split_default_left_[smaller_leaf_index_],
      larger_leaf_index_ >= 0 ? &leaf_best_split_feature_[larger_leaf_index_] : nullptr,
      larger_leaf_index_ >= 0 ? &leaf_best_split_threshold_[larger_leaf_index_] : nullptr,
      larger_leaf_index_ >= 0 ? &leaf_best_split_default_left_[larger_leaf_index_] : nullptr);

    // compute forced split info for every pending JSON node whose target leaf is active
    auto compute_for = [&](const Json& node_json, const int leaf_index) {
      if (node_json.is_null() ||
          node_json.object_items().count("feature") == 0 ||
          node_json.object_items().count("threshold") == 0) {
        return;
      }
      const int real_feature_index = node_json["feature"].int_value();
      const double threshold_double = node_json["threshold"].number_value();
      const int inner_feature_index = train_data_->InnerFeatureIndex(real_feature_index);
      if (inner_feature_index < 0) {
        Log::Warning("Forced split feature %d not in training data; skipping.", real_feature_index);
        return;
      }
      if (train_data_->FeatureBinMapper(inner_feature_index)->bin_type() == BinType::CategoricalBin) {
        Log::Warning("Forced splits on categorical features are not supported on CUDA; skipping feature %d.",
                     real_feature_index);
        return;
      }
      const uint32_t threshold_bin = train_data_->BinThreshold(inner_feature_index, threshold_double);
      const bool is_smaller = (leaf_index == smaller_leaf_index_);
      const CUDALeafSplitsStruct* leaf_struct = is_smaller ?
        cuda_smaller_leaf_splits_->GetCUDAStruct() : cuda_larger_leaf_splits_->GetCUDAStruct();
      ForcedSplitEntry entry;
      entry.inner_feature_index = inner_feature_index;
      entry.threshold_bin = threshold_bin;
      entry.device_info = cuda_best_split_finder_->ComputeForcedSplit(
        leaf_struct, inner_feature_index, threshold_bin, leaf_num_data_[leaf_index],
        leaf_index, &entry.host_info);
      if (entry.device_info == nullptr || !entry.host_info.is_valid) {
        Log::Warning("'Forced Split' will be ignored since the gain getting worse.");
        return;
      }
      force_split_map[leaf_index] = entry;
    };
    {
      std::queue<std::pair<Json, int>> q_copy = q;
      while (!q_copy.empty()) {
        const std::pair<Json, int>& pending = q_copy.front();
        const int leaf_index = pending.second;
        if ((leaf_index == smaller_leaf_index_ || leaf_index == larger_leaf_index_ ||
             (larger_leaf_index_ < 0 && leaf_index == 0)) &&
            force_split_map.find(leaf_index) == force_split_map.end()) {
          compute_for(pending.first, leaf_index);
        }
        q_copy.pop();
      }
    }

    std::pair<Json, int> pair = q.front();
    q.pop();
    const int current_leaf = pair.second;
    auto it = force_split_map.find(current_leaf);
    if (it == force_split_map.end()) {
      // forced split for this leaf was invalid / unsupported: stop forcing (mirrors CPU abort)
      break;
    }
    const ForcedSplitEntry& entry = it->second;
    leaf_best_split_feature_[current_leaf] = entry.inner_feature_index;
    leaf_best_split_threshold_[current_leaf] = entry.threshold_bin;
    leaf_best_split_default_left_[current_leaf] = 1;
    num_cat_threshold_ = 0;
    const int right_leaf_index = ApplySplit(tree, entry.device_info, current_leaf);
    force_split_map.erase(current_leaf);
    // both children's cached entries (host + device) describe the pre-split leaf:
    // invalidate the device entries; the host arrays will be overwritten when the
    // children are next searched and synced.
    cuda_best_split_finder_->InvalidateLeafBestSplit(current_leaf, right_leaf_index);
    smaller_leaf_index_ = (leaf_num_data_[current_leaf] < leaf_num_data_[right_leaf_index] ?
      current_leaf : right_leaf_index);
    larger_leaf_index_ = (smaller_leaf_index_ == current_leaf ? right_leaf_index : current_leaf);
    ++result_count;
    if (pair.first.object_items().count("left") > 0) {
      const Json& left = pair.first["left"];
      if (left.object_items().count("feature") > 0 && left.object_items().count("threshold") > 0) {
        q.push(std::make_pair(left, current_leaf));
      }
    }
    if (pair.first.object_items().count("right") > 0) {
      const Json& right = pair.first["right"];
      if (right.object_items().count("feature") > 0 && right.object_items().count("threshold") > 0) {
        q.push(std::make_pair(right, right_leaf_index));
      }
    }
  }
  *num_splits_done = result_count;
  return result_count;
}

void CUDASingleGPUTreeLearner::UpdateLeafConstraints(
  const int left_leaf, const int right_leaf, const int inner_feature_index,
  const double left_value, const double right_value, const bool is_numerical_split) {
  // Mirror of BasicLeafConstraints::Update (monotone_constraints.hpp). The new leaf
  // inherits the parent's [min,max], then, for a numerical split on a monotone
  // feature, the mid-point of the (clamped) child outputs becomes a min/max bound
  // for the two children.
  leaf_constraint_min_[right_leaf] = leaf_constraint_min_[left_leaf];
  leaf_constraint_max_[right_leaf] = leaf_constraint_max_[left_leaf];
  if (!is_numerical_split) {
    return;
  }
  const int8_t monotone_type = monotone_constraints_[inner_feature_index];
  if (monotone_type == 0) {
    return;
  }
  const double mid = (left_value + right_value) / 2.0;
  if (monotone_type < 0) {
    // decreasing: left child (parent leaf) gets a min bound, right child a max bound
    leaf_constraint_min_[left_leaf] = std::max(mid, leaf_constraint_min_[left_leaf]);
    leaf_constraint_max_[right_leaf] = std::min(mid, leaf_constraint_max_[right_leaf]);
  } else {
    // increasing: left child gets a max bound, right child a min bound
    leaf_constraint_max_[left_leaf] = std::min(mid, leaf_constraint_max_[left_leaf]);
    leaf_constraint_min_[right_leaf] = std::max(mid, leaf_constraint_min_[right_leaf]);
  }
}

void CUDASingleGPUTreeLearner::ResetTrainingData(
  const Dataset* train_data,
  bool is_constant_hessian) {
  SerialTreeLearner::ResetTrainingData(train_data, is_constant_hessian);
  CHECK_EQ(num_features_, train_data_->num_features());
  cuda_histogram_constructor_->ResetTrainingData(train_data, share_state_.get());
  cuda_data_partition_->ResetTrainingData(train_data,
    static_cast<int>(share_state_->feature_hist_offsets().back()) * std::max(1, vec_num_targets_),
    cuda_histogram_constructor_->cuda_hist_pointer());
  cuda_best_split_finder_->ResetTrainingData(
    cuda_histogram_constructor_->cuda_hist(),
    train_data,
    share_state_->feature_hist_offsets());
  SyncHistFP32();
  cuda_smaller_leaf_splits_->Resize(num_data_);
  cuda_larger_leaf_splits_->Resize(num_data_);
  CHECK_EQ(is_constant_hessian, share_state_->is_constant_hessian);
  if (!boosting_on_cuda_) {
    const size_t grad_planes = static_cast<size_t>(std::max(1, vec_num_targets_));
    cuda_gradients_.Resize(static_cast<size_t>(num_data_) * grad_planes);
    cuda_hessians_.Resize(static_cast<size_t>(num_data_) * grad_planes);
  }
  // training data changed: the predrawn column sample and any prefilled
  // compact view belong to the old data / re-seeded sampler
  next_tree_col_sample_ready_ = false;
  cuda_histogram_constructor_->InvalidateCompactPrefill();
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  ReleaseHybridGraphs();  // captured buffer pointers may have changed
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
}

void CUDASingleGPUTreeLearner::ResetConfig(const Config* config) {
  const int old_num_leaves = config_->num_leaves;
  SerialTreeLearner::ResetConfig(config);
  if (config_->gpu_device_id >= 0 && config_->gpu_device_id != gpu_device_id_) {
    Log::Fatal("Changing gpu device ID by resetting configuration parameter is not allowed for CUDA tree learner.");
  }
  num_threads_ = OMP_NUM_THREADS();
  if (config_->num_leaves != old_num_leaves) {
    leaf_best_split_feature_.resize(config_->num_leaves, -1);
    leaf_best_split_threshold_.resize(config_->num_leaves, 0);
    leaf_best_split_default_left_.resize(config_->num_leaves, 0);
    leaf_num_data_.resize(config_->num_leaves, 0);
    leaf_data_start_.resize(config_->num_leaves, 0);
    leaf_sum_gradients_.resize(config_->num_leaves, 0.0f);
    leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);
  }
  cuda_histogram_constructor_->ResetConfig(config);
  cuda_best_split_finder_->ResetConfig(config, cuda_histogram_constructor_->cuda_hist());
  cuda_data_partition_->ResetConfig(config, cuda_histogram_constructor_->cuda_hist_pointer());
  SyncHistFP32();
  // config changed: SerialTreeLearner::ResetConfig re-seeded the column
  // sampler, so the predrawn sample / prefilled view are stale
  next_tree_col_sample_ready_ = false;
  cuda_histogram_constructor_->InvalidateCompactPrefill();
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  ReleaseHybridGraphs();  // captured buffer pointers / budgets may have changed
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
}

void CUDASingleGPUTreeLearner::SetBaggingData(const Dataset* /*subset*/,
  const data_size_t* used_indices, data_size_t num_data) {
  cuda_data_partition_->SetUsedDataIndices(used_indices, num_data);
}

void CUDASingleGPUTreeLearner::RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                                               data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt, const double* train_score) const {
  CHECK(tree->is_cuda_tree());
  CUDATree* cuda_tree = reinterpret_cast<CUDATree*>(tree);
  if (obj != nullptr && obj->IsRenewTreeOutput()) {
    CHECK_LE(cuda_tree->num_leaves(), data_partition_->num_leaves());
    if (boosting_on_cuda_) {
      obj->RenewTreeOutputCUDA(train_score, cuda_data_partition_->cuda_data_indices(),
                               cuda_data_partition_->cuda_leaf_num_data(), cuda_data_partition_->cuda_leaf_data_start(),
                               cuda_tree->num_leaves(), cuda_tree->cuda_leaf_value_ref());
      cuda_tree->SyncLeafOutputFromCUDAToHost();
    } else {
      const data_size_t* bag_mapper = nullptr;
      if (total_num_data != num_data_) {
        CHECK_EQ(bag_cnt, num_data_);
        bag_mapper = bag_indices;
      }
      std::vector<int> n_nozeroworker_perleaf(cuda_tree->num_leaves(), 1);
      int num_machines = Network::num_machines();
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (int i = 0; i < cuda_tree->num_leaves(); ++i) {
        const double output = static_cast<double>(cuda_tree->LeafOutput(i));
        data_size_t cnt_leaf_data = leaf_num_data_[i];
        std::vector<data_size_t> index_mapper(cnt_leaf_data, -1);
        CopyFromCUDADeviceToHost<data_size_t>(index_mapper.data(),
          cuda_data_partition_->cuda_data_indices() + leaf_data_start_[i],
          static_cast<size_t>(cnt_leaf_data), __FILE__, __LINE__);
        if (cnt_leaf_data > 0) {
          const double new_output = obj->RenewTreeOutput(output, residual_getter, index_mapper.data(), bag_mapper, cnt_leaf_data);
          cuda_tree->SetLeafOutput(i, new_output);
        } else {
          CHECK_GT(num_machines, 1);
          cuda_tree->SetLeafOutput(i, 0.0);
          n_nozeroworker_perleaf[i] = 0;
        }
      }
      if (num_machines > 1) {
        std::vector<double> outputs(cuda_tree->num_leaves());
        for (int i = 0; i < cuda_tree->num_leaves(); ++i) {
          outputs[i] = static_cast<double>(cuda_tree->LeafOutput(i));
        }
        outputs = Network::GlobalSum(&outputs);
        n_nozeroworker_perleaf = Network::GlobalSum(&n_nozeroworker_perleaf);
        for (int i = 0; i < cuda_tree->num_leaves(); ++i) {
          cuda_tree->SetLeafOutput(i, outputs[i] / n_nozeroworker_perleaf[i]);
        }
      }
    }
    cuda_tree->SyncLeafOutputFromHostToCUDA();
  }
}

Tree* CUDASingleGPUTreeLearner::FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const {
  std::unique_ptr<CUDATree> cuda_tree(new CUDATree(old_tree));
  cuda_leaf_gradient_stat_buffer_.SetValue(0);
  cuda_leaf_hessian_stat_buffer_.SetValue(0);
  ReduceLeafStat(cuda_tree.get(), gradients, hessians, cuda_data_partition_->cuda_data_indices());
  cuda_tree->SyncLeafOutputFromCUDAToHost();
  return cuda_tree.release();
}

Tree* CUDASingleGPUTreeLearner::FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                                                  const score_t* gradients, const score_t* hessians) const {
  cuda_data_partition_->ResetByLeafPred(leaf_pred, old_tree->num_leaves());
  refit_num_data_ = static_cast<data_size_t>(leaf_pred.size());
  data_size_t buffer_size = static_cast<data_size_t>(old_tree->num_leaves());
  if (old_tree->num_leaves() > 2048) {
    const int num_block = (refit_num_data_ + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
    buffer_size *= static_cast<data_size_t>(num_block + 1);
  }
  if (static_cast<size_t>(buffer_size) > cuda_leaf_gradient_stat_buffer_.Size()) {
    cuda_leaf_gradient_stat_buffer_.Resize(buffer_size);
    cuda_leaf_hessian_stat_buffer_.Resize(buffer_size);
  }
  return FitByExistingTree(old_tree, gradients, hessians);
}

void CUDASingleGPUTreeLearner::ReduceLeafStat(
  CUDATree* old_tree, const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf) const {
  LaunchReduceLeafStatKernel(gradients, hessians, num_data_in_leaf, old_tree->cuda_leaf_parent(),
    old_tree->cuda_left_child(), old_tree->cuda_right_child(),
    old_tree->num_leaves(), refit_num_data_, old_tree->cuda_leaf_value_ref(), old_tree->shrinkage());
}

void CUDASingleGPUTreeLearner::ConstructBitsetForCategoricalSplit(
  const CUDASplitInfo* best_split_info) {
  LaunchConstructBitsetForCategoricalSplitKernel(best_split_info);
}

void CUDASingleGPUTreeLearner::AllocateBitset() {
  has_categorical_feature_ = false;
  categorical_bin_offsets_.clear();
  categorical_bin_offsets_.push_back(0);
  categorical_bin_to_value_.clear();
  for (int i = 0; i < train_data_->num_features(); ++i) {
    const BinMapper* bin_mapper = train_data_->FeatureBinMapper(i);
    if (bin_mapper->bin_type() == BinType::CategoricalBin) {
      has_categorical_feature_ = true;
      break;
    }
  }
  if (has_categorical_feature_) {
    int max_cat_value = 0;
    int max_cat_num_bin = 0;
    for (int i = 0; i < train_data_->num_features(); ++i) {
      const BinMapper* bin_mapper = train_data_->FeatureBinMapper(i);
      if (bin_mapper->bin_type() == BinType::CategoricalBin) {
        max_cat_value = std::max(bin_mapper->MaxCatValue(), max_cat_value);
        max_cat_num_bin = std::max(bin_mapper->num_bin(), max_cat_num_bin);
        if (bin_mapper->num_bin() > 256) {
          Log::Warning("Categorical feature %d has %d category bins; the CUDA "
                       "many-vs-many split search considers the 255 most frequent "
                       "categories (rarer ones route to the right child). Group "
                       "rare categories to avoid this.", i, bin_mapper->num_bin());
        }
      }
    }
    // std::max(..., 1) to avoid error in the case when there are NaN's in the categorical values.
    // The second argument must be size_t (not the 1UL literal used previously): unsigned long is
    // 32-bit on LLP64 platforms (MSVC/Windows) while size_t is 64-bit there, so std::max's matching
    // template argument deduction failed to compile on Windows even though it worked on Linux.
    const size_t cuda_bitset_max_size = std::max(static_cast<size_t>((max_cat_value + 31) / 32), static_cast<size_t>(1));
    const size_t cuda_bitset_inner_max_size = std::max(static_cast<size_t>((max_cat_num_bin + 31) / 32), static_cast<size_t>(1));
    batch_cat_bitset_stride_ = cuda_bitset_max_size;
    batch_cat_bitset_inner_stride_ = cuda_bitset_inner_max_size;
    AllocateCUDAMemory<uint32_t>(&cuda_bitset_, cuda_bitset_max_size, __FILE__, __LINE__);
    AllocateCUDAMemory<uint32_t>(&cuda_bitset_inner_, cuda_bitset_inner_max_size, __FILE__, __LINE__);
    const int max_cat_in_split = std::min(config_->max_cat_threshold, max_cat_num_bin / 2);
    const int num_blocks = (max_cat_in_split + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
    AllocateCUDAMemory<size_t>(&cuda_block_bitset_len_buffer_, num_blocks, __FILE__, __LINE__);

    for (int i = 0; i < train_data_->num_features(); ++i) {
      const BinMapper* bin_mapper = train_data_->FeatureBinMapper(i);
      if (bin_mapper->bin_type() == BinType::CategoricalBin) {
        categorical_bin_offsets_.push_back(bin_mapper->num_bin());
      } else {
        categorical_bin_offsets_.push_back(0);
      }
    }
    for (size_t i = 1; i < categorical_bin_offsets_.size(); ++i) {
      categorical_bin_offsets_[i] += categorical_bin_offsets_[i - 1];
    }
    categorical_bin_to_value_.resize(categorical_bin_offsets_.back(), 0);
    for (int i = 0; i < train_data_->num_features(); ++i) {
      const BinMapper* bin_mapper = train_data_->FeatureBinMapper(i);
      if (bin_mapper->bin_type() == BinType::CategoricalBin) {
        const int offset = categorical_bin_offsets_[i];
        for (int bin = 0; bin < bin_mapper->num_bin(); ++bin) {
          categorical_bin_to_value_[offset + bin] = static_cast<int>(bin_mapper->BinToValue(bin));
        }
      }
    }
    InitCUDAMemoryFromHostMemory<int>(&cuda_categorical_bin_offsets_, categorical_bin_offsets_.data(), categorical_bin_offsets_.size(), __FILE__, __LINE__);
    InitCUDAMemoryFromHostMemory<int>(&cuda_categorical_bin_to_value_, categorical_bin_to_value_.data(), categorical_bin_to_value_.size(), __FILE__, __LINE__);
  } else {
    cuda_bitset_ = nullptr;
    cuda_bitset_inner_ = nullptr;
  }
  cuda_bitset_len_ = 0;
  cuda_bitset_inner_len_ = 0;
}

void CUDASingleGPUTreeLearner::ResetBoostingOnGPU(const bool boosting_on_cuda) {
  boosting_on_cuda_ = boosting_on_cuda;
  cuda_gradients_.Clear();
  cuda_hessians_.Clear();
  if (!boosting_on_cuda_) {
    cuda_gradients_.Resize(static_cast<size_t>(num_data_));
    cuda_hessians_.Resize(static_cast<size_t>(num_data_));
  }
}

void CUDASingleGPUTreeLearner::SelectFeatureByNode(const Tree* tree) {
  if (select_features_by_node_) {
    // use feature interaction constraint or sample features by node
    const std::vector<int8_t>& is_feature_used_by_smaller_node = col_sampler_.GetByNode(tree, smaller_leaf_index_);
    std::vector<int8_t> is_feature_used_by_larger_node;
    if (larger_leaf_index_ >= 0) {
      is_feature_used_by_larger_node = col_sampler_.GetByNode(tree, larger_leaf_index_);
    }
    cuda_best_split_finder_->SetUsedFeatureByNode(is_feature_used_by_smaller_node, is_feature_used_by_larger_node);
  }
}

#ifdef DEBUG
void CUDASingleGPUTreeLearner::CheckSplitValid(
  const int left_leaf,
  const int right_leaf) {
  std::vector<data_size_t> left_data_indices(leaf_num_data_[left_leaf]);
  std::vector<data_size_t> right_data_indices(leaf_num_data_[right_leaf]);
  CopyFromCUDADeviceToHost<data_size_t>(left_data_indices.data(),
    cuda_data_partition_->cuda_data_indices() + leaf_data_start_[left_leaf],
    leaf_num_data_[left_leaf], __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<data_size_t>(right_data_indices.data(),
    cuda_data_partition_->cuda_data_indices() + leaf_data_start_[right_leaf],
    leaf_num_data_[right_leaf], __FILE__, __LINE__);
  double sum_left_gradients = 0.0f, sum_left_hessians = 0.0f;
  double sum_right_gradients = 0.0f, sum_right_hessians = 0.0f;
  for (size_t i = 0; i < left_data_indices.size(); ++i) {
    const data_size_t index = left_data_indices[i];
    sum_left_gradients += host_gradients_[index];
    sum_left_hessians += host_hessians_[index];
  }
  for (size_t i = 0; i < right_data_indices.size(); ++i) {
    const data_size_t index = right_data_indices[i];
    sum_right_gradients += host_gradients_[index];
    sum_right_hessians += host_hessians_[index];
  }
  if (nccl_communicator_ != nullptr) {
    sum_left_gradients = NCCLAllReduce<double>(sum_left_gradients, ncclFloat64, ncclSum, nccl_communicator_);
    sum_left_hessians = NCCLAllReduce<double>(sum_left_hessians, ncclFloat64, ncclSum, nccl_communicator_);
    sum_right_gradients = NCCLAllReduce<double>(sum_right_gradients, ncclFloat64, ncclSum, nccl_communicator_);
    sum_right_hessians = NCCLAllReduce<double>(sum_right_hessians, ncclFloat64, ncclSum, nccl_communicator_);
  }
  CHECK_LE(std::fabs(sum_left_gradients - leaf_sum_gradients_[left_leaf]), 1e-6f);
  CHECK_LE(std::fabs(sum_left_hessians - leaf_sum_hessians_[left_leaf]), 1e-6f);
  CHECK_LE(std::fabs(sum_right_gradients - leaf_sum_gradients_[right_leaf]), 1e-6f);
  CHECK_LE(std::fabs(sum_right_hessians - leaf_sum_hessians_[right_leaf]), 1e-6f);
}
#endif  // DEBUG

void CUDASingleGPUTreeLearner::RenewDiscretizedTreeLeaves(CUDATree* cuda_tree) {
  cuda_data_partition_->ReduceLeafGradStat(
    gradients_, hessians_, cuda_tree,
    cuda_leaf_gradient_stat_buffer_.RawData(),
    cuda_leaf_hessian_stat_buffer_.RawData());
  LaunchCalcLeafValuesGivenGradStat(cuda_tree, cuda_data_partition_->cuda_data_indices());
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDASingleGPUTreeLearner::SetNCCLInfo(
  ncclComm_t nccl_communicator,
  int nccl_gpu_rank,
  int local_gpu_rank,
  int gpu_device_id,
  data_size_t global_num_data) {
  NCCLInfo::SetNCCLInfo(nccl_communicator, nccl_gpu_rank, local_gpu_rank, gpu_device_id, global_num_data);
  leaf_to_hist_index_map_.resize(config_->num_leaves - 1);
  global_num_data_in_leaf_.resize(config_->num_leaves, 0);
  nccl_stream_ = CUDAStreamCreate();
  CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&nccl_reduce_done_event_, cudaEventDisableTiming));
}

void CUDASingleGPUTreeLearner::NCCLReduceLevelHistograms(const int num_pairs) {
  if (num_pairs <= 0) return;
  global_timer.Start("CUDASingleGPUTreeLearner::NCCLReduceLevelHistograms");
  // ConstructHistogramsForLevel(defer_subtract=true) recorded this event on
  // its stream right after the batched construct launch
  CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(
      nccl_stream_, cuda_histogram_constructor_->construct_done_events()[0], 0));
  const int num_bins = nccl_reduce_num_bins_ > 0 ? nccl_reduce_num_bins_ : num_total_bin_;
  const int* idx = nccl_reduce_num_bins_ > 0 ? cuda_nccl_reduce_bin_idx_.RawDataReadOnly() : nullptr;
  const size_t elems = static_cast<size_t>(num_pairs) * num_bins * 2;
  if (cuda_nccl_reduce_buf_.Size() < elems) {
    // growth frees the old buffer: the previous level's scatter must be done
    // reading it. Rare (monotonic high-water growth), so a sync here is cheap.
    SynchronizeCUDAStream(nccl_stream_, __FILE__, __LINE__);
    cuda_nccl_reduce_buf_.Resize(elems);
  }
  const CUDAHybridPairDescriptor* descs = cuda_hybrid_pair_descs_.RawDataReadOnly();
  if (config_->use_quantized_grad) {
    // integer lanes: one int64 per bin (half the fp64-pair payload), exact sums
    int64_t* ibuf = reinterpret_cast<int64_t*>(cuda_nccl_reduce_buf_.RawData());
    LaunchNCCLGatherLevelQuant(descs, idx, ibuf, num_bins, num_pairs, nccl_stream_);
    NCCLAllReduce<int64_t>(ibuf, ibuf,
                           static_cast<size_t>(num_pairs) * num_bins,
                           ncclInt64, ncclSum, nccl_communicator_, nccl_stream_);
    LaunchNCCLScatterLevelQuant(ibuf, idx, descs, num_bins, num_pairs, nccl_stream_);
  } else {
    LaunchNCCLGatherLevel(descs, idx, cuda_nccl_reduce_buf_.RawData(),
                          num_bins, num_pairs, nccl_stream_);
    NCCLAllReduce<double>(cuda_nccl_reduce_buf_.RawData(), cuda_nccl_reduce_buf_.RawData(),
                          elems, ncclFloat64, ncclSum, nccl_communicator_, nccl_stream_);
    LaunchNCCLScatterLevel(cuda_nccl_reduce_buf_.RawDataReadOnly(), idx, descs,
                           num_bins, num_pairs, nccl_stream_);
  }
  // The deferred SubtractHistogramsForLevel launches next on the histogram
  // stream and must see the globally reduced smaller histograms: order it
  // with a cross-stream event instead of a host-blocking sync (the collective
  // itself synchronizes the ranks; the host has no reason to wait with it).
  CUDASUCCESS_OR_FATAL(cudaEventRecord(nccl_reduce_done_event_, nccl_stream_));
  CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(
      cuda_histogram_constructor_->hist_stream(), nccl_reduce_done_event_, 0));
  if (FalcataDebug().dump) {
    // per-pair reduce diagnostics: the hessian total of the reduced smaller
    // histogram must match the (global) struct sum the finder will gate on
    std::vector<double> buf(static_cast<size_t>(num_pairs) * num_bins * 2);
    CopyFromCUDADeviceToHost<double>(buf.data(), cuda_nccl_reduce_buf_.RawDataReadOnly(),
                                     buf.size(), __FILE__, __LINE__);
    for (int i = 0; i < num_pairs; ++i) {
      double hess_total = 0.0;
      for (int b = 0; b < num_bins; ++b) {
        hess_total += buf[static_cast<size_t>(i) * num_bins * 2 + 2 * b + 1];
      }
      const CUDAHybridPairDescriptor& d = host_hybrid_pair_descs_[i];
      CUDALeafSplitsStruct s{};
      CopyFromCUDADeviceToHost<CUDALeafSplitsStruct>(&s, d.smaller_struct, 1, __FILE__, __LINE__);
      Log::Warning("[nccl-lvl rank%d] pair%d leaf=%d desc_n=%d struct_n=%d struct_hess=%.2f hist_hess=%.2f cvalid=%d",
                   nccl_gpu_rank_, i, d.smaller_leaf_index, d.num_data_in_smaller_leaf,
                   s.num_data_in_leaf, s.sum_of_hessians, hess_total,
                   static_cast<int>(d.construct_valid));
    }
  }
  global_timer.Stop("CUDASingleGPUTreeLearner::NCCLReduceLevelHistograms");
}

bool CUDASingleGPUTreeLearner::BuildNCCLReduceBinIndex() {
  // Only worth it when feature_fraction sampling actually excludes features.
  if (!cuda_histogram_constructor_->any_feature_unused_bytree()) {
    nccl_reduce_num_bins_ = 0;
    return false;
  }
  const std::vector<uint8_t>& used = cuda_histogram_constructor_->host_bin_used_bytree();
  if (static_cast<int>(used.size()) != num_total_bin_) {
    nccl_reduce_num_bins_ = 0;
    return false;
  }
  std::vector<int> idx;
  idx.reserve(used.size());
  for (int b = 0; b < num_total_bin_; ++b) {
    if (used[b]) idx.push_back(b);
  }
  // compaction costs a gather + a scatter; only pay it when it removes real
  // traffic (the collective is latency-bound below a few tens of KB anyway)
  if (idx.empty() || static_cast<double>(idx.size()) > 0.75 * num_total_bin_) {
    nccl_reduce_num_bins_ = 0;
    return false;
  }
  nccl_reduce_num_bins_ = static_cast<int>(idx.size());
  if (cuda_nccl_reduce_bin_idx_.Size() < idx.size()) {
    cuda_nccl_reduce_bin_idx_.Resize(idx.size());
    cuda_nccl_reduce_buf_.Resize(idx.size() * 2);
  }
  CopyFromHostToCUDADevice<int>(cuda_nccl_reduce_bin_idx_.RawData(), idx.data(),
                                idx.size(), __FILE__, __LINE__);
  return true;
}

void CUDASingleGPUTreeLearner::NCCLReduceHistogram(const int pipeline) {
  // ConstructHistogramForLeaf is ASYNC: it records construct_done_events_ on
  // the construct stream rather than syncing. The all-reduce below runs on
  // nccl_stream_, a DIFFERENT stream, so without this wait it can read the
  // histogram while it is still being built -- reducing garbage, and doing so
  // nondeterministically per rank. That produced wrong gains, ranks that grew
  // different trees, and ultimately a hang when they issued different numbers
  // of collectives. (The trailing SynchronizeCUDAStream already orders the
  // finder AFTER the reduce; only this edge was missing.)
  const cudaEvent_t hist_event = cuda_histogram_constructor_->construct_done_events()[pipeline];
  CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(nccl_stream_, hist_event, 0));
  if (config_->use_quantized_grad) {
    hist_t* smaller_leaf_hist_pointer = cuda_histogram_constructor_->cuda_hist_pointer() +
      leaf_to_hist_index_map_[smaller_leaf_index_] * num_total_bin_;
    const int8_t bit_size = cuda_gradient_discretizer_->GetHistBitsInNode<true>(smaller_leaf_index_);
    if (bit_size == 32) {
      NCCLAllReduce<int64_t>(
        reinterpret_cast<const int64_t*>(smaller_leaf_hist_pointer),
        reinterpret_cast<int64_t*>(smaller_leaf_hist_pointer),
        static_cast<size_t>(num_total_bin_),
        ncclInt64,
        ncclSum,
        nccl_communicator_,
        nccl_stream_);
    } else if (bit_size <= 16) {
      NCCLAllReduce<int32_t>(
        reinterpret_cast<const int32_t*>(smaller_leaf_hist_pointer),
        reinterpret_cast<int32_t*>(smaller_leaf_hist_pointer),
        static_cast<size_t>(num_total_bin_),
        ncclInt32,
        ncclSum,
        nccl_communicator_,
        nccl_stream_);
    }
  } else {
    hist_t* smaller_leaf_hist_pointer = cuda_histogram_constructor_->cuda_hist_pointer() +
        leaf_to_hist_index_map_[smaller_leaf_index_] * num_total_bin_ * 2;
    if (nccl_reduce_num_bins_ > 0) {
      // ship only the bins this tree's feature sample can read
      LaunchNCCLGatherBins(smaller_leaf_hist_pointer, cuda_nccl_reduce_buf_.RawData(),
                           nccl_reduce_num_bins_, nccl_stream_);
      NCCLAllReduce<double>(
        cuda_nccl_reduce_buf_.RawData(),
        cuda_nccl_reduce_buf_.RawData(),
        static_cast<size_t>(nccl_reduce_num_bins_) * 2,
        ncclFloat64,
        ncclSum,
        nccl_communicator_,
        nccl_stream_);
      LaunchNCCLScatterBins(cuda_nccl_reduce_buf_.RawData(), smaller_leaf_hist_pointer,
                            nccl_reduce_num_bins_, nccl_stream_);
    } else {
      NCCLAllReduce<hist_t>(
        smaller_leaf_hist_pointer,
        smaller_leaf_hist_pointer,
        static_cast<size_t>(num_total_bin_) * 2,
        ncclFloat64,
        ncclSum,
        nccl_communicator_,
        nccl_stream_);
    }
  }
  SynchronizeCUDAStream(nccl_stream_, __FILE__, __LINE__);
}

void CUDASingleGPUTreeLearner::InitLinearTreeCUDA(const Dataset* train_data) {
  num_numeric_features_ = train_data->num_numeric_features();
  const int num_features = train_data->num_features();
  // Reproduce Dataset::numeric_feature_map_: numeric features numbered in feature order.
  inner_feature_to_numeric_col_.assign(num_features, -1);
  int col = 0;
  for (int feat = 0; feat < num_features; ++feat) {
    if (train_data->FeatureBinMapper(feat)->bin_type() == BinType::NumericalBin) {
      inner_feature_to_numeric_col_[feat] = col;
      ++col;
    }
  }
  CHECK_EQ(col, num_numeric_features_);
  if (num_numeric_features_ == 0) {
    return;
  }
  // Upload raw (un-binned) float columns, feature-major, matching host raw_index().
  const size_t total = static_cast<size_t>(num_numeric_features_) * static_cast<size_t>(num_data_);
  cuda_raw_data_.Resize(total);
  for (int feat = 0; feat < num_features; ++feat) {
    const int c = inner_feature_to_numeric_col_[feat];
    if (c < 0) {
      continue;
    }
    CopyFromHostToCUDADevice<float>(
      cuda_raw_data_.RawData() + static_cast<size_t>(c) * static_cast<size_t>(num_data_),
      train_data->raw_index(feat), static_cast<size_t>(num_data_), __FILE__, __LINE__);
  }
}

void CUDASingleGPUTreeLearner::CalculateLinearCUDA(
    CUDATree* tree, const score_t* gradients, const score_t* hessians, bool is_first_tree) {
  tree->SetIsLinear(true);
  const int num_leaves = tree->num_leaves();

  if (is_first_tree) {
    // First tree: leaf_const = leaf output, no coefficients (matches CPU CalculateLinear).
    for (int leaf = 0; leaf < num_leaves; ++leaf) {
      tree->SetLeafConst(leaf, tree->LeafOutput(leaf));
    }
    return;
  }

  // ---- build per-leaf numeric feature lists from branch features (non-refit path) ----
  std::vector<std::vector<int>> leaf_feat_inner(num_leaves);  // inner feature indices
  std::vector<int> host_num_feat(num_leaves, 0);
  std::vector<int> feat_offset(num_leaves + 1, 0);
  std::vector<int> xthx_offset(num_leaves + 1, 0);
  std::vector<int> xtg_offset(num_leaves + 1, 0);
  std::vector<int> feat_col_flat;
  for (int leaf = 0; leaf < num_leaves; ++leaf) {
    std::vector<int> raw_features = tree->branch_features(leaf);
    std::sort(raw_features.begin(), raw_features.end());
    raw_features.erase(std::unique(raw_features.begin(), raw_features.end()), raw_features.end());
    for (int real_feat : raw_features) {
      const int inner = train_data_->InnerFeatureIndex(real_feat);
      if (inner < 0) {
        continue;
      }
      if (train_data_->FeatureBinMapper(inner)->bin_type() == BinType::NumericalBin) {
        leaf_feat_inner[leaf].push_back(inner);
        feat_col_flat.push_back(inner_feature_to_numeric_col_[inner]);
      }
    }
    const int nf = static_cast<int>(leaf_feat_inner[leaf].size());
    // the gram kernel holds one row of features in a fixed-size register array
    CHECK_LE(nf, 256);
    host_num_feat[leaf] = nf;
    feat_offset[leaf + 1] = feat_offset[leaf] + nf;
    xthx_offset[leaf + 1] = xthx_offset[leaf] + (nf + 1) * (nf + 2) / 2;
    xtg_offset[leaf + 1] = xtg_offset[leaf] + (nf + 1);
  }

  // ---- device buffers + accumulation kernel ----
  CUDAVector<int> d_num_feat, d_feat_offset, d_feat_col, d_xthx_offset, d_xtg_offset;
  d_num_feat.InitFromHostVector(host_num_feat);
  d_feat_offset.InitFromHostVector(feat_offset);
  d_xthx_offset.InitFromHostVector(xthx_offset);
  d_xtg_offset.InitFromHostVector(xtg_offset);
  if (feat_col_flat.empty()) {
    d_feat_col.Resize(1);
  } else {
    d_feat_col.InitFromHostVector(feat_col_flat);
  }
  const int total_xthx = xthx_offset[num_leaves];
  const int total_xtg = xtg_offset[num_leaves];
  cuda_linear_xthx_.Resize(static_cast<size_t>(total_xthx));
  cuda_linear_xtg_.Resize(static_cast<size_t>(total_xtg));
  cuda_linear_nonzero_.Resize(static_cast<size_t>(num_leaves));
  cuda_linear_xthx_.SetValue(0);
  cuda_linear_xtg_.SetValue(0);
  cuda_linear_nonzero_.SetValue(0);

  LaunchCalcLinearGramKernel(
    cuda_data_partition_->cuda_data_index_to_leaf_index(), gradients, hessians,
    d_num_feat.RawData(), d_feat_offset.RawData(), d_feat_col.RawData(),
    d_xthx_offset.RawData(), d_xtg_offset.RawData(), num_leaves,
    cuda_linear_xthx_.RawData(), cuda_linear_xtg_.RawData(), cuda_linear_nonzero_.RawData());

  std::vector<double> h_xthx(total_xthx), h_xtg(total_xtg);
  std::vector<int> h_nonzero(num_leaves);
  CopyFromCUDADeviceToHost<double>(h_xthx.data(), cuda_linear_xthx_.RawData(), total_xthx, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(h_xtg.data(), cuda_linear_xtg_.RawData(), total_xtg, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(h_nonzero.data(), cuda_linear_nonzero_.RawData(), num_leaves, __FILE__, __LINE__);

  // ---- per-leaf solve (host Eigen, matching CPU exactly) + store into tree ----
  const double lambda = config_->linear_lambda;
  for (int leaf = 0; leaf < num_leaves; ++leaf) {
    const int num_feat = host_num_feat[leaf];
    if (h_nonzero[leaf] < num_feat + 1) {
      // degenerate leaf: constant model (matches CPU non-refit branch)
      tree->SetLeafConst(leaf, tree->LeafOutput(leaf));
      continue;
    }
    // Device-agnostic solve + store, shared with the CPU LinearTreeLearner.
    LinearLeafSolver::SolveAndStore(
        tree, leaf, num_feat, h_xthx.data() + xthx_offset[leaf], h_xtg.data() + xtg_offset[leaf],
        leaf_feat_inner[leaf], train_data_, lambda,
        /*is_refit=*/false, /*decay_rate=*/0.0, /*shrinkage=*/0.0, /*old_coeff_idx=*/{});
  }
}

void CUDASingleGPUTreeLearner::BuildAndUploadLinearScoreModel(const Tree* tree) const {
  // Read the tree's current (already-shrunk-by-learning-rate) per-leaf linear model
  // and flatten it for the score kernel. Mirrors the CPU LinearTreeLearner reading
  // tree->LeafConst/LeafCoeffs/LeafFeaturesInner at score time.
  const int num_leaves = tree->num_leaves();
  std::vector<double> leaf_const(num_leaves), leaf_output(num_leaves);
  std::vector<int> num_coeff(num_leaves, 0), coeff_offset(num_leaves + 1, 0);
  std::vector<double> coeff_flat;
  std::vector<int> coeff_col;
  for (int leaf = 0; leaf < num_leaves; ++leaf) {
    leaf_const[leaf] = tree->LeafConst(leaf);
    leaf_output[leaf] = tree->LeafOutput(leaf);
    const std::vector<double>& c = tree->LeafCoeffs(leaf);
    const std::vector<int>& inner = tree->LeafFeaturesInner(leaf);
    for (size_t i = 0; i < c.size(); ++i) {
      coeff_flat.push_back(c[i]);
      coeff_col.push_back(inner_feature_to_numeric_col_[inner[i]]);
    }
    num_coeff[leaf] = static_cast<int>(c.size());
    coeff_offset[leaf + 1] = coeff_offset[leaf] + num_coeff[leaf];
  }
  cuda_linear_leaf_const_.InitFromHostVector(leaf_const);
  cuda_linear_leaf_output_.InitFromHostVector(leaf_output);
  cuda_linear_leaf_num_coeff_.InitFromHostVector(num_coeff);
  cuda_linear_leaf_coeff_offset_.InitFromHostVector(coeff_offset);
  if (coeff_flat.empty()) {  // CUDAVector cannot init from an empty vector
    cuda_linear_leaf_coeff_.Resize(1);
    cuda_linear_leaf_coeff_col_.Resize(1);
  } else {
    cuda_linear_leaf_coeff_.InitFromHostVector(coeff_flat);
    cuda_linear_leaf_coeff_col_.InitFromHostVector(coeff_col);
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
