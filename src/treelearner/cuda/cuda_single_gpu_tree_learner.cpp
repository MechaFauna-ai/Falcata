/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/falcata_plan.h>
#include <Falcata/feature_group.h>
#include <Falcata/network.h>
#include <Falcata/objective_function.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <limits>
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
#ifdef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
#include "../../../tests/cpp_tests/whole_map_only_fault_test.hpp"
#endif

namespace Falcata {
#ifdef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
namespace {
thread_local WholeMapOnlyFaultCallback whole_map_only_fault_callback = nullptr;
}
void SetWholeMapOnlyFaultCallback(WholeMapOnlyFaultCallback callback) {
  whole_map_only_fault_callback = callback;
}
void WholeMapOnlyFaultCheckpoint(CUDASingleGPUTreeLearner* learner, WholeMapOnlyFaultPoint point) {
  if (whole_map_only_fault_callback != nullptr) whole_map_only_fault_callback(learner, point);
}
#endif

CUDASingleGPUTreeLearner::CUDASingleGPUTreeLearner(const Config* config, const bool boosting_on_cuda): SerialTreeLearner(config), boosting_on_cuda_(boosting_on_cuda) {}

CUDASingleGPUTreeLearner::~CUDASingleGPUTreeLearner() {
  if (nccl_communicator_ != nullptr) {
    CUDAStreamDestroy(nccl_stream_);
  }
  if (readback_staging_ != nullptr) {
    cudaFreeHost(readback_staging_);
  }
  if (tree_record_done_event_ != nullptr) {
    cudaEventDestroy(tree_record_done_event_);
  }
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  device_level_graphs_.clear();
  if (pinned_device_level_journal_ != nullptr) {
    cudaFreeHost(pinned_device_level_journal_);
  }
  if (pinned_selective_prefix_split_infos_ != nullptr) {
    cudaFreeHost(pinned_selective_prefix_split_infos_);
  }
  if (device_level_select_event_ != nullptr) {
    cudaEventDestroy(device_level_select_event_);
  }
  if (bynode_fork_event_ != nullptr) {
    cudaEventDestroy(bynode_fork_event_);
  }
  if (bynode_join_event_ != nullptr) {
    cudaEventDestroy(bynode_join_event_);
  }
  if (bynode_stream_ != nullptr) {
    cudaStreamDestroy(bynode_stream_);
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
  construct_saturation_floor_ = 0;
  if (config_->use_quantized_grad) {
    // the batched quantized construct's saturation floor (FalcataPlan::batch_construct_saturation_floor): 160 was
    // tuned on the RTX 5090's 170 SMs and scales with the device's SM count, clamped to [10, 5120] -- the seed rule
    // of the runtime tuner removed 2026-10-15, now static. A 170-SM device keeps 160. The quantized small-leaf
    // threshold keeps its compile-time 1024. Both are schedule-only: any value trains the same model. Set again at
    // every tree (Train), as the tuner did: a Dataset constructed after this Init re-resolves the plan.
    int sm_count = 0;
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, gpu_device_id_ < 0 ? 0 : gpu_device_id_) ==
          cudaSuccess && sm_count > 0) {
      construct_saturation_floor_ = std::max(10, std::min(5120, static_cast<int>(160.0 * sm_count / 170.0 + 0.5)));
      FalcataPlan::Mutable().batch_construct_saturation_floor = construct_saturation_floor_;
    }
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
  if (forest_row_data_source_ != nullptr) {
    // cuda_plan key forest_batched: a slot reads the booster learner's bin matrix (same data, read-only after Init)
    cuda_histogram_constructor_->ShareRowData(forest_row_data_source_->cuda_histogram_constructor_->row_data_shared());
  }
  cuda_histogram_constructor_->Init(train_data_, share_state_.get());
  // cuda_plan key forest_count_hist: a forest slot shares the booster learner's codes (same gradient arrays)
  count_hist_cache_ = forest_row_data_source_ != nullptr && forest_row_data_source_->count_hist_cache_ != nullptr ?
    forest_row_data_source_->count_hist_cache_ : std::make_shared<CUDACountHistCache>();

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
  if (forest_private_column_view_) {
    // cuda_plan key forest_batched: this slot's per-tree views go to a view of its own (the dataset's is the
    // booster learner's, and the other slots install theirs concurrently)
    private_column_view_.reset(new CUDAColumnData(num_data_, gpu_device_id_));
    private_column_view_->InitViewOf(train_data_->cuda_column_data());
    cuda_data_partition_->SetColumnData(private_column_view_.get());
  }
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
  // single-sync (speculative) level pipeline for the hybrid prefix: each level's
  // apply and the children's search are enqueued back to back and read back with
  // ONE device synchronization per level; "0" keeps the classic two-sync flow
  use_hybrid_one_sync_ = FalcataPlan::Get().one_sync;
  // vector-leaf training rides the hybrid TWO-SYNC level flows: the exact-fit
  // prefix in depth-limited configs and the SELECTIVE (grow-then-prune) prefix
  // in budget-limited ones, both on the per-plane histogram loop and the
  // batched vector find. The speculative (one-sync) flow derives its child
  // gates from device structs the plane fan-out has not refreshed yet, so it
  // falls back to the classic per-split loop here, as does the complete
  // non-quantized device tree (NonquantDeviceTreeShape).
  if (vec_num_targets_ > 1) {
    use_hybrid_one_sync_ = false;
  }
  cuda_best_split_finder_.reset(new CUDABestSplitFinder(cuda_histogram_constructor_->cuda_hist(),
    train_data_, this->share_state_->feature_hist_offsets(), select_features_by_node_, config_));
  cuda_best_split_finder_->Init();
  // cuda_plan key level_bynode_masks: feature_fraction_bynode < 1 without interaction constraints draws keyed
  // per-node masks in every flow, and the level flows take them (LevelFlowsTakeNodeMasks); interaction constraints
  // keep the shared-stream draws and the per-split loop
  bynode_keyed_ = FalcataPlan::Get().level_bynode_masks && config_->feature_fraction_bynode < 1.0 &&
    config_->interaction_constraints_vector.empty();
  bynode_level_masks_ = bynode_keyed_ && vec_num_targets_ <= 1;
  cuda_best_split_finder_->EnableLevelNodeMasks(bynode_level_masks_);
  bynode_tree_count_ = 0;
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
  // Wire the histogram subtraction completion events into the best split finder so
  // per-leaf FindBestSplits kernels are ordered after histogram subtraction
  // via cudaStreamWaitEvent instead of per-split device syncs.
  cuda_best_split_finder_->SetHistogramEvents(
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
    // vector-leaf mode discretizes one plane per target, each at its own
    // gradient scale; scalar training is the single-plane case
    cuda_gradient_discretizer_->SetNumPlanes(std::max(1, vec_num_targets_));
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
  colmajor_fill_pending_ = cuda_histogram_constructor_->ColMajorFillApplicable();
  colmajor_fill_trees_seen_ = 0;
  // colmajor_direct: the bin matrix has no row-major copy yet; its view is chosen now that the split finder, the
  // data partition and the histograms are allocated, so free memory already accounts for them
  InitViewRegime("train()");
}

// colmajor_direct: what training can still allocate once the regime's view exists, the compact regime's memory
// check (CUDAHistogramConstructor::ChooseViewRegime). As colmajor_fill's reserve, without the compact-view growth
// (the check counts the whole compact view), plus the booster's per-row state allocated after this learner's Init
// (scores, gradients and hessians per output; already allocated, and so counted twice, at a later re-choice):
//  - the split finder's per-level output and the quantized bit-change scratch for up to num_leaves / 2 + 2 pairs,
//    and with NCCL the level reduce buffer;
//  - 256 MiB for everything smaller.
size_t CUDASingleGPUTreeLearner::DirectViewReserveBytes() const {
  constexpr size_t kHeadroom = 256ULL << 20;
  const int max_pairs = config_->num_leaves / 2 + 2;
  size_t reserve = kHeadroom + cuda_best_split_finder_->HybridLevelGrowthBytes(max_pairs) +
                   cuda_histogram_constructor_->BitChangeScratchGrowthBytes(max_pairs);
  if (nccl_communicator_ != nullptr) {
    const size_t elems = static_cast<size_t>(max_pairs) * static_cast<size_t>(num_total_bin_) * 2;
    reserve += cuda_nccl_reduce_buf_.Size() >= elems ? 0 : elems * sizeof(double);
  }
  const size_t outputs = static_cast<size_t>(std::max(1, config_->num_class));
  reserve += static_cast<size_t>(num_data_) * outputs * (sizeof(double) + 2 * sizeof(score_t));
  return reserve;
}

void CUDASingleGPUTreeLearner::InitViewRegime(const char* why) {
  if (!cuda_histogram_constructor_->colmajor_direct()) {
    return;
  }
  view_decided_ff_ = config_->feature_fraction;
  cuda_histogram_constructor_->ChooseViewRegime(DirectViewReserveBytes(), BuildsOneByteSplitView(),
                                                TreesReadFullMatrix(), why);
}

// colmajor_fill sizing. The column-major copy of the 4-bit row matrix is the largest optional allocation training
// makes (as large as the row data itself) and, being optional, must never be what runs the device out of memory.
// It is therefore made before the SECOND tree, not at Init: by then the booster's scores, gradients and bag
// indices, the training and validation sets' device columns, the data partition, the gradient discretizer,
// linear-tree data and the first tree's compact views and level buffers all exist, so free device memory already
// accounts for them, multiclass and vector-leaf per-row state included. This returns what can still be allocated
// after that point, as an upper bound computed from the shape:
//  - the split finder's per-level output for up to num_leaves / 2 + 2 pairs (the graph loop's worst case) with its
//    categorical and vector-payload slabs, the quantized bit-change histogram scratch (pairs x total bins) and,
//    with NCCL, the level reduce buffer, each counted in full while short of that bound, since a growing Resize
//    holds the old block until the new one is allocated;
//  - the compact views growing as later trees' samples fall differently into the feature partitions;
//  - 256 MiB (the same headroom EnsureCompactColumnBuffer keeps) for everything smaller: per-level metadata, tree
//    buffers, allocator granularity.
// Nothing is reserved for the classic per-column split view (one byte per sampled value): the flows without the
// packed split read build it with each tree's compact view (BuildsOneByteSplitView), and the flows with it build it
// only in a leaf-wise tail (ApplySplit -> EnsureClassicColumnView), which they reach only when the batched root
// search found no split and the classic one did (the device tree's root tail): depth-limited growth (num_leaves + 1 >=
// 2^max_depth) cannot bind the budget before its last level, whose children all sit at max_depth and are applied as
// one batched partial level (ArbitrateLevelBudget), and selective growth builds the complete tree.
// Not covered: device memory taken outside this booster's training once the copy exists, e.g. a validation set
// added after the first tree, GPU prediction from a callback, or a ResetConfig that raises num_leaves or
// feature_fraction; colmajor_fill:off leaves that memory free.
size_t CUDASingleGPUTreeLearner::ColMajorFillReserveBytes() const {
  constexpr size_t kHeadroom = 256ULL << 20;
  const int max_pairs = config_->num_leaves / 2 + 2;
  size_t reserve = kHeadroom + cuda_best_split_finder_->HybridLevelGrowthBytes(max_pairs) +
                   cuda_histogram_constructor_->BitChangeScratchGrowthBytes(max_pairs) +
                   cuda_histogram_constructor_->CompactViewGrowthBytes();
  if (nccl_communicator_ != nullptr) {
    const size_t elems = static_cast<size_t>(max_pairs) * static_cast<size_t>(num_total_bin_) * 2;
    reserve += cuda_nccl_reduce_buf_.Size() >= elems ? 0 : elems * sizeof(double);
  }
  return reserve;
}

void CUDASingleGPUTreeLearner::MaybeInitColMajorFill() {
  if (!colmajor_fill_pending_ || colmajor_fill_trees_seen_++ == 0) {
    return;
  }
  colmajor_fill_pending_ = false;
  cuda_histogram_constructor_->InitColMajorFill(ColMajorFillReserveBytes());
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
#ifdef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
  WholeMapOnlyFaultCheckpoint(this, WholeMapOnlyFaultPoint::BeforeRootReuse);
#endif
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
#ifdef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
  WholeMapOnlyFaultCheckpoint(this, WholeMapOnlyFaultPoint::AfterRootInit);
#endif
  if (config_->use_quantized_grad) {
    // l2 ridge in hessian quanta in every discretized find kernel -- scalar
    // (FindBestSplitsDiscretizedForLeafKernel,
    // FindBestSplitsDiscretizedForLevelKernel) and vector
    // (FindBestSplitsDiscretizedVectorInner) alike -- wherever a child's
    // quantized hessian sum can fall far below its true hessian mass:
    //  - fixedpoint training whose hessians vary per row (softmax, logistic,
    //    weighted rows), bagged or not: kFixedpointHessianRidgeQuanta. Each
    //    subquantum hessian arrives as error-feedback quanta that are off by
    //    up to one quantum apiece, so a child's recorded H is uncertain by
    //    several quanta, not one; four is the size that keeps lambda_l2 = 0
    //    softmax training with saturating rare classes out of the collapse
    //    (tests/gates/canonical.py covtype-fixedpoint holds it);
    //  - other bagged training: one quantum, against the per-bag redraw of
    //    the rounding noise;
    //  - otherwise none: a constant unweighted hessian quantizes to exactly
    //    quant_bins per row, and unbagged stochastic training keeps the
    //    unridged math, so those models are bit-identical.
    constexpr double kFixedpointHessianRidgeQuanta = 4.0;
    constexpr double kBaggedHessianRidgeQuanta = 1.0;
    const bool hessians_exact = share_state_->is_constant_hessian &&
                                train_data_->metadata().weights() == nullptr;
    cuda_best_split_finder_->SetHessianRidgeQuanta(
        fixedpoint_quant_ && !hessians_exact ? kFixedpointHessianRidgeQuanta :
        cuda_data_partition_->use_bagging() ? kBaggedHessianRidgeQuanta : 0.0);
    if (cuda_data_partition_->use_bagging()) {
      cuda_gradient_discretizer_->SetBagForThisTree(
          cuda_data_partition_->cuda_data_indices(), cuda_data_partition_->root_num_data());
    } else {
      cuda_gradient_discretizer_->SetBagForThisTree(nullptr, 0);
    }
    // cuda_plan key const_hess_reads: the objective's IsConstantHessian (false under GOSS, which rescales
    // hessians) promises one hessian value for every row; dataset weights would scale it per row (MAPE reports a
    // constant hessian even then), and vector leaves share the array across planes
    cuda_gradient_discretizer_->SetHessiansConstant(share_state_->is_constant_hessian &&
                                                    train_data_->metadata().weights() == nullptr &&
                                                    vec_num_targets_ <= 1);
    cuda_gradient_discretizer_->DiscretizeGradients(num_data_, gradients_, hessians_);
    for (int t = 1; t < vec_num_targets_; ++t) {
      // plane t: target t's gradients at their own scale, plane 0's hessians
      cuda_gradient_discretizer_->DiscretizeGradientsForPlane(
        t, num_data_, gradients_ + static_cast<size_t>(t) * num_data_, hessians_);
    }
    cuda_histogram_constructor_->BeforeTrain(
      reinterpret_cast<const score_t*>(cuda_gradient_discretizer_->discretized_gradients_and_hessians()), nullptr);
    // cuda_plan key async_tree_start: the root sums are first needed by the level prefix
    // (TrainLevelWisePrefix's EnsureRootSumsReadBack; Train reads them back itself before any other flow), so
    // the synchronous readback -- which waits for the histogram zeroing and the root sum kernels -- moves behind
    // the column sample, the metadata uploads and the fill launch instead of holding them back
    root_sums_deferred_ = AsyncTreeStart() && vec_num_targets_ <= 1 && nccl_communicator_ == nullptr &&
                          fp_merge_state_ == nullptr && HybridGrowthUsable() && !UseSelectiveGrowth();
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
      cuda_gradient_discretizer_->hess_scale_ptr(),
      root_sums_deferred_);
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
    // One ownership model of the histogram storage per run (see DetDenseBatchedEligible): a configuration the
    // complete device tree takes never runs the host batched deterministic construct, on any of its trees (the
    // root level and trees the one-sync flow grows build in fixed point, like the device levels)
    cuda_histogram_constructor_->SetDetBatchedAllowed(!NonquantDeviceTreeShape());
    cuda_histogram_constructor_->SetCountCodes(CountHistCodesForTree());
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
  if (forest_next_sample_ready_) {
    // cuda_plan key forest_batched: the forest trainer drew this tree's sample in tree order
    col_sampler_.SetByTreeSample(forest_next_sample_);
    forest_next_sample_ready_ = false;
  } else if (next_tree_col_sample_ready_) {
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
    // cuda_plan key tree_meta_batch: the uploads from here to the compact fill go out as one copy and one scatter
    // (flushed before the fill and any other GPU operation in between, and at the end of this block)
    TreeStartMetaBatchScope meta_batch_scope(
      FalcataPlan::Get().tree_meta_batch && AsyncTreeStart() && nccl_communicator_ == nullptr ?
        &tree_meta_batch_ : nullptr);
    cuda_best_split_finder_->BeforeTrain(col_sampler_.is_feature_used_bytree());
    cuda_histogram_constructor_->SetFeatureUsedBytree(col_sampler_.is_feature_used_bytree());
    // fused root histogram: the fill sees every row, which is the root leaf only without bagging; the root level
    // of the batched host flow consumes it (EnqueueLevelBestSplitSearch)
    // cuda_plan key bagged_code_view: under bagging the root is the bag, so the fill's histogram of every row is
    // discarded, but the fused fill still runs for the pair-joint code view it writes (pair_code5), which the bagged
    // levels' gathering construct reads like the unbagged ones
    const bool fused_root_shape = FalcataPlan::Get().fused_root_hist && config_->use_quantized_grad &&
      use_hybrid_growth_ && vec_num_targets_ <= 1 && nccl_communicator_ == nullptr;
    const bool root_is_every_row = !cuda_data_partition_->use_bagging() && root_num_data == num_data_;
    const bool bagged_code_view = FalcataPlan::Get().bagged_code_view && cuda_data_partition_->use_bagging();
    cuda_histogram_constructor_->RequestFusedRootHist(
      fused_root_shape && (root_is_every_row || bagged_code_view),
      config_->use_quantized_grad && cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(0) <= 16,
      !root_is_every_row);
    // cuda_plan key pair_code5: no code view for a tree whose split path reads the nibble view anyway
    cuda_histogram_constructor_->AllowCode5View(!BuildsOneByteSplitView());
    cuda_histogram_constructor_->BuildCompactView(col_sampler_.is_feature_used_bytree());
    cuda_histogram_constructor_->RequestFusedRootHist(false, false);
    meta_batch_scope.End();
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
  // Only leaf 0 is active now. Its count/start and quantized bit width were reset above; its sums retain
  // the existing root_sums_deferred_ contract. Old nonroot entries stay dead until an ordinary split writes them.
  whole_host_leaf_cache_stale_ = false;

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
  if (FalcataPlan::Get().compact_prefill && !select_features_by_node_ && !forest_slot_ &&
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

bool CUDASingleGPUTreeLearner::PackedSplitReadUsable() const {
  return SplitPackedReadEnabled() && HybridGrowthUsable() && !has_categorical_feature_;
}

// The one-byte split view (sampled columns x rows) is built by BuildCompactColumnView's gather whenever the packed
// read does not apply (otherwise a leaf-wise tail's first classic split builds it lazily, EnsureClassicColumnView;
// see ColMajorFillReserveBytes for when one runs). Forced splits switch the hybrid flow off but reach this learner after
// its Init (GBDT::SetForcedSplit), where HybridGrowthUsable cannot see them yet, so they count directly.
bool CUDASingleGPUTreeLearner::BuildsOneByteSplitView() const {
  return !PackedSplitReadUsable() || !config_->forcedsplits_filename.empty();
}

// Quantized training outside the hybrid level flow (the classic loop: per-node sampling, forced splits, two leaves,
// monotone constraints, CEGB, ...) builds every histogram with the per-leaf construct, which reads the whole matrix
// row-major and has no compact-view branch; colmajor_direct then takes the mask regime.
bool CUDASingleGPUTreeLearner::TreesReadFullMatrix() const {
  return config_->use_quantized_grad && (!HybridGrowthUsable() || !config_->forcedsplits_filename.empty());
}

void CUDASingleGPUTreeLearner::BuildCompactColumnView() {
  const auto* row_data = cuda_histogram_constructor_->cuda_row_data_internal();
  if (row_data == nullptr || train_data_->num_features() == 0) return;  // no row data was built
  if (row_data->bit_type() != 8 || row_data->is_sparse()) return;  // only dense uint8 path supported
  // The whole matrix in the row-major layout is read below only when the compact matrix cannot serve this tree's
  // columns: the row data's matrix, or under colmajor_direct the full view (EnsureFullView there, not here).

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
  // ... the row data itself (ResetTrainingData replaces it, and the matrix a view may point into) and colmajor_direct's
  // full view (released and refilled at another address when trees stop and resume needing it, or on a regime change)
  sig = sig * 1099511628211ULL ^ static_cast<uint64_t>(reinterpret_cast<uintptr_t>(row_data));
  sig = sig * 1099511628211ULL ^
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cuda_histogram_constructor_->full_view_device()));
  // ... and the column-major store colmajor_split points the packed view into (released on a switch to the mask
  // regime and re-uploaded, possibly elsewhere, on the way back)
  sig = sig * 1099511628211ULL ^
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cuda_histogram_constructor_->colmajor_bin()));
  // Someone else may have replaced the published table since we installed ours
  // -- tree traversal restores the original per-column view (see
  // CUDATree::LaunchAddPredictionToScoreKernel) -- and then an unchanged layout
  // no longer means our view is the one in force.
  CUDAColumnData* col_data = ColumnView();
  if (sig == compact_col_signature_ && !compact_column_to_orig_.empty() &&
      col_view_generation_seen_ == col_data->column_view_generation()) {
    // Cache hit: layout, source pointers AND published view unchanged.
    if (column_view_reads_full_view_) {
      cuda_histogram_constructor_->EnsureFullView("a split view the compact matrix cannot serve");  // still in use
    }
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

  // NOT allocated here. Two of the three paths below return before the gather
  // and never read compact_column_buffer_: the fused-second-output path points
  // CUDAColumnData straight at the histogram constructor's column-major view,
  // and the packed-split-read path only rewrites a pointer table. Allocating up
  // front therefore reserved one byte per (sampled feature x row) — 10.69 GiB
  // at 3555 features and 6.3M rows — that nothing ever touched, and it was the
  // single largest contributor to the colsample OOM ceiling. The gather path
  // allocates it at its own call site below; EnsureClassicColumnView, the only
  // other consumer, already calls this itself.

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
  column_view_reads_full_view_ = !compact_src;
  // the gather (and the packed split view, and the out-of-bag pass that traverses it) reads every column in the
  // row-major layout unless the compact matrix serves: the row data's matrix, or colmajor_direct's full view
  const uint8_t* gather_src = compact_src ? cuda_histogram_constructor_->compact_data_device() :
    cuda_histogram_constructor_->EnsureFullView("a split view the compact matrix cannot serve");
  if (gather_src == nullptr) {
    Log::Fatal("BuildCompactColumnView: the row-major bin matrix has no device pointer");
  }
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
  // cuda_plan key async_tree_start: no host wait (the live fill may still be running); the sources are local, so
  // their bytes are kept until the tree end
  tree_start_uploads_.Upload<size_t>(cuda_slot_p_byte_.RawData(), slot_p_byte_h.data(), num_compact_cols);
  tree_start_uploads_.Upload<int>(cuda_slot_p_stride_.RawData(), slot_p_stride_h.data(), num_compact_cols);
  tree_start_uploads_.Upload<int>(cuda_slot_col_in_p_.RawData(), slot_col_in_p_h.data(), num_compact_cols);

  compact_packed_view_active_ = false;
  if (gather_src_is_4bit && PackedSplitReadUsable()) {
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
    const bool colmajor_split =
      FalcataPlan::Get().colmajor_split && cuda_histogram_constructor_->colmajor_bin() != nullptr;
    col_data->SetCompactPackedColumnView(orig_column_to_compact_slot_, gather_src,
                                         col_base_byte_h, slot_p_stride_h, col_shift_h,
                                         colmajor_split ? cuda_histogram_constructor_->colmajor_bin() : nullptr,
                                         colmajor_split ? cuda_histogram_constructor_->colmajor_pad() : 0);
    compact_packed_view_active_ = true;
    compact_gather_src_ = gather_src;
    compact_gather_src_is_4bit_ = gather_src_is_4bit;
    compact_col_signature_ = sig;
    col_view_generation_seen_ = col_data->column_view_generation();
    return;
  }

  // The gather is the buffer's only consumer on this path, so this is where it
  // gets allocated (moved down from the top of the function).
  EnsureCompactColumnBuffer(needed_bytes, num_compact_cols, num_data);
  if (compact_src) {
    // cuda_plan key pair_code5: the compact view's nibbles may not be written yet
    cuda_histogram_constructor_->EnsureNibbleView();
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
  if (compact_gather_src_ == cuda_histogram_constructor_->compact_data_device()) {
    // cuda_plan key pair_code5: the compact view's nibbles may not be written yet
    cuda_histogram_constructor_->EnsureNibbleView();
  }
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
  CUDAColumnData* col_data = ColumnView();
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

void CUDASingleGPUTreeLearner::RestoreBagOrder() {
  cuda_data_partition_->RestoreBagOrder();
}

bool CUDASingleGPUTreeLearner::HybridGrowthUsable() const {
  // Insufficient fixed shared memory: the existing classic per-pair kernels are deterministic for every layout.
  if (cuda_histogram_constructor_ != nullptr && cuda_histogram_constructor_->NonquantClassicFallback()) {
    return false;
  }
  // Plain level batching is exactly leaf-wise-equivalent only in the
  // depth-limited regime (2^max_depth <= num_leaves + 1): there the budget can
  // only bind at the final (max-depth) level, where the batched top-K-by-gain
  // selection is optimal because no deeper candidates can exist. In
  // budget-limited configs (num_leaves << 2^max_depth) the SELECTIVE
  // grow-then-prune mode (UseSelectiveGrowth) provides exact leaf-wise
  // equivalence; where it does not apply, the classic one-split-at-a-time loop
  // runs.
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
  // Multi-GPU rides the hybrid TWO-SYNC batched flow only: that is the flow
  // whose per-level structure allows ONE grouped all-reduce per level instead
  // of one per split, which is the dominant multi-GPU cost. The one-sync,
  // device tree and selective flows derive leaf counts on-device or reorder the
  // readback, and their rank-visible decisions have not been made global, so
  // they stay single-GPU (see UseOneSyncPrefix / NonquantDeviceTreeShape /
  // UseSelectiveGrowth).
  const bool nccl_ok = nccl_communicator_ == nullptr ||
      (cuda_histogram_constructor_ != nullptr && cuda_best_split_finder_ != nullptr &&
       cuda_histogram_constructor_->SupportsBatchedLevel() &&
       cuda_best_split_finder_->SupportsBatchedLevel());
  // Vector-leaf mode rides the two-sync level flows (exact-fit prefix and
  // selective grow-then-prune) with the batched level kernels (per-plane
  // construct/fix/subtract + the batched vector find) and the batched apply,
  // which is where the per-level plane fan-out lives.
  const bool vec_ok = vec_num_targets_ <= 1 ||
      (cuda_histogram_constructor_ != nullptr && cuda_best_split_finder_ != nullptr &&
       cuda_histogram_constructor_->SupportsBatchedLevel() &&
       cuda_best_split_finder_->SupportsBatchedLevel());
  const bool base = use_hybrid_growth_ &&
         nccl_ok &&
         vec_ok &&
         (!select_features_by_node_ || LevelFlowsTakeNodeMasks()) &&
         (cuda_best_split_finder_ == nullptr || !cuda_best_split_finder_->use_global_memory()) &&
         (forced_split_json_ == nullptr || forced_split_json_->is_null()) &&
         config_->num_leaves > 2;
  if (!base) {
    return false;
  }
  return depth_limited || UseSelectiveGrowth();
}

void CUDASingleGPUTreeLearner::CheckMultiGPUFlowUsable(int num_splits_done) const {
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
}

bool CUDASingleGPUTreeLearner::UseSelectiveGrowth() const {
  // Insufficient fixed shared memory: the existing classic per-pair kernels are deterministic for every layout.
  if (cuda_histogram_constructor_ != nullptr && cuda_histogram_constructor_->NonquantClassicFallback()) {
    return false;
  }
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
  if (depth_limited) {
    // depth-limited configs keep the (bit-exact-locked) exact-fit prefix
    return false;
  }
  // Selective growth runs on the batched apply phase (host-assigned, recyclable right-child leaf indices). Linear
  // trees renumber leaves through their own device paths, so they keep the classic loop.
  return !config_->linear_tree &&
         nccl_communicator_ == nullptr &&
         (!select_features_by_node_ || LevelFlowsTakeNodeMasks()) &&
         (forced_split_json_ == nullptr || forced_split_json_->is_null()) &&
         config_->num_leaves > 2;
}

bool CUDASingleGPUTreeLearner::LevelFlowsTakeNodeMasks() const {
  // the per-pair search fallback of the level loops keeps one mask pair on the device (the per-split loop's), so
  // per-node masks ride the level flows only with the batched level kernels; single GPU only (the level flows' node
  // masks are not striped for feature-parallel and are not part of the multi-GPU two-sync flow)
  return bynode_level_masks_ && nccl_communicator_ == nullptr && fp_merge_state_ == nullptr &&
         vec_num_targets_ <= 1 &&
         cuda_histogram_constructor_ != nullptr && cuda_best_split_finder_ != nullptr &&
         cuda_histogram_constructor_->SupportsBatchedLevel() && cuda_best_split_finder_->SupportsBatchedLevel();
}

// cuda_plan key leaf_map_no_zero: the flows whose tree end writes every in-bag row's leaf before the map's first
// reader: Train's two-sync level prefix (no forced splits, so the prefix starts at the root; not the selective flow)
// with its batched apply, whose tree end materializes the map from the final windows (every leaf: the final level's
// split-inner and the residual pass; a leaf-wise tail's classic splits write their rows inline, its leaves' windows
// are rewritten by the residual pass), or, when the prefix splits nothing, the stump pass in Train. Single GPU,
// quantized training, no linear trees (the full-precision flows' and linear trees' models differ run to run with or
// without the key, so no md5 check can confirm them; they keep the zero store).
bool CUDASingleGPUTreeLearner::LeafMapZeroDead() const {
  return FalcataPlan::Get().leaf_map_no_zero && config_->use_quantized_grad && !config_->linear_tree &&
         nccl_communicator_ == nullptr &&
         fp_merge_state_ == nullptr && (forced_split_json_ == nullptr || forced_split_json_->is_null()) &&
         HybridGrowthUsable() && !UseSelectiveGrowth();
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
         cuda_histogram_constructor_->SupportsBatchedLevel() &&
         cuda_best_split_finder_->SupportsBatchedLevel();
}

bool CUDASingleGPUTreeLearner::NonquantDeviceTreeShape() const {
  // the configurations the non-quantized graph loop (L1.5, removed in 1.2) served: depth-limited (the device select
  // applies a binding final level's top-K in the host arbitration order), single GPU, scalar leaves, no categorical
  // features, no per-node masks, no debug hooks, not a forest slot; and its shape bound, where the device levels'
  // fixed-point construct trains its models (cuda_device_levels.hpp)
  if (config_->use_quantized_grad || nccl_communicator_ != nullptr || fp_merge_state_ != nullptr ||
      select_features_by_node_ || vec_num_targets_ > 1 || has_categorical_feature_ || ForestSlotShared() ||
      forest_slot_ || FalcataDebug().debug || FalcataDebug().dump) {
    return false;
  }
  // insufficient shared memory for the fixed-point table: the classic loop's deterministic per-pair kernels
  if (cuda_histogram_constructor_ != nullptr && cuda_histogram_constructor_->NonquantClassicFallback()) {
    return false;
  }
  const FalcataPlan& plan = FalcataPlan::Get();
  const int depth = config_->max_depth;
  return plan.device_levels && plan.device_tree && depth >= 1 && depth <= kNonquantDeviceTreeMaxDepth &&
    (1LL << depth) <= static_cast<int64_t>(config_->num_leaves) + 1 &&
    config_->num_leaves <= kNonquantDeviceTreeMaxLeaves;
}

bool CUDASingleGPUTreeLearner::NonquantDeviceTreeUsable() const {
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  return NonquantDeviceTreeShape() && UseOneSyncPrefix() && DeviceTreeRequested() && !device_levels_disabled_ &&
    !device_tree_scratch_refused_ &&
    cuda_best_split_finder_->num_used_tasks() > 0;
#else
  return false;
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
}

bool CUDASingleGPUTreeLearner::EnsureReadbackStaging(const size_t bytes) {
  if (!FalcataPlan::Get().readback_kernel) {
    return false;
  }
  if (readback_staging_bytes_ < bytes) {
    if (readback_staging_ != nullptr) {
      CUDASUCCESS_OR_FATAL(cudaFreeHost(readback_staging_));
    }
    // sized for the largest tree-end readback up front (the tree's pooled slab), so it is allocated once
    const size_t capacity = std::max(bytes, CUDATree::PooledDeviceBufferSize(config_->num_leaves));
    AllocatePinnedMappedHost(&readback_staging_, capacity, __FILE__, __LINE__);
    readback_staging_device_ = MappedHostDevicePointer(readback_staging_);
    readback_staging_bytes_ = capacity;
  }
  return readback_staging_device_ != nullptr;
}

void CUDASingleGPUTreeLearner::EnsureRootSumsReadBack(CUDATree* tree) {
  if (!root_sums_deferred_) {
    return;
  }
  root_sums_deferred_ = false;
  if (EnsureReadbackStaging(2 * sizeof(double))) {
    // cuda_plan key readback_kernel: both sums through the mapped pinned staging, one synchronize
    cuda_smaller_leaf_splits_->CopyRootSumsToHost(&leaf_sum_gradients_[0], &leaf_sum_hessians_[0],
                                                  readback_staging_, readback_staging_device_);
  } else {
    cuda_smaller_leaf_splits_->CopyRootSumsToHost(&leaf_sum_gradients_[0], &leaf_sum_hessians_[0]);
  }
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
  CHECK(!whole_host_leaf_cache_stale_);
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

  SelectFeatureByNode(tree, smaller_leaf_index, larger_leaf_index);
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
  if (has_categorical_feature_ && config_->use_quantized_grad) {
    Log::Fatal("tree_mode=vector_leaf multi-target training does not support "
               "categorical features under quantized training yet (the "
               "quantized vector search covers numerical tasks only). Set "
               "quant_mode=none.");
  }
  if (config_->use_quantized_grad && config_->quant_train_renew_leaf) {
    // the renewal reduces ONE exact gradient stream per leaf and overwrites a
    // scalar leaf value; a vector leaf carries T of them
    Log::Fatal("tree_mode=vector_leaf multi-target training does not support "
               "quant_train_renew_leaf yet.");
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
    if (config_->use_quantized_grad) {
      cuda_smaller_leaf_splits_->InitValues(
        config_->lambda_l1,
        config_->lambda_l2,
        config_->max_delta_step,
        reinterpret_cast<const int16_t*>(
          cuda_gradient_discretizer_->discretized_gradients_and_hessians(t)),
        leaf_splits_init_indices,
        cuda_data_partition_->cuda_data_indices(),
        root_num_data,
        cuda_histogram_constructor_->cuda_hist_pointer() +
          static_cast<size_t>(t) * 2 * num_total_bin_,
        &vec_root_sum_gradients_[t],
        &vec_root_sum_hessians_[t],
        cuda_gradient_discretizer_->grad_scale_ptr(t),
        cuda_gradient_discretizer_->hess_scale_ptr());
    } else {
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
    }
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
  CUDABestSplitFinder::VectorQuantArgs quant;
  uint8_t parent_num_bits = 0, smaller_num_bits = 0, larger_num_bits = 0;
  if (config_->use_quantized_grad) {
    quant.grad_scales = cuda_gradient_discretizer_->grad_scale_ptr(0);
    quant.hess_scale = cuda_gradient_discretizer_->hess_scale_ptr();
    // bit widths are a function of the leaf's row count, which every plane
    // shares, so one lookup covers all T planes (single-GPU: vector mode is
    // fenced to one rank, so the local counts are the global ones)
    if (larger_leaf_index >= 0) {
      const int parent_leaf_index = std::min(smaller_leaf_index, larger_leaf_index);
      parent_num_bits = cuda_gradient_discretizer_->GetHistBitsInNode<false>(parent_leaf_index);
      smaller_num_bits = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(smaller_leaf_index);
      larger_num_bits = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(larger_leaf_index);
    } else {
      parent_num_bits = cuda_gradient_discretizer_->GetHistBitsInLeaf<false>(0);
      smaller_num_bits = parent_num_bits;
      larger_num_bits = parent_num_bits;
    }
    quant.smaller_num_bits = smaller_num_bits;
    quant.larger_num_bits = larger_num_bits;
  }
  global_timer.Start("CUDASingleGPUTreeLearner::ConstructHistogramForLeaf");
  for (int t = 0; t < num_targets; ++t) {
    // planes 1..T-1 accumulate gradients only: the hessian is target-independent
    // and the finder reads it from plane 0, so their hessian cells stay zero.
    // (Quantized planes pack gradient and hessian into one word per row, so
    // there the flag is inert -- plane t already carries plane 0's hessians.)
    if (config_->use_quantized_grad) {
      cuda_histogram_constructor_->SelectGradientPlane(
        reinterpret_cast<const score_t*>(
          cuda_gradient_discretizer_->discretized_gradients_and_hessians(t)),
        nullptr, /*grad_only=*/t > 0);
    } else {
      cuda_histogram_constructor_->SelectGradientPlane(
        gradients_ + static_cast<size_t>(t) * num_data_,
        hessians_ + static_cast<size_t>(t) * num_data_,
        /*grad_only=*/t > 0);
    }
    cuda_histogram_constructor_->ConstructHistogramForLeaf(
      planes + t,
      planes + num_targets + t,
      num_data_in_smaller_leaf,
      num_data_in_larger_leaf,
      num_data_in_smaller_leaf,
      num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf,
      sum_hessians_in_larger_leaf,
      smaller_num_bits);
    cuda_histogram_constructor_->SubtractHistogramForLeaf(
      planes + t,
      planes + num_targets + t,
      config_->use_quantized_grad, parent_num_bits, smaller_num_bits, larger_num_bits);
  }
  if (config_->use_quantized_grad) {
    cuda_histogram_constructor_->SelectGradientPlane(
      reinterpret_cast<const score_t*>(
        cuda_gradient_discretizer_->discretized_gradients_and_hessians(0)),
      nullptr);
  } else {
    cuda_histogram_constructor_->SelectGradientPlane(gradients_, hessians_);
  }
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
    quant,
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
      // planes 1..T-1 accumulate gradients only: the hessian is
      // target-independent and the finder reads it from plane 0. (Quantized
      // planes pack gradient and hessian into one word per row, so there the
      // flag is inert -- plane t already carries plane 0's hessians.)
      if (config_->use_quantized_grad) {
        cuda_histogram_constructor_->SelectGradientPlane(
          reinterpret_cast<const score_t*>(
            cuda_gradient_discretizer_->discretized_gradients_and_hessians(t)),
          nullptr, /*grad_only=*/t > 0);
      } else {
        cuda_histogram_constructor_->SelectGradientPlane(
          gradients_ + static_cast<size_t>(t) * num_data_,
          hessians_ + static_cast<size_t>(t) * num_data_,
          /*grad_only=*/t > 0);
      }
      cuda_histogram_constructor_->ConstructHistogramForLeaf(
        smaller_planes + t, larger_planes + t,
        desc.num_data_in_smaller_leaf, desc.num_data_in_larger_leaf,
        desc.num_data_in_smaller_leaf, desc.num_data_in_larger_leaf,
        sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf,
        desc.smaller_num_bits);
      cuda_histogram_constructor_->SubtractHistogramForLeaf(
        smaller_planes + t, larger_planes + t, config_->use_quantized_grad,
        desc.parent_num_bits, desc.smaller_num_bits, desc.larger_num_bits);
    }
  }
  CUDABestSplitFinder::VectorQuantArgs quant;
  if (config_->use_quantized_grad) {
    quant.grad_scales = cuda_gradient_discretizer_->grad_scale_ptr(0);
    quant.hess_scale = cuda_gradient_discretizer_->hess_scale_ptr();
    cuda_histogram_constructor_->SelectGradientPlane(
      reinterpret_cast<const score_t*>(
        cuda_gradient_discretizer_->discretized_gradients_and_hessians(0)),
      nullptr);
  } else {
    cuda_histogram_constructor_->SelectGradientPlane(gradients_, hessians_);
  }
  cuda_best_split_finder_->FindBestSplitsForLevelVector(
    cuda_hybrid_pair_descs_.RawDataReadOnly(), num_pairs, plane_slab, quant);
}

void CUDASingleGPUTreeLearner::WaitForInFlightApply() {
  if (!cuda_data_partition_->ApplyInFlight()) {
    return;
  }
  // cuda_plan key split_counts_early: the previous level's readback waited only for its aggregate kernel, so two
  // orderings its device synchronize gave are restored by events here. (1) The histogram stream (construct, fix,
  // subtract; the level's find and sync kernels wait on its subtract_done event) waits for the partition kernels
  // (apply_done_event, recorded after the level's last apply kernel). (2) The best-split finder's stream waits for the
  // previous level's tree record (tree_record_done_event_, recorded on the tree stream after its last SplitBatch /
  // SplitCategorical): the record reads the split leaves' cuda_leaf_best_split_info_ entries, which this level's
  // sync kernel overwrites for the same left-child leaf indices.
  CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_histogram_constructor_->hist_stream(),
                                           cuda_data_partition_->apply_done_event(), 0));
  CHECK(tree_record_done_event_ != nullptr);
  cuda_best_split_finder_->WaitOnCatSlabConsumers(nullptr, tree_record_done_event_);
  cuda_data_partition_->ClearApplyInFlight();
}

void CUDASingleGPUTreeLearner::RecordTreeRecordDone(const CUDATree* tree) {
  if (tree_record_done_event_ == nullptr) {
    CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&tree_record_done_event_, cudaEventDisableTiming));
  }
  CUDASUCCESS_OR_FATAL(cudaEventRecord(tree_record_done_event_, tree->tree_stream()));
}

void CUDASingleGPUTreeLearner::EnqueueLevelBestSplitSearch(const CUDATree* tree,
    const std::vector<HybridPendingPair>& pairs,
    CUDALeafSplitsStruct* vec_plane_slab) {
  CHECK(!whole_host_leaf_cache_stale_);
  const int num_pairs = static_cast<int>(pairs.size());
  if (num_pairs <= 0) {
    // nothing to search, but an in-flight apply is still consumed here (a non-final level always yields pairs today,
    // so this only keeps apply_in_flight_ from outliving the level if that changes)
    WaitForInFlightApply();
    return;
  }
  global_timer.Start("CUDASingleGPUTreeLearner::EnqueueLevelBestSplitSearch");
  host_hybrid_pair_descs_.resize(static_cast<size_t>(num_pairs));
  data_size_t max_num_data_in_smaller_leaf = 0;
  bool any_bit_change_copy = false;
  const bool skip_unsplittable = FalcataPlan::Get().skip_unsplittable && config_->forcedsplits_filename.empty() &&
    (forced_split_json_ == nullptr || forced_split_json_->is_null());
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
    // cuda_plan key skip_unsplittable: a leaf of n rows cannot be split when n + 2 + n / 2^20 < 2 * min_data_in_leaf:
    // both children need min_data_in_leaf rows, and a threshold's left and right counts sum to n (one is n minus the
    // other) or, rounded separately from hessian sums, to at most n + 1 plus the floating-point error of those sums
    // (the n / 2^20 margin covers fp32 gains). Such a leaf gets no split search and, when its partner is also
    // unsplittable, no histogram.
    const auto unsplittable_by_count = [&](const data_size_t n) {
      const int64_t n64 = static_cast<int64_t>(n);
      return skip_unsplittable && n64 + 2 + (n64 >> 20) < 2 * static_cast<int64_t>(config_->min_data_in_leaf);
    };
    const bool smaller_unsplittable = unsplittable_by_count(num_data_in_smaller_leaf);
    const bool larger_unsplittable = unsplittable_by_count(num_data_in_larger_leaf);
    // mirror of ConstructHistogramForLeaf's min_data/min_hessian early return
    desc.construct_valid =
      ((num_data_in_smaller_leaf <= config_->min_data_in_leaf || smaller_unsplittable ||
        sum_hessians_in_smaller_leaf <= config_->min_sum_hessian_in_leaf) &&
       (num_data_in_larger_leaf <= config_->min_data_in_leaf || larger_unsplittable ||
        sum_hessians_in_larger_leaf <= config_->min_sum_hessian_in_leaf)) ? 0 : 1;
    // mirror of CUDABestSplitFinder::FindBestSplitsForLeaf's leaf validity checks
    const bool smaller_below_max_depth =
      config_->max_depth <= 0 || GrowthLeafDepth(tree, pair.smaller) < config_->max_depth;
    const bool larger_below_max_depth = pair.larger < 0 ||
      config_->max_depth <= 0 || GrowthLeafDepth(tree, pair.larger) < config_->max_depth;
    desc.smaller_valid = (num_data_in_smaller_leaf > config_->min_data_in_leaf && !smaller_unsplittable &&
                          sum_hessians_in_smaller_leaf > config_->min_sum_hessian_in_leaf &&
                          smaller_below_max_depth) ? 1 : 0;
    desc.larger_valid = (pair.larger >= 0 &&
                         num_data_in_larger_leaf > config_->min_data_in_leaf && !larger_unsplittable &&
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
  // subtract_done_events_[0]. Reusing host_hybrid_pair_descs_ next level is
  // safe because it is a pageable std::vector: a pageable async H2D copies it
  // into the driver's staging before it returns. No full sync is needed for
  // that (under split_counts_early a non-final level's FinishSplitBatch waits
  // for the aggregate kernel only, with the partition still in flight); moving
  // this buffer to pinned memory would need an event wait before the rewrite.
  CopyFromHostToCUDADeviceAsync<CUDAHybridPairDescriptor>(
    cuda_hybrid_pair_descs_.RawData(), host_hybrid_pair_descs_.data(),
    static_cast<size_t>(num_pairs), cuda_histogram_constructor_->hist_stream(),
    __FILE__, __LINE__);
  if (bynode_level_masks_) {
    // cuda_plan key level_bynode_masks: one mask row per leaf the level searches (the host-valid roles; the finder
    // returns before the mask for the others), in pair order, smaller role first
    level_search_leaves_.clear();
    for (const CUDAHybridPairDescriptor& desc : host_hybrid_pair_descs_) {
      if (desc.smaller_valid) {
        level_search_leaves_.push_back(desc.smaller_leaf_index);
      }
      if (desc.larger_valid) {
        level_search_leaves_.push_back(desc.larger_leaf_index);
      }
    }
    UploadLevelNodeMasks(level_search_leaves_);
  }
  // the previous level's partition and tree record, where its readback did not wait for them (split_counts_early);
  // after the descriptor upload above, which reads no partition output and so goes ahead of the wait
  WaitForInFlightApply();
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
  // the root level reuses the root histogram this tree's compact fill accumulated, if it did
  const bool use_fused_root = num_pairs == 1 && pairs[0].smaller == 0 && pairs[0].larger < 0 &&
    nccl_communicator_ == nullptr && config_->use_quantized_grad &&
    host_hybrid_pair_descs_[0].num_data_in_smaller_leaf == num_data_ &&
    cuda_histogram_constructor_->ConsumeFusedRootHist(host_hybrid_pair_descs_[0].smaller_num_bits <= 16);
  // cuda_plan key pair_block_map: the host leaf counts are the device structs' (under NCCL the descriptors hold
  // global counts, so the constructor gets none)
  cuda_histogram_constructor_->SetLevelHostPairDescs(
    nccl_communicator_ == nullptr ? host_hybrid_pair_descs_.data() : nullptr, num_pairs);
  cuda_histogram_constructor_->ConstructHistogramsForLevel(
    cuda_hybrid_pair_descs_.RawDataReadOnly(), num_pairs,
    max_num_data_in_smaller_leaf, any_bit_change_copy,
    /*level_smaller_num_data=*/nullptr,
    /*defer_subtract=*/nccl_communicator_ != nullptr,
    use_fused_root);
  cuda_histogram_constructor_->SetLevelHostPairDescs(nullptr, 0);
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
    const std::vector<HybridAppliedSplit>& applied, cudaEvent_t tree_record_done) {
  CHECK(!whole_host_leaf_cache_stale_);
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
  if (bynode_level_masks_) {
    // cuda_plan key level_bynode_masks: both children of every applied split (the smaller/larger roles are decided
    // on the device; the finder reads the row of the leaf index the apply wrote into the struct)
    level_search_leaves_.clear();
    for (int i = 0; i < num_pairs; ++i) {
      if (host_hybrid_pair_descs_[i].smaller_valid) {
        level_search_leaves_.push_back(applied[i].left);
        level_search_leaves_.push_back(applied[i].right);
      }
    }
    UploadLevelNodeMasks(level_search_leaves_);
  }
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
  // the level's tree record (SplitBatchKernel on the tree stream) reads cuda_leaf_best_split_info_[parent], and the
  // sync kernel FindBestSplitsForLevel launches on the finder's stream overwrites that entry for the left child (the
  // parent's index); the cat_consumed_* events above are re-recorded only on categorical levels, and no device
  // synchronize separates the two in this flow, so the finder's stream waits for the record here
  cuda_best_split_finder_->WaitOnCatSlabConsumers(nullptr, tree_record_done);
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
  level_completes_tree_ = false;
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
    level_completes_tree_ = all_children_depth_capped;  // every candidate leaf is split (cuda_plan key skip_empty_tail)
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
  return true;
}

void CUDASingleGPUTreeLearner::ApplyLevelBatched(CUDATree* tree,
    const std::vector<int>& splittable, std::vector<HybridAppliedSplit>* applied,
    const bool final_level) {
  CHECK(!whole_host_leaf_cache_stale_);
  // one launch per kernel family for the whole level: batched tree-structure
  // update (CUDATree::SplitBatch) + batched partition (SplitLevelBatched);
  // right children take consecutive leaf indices exactly as the per-split
  // loop would assign them
  applied->clear();
  applied->reserve(splittable.size());
  const int base_num_leaves = tree->num_leaves();
  MaybeInjectTreeFailure(base_num_leaves);
  host_tree_batch_splits_.clear();
  host_apply_split_inputs_.clear();
  // cuda_plan key level_apply_first: without categorical splits or vector leaves the partition's apply goes out
  // before the tree's record (disjoint buffers, separate streams; see the key), and the record's entries (real
  // feature index and threshold, missing type: per-feature lookups only the record reads) are built after it
  bool apply_first = FalcataPlan::Get().level_apply_first && vec_num_targets_ <= 1;
  for (const int leaf : splittable) {
    if (host_leaf_best_splits_[leaf].num_cat_threshold > 0) {
      apply_first = false;
      break;
    }
  }
  // cuda_plan key final_map_only: a final level that completes the tree (the tail is known empty) with no reader of
  // its children's windows after it: the apply writes the map and leaves the windows unpartitioned (any reader
  // that turns up after all, e.g. an objective's leaf renewal, completes the partition first)
  const FalcataPlan& map_plan = FalcataPlan::Get();
  const bool map_only = final_level && level_completes_tree_ && map_plan.final_map_only && map_plan.skip_empty_tail &&
    !cuda_data_partition_->use_bagging() && !config_->linear_tree &&
    !(config_->use_quantized_grad && config_->quant_train_renew_leaf) && nccl_communicator_ == nullptr &&
    fp_merge_state_ == nullptr && vec_num_targets_ <= 1 && !UseSelectiveGrowth();
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
    // Categorical splits: the entry's real_threshold is never read (the
    // per-split loop below routes them through SplitCategorical; under
    // cuda_plan key cat_level_record SplitLevel reads the rest of the entry
    // but not the threshold), and RealThreshold() would OOB-index the
    // bin-upper-bound array with the categorical threshold payload -- do not
    // call it.
    const bool is_cat_split = host_leaf_best_splits_[leaf].num_cat_threshold > 0;
    if (!apply_first) {
      host_tree_batch_splits_.push_back({
        leaf,
        right_leaf_index,  // == tree num_leaves at the time of this split
        train_data_->RealFeatureIndex(inner_feature_index),
        is_cat_split ? 0.0 :
          train_data_->RealThreshold(inner_feature_index, leaf_best_split_threshold_[leaf]),
        static_cast<int>(train_data_->FeatureBinMapper(inner_feature_index)->missing_type()),
        best_split_info});
    }
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
    DebugPrintBynodeSplit(leaf, inner_feature_index);
    BynodeKeySplit(leaf, right_leaf_index);
  }
  // ZeroHistForLeaf is a no-op (BeforeTrain zeroes the full histogram
  // buffer), so the batched path skips the per-split calls entirely.
  //
  // Tree recording: numerical runs go through SplitBatch chunks; the level's
  // categorical splits (rare) interleave through the per-split categorical
  // path IN LEVEL ORDER, preserving the consecutive right-child numbering.
  // The data partition still applies EVERYTHING through the batched arena
  // path below.
  if (apply_first) {
    cuda_data_partition_->SplitLevelBatched(host_apply_split_inputs_, final_level, map_only);
    // the tree record's entries, in level order with the values the loop above would have given them (no split of
    // this level is categorical)
    for (const CUDAHybridApplySplitInput& in : host_apply_split_inputs_) {
      host_tree_batch_splits_.push_back({
        in.left_leaf_index,
        in.right_leaf_index,
        train_data_->RealFeatureIndex(in.split_feature),
        train_data_->RealThreshold(in.split_feature, in.split_threshold),
        static_cast<int>(train_data_->FeatureBinMapper(in.split_feature)->missing_type()),
        in.best_split_info});
    }
  }
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
    // cuda_plan key cat_level_record: the level's categorical splits are recorded by one kernel at device-summed
    // offsets (CUDATree::SplitLevel), so their lengths are never read back (vector leaves keep the per-split loop)
    const bool cat_level_record = num_cat_splits > 0 && FalcataPlan::Get().cat_level_record && vec_num_targets_ <= 1;
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
      if (!cat_level_record) {
        host_batch_cat_lens_.resize(2 * static_cast<size_t>(num_cat_splits));
        CopyFromCUDADeviceToHost<uint64_t>(host_batch_cat_lens_.data(),
          cuda_batch_cat_lens_.RawData(), 2 * static_cast<size_t>(num_cat_splits),
          __FILE__, __LINE__);
      }
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
    if (cat_level_record) {
      // the bitset kernel above (legacy stream) is ordered before SplitLevel's launches (blocking tree stream), and
      // the next level's bitset kernel after them, so the slab and the lengths are read before they are rewritten
      host_level_is_cat_.resize(host_tree_batch_splits_.size());
      for (size_t k = 0; k < host_tree_batch_splits_.size(); ++k) {
        host_level_is_cat_[k] = host_apply_split_inputs_[k].num_cat_threshold > 0 ? 1 : 0;
      }
      CUDATreeCatLevel cat_level;
      cat_level.bitset_slab = cuda_batch_cat_bitset_.RawDataReadOnly();
      cat_level.bitset_stride = batch_cat_bitset_stride_;
      cat_level.bitset_inner_slab = cuda_batch_cat_bitset_inner_.RawDataReadOnly();
      cat_level.bitset_inner_stride = batch_cat_bitset_inner_stride_;
      cat_level.lens = cuda_batch_cat_lens_.RawDataReadOnly();
      tree->SplitLevel(host_tree_batch_splits_, host_level_is_cat_, cat_level);
      cat_index = num_cat_splits;
    }
    for (size_t k = 0; k < host_tree_batch_splits_.size() && !cat_level_record; ++k) {
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
  if (!apply_first) {
    cuda_data_partition_->SplitLevelBatched(host_apply_split_inputs_, final_level, map_only);
  }
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
    // cuda_plan key early_leaf_map: the residual leaves' windows (in the main index array the apply kernels above
    // finalize, ordered before this default-stream launch) and their leaf indices are final now, and nothing up
    // to the tree end writes the map or those windows, so the map pass goes out here and runs while the host
    // reads the level back and finalizes the tree, instead of after those readbacks
    hybrid_map_early_written_ = false;
    hybrid_map_early_pending_ = false;
    if (FalcataPlan::Get().early_leaf_map && AsyncTreeStart()) {
      if (FalcataPlan::Get().final_readback_first) {
        // cuda_plan key final_readback_first: launched at the tree end right after the tree's device copy
        // (FlushEarlyLeafMap): here it would sit ahead of this level's split batch readback and the tree-end
        // readbacks on the default stream, each of which would wait for it
        hybrid_map_early_pending_ = true;
      } else {
        cuda_data_partition_->MaterializeHybridLeafMapSubset(hybrid_map_residual_leaves_);
        hybrid_map_early_written_ = true;
        hybrid_map_early_num_leaves_ = tree->num_leaves();
      }
    }
    if (FalcataPlan::Get().tree_end_prealloc && vec_num_targets_ <= 1) {
      // cuda_plan key tree_end_prealloc: the tree's leaf count after its final level; the allocation runs while
      // the GPU applies the level
      tree->PrepareRetainedLeafValues(tree->num_leaves());
    }
  }
}

void CUDASingleGPUTreeLearner::PrefetchTreeReadback(const CUDATree* tree) {
  // cuda_plan key final_readback_first: after a final level that completes the tree (every candidate split, the
  // leaf budget not binding, every child at max_depth; skip_empty_tail then runs no leaf-wise tail), nothing writes
  // the tree's arrays or the partition's leaf counts before the tree end reads them back (Train's early_to_host),
  // so those two copy kernels go out here, behind the level's apply and tree record and ahead of the level's own
  // split batch readback, whose default-stream synchronize then covers all three: one host round trip, not two
  prefetched_tree_num_leaves_ = 0;
  prefetched_tree_counts_ = false;
  if (!FalcataPlan::Get().final_readback_first || !FalcataPlan::Get().skip_empty_tail || !level_completes_tree_ ||
      !AsyncTreeStart() || (config_->use_quantized_grad && config_->quant_train_renew_leaf) ||
      config_->linear_tree || tree->num_leaves() <= 0) {
    return;
  }
  const size_t slab_bytes = CUDATree::PooledDeviceBufferSize(config_->num_leaves);
  const size_t counts_offset = (slab_bytes + 255) / 256 * 256;
  const size_t count_bytes = static_cast<size_t>(tree->num_leaves()) * sizeof(data_size_t);
  if (!EnsureReadbackStaging(counts_offset + count_bytes) ||
      !tree->LaunchSlabReadback(readback_staging_device_, readback_staging_bytes_)) {
    return;
  }
  prefetched_tree_num_leaves_ = tree->num_leaves();
  if (nccl_communicator_ == nullptr) {
    LaunchCopyToMappedHost(static_cast<uint8_t*>(readback_staging_device_) + counts_offset,
                           cuda_data_partition_->cuda_leaf_num_data(), count_bytes, __FILE__, __LINE__);
    prefetched_tree_counts_ = true;
  }
}

void CUDASingleGPUTreeLearner::FlushEarlyLeafMap(const CUDATree* tree) {
  // cuda_plan key final_readback_first: called at the tree end, after its last split, between the tree's device
  // copy and the host-side finish of ToHost: the residual leaves' windows and leaf indices are final (a leaf-wise
  // tail split since the final level updated them on the device, as the tree-end pass reads them), and the pass
  // reads none of the tree's arrays, so it writes what the tree-end pass would write while the host finishes
  if (!hybrid_map_early_pending_) {
    return;
  }
  hybrid_map_early_pending_ = false;
  cuda_data_partition_->MaterializeHybridLeafMapSubset(hybrid_map_residual_leaves_);
  hybrid_map_early_written_ = true;
  hybrid_map_early_num_leaves_ = tree->num_leaves();
}

void CUDASingleGPUTreeLearner::FinishLevelBookkeeping(
    const std::vector<HybridAppliedSplit>& applied,
    const std::vector<int>& batch_info,
    std::vector<HybridPendingPair>* next_pairs,
    int* num_splits) {
  CHECK_GE(batch_info.size(), 18 * applied.size());
  FinishLevelBookkeeping(applied, batch_info.data(), next_pairs, num_splits);
}

void CUDASingleGPUTreeLearner::FinishLevelBookkeeping(
    const std::vector<HybridAppliedSplit>& applied,
    const int* batch_info,
    std::vector<HybridPendingPair>* next_pairs,
    int* num_splits) {
  CHECK(!whole_host_leaf_cache_stale_);
  for (size_t k = 0; k < applied.size(); ++k) {
    const HybridAppliedSplit& a = applied[k];
    const int* info = batch_info + 18 * k;
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
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  // before SetupDeviceLevels sizes a complete tree's search scratch: its pair chunk
  PlanDeviceTreeChunks();
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
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
  // config is outside the batched kernels' support.
  const bool use_batched_level_kernels =
    cuda_histogram_constructor_->SupportsBatchedLevel() &&
    cuda_best_split_finder_->SupportsBatchedLevel();
  const bool hybrid_diag = FalcataDebug().diag;
  if (hybrid_diag) {
    static bool diag_logged = false;
    if (!diag_logged) {
      diag_logged = true;
      fprintf(stderr, "[hybrid-diag] prefix: batched_level_kernels=%d one_sync=%d\n",
                   static_cast<int>(use_batched_level_kernels),
                   static_cast<int>(UseOneSyncPrefix()));
      fprintf(stderr, "[hybrid-diag] %s\n", cuda_histogram_constructor_->BatchedLevelGateDiag().c_str());
      fprintf(stderr, "[hybrid-diag] %s\n", cuda_best_split_finder_->BatchedLevelGateDiag().c_str());
    }
  }
  // batched apply: all tree-structure + partition updates of a level in one
  // launch per kernel family with zero per-split host syncs (ApplyLevelBatched).
  // vector mode reaches the prefix only through the batched level kernels
  // (HybridGrowthUsable): the per-pair search fallback below is scalar-only
  CHECK(vec_num_targets_ <= 1 || use_batched_level_kernels);
  // single-sync (speculative) level pipeline: requires both batched phases and
  // non-quantized training (the quantized HOST path selects histogram kernels
  // host-side from per-leaf bit widths, which needs the classic readback
  // ordering).
  // (cuda_plan key device_tree: a non-quantized tree inside the device tree's bound takes the complete device tree)
  if (UseOneSyncPrefix() && !NonquantDeviceTreeUsable()) {
    growth_route_ = GrowthRoute::kOneSync;
    return TrainLevelWisePrefixOneSync(tree);
  }
  // The complete device path evaluates root validity from the device root struct. Keep root sums deferred there;
  // the classic host flow still needs them for its root descriptor. Capacity setup precedes all root enqueues.
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  const int device_levels = DeviceLevelsToRun(use_batched_level_kernels, /*batched_level_apply=*/true);
  if (device_levels > 0 && !device_levels_ready_ && !SetupDeviceLevels(device_levels)) {
    device_levels_disabled_ = true;
  }
  const bool whole_device_tree = DeviceTreeRequested() && device_levels > 0 && device_levels_ready_;
  if (UseOneSyncPrefix() && !whole_device_tree) {
    // a non-quantized device tree whose setup failed: the one-sync flow grows it
    growth_route_ = GrowthRoute::kOneSync;
    return TrainLevelWisePrefixOneSync(tree);
  }
  if (!whole_device_tree) {
    EnsureRootSumsReadBack(tree);
  }
#else
  EnsureRootSumsReadBack(tree);
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
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
  // cuda_plan key readback_fused: the level's two readbacks written by their producing kernels into the mapped
  // staging (single GPU: the host reads its own device; no feature-parallel merge rewriting the device cache; scalar
  // leaves; no compact prefill on the non-blocking side stream, which the device synchronize would wait for)
  const FalcataPlan& readback_plan = FalcataPlan::Get();
  const bool readback_fused = readback_plan.readback_fused && readback_plan.readback_kernel &&
    !readback_plan.compact_prefill && nccl_communicator_ == nullptr && fp_merge_state_ == nullptr &&
    vec_num_targets_ <= 1 && use_batched_level_kernels;
  // per leaf of this tree: the best-split staging holds the device cache's bytes (written by a mirrored level sync
  // kernel or a full copy of this tree's loop; nothing else writes those entries inside the loop)
  std::vector<char> leaf_staging_coherent;
  int num_splits = 0;
  // cuda_plan key device_levels: the levels whose decisions are all "split every valid leaf" run device-driven
  // after the root's search; their one readback leaves the host where its own readback of the next level would
  bool search_read_back = false;
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  if (device_levels > 0 && device_levels_ready_) {
    if (FalcataDebug().diag) {
      if (!device_levels_logged_) {
        device_levels_logged_ = true;
        fprintf(stderr, "[hybrid-diag] device_levels: %d device-driven levels per tree\n", device_levels);
      }
    }
    if (whole_device_tree) {
      // Prepare required descendant nibbles before the root completion event: their fill stays on the original
      // histogram stream, while device levels exchange that stream for the find stream. Fused root scratch is
      // independent; descendants that can read the code view keep it.
      if (!cuda_histogram_constructor_->DeviceLevelGraphCapturable()) {
        cuda_histogram_constructor_->EnsureNibbleView();
      }
      if (select_features_by_node_) {
        // cuda_plan key device_bynode_masks (DeviceLevelsToRun admitted the per-node masks only for it)
        BeginDeviceBynode();
      }
      EnqueueRootLevelSearchOneSync(/*reuse_fused_root=*/true);
    } else {
      EnqueueLevelBestSplitSearch(tree, pairs, vec_plane_slab);
    }
    num_splits = TrainDeviceLevels(tree, device_levels, &pairs);
    search_read_back = true;
    // the root tail continues in Train's leaf-wise loop
    growth_route_ = device_tree_completed_ ? GrowthRoute::kDeviceTree :
      device_tree_root_tail_ ? GrowthRoute::kClassic : GrowthRoute::kPrefix;
    if (device_tree_completed_ || device_tree_root_tail_) {
      return num_splits;
    }
  }
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  if (growth_route_ != GrowthRoute::kPrefix) {
    growth_route_ = GrowthRoute::kHostTwoSync;
  }
  while (true) {
    if (!search_read_back) {
      // enqueue histogram + best-split search for every pair of this level; device
      // work only, ordered per pair by the histogram-completion events
      if (readback_fused) {
        cuda_best_split_finder_->RequestLeafBestSplitMirror();
      }
      if (use_batched_level_kernels) {
        EnqueueLevelBestSplitSearch(tree, pairs, vec_plane_slab);
      } else {
        int pair_counter = 0;
        for (const HybridPendingPair& pair : pairs) {
          EnqueuePairBestSplitSearch(tree, pair.smaller_struct, pair.larger_struct,
                                     pair.smaller, pair.larger, /*synchronize=*/false,
                                     pair_counter % CUDAHistogramConstructor::kNumHistPipelines);
          ++pair_counter;
        }
      }
      // one device synchronization + one transfer for the whole level
      if (readback_fused) {
        const int level_num_leaves = tree->num_leaves();
        if (leaf_staging_coherent.size() < static_cast<size_t>(level_num_leaves)) {
          leaf_staging_coherent.resize(static_cast<size_t>(level_num_leaves), 0);
        }
        bool staging_coherent = cuda_best_split_finder_->LeafBestSplitMirrorWritten();
        if (staging_coherent) {
          // the level sync kernel wrote (and mirrored) both leaves of every pair
          for (const HybridPendingPair& pair : pairs) {
            leaf_staging_coherent[pair.smaller] = 1;
            if (pair.larger >= 0) {
              leaf_staging_coherent[pair.larger] = 1;
            }
          }
          for (int leaf = 0; staging_coherent && leaf < level_num_leaves; ++leaf) {
            staging_coherent = leaf_staging_coherent[leaf] != 0;
          }
        }
        cuda_best_split_finder_->SyncLevelLeafBestSplitsToHost(level_num_leaves, staging_coherent,
                                                               &host_leaf_best_splits_);
        // either way the staging now holds the device entries of leaves [0, level_num_leaves)
        std::fill(leaf_staging_coherent.begin(), leaf_staging_coherent.begin() + level_num_leaves, 1);
      } else {
        cuda_best_split_finder_->SyncAllLeafBestSplitsToHost(tree->num_leaves(), &host_leaf_best_splits_);
      }
    }
    search_read_back = false;
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
    const int leaves_before_level = tree->num_leaves();
    // apply the whole level without any device synchronization: every launch
    // parameter is host-known from the previous level, and split info is read
    // back once for the whole level below (FinishSplitBatch waits for it: a device
    // synchronize or the copy's stream sync, or, for a non-final level under
    // split_counts_early, only the aggregate kernel's event)
    std::vector<HybridAppliedSplit> applied;
    if (readback_fused) {
      // cuda_plan key split_counts_early: a level followed by another level's search (the readback's only consumer
      // before the next device synchronize is EnqueueLevelBestSplitSearch, which orders the histogram stream
      // behind the partition); the final level keeps the full synchronize (its tree readback prefetch and the
      // tree end rely on it)
      if (readback_plan.split_counts_early && !final_partial_level) {
        cuda_data_partition_->RequestSplitInfoEarly();
      } else {
        cuda_data_partition_->RequestSplitInfoMirror();
      }
    }
    ApplyLevelBatched(tree, splittable, &applied, final_partial_level);
    if (readback_fused && readback_plan.split_counts_early && !final_partial_level) {
      // cuda_plan key split_counts_early: every record kernel of the level is on the tree stream by now; the next
      // level's search orders the finder's stream behind this (EnqueueLevelBestSplitSearch)
      RecordTreeRecordDone(tree);
    }
    std::vector<int> batch_info;
    if (final_partial_level) {
      PrefetchTreeReadback(tree);
    }
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
      prefix_completes_tree_ = level_completes_tree_;
      prefix_final_leaves_before_ = leaves_before_level;
      break;
    }
  }
  return num_splits;
}

void CUDASingleGPUTreeLearner::EnqueueRootLevelSearchOneSync(const bool reuse_fused_root) {
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
  if (bynode_level_masks_ && desc.smaller_valid && !device_bynode_active_) {
    // cuda_plan key level_bynode_masks: the root's mask row (device_bynode_masks: drawn on the device instead)
    level_search_leaves_.assign(1, 0);
    UploadLevelNodeMasks(level_search_leaves_);
  }
  // The complete quantized path can consume the fill's root histogram just like the classic host root. Besides
  // avoiding a redundant construct, this preserves the code view that the device levels will capture/read.
  const bool use_fused_root = reuse_fused_root && config_->use_quantized_grad && leaf_num_data_[0] == num_data_ &&
    cuda_histogram_constructor_->ConsumeFusedRootHist(desc.smaller_num_bits <= 16);
  cuda_histogram_constructor_->ConstructHistogramsForLevel(
    cuda_hybrid_pair_descs_.RawDataReadOnly(), 1, leaf_num_data_[0],
    /*any_pair_needs_bit_change_copy=*/false, /*level_smaller_num_data=*/nullptr,
    /*defer_subtract=*/false, use_fused_root);
  cuda_best_split_finder_->WaitOnCatSlabConsumers(cat_consumed_default_event_, cat_consumed_tree_event_);
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  JoinDeviceBynode();  // cuda_plan key device_bynode_masks: the root's mask
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
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
    const int leaves_before_level = tree->num_leaves();
    ApplyLevelBatched(tree, splittable, &applied, final_partial_level);
    if (final_partial_level) {
      // the tree is full and every child sits at max_depth: nothing is left to
      // search, so read the apply info back immediately and stop
      cuda_data_partition_->FinishSplitBatch(static_cast<int>(applied.size()), &batch_info);
      FinishLevelBookkeeping(applied, batch_info, nullptr, &num_splits);
      prefix_completes_tree_ = level_completes_tree_;
      prefix_final_leaves_before_ = leaves_before_level;
      break;
    }
    // every record kernel of the level (SplitBatch, SplitCategorical) is on the
    // tree stream by now: the children's sync kernel waits for them
    RecordTreeRecordDone(tree);
    // speculative: the children's search goes out before their statistics are
    // host-known; the next iteration's readback collects both
    EnqueueLevelBestSplitSearchSpeculative(tree, applied, tree_record_done_event_);
    pending.swap(applied);
  }
  if (num_splits == 0) {
    // the root has no valid split: the leaf-wise tail (and a possible stump)
    // needs the host root sums after all
    EnsureRootSumsReadBack(tree);
  }
  return num_splits;
}

#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED

void CUDASingleGPUTreeLearner::ReleaseDeviceLevelGraphs() {
  // ResetConfig / ResetTrainingData can replace every pointer and static feature table captured by the prefix.
  device_level_graphs_.clear();
  device_levels_ready_ = false;
  device_levels_disabled_ = false;
  selective_prefix_logged_ = false;  // a new config reports its levels again
  device_tree_chunk_pairs_ = -1;
  device_tree_scratch_refused_ = false;
  device_level_graph_disabled_ = false;
  device_level_graph_hits_ = 0;
  device_level_graph_misses_ = 0;
  device_level_graph_update_failed_ = false;
  device_level_graph_update_hits_ = 0;
  device_level_graph_update_misses_ = 0;
  device_level_graph_update_last_refusal_ = -1;
}

void CUDASingleGPUTreeLearner::LogDeviceLevelsFallback(const DeviceTreeReason reason) {
  device_levels_refusal_ = reason;
  // a refused complete device tree reports on its reason line (LogDeviceTreeFallback)
  if (FalcataDebug().diag && !device_levels_fallback_logged_ && !DeviceTreeRequested()) {
    device_levels_fallback_logged_ = true;
    fprintf(stderr, "[hybrid-diag] device_levels: host fallback (%s)\n", DeviceTreeReasonToken(reason));
  }
}

DeviceTreeReason CUDASingleGPUTreeLearner::DeviceTreeRefusal() const {
  // The order of FALCATA_DEVICE_TREE_REASONS: the first refusal that holds names the tree's fallback. Each test is a
  // sufficient condition for the routing predicates (HybridGrowthUsable, DeviceTreeRequested, DeviceLevelsToRun) to
  // refuse the complete device tree; a tree none of them explains reports kUnclassified.
  using R = DeviceTreeReason;
  const FalcataPlan& plan = FalcataPlan::Get();
  const FalcataDebugOptions& debug = FalcataDebug();
  if (debug.debug || debug.dump || (debug.bynode && select_features_by_node_)) {
    return R::kDebugHook;
  }
  const bool bynode_unkeyed_by_key = config_->feature_fraction_bynode < 1.0 &&
    config_->interaction_constraints_vector.empty() && !plan.level_bynode_masks;
  if (!plan.hybrid || !plan.device_levels || !plan.device_tree || bynode_unkeyed_by_key ||
      (bynode_keyed_ && !plan.device_bynode_masks) || (!config_->use_quantized_grad && !use_hybrid_one_sync_)) {
    return R::kPlanOff;
  }
  if (nccl_communicator_ != nullptr) {
    return R::kMultiGpu;
  }
  if (fp_merge_state_ != nullptr) {
    return R::kFeatureParallel;
  }
  if (vec_num_targets_ > 1) {
    return R::kVectorLeaves;
  }
  if (ForestSlotShared()) {
    return R::kForestSlots;
  }
  const R layout = cuda_histogram_constructor_->BatchedLevelRefusal();
  const R finder = cuda_best_split_finder_->BatchedLevelRefusal();
  if (layout == R::kNoSharedMemory) {
    return R::kNoSharedMemory;
  }
  if (finder == R::kNonquantGt256Bins) {
    return R::kNonquantGt256Bins;
  }
  if (layout == R::kSparseLayout || layout == R::kEmptyTaskList) {
    return layout;
  }
  if (finder == R::kExtraTrees || finder == R::kCatRandomSearch) {
    return finder;
  }
  if (!config_->interaction_constraints_vector.empty()) {
    return R::kInteractionConstraints;
  }
  const int depth = config_->max_depth;
  const bool depth_limited = depth > 0 && depth < 31 && (1LL << depth) <= static_cast<int64_t>(config_->num_leaves) + 1;
  if (!config_->monotone_constraints.empty()) {
    return R::kMonotone;
  }
  if (CostEfficientGradientBoosting::IsEnable(config_)) {
    return R::kCegb;
  }
  if ((forced_split_json_ != nullptr && !forced_split_json_->is_null()) || !config_->forcedsplits_filename.empty()) {
    return R::kForcedSplits;
  }
  if (config_->linear_tree && !depth_limited) {
    return R::kLinearBudget;
  }
  if (config_->num_leaves <= 2) {
    return R::kNumLeavesLe2;
  }
  if (!config_->use_quantized_grad) {
    if (!depth_limited) {
      return R::kNonquantSelective;
    }
    // the complete device tree's non-quantized bound (NonquantDeviceTreeShape)
    if (depth > kNonquantDeviceTreeMaxDepth || config_->num_leaves > kNonquantDeviceTreeMaxLeaves) {
      return R::kNonquantOnesync;
    }
  }
  if (finder != R::kNone) {
    return finder;  // lambda_l1, path_smooth
  }
  if (layout != R::kNone) {
    return layout;  // large_bin_layout, compact_16bit
  }
  if (has_categorical_feature_) {
    return R::kCategorical;
  }
  if (!config_->use_quantized_grad) {
    // NonquantDeviceTreeShape's own refusals: per-node masks (drawn on the host), a forest slot's tree
    if (select_features_by_node_) {
      return R::kBynodeHostMasks;
    }
    if (forest_slot_) {
      return R::kForestSlots;
    }
  }
  if (select_features_by_node_ && !depth_limited) {
    return R::kBynodeOutsideBound;
  }
  if (select_features_by_node_ && !DeviceBynodeMasksUsable()) {
    return R::kBynodeHostMasks;
  }
  if (depth <= 0) {
    return R::kSelectiveUnbounded;
  }
  if (!depth_limited) {
    return R::kSelectiveBudget;
  }
  if (device_tree_scratch_refused_) {
    return R::kScratchTooLarge;
  }
  if (device_levels_disabled_) {
    return R::kSetupFailed;
  }
  if (cuda_best_split_finder_->num_used_tasks() == 0) {
    return R::kEmptyTaskList;
  }
  if (device_tree_root_tail_) {
    return R::kRootTailParity;
  }
  return R::kUnclassified;
}

void CUDASingleGPUTreeLearner::LogDeviceTreeFallback() {
  if (device_tree_completed_) {
    return;  // its `device_tree: complete` line
  }
  const DeviceTreeReason reason = DeviceTreeRefusal();
  const GrowthRoute route = ForestSlotShared() ? GrowthRoute::kForestSlots : growth_route_;
  std::string line = std::string("[hybrid-diag] device_tree: fallback (") + DeviceTreeReasonToken(reason) +
    ") route=" + GrowthRouteToken(route);
  if ((route == GrowthRoute::kHostTwoSync || route == GrowthRoute::kSelectiveTwoSync) &&
      device_levels_refusal_ != DeviceTreeReason::kNone && device_levels_refusal_ != reason) {
    line += std::string(" prefix=") + DeviceTreeReasonToken(device_levels_refusal_);
  }
  if (line != last_device_tree_fallback_line_) {
    fprintf(stderr, "%s\n", line.c_str());
    last_device_tree_fallback_line_ = line;
  }
}

// ---- cuda_plan key device_bynode_masks (see CUDADeviceBynodeConfig) ----

bool CUDASingleGPUTreeLearner::DeviceBynodeMasksUsable() const {
  // a forest slot beside other slot threads keeps the host draw; a group's only tree (forest_single_job_device) runs
  // alone and takes the device masks (same keys: the trainer gives it the sequential tree counters)
  return FalcataPlan::Get().device_bynode_masks && bynode_keyed_ && LevelFlowsTakeNodeMasks() && !ForestSlotShared() &&
    !FalcataDebug().bynode &&
    // the kernels' static shared memory (the block scan's ~132 bytes) beside the dynamic bitmap: a 1 KB margin
    DeviceBynodeSharedBytes(train_data_->num_features()) + 1024 <= 48 * 1024;
}

void CUDASingleGPUTreeLearner::BeginDeviceBynode() {
  CHECK(!device_bynode_active_);
  const int num_features = train_data_->num_features();
  // a complete tree's leaf indices stay below 2^max_depth (DeviceTreeRequested: max_depth >= 1)
  CHECK_GE(config_->max_depth, 1);
  CHECK_LE(config_->max_depth, 30);
  const int slots = static_cast<int>(std::min<int64_t>(config_->num_leaves, int64_t{1} << config_->max_depth)) + 2;
  const ColSampler::KeyedNodeSample shape = col_sampler_.GetByNodeKeyedShape(&bynode_candidates_scratch_);
  CHECK_LE(shape.num_candidates, num_features);
  std::vector<int>& sample = bynode_sample_scratch_;
  sample.assign({shape.num_candidates, shape.num_sampled, shape.mode,
                 static_cast<int>(static_cast<uint32_t>(shape.seed_mix)),
                 static_cast<int>(static_cast<uint32_t>(shape.seed_mix >> 32))});
  sample.insert(sample.end(), bynode_candidates_scratch_.begin(), bynode_candidates_scratch_.end());
  cudaStream_t find_stream = cuda_best_split_finder_->find_stream();
  CUDADeviceBynodeConfig& cfg = device_bynode_cfg_;
  const int num_tasks = cuda_best_split_finder_->num_tasks();
  // pair chunking (PlanDeviceTreeChunks): rows and task lists for one chunk's children (row slot 0 the root, then
  // 1 + 2 k + side), the mask kernels pointing each leaf's table entry and list range at its slot
  const bool chunked = DeviceTreeChunks(device_level_max_pairs_) > 1;
  const int row_slots = chunked ? 1 + 2 * device_tree_chunk_pairs_ : slots;
  // host_bynode_sample_ holds the sample size the static tables were built for (its [1]) until a rebuild clears it
  const bool rebuild = cfg.rows == nullptr || cfg.num_slots != slots || cfg.num_features != num_features ||
    cfg.num_tasks != num_tasks || host_bynode_sample_.size() < 5 || host_bynode_sample_[1] != shape.num_sampled ||
    bynode_row_slots_ != row_slots || (cfg.table != nullptr) != chunked;
  if (rebuild) {
    std::vector<int> task_inner;
    int max_tasks = 0;
    cuda_best_split_finder_->NodeTaskTables(shape.num_sampled, &task_inner, &max_tasks);
    CHECK_LE(2 * static_cast<int64_t>(slots) + static_cast<int64_t>(row_slots) * std::max(1, max_tasks),
             static_cast<int64_t>(std::numeric_limits<int>::max()));
    // every buffer at its training-long size: the level kernels and the finder's captured launches keep them
    bynode_table_bytes_ = (static_cast<size_t>(slots) * sizeof(int8_t*) + 255) / 256 * 256;
    bynode_row_slots_ = row_slots;
    cuda_bynode_rows_.Resize(bynode_table_bytes_ + static_cast<size_t>(row_slots) * static_cast<size_t>(num_features));
    std::vector<int8_t> table(bynode_table_bytes_, 0);
    int8_t* rows = cuda_bynode_rows_.RawData() + bynode_table_bytes_;
    for (int leaf = 0; leaf < slots; ++leaf) {
      // chunked: every entry starts at the root's slot; the mask kernels repoint the leaves they draw
      int8_t* row = rows + static_cast<size_t>(chunked ? 0 : leaf) * static_cast<size_t>(num_features);
      std::memcpy(table.data() + static_cast<size_t>(leaf) * sizeof(int8_t*), &row, sizeof(int8_t*));
    }
    CopyFromHostToCUDADevice<int8_t>(cuda_bynode_rows_.RawData(), table.data(), table.size(), __FILE__, __LINE__);
    const size_t task_ints = 2 * static_cast<size_t>(slots) + static_cast<size_t>(row_slots) * std::max(1, max_tasks);
    cuda_bynode_tasks_.Resize(task_ints);
    std::vector<int> ranges(2 * static_cast<size_t>(slots), 0);
    for (int leaf = 0; leaf < slots; ++leaf) {
      // in int range: PlanDeviceTreeChunks caps the chunk (DeviceTreeBynodeChunkCap)
      const int64_t offset = 2 * static_cast<int64_t>(slots) + static_cast<int64_t>(chunked ? 0 : leaf) * max_tasks;
      CHECK_LE(offset, static_cast<int64_t>(std::numeric_limits<int>::max()));
      ranges[2 * leaf] = static_cast<int>(offset);
    }
    CopyFromHostToCUDADevice<int>(cuda_bynode_tasks_.RawData(), ranges.data(), ranges.size(), __FILE__, __LINE__);
    cuda_bynode_keys_.Resize(2 * static_cast<size_t>(slots));
    SetCUDAMemory<uint64_t>(cuda_bynode_keys_.RawData(), 0, 2 * static_cast<size_t>(slots), __FILE__, __LINE__);
    cuda_bynode_sample_.Resize(5 + static_cast<size_t>(num_features));
    cuda_bynode_task_inner_.Resize(static_cast<size_t>(std::max(1, num_tasks)));
    if (num_tasks > 0) {
      CopyFromHostToCUDADevice<int>(cuda_bynode_task_inner_.RawData(), task_inner.data(), task_inner.size(),
                                    __FILE__, __LINE__);
    }
    cfg.sample = cuda_bynode_sample_.RawDataReadOnly();
    cfg.task_inner_feature = cuda_bynode_task_inner_.RawDataReadOnly();
    cfg.num_tasks = num_tasks;
    cfg.num_features = num_features;
    cfg.num_slots = slots;
    cfg.max_tasks = max_tasks;
    cfg.rows = rows;
    cfg.tasks = cuda_bynode_tasks_.RawData();
    cfg.keys = cuda_bynode_keys_.RawData();
    cfg.table = chunked ? reinterpret_cast<int8_t**>(cuda_bynode_rows_.RawData()) : nullptr;
    host_bynode_sample_.clear();
    if (bynode_stream_ == nullptr) {
      CUDASUCCESS_OR_FATAL(cudaStreamCreateWithFlags(&bynode_stream_, cudaStreamNonBlocking));
      CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&bynode_fork_event_, cudaEventDisableTiming));
      CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&bynode_join_event_, cudaEventDisableTiming));
    }
    if (FalcataDebug().diag) {
      fprintf(stderr, "[bynode-diag] device masks: candidates=%d sampled=%d mode=%d tasks=%d max_tasks=%d slots=%d "
              "row_slots=%d\n", shape.num_candidates, shape.num_sampled, shape.mode, num_tasks, max_tasks, slots,
              row_slots);
    }
  }
  // the tree's candidates (a feature_fraction sample changes them every tree), ahead of the root's mask kernel
  if (sample != host_bynode_sample_) {
    host_bynode_sample_ = sample;
    CopyFromHostToCUDADeviceAsync<int>(cuda_bynode_sample_.RawData(), host_bynode_sample_.data(),
                                       host_bynode_sample_.size(), find_stream, __FILE__, __LINE__);
  }
  cuda_best_split_finder_->SetDeviceNodeMasks(
    reinterpret_cast<const int8_t* const*>(cuda_bynode_rows_.RawDataReadOnly()), cuda_bynode_tasks_.RawDataReadOnly(),
    cfg.max_tasks);
  device_bynode_active_ = true;
  EnqueueDeviceBynodeMasks(find_stream, -1, 1);
}

void CUDASingleGPUTreeLearner::EndDeviceBynode() {
  cuda_best_split_finder_->ClearDeviceNodeMasks();
  device_bynode_active_ = false;
  bynode_join_pending_ = false;
}

void CUDASingleGPUTreeLearner::EnqueueDeviceBynodeMasks(cudaStream_t stream, const int level, const int max_splits) {
  CHECK(!bynode_join_pending_);
  CUDASUCCESS_OR_FATAL(cudaEventRecord(bynode_fork_event_, stream));
  CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(bynode_stream_, bynode_fork_event_, 0));
  if (level < 0) {
    LaunchDeviceBynodeRootMask(device_bynode_cfg_, bynode_node_key_[0], bynode_stream_);
  } else {
    LaunchDeviceBynodeLevelMasks(device_bynode_cfg_, cuda_device_level_state_.RawDataReadOnly(),
                                 cuda_data_partition_->device_level_apply_descs(), level, max_splits,
                                 /*chunk_base=*/0, bynode_stream_);
  }
  CUDASUCCESS_OR_FATAL(cudaEventRecord(bynode_join_event_, bynode_stream_));
  bynode_join_pending_ = true;
}

void CUDASingleGPUTreeLearner::EnqueueDeviceBynodeChunkMasks(cudaStream_t stream, const int level,
                                                            const int chunk_base, const int count) {
  CHECK(!bynode_join_pending_);
  CHECK(device_bynode_cfg_.table != nullptr);
  LaunchDeviceBynodeLevelMasks(device_bynode_cfg_, cuda_device_level_state_.RawDataReadOnly(),
                               cuda_data_partition_->device_level_apply_descs(), level, count, chunk_base, stream);
}

void CUDASingleGPUTreeLearner::JoinDeviceBynode() {
  if (bynode_join_pending_) {
    cuda_best_split_finder_->WaitOnCatSlabConsumers(bynode_join_event_, nullptr);
    bynode_join_pending_ = false;
  }
}

// ---- cuda_plan key device_levels (see cuda_device_levels.hpp) ----

bool CUDASingleGPUTreeLearner::DeviceTreeRequested() const {
  // cuda_plan key device_tree (default on, with device_levels): only where the complete tree's depth bound holds, so
  // no other shape loses the device_levels prefix or the host flow. Match HybridGrowthUsable's depth-limited regime:
  // only the final depth-capped level can bind, and the device select applies its top-K candidates in the host
  // arbitration order. Earlier levels still split leaf-ascending. Any depth: a level wider than the shared select
  // selects over global scratch, and a search whose scratch does not fit the memory budget runs in pair chunks
  // (PlanDeviceTreeChunks).
  const FalcataPlan& plan = FalcataPlan::Get();
  const int depth = config_->max_depth;
  return plan.device_tree && plan.device_levels && depth >= 1 && depth < 31 &&
    (1LL << depth) <= static_cast<int64_t>(config_->num_leaves) + 1;
}

int CUDASingleGPUTreeLearner::DeviceLevelsToRun(const bool batched_level_kernels, const bool batched_level_apply) {
  const FalcataPlan& plan = FalcataPlan::Get();
  const bool device_tree = DeviceTreeRequested();
  if (!plan.device_levels || device_levels_disabled_) {
    return 0;
  }
  // Slot workers use thread-local stream ownership. Neither capture nor the plain device enqueue is supported here.
  // A group's only tree (cuda_plan key forest_single_job_device) runs alone, as the sequential learner's trees do.
  if (ForestSlotShared()) {
    LogDeviceLevelsFallback(DeviceTreeReason::kForestSlots);
    return 0;
  }
  // When sampled layouts keep missing the graph cache, use the existing host loop on subsequent trees.
  if (!device_tree && plan.device_level_graph && device_level_graph_disabled_) {
    LogDeviceLevelsFallback(DeviceTreeReason::kPrefixGraphChurn);
    return 0;
  }
  // the flow this replaces: the classic quantized prefix with both batched phases, single GPU, scalar leaves, no
  // per-node masks (drawn on the host per level), no categorical splits (their bitsets are built through host
  // readbacks), no debug hooks (they steer the host loop's decisions); the device counts reach only the fused
  // tree-structure + gap apply ids and the dense construct's block map
  // (cuda_plan key device_bynode_masks: the complete tree draws keyed per-node masks on the device)
  const bool node_masks_on_host = select_features_by_node_ && !(device_tree && DeviceBynodeMasksUsable());
  DeviceTreeReason refusal = DeviceTreeReason::kNone;
  if (FalcataDebug().debug || FalcataDebug().dump) {
    refusal = DeviceTreeReason::kDebugHook;
  } else if (nccl_communicator_ != nullptr) {
    refusal = DeviceTreeReason::kMultiGpu;
  } else if (fp_merge_state_ != nullptr) {
    refusal = DeviceTreeReason::kFeatureParallel;
  } else if (vec_num_targets_ > 1) {
    refusal = DeviceTreeReason::kVectorLeaves;
  } else if (!config_->use_quantized_grad && !NonquantDeviceTreeUsable()) {
    refusal = DeviceTreeRefusal();  // the non-quantized flavour
  } else if (!batched_level_kernels || !batched_level_apply ||
             !cuda_histogram_constructor_->DeviceLevelConstructSupported()) {
    refusal = DeviceTreeRefusal();  // which layout or finder option keeps the batched level kernels off
  } else if (has_categorical_feature_) {
    refusal = DeviceTreeReason::kCategorical;
  } else if (node_masks_on_host) {
    refusal = device_tree ? DeviceTreeReason::kBynodeHostMasks : DeviceTreeReason::kBynodeOutsideBound;
  } else if (cuda_best_split_finder_->num_used_tasks() == 0) {
    refusal = DeviceTreeReason::kEmptyTaskList;
  }
  if (refusal != DeviceTreeReason::kNone) {
    LogDeviceLevelsFallback(refusal);
    return 0;
  }
  if (device_tree && device_tree_scratch_refused_) {
    // PlanDeviceTreeChunks: not even one searched pair per chunk fits: the host level loop, not the prefix
    LogDeviceLevelsFallback(DeviceTreeReason::kScratchTooLarge);
    return 0;
  }
  if (device_tree) {
    return config_->max_depth;
  }
  // levels L with 2^(L + 1) <= num_leaves and L + 1 < max_depth: every valid leaf splits and every child is searched
  int levels = 0;
  while ((2LL << levels) <= static_cast<int64_t>(config_->num_leaves)) {
    ++levels;
  }
  if (config_->max_depth > 0) {
    levels = std::min(levels, config_->max_depth - 1);
  }
  // the select / pairs kernels' block covers a level of at most kDeviceLevelMaxPairs splits
  int max_levels = 0;
  while ((1 << max_levels) < kDeviceLevelMaxPairs) {
    ++max_levels;
  }
  levels = std::min(levels, max_levels + 1);
  return levels >= 2 ? levels : 0;
}

// The whole tree's searched pairs: its final level (2^(max_depth - 1) splits) is applied without a child search
// (EnqueueDeviceLevels breaks after the final apply; DeviceLevelPairsBlock writes no descriptors there), so the
// widest search is the children of level max_depth - 2: 2^(max_depth - 2) pairs. The prefix searches the children of
// each of its levels, the last one included: 2^(num_levels - 1) pairs.
int CUDASingleGPUTreeLearner::DeviceLevelSearchPairs(const int num_levels) const {
  return DeviceTreeRequested() ? std::max(1, (1 << num_levels) / 4) : (1 << num_levels) / 2;
}

// cuda_plan key device_tree: the pair chunk of a whole tree of max_depth num_levels, against free device memory
// (FALCATA_DEBUG=vramfree=N caps it, as ChooseViewRegime's), before the first tree's level flow; ReleaseDeviceLevelGraphs
// (ResetConfig, ResetTrainingData) clears it, so the next tree plans again. The measure is the bytes SetupDeviceLevels
// and BeginDeviceBynode still have to allocate (a growing Resize holds the old block until the new one exists): the
// tree's fixed scratch (per-split descriptors, slots and journal, the block map, the select's global scratch, the
// bynode keys and tables) plus, for a chunk of P pairs, the split finder's output (2 * num_tasks * P CUDASplitInfo),
// the quantized bit-change scratch (P * total bins int32), the bynode rows and task lists of 1 + 2 P row slots and the
// chunk states. The widest search's pairs when that fits half of the free memory; otherwise the most pairs that do
// (at least one). Chunking changes no float result: a pair's fix/subtract and find read and write only its own
// histograms, scratch region and leaf entries, and the select reads the level's leaf entries after every chunk.
void CUDASingleGPUTreeLearner::PlanDeviceTreeChunks() {
  if (device_tree_chunk_pairs_ >= 0 || !DeviceTreeRequested()) {
    return;
  }
  const int num_levels = config_->max_depth;
  const int pairs = DeviceLevelSearchPairs(num_levels);
  const size_t leaves = static_cast<size_t>(1) << num_levels;
  const size_t final_splits = leaves / 2;
  const size_t select_bytes = final_splits > static_cast<size_t>(kDeviceTreeSharedSplits) ?
    final_splits * (sizeof(int) + sizeof(double)) + leaves / 32 * sizeof(unsigned int) + leaves * sizeof(int2) : 0;
  const size_t other_bytes =
    (static_cast<size_t>(kDeviceLevelJournalHeader) + kDeviceLevelJournalRecord * leaves) * sizeof(int) +
    (leaves + 2) * sizeof(CUDALeafSplitsStruct) + (leaves + 4) * sizeof(CUDAHybridApplyDescriptor) +
    static_cast<size_t>(pairs) * sizeof(CUDAHybridPairDescriptor) + (static_cast<size_t>(pairs) + 1) * sizeof(int) +
    select_bytes;
  // cuda_plan key device_bynode_masks: BeginDeviceBynode's table, ranges, keys and sample (per leaf slot) and its rows
  // and task lists (per row slot: one per leaf slot unchunked, 1 + 2 P chunked; slots x (num_features + 4 x max_tasks)
  // bytes dominate: hundreds of MB at depth 12 with ~10k features)
  const bool bynode = select_features_by_node_ && DeviceBynodeMasksUsable();
  const size_t slots = std::min(static_cast<size_t>(config_->num_leaves), leaves) + 2;
  const size_t num_features = static_cast<size_t>(train_data_->num_features());
  size_t bynode_fixed = 0;
  size_t bynode_per_slot = 0;
  // the most pairs per chunk whose per-node task-list offsets stay in int range (depth 17 unchunked: 131074 slots)
  int bynode_cap = pairs;
  if (bynode) {
    std::vector<int> candidates;
    std::vector<int> task_inner;
    int max_tasks = 0;
    cuda_best_split_finder_->NodeTaskTables(col_sampler_.GetByNodeKeyedShape(&candidates).num_sampled, &task_inner,
                                            &max_tasks);
    bynode_fixed = (slots * sizeof(int8_t*) + 255) / 256 * 256 + 2 * slots * sizeof(int) +
      2 * slots * sizeof(uint64_t) + (5 + num_features + task_inner.size()) * sizeof(int);
    bynode_per_slot = num_features + static_cast<size_t>(std::max(1, max_tasks)) * sizeof(int);
    bynode_cap = DeviceTreeBynodeChunkCap(static_cast<int64_t>(slots), max_tasks, pairs);
  }
  const auto bynode_bytes = [&](const int chunk) {
    return bynode ? bynode_fixed + (chunk < pairs ? 1 + 2 * static_cast<size_t>(chunk) : slots) * bynode_per_slot : 0;
  };
  const auto growth_bytes = [&](const int chunk) {
    const size_t chunks = static_cast<size_t>((pairs + chunk - 1) / chunk);
    return cuda_best_split_finder_->HybridLevelGrowthBytes(chunk) +
      cuda_histogram_constructor_->BitChangeScratchGrowthBytes(chunk) + other_bytes + bynode_bytes(chunk) +
      (chunks > 1 ? chunks * sizeof(CUDADeviceLevelState) : 0);
  };
  size_t free_bytes = 0;
  size_t device_bytes = 0;
  CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_bytes, &device_bytes));
  const int64_t cap_mib = FalcataDebug().vramfree_mib;
  if (cap_mib >= 0) {
    free_bytes = std::min(free_bytes, static_cast<size_t>(cap_mib) << 20);
  }
  const size_t budget = free_bytes / 2;
  int chunk = pairs;
  // pair chunking serves the quantized device tree; a non-quantized tree (NonquantDeviceTreeShape's bound) keeps the
  // whole search's scratch or, over the budget, its one-sync host flow (scratch_too_large), as before P4
  const bool chunkable = config_->use_quantized_grad;
  if (chunkable && (growth_bytes(pairs) > budget || bynode_cap < pairs)) {
    // the most pairs within the budget (growth_bytes rises with the chunk) and the bynode offset cap, at least one
    chunk = 1;
    int lo = 2;
    int hi = std::min(pairs - 1, bynode_cap);
    while (lo <= hi) {
      const int mid = lo + (hi - lo) / 2;
      if (growth_bytes(mid) <= budget) {
        chunk = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
  }
  device_tree_chunk_pairs_ = chunk;
  // even one pair per chunk exceeds the budget (half of the free memory; vramfree caps it), or the bynode task offsets
  // overflow at one pair: the host level loop grows this learner's trees (reason scratch_too_large), no CUDA Fatal
  device_tree_scratch_refused_ = growth_bytes(chunk) > budget || bynode_cap < 1;
  if (device_tree_scratch_refused_ && FalcataDebug().diag) {
    fprintf(stderr, "[hybrid-diag] device_tree: scratch too large: max_depth %d needs %.1f MiB at one searched pair "
            "per chunk, %.1f MiB free (budget half): host level loop\n", num_levels,
            growth_bytes(chunk) / (1024.0 * 1024.0), free_bytes / (1024.0 * 1024.0));
  }
  if (FalcataDebug().diag) {
    constexpr double kMiB = 1024.0 * 1024.0;
    const size_t finder_bytes = 2 * static_cast<size_t>(cuda_best_split_finder_->num_tasks()) *
      static_cast<size_t>(chunk) * sizeof(CUDASplitInfo);
    const size_t bit_change_bytes = config_->use_quantized_grad ?
      cuda_histogram_constructor_->BitChangeScratchEntries(chunk) * sizeof(hist_t) : 0;
    fprintf(stderr, "[hybrid-diag] device_tree: scratch estimate %.1f MiB (finder %.1f, bit-change %.1f, other %.1f, "
            "bynode %.1f; to allocate %.1f) for max_depth %d, %d searched pairs, %d chunks of %d pairs, %d tasks, %d "
            "total bins; free %.1f of %.1f MiB\n", (finder_bytes + bit_change_bytes + other_bytes +
            bynode_bytes(chunk)) / kMiB, finder_bytes / kMiB, bit_change_bytes / kMiB, other_bytes / kMiB,
            bynode_bytes(chunk) / kMiB, growth_bytes(chunk) / kMiB, num_levels, pairs, DeviceTreeChunks(pairs),
            chunk, cuda_best_split_finder_->num_tasks(), num_total_bin_, free_bytes / kMiB, device_bytes / kMiB);
  }
}

bool CUDASingleGPUTreeLearner::SetupDeviceLevels(const int num_levels) {
  // static feature tables of the select kernel: SplitLevelBatched's per-feature metadata and the real thresholds
  std::vector<CUDAHybridGraphFeatureMeta> meta;
  cuda_data_partition_->BuildDeviceLevelFeatureMeta(&meta);
  CHECK_EQ(static_cast<int>(meta.size()), train_data_->num_features());
  std::vector<double> real_thresholds;
  uint32_t threshold_offset = 0;
  for (int feature_index = 0; feature_index < train_data_->num_features(); ++feature_index) {
    meta[feature_index].real_feature_index = train_data_->RealFeatureIndex(feature_index);
    meta[feature_index].missing_type = static_cast<int>(train_data_->FeatureBinMapper(feature_index)->missing_type());
    meta[feature_index].real_threshold_offset = threshold_offset;
    const int num_bin = train_data_->FeatureBinMapper(feature_index)->num_bin();
    for (int bin = 0; bin < num_bin; ++bin) {
      real_thresholds.push_back(train_data_->RealThreshold(feature_index, static_cast<uint32_t>(bin)));
    }
    threshold_offset += static_cast<uint32_t>(num_bin);
  }
  cuda_device_level_feature_meta_.InitFromHostVector(meta);
  cuda_device_level_real_thresholds_.InitFromHostVector(real_thresholds);
  cuda_device_level_feature_source_.Resize(static_cast<size_t>(train_data_->num_features()));
  cuda_device_level_state_.Resize(1);
  cuda_device_level_config_.Resize(1);
  const int max_prefix_leaves = 1 << num_levels;
  device_level_journal_size_ = static_cast<size_t>(kDeviceLevelJournalHeader) +
    kDeviceLevelJournalRecord * static_cast<size_t>(max_prefix_leaves);
  cuda_device_level_journal_.Resize(device_level_journal_size_);
  if (pinned_device_level_journal_ != nullptr) {
    cudaFreeHost(pinned_device_level_journal_);
    pinned_device_level_journal_ = nullptr;
  }
  if (cudaHostAlloc(reinterpret_cast<void**>(&pinned_device_level_journal_),
                    device_level_journal_size_ * sizeof(int), cudaHostAllocDefault) != cudaSuccess) {
    return false;
  }
  if (device_level_select_event_ == nullptr) {
    CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&device_level_select_event_, cudaEventDisableTiming));
  }
  if (tree_record_done_event_ == nullptr) {
    CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&tree_record_done_event_, cudaEventDisableTiming));
  }
  // every buffer a device level writes, at its worst case: a growth is a device synchronize the enqueue cannot
  // pay mid-level (it would also reallocate under in-flight kernels' pointers)
  const int max_pairs = DeviceLevelSearchPairs(num_levels);
  device_level_max_pairs_ = max_pairs;
  // the per-pair search scratch holds one pair chunk of a complete tree (PlanDeviceTreeChunks), the widest search
  // otherwise
  const bool whole_tree = DeviceTreeRequested();
  const int scratch_pairs = whole_tree && device_tree_chunk_pairs_ > 0 ? std::min(device_tree_chunk_pairs_, max_pairs) :
    max_pairs;
  device_level_scratch_pairs_ = scratch_pairs;
  const size_t max_slots = static_cast<size_t>(max_prefix_leaves) + 2;
  if (hybrid_pair_slots_.Size() < max_slots) {
    hybrid_pair_slots_.Resize(max_slots);
  }
  if (cuda_hybrid_pair_descs_.Size() < static_cast<size_t>(max_pairs)) {
    cuda_hybrid_pair_descs_.Resize(static_cast<size_t>(max_pairs));
  }
  if (cuda_device_level_block_start_.Size() < static_cast<size_t>(max_pairs) + 1) {
    cuda_device_level_block_start_.Resize(static_cast<size_t>(max_pairs) + 1);
  }
  // a complete tree's levels wider than the shared select: its arrays in global scratch (the final level's splits)
  const size_t final_splits = static_cast<size_t>(max_prefix_leaves) / 2;
  if (whole_tree && final_splits > static_cast<size_t>(kDeviceTreeSharedSplits) &&
      cuda_device_level_select_leaves_.Size() < final_splits) {
    cuda_device_level_select_leaves_.Resize(final_splits);
    cuda_device_level_select_gains_.Resize(final_splits);
    cuda_device_level_select_bits_.Resize(2 * final_splits / 32);
    cuda_device_level_select_gaps_.Resize(2 * final_splits);
  }
  const int chunks = whole_tree ? DeviceTreeChunks(max_pairs) : 1;
  if (chunks > 1 && cuda_device_level_chunk_states_.Size() < static_cast<size_t>(chunks)) {
    cuda_device_level_chunk_states_.Resize(static_cast<size_t>(chunks));
  }
  cuda_data_partition_->EnsureDeviceLevelCapacity(max_prefix_leaves);
  cuda_best_split_finder_->EnsureDeviceLevelCapacity(scratch_pairs);
  cuda_histogram_constructor_->EnsureDeviceLevelCapacity(scratch_pairs);
  device_levels_ready_ = true;
  return true;
}

void CUDASingleGPUTreeLearner::EnqueueDeviceLevels(CUDATree* tree, const CUDADeviceLevelConfig& cfg,
                                                   const int num_levels, const bool inline_all_map,
                                                   const bool final_map_only) {
  CHECK(!inline_all_map || cfg.whole_tree);
  CHECK(!final_map_only || (cfg.whole_tree && inline_all_map));
  // one stream for the levels' chain (select, apply, pairs, construct, fix/subtract, find, sync): every launch
  // depends on the one before it, and a cross-stream hand-off costs a few microseconds per edge; the tree record
  // stays on the tree stream beside it
  cudaStream_t find_stream = cuda_best_split_finder_->find_stream();
  cudaStream_t apply_stream_saved = cuda_data_partition_->ExchangeApplyStream(find_stream);
  cudaStream_t hist_stream_saved = cuda_histogram_constructor_->ExchangeHistStream(find_stream);
  struct RestoreDeviceLevelStreams {
    CUDADataPartition* partition;
    CUDAHistogramConstructor* histogram;
    CUDABestSplitFinder* finder;
    cudaStream_t apply;
    cudaStream_t hist;
    ~RestoreDeviceLevelStreams() {
      histogram->SetDeviceLevelState(nullptr);
      finder->SetDeviceLevelState(nullptr);
      finder->SetDeviceLevelSelect(nullptr);
      partition->ExchangeApplyStream(apply);
      histogram->ExchangeHistStream(hist);
    }
  } restore{cuda_data_partition_.get(), cuda_histogram_constructor_.get(), cuda_best_split_finder_.get(),
            apply_stream_saved, hist_stream_saved};
  cudaStream_t tree_stream = tree->tree_stream();
  CUDADeviceLevelState* dls = cfg.state;
  // ordered after the root level's search (its sync kernel ran on the find stream)
  LaunchDeviceLevelReset(cfg, 1, find_stream);
  cuda_histogram_constructor_->SetDeviceLevelState(dls, cfg.whole_tree != 0);
  cuda_best_split_finder_->SetDeviceLevelState(dls);
  // (non-quantized training: the level finder on hist_t histograms)
  const score_t* grad_scale = config_->use_quantized_grad ? cuda_gradient_discretizer_->grad_scale_ptr() : nullptr;
  const score_t* hess_scale = config_->use_quantized_grad ? cuda_gradient_discretizer_->hess_scale_ptr() : nullptr;
  // The prefix can select in the previous sync's last block; whole trees select in a standalone launch.
  bool select_ran = false;
  const CUDADeviceLevelConfig* cfg_device = cuda_device_level_config_.RawDataReadOnly();
  // the selective prefix records no tree on the device (nothing reads it;
  // the selective flow rebuilds the tree from its own records), so the tree stream never joins the levels
  const bool record_tree = cfg.skip_tree_record == 0;
  CHECK(record_tree || (!cfg.whole_tree && cfg.split_info_journal != nullptr));
  for (int level = 0; level < num_levels; ++level) {
    const int max_splits = 1 << level;
    // the level's host decisions (after its search on this stream), then the apply and the tree record behind them
    if (!select_ran) {
      LaunchDeviceLevelSelect(cfg, level, max_splits, find_stream);
    }
    if (record_tree) {
      CUDASUCCESS_OR_FATAL(cudaEventRecord(device_level_select_event_, find_stream));
      CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(tree_stream, device_level_select_event_, 0));
    }
    // (the apply's last block writes the children's descriptors and block map: DeviceLevelPairsBlock)
    // the selective prefix's partition launch takes the whole tree's joint
    // worker bound (a level of a root-start tree partitions at most every row: the same bound holds)
    cuda_data_partition_->LaunchDeviceLevelApply(max_splits, dls, cfg_device,
      inline_all_map && cfg.whole_tree && level + 1 == num_levels,
      final_map_only && level + 1 == num_levels, cfg.whole_tree != 0 || cfg.joint_grid_bound != 0);
    if (record_tree) {
      tree->LaunchSplitBatchKernelDevice(max_splits, &dls->live_splits);
      CUDASUCCESS_OR_FATAL(cudaEventRecord(tree_record_done_event_, tree_stream));
    }
    if (cfg.whole_tree && level + 1 == num_levels) {
      // The apply has produced the final partition and tree records. No final child histogram or best-split
      // search can affect a depth-bounded tree. Join the record stream before the single end-of-tree journal.
      CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(find_stream, tree_record_done_event_, 0));
      break;
    }
    // the children's search behind the apply, inside the scratch SetupDeviceLevels presized (a growth here would
    // reallocate under in-flight or captured kernels)
    CHECK_LE(max_splits, device_level_max_pairs_);
    const int chunks = cfg.num_chunks > 1 ? DeviceTreeChunks(max_splits) : 1;
    if (chunks > 1) {
      // pair chunking: the level's construct over all its pairs (no per-pair scratch), then per chunk of
      // cfg.chunk_pairs pairs the fix/subtract and the find, each reading the chunk's state (the pairs block wrote
      // its live pairs) and descriptors and reusing one chunk of scratch; the select reads every chunk's leaf entries
      CHECK_EQ(cfg.chunk_pairs, device_tree_chunk_pairs_);
      CHECK_LE(cfg.chunk_pairs, device_level_scratch_pairs_);
      CHECK_LE(chunks, cfg.num_chunks);
      cuda_histogram_constructor_->ConstructHistogramsForLevelOnly(cuda_hybrid_pair_descs_.RawDataReadOnly(),
                                                                   max_splits);
      // (the record reads the split leaves' best-split entries, which the first chunk's sync starts overwriting)
      cuda_best_split_finder_->WaitOnCatSlabConsumers(nullptr, record_tree ? tree_record_done_event_ : nullptr);
      for (int c = 0; c < chunks; ++c) {
        const int base = c * cfg.chunk_pairs;
        const int count = std::min(cfg.chunk_pairs, max_splits - base);
        const CUDADeviceLevelState* chunk_state = cfg.chunk_states + c;
        const CUDAHybridPairDescriptor* chunk_descs = cuda_hybrid_pair_descs_.RawDataReadOnly() + base;
        cuda_histogram_constructor_->SetDeviceLevelState(chunk_state, /*whole_tree=*/true);
        cuda_best_split_finder_->SetDeviceLevelState(chunk_state);
        if (device_bynode_active_) {
          // cuda_plan key device_bynode_masks: the chunk's children's masks into the chunk's row slots
          EnqueueDeviceBynodeChunkMasks(find_stream, level, base, count);
        }
        cuda_histogram_constructor_->SubtractHistogramsForLevel(chunk_descs, count,
                                                                /*any_pair_needs_bit_change_copy=*/true);
        cuda_best_split_finder_->FindBestSplitsForLevel(chunk_descs, count, grad_scale, hess_scale);
      }
      cuda_histogram_constructor_->SetDeviceLevelState(dls, /*whole_tree=*/true);
      cuda_best_split_finder_->SetDeviceLevelState(dls);
      select_ran = false;
      continue;
    }
    // cuda_plan key device_bynode_masks: the children's masks on the side stream, beside their construct
    if (device_bynode_active_) {
      EnqueueDeviceBynodeMasks(find_stream, level, max_splits);
    }
    cuda_histogram_constructor_->ConstructHistogramsForLevel(
      cuda_hybrid_pair_descs_.RawDataReadOnly(), max_splits, /*max_num_data_in_smaller_leaf=*/0,
      /*any_pair_needs_bit_change_copy=*/true);
    // (the record reads the split leaves' best-split entries, which this level's sync overwrites)
    cuda_best_split_finder_->WaitOnCatSlabConsumers(nullptr, record_tree ? tree_record_done_event_ : nullptr);
    JoinDeviceBynode();
    cuda_best_split_finder_->SetDeviceLevelSelect(!cfg.whole_tree && level + 1 < num_levels ? cfg_device : nullptr);
    cuda_best_split_finder_->FindBestSplitsForLevel(
      cuda_hybrid_pair_descs_.RawDataReadOnly(), max_splits, grad_scale, hess_scale);
    select_ran = cuda_best_split_finder_->LastSyncRanSelect();
    cuda_best_split_finder_->SetDeviceLevelSelect(nullptr);
  }
}

bool CUDASingleGPUTreeLearner::LaunchDeviceLevelGraph(CUDATree* tree, const CUDADeviceLevelConfig& cfg,
                                                      const int num_levels, const bool inline_all_map,
                                                   const bool final_map_only) {
  // the pointers and per-tree launch shapes the captured launches bake in
  std::vector<const void*> key;
  key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(num_levels)));
  key.push_back(static_cast<const void*>(tree->hybrid_graph_batch_splits()));
  key.push_back(static_cast<const void*>(cfg.root_struct));
  key.push_back(static_cast<const void*>(cfg.tree_leaf_value));
  key.push_back(static_cast<const void*>(cuda_hybrid_pair_descs_.RawDataReadOnly()));
  key.push_back(static_cast<const void*>(hybrid_pair_slots_.RawDataReadOnly()));
  // Sampled compact views may select a different decoder while retaining the same allocation and padded shape.
  key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(cuda_histogram_constructor_->compact_codec())));
  cuda_histogram_constructor_->DeviceLevelGraphKey(&key, cfg.whole_tree ? &cfg.shape : nullptr);
  cuda_best_split_finder_->DeviceLevelGraphKey(&key);
  cuda_data_partition_->DeviceLevelGraphKey(&key);
  if (cfg.split_info_journal != nullptr) {
    // cuda_plan key selective_device_prefix: the select captures cfg by value, the journal pointer included (append
    // only: a prefix without it keeps its key)
    key.push_back(static_cast<const void*>(cfg.split_info_journal));
  }
  if (cfg.whole_tree) {
    // Append only: leave prefix keys and old field indices unchanged; map allocation is independent.
    key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(inline_all_map)));
    key.push_back(static_cast<const void*>(cuda_data_partition_->cuda_data_index_to_leaf_index()));
    // The final warm clear AND whole joint worker grid capture the real row extent, even on retained allocations.
    key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(num_data_)));
    // Append only for whole trees: preserve every old field index and the exact prefix tuple.
    key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(final_map_only)));
    // Select captures cfg BY VALUE; keep descriptor units identical to the new host receipt after a plan change.
    key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(cfg.gap_block_dim)));
    // The select kernel captures this value too, including when reset/reuse retains the same allocations.
    key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(cfg.num_leaves_budget)));
    // cuda_device_levels: the block map, the select's global scratch and the pair chunks (the chunk loop's
    // launches and the states the pairs block writes)
    key.push_back(static_cast<const void*>(cfg.construct_block_start));
    key.push_back(static_cast<const void*>(cfg.select_leaves));
    key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(cfg.num_chunks)));
    key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(cfg.chunk_pairs)));
    key.push_back(static_cast<const void*>(cfg.chunk_states));
    // cuda_plan key device_bynode_masks: the mask kernels and the buffers they capture
    key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(device_bynode_active_ ? 1 : 0)));
    if (!config_->use_quantized_grad) {
      // non-quantized training: the construct's instantiation (fixed point or float atomics, interleaved pairs)
      key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(
        cuda_histogram_constructor_->NonquantDeviceLevelKey())));
    }
    if (device_bynode_active_) {
      key.push_back(static_cast<const void*>(device_bynode_cfg_.sample));
      key.push_back(static_cast<const void*>(device_bynode_cfg_.task_inner_feature));
      key.push_back(static_cast<const void*>(device_bynode_cfg_.rows));
      key.push_back(static_cast<const void*>(device_bynode_cfg_.keys));
      key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(device_bynode_cfg_.num_slots)));
      key.push_back(reinterpret_cast<const void*>(static_cast<uintptr_t>(device_bynode_cfg_.num_tasks)));
      key.push_back(static_cast<const void*>(device_bynode_cfg_.table));
    }
  }
  if (!cuda_histogram_constructor_->DeviceLevelGraphCapturable()) {
    return false;  // the construct would rewrite this tree's compact view first (a per-tree state, not a launch)
  }
  cudaStream_t find_stream = cuda_best_split_finder_->find_stream();
  for (size_t i = 0; i < device_level_graphs_.size(); ++i) {
    if (device_level_graphs_[i]->key == key) {
      if (i + 1 != device_level_graphs_.size()) {
        std::rotate(device_level_graphs_.begin() + i, device_level_graphs_.begin() + i + 1, device_level_graphs_.end());
      }
      CUDASUCCESS_OR_FATAL(cudaGraphLaunch(device_level_graphs_.back()->exec, find_stream));
      // the host halves of the captured enqueue: the index buffers swap once per level
      cuda_data_partition_->SwapIndexBuffers(num_levels);
      ++device_level_graph_hits_;
      if (FalcataDebug().diag && device_level_graph_hits_ + device_level_graph_misses_ == 64) {
        fprintf(stderr, "[hybrid-diag] device_level_graph: %d hits, %d misses in the first 64 trees (%d cached)\n",
                device_level_graph_hits_, device_level_graph_misses_, static_cast<int>(device_level_graphs_.size()));
      }
      return true;
    }
  }
  // Diagnostic-only evidence of which exact launch-key fields changed; never printed during timings.
  if (FalcataDebug().diag && cfg.whole_tree && !device_level_graphs_.empty()) {
    const auto& previous = device_level_graphs_.back()->key;
    for (size_t i = 0; i < std::min(key.size(), previous.size()); ++i) {
      if (key[i] != previous[i]) {
        fprintf(stderr, "[hybrid-diag] device_tree: graph key[%zu] %p -> %p\n", i, previous[i], key[i]);
      }
    }
  }
  // a new key: capture this tree's enqueue (its host halves run now; the graph runs right after)
  ++device_level_graph_misses_;
  const bool churning = device_level_graph_misses_ > 8 && device_level_graph_misses_ > device_level_graph_hits_;
  const bool update = FalcataPlan::Get().device_level_graph_update && !device_level_graph_update_failed_;
  if (!cfg.whole_tree && churning) {
    // per-tree keys keep changing (e.g. a sampled feature set every tree): the plain enqueue from here on
    device_level_graph_disabled_ = true;
    if (FalcataDebug().diag) {
      fprintf(stderr, "[hybrid-diag] device_level_graph: %d misses, %d hits: graphs off\n",
              device_level_graph_misses_, device_level_graph_hits_);
    }
    return false;
  }
  std::unique_ptr<DeviceLevelGraph> graph(new DeviceLevelGraph());
  graph->key = key;
  CUDASUCCESS_OR_FATAL(cudaStreamBeginCapture(find_stream, cudaStreamCaptureModeThreadLocal));
  try {
    EnqueueDeviceLevels(tree, cfg, num_levels, inline_all_map, final_map_only);
  } catch (...) {
    // End even an invalidated capture before the next tree can use these streams.
    cudaGraph_t abandoned = nullptr;
    cudaStreamEndCapture(find_stream, &abandoned);
    if (abandoned != nullptr) {
      cudaGraphDestroy(abandoned);
    }
    throw;
  }
  CUDASUCCESS_OR_FATAL(cudaStreamEndCapture(find_stream, &graph->graph));
  if (FalcataDebug().diag && cfg.whole_tree && inline_all_map) {
    // Validate the actual captured memset extent and FULL-completion edges, including with PDL enabled.
    // Graph support already requires CUDA>=12.4, including the edge-data query API (no lossy legacy query).
#if CUDART_VERSION >= 13000
    const auto get_edges = cudaGraphGetEdges;
#else
    const auto get_edges = cudaGraphGetEdges_v2;
#endif
    size_t node_count = 0;
    size_t edge_count = 0;
    CUDASUCCESS_OR_FATAL(cudaGraphGetNodes(graph->graph, nullptr, &node_count));
    CUDASUCCESS_OR_FATAL(get_edges(graph->graph, nullptr, nullptr, nullptr, &edge_count));
    std::vector<cudaGraphNode_t> nodes(node_count), from(edge_count), to(edge_count);
    std::vector<cudaGraphEdgeData> edges(edge_count);
    CUDASUCCESS_OR_FATAL(cudaGraphGetNodes(graph->graph, nodes.data(), &node_count));
    CUDASUCCESS_OR_FATAL(get_edges(graph->graph, from.data(), to.data(), edges.data(), &edge_count));
    int clears = 0;
    for (const cudaGraphNode_t node : nodes) {
      cudaGraphNodeType type;
      CUDASUCCESS_OR_FATAL(cudaGraphNodeGetType(node, &type));
      if (type != cudaGraphNodeTypeMemset) {
        continue;
      }
      cudaMemsetParams params = {};
      CUDASUCCESS_OR_FATAL(cudaGraphMemsetNodeGetParams(node, &params));
      if (params.dst != cuda_data_partition_->cuda_data_index_to_leaf_index()) {
        continue;
      }
      const size_t bytes = params.width * params.height * params.elementSize;
      CHECK_EQ(bytes, static_cast<size_t>(num_data_) * sizeof(int));
      CHECK_EQ(params.value, 0u);
      int incoming = 0;
      int outgoing = 0;
      for (size_t i = 0; i < edge_count; ++i) {
        if (from[i] == node || to[i] == node) {
          CHECK_EQ(static_cast<int>(edges[i].type), static_cast<int>(cudaGraphDependencyTypeDefault));
          CHECK_EQ(static_cast<int>(edges[i].from_port), 0);
          CHECK_EQ(static_cast<int>(edges[i].to_port), 0);
          incoming += to[i] == node ? 1 : 0;
          outgoing += from[i] == node ? 1 : 0;
        }
      }
      CHECK_GT(incoming, 0);
      CHECK_GT(outgoing, 0);
      ++clears;
      fprintf(stderr, "[hybrid-diag] device_tree_warm_clear_graph: bytes=%zu incoming_full=%d outgoing_full=%d\n",
              bytes, incoming, outgoing);
    }
    CHECK_EQ(clears, 1);
  }
  // cuda_plan key device_level_graph_update: whole-tree keys that keep changing (a sampled feature set every tree)
  // retarget the least recently used exec to this capture in place: the same launches and parameters as a new
  // instantiation, without its instantiate, upload and destroy. A topology change fails the update: instantiate.
  if (cfg.whole_tree && update && churning && !device_level_graphs_.empty()) {
    DeviceLevelGraph* lru = device_level_graphs_.front().get();
    cudaGraphExecUpdateResultInfo info = {};
    const cudaError_t updated = cudaGraphExecUpdate(lru->exec, graph->graph, &info);
    if (updated == cudaSuccess) {
      ++device_level_graph_update_hits_;
      std::swap(lru->graph, graph->graph);
      lru->key.swap(graph->key);
      std::rotate(device_level_graphs_.begin(), device_level_graphs_.begin() + 1, device_level_graphs_.end());
      CUDASUCCESS_OR_FATAL(cudaGraphLaunch(lru->exec, find_stream));
      return true;  // graph (the old capture) is destroyed here
    }
    // only a refused update (cudaErrorGraphExecUpdateFailure: the capture's topology or node parameters do not fit
    // this exec) falls back to a new instantiation; any other error (an invalid argument, a fault from earlier queued
    // work) is fatal
    if (updated != cudaErrorGraphExecUpdateFailure) {
      CUDASUCCESS_OR_FATAL(updated);
    }
    cudaGetLastError();  // clear the refusal: instantiate below
    device_level_graph_update_last_refusal_ = static_cast<int>(info.result);
    if (FalcataDebug().diag) {
      fprintf(stderr, "[hybrid-diag] device_level_graph_update: refused result=%d (cudaGraphExecUpdateResult)\n",
              static_cast<int>(info.result));
    }
    // topologies that keep changing: no more update attempts
    ++device_level_graph_update_misses_;
    device_level_graph_update_failed_ = device_level_graph_update_misses_ > 8 &&
      device_level_graph_update_misses_ > device_level_graph_update_hits_;
  }
  CUDASUCCESS_OR_FATAL(cudaGraphInstantiate(&graph->exec, graph->graph, 0));
  CUDASUCCESS_OR_FATAL(cudaGraphUpload(graph->exec, find_stream));
  CUDASUCCESS_OR_FATAL(cudaGraphLaunch(graph->exec, find_stream));
  const size_t cache_limit = cfg.whole_tree ? 256 : 32;
  if (device_level_graphs_.size() >= cache_limit) {
    device_level_graphs_.erase(device_level_graphs_.begin());
  }
  device_level_graphs_.push_back(std::move(graph));
  return true;
}

CUDASingleGPUTreeLearner::DeviceLevelGraph::~DeviceLevelGraph() {
  if (exec != nullptr) {
    cudaGraphExecDestroy(exec);
  }
  if (graph != nullptr) {
    cudaGraphDestroy(graph);
  }
}

int CUDASingleGPUTreeLearner::TrainDeviceLevels(CUDATree* tree, const int num_levels,
                                                std::vector<HybridPendingPair>* pairs, const bool record_split_infos) {
  global_timer.Start("CUDASingleGPUTreeLearner::TrainDeviceLevels");
  cudaStream_t find_stream = cuda_best_split_finder_->find_stream();
  // the tree's column sources (the per-tree compact or packed view), as SplitLevelBatched reads them per split
  cuda_data_partition_->BuildDeviceLevelFeatureSource(&host_device_level_feature_source_);
  CopyFromHostToCUDADeviceAsync<CUDAHybridGraphFeatureSource>(
    cuda_device_level_feature_source_.RawData(), host_device_level_feature_source_.data(),
    host_device_level_feature_source_.size(), find_stream, __FILE__, __LINE__);
  CUDADeviceLevelState* dls = cuda_device_level_state_.RawData();
  CUDADeviceLevelConfig& cfg = host_device_level_config_;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.state = dls;
  cfg.leaf_best_split_info = cuda_best_split_finder_->leaf_best_split_info_ptr(0);
  cfg.root_struct = cuda_smaller_leaf_splits_->GetCUDAStruct();
  // (non-quantized trees keep the root output of the one-sync flow, which reads the root sums back only for a stump:
  // the first split records internal_value[0] = 0, as that flow and the graph loop it replaced wrote it)
  cfg.tree_leaf_value = config_->use_quantized_grad ? tree->cuda_leaf_value_ref() : nullptr;
  cfg.leaf_num_data = cuda_data_partition_->cuda_leaf_num_data();
  cfg.leaf_data_start = cuda_data_partition_->cuda_leaf_data_start();
  cfg.tree_batch_splits = tree->hybrid_graph_batch_splits();
  cfg.apply_descs = cuda_data_partition_->device_level_apply_descs();
  cfg.pair_descs = cuda_hybrid_pair_descs_.RawData();
  cfg.pair_slots = hybrid_pair_slots_.RawData();
  cfg.feature_meta = cuda_device_level_feature_meta_.RawDataReadOnly();
  cfg.feature_source = cuda_device_level_feature_source_.RawDataReadOnly();
  cfg.real_thresholds = cuda_device_level_real_thresholds_.RawDataReadOnly();
  cfg.journal = cuda_device_level_journal_.RawData();
  cfg.max_depth = config_->max_depth;
  cfg.num_leaves_budget = config_->num_leaves;
  cfg.whole_tree = DeviceTreeRequested() ? 1 : 0;
  cfg.construct_block_start = cuda_device_level_block_start_.RawData();
  if (cfg.whole_tree && (1 << (num_levels - 1)) > kDeviceTreeSharedSplits) {
    // the levels wider than the shared select (SetupDeviceLevels sized the scratch for the final level)
    CHECK_GE(cuda_device_level_select_leaves_.Size(), static_cast<size_t>(1) << (num_levels - 1));
    cfg.select_leaves = cuda_device_level_select_leaves_.RawData();
    cfg.select_gains = cuda_device_level_select_gains_.RawData();
    cfg.select_split_bits = cuda_device_level_select_bits_.RawData();
    cfg.select_gaps = cuda_device_level_select_gaps_.RawData();
    cfg.select_capacity = 1 << (num_levels - 1);
  }
  // pair chunking (PlanDeviceTreeChunks): the chunks of the widest search; narrower levels use the first ones
  cfg.num_chunks = cfg.whole_tree ? DeviceTreeChunks(device_level_max_pairs_) : 1;
  if (cfg.num_chunks > 1) {
    CHECK_GE(cuda_device_level_chunk_states_.Size(), static_cast<size_t>(cfg.num_chunks));
    cfg.chunk_states = cuda_device_level_chunk_states_.RawData();
    cfg.chunk_pairs = device_tree_chunk_pairs_;
  }
  // cuda_plan key selective_device_prefix: the applied splits' best-split records, for the selective records. Sized
  // here, outside SetupDeviceLevels, because only selective trees ask for them (the device tree and the plain prefix
  // never allocate them); the growth happens before any level is enqueued or captured, and K is config-constant, so
  // it happens once per learner (ReleaseDeviceLevelGraphs keeps the buffers: a larger K regrows them here)
  const int max_prefix_splits = std::min((1 << num_levels) - 1, config_->num_leaves - 1);
  if (record_split_infos) {
    CHECK(!cfg.whole_tree);
    if (cuda_selective_prefix_split_infos_.Size() < static_cast<size_t>(max_prefix_splits)) {
      cuda_selective_prefix_split_infos_.Resize(static_cast<size_t>(max_prefix_splits));
    }
    if (pinned_selective_prefix_split_infos_count_ < static_cast<size_t>(max_prefix_splits)) {
      if (pinned_selective_prefix_split_infos_ != nullptr) {
        CUDASUCCESS_OR_FATAL(cudaFreeHost(pinned_selective_prefix_split_infos_));
        pinned_selective_prefix_split_infos_ = nullptr;
      }
      CUDASUCCESS_OR_FATAL(cudaHostAlloc(&pinned_selective_prefix_split_infos_,
                                         static_cast<size_t>(max_prefix_splits) * sizeof(CUDASplitInfo),
                                         cudaHostAllocDefault));
      pinned_selective_prefix_split_infos_count_ = static_cast<size_t>(max_prefix_splits);
    }
    cfg.split_info_journal = cuda_selective_prefix_split_infos_.RawData();
    // the selective prefix: no device tree record and the joint partition bound (falcata_plan.h,
    // selective_device_prefix)
    cfg.skip_tree_record = 1;
    cfg.joint_grid_bound = 1;
  }
  // No current-tree host tail exists for a completed whole tree. Plain unweighted L2 never renews its
  // outputs, and its next BeforeTrain rebuilds root counts/start/bits before any host search. All other
  // modes retain full child statistics; GPU split-info storage/production and full journal replay stay intact.
  // cuda_plan key device_tree_lean_end: every objective boosted on the device, not only constant-hessian L2 (the
  // skipped cache is read only by host searches/tails and the host RenewTreeOutput, which CHECK it; the device renewal
  // of L1 / quantile / MAPE reads the device leaf windows), with weights and under row bagging. Trees of two or more
  // levels only: a one-level tree (a stump shape, max_depth 1) measured slower on the lean end (fraud stumps, binary:
  // 0.090 -> 0.094 ms per tree, same binary, 1000 rounds); it keeps the full tree end (constant-hessian L2 as before).
  const bool lean_end = FalcataPlan::Get().device_tree_lean_end && num_levels >= 2;
  bool skip_host_cache = cfg.whole_tree && config_->boosting == "gbdt" &&
    (lean_end || (config_->objective == "regression" && share_state_->is_constant_hessian)) &&
    boosting_on_cuda_ && (lean_end || train_data_->metadata().weights() == nullptr) &&
    config_->use_quantized_grad && vec_num_targets_ <= 1 && !has_categorical_feature_ &&
    nccl_communicator_ == nullptr && fp_merge_state_ == nullptr && !forest_slot_ &&
    config_->data_sample_strategy == "bagging" && !config_->bagging_by_query &&
    // device_tree_lean_end: under row bagging too (the bag is partitioned on the device; the cache is not read)
    (lean_end ||
     (config_->bagging_freq == 0 && config_->bagging_fraction == 1.0 && config_->pos_bagging_fraction == 1.0 &&
      config_->neg_bagging_fraction == 1.0 && !cuda_data_partition_->use_bagging())) &&
    !config_->quant_train_renew_leaf && !config_->linear_tree &&
    (!select_features_by_node_ || device_bynode_active_) &&
    config_->interaction_constraints_vector.empty() && config_->monotone_constraints.empty() &&
    !use_monotone_constraints_ && config_->monotone_penalty == 0.0 && config_->feature_contri.empty() &&
    config_->cegb_penalty_split == 0.0 && config_->cegb_penalty_feature_lazy.empty() &&
    config_->cegb_penalty_feature_coupled.empty() && config_->forcedsplits_filename.empty() &&
    (forced_split_json_ == nullptr || forced_split_json_->is_null()) &&
    !FalcataDebug().debug && !FalcataDebug().dump;
#ifdef DEBUG
  skip_host_cache = false;  // CheckSplitValid consumes the host statistics/windows.
#endif  // DEBUG
  // Same conservative whole-L2 subset as V10; one mode feeds capture identity and both enqueue routes.
  // cuda_plan key device_tree_lean_end: a row bag's windows too (the tree-end full pass writes the same in-bag rows)
  const bool all_rows_root = cuda_data_partition_->root_num_data() == num_data_;
  const bool inline_all_map = skip_host_cache && (all_rows_root || lean_end) &&
    cuda_data_partition_->cuda_data_index_to_leaf_index() != nullptr;
  cfg.gap_block_dim = CUDADataPartition::DeviceLevelGapBlockDim();
  cfg.min_data_in_leaf = static_cast<data_size_t>(config_->min_data_in_leaf);
  cfg.min_sum_hessian_in_leaf = config_->min_sum_hessian_in_leaf;
  // EnqueueLevelBestSplitSearch's skip_unsplittable gate (no forced splits reach the prefix)
  cfg.skip_unsplittable = (FalcataPlan::Get().skip_unsplittable && config_->forcedsplits_filename.empty() &&
    (forced_split_json_ == nullptr || forced_split_json_->is_null())) ? 1 : 0;
  cfg.num_grad_quant_bins = cuda_histogram_constructor_->DeviceLevelNumGradQuantBins();
  cfg.construct_min_grid_dim_y = cuda_histogram_constructor_->DeviceLevelConstructMinGridDimY();
  cfg.construct_min_rows_per_thread = CUDAHistogramConstructor::BatchConstructMinRowsPerThread();
  cfg.construct_saturation_floor = CUDAHistogramConstructor::BatchConstructSaturationFloor();
  cuda_histogram_constructor_->DeviceLevelConstructShape(&cfg.shape);
  // (the non-quantized dense kernel has no rows-per-thread floor: each block row takes its pair's grid share)
  cfg.level_row_blocks = config_->use_quantized_grad && (cfg.shape.pair_mode == 0 ?
    FalcataPlan::Get().dense_level_row_blocks : FalcataPlan::Get().level_row_blocks) ? 1 : 0;
  // the apply's pairs block reads the config from device memory (stream-ordered before the levels; the host copy
  // stays alive in the learner)
  CopyFromHostToCUDADeviceAsync<CUDADeviceLevelConfig>(cuda_device_level_config_.RawData(), &cfg, 1, find_stream,
                                                       __FILE__, __LINE__);
  // Reserve the whole tree's final slab and maximum count prefix before growth can be in flight. These bytes
  // are copied outside capture; the pooled slab is stable through ToHost, and the growth fence covers the DMA.
  size_t end_slab_bytes = 0;
  size_t end_counts_offset = 0;
  bool prefetch_tree_end = false;
  const FalcataPlan& end_plan = FalcataPlan::Get();
  if (cfg.whole_tree && end_plan.final_readback_first && end_plan.readback_kernel && AsyncTreeStart() &&
      nccl_communicator_ == nullptr && vec_num_targets_ <= 1 && !config_->linear_tree &&
      !(config_->use_quantized_grad && config_->quant_train_renew_leaf)) {
    end_slab_bytes = CUDATree::PooledDeviceBufferSize(config_->num_leaves);
    end_counts_offset = (end_slab_bytes + 255) / 256 * 256;
    const size_t max_count_bytes = static_cast<size_t>(config_->num_leaves) * sizeof(data_size_t);
    if (cuda_tree_pool_buffer_.Size() >= end_slab_bytes &&
        tree->cuda_threshold() == reinterpret_cast<const double*>(cuda_tree_pool_buffer_.RawDataReadOnly())) {
      prefetch_tree_end = EnsureReadbackStaging(end_counts_offset + max_count_bytes);
    }
  }
  // Count staging must already be reserved: the validated journal + exact final counts own all repair metadata.
  // No intermediate readback, new DMA/kernel, plan key or speculative host readiness is introduced.
  const bool final_map_only = inline_all_map && all_rows_root && prefetch_tree_end && end_plan.final_map_only &&
    end_plan.skip_empty_tail;
  // cuda_plan key device_level_graph: the levels' launches as one cached CUDA graph (captured on the first tree of a
  // pointer key, replayed on the next ones); otherwise enqueued as they are
  bool launched = false;
  if (FalcataPlan::Get().device_level_graph && !ForestSlotShared() && !device_level_graph_disabled_) {
    launched = LaunchDeviceLevelGraph(tree, cfg, num_levels, inline_all_map, final_map_only);
  }
  if (!launched) {
    EnqueueDeviceLevels(tree, cfg, num_levels, inline_all_map, final_map_only);
  }
  if (device_bynode_active_) {
    // every mask kernel joined the find stream inside the levels; a root tail draws on the host (SelectFeatureByNode)
    EndDeviceBynode();
  }
  const int max_total_splits = max_prefix_splits;
  bool active_prefix_readback = false;
  if (prefetch_tree_end) {
    // Same canonical whole/full-root subset; keep every V9 staging/producer/fence guard.
    // Launch OUTSIDE capture, behind its final apply/record join, on the existing find stream.
    if (inline_all_map) {
      active_prefix_readback = tree->LaunchActiveSlabReadback(readback_staging_device_, readback_staging_bytes_,
        cuda_data_partition_->cuda_leaf_num_data(), cuda_device_level_journal_.RawDataReadOnly(),
        max_total_splits, find_stream);
    }
    if (!active_prefix_readback) {
      // Unchanged DMA fallback uses the PINNED HOST pointer, not its mapped device alias.
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(readback_staging_, tree->cuda_threshold(), end_slab_bytes,
                                           cudaMemcpyDeviceToHost, find_stream));
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(static_cast<uint8_t*>(readback_staging_) + end_counts_offset,
        cuda_data_partition_->cuda_leaf_num_data(), static_cast<size_t>(config_->num_leaves) * sizeof(data_size_t),
        cudaMemcpyDeviceToHost, find_stream));
    }
  }
  // ---- the one growth readback: journal and applied split infos; the prefix also reads leaf-best entries.
  // Whole-tree slab/count copies, when requested, are covered by this same fence. ----
  CopyFromCUDADeviceToHostAsync<int>(pinned_device_level_journal_, cuda_device_level_journal_.RawData(),
    static_cast<size_t>(kDeviceLevelJournalHeader) + kDeviceLevelJournalRecord * static_cast<size_t>(max_total_splits),
    find_stream, __FILE__, __LINE__);
  if (!skip_host_cache) {
    cuda_data_partition_->PrefetchSplitBatchAsync(max_total_splits, find_stream);
  }
  if (!cfg.whole_tree) {
    cuda_best_split_finder_->PrefetchLeafBestSplitsAsync(max_total_splits + 1, find_stream);
  }
  if (record_split_infos && max_total_splits > 0) {
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(pinned_selective_prefix_split_infos_, cfg.split_info_journal,
                                         static_cast<size_t>(max_total_splits) * sizeof(CUDASplitInfo),
                                         cudaMemcpyDeviceToHost, find_stream));
  }
  CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(find_stream));
  const int* journal = pinned_device_level_journal_;
  if (journal[2] != 0) {
    Log::Fatal("device_levels: a device level left the gated regime (flags 0x%x)", journal[2]);
  }
  const int levels_applied = journal[0];
  const int total_splits = journal[1];
  CHECK_GE(levels_applied, 0);
  CHECK_LE(levels_applied, num_levels);
  CHECK_GE(total_splits, 0);
  CHECK_LE(total_splits, max_total_splits);
  // the splits of each level, from the records' level tags
  std::vector<int>& level_counts = device_level_journal_counts_;
  level_counts.resize(static_cast<size_t>(num_levels));
  CHECK(DeviceLevelJournalCounts(journal, total_splits, num_levels, level_counts.data()));
  for (int level = levels_applied; level < num_levels; ++level) {
    CHECK_EQ(level_counts[level], 0);
  }
  cuda_data_partition_->FinishDeviceLevels(num_levels, levels_applied, total_splits);
  if (!skip_host_cache) {
    cuda_data_partition_->ReadPrefetchedSplitBatch(total_splits, &device_level_batch_info_);
  }
  // ---- the host bookkeeping of every applied level, in level order (ApplyLevelBatched's tree mirrors,
  // FinishLevelBookkeeping's leaf arrays, bit widths and pairs) ----
  int num_splits = 0;
  int cursor = 0;
  pairs->clear();
  // the selective prefix: no device record to mirror (the selective flow starts its tree from the
  // constructor state; before, the mirrors replayed here were undone right after)
  const bool replay_mirrors = cfg.skip_tree_record == 0;
  // the selective prefix: the level bookkeeping reads the split-batch readback in place
  const bool lean_batch = record_split_infos;
  for (int level = 0; level < levels_applied; ++level) {
    const int level_splits = level_counts[level];
    CHECK_GE(level_splits, 1);
    CHECK_LE(cursor + level_splits, total_splits);
    if (cfg.whole_tree && level + 1 == num_levels) {
      prefix_final_leaves_before_ = tree->num_leaves();
    }
    if (!skip_host_cache) {
      device_level_applied_.clear();
    }
    for (int k = 0; k < level_splits; ++k) {
      const int* record = journal + kDeviceLevelJournalHeader + kDeviceLevelJournalRecord * (cursor + k);
      const int left_leaf = record[0];
      const int inner_feature = record[1];
      // the device numbers the level's right children after the leaves before it, in split order
      const int right_leaf = cursor + k + 1;
      if (replay_mirrors) {
        CHECK_EQ(tree->num_leaves(), right_leaf);
        tree->ReplaySplitHostMirrors(left_leaf, train_data_->RealFeatureIndex(inner_feature));
      }
      // the host's node keys follow the device's (a no-op without keyed per-node masks)
      BynodeKeySplit(left_leaf, right_leaf);
      if (!skip_host_cache) {
        device_level_applied_.push_back({left_leaf, right_leaf, hybrid_pair_slots_.RawData() + 2 * k,
                                         hybrid_pair_slots_.RawData() + 2 * k + 1});
      }
    }
    if (!skip_host_cache) {
      pairs->clear();
      if (lean_batch) {
        FinishLevelBookkeeping(device_level_applied_, device_level_batch_info_.data() + 18 * static_cast<size_t>(cursor),
                               pairs, &num_splits);
      } else {
        device_level_batch_slice_.assign(device_level_batch_info_.begin() + 18 * static_cast<size_t>(cursor),
                                         device_level_batch_info_.begin() + 18 * static_cast<size_t>(cursor + level_splits));
        FinishLevelBookkeeping(device_level_applied_, device_level_batch_slice_, pairs, &num_splits);
      }
    }
    cursor += level_splits;
  }
  CHECK_EQ(cursor, total_splits);
  CHECK_EQ(tree->num_leaves(), replay_mirrors ? total_splits + 1 : 1);
  const bool all_map_complete = inline_all_map && levels_applied == num_levels && level_counts[num_levels - 1] > 0;
  const bool deferred_final = final_map_only && all_map_complete;
  int deferred_splits = 0;
  int deferred_gaps = 0;
  int deferred_max_gap_blocks = 0;
  int64_t deferred_flat_blocks = 0;
  if (deferred_final) {
    // The growth fence covers both V15 mapped-copy and unchanged DMA fallback. Four owned counts AND the creation gap width survive staging
    // reuse. Reconstruct exactly the final select's leaf-ordered split/gap descriptors, with wide count sums.
    const int actual_leaves = total_splits + 1;
    CHECK_LE(actual_leaves, config_->num_leaves);
    deferred_splits = level_counts[num_levels - 1];
    const int leaves_before = actual_leaves - deferred_splits;
    CHECK_EQ(leaves_before, prefix_final_leaves_before_);
    const auto* counts = reinterpret_cast<const data_size_t*>(
      static_cast<const uint8_t*>(readback_staging_) + end_counts_offset);
    std::vector<uint8_t> selected(static_cast<size_t>(leaves_before), 0);
    int64_t root_rows = 0;
    for (int leaf = 0; leaf < actual_leaves; ++leaf) {
      CHECK_GE(counts[leaf], 0);
      root_rows += counts[leaf];
    }
    CHECK_EQ(root_rows, static_cast<int64_t>(num_data_));
    for (int k = 0; k < deferred_splits; ++k) {
      const int left = journal[kDeviceLevelJournalHeader +
                               kDeviceLevelJournalRecord * (total_splits - deferred_splits + k)];
      CHECK_GE(left, 0);
      CHECK_LT(left, leaves_before);
      CHECK_EQ(selected[left], 0);
      selected[left] = 1;
      const int64_t parent_rows = static_cast<int64_t>(counts[left]) + counts[leaves_before + k];
      CHECK_GT(parent_rows, 0);
      deferred_flat_blocks += (parent_rows + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) /
        SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
    }
    const int gap_block = cfg.gap_block_dim;  // descriptor creation unit, snapshotted BEFORE growth
    for (int leaf = 0; leaf < leaves_before; ++leaf) {
      if (selected[leaf] == 0 && counts[leaf] > 0) {
        ++deferred_gaps;
        const int blocks = static_cast<int>((static_cast<int64_t>(counts[leaf]) + gap_block - 1) / gap_block);
        deferred_max_gap_blocks = std::max(deferred_max_gap_blocks, blocks);
      }
    }
    CHECK_LE(deferred_flat_blocks, std::numeric_limits<int>::max());
    // FinishDeviceLevels has corrected K to R. This is positive STATIC completion (R=K), so undo just the final
    // role swap. Do not statically skip a swap during enqueue/capture: early R<K needs the original correction.
#ifdef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
    // Exact test boundary: every journal/count/mirror check has passed; NONE of the receipt is published.
    WholeMapOnlyFaultCheckpoint(this, WholeMapOnlyFaultPoint::BeforePublish);
#endif
    cuda_data_partition_->DeferDeviceLevelPartition(deferred_splits, deferred_gaps, deferred_max_gap_blocks,
                                                   static_cast<int>(deferred_flat_blocks), cfg.gap_block_dim);
  }
#ifdef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
  if (all_map_complete && !deferred_final) {
    // Matched deterministic eager control: same completed journal, no deferred receipt in this arm.
    WholeMapOnlyFaultCheckpoint(this, WholeMapOnlyFaultPoint::BeforePublish);
  }
#endif
  if (all_map_complete) {
    // Split children AND terminal gaps wrote every row, from old parent IDs when final partition is deferred.
    hybrid_map_final_written_ = true;
    hybrid_map_early_written_ = true;
    hybrid_map_early_num_leaves_ = tree->num_leaves();
    hybrid_map_early_pending_ = false;
    hybrid_map_residual_leaves_.clear();  // no current-tree subset upload exists in this whole path
  }
  if (skip_host_cache) {
    num_splits = total_splits;
    whole_host_leaf_cache_stale_ = total_splits > 0;
  }
  if (cfg.whole_tree) {
    if (num_splits == 0) {
      // The root output survives only for a stump. Read its original root struct before tree-end consumers.
      EnsureRootSumsReadBack(tree);
      // A level root search without a valid split is not final in the host level flows: their leaf-wise tail first
      // searches the root again with the classic per-leaf search (e.g. on a sampled compact layout whose batched
      // root histogram misses the sampled features). Run that same single search here. A valid split hands the tree
      // to Train's tail with this search cached; otherwise the stump is the host flows' result as well. Its classic
      // construct accumulates into an assumed-zero slot: clear the batched root histogram first, as Train's tail does.
      cuda_histogram_constructor_->ZeroHistSlots({0});
      EnqueuePairBestSplitSearch(tree, cuda_smaller_leaf_splits_->GetCUDAStruct(),
                                 cuda_larger_leaf_splits_->GetCUDAStruct(), smaller_leaf_index_, larger_leaf_index_,
                                 /*synchronize=*/true);
      // FindBestFromAllSplitsKernel's root criterion; that kernel consumes the leaf it picks, so the tail runs it
      SynchronizeCUDADevice(__FILE__, __LINE__);
      CUDASplitInfo root_split;
      CopyFromCUDADeviceToHost<CUDASplitInfo>(&root_split, cuda_best_split_finder_->leaf_best_split_info_ptr(0), 1,
                                              __FILE__, __LINE__);
      device_tree_root_tail_ = root_split.is_valid && root_split.gain > 0.0;
    }
    device_tree_completed_ = !device_tree_root_tail_;
    prefix_completes_tree_ = !device_tree_root_tail_;
    if (prefetch_tree_end && total_splits > 0) {
      CHECK_EQ(tree->num_leaves(), total_splits + 1);
      prefetched_tree_num_leaves_ = tree->num_leaves();
      prefetched_tree_counts_ = true;
    }
#ifdef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
    if (all_map_complete) {
      // Receipt plus map/completion/prefetch flags are ALL published; never throw half-way through a setter.
      WholeMapOnlyFaultCheckpoint(this, WholeMapOnlyFaultPoint::AfterPublish);
    }
#endif
    if (FalcataDebug().diag) {
      fprintf(stderr, "[hybrid-diag] device_tree_final_all_map: enabled=%d complete=%d "
              "final_splits=%d leaves=%d\n", inline_all_map ? 1 : 0, all_map_complete ? 1 : 0,
              all_map_complete ? level_counts[num_levels - 1] : 0,
              tree->num_leaves());
      fprintf(stderr, "[hybrid-diag] device_tree_final_map_only: enabled=%d deferred=%d "
              "splits=%d gaps=%d flat=%" PRId64 "\n", final_map_only ? 1 : 0, deferred_final ? 1 : 0,
              deferred_splits, deferred_gaps, deferred_flat_blocks);
      fprintf(stderr, "[hybrid-diag] device_tree_host_cache: skipped=%d split_info_bytes=%zu stale=%d\n",
              skip_host_cache ? 1 : 0,
              skip_host_cache ? size_t{0} : 18 * sizeof(int) * static_cast<size_t>(max_total_splits),
              whole_host_leaf_cache_stale_ ? 1 : 0);
      fprintf(stderr, "[hybrid-diag] device_tree_active_readback: queued=%d ready=%d leaves=%d\n",
              active_prefix_readback ? 1 : 0,
              active_prefix_readback && prefetched_tree_num_leaves_ > 0 ? 1 : 0, tree->num_leaves());
      // Separate line: preserve the existing completion marker and its frozen measurement parsers verbatim.
      fprintf(stderr, "[hybrid-diag] device_tree_end_readback: queued=%d ready=%d leaves=%d counts=%d\n",
              prefetch_tree_end ? 1 : 0, prefetched_tree_num_leaves_ > 0 ? 1 : 0, tree->num_leaves(),
              prefetched_tree_counts_ ? 1 : 0);
      fprintf(stderr, "[hybrid-diag] device_tree: complete depth=%d applied_levels=%d splits=%d "
              "growth_readbacks=1 host_tail=%d graph=%d hits=%d misses=%d\n", num_levels, levels_applied,
              total_splits, device_tree_root_tail_ ? 1 : 0, launched ? 1 : 0, device_level_graph_hits_,
              device_level_graph_misses_);
      // cuda_plan key device_level_graph_update: in-place exec updates under key churn (separate line, see above)
      fprintf(stderr, "[hybrid-diag] device_level_graph_update: updates=%d refused=%d stopped=%d last_refusal=%d\n",
              device_level_graph_update_hits_, device_level_graph_update_misses_,
              device_level_graph_update_failed_ ? 1 : 0, device_level_graph_update_last_refusal_);
    }
  } else {
    // the last search's result, as the host loop's own readback leaves it
    cuda_best_split_finder_->ReadPrefetchedLeafBestSplits(total_splits + 1, &host_leaf_best_splits_);
  }
  global_timer.Stop("CUDASingleGPUTreeLearner::TrainDeviceLevels");
  return num_splits;
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
    if (vec_num_targets_ > 1) {
      // snapshot the per-target child sums/outputs now, for the same reason
      // the categorical thresholds are snapshotted: the leaf's payload slot is
      // reused as soon as the (possibly recycled) leaf index is searched again
      const size_t fields = static_cast<size_t>(kNumVecPayloadFields) *
        static_cast<size_t>(vec_num_targets_);
      const size_t base = static_cast<size_t>(f.leaf) * fields;
      CHECK_LE(base + fields, sel_leaf_vec_payloads_.size());
      node.vec_payload.assign(sel_leaf_vec_payloads_.begin() + base,
                              sel_leaf_vec_payloads_.begin() + base + fields);
    }
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
    DebugPrintBynodeSplit(f.leaf, host_leaf_best_splits_[f.leaf].inner_feature_index);
    BynodeKeySplit(f.leaf, right);
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
  // the speculative flow gates the children on device structs the vector plane
  // fan-out has not refreshed yet (UseOneSyncPrefix is forced off in Init)
  CHECK_LE(vec_num_targets_, 1);
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
    // host-known; the next iteration's readback collects both. Nothing is
    // recorded on the tree stream here (SelectiveFinalize records the tree
    // after the loop), so the sync kernel has no record to wait for.
    EnqueueLevelBestSplitSearchSpeculative(tree, applied, nullptr);
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
  const bool use_batched_level_kernels =
    cuda_histogram_constructor_->SupportsBatchedLevel() &&
    cuda_best_split_finder_->SupportsBatchedLevel();
  // vector mode reaches selective growth only through the batched phases
  // (HybridGrowthUsable): the per-pair search fallback below is scalar-only
  CHECK(vec_num_targets_ <= 1 || use_batched_level_kernels);
  const bool hybrid_diag = FalcataDebug().diag;
  if (hybrid_diag) {
    static bool diag_logged = false;
    if (!diag_logged) {
      diag_logged = true;
      fprintf(stderr, "[hybrid-diag] selective-two-sync: batched_level_kernels=%d\n",
                   static_cast<int>(use_batched_level_kernels));
      fprintf(stderr, "[hybrid-diag] %s\n", cuda_histogram_constructor_->BatchedLevelGateDiag().c_str());
      fprintf(stderr, "[hybrid-diag] %s\n", cuda_best_split_finder_->BatchedLevelGateDiag().c_str());
    }
  }
  // vector mode: the ROOT level's planes are the fixed slab BeforeTrain filled
  // (pair 0's smaller role at [0, T), larger at [T, 2T)); every later level
  // reads the per-slot slab the plane fan-out writes
  CUDALeafSplitsStruct* vec_plane_slab =
    vec_num_targets_ > 1 ? vec_plane_structs_.RawData() : nullptr;
  std::vector<HybridAppliedSplit> applied;
  std::vector<int> batch_info;
  std::vector<int> child_leaves;
  // cuda_plan key selective_device_prefix: the first levels (those on which every valid candidate is selected) ran
  // as the device_levels prefix; its readback stands in for the next level's search and readback
  bool search_read_back = false;
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  if (use_batched_level_kernels) {
    search_read_back = TrainSelectiveDevicePrefix(tree, &pairs);
  }
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  growth_route_ = search_read_back ? GrowthRoute::kSelectivePrefix : GrowthRoute::kSelectiveTwoSync;
  while (true) {
    if (search_read_back) {
      search_read_back = false;
    } else {
      if (use_batched_level_kernels) {
        EnqueueLevelBestSplitSearch(tree, pairs, vec_plane_slab);
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
      if (vec_num_targets_ > 1) {
        cuda_best_split_finder_->CopyLeafVecPayloadsToHost(sel_num_allocated_, &sel_leaf_vec_payloads_);
      }
    }
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
    if (vec_num_targets_ > 1) {
      // the partition seeded the primary child structs (target-0 sums, plane-0
      // histogram pointers); fan them out into this level's plane structs with
      // per-target sums/values from each parent split's vector payload
      VectorLevelPlaneFanOut(applied);
      vec_plane_slab = vec_level_plane_slots_.RawData();
    }
  }
}

#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
bool CUDASingleGPUTreeLearner::TrainSelectiveDevicePrefix(CUDATree* tree, std::vector<HybridPendingPair>* pairs) {
  // Exactness (runs/device_default/SELECTIVE_PREFIX.md in the development notes): at level k the host's greedy
  // simulation (RunSelectiveLevel) holds the applied splits of levels < k plus the level's valid frontier, at most
  // (2^k - 1) + 2^k = 2^(k+1) - 1 candidates, all available (every frontier candidate's parent is applied and
  // selected). While 2^(k+1) <= num_leaves that is at most the budget num_leaves - 1, so the simulation pops every
  // candidate before the budget binds: no collapse, every valid leaf split. Invalid leaves never enter the frontier and
  // only lower the count. That is the device_levels prefix's "split every valid leaf" (the same is_valid scan), whose
  // levels DeviceLevelsToRun bounds by exactly 2^levels <= num_leaves (and max_depth - 1, kDeviceLevelMaxPairs).
  if (!FalcataPlan::Get().selective_device_prefix || UseOneSyncPrefix()) {
    return false;
  }
  CHECK(!DeviceTreeRequested());
  const int levels = DeviceLevelsToRun(/*batched_level_kernels=*/true, /*batched_level_apply=*/true);
  if (levels <= 0) {
    return false;
  }
  CHECK_LE(int64_t{1} << levels, static_cast<int64_t>(config_->num_leaves));
  if (!device_levels_ready_ && !SetupDeviceLevels(levels)) {
    device_levels_disabled_ = true;
    return false;
  }
  if (FalcataDebug().diag && !selective_prefix_logged_) {
    selective_prefix_logged_ = true;
    fprintf(stderr, "[hybrid-diag] selective_device_prefix: %d device-driven levels per tree\n", levels);
  }
  CHECK_EQ(static_cast<int>(pairs->size()), 1);
  const HybridPendingPair root_pair = (*pairs)[0];
  // the selective record of a split holds its parent's window, as leaf_data_start_ / leaf_num_data_ held it when
  // the host created the record; TrainDeviceLevels' bookkeeping overwrites those arrays level by level
  std::vector<data_size_t> window_start(static_cast<size_t>(config_->num_leaves) + 1, 0);
  std::vector<data_size_t> window_num(static_cast<size_t>(config_->num_leaves) + 1, 0);
  window_start[0] = leaf_data_start_[0];
  window_num[0] = leaf_num_data_[0];
  // the root level's search, as the host loop's first iteration enqueues it
  EnqueueLevelBestSplitSearch(tree, *pairs, nullptr);
  const int num_splits = TrainDeviceLevels(tree, levels, pairs, /*record_split_infos=*/true);
  const int* journal = pinned_device_level_journal_;
  const int levels_applied = journal[0];
  CHECK_EQ(journal[1], num_splits);
  // (no tree mirrors were replayed: the selective prefix records no device tree)
  CHECK_EQ(tree->num_leaves(), 1);
  // ---- the selective records, level by level as RunSelectiveLevel would have created them (every candidate
  // selected, no collapse, right-child indices fresh). Within a level the device numbers right children in
  // leaf-ascending order where the host numbers them in greedy order; which hybrid index (and histogram slot) a leaf
  // holds is read by no search, simulation or renumbering (see the notes), so the model is the same. ----
  const CUDASplitInfo* raw_infos = static_cast<const CUDASplitInfo*>(pinned_selective_prefix_split_infos_);
  int cursor = 0;
  int leaves = 1;
  const std::vector<int>& level_counts = device_level_journal_counts_;  // TrainDeviceLevels' tag counts
  CHECK_GE(static_cast<int>(level_counts.size()), levels_applied);
  for (int level = 0; level < levels_applied; ++level) {
    const int level_splits = level_counts[level];
    CHECK_GE(level_splits, 1);
    for (int k = 0; k < level_splits; ++k) {
      const int s = cursor + k;
      const int left = journal[kDeviceLevelJournalHeader + kDeviceLevelJournalRecord * s];
      const int right = leaves + k;
      const int* batch = device_level_batch_info_.data() + 18 * static_cast<size_t>(s);
      CHECK_EQ(batch[0], left);
      CHECK_EQ(batch[3], right);
      CHECK_LT(left, leaves);
      CHECK_LT(right, config_->num_leaves);
      CHECK(sel_leaf_live_[left]);
      // the children's windows tile the parent's (the left child keeps its start)
      CHECK_EQ(batch[2], window_start[left]);
      CHECK_EQ(batch[1] + batch[4], window_num[left]);
      CHECK_EQ(batch[5], batch[2] + batch[1]);
      const int new_id = static_cast<int>(sel_applied_.size());
      const int parent = sel_leaf_parent_[left];
      // the record written in place (one copy of the journaled record); CUDASplitInfo's assignment copies
      // the fields and never the device pointers (a default-constructed destination keeps them null), which is
      // what the host's own scrubbed best-split readback (ReadPrefetchedLeafBestSplits) hands over
      const CUDASplitInfo& raw = raw_infos[s];
      CHECK(raw.is_valid);
      CHECK_EQ(raw.inner_feature_index, journal[kDeviceLevelJournalHeader + kDeviceLevelJournalRecord * s + 1]);
      CHECK_EQ(raw.num_cat_threshold, 0);
      sel_applied_.emplace_back();
      SelectiveApplied& node = sel_applied_.back();
      node.parent = parent;
      node.left_leaf = left;
      node.right_leaf = right;
      node.kid_left = -1;
      node.kid_right = -1;
      node.dead = false;
      node.selected = true;
      node.data_start = window_start[left];
      node.num_data = window_num[left];
      node.info = raw;
      node.gain = node.info.gain;
      if (parent >= 0) {
        SelectiveApplied& p = sel_applied_[parent];
        if (p.left_leaf == left) {
          p.kid_left = new_id;
        } else {
          CHECK_EQ(p.right_leaf, left);
          p.kid_right = new_id;
        }
      }
      sel_leaf_parent_[left] = new_id;
      sel_leaf_parent_[right] = new_id;
      sel_leaf_live_[right] = 1;
      sel_leaf_depth_[right] = sel_leaf_depth_[left] + 1;
      ++sel_leaf_depth_[left];
      window_start[left] = batch[2];
      window_num[left] = batch[1];
      window_start[right] = batch[5];
      window_num[right] = batch[4];
    }
    cursor += level_splits;
    leaves += level_splits;
  }
  CHECK_EQ(cursor, num_splits);
  sel_num_allocated_ = leaves;
  sel_num_splits_ = num_splits;
  sel_stat_applied_ += num_splits;
  sel_stat_levels_ += levels_applied;
  // the tree is rebuilt from the records (SelectiveFinalize); the prefix recorded no device tree and replayed no host
  // mirrors, so the tree is still in its constructor state
  if (levels_applied == 0) {
    // the root found no valid split: the host loop's first readback, with its root pair
    pairs->assign(1, root_pair);
  }
  if (FalcataDebug().diag) {
    int frontier = 0;
    for (const HybridPendingPair& pair : *pairs) {
      frontier += host_leaf_best_splits_[pair.smaller].is_valid ? 1 : 0;
      frontier += pair.larger >= 0 && host_leaf_best_splits_[pair.larger].is_valid ? 1 : 0;
    }
    fprintf(stderr, "[hybrid-diag] selective_device_prefix: levels=%d applied_levels=%d splits=%d leaves=%d "
            "frontier=%d graph_hits=%d graph_misses=%d\n", levels, levels_applied, num_splits, leaves, frontier,
            device_level_graph_hits_, device_level_graph_misses_);
  }
  return true;
}
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED

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
    if (vec_num_targets_ > 1) {
      // both children's per-target outputs from the snapshotted payload; the
      // scalar fields above are its target-0 entries
      const int num_targets = vec_num_targets_;
      CHECK_EQ(static_cast<int>(node.vec_payload.size()), kNumVecPayloadFields * num_targets);
      const auto left_begin = node.vec_payload.begin() + kVecLeftValue * num_targets;
      const auto right_begin = node.vec_payload.begin() + kVecRightValue * num_targets;
      s.vec_left_values.assign(left_begin, left_begin + num_targets);
      s.vec_right_values.assign(right_begin, right_begin + num_targets);
    }
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
  CHECK(!whole_host_leaf_cache_stale_);
  global_timer.Start("CUDASingleGPUTreeLearner::TrainSelective");
  selective_active_ = true;
  SelectiveReset();
  // two struct slots per split of a level (levels hold at most (num_leaves+1)/2
  // splits: a level's splits plus their ancestor chains all fit the selection)
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
  if (UseOneSyncPrefix()) {
    growth_route_ = GrowthRoute::kSelectiveOneSync;
    TrainSelectiveOneSync(tree);
  } else {
    TrainSelectiveTwoSync(tree);
  }
  SelectiveFinalize(tree);
  selective_active_ = false;
  global_timer.Stop("CUDASingleGPUTreeLearner::TrainSelective");
}

Tree* CUDASingleGPUTreeLearner::Train(const score_t* gradients,
  const score_t* hessians, bool is_first_tree) {
  MaybeInitColMajorFill();
  if (construct_saturation_floor_ > 0 &&
      FalcataPlan::Get().batch_construct_saturation_floor != construct_saturation_floor_) {
    // see Init; written only on a change (forest slots train concurrently and all hold the same value)
    FalcataPlan::Mutable().batch_construct_saturation_floor = construct_saturation_floor_;
  }
  gradients_ = gradients;
  hessians_ = hessians;
  // cuda_plan key leaf_map_no_zero: the gate is config-constant, so the bagged subset of the next round (set before
  // that round's Train) sees the same verdict
  const bool leaf_map_zero_dead = LeafMapZeroDead();
  cuda_data_partition_->SetLeafMapZeroDead(leaf_map_zero_dead);
  if (tree_in_progress_) {
    // the last tree threw mid-way (a device-memory failure, which a forest group retries, or any error a caller
    // caught): it may have written histogram slots beyond the last completed tree's count, and the accumulating
    // constructs (quantized, float atomics) would add the leftovers into this tree
    AbortedTreeReset();
  }
  tree_in_progress_ = true;
  if (forest_slot_ && !forest_private_column_view_) {
    ++forest_main_trees_started_;
  }
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  if (device_bynode_active_) {
    EndDeviceBynode();  // the last tree threw between its root search and its device levels
  }
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  global_timer.Start("CUDASingleGPUTreeLearner::BeforeTrain");
  BeforeTrain();
  global_timer.Stop("CUDASingleGPUTreeLearner::BeforeTrain");
  if (bynode_keyed_) {
    // cuda_plan key level_bynode_masks: the root's node key names the tree (this learner's tree count); every
    // other node's key follows from its path (BynodeKeySplit)
    if (bynode_node_key_.size() < static_cast<size_t>(config_->num_leaves) + 2) {
      bynode_node_key_.resize(static_cast<size_t>(config_->num_leaves) + 2, 0);
    }
    bynode_node_key_[0] = ColSampler::MixKey(bynode_tree_count_ ^ 0x3C6EF372FE94F82BULL);
    ++bynode_tree_count_;
  }
  hybrid_map_final_written_ = false;
  hybrid_map_early_written_ = false;
  hybrid_map_early_num_leaves_ = 0;
  hybrid_map_early_pending_ = false;  // map completion belongs only to the current tree
  prefetched_tree_num_leaves_ = 0;
  prefetched_tree_counts_ = false;
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
    cuda_tree_pool_buffer_.RawData(), std::max(1, vec_num_targets_), &cuda_tree_cat_arena_));
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
  CheckMultiGPUFlowUsable(num_splits_done);
  if (config_->use_quantized_grad && !(num_splits_done == 0 && HybridGrowthUsable() && !UseSelectiveGrowth())) {
    // every quantized flow but the level prefix reads the root sums on the host: a no-op unless BeforeTrain
    // deferred them (async_tree_start; TrainLevelWisePrefix reads them back itself)
    EnsureRootSumsReadBack(tree.get());
  }
  prefix_completes_tree_ = false;
  prefix_final_leaves_before_ = -1;
  device_tree_root_tail_ = false;
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  device_tree_completed_ = false;
#ifdef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
  WholeMapOnlyFaultCheckpoint(this, WholeMapOnlyFaultPoint::AfterTrainFlagsReset);
#endif
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  if (FalcataDebug().diag && select_features_by_node_) {
    // per-node masks: which draws and which flow (once per process; tests/python_package_test/test_cuda_bynode_level.py)
    static bool bynode_diag_logged = false;
    if (!bynode_diag_logged) {
      bynode_diag_logged = true;
      const bool level = num_splits_done == 0 && HybridGrowthUsable();
      fprintf(stderr, "[bynode-diag] masks=%s flow=%s%s\n", bynode_keyed_ ? "keyed" : "shared-stream",
              !level ? "per-split" : UseSelectiveGrowth() ? "selective" : "prefix",
              !level ? "" : UseOneSyncPrefix() ? "-one-sync" : "-two-sync");
    }
  }
  growth_route_ = GrowthRoute::kClassic;  // the controllers below name themselves
  device_levels_refusal_ = DeviceTreeReason::kNone;
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  if (FalcataDebug().diag && FalcataPlan::Get().device_levels && !DeviceTreeRequested()) {
    // the device_levels prefix's own fallbacks (a refused complete device tree reports on its reason line instead)
    if (!config_->use_quantized_grad || num_splits_done != 0 || !HybridGrowthUsable()) {
      LogDeviceLevelsFallback(DeviceTreeRefusal());
    } else if (UseSelectiveGrowth() && !FalcataPlan::Get().selective_device_prefix) {
      // (with selective_device_prefix the prefix reports its own fallbacks)
      LogDeviceLevelsFallback(DeviceTreeReason::kPlanOff);
    }
    device_levels_refusal_ = DeviceTreeReason::kNone;  // only the flows' own refusals name a route's prefix
  }
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  // a level prefix without a split leaves its batched root histogram in leaf 0's slot, which the tail's classic
  // root construct would accumulate onto
  bool level_root_hist_built = false;
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
      level_root_hist_built = num_splits_done == 0;
      batched_apply_ran = num_splits_done > 0;
      // every current leaf already has a cached best-split candidate; the first
      // tail iteration must not redo the pair search (nor the whole tree's classic root search)
      pair_search_cached = num_splits_done > 0 || device_tree_root_tail_;
    }
  }
  // cuda_plan key skip_empty_tail: after a level prefix that ended on a complete final level the tail's search can
  // only find every cached candidate invalid; report the stop it would report without running it (the batched
  // apply invalidates the final level's children)
  const bool tail_known_empty =
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
    device_tree_completed_ ||
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
    (FalcataPlan::Get().skip_empty_tail && batched_apply_ran && prefix_completes_tree_);
  CHECK(!whole_host_leaf_cache_stale_ || tail_known_empty);
  if (tail_known_empty && !selective_handled && num_splits_done < config_->num_leaves - 1) {
    Log::Warning("No further splits with positive gain, training stopped with %d leaves.", (num_splits_done + 1));
  }
  for (int i = num_splits_done; !selective_handled && !tail_known_empty && i < config_->num_leaves - 1; ++i) {
    if (!pair_search_cached) {
      if (i == 0 && level_root_hist_built) {
        // the classic root construct accumulates into an assumed-zero slot: clear the batched root histogram
        cuda_histogram_constructor_->ZeroHistSlots({0});
      }
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

    if (FalcataDebug().bynode && select_features_by_node_ &&
        best_leaf_index_ < static_cast<int>(debug_bynode_masks_.size())) {
      const std::vector<int8_t>& mask = debug_bynode_masks_[best_leaf_index_];
      std::string bits(mask.size(), '0');
      for (size_t f = 0; f < mask.size(); ++f) {
        bits[f] = mask[f] ? '1' : '0';
      }
      fprintf(stderr, "BYNODE leaf=%d feat=%d mask=%s\n", best_leaf_index_,
              leaf_best_split_feature_[best_leaf_index_], bits.c_str());
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
  // cuda_plan key async_tree_start: every upload this tree made from a function-local buffer has completed
  cuda_histogram_constructor_->ReleaseTreeStartUploads();
  tree_start_uploads_.Release();
  tree_meta_batch_.Release();
  // The counts the split finder recorded are estimates (see below): the data partition's exact per-leaf counts
  // cuda_plan key readback_kernel: when the exact counts were already copied into the staging buffer behind the
  // tree's slab (one stream synchronize for both, see early_to_host below), their offset there; 0 = not prefetched
  size_t prefetched_counts_offset = 0;
  const auto sync_node_counts = [&]() {
    if (nccl_communicator_ == nullptr && tree->num_leaves() > 0) {
      std::vector<data_size_t> leaf_num_data(static_cast<size_t>(tree->num_leaves()));
      const size_t count_bytes = leaf_num_data.size() * sizeof(data_size_t);
      if (prefetched_counts_offset > 0) {
        std::memcpy(leaf_num_data.data(), static_cast<const uint8_t*>(readback_staging_) + prefetched_counts_offset,
                    count_bytes);
      } else if (EnsureReadbackStaging(count_bytes)) {
        // cuda_plan key readback_kernel: copy kernel on the default stream into the mapped pinned staging
        CopyFromCUDADeviceToMappedHost(readback_staging_device_, readback_staging_,
                                       cuda_data_partition_->cuda_leaf_num_data(), count_bytes, __FILE__, __LINE__);
        std::memcpy(leaf_num_data.data(), readback_staging_, count_bytes);
      } else {
        CopyFromCUDADeviceToHost<data_size_t>(leaf_num_data.data(),
          cuda_data_partition_->cuda_leaf_num_data(), leaf_num_data.size(), __FILE__, __LINE__);
      }
      tree->SyncNodeCountsFromPartition(leaf_num_data);
    }
  };
  // cuda_plan key async_tree_start: the tree's host copy and the exact counts read nothing the leaf-map pass
  // writes, so they are read back first and the pass runs while the host finalizes the tree (the readbacks
  // would otherwise wait for it); not with leaf renewal or linear trees, which use the map before the copy
  const bool early_to_host = AsyncTreeStart() && batched_apply_ran && !selective_handled &&
    !(config_->use_quantized_grad && config_->quant_train_renew_leaf) && !config_->linear_tree;
  // cuda_plan key readback_kernel: the tree's pooled slab is read back through the mapped pinned staging
  const auto tree_to_host = [&](const std::function<void()>& after_device_copy, const bool slab_in_staging) {
    const size_t slab_bytes = CUDATree::PooledDeviceBufferSize(config_->num_leaves);
    if (EnsureReadbackStaging(slab_bytes)) {
      tree->ToHost(readback_staging_, readback_staging_device_, readback_staging_bytes_, after_device_copy,
                   slab_in_staging);
    } else {
      tree->ToHost(nullptr, nullptr, 0, after_device_copy);
    }
  };
  if (early_to_host) {
    // cuda_plan key readback_kernel: the exact counts' copy kernel goes out first into the staging buffer past the
    // slab, so the slab's synchronize in ToHost covers both copies (same default stream; nothing between writes the
    // counts) and the counts need no round trip of their own
    const size_t slab_bytes = CUDATree::PooledDeviceBufferSize(config_->num_leaves);
    const size_t counts_offset = (slab_bytes + 255) / 256 * 256;
    const size_t count_bytes = static_cast<size_t>(tree->num_leaves()) * sizeof(data_size_t);
    // cuda_plan key final_readback_first: both already copied with the final level's readback (PrefetchTreeReadback)
    const bool slab_prefetched = prefetched_tree_num_leaves_ > 0 && prefetched_tree_num_leaves_ == tree->num_leaves();
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
    if (FalcataDebug().diag && device_tree_completed_) {
      fprintf(stderr, "[hybrid-diag] device_tree_end_readback: consume slab=%d counts=%d leaves=%d\n",
              slab_prefetched ? 1 : 0, slab_prefetched && prefetched_tree_counts_ ? 1 : 0, tree->num_leaves());
    }
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
    if (slab_prefetched) {
      if (prefetched_tree_counts_) {
        prefetched_counts_offset = counts_offset;
      }
    } else if (nccl_communicator_ == nullptr && tree->num_leaves() > 0 &&
        EnsureReadbackStaging(counts_offset + count_bytes)) {
      LaunchCopyToMappedHost(static_cast<uint8_t*>(readback_staging_device_) + counts_offset,
                             cuda_data_partition_->cuda_leaf_num_data(), count_bytes, __FILE__, __LINE__);
      prefetched_counts_offset = counts_offset;
    }
    // cuda_plan key final_readback_first: the final level's residual-leaf map pass goes out right after the tree's
    // device copy and runs while the host scatters and releases it; the exact counts are read before it (by the
    // slab's synchronize when prefetched above, otherwise here, applied after the copy as below)
    std::function<void()> after_device_copy;
    std::vector<data_size_t> counts_read_first;
    if (hybrid_map_early_pending_) {
      if (prefetched_counts_offset == 0 && nccl_communicator_ == nullptr && tree->num_leaves() > 0) {
        counts_read_first.resize(static_cast<size_t>(tree->num_leaves()));
        CopyFromCUDADeviceToHost<data_size_t>(counts_read_first.data(),
          cuda_data_partition_->cuda_leaf_num_data(), counts_read_first.size(), __FILE__, __LINE__);
      }
      CUDATree* const cuda_tree = tree.get();
      after_device_copy = [this, cuda_tree]() { FlushEarlyLeafMap(cuda_tree); };
    }
    tree_to_host(after_device_copy, slab_prefetched);
    if (!counts_read_first.empty()) {
      tree->SyncNodeCountsFromPartition(counts_read_first);
    } else {
      sync_node_counts();
    }
  }
  const bool device_tree_map_diag =
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
    FalcataDebug().diag && device_tree_completed_;
#else
    false;
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  const char* device_tree_map_route = "none";
  if (batched_apply_ran) {
    // the batched apply defers the per-level row -> leaf map scatter; write the
    // map once from the final leaf windows before any tree-end consumer
    // (AddPredictionToScore / refit stats / linear trees) reads it. The classic
    // tail splits above kept their inline updates, which this pass subsumes.
    // When the final level's split-inner already wrote the map inline
    // (final_partial_level), only earlier-finalized leaves remain.
    if (hybrid_map_final_written_) {
      // cuda_plan key early_leaf_map: already written with the final level, unless the leaf-wise tail split since
      // (its classic splits wrote their rows inline; the pass then runs here as without the key)
      if (!hybrid_map_early_written_ || tree->num_leaves() != hybrid_map_early_num_leaves_) {
        cuda_data_partition_->MaterializeHybridLeafMapSubset(hybrid_map_residual_leaves_);
        device_tree_map_route = "subset";
      } else {
        device_tree_map_route = "inline-all";
      }
    } else {
      cuda_data_partition_->MaterializeHybridLeafMap(tree->num_leaves());
      device_tree_map_route = "full";
    }
  } else if (leaf_map_zero_dead) {
    // cuda_plan key leaf_map_no_zero: no batched apply ran (a stump: the level prefix found no split), so the map was
    // neither zeroed nor written; the pass writes leaf 0's window, every in-bag row, as the zero store did, for the
    // tree-end readers below (GBDT adds a stump's value without the map)
    cuda_data_partition_->MaterializeHybridLeafMap(tree->num_leaves());
    device_tree_map_route = "stump";
  }
  if (device_tree_map_diag) {
    fprintf(stderr, "[hybrid-diag] device_tree_final_all_map_branch: route=%s leaves=%d\n",
            device_tree_map_route, tree->num_leaves());
  }
  hybrid_map_final_written_ = false;
  hybrid_map_early_written_ = false;
  hybrid_map_early_pending_ = false;
  prefetched_tree_num_leaves_ = 0;
  prefetched_tree_counts_ = false;
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
  if (!selective_handled && !early_to_host) {
    // the selective path rebuilt the host tree from captured split info and
    // released the device arrays in RebuildFromHostSplits already
    tree_to_host(std::function<void()>(), false);
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
  if (!early_to_host) {
    sync_node_counts();
  }
  // only the leaf histogram slots this tree used can be dirty; the next
  // BeforeTrain zeroes just that prefix (single-GPU only: the NCCL path keeps
  // the conservative full zeroing). Selective growth dirties every hybrid slot
  // it ever allocated (recycled slots are re-zeroed on the fly, but the last
  // occupants remain), so it reports its allocation peak instead.
  // cuda_plan key dirty_final_slots: when the level prefix ended on a final level that completed the tree and the
  // tail was skipped, every histogram of the tree was built before that level, into the slots of the leaves that
  // existed then (a slot is handed to a leaf at its creation, by its index); the slots the final level handed to
  // its children were never written, so the prefix of the leaves before it is all that can be dirty
  if (nccl_communicator_ == nullptr) {
    int num_dirty = selective_handled ? sel_last_peak_ : tree->num_leaves();
    if (!selective_handled && tail_known_empty && prefix_final_leaves_before_ > 0 &&
        FalcataPlan::Get().dirty_final_slots) {
      num_dirty = std::min(num_dirty, prefix_final_leaves_before_);
    }
    cuda_histogram_constructor_->SetNumDirtyLeaves(num_dirty);
  }
  if (FalcataDebug().diag && tree->cat_level_launches() > 0) {
    // cuda_plan key cat_level_record: what the level-batched record saved on this tree against the per-split loop
    // (one SplitCategoricalKernel launch and two blocking 4-byte D2D copies per categorical split, one blocking
    // length readback per categorical level, four record copies in ToHost)
    const int levels = tree->cat_level_launches();
    const int splits = tree->cat_level_splits();
    fprintf(stderr, "[hybrid-diag] cat_level_record: levels=%d cat_splits=%d record_launches=%d "
            "saved_launches=%d saved_d2d=%d saved_length_readbacks=%d tohost_copies=%d overflow_readbacks=%d\n",
            levels, splits, levels, splits - levels, 2 * splits, levels, tree->cat_record_one_copy() ? 1 : 4,
            tree->cat_overflow_readbacks());
  }
  last_tree_is_linear_ = false;
  if (config_->linear_tree) {
    // gradients_/hessians_ are device pointers after BeforeTrain (the Train args stay on host)
    CalculateLinearCUDA(tree.get(), gradients_, hessians_, is_first_tree);
    last_tree_is_linear_ = true;
  }
  // colmajor_direct, compact regime: the full view stays only while consecutive trees read it (decided after the
  // tree, since the per-leaf construct may first read it mid-tree)
  cuda_histogram_constructor_->ReleaseUnusedFullView();
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  if (FalcataDebug().diag) {
    LogDeviceTreeFallback();
  }
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  tree_in_progress_ = false;
  return tree.release();
}

int CUDASingleGPUTreeLearner::ApplySplit(CUDATree* tree, const CUDASplitInfo* best_split_info,
                                         const int leaf_index) {
  CHECK(!whole_host_leaf_cache_stale_);
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

  BynodeKeySplit(leaf_index, right_leaf_index);
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
                              cuda_smaller_leaf_splits_->GetCUDAStructRef(),
                              cuda_larger_leaf_splits_->GetCUDAStructRef(),
                              &leaf_num_data_[leaf_index],
                              &leaf_num_data_[right_leaf_index],
                              &leaf_data_start_[leaf_index],
                              &leaf_data_start_[right_leaf_index],
                              &leaf_sum_hessians_[leaf_index],
                              &leaf_sum_hessians_[right_leaf_index],
                              &leaf_sum_gradients_[leaf_index],
                              &leaf_sum_gradients_[right_leaf_index],
                              global_num_data_in_leaf_.data() + leaf_index,
                              global_num_data_in_leaf_.data() + right_leaf_index);
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
    SelectFeatureByNode(tree, smaller_leaf_index_, larger_leaf_index_);
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
  if (count_hist_cache_ != nullptr) {
    count_hist_cache_->Clear();  // cuda_plan key forest_count_hist: the gradient arrays get other values
  }
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
  if (cuda_gradient_discretizer_ != nullptr) {
    // Row buffers, reduction geometry and error-feedback residuals belong to the old dataset. Keep the Philox
    // tree counter, but initialize new rows with no residual history.
    const int iteration = cuda_gradient_discretizer_->iteration();
    cuda_gradient_discretizer_->Init(num_data_, config_->num_leaves, train_data_->num_features(), train_data_);
    cuda_gradient_discretizer_->SetIteration(iteration);
  }
  // training data changed: the predrawn column sample and any prefilled
  // compact view belong to the old data / re-seeded sampler
  next_tree_col_sample_ready_ = false;
  cuda_histogram_constructor_->InvalidateCompactPrefill();
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  ReleaseDeviceLevelGraphs();  // captured buffer pointers may have changed
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  // the histogram constructor dropped the old data's column-major copy: decide again after a tree on the new data
  colmajor_fill_pending_ = cuda_histogram_constructor_->ColMajorFillApplicable();
  colmajor_fill_trees_seen_ = 0;
  // colmajor_direct: the old data's view is gone; choose a regime for the new data and build its view
  InitViewRegime("update(train_set=...)");
}

void CUDASingleGPUTreeLearner::ResetConfig(const Config* config) {
  // Consume saved apply scratch before any config-dependent histogram/partition resize. No-op normally.
  cuda_data_partition_->EnsureLevelPartitioned();
  const int old_num_leaves = config_->num_leaves;
  SerialTreeLearner::ResetConfig(config);
  // cuda_plan key leaf_map_no_zero: the verdict is re-derived by the next Train; a bagged subset set before it zeroes
  if (cuda_data_partition_ != nullptr) {
    cuda_data_partition_->SetLeafMapZeroDead(false);
  }
  if (count_hist_cache_ != nullptr) {
    count_hist_cache_->Clear();  // cuda_plan key forest_count_hist: built again on the next tree
  }
  count_hist_logged_ = false;
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
    if (cuda_gradient_discretizer_ != nullptr) {
      // Host bookkeeping of histogram widths is leaf-indexed as well. A budget increase must not index the old
      // vectors; reinitialize only the base metadata, retaining the device residuals and stochastic tree counter.
      cuda_gradient_discretizer_->GradientDiscretizer::Init(
        num_data_, config_->num_leaves, train_data_->num_features(), train_data_);
      cuda_leaf_gradient_stat_buffer_.Resize(config_->num_leaves);
      cuda_leaf_hessian_stat_buffer_.Resize(config_->num_leaves);
    }
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
  ReleaseDeviceLevelGraphs();  // captured buffer pointers / budgets may have changed
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  // colmajor_direct: a new feature_fraction may call for the other regime (other config changes keep it, so a
  // per-round reset_parameter callback does not rebuild the view)
  if (cuda_histogram_constructor_->colmajor_direct() && config_->feature_fraction != view_decided_ff_) {
    InitViewRegime("a feature_fraction change");
  }
}

void CUDASingleGPUTreeLearner::PresizeLevelBuffers() {
  size_t free_bytes = 0, total_bytes = 0;
  CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_bytes, &total_bytes));
  const size_t kMaxPresizeBytes = std::max<size_t>(size_t{256} << 20, total_bytes / 8);
  int max_pairs = config_->num_leaves / 2 + 2;
  if (config_->max_depth > 0 && config_->max_depth < 30) {
    // a level of a depth-limited tree has at most 2^(max_depth - 1) pairs
    max_pairs = std::min(max_pairs, (1 << (config_->max_depth - 1)) + 2);
  }
  cuda_data_partition_->PresizeLevelBuffers();
  cuda_best_split_finder_->PresizeLevelOutput(max_pairs, kMaxPresizeBytes);
  if (config_->use_quantized_grad) {
    cuda_histogram_constructor_->PresizeBitChangeScratch(max_pairs, kMaxPresizeBytes);
  }
}

CUDASingleGPUTreeLearner::TreeCounters CUDASingleGPUTreeLearner::tree_counters() const {
  TreeCounters counters;
  counters.bynode = bynode_tree_count_;
  counters.discretizer = cuda_gradient_discretizer_ ? cuda_gradient_discretizer_->iteration() : 0;
  return counters;
}

void CUDASingleGPUTreeLearner::SetTreeCounters(const TreeCounters& counters) {
  bynode_tree_count_ = counters.bynode;
  if (cuda_gradient_discretizer_) {
    cuda_gradient_discretizer_->SetIteration(counters.discretizer);
  }
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
    cuda_data_partition_->EnsureLevelPartitioned();  // cuda_plan key final_map_only: reads the leaf windows
    if (boosting_on_cuda_) {
      obj->RenewTreeOutputCUDA(train_score, cuda_data_partition_->cuda_data_indices(),
                               cuda_data_partition_->cuda_leaf_num_data(), cuda_data_partition_->cuda_leaf_data_start(),
                               cuda_tree->num_leaves(), cuda_tree->cuda_leaf_value_ref());
      cuda_tree->SyncLeafOutputFromCUDAToHost();
    } else {
      const data_size_t* bag_mapper = nullptr;
      CHECK(!whole_host_leaf_cache_stale_);
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
  cuda_data_partition_->EnsureLevelPartitioned();  // cuda_plan key final_map_only: reads the leaf windows
  ReduceLeafStat(cuda_tree.get(), gradients, hessians, cuda_data_partition_->cuda_data_indices());
  cuda_tree->SyncLeafOutputFromCUDAToHost();
  return cuda_tree.release();
}

Tree* CUDASingleGPUTreeLearner::FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                                                  const score_t* gradients, const score_t* hessians) const {
  // ResetByLeafPred replaces only the map; smoothed refit can still read the prior index bytes.
  cuda_data_partition_->EnsureLevelPartitioned();
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
    // at least one word, also when there are NaN's in the categorical values; a split holding the value max_cat_value needs word max_cat_value / 32, so max_cat_value / 32 + 1 words: the
    // former (max_cat_value + 31) / 32 was one word short when max_cat_value is a positive multiple of 32, and the
    // level bitset kernel then wrote that word into the next split's slab region (or past the slab)
    const size_t cuda_bitset_max_size = static_cast<size_t>(std::max(max_cat_value, 0)) / 32 + 1;
    const size_t cuda_bitset_inner_max_size = std::max(static_cast<size_t>((max_cat_num_bin + 31) / 32), static_cast<size_t>(1));
    batch_cat_bitset_stride_ = cuda_bitset_max_size;
    batch_cat_bitset_inner_stride_ = cuda_bitset_inner_max_size;
    AllocateCUDAMemory<uint32_t>(&cuda_bitset_, cuda_bitset_max_size, __FILE__, __LINE__);
    AllocateCUDAMemory<uint32_t>(&cuda_bitset_inner_, cuda_bitset_inner_max_size, __FILE__, __LINE__);
    // presize the trees' categorical record buffers to the per-tree bound (num_leaves - 1 categorical splits of
    // at most the per-split bitset words), capped at 1 MiB each; past the cap they grow by doubling
    const size_t max_cat_splits = static_cast<size_t>(std::max(config_->num_leaves - 1, 1));
    const size_t kCatArenaReserveCapWords = static_cast<size_t>(1) << 18;
    cuda_tree_cat_arena_.bitset.Reserve(std::min(max_cat_splits * cuda_bitset_max_size, kCatArenaReserveCapWords));
    cuda_tree_cat_arena_.bitset_inner.Reserve(
      std::min(max_cat_splits * cuda_bitset_inner_max_size, kCatArenaReserveCapWords));
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

void CUDASingleGPUTreeLearner::AbortedTreeReset() {
  // what the next tree would otherwise take from the last completed one: every histogram slot zeroed (as after
  // Init), the count scratch zeroed, no selective-growth peak or pending readback carried over
  SynchronizeCUDADevice(__FILE__, __LINE__);  // the aborted tree's kernels finish before the next tree's zeroing
  // If a validated completion was published before a later error, consume it before scratch is reused.
  // Failures before publication never create a pending receipt; BeforeTrain replaces the root assignment.
#ifdef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
  WholeMapOnlyFaultCheckpoint(this, WholeMapOnlyFaultPoint::BeforeRecoveryRepair);
#endif
  cuda_data_partition_->EnsureLevelPartitioned();
#ifdef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
  WholeMapOnlyFaultCheckpoint(this, WholeMapOnlyFaultPoint::AfterRecoveryRepair);
#endif
  cuda_histogram_constructor_->SetNumDirtyLeaves(-1);
  cuda_histogram_constructor_->MarkCountScratchDirty();
  sel_last_peak_ = 0;
  hybrid_map_final_written_ = false;
  hybrid_map_early_written_ = false;
  hybrid_map_early_num_leaves_ = 0;
  hybrid_map_early_pending_ = false;
  prefetched_tree_num_leaves_ = 0;
  prefetched_tree_counts_ = false;
#ifdef FALCATA_WHOLE_MAP_ONLY_FAULT_TEST
  WholeMapOnlyFaultCheckpoint(this, WholeMapOnlyFaultPoint::RecoveryCleared);
#endif
}

void CUDASingleGPUTreeLearner::MaybeInjectTreeFailure(const int num_leaves) {
  // test hook (FALCATA_TEST_FOREST_ABORT_TREE=n): the n-th tree (0-based) the booster's learner grows in forest
  // mode fails once it has at least 4 leaves, as a device-memory failure inside a tree would
  static const int inject_tree = [] {
    const char* v = std::getenv("FALCATA_TEST_FOREST_ABORT_TREE");
    return v != nullptr ? std::atoi(v) : -1;
  }();
  if (inject_tree < 0 || !forest_slot_ || forest_private_column_view_ || num_leaves < 4 || injected_tree_failure_ ||
      forest_main_trees_started_ - 1 != inject_tree) {
    return;
  }
  injected_tree_failure_ = true;
  throw std::runtime_error("[CUDA] out of memory allocating 0.00 GiB (injected inside a tree for a test)");
}

const CUDACountHistCodes* CUDASingleGPUTreeLearner::CountHistCodesForTree() {
  // cuda_plan key forest_count_hist (docs/design/forest-count-histograms.md): a random forest's constant gradient
  // array of a qualifying objective, on the deterministic construct's path, whose values pass the exactness check
  if (!FalcataPlan::Get().forest_count_hist || count_hist_cache_ == nullptr ||
      config_->boosting != std::string("rf")) {
    return nullptr;
  }
  const std::string& objective = config_->objective;
  std::string why;
  if (objective != "binary" && objective != "multiclass" && objective != "multiclassova" && objective != "regression") {
    why = "objective " + objective;
  } else if (!boosting_on_cuda_) {
    why = "host objective";
  } else if (config_->use_quantized_grad) {
    why = "quantized gradients";
  } else if (config_->data_sample_strategy == std::string("goss")) {
    why = "goss";
  } else if (train_data_->metadata().weights() != nullptr) {
    why = "weights";
  } else if (nccl_communicator_ != nullptr || vec_num_targets_ > 1) {
    why = "multi-GPU or vector leaves";
  } else if (NonquantDeviceTreeShape()) {
    why = "device tree";
  } else if (!cuda_histogram_constructor_->CountConstructSupported()) {
    why = "construct (sparse, large-bin partition, fp32 histograms or the shared table)";
  }
  const CUDACountHistCodes* codes = nullptr;
  if (why.empty()) {
    codes = count_hist_cache_->Get(gradients_, hessians_, num_data_, &why);
  }
  if (FalcataDebug().diag && !count_hist_logged_) {
    count_hist_logged_ = true;
    fprintf(stderr, "[count-hist-diag] %s%s\n", codes != nullptr ? "count construct" : "deterministic construct: ",
            codes != nullptr ? "" : why.c_str());
  }
  return codes;
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

void CUDASingleGPUTreeLearner::DrawKeyedNodeMask(const int leaf, int8_t* out, std::vector<int>* inner_features) {
  col_sampler_.GetByNodeKeyed(bynode_node_key_[leaf], out, inner_features);
  if (FalcataDebug().bynode) {
    if (static_cast<int>(debug_bynode_masks_.size()) < config_->num_leaves + 2) {
      debug_bynode_masks_.resize(static_cast<size_t>(config_->num_leaves) + 2);
    }
    debug_bynode_masks_[leaf].assign(out, out + train_data_->num_features());
  }
}

void CUDASingleGPUTreeLearner::UploadLevelNodeMasks(const std::vector<int>& leaves) {
  if (!bynode_level_masks_) {
    return;
  }
  bynode_level_leaves_.clear();
  for (const int leaf : leaves) {
    if (leaf >= 0) {
      bynode_level_leaves_.push_back(leaf);
    }
  }
  if (bynode_level_leaves_.empty()) {
    return;
  }
  const size_t num_features = static_cast<size_t>(train_data_->num_features());
  int8_t* rows = cuda_best_split_finder_->LevelNodeMaskRows(static_cast<int>(bynode_level_leaves_.size()));
  bynode_row_feature_begin_.assign(1, 0);
  bynode_row_features_.clear();
  for (size_t i = 0; i < bynode_level_leaves_.size(); ++i) {
    DrawKeyedNodeMask(bynode_level_leaves_[i], rows + i * num_features, &bynode_one_row_features_);
    bynode_row_features_.insert(bynode_row_features_.end(), bynode_one_row_features_.begin(),
                                bynode_one_row_features_.end());
    bynode_row_feature_begin_.push_back(static_cast<int>(bynode_row_features_.size()));
  }
  cuda_best_split_finder_->CommitLevelNodeMasks(bynode_level_leaves_, bynode_row_feature_begin_, bynode_row_features_,
                                                cuda_histogram_constructor_->hist_stream());
}

void CUDASingleGPUTreeLearner::DebugPrintBynodeSplit(const int leaf, const int inner_feature) const {
  if (!FalcataDebug().bynode || !select_features_by_node_ || leaf >= static_cast<int>(debug_bynode_masks_.size())) {
    return;
  }
  const std::vector<int8_t>& mask = debug_bynode_masks_[leaf];
  std::string bits(mask.size(), '0');
  for (size_t f = 0; f < mask.size(); ++f) {
    bits[f] = mask[f] ? '1' : '0';
  }
  fprintf(stderr, "BYNODE leaf=%d feat=%d mask=%s\n", leaf, inner_feature, bits.c_str());
}

void CUDASingleGPUTreeLearner::SelectFeatureByNode(const Tree* tree, const int smaller_leaf_index,
                                                   const int larger_leaf_index) {
  if (select_features_by_node_ && bynode_keyed_) {
    // cuda_plan key level_bynode_masks: the same keyed masks the level flows draw
    const size_t num_features = static_cast<size_t>(train_data_->num_features());
    bynode_pair_masks_[0].resize(num_features);
    DrawKeyedNodeMask(smaller_leaf_index, bynode_pair_masks_[0].data());
    bynode_pair_masks_[1].clear();
    if (larger_leaf_index >= 0) {
      bynode_pair_masks_[1].resize(num_features);
      DrawKeyedNodeMask(larger_leaf_index, bynode_pair_masks_[1].data());
    }
    cuda_best_split_finder_->SetUsedFeatureByNode(bynode_pair_masks_[0], bynode_pair_masks_[1]);
  } else if (select_features_by_node_) {
    // use feature interaction constraint or sample features by node
    const std::vector<int8_t>& is_feature_used_by_smaller_node = col_sampler_.GetByNode(tree, smaller_leaf_index);
    std::vector<int8_t> is_feature_used_by_larger_node;
    if (larger_leaf_index >= 0) {
      is_feature_used_by_larger_node = col_sampler_.GetByNode(tree, larger_leaf_index);
    }
    cuda_best_split_finder_->SetUsedFeatureByNode(is_feature_used_by_smaller_node, is_feature_used_by_larger_node);
    if (FalcataDebug().bynode) {
      if (static_cast<int>(debug_bynode_masks_.size()) < config_->num_leaves) {
        debug_bynode_masks_.resize(config_->num_leaves);
      }
      debug_bynode_masks_[smaller_leaf_index] = is_feature_used_by_smaller_node;
      if (larger_leaf_index >= 0) {
        debug_bynode_masks_[larger_leaf_index] = is_feature_used_by_larger_node;
      }
    }
  }
}

#ifdef DEBUG
void CUDASingleGPUTreeLearner::CheckSplitValid(
  const int left_leaf,
  const int right_leaf) {
  CHECK(!whole_host_leaf_cache_stale_);
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
