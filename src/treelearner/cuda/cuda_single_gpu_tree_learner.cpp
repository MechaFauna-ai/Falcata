/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_single_gpu_tree_learner.hpp"

#include <LightGBM/cuda/cuda_tree.hpp>
#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/feature_group.h>
#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../linear_leaf_solver.h"

namespace LightGBM {

CUDASingleGPUTreeLearner::CUDASingleGPUTreeLearner(const Config* config, const bool boosting_on_cuda): SerialTreeLearner(config), boosting_on_cuda_(boosting_on_cuda) {}

CUDASingleGPUTreeLearner::~CUDASingleGPUTreeLearner() {
  if (nccl_communicator_ != nullptr) {
    CUDAStreamDestroy(nccl_stream_);
  }
}

void CUDASingleGPUTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  num_threads_ = OMP_NUM_THREADS();
  // use the first gpu by default
  if (nccl_communicator_ == nullptr) {
    gpu_device_id_ = config_->gpu_device_id >= 0 ? config_->gpu_device_id : 0;
    SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
  }
  cuda_smaller_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  cuda_smaller_leaf_splits_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, global_num_data_);
  cuda_smaller_leaf_splits_->Init(config_->use_quantized_grad);
  cuda_larger_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  cuda_larger_leaf_splits_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, global_num_data_);
  cuda_larger_leaf_splits_->Init(config_->use_quantized_grad);

  cuda_histogram_constructor_.reset(new CUDAHistogramConstructor(train_data_, config_->num_leaves, num_threads_,
    share_state_->feature_hist_offsets(),
    config_->min_data_in_leaf, config_->min_sum_hessian_in_leaf, gpu_device_id_, config_->gpu_use_dp,
    config_->use_quantized_grad, config_->num_grad_quant_bins));
  cuda_histogram_constructor_->Init(train_data_, share_state_.get());

  const auto& feature_hist_offsets = share_state_->feature_hist_offsets();
  num_total_bin_ = feature_hist_offsets.empty() ? 0 : static_cast<int>(feature_hist_offsets.back());
  cuda_data_partition_.reset(new CUDADataPartition(
    train_data_, num_total_bin_, config_->num_leaves, num_threads_, config_->use_quantized_grad,
    cuda_histogram_constructor_->cuda_hist_pointer()));
  cuda_data_partition_->SetNCCLInfo(nccl_communicator_, nccl_gpu_rank_, local_gpu_rank_, gpu_device_id_, global_num_data_);
  cuda_data_partition_->Init();

  select_features_by_node_ = !config_->interaction_constraints_vector.empty() || config_->feature_fraction_bynode < 1.0;
  // hybrid level-batched growth is on by default; EXABOOST_HYBRID_GROWTH=0 disables
  // it (falls back to the classic one-split-at-a-time leaf-wise loop everywhere)
  const char* hybrid_env = std::getenv("EXABOOST_HYBRID_GROWTH");
  use_hybrid_growth_ = (hybrid_env == nullptr || std::string(hybrid_env) != std::string("0"));
  cuda_best_split_finder_.reset(new CUDABestSplitFinder(cuda_histogram_constructor_->cuda_hist(),
    train_data_, this->share_state_->feature_hist_offsets(), select_features_by_node_, config_));
  cuda_best_split_finder_->Init();
  // Wire the histogram constructor's completion events into the best split finder so
  // per-leaf FindBestSplits kernels are ordered after histogram construction/subtraction
  // via cudaStreamWaitEvent instead of per-split device syncs.
  cuda_best_split_finder_->SetHistogramEvents(
    cuda_histogram_constructor_->construct_done_event(),
    cuda_histogram_constructor_->subtract_done_event());

  leaf_best_split_feature_.resize(config_->num_leaves, -1);
  leaf_best_split_threshold_.resize(config_->num_leaves, 0);
  leaf_best_split_default_left_.resize(config_->num_leaves, 0);
  leaf_num_data_.resize(config_->num_leaves, 0);
  leaf_data_start_.resize(config_->num_leaves, 0);
  leaf_sum_gradients_.resize(config_->num_leaves, 0.0f);
  leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);

  if (!boosting_on_cuda_) {
    cuda_gradients_.Resize(static_cast<size_t>(num_data_));
    cuda_hessians_.Resize(static_cast<size_t>(num_data_));
  }
  AllocateBitset();

  leaf_stat_buffer_size_ = 0;
  num_cat_threshold_ = 0;

  if (config_->use_quantized_grad) {
    cuda_leaf_gradient_stat_buffer_.Resize(config_->num_leaves);
    cuda_leaf_hessian_stat_buffer_.Resize(config_->num_leaves);
    // one random-offset slot per TREE: multiclass trains num_class trees per
    // iteration, and the discretizer's iter_ counter advances once per tree
    cuda_gradient_discretizer_.reset(new CUDAGradientDiscretizer(
      config_->num_grad_quant_bins, config_->num_iterations * std::max(config_->num_class, 1), config_->seed, is_constant_hessian, config_->stochastic_rounding));
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

void CUDASingleGPUTreeLearner::BeforeTrain() {
  const data_size_t root_num_data = cuda_data_partition_->root_num_data();
  if (!boosting_on_cuda_) {
    CopyFromHostToCUDADevice<score_t>(cuda_gradients_.RawData(), gradients_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<score_t>(cuda_hessians_.RawData(), hessians_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
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
    cuda_gradient_discretizer_->DiscretizeGradients(num_data_, gradients_, hessians_);
    cuda_histogram_constructor_->BeforeTrain(
      reinterpret_cast<const score_t*>(cuda_gradient_discretizer_->discretized_gradients_and_hessians()), nullptr);
    cuda_smaller_leaf_splits_->InitValues(
      config_->lambda_l1,
      config_->lambda_l2,
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
    cuda_smaller_leaf_splits_->InitValues(
      config_->lambda_l1,
      config_->lambda_l2,
      gradients_,
      hessians_,
      leaf_splits_init_indices,
      cuda_data_partition_->cuda_data_indices(),
      root_num_data,
      cuda_histogram_constructor_->cuda_hist_pointer(),
      &leaf_sum_gradients_[0],
      &leaf_sum_hessians_[0]);
  }
  leaf_num_data_[0] = root_num_data;
  cuda_larger_leaf_splits_->InitValues();
  col_sampler_.ResetByTree();
  cuda_best_split_finder_->BeforeTrain(col_sampler_.is_feature_used_bytree());
  cuda_histogram_constructor_->SetFeatureUsedBytree(col_sampler_.is_feature_used_bytree());
  cuda_histogram_constructor_->BuildCompactView(col_sampler_.is_feature_used_bytree());
  // Build per-tree compact column data for the partition kernels.
  BuildCompactColumnView();
  leaf_data_start_[0] = 0;
  smaller_leaf_index_ = 0;
  larger_leaf_index_ = -1;

  if (nccl_communicator_ != nullptr) {
    leaf_to_hist_index_map_.resize(config_->num_leaves, -1);
    leaf_to_hist_index_map_[0] = 0;
    global_num_data_in_leaf_[0] = global_num_data_;
  }
}

// Forward decl: defined in cuda_histogram_constructor.cu.
extern void LaunchRowToColCompactKernel(
    cudaStream_t stream,
    const uint8_t* src_data,
    uint8_t* compact_col_buf,
    const size_t* slot_p_byte,
    const int* slot_p_stride,
    const int* slot_col_in_p,
    int num_compact_cols,
    data_size_t num_data);

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
  if (sig == compact_col_signature_ && !compact_column_to_orig_.empty()) {
    // Cache hit: layout unchanged → pointers still valid.
    return;
  }

  // Build column->slot mapping. CUDAColumnData uses column index, not feature index.
  // For dense single-feature groups (Numerai), feature_to_column[f] returns the column index.
  CUDAColumnData* col_data = const_cast<CUDAColumnData*>(train_data_->cuda_column_data());
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
  if (compact_column_buffer_.Size() < needed_bytes) {
    compact_column_buffer_.Resize(needed_bytes);
  }

  // Build per-slot source-frame metadata host-side (one entry per compact slot).
  const auto& src_part_col_offsets_h = row_data->host_feature_partition_column_index_offsets();
  std::vector<size_t> slot_p_byte_h(num_compact_cols);
  std::vector<int> slot_p_stride_h(num_compact_cols);
  std::vector<int> slot_col_in_p_h(num_compact_cols);
  for (int s = 0; s < num_compact_cols; ++s) {
    const int orig_col = compact_column_to_orig_[s];
    int p = 0;
    while (p + 1 < static_cast<int>(src_part_col_offsets_h.size()) &&
           orig_col >= src_part_col_offsets_h[p + 1]) ++p;
    const int p_start = src_part_col_offsets_h[p];
    const int p_end = src_part_col_offsets_h[p + 1];
    slot_p_byte_h[s] = static_cast<size_t>(p_start) * static_cast<size_t>(num_data);
    slot_p_stride_h[s] = p_end - p_start;
    slot_col_in_p_h[s] = orig_col - p_start;
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

  LaunchRowToColCompactKernel(
      0,
      src,
      compact_column_buffer_.RawData(),
      cuda_slot_p_byte_.RawData(),
      cuda_slot_p_stride_.RawData(),
      cuda_slot_col_in_p_.RawData(),
      num_compact_cols,
      num_data);
  CUDASUCCESS_OR_FATAL(cudaDeviceSynchronize());

  // Update CUDAColumnData's data_by_column_ pointers to slots in compact buffer.
  col_data->SetCompactColumnView(orig_column_to_compact_slot_,
                                 compact_column_buffer_.RawData(),
                                 static_cast<size_t>(num_data));
  compact_col_signature_ = sig;
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

bool CUDASingleGPUTreeLearner::HybridGrowthUsable() const {
  return use_hybrid_growth_ &&
         nccl_communicator_ == nullptr &&
         !has_categorical_feature_ &&
         !select_features_by_node_ &&
         (forced_split_json_ == nullptr || forced_split_json_->is_null()) &&
         config_->num_leaves > 2;
}

void CUDASingleGPUTreeLearner::EnqueuePairBestSplitSearch(const CUDATree* tree,
    const CUDALeafSplitsStruct* smaller_struct, const CUDALeafSplitsStruct* larger_struct,
    const int smaller_leaf_index, const int larger_leaf_index, const bool synchronize) {
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
    NCCLReduceHistogram();
  }
  global_timer.Stop("CUDASingleGPUTreeLearner::NCCLReduceHistogram");

  global_timer.Start("CUDASingleGPUTreeLearner::FindBestSplitsForLeaf");
  uint8_t parent_num_bits_bin = 0;
  uint8_t smaller_num_bits_bin = 0;
  uint8_t larger_num_bits_bin = 0;
  if (config_->use_quantized_grad) {
    if (larger_leaf_index != -1) {
      const int parent_leaf_index = std::min(smaller_leaf_index, larger_leaf_index);
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
      config_->max_depth <= 0 || tree->leaf_depth(smaller_leaf_index) < config_->max_depth,
      larger_leaf_index < 0 || config_->max_depth <= 0 ||
        tree->leaf_depth(larger_leaf_index) < config_->max_depth,
      synchronize);
  } else {
    cuda_best_split_finder_->FindBestSplitsForLeaf(
      smaller_struct,
      larger_struct,
      smaller_leaf_index, larger_leaf_index,
      global_num_data_in_smaller_leaf, global_num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf,
      nullptr, nullptr, 0, 0,
      config_->max_depth <= 0 || tree->leaf_depth(smaller_leaf_index) < config_->max_depth,
      larger_leaf_index < 0 || config_->max_depth <= 0 ||
        tree->leaf_depth(larger_leaf_index) < config_->max_depth,
      synchronize);
  }
  global_timer.Stop("CUDASingleGPUTreeLearner::FindBestSplitsForLeaf");
}

int CUDASingleGPUTreeLearner::TrainLevelWisePrefix(CUDATree* tree) {
  // two struct slots per split of a level, reused across levels
  const size_t max_slots = static_cast<size_t>(config_->num_leaves) + 2;
  if (hybrid_pair_slots_.size() < max_slots) {
    const size_t old_size = hybrid_pair_slots_.size();
    hybrid_pair_slots_.resize(max_slots);
    for (size_t s = old_size; s < max_slots; ++s) {
      hybrid_pair_slots_[s].reset(new CUDALeafSplits(1));
      hybrid_pair_slots_[s]->Init(config_->use_quantized_grad);
    }
  }
  struct PendingPair {
    int smaller;
    int larger;
    const CUDALeafSplitsStruct* smaller_struct;
    const CUDALeafSplitsStruct* larger_struct;
  };
  // the root "pair" lives in the fixed pair structs, initialized by BeforeTrain
  std::vector<PendingPair> pairs;
  pairs.push_back({smaller_leaf_index_, larger_leaf_index_,
                   cuda_smaller_leaf_splits_->GetCUDAStruct(),
                   cuda_larger_leaf_splits_->GetCUDAStruct()});
  int num_splits = 0;
  while (true) {
    // enqueue histogram + best-split search for every pair of this level; device
    // work only, ordered per pair by the histogram-completion events
    for (const PendingPair& pair : pairs) {
      static const bool sync_pairs = std::getenv("EXABOOST_HYBRID_SYNCPAIRS") != nullptr;
      EnqueuePairBestSplitSearch(tree, pair.smaller_struct, pair.larger_struct,
                                 pair.smaller, pair.larger, /*synchronize=*/sync_pairs);
    }
    // one device synchronization + one transfer for the whole level
    cuda_best_split_finder_->SyncAllLeafBestSplitsToHost(tree->num_leaves(), &host_leaf_best_splits_);
    std::vector<int> splittable;
    for (int leaf = 0; leaf < tree->num_leaves(); ++leaf) {
      const CUDASplitInfo& info = host_leaf_best_splits_[leaf];
      if (info.is_valid) {
        // keep the host-side split arrays consistent with the device cache for
        // every candidate leaf: the leaf-wise tail may apply any of them later
        leaf_best_split_feature_[leaf] = info.inner_feature_index;
        leaf_best_split_threshold_[leaf] = info.threshold;
        leaf_best_split_default_left_[leaf] = static_cast<uint8_t>(info.default_left);
        splittable.push_back(leaf);
      }
    }
    static const bool hybrid_debug = std::getenv("EXABOOST_HYBRID_DEBUG") != nullptr;
    if (hybrid_debug) {
      // (heavy per-pair diagnostics removed; see git history of the branch)
  }
    if (hybrid_debug) {
      double max_gain = -1e300, min_gain = 1e300;
      for (const int leaf : splittable) {
        max_gain = std::max(max_gain, host_leaf_best_splits_[leaf].gain);
        min_gain = std::min(min_gain, host_leaf_best_splits_[leaf].gain);
      }
      Log::Warning("[hybrid] leaves=%d splittable=%d gain=[%g, %g]",
                   tree->num_leaves(), static_cast<int>(splittable.size()), min_gain, max_gain);
    }
    if (splittable.empty()) {
      break;
    }
    // stop level batching as soon as the leaf budget could bind or applying the level
    // would leave no headroom for ranking its children; the leaf-wise tail then
    // selects among the cached candidates in exact best-gain order
    if (tree->num_leaves() + 2 * static_cast<int>(splittable.size()) > config_->num_leaves) {
      break;
    }
    // debug: cap splits per level to isolate multi-pair interactions
    static const char* max_splits_env = std::getenv("EXABOOST_HYBRID_MAXSPLITS");
    if (max_splits_env != nullptr) {
      const size_t cap = static_cast<size_t>(std::atoi(max_splits_env));
      if (splittable.size() > cap) splittable.resize(cap);
    }
    pairs.clear();
    // apply the whole level without any device synchronization: every launch
    // parameter is host-known from the previous level, and split info is read
    // back once for the whole level below (FinishSplitBatch synchronizes)
    struct AppliedSplit {
      int left;
      int right;
      CUDALeafSplitsStruct* smaller_slot;
      CUDALeafSplitsStruct* larger_slot;
    };
    std::vector<AppliedSplit> applied;
    applied.reserve(splittable.size());
    size_t slot_id = 0;
    for (const int leaf : splittable) {
      CUDALeafSplitsStruct* smaller_slot = hybrid_pair_slots_[slot_id]->GetCUDAStructRef();
      CUDALeafSplitsStruct* larger_slot = hybrid_pair_slots_[slot_id + 1]->GetCUDAStructRef();
      const int right_leaf_index = ApplySplit(
        tree, cuda_best_split_finder_->leaf_best_split_info_ptr(leaf), leaf,
        smaller_slot, larger_slot, static_cast<int>(applied.size()));
      applied.push_back({leaf, right_leaf_index, smaller_slot, larger_slot});
      slot_id += 2;
    }
    std::vector<int> batch_info;
    cuda_data_partition_->FinishSplitBatch(static_cast<int>(applied.size()), &batch_info);
    for (size_t k = 0; k < applied.size(); ++k) {
      const AppliedSplit& a = applied[k];
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
      const int smaller = leaf_num_data_[a.left] < leaf_num_data_[a.right] ? a.left : a.right;
      const int larger = smaller == a.left ? a.right : a.left;
      if (config_->use_quantized_grad) {
        cuda_gradient_discretizer_->SetNumBitsInHistogramBin<false>(
          a.left, a.right, leaf_num_data_[a.left], leaf_num_data_[a.right]);
      }
      pairs.push_back({smaller, larger, a.smaller_slot, a.larger_slot});
      smaller_leaf_index_ = smaller;
      larger_leaf_index_ = larger;
      ++num_splits;
    }
  }
  return num_splits;
}

Tree* CUDASingleGPUTreeLearner::Train(const score_t* gradients,
  const score_t* hessians, bool is_first_tree) {
  gradients_ = gradients;
  hessians_ = hessians;
  global_timer.Start("CUDASingleGPUTreeLearner::BeforeTrain");
  BeforeTrain();
  global_timer.Stop("CUDASingleGPUTreeLearner::BeforeTrain");
  // linear trees need per-leaf branch features to know which features each leaf's
  // linear model uses (mirrors the CPU LinearTreeLearner using new Tree(.., true, true)).
  const bool track_branch_features = !(config_->interaction_constraints_vector.empty()) || config_->linear_tree;
  std::unique_ptr<CUDATree> tree(new CUDATree(config_->num_leaves, track_branch_features,
    config_->linear_tree, gpu_device_id_, has_categorical_feature_));
  // set the root value by hand, as it is not handled by splits
  tree->SetLeafOutput(0, CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(
    leaf_sum_gradients_[smaller_leaf_index_], leaf_sum_hessians_[smaller_leaf_index_],
    config_->lambda_l1, config_->lambda_l2,  config_->path_smooth,
    static_cast<data_size_t>(num_data_), 0));
  tree->SyncLeafOutputFromHostToCUDA();
  int num_splits_done = 0;
  ForceSplitsCUDA(tree.get(), &num_splits_done);
  bool pair_search_cached = false;
  if (num_splits_done == 0 && HybridGrowthUsable()) {
    global_timer.Start("CUDASingleGPUTreeLearner::TrainLevelWisePrefix");
    num_splits_done = TrainLevelWisePrefix(tree.get());
    global_timer.Stop("CUDASingleGPUTreeLearner::TrainLevelWisePrefix");
    // every current leaf already has a cached best-split candidate; the first
    // tail iteration must not redo the pair search
    pair_search_cached = num_splits_done > 0;
  }
  for (int i = num_splits_done; i < config_->num_leaves - 1; ++i) {
    if (!pair_search_cached) {
      EnqueuePairBestSplitSearch(tree.get(),
        cuda_smaller_leaf_splits_->GetCUDAStruct(),
        cuda_larger_leaf_splits_->GetCUDAStruct(),
        smaller_leaf_index_, larger_leaf_index_, /*synchronize=*/true);
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
      Log::Warning("No further splits with positive gain, training stopped with %d leaves.", (i + 1));
      break;
    }

    global_timer.Start("CUDASingleGPUTreeLearner::Split");
    if (num_cat_threshold_ > 0) {
      ConstructBitsetForCategoricalSplit(best_split_info);
    }

    const int right_leaf_index = ApplySplit(tree.get(), best_split_info, best_leaf_index_);

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
  if (config_->use_quantized_grad && config_->quant_train_renew_leaf) {
    global_timer.Start("CUDASingleGPUTreeLearner::RenewDiscretizedTreeLeaves");
    RenewDiscretizedTreeLeaves(tree.get());
    global_timer.Stop("CUDASingleGPUTreeLearner::RenewDiscretizedTreeLeaves");
  }
  tree->ToHost();
  last_tree_is_linear_ = false;
  if (config_->linear_tree) {
    // gradients_/hessians_ are device pointers after BeforeTrain (the Train args stay on host)
    CalculateLinearCUDA(tree.get(), gradients_, hessians_, is_first_tree);
    last_tree_is_linear_ = true;
  }
  return tree.release();
}

int CUDASingleGPUTreeLearner::ApplySplit(CUDATree* tree, const CUDASplitInfo* best_split_info, const int leaf_index,
                                         CUDALeafSplitsStruct* smaller_slot, CUDALeafSplitsStruct* larger_slot,
                                         int deferred_slot) {
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
  } else {
    right_leaf_index = tree->Split(leaf_index,
                                     train_data_->RealFeatureIndex(leaf_best_split_feature_[leaf_index]),
                                     train_data_->RealThreshold(leaf_best_split_feature_[leaf_index],
                                      leaf_best_split_threshold_[leaf_index]),
                                     train_data_->FeatureBinMapper(leaf_best_split_feature_[leaf_index])->missing_type(),
                                     best_split_info);
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
    cuda_best_split_finder_->FindBestSplitsForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      smaller_leaf_index_, larger_leaf_index_,
      num_data_in_smaller_leaf, num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf,
      nullptr, nullptr, 0, 0,
      config_->max_depth <= 0 || tree->leaf_depth(smaller_leaf_index_) < config_->max_depth,
      larger_leaf_index_ < 0 || config_->max_depth <= 0 ||
        tree->leaf_depth(larger_leaf_index_) < config_->max_depth);
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

void CUDASingleGPUTreeLearner::ResetTrainingData(
  const Dataset* train_data,
  bool is_constant_hessian) {
  SerialTreeLearner::ResetTrainingData(train_data, is_constant_hessian);
  CHECK_EQ(num_features_, train_data_->num_features());
  cuda_histogram_constructor_->ResetTrainingData(train_data, share_state_.get());
  cuda_data_partition_->ResetTrainingData(train_data,
    static_cast<int>(share_state_->feature_hist_offsets().back()),
    cuda_histogram_constructor_->cuda_hist_pointer());
  cuda_best_split_finder_->ResetTrainingData(
    cuda_histogram_constructor_->cuda_hist(),
    train_data,
    share_state_->feature_hist_offsets());
  cuda_smaller_leaf_splits_->Resize(num_data_);
  cuda_larger_leaf_splits_->Resize(num_data_);
  CHECK_EQ(is_constant_hessian, share_state_->is_constant_hessian);
  if (!boosting_on_cuda_) {
    cuda_gradients_.Resize(static_cast<size_t>(num_data_));
    cuda_hessians_.Resize(static_cast<size_t>(num_data_));
  }
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
      }
    }
    // std::max(..., 1UL) to avoid error in the case when there are NaN's in the categorical values
    const size_t cuda_bitset_max_size = std::max(static_cast<size_t>((max_cat_value + 31) / 32), 1UL);
    const size_t cuda_bitset_inner_max_size = std::max(static_cast<size_t>((max_cat_num_bin + 31) / 32), 1UL);
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
}

void CUDASingleGPUTreeLearner::NCCLReduceHistogram() {
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
    NCCLAllReduce<hist_t>(
      smaller_leaf_hist_pointer,
      smaller_leaf_hist_pointer,
      static_cast<size_t>(num_total_bin_) * 2,
      ncclFloat64,
      ncclSum,
      nccl_communicator_,
      nccl_stream_);
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

}  // namespace LightGBM

#endif  // USE_CUDA
