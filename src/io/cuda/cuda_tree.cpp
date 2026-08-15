/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_tree.hpp>

#include <algorithm>
#include <cstring>
#include <vector>

namespace Falcata {

CUDATree::CUDATree(int max_leaves, bool track_branch_features, bool is_linear,
  const int gpu_device_id, const bool has_categorical_feature,
  uint8_t* pooled_device_buffer):
Tree(max_leaves, track_branch_features, is_linear),
num_threads_per_block_add_prediction_to_score_(1024) {
  is_cuda_tree_ = true;
  if (gpu_device_id >= 0) {
    SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
  } else {
    SetCUDADevice(0, __FILE__, __LINE__);
  }
  if (has_categorical_feature) {
    cuda_cat_boundaries_.Resize(max_leaves);
    cuda_cat_boundaries_inner_.Resize(max_leaves);
  }
  InitCUDAMemory(pooled_device_buffer);
}

CUDATree::CUDATree(const Tree* host_tree):
  Tree(*host_tree),
  num_threads_per_block_add_prediction_to_score_(1024) {
  is_cuda_tree_ = true;
  InitCUDA();
}

CUDATree::~CUDATree() {
  // CUDAVector members handle their own deallocation. ToHost() may have
  // already destroyed the stream — guard against double-destroy.
  if (cuda_stream_ != nullptr) {
    gpuAssert(cudaStreamDestroy(cuda_stream_), __FILE__, __LINE__);
  }
}

void CUDATree::InitCUDAMemory(uint8_t* pooled_device_buffer) {
  if (pooled_device_buffer != nullptr) {
    // carve every per-tree array from the caller-owned slab as non-owning views:
    // no per-tree cudaMalloc, and the whole set reads back with one D2H copy in
    // ToHost(). Layout must match PooledDeviceBufferSize() and the fused ToHost
    // readback: 8-byte types first, then 4-byte, then decision_type (int8).
    // Fresh cudaMalloc'd memory was never zeroed either, so reusing a slab with
    // stale content across trees is equivalent (every read is write-first,
    // except the three first-element memsets below).
    pooled_ = true;
    const size_t L = static_cast<size_t>(max_leaves_);
    uint8_t* ptr = pooled_device_buffer;
    const auto take = [&ptr](size_t bytes) { uint8_t* p = ptr; ptr += bytes; return p; };
    cuda_threshold_.SetView(reinterpret_cast<double*>(take(sizeof(double) * L)), L);
    cuda_internal_weight_.SetView(reinterpret_cast<double*>(take(sizeof(double) * L)), L);
    cuda_internal_value_.SetView(reinterpret_cast<double*>(take(sizeof(double) * L)), L);
    cuda_leaf_value_.SetView(reinterpret_cast<double*>(take(sizeof(double) * L)), L);
    cuda_leaf_weight_.SetView(reinterpret_cast<double*>(take(sizeof(double) * L)), L);
    const size_t num_batch_splits = L / 2 + 2;
    cuda_batch_splits_.SetView(reinterpret_cast<CUDATreeBatchSplit*>(
      take(sizeof(CUDATreeBatchSplit) * num_batch_splits)), num_batch_splits);
    cuda_left_child_.SetView(reinterpret_cast<int*>(take(sizeof(int) * L)), L);
    cuda_right_child_.SetView(reinterpret_cast<int*>(take(sizeof(int) * L)), L);
    cuda_split_feature_inner_.SetView(reinterpret_cast<int*>(take(sizeof(int) * L)), L);
    cuda_split_feature_.SetView(reinterpret_cast<int*>(take(sizeof(int) * L)), L);
    cuda_leaf_depth_.SetView(reinterpret_cast<int*>(take(sizeof(int) * L)), L);
    cuda_leaf_parent_.SetView(reinterpret_cast<int*>(take(sizeof(int) * L)), L);
    cuda_threshold_in_bin_.SetView(reinterpret_cast<uint32_t*>(take(sizeof(uint32_t) * L)), L);
    cuda_leaf_count_.SetView(reinterpret_cast<data_size_t*>(take(sizeof(data_size_t) * L)), L);
    cuda_internal_count_.SetView(reinterpret_cast<data_size_t*>(take(sizeof(data_size_t) * L)), L);
    cuda_split_gain_.SetView(reinterpret_cast<float*>(take(sizeof(float) * L)), L);
    cuda_decision_type_.SetView(reinterpret_cast<int8_t*>(take(sizeof(int8_t) * L)), L);
  } else {
    cuda_left_child_.Resize(static_cast<size_t>(max_leaves_));
    cuda_right_child_.Resize(static_cast<size_t>(max_leaves_));
    cuda_split_feature_inner_.Resize(static_cast<size_t>(max_leaves_));
    cuda_split_feature_.Resize(static_cast<size_t>(max_leaves_));
    cuda_leaf_depth_.Resize(static_cast<size_t>(max_leaves_));
    cuda_leaf_parent_.Resize(static_cast<size_t>(max_leaves_));
    cuda_threshold_in_bin_.Resize(static_cast<size_t>(max_leaves_));
    cuda_threshold_.Resize(static_cast<size_t>(max_leaves_));
    cuda_decision_type_.Resize(static_cast<size_t>(max_leaves_));
    cuda_leaf_value_.Resize(static_cast<size_t>(max_leaves_));
    cuda_internal_weight_.Resize(static_cast<size_t>(max_leaves_));
    cuda_internal_value_.Resize(static_cast<size_t>(max_leaves_));
    cuda_leaf_weight_.Resize(static_cast<size_t>(max_leaves_));
    cuda_leaf_count_.Resize(static_cast<size_t>(max_leaves_));
    cuda_internal_count_.Resize(static_cast<size_t>(max_leaves_));
    cuda_split_gain_.Resize(static_cast<size_t>(max_leaves_));
  }
  // async memsets on the default stream, stream-ordered before every consumer
  // (SetCUDAMemory would pay one full device sync each)
  CUDASUCCESS_OR_FATAL(cudaMemset(reinterpret_cast<void*>(cuda_leaf_value_.RawData()), 0, sizeof(double)));
  CUDASUCCESS_OR_FATAL(cudaMemset(reinterpret_cast<void*>(cuda_leaf_weight_.RawData()), 0, sizeof(double)));
  CUDASUCCESS_OR_FATAL(cudaMemset(reinterpret_cast<void*>(cuda_leaf_parent_.RawData()), -1, sizeof(int)));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));
  if (!pooled_) {
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }
}

void CUDATree::InitCUDA() {
  cuda_left_child_.InitFromHostVector(left_child_);
  cuda_right_child_.InitFromHostVector(right_child_);
  cuda_split_feature_inner_.InitFromHostVector(split_feature_inner_);
  cuda_split_feature_.InitFromHostVector(split_feature_);
  cuda_threshold_in_bin_.InitFromHostVector(threshold_in_bin_);
  cuda_threshold_.InitFromHostVector(threshold_);
  cuda_leaf_depth_.InitFromHostVector(leaf_depth_);
  cuda_decision_type_.InitFromHostVector(decision_type_);
  cuda_internal_weight_.InitFromHostVector(internal_weight_);
  cuda_internal_value_.InitFromHostVector(internal_value_);
  cuda_internal_count_.InitFromHostVector(internal_count_);
  cuda_leaf_count_.InitFromHostVector(leaf_count_);
  cuda_split_gain_.InitFromHostVector(split_gain_);
  cuda_leaf_value_.InitFromHostVector(leaf_value_);
  cuda_leaf_weight_.InitFromHostVector(leaf_weight_);
  cuda_leaf_parent_.InitFromHostVector(leaf_parent_);
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

int CUDATree::Split(const int leaf_index,
           const int real_feature_index,
           const double real_threshold,
           const MissingType missing_type,
           const CUDASplitInfo* cuda_split_info) {
  LaunchSplitKernel(leaf_index, real_feature_index, real_threshold, missing_type, cuda_split_info);
  RecordBranchFeatures(leaf_index, num_leaves_, real_feature_index);
  // mirror CPU Tree::Split: keep host-side leaf_depth_ in sync so leaf_depth(idx)
  // returns the correct value for max_depth and other host-side checks.
  leaf_depth_[num_leaves_] = leaf_depth_[leaf_index] + 1;
  leaf_depth_[leaf_index]++;
  ++num_leaves_;
  return num_leaves_ - 1;
}

void CUDATree::SplitBatch(const std::vector<CUDATreeBatchSplit>& splits) {
  if (splits.empty()) {
    return;
  }
  if (cuda_batch_splits_.Size() < splits.size()) {
    // preallocate for the deepest possible level so this resizes at most once
    cuda_batch_splits_.Resize(std::max(splits.size(),
      static_cast<size_t>(max_leaves_ / 2 + 2)));
  }
  // async H2D on cuda_stream_: ordered after any prior SplitBatchKernel still
  // reading the buffer and before the launch below (same stream). The host
  // vector is only rewritten after the caller's per-level full sync, and a
  // pageable async H2D returns only once the data is staged, so reuse is safe.
  CopyFromHostToCUDADeviceAsync<CUDATreeBatchSplit>(cuda_batch_splits_.RawData(),
    splits.data(), splits.size(), cuda_stream_, __FILE__, __LINE__);
  LaunchSplitBatchKernel(static_cast<int>(splits.size()));
  // host mirrors, in the exact order the per-split Split() loop would apply them
  for (const CUDATreeBatchSplit& split : splits) {
    CHECK_EQ(split.num_leaves_at_split, num_leaves_);
    RecordBranchFeatures(split.leaf_index, num_leaves_, split.real_feature_index);
    leaf_depth_[num_leaves_] = leaf_depth_[split.leaf_index] + 1;
    leaf_depth_[split.leaf_index]++;
    ++num_leaves_;
  }
}

int CUDATree::SplitCategorical(const int leaf_index,
           const int real_feature_index,
           const MissingType missing_type,
           const CUDASplitInfo* cuda_split_info,
           uint32_t* cuda_bitset,
           size_t cuda_bitset_len,
           uint32_t* cuda_bitset_inner,
           size_t cuda_bitset_inner_len) {
  LaunchSplitCategoricalKernel(leaf_index, real_feature_index,
    missing_type, cuda_split_info,
    cuda_bitset_len, cuda_bitset_inner_len);
  cuda_bitset_.PushBack(cuda_bitset, cuda_bitset_len);
  cuda_bitset_inner_.PushBack(cuda_bitset_inner, cuda_bitset_inner_len);
  // The new leaf is num_leaves_ BEFORE the increment, as in Split(). Recording
  // it afterwards addressed branch_features_ one past the new leaf: out of
  // bounds on the last split of a full tree (branch_features_ holds exactly
  // max_leaves_ entries), and one leaf's ancestors written over another's for
  // every earlier split. linear_tree turns this tracking on for every tree, so
  // a categorical split under linear_tree corrupted the heap.
  RecordBranchFeatures(leaf_index, num_leaves_, real_feature_index);
  // mirror CPU Tree::Split: keep host-side leaf_depth_ in sync.
  leaf_depth_[num_leaves_] = leaf_depth_[leaf_index] + 1;
  leaf_depth_[leaf_index]++;
  ++num_leaves_;
  ++num_cat_;
  return num_leaves_ - 1;
}

void CUDATree::RecordBranchFeatures(const int left_leaf_index,
                                    const int right_leaf_index,
                                    const int real_feature_index) {
  if (track_branch_features_) {
    branch_features_[right_leaf_index] = branch_features_[left_leaf_index];
    branch_features_[right_leaf_index].push_back(real_feature_index);
    branch_features_[left_leaf_index].push_back(real_feature_index);
  }
}

void CUDATree::AddPredictionToScore(const Dataset* data,
                                    data_size_t num_data,
                                    double* score) const {
  LaunchAddPredictionToScoreKernel(data, nullptr, num_data, score);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDATree::AddPredictionToScore(const Dataset* data,
                                    const data_size_t* used_data_indices,
                                    data_size_t num_data, double* score) const {
  LaunchAddPredictionToScoreKernel(data, used_data_indices, num_data, score);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

inline void CUDATree::Shrinkage(double rate) {
  Tree::Shrinkage(rate);
  LaunchShrinkageKernel(rate);
}

inline void CUDATree::AddBias(double val) {
  Tree::AddBias(val);
  LaunchAddBiasKernel(val);
}

void CUDATree::ToHost() {
  left_child_.resize(max_leaves_ - 1);
  right_child_.resize(max_leaves_ - 1);
  split_feature_inner_.resize(max_leaves_ - 1);
  split_feature_.resize(max_leaves_ - 1);
  threshold_in_bin_.resize(max_leaves_ - 1);
  threshold_.resize(max_leaves_ - 1);
  decision_type_.resize(max_leaves_ - 1, 0);
  split_gain_.resize(max_leaves_ - 1);
  leaf_parent_.resize(max_leaves_);
  leaf_value_.resize(max_leaves_);
  leaf_weight_.resize(max_leaves_);
  leaf_count_.resize(max_leaves_);
  internal_value_.resize(max_leaves_ - 1);
  internal_weight_.resize(max_leaves_ - 1);
  internal_count_.resize(max_leaves_ - 1);
  leaf_depth_.resize(max_leaves_);

  const size_t num_leaves_size = static_cast<size_t>(num_leaves_);
  if (pooled_) {
    // all per-tree arrays live in one contiguous slab: ONE D2H copy replaces the
    // 16 individual synchronous copies below (each pays a full launch+sync
    // round trip). The scatter into the host vectors is plain host memcpy.
    const size_t L = static_cast<size_t>(max_leaves_);
    const size_t slab_bytes = PooledDeviceBufferSize(max_leaves_);
    std::vector<uint8_t> staging(slab_bytes);
    // cuda_threshold_ is the first view carved from the slab (see InitCUDAMemory)
    CopyFromCUDADeviceToHost<uint8_t>(staging.data(),
      reinterpret_cast<const uint8_t*>(cuda_threshold_.RawData()), slab_bytes, __FILE__, __LINE__);
    const uint8_t* p = staging.data();
    const auto scatter = [&p](void* dst, const size_t bytes, const size_t stride) {
      if (bytes > 0) std::memcpy(dst, p, bytes);
      p += stride;
    };
    scatter(threshold_.data(), (num_leaves_size - 1) * sizeof(double), L * sizeof(double));
    scatter(internal_weight_.data(), (num_leaves_size - 1) * sizeof(double), L * sizeof(double));
    scatter(internal_value_.data(), (num_leaves_size - 1) * sizeof(double), L * sizeof(double));
    scatter(leaf_value_.data(), num_leaves_size * sizeof(double), L * sizeof(double));
    scatter(leaf_weight_.data(), num_leaves_size * sizeof(double), L * sizeof(double));
    p += sizeof(CUDATreeBatchSplit) * (L / 2 + 2);  // skip the batched split descriptors
    scatter(left_child_.data(), (num_leaves_size - 1) * sizeof(int), L * sizeof(int));
    scatter(right_child_.data(), (num_leaves_size - 1) * sizeof(int), L * sizeof(int));
    scatter(split_feature_inner_.data(), (num_leaves_size - 1) * sizeof(int), L * sizeof(int));
    scatter(split_feature_.data(), (num_leaves_size - 1) * sizeof(int), L * sizeof(int));
    scatter(leaf_depth_.data(), num_leaves_size * sizeof(int), L * sizeof(int));
    scatter(leaf_parent_.data(), (num_leaves_size - 1) * sizeof(int), L * sizeof(int));
    scatter(threshold_in_bin_.data(), (num_leaves_size - 1) * sizeof(uint32_t), L * sizeof(uint32_t));
    scatter(leaf_count_.data(), num_leaves_size * sizeof(data_size_t), L * sizeof(data_size_t));
    scatter(internal_count_.data(), (num_leaves_size - 1) * sizeof(data_size_t), L * sizeof(data_size_t));
    scatter(split_gain_.data(), (num_leaves_size - 1) * sizeof(float), L * sizeof(float));
    scatter(decision_type_.data(), (num_leaves_size - 1) * sizeof(int8_t), L * sizeof(int8_t));
  } else {
  CopyFromCUDADeviceToHost<int>(left_child_.data(), cuda_left_child_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(right_child_.data(), cuda_right_child_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(split_feature_inner_.data(), cuda_split_feature_inner_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(split_feature_.data(), cuda_split_feature_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<uint32_t>(threshold_in_bin_.data(), cuda_threshold_in_bin_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(threshold_.data(), cuda_threshold_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int8_t>(decision_type_.data(), cuda_decision_type_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<float>(split_gain_.data(), cuda_split_gain_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(leaf_parent_.data(), cuda_leaf_parent_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(leaf_value_.data(), cuda_leaf_value_.RawData(), num_leaves_size, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(leaf_weight_.data(), cuda_leaf_weight_.RawData(), num_leaves_size, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<data_size_t>(leaf_count_.data(), cuda_leaf_count_.RawData(), num_leaves_size, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(internal_value_.data(), cuda_internal_value_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(internal_weight_.data(), cuda_internal_weight_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<data_size_t>(internal_count_.data(), cuda_internal_count_.RawData(), num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(leaf_depth_.data(), cuda_leaf_depth_.RawData(), num_leaves_size, __FILE__, __LINE__);
  }

  if (num_cat_ > 0) {
    cuda_cat_boundaries_inner_.Resize(num_cat_ + 1);
    cuda_cat_boundaries_.Resize(num_cat_ + 1);
    cat_boundaries_ = cuda_cat_boundaries_.ToHost();
    cat_boundaries_inner_ = cuda_cat_boundaries_inner_.ToHost();
    cat_threshold_ = cuda_bitset_.ToHost();
    cat_threshold_inner_ = cuda_bitset_inner_.ToHost();
  }

  ShrinkHostVectorsAndReleaseDevice();
}

void CUDATree::ShrinkHostVectorsAndReleaseDevice() {
  // Shrink host vectors to actual size before they're kept long-term in the
  // Booster's model list. With max_leaves_=8192 but actual num_leaves_~125 on
  // Numerai prod, this drops per-tree CPU memory from ~650 KB to ~10 KB.
  if (num_leaves_ > 0 && num_leaves_ < max_leaves_) {
    const size_t n_internal = static_cast<size_t>(num_leaves_) - 1;
    const size_t n_leaves = static_cast<size_t>(num_leaves_);
    left_child_.resize(n_internal); left_child_.shrink_to_fit();
    right_child_.resize(n_internal); right_child_.shrink_to_fit();
    split_feature_inner_.resize(n_internal); split_feature_inner_.shrink_to_fit();
    split_feature_.resize(n_internal); split_feature_.shrink_to_fit();
    threshold_in_bin_.resize(n_internal); threshold_in_bin_.shrink_to_fit();
    threshold_.resize(n_internal); threshold_.shrink_to_fit();
    decision_type_.resize(n_internal); decision_type_.shrink_to_fit();
    split_gain_.resize(n_internal); split_gain_.shrink_to_fit();
    leaf_parent_.resize(n_internal); leaf_parent_.shrink_to_fit();
    internal_value_.resize(n_internal); internal_value_.shrink_to_fit();
    internal_weight_.resize(n_internal); internal_weight_.shrink_to_fit();
    internal_count_.resize(n_internal); internal_count_.shrink_to_fit();
    leaf_value_.resize(n_leaves); leaf_value_.shrink_to_fit();
    leaf_weight_.resize(n_leaves); leaf_weight_.shrink_to_fit();
    leaf_count_.resize(n_leaves); leaf_count_.shrink_to_fit();
    leaf_depth_.resize(n_leaves); leaf_depth_.shrink_to_fit();
  }

  SynchronizeCUDADevice(__FILE__, __LINE__);

  // Free per-tree GPU buffers no longer needed after ToHost. Without this,
  // accumulated per-tree GPU memory OOMs a 32GB device after ~6k trees on
  // the Numerai prod config. cuda_leaf_value_ is the only field needed
  // post-train (AddPredictionToScore reads it); shrink to actual num_leaves_.
  // For pooled trees the slab is reused by the next tree, so the retained leaf
  // values must be materialized into owned memory in every case (Resize on a
  // view copies-then-owns; Materialize covers num_leaves_ == max_leaves_).
  if (num_leaves_ > 0 && num_leaves_ < max_leaves_ && cuda_leaf_value_.Size() > 0) {
    cuda_leaf_value_.Resize(static_cast<size_t>(num_leaves_));
  }
  cuda_leaf_value_.Materialize();
  cuda_left_child_.Clear();
  cuda_right_child_.Clear();
  cuda_split_feature_inner_.Clear();
  cuda_split_feature_.Clear();
  cuda_leaf_depth_.Clear();
  cuda_leaf_parent_.Clear();
  cuda_threshold_in_bin_.Clear();
  cuda_threshold_.Clear();
  cuda_internal_weight_.Clear();
  cuda_internal_value_.Clear();
  cuda_decision_type_.Clear();
  cuda_leaf_count_.Clear();
  cuda_leaf_weight_.Clear();
  cuda_internal_count_.Clear();
  cuda_split_gain_.Clear();
  cuda_batch_splits_.Clear();

  // Destroy the per-tree CUDA stream — only used during construction.
  if (cuda_stream_ != nullptr) {
    gpuAssert(cudaStreamDestroy(cuda_stream_), __FILE__, __LINE__);
    cuda_stream_ = nullptr;
  }
}

void CUDATree::RebuildFromHostSplits(const std::vector<CUDATreeHostSplit>& splits) {
  // Host replay of SplitKernel, one split at a time in classic (greedy) order.
  // The host mirrors are in their constructor state (the selective growth phase
  // never calls Split/SplitBatch), so leaf_parent_[0] == -1 and leaf_depth_[0]
  // == 0 hold; leaf_value_[0] holds the root output (set via SetLeafOutput).
  num_leaves_ = 1;
  for (const CUDATreeHostSplit& s : splits) {
    const int new_node_index = num_leaves_ - 1;
    const int leaf_index = s.leaf_index;
    CHECK_LT(leaf_index, num_leaves_);
    const int parent_index = leaf_parent_[leaf_index];
    if (parent_index >= 0) {
      // if cur node is left child
      if (left_child_[parent_index] == ~leaf_index) {
        left_child_[parent_index] = new_node_index;
      } else {
        right_child_[parent_index] = new_node_index;
      }
    }
    left_child_[new_node_index] = ~leaf_index;
    right_child_[new_node_index] = ~num_leaves_;
    leaf_parent_[leaf_index] = new_node_index;
    leaf_parent_[num_leaves_] = new_node_index;
    split_feature_inner_[new_node_index] = s.inner_feature_index;
    split_feature_[new_node_index] = s.real_feature_index;
    split_gain_[new_node_index] = static_cast<float>(s.gain);
    internal_weight_[new_node_index] = s.left_sum_hessians + s.right_sum_hessians;
    leaf_weight_[leaf_index] = s.left_sum_hessians;
    // save current leaf value to internal node before change (SplitKernel order)
    internal_value_[new_node_index] = leaf_value_[leaf_index];
    leaf_value_[leaf_index] = std::isnan(s.left_value) ? 0.0f : s.left_value;
    internal_count_[new_node_index] = s.left_count + s.right_count;
    leaf_count_[leaf_index] = s.left_count;
    leaf_value_[num_leaves_] = std::isnan(s.right_value) ? 0.0f : s.right_value;
    leaf_weight_[num_leaves_] = s.right_sum_hessians;
    leaf_count_[num_leaves_] = s.right_count;
    leaf_depth_[num_leaves_] = leaf_depth_[leaf_index] + 1;
    ++leaf_depth_[leaf_index];
    if (s.num_cat_threshold > 0) {
      // replay of Tree::SplitCategorical's threshold bookkeeping: the node's
      // threshold is an index into the categorical bitset tables, whose words
      // were prebuilt on host (SelectiveFinalize) in the model-format layout
      int8_t decision_type = 0;
      SetDecisionType(&decision_type, true, kCategoricalMask);
      SetMissingType(&decision_type, static_cast<int8_t>(s.missing_type));
      decision_type_[new_node_index] = decision_type;
      threshold_in_bin_[new_node_index] = static_cast<uint32_t>(num_cat_);
      threshold_[new_node_index] = num_cat_;
      ++num_cat_;
      cat_boundaries_.push_back(cat_boundaries_.back() + static_cast<int>(s.cat_bitset.size()));
      for (const uint32_t word : s.cat_bitset) {
        cat_threshold_.push_back(word);
      }
      cat_boundaries_inner_.push_back(cat_boundaries_inner_.back() + static_cast<int>(s.cat_bitset_inner.size()));
      for (const uint32_t word : s.cat_bitset_inner) {
        cat_threshold_inner_.push_back(word);
      }
    } else {
      int8_t decision_type = 0;
      SetDecisionType(&decision_type, false, kCategoricalMask);
      SetDecisionType(&decision_type, s.default_left != 0, kDefaultLeftMask);
      SetMissingType(&decision_type, static_cast<int8_t>(s.missing_type));
      decision_type_[new_node_index] = decision_type;
      threshold_in_bin_[new_node_index] = s.threshold_in_bin;
      threshold_[new_node_index] = s.real_threshold;
    }
    RecordBranchFeatures(leaf_index, num_leaves_, s.real_feature_index);
    ++num_leaves_;
  }
  // upload the final leaf values into the (still max_leaves_-sized, possibly
  // pooled-view) device array: AddPredictionToScore, Shrinkage and the leaf
  // renewal kernels read them
  CopyFromHostToCUDADevice<double>(cuda_leaf_value_.RawData(), leaf_value_.data(),
    static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  ShrinkHostVectorsAndReleaseDevice();
}

void CUDATree::SyncLeafOutputFromHostToCUDA() {
  CopyFromHostToCUDADevice<double>(cuda_leaf_value_.RawData(), leaf_value_.data(), leaf_value_.size(), __FILE__, __LINE__);
}

void CUDATree::SyncLeafOutputFromCUDAToHost() {
  CopyFromCUDADeviceToHost<double>(leaf_value_.data(), cuda_leaf_value_.RawData(), leaf_value_.size(), __FILE__, __LINE__);
}

void CUDATree::AsConstantTree(double val, int count) {
  Tree::AsConstantTree(val, count);
  // After ToHost, cuda_leaf_value_ may have been Resize()d to a smaller size
  // and cuda_leaf_count_ Clear()ed. GBDT calls AsConstantTree on 1-leaf trees,
  // so realloc cuda_leaf_value_ at size 1 if needed and skip the count write
  // when its storage was freed.
  if (cuda_leaf_value_.Size() == 0) {
    cuda_leaf_value_.Resize(1);
  }
  CopyFromHostToCUDADevice<double>(cuda_leaf_value_.RawData(), &val, 1, __FILE__, __LINE__);
  if (cuda_leaf_count_.Size() > 0) {
    CopyFromHostToCUDADevice<int>(cuda_leaf_count_.RawData(), &count, 1, __FILE__, __LINE__);
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
