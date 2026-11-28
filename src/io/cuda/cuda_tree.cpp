/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_tree.hpp>
#include <Falcata/falcata_plan.h>

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

namespace Falcata {

CUDATree::CUDATree(int max_leaves, bool track_branch_features, bool is_linear,
  const int gpu_device_id, const bool has_categorical_feature,
  uint8_t* pooled_device_buffer, int leaf_value_dim, CUDATreeCatArena* cat_arena,
  cudaStream_t tree_stream, CUDATreeLeafArena* leaf_arena,
  int host_reserve_leaves, bool defer_root_init):
Tree(max_leaves, track_branch_features, is_linear, leaf_value_dim, DeferredHostStorageTag{}, host_reserve_leaves),
leaf_arena_(leaf_arena),
num_threads_per_block_add_prediction_to_score_(1024) {
  is_cuda_tree_ = true;
  if (gpu_device_id >= 0) {
    SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
  } else {
    SetCUDADevice(0, __FILE__, __LINE__);
  }
  if (has_categorical_feature && cat_arena != nullptr) {
    // take over the learner's growth buffers (capacity kept from earlier trees); the bitsets restart empty and the
    // boundaries need max_leaves entries (SplitCategoricalKernel writes [num_cat + 1], the stale rest is never read)
    cat_arena_ = cat_arena;
    cuda_bitset_.Swap(&cat_arena->bitset);
    cuda_bitset_inner_.Swap(&cat_arena->bitset_inner);
    cuda_cat_boundaries_.Swap(&cat_arena->boundaries);
    cuda_cat_boundaries_inner_.Swap(&cat_arena->boundaries_inner);
    cuda_bitset_.ClearKeepCapacity();
    cuda_bitset_inner_.ClearKeepCapacity();
    if (cuda_cat_boundaries_.Size() < static_cast<size_t>(max_leaves)) {
      cuda_cat_boundaries_.ResizeDiscard(max_leaves);
    }
    if (cuda_cat_boundaries_inner_.Size() < static_cast<size_t>(max_leaves)) {
      cuda_cat_boundaries_inner_.ResizeDiscard(max_leaves);
    }
  } else if (has_categorical_feature) {
    cuda_cat_boundaries_.Resize(max_leaves);
    cuda_cat_boundaries_inner_.Resize(max_leaves);
  }
  if (leaf_value_dim_ > 1) {
    cuda_leaf_values_vec_.Resize(static_cast<size_t>(max_leaves_) * leaf_value_dim_);
  }
  InitCUDAMemory(pooled_device_buffer, tree_stream, defer_root_init && leaf_value_dim_ <= 1);
}

CUDATree::CUDATree(const Tree* host_tree):
  Tree(*host_tree),
  num_threads_per_block_add_prediction_to_score_(1024) {
  is_cuda_tree_ = true;
  InitCUDA();
}

CUDATree::~CUDATree() {
  // CUDAVector members handle their own deallocation. ToHost() may have
  // already released the stream — guard against double-destroy and never
  // destroy a stream borrowed from the tree learner.
  if (cuda_stream_ != nullptr && owns_cuda_stream_) {
    gpuAssert(cudaStreamDestroy(cuda_stream_), __FILE__, __LINE__);
  }
  if (prepared_leaf_value_ != nullptr) {
    gpuAssert(cudaFree(prepared_leaf_value_), __FILE__, __LINE__);
  }
}

void CUDATree::PrepareRetainedLeafValues(const int num_leaves) {
  if (leaf_arena_ != nullptr || num_leaves <= 0 || num_leaves > max_leaves_ || !cuda_leaf_value_.IsView()) {
    return;
  }
  if (prepared_leaf_value_ != nullptr && prepared_leaf_value_size_ == static_cast<size_t>(num_leaves)) {
    return;
  }
  DeallocateCUDAMemory<double>(&prepared_leaf_value_, __FILE__, __LINE__);
  prepared_leaf_value_size_ = 0;
  AllocateCUDAMemory<double>(&prepared_leaf_value_, static_cast<size_t>(num_leaves), __FILE__, __LINE__);
  prepared_leaf_value_size_ = static_cast<size_t>(num_leaves);
}

void CUDATree::InitCUDAMemory(uint8_t* pooled_device_buffer, cudaStream_t tree_stream, bool defer_root_init) {
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
    const CUDATreePooledLayout layout(max_leaves_);
    const auto address = [&](CUDATreePooledLayout::Field field) {
      return pooled_device_buffer + layout.fields[field].offset;
    };
    cuda_threshold_.SetView(reinterpret_cast<double*>(address(CUDATreePooledLayout::kThreshold)), L);
    cuda_internal_weight_.SetView(reinterpret_cast<double*>(address(CUDATreePooledLayout::kInternalWeight)), L);
    cuda_internal_value_.SetView(reinterpret_cast<double*>(address(CUDATreePooledLayout::kInternalValue)), L);
    cuda_leaf_value_.SetView(reinterpret_cast<double*>(address(CUDATreePooledLayout::kLeafValue)), L);
    cuda_leaf_weight_.SetView(reinterpret_cast<double*>(address(CUDATreePooledLayout::kLeafWeight)), L);
    cuda_batch_splits_.SetView(reinterpret_cast<CUDATreeBatchSplit*>(
      pooled_device_buffer + layout.batch_splits_offset), layout.batch_splits_count);
    cuda_left_child_.SetView(reinterpret_cast<int*>(address(CUDATreePooledLayout::kLeftChild)), L);
    cuda_right_child_.SetView(reinterpret_cast<int*>(address(CUDATreePooledLayout::kRightChild)), L);
    cuda_split_feature_inner_.SetView(reinterpret_cast<int*>(address(CUDATreePooledLayout::kSplitFeatureInner)), L);
    cuda_split_feature_.SetView(reinterpret_cast<int*>(address(CUDATreePooledLayout::kSplitFeature)), L);
    cuda_leaf_depth_.SetView(reinterpret_cast<int*>(address(CUDATreePooledLayout::kLeafDepth)), L);
    cuda_leaf_parent_.SetView(reinterpret_cast<int*>(address(CUDATreePooledLayout::kLeafParent)), L);
    cuda_threshold_in_bin_.SetView(reinterpret_cast<uint32_t*>(address(CUDATreePooledLayout::kThresholdInBin)), L);
    cuda_leaf_count_.SetView(reinterpret_cast<data_size_t*>(address(CUDATreePooledLayout::kLeafCount)), L);
    cuda_internal_count_.SetView(reinterpret_cast<data_size_t*>(address(CUDATreePooledLayout::kInternalCount)), L);
    cuda_split_gain_.SetView(reinterpret_cast<float*>(address(CUDATreePooledLayout::kSplitGain)), L);
    cuda_decision_type_.SetView(reinterpret_cast<int8_t*>(address(CUDATreePooledLayout::kDecisionType)), L);
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
  if (!defer_root_init) {
    // Single tiny default-stream launch initializes root leaf_value/leaf_weight/leaf_parent
    // (and root vector outputs when leaf_value_dim_ > 1), ordered after any previous tree's
    // stream-0 D2D leaf copy and before every blocking-stream consumer (find_stream, tree_stream).
    LaunchInitTreeRootKernel();
    root_initialized_ = true;
  } else {
    root_initialized_ = false;
  }
  if (tree_stream != nullptr) {
    cuda_stream_ = tree_stream;
    owns_cuda_stream_ = false;
  } else {
    CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));
    owns_cuda_stream_ = true;
  }
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
  if (leaf_value_dim_ > 1) {
    // a vector-leaf tree's outputs (leaf-major, leaf_value_dim_ per leaf): the vector score update and shrinkage
    // read them from here
    cuda_leaf_values_vec_.InitFromHostVector(leaf_values_vec_);
  }
  cuda_stream_ = nullptr;
  owns_cuda_stream_ = false;
  root_initialized_ = true;
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

int CUDATree::Split(const int leaf_index,
           const int real_feature_index,
           const double real_threshold,
           const MissingType missing_type,
           const CUDASplitInfo* cuda_split_info) {
  EnsureRootInitializedOnDevice();
  LaunchSplitKernel(leaf_index, real_feature_index, real_threshold, missing_type, cuda_split_info);
  RecordBranchFeatures(leaf_index, num_leaves_, real_feature_index);
  // mirror CPU Tree::Split: keep host-side leaf_depth_ in sync so leaf_depth(idx)
  // returns the correct value for max_depth and other host-side checks.
  UpdateLeafDepthOnSplit(leaf_index);
  ++num_leaves_;
  return num_leaves_ - 1;
}

void CUDATree::SplitBatch(const std::vector<CUDATreeBatchSplit>& splits) {
  if (splits.empty()) {
    return;
  }
  EnsureRootInitializedOnDevice();
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
    UpdateLeafDepthOnSplit(split.leaf_index);
    ++num_leaves_;
  }
}

void CUDATree::SplitLevel(const std::vector<CUDATreeBatchSplit>& splits, const std::vector<uint8_t>& is_categorical,
                          const CUDATreeCatLevel& cat) {
  CHECK_EQ(splits.size(), is_categorical.size());
  if (splits.empty()) {
    return;
  }
  EnsureRootInitializedOnDevice();
  if (cuda_batch_splits_.Size() < splits.size()) {
    // preallocate for the deepest possible level so this resizes at most once (as SplitBatch)
    cuda_batch_splits_.Resize(std::max(splits.size(), static_cast<size_t>(max_leaves_ / 2 + 2)));
  }
  // launch order: the numerical splits (SplitBatchKernel reads [0, num_numerical)), then the categorical ones in
  // level order (the categorical kernel reads [num_numerical, size)); the c-th categorical split is the level's c-th
  host_level_splits_.clear();
  for (size_t k = 0; k < splits.size(); ++k) {
    if (is_categorical[k] == 0) {
      host_level_splits_.push_back(splits[k]);
    }
  }
  const int num_numerical = static_cast<int>(host_level_splits_.size());
  for (size_t k = 0; k < splits.size(); ++k) {
    if (is_categorical[k] != 0) {
      host_level_splits_.push_back(splits[k]);
    }
  }
  const int num_cat_splits = static_cast<int>(splits.size()) - num_numerical;
  if (num_cat_splits > 0) {
    // the level's worst case must fit before the launch (its lengths are known only on the device); the bound is
    // exact again after a readback, so a buffer that keeps up with the record reads nothing back
    const size_t level_words = static_cast<size_t>(num_cat_splits) * cat.bitset_stride;
    const size_t level_words_inner = static_cast<size_t>(num_cat_splits) * cat.bitset_inner_stride;
    if (!cat_record_on_device_) {
      cat_words_bound_ = cuda_bitset_.Size();
      cat_words_inner_bound_ = cuda_bitset_inner_.Size();
    }
    if (cat_words_bound_ + level_words > cuda_bitset_.Capacity() ||
        cat_words_inner_bound_ + level_words_inner > cuda_bitset_inner_.Capacity()) {
      if (cat_record_on_device_) {
        ++cat_overflow_readbacks_;
      }
      SettleCategoricalRecordSizes();
      cat_words_bound_ = cuda_bitset_.Size();
      cat_words_inner_bound_ = cuda_bitset_inner_.Size();
      // grow by doubling, as PushBack (Reserve keeps the Size() words already recorded)
      if (cat_words_bound_ + level_words > cuda_bitset_.Capacity()) {
        cuda_bitset_.Reserve(std::max(cat_words_bound_ + level_words, 2 * cuda_bitset_.Capacity()));
      }
      if (cat_words_inner_bound_ + level_words_inner > cuda_bitset_inner_.Capacity()) {
        cuda_bitset_inner_.Reserve(std::max(cat_words_inner_bound_ + level_words_inner,
                                            2 * cuda_bitset_inner_.Capacity()));
      }
    }
  }
  // async H2D on cuda_stream_, ordered after any earlier kernel of this stream reading the buffer (see SplitBatch)
  CopyFromHostToCUDADeviceAsync<CUDATreeBatchSplit>(cuda_batch_splits_.RawData(),
    host_level_splits_.data(), host_level_splits_.size(), cuda_stream_, __FILE__, __LINE__);
  if (num_numerical > 0) {
    LaunchSplitBatchKernel(num_numerical);
  }
  if (num_cat_splits > 0) {
    LaunchSplitCategoricalLevelKernel(num_numerical, num_cat_splits, cat);
    cat_words_bound_ += static_cast<size_t>(num_cat_splits) * cat.bitset_stride;
    cat_words_inner_bound_ += static_cast<size_t>(num_cat_splits) * cat.bitset_inner_stride;
    cat_record_on_device_ = true;
    ++cat_level_launches_;
    cat_level_splits_ += num_cat_splits;
  }
  // host mirrors, in the exact order the per-split loop would apply them
  for (size_t k = 0; k < splits.size(); ++k) {
    const CUDATreeBatchSplit& split = splits[k];
    CHECK_EQ(split.num_leaves_at_split, num_leaves_);
    RecordBranchFeatures(split.leaf_index, num_leaves_, split.real_feature_index);
    UpdateLeafDepthOnSplit(split.leaf_index);
    ++num_leaves_;
    if (is_categorical[k] != 0) {
      ++num_cat_;
    }
  }
}

void CUDATree::ReserveCategoricalRecord(const size_t words, const size_t words_inner) {
  // before any record of this tree (the constructor left both bitsets empty): a Reserve copies nothing
  CHECK(!cat_record_on_device_);
  CHECK_EQ(num_cat_, 0);
  cuda_bitset_.Reserve(words);
  cuda_bitset_inner_.Reserve(words_inner);
}

void CUDATree::ReplayDeviceCategoricalRecord(const int levels_with_cat, const int num_cat_splits,
                                             const size_t bitset_stride, const size_t bitset_inner_stride) {
  if (num_cat_splits == 0) {
    return;
  }
  CHECK(!cat_record_on_device_);
  CHECK_EQ(num_cat_, 0);
  // what SplitLevel's bookkeeping leaves after the same levels: the bound counts every categorical split at its
  // stride on top of the exact (empty) start; the record's words stay on the device until ToHost
  cat_words_bound_ = cuda_bitset_.Size() + static_cast<size_t>(num_cat_splits) * bitset_stride;
  cat_words_inner_bound_ = cuda_bitset_inner_.Size() + static_cast<size_t>(num_cat_splits) * bitset_inner_stride;
  CHECK_LE(cat_words_bound_, cuda_bitset_.Capacity());
  CHECK_LE(cat_words_inner_bound_, cuda_bitset_inner_.Capacity());
  cat_record_on_device_ = true;
  cat_level_launches_ += levels_with_cat;
  cat_level_splits_ += num_cat_splits;
  num_cat_ += num_cat_splits;
}

void CUDATree::SettleCategoricalRecordSizes() {
  if (!cat_record_on_device_) {
    return;
  }
  cat_record_on_device_ = false;
  if (num_cat_ == 0) {
    cuda_bitset_.SetSizeWithinCapacity(0);
    cuda_bitset_inner_.SetSizeWithinCapacity(0);
    return;
  }
  // boundaries[num_cat_] is the record's word count (legacy-stream copies wait for the blocking tree stream)
  int words = 0;
  int words_inner = 0;
  CopyFromCUDADeviceToHost<int>(&words, cuda_cat_boundaries_.RawData() + num_cat_, 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(&words_inner, cuda_cat_boundaries_inner_.RawData() + num_cat_, 1, __FILE__, __LINE__);
  cuda_bitset_.SetSizeWithinCapacity(static_cast<size_t>(words));
  cuda_bitset_inner_.SetSizeWithinCapacity(static_cast<size_t>(words_inner));
}

int CUDATree::SplitCategorical(const int leaf_index,
           const int real_feature_index,
           const MissingType missing_type,
           const CUDASplitInfo* cuda_split_info,
           uint32_t* cuda_bitset,
           size_t cuda_bitset_len,
           uint32_t* cuda_bitset_inner,
           size_t cuda_bitset_inner_len) {
  // a tree records its categorical splits either all by level (SplitLevel, or the complete device tree's
  // LaunchSplitLevelKernelsDevice) or all here: the level flows that record by level grow the tree to the end, so no
  // per-split categorical split follows a level record. The host level loop runs only where HybridGrowthUsable admits
  // it (depth-limited, or selective growth, which records the whole tree from its own records); depth-limited, the
  // budget binds at most on the final level, which CUDASingleGPUTreeLearner::ArbitrateLevelBudget applies as one
  // partial level, never through the leaf-wise tail. The device tree's only tail is the root tail, after no split.
  CHECK(!cat_record_on_device_);
  EnsureRootInitializedOnDevice();
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
  UpdateLeafDepthOnSplit(leaf_index);
  ++num_leaves_;
  ++num_cat_;
  return num_leaves_ - 1;
}

data_size_t CUDATree::RebuildInternalCounts(const int node) {
  const int left = left_child_[node];
  const int right = right_child_[node];
  const data_size_t left_count = left >= 0 ? RebuildInternalCounts(left) : leaf_count_[~left];
  const data_size_t right_count = right >= 0 ? RebuildInternalCounts(right) : leaf_count_[~right];
  internal_count_[node] = left_count + right_count;
  return internal_count_[node];
}

void CUDATree::SyncNodeCountsFromPartition(const data_size_t* leaf_num_data, const int num_leaves) {
  if (leaf_num_data == nullptr || num_leaves < num_leaves_ ||
      static_cast<int>(leaf_count_.size()) < num_leaves_) {
    return;
  }
  if (num_leaves_ > 0) {
    std::memcpy(leaf_count_.data(), leaf_num_data, static_cast<size_t>(num_leaves_) * sizeof(data_size_t));
  }
  if (num_leaves_ > 1 && static_cast<int>(internal_count_.size()) >= num_leaves_ - 1) {
    RebuildInternalCounts(0);
  }
}

void CUDATree::SyncNodeCountsFromPartition(const std::vector<data_size_t>& leaf_num_data) {
  SyncNodeCountsFromPartition(leaf_num_data.data(), static_cast<int>(leaf_num_data.size()));
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
  if (!FalcataPlan::Get().async_tree_start) {
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }
}

void CUDATree::AddPredictionToScore(const Dataset* data,
                                    const data_size_t* used_data_indices,
                                    data_size_t num_data, double* score) const {
  LaunchAddPredictionToScoreKernel(data, used_data_indices, num_data, score);
  if (!FalcataPlan::Get().async_tree_start) {
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }
}

void CUDATree::LeafIndexOnDevice(const Dataset* data, const data_size_t* used_data_indices,
                                 data_size_t num_data, int* leaf_index) const {
  if (num_leaves_ <= 1) {
    // a stump has no internal node to traverse: every row is in leaf 0
    SetCUDAMemory<int>(leaf_index, 0, static_cast<size_t>(data->num_data()), __FILE__, __LINE__);
    return;
  }
  LaunchAddPredictionToScoreKernel(data, used_data_indices, num_data, nullptr, leaf_index);
}

void CUDATree::SetVectorLeafValuesFromSplit(const int left_leaf_index, const int right_leaf_index,
                                            const CUDASplitInfo* cuda_split_info) {
  CHECK_GT(leaf_value_dim_, 1);
  LaunchSetVectorLeafValuesFromSplitKernel(left_leaf_index, right_leaf_index, cuda_split_info);
}

void CUDATree::SetVectorLeafValuesFromSplitBatch(const int num_splits) {
  CHECK_GT(leaf_value_dim_, 1);
  if (num_splits <= 0) {
    return;
  }
  LaunchSetVectorLeafValuesFromSplitBatchKernel(num_splits);
}

inline void CUDATree::Shrinkage(double rate) {
  Tree::Shrinkage(rate);
  LaunchShrinkageKernel(rate);
}

inline void CUDATree::AddBias(double val) {
  Tree::AddBias(val);
  LaunchAddBiasKernel(val);
}

bool CUDATree::LaunchSlabReadback(void* readback_staging_device, size_t readback_staging_bytes) const {
  const size_t slab_bytes = PooledDeviceBufferSize(max_leaves_);
  if (!pooled_ || readback_staging_device == nullptr || readback_staging_bytes < slab_bytes) {
    return false;
  }
  // cuda_threshold_ is the first view carved from the slab (see InitCUDAMemory)
  LaunchCopyToMappedHost(readback_staging_device, cuda_threshold_.RawDataReadOnly(), slab_bytes, __FILE__, __LINE__);
  return true;
}

void CUDATree::ToHost(void* readback_staging_host, void* readback_staging_device,
                      size_t readback_staging_bytes, const std::function<void()>& after_device_copy,
                      bool slab_in_staging) {
  EnsureRootInitializedOnDevice();
  const size_t num_leaves_size = static_cast<size_t>(std::max(num_leaves_, 0));
  const size_t n_internal = num_leaves_size > 1 ? num_leaves_size - 1 : 0;

  if (pooled_) {
    // all per-tree arrays live in one contiguous slab: ONE D2H copy replaces the
    // 16 individual synchronous copies below (each pays a full launch+sync
    // round trip). The scatter into the host vectors uses assign() over the
    // constructor's pre-reserved capacity (zero malloc, zero memset(0) on the
    // GPU-idle critical path).
    const CUDATreePooledLayout layout(max_leaves_);
    const size_t slab_bytes = layout.slab_bytes;
    std::vector<uint8_t> staging;
    // cuda_threshold_ is the first view carved from the slab (see InitCUDAMemory)
    const uint8_t* slab = reinterpret_cast<const uint8_t*>(cuda_threshold_.RawData());
    const uint8_t* p = nullptr;
    if (readback_staging_device != nullptr && readback_staging_bytes >= slab_bytes) {
      // cuda_plan key readback_kernel: copy kernel on the default stream into the caller's mapped pinned buffer
      // (cuda_plan key final_readback_first: already copied there and synchronized, see LaunchSlabReadback)
      // CopyFromCUDADeviceToMappedHost syncs stream 0 before returning, so p is safe to read immediately
      if (!slab_in_staging) {
        CopyFromCUDADeviceToMappedHost(readback_staging_device, readback_staging_host, slab, slab_bytes,
                                       __FILE__, __LINE__);
      }
      p = static_cast<const uint8_t*>(readback_staging_host);
    } else {
      staging.resize(slab_bytes);
      CopyFromCUDADeviceToHost<uint8_t>(staging.data(), slab, slab_bytes, __FILE__, __LINE__);
      p = staging.data();
    }
    if (after_device_copy) after_device_copy();
    // Preserve all existing reads, including leaf_parent I, through the same layout as the producer.
    const auto scatter_assign = [&](CUDATreePooledLayout::Field field, auto* dst_vec) {
      using Elem = typename std::remove_reference<decltype(*dst_vec)>::type::value_type;
      const auto& prefix = layout.fields[field];
      const size_t entries = prefix.leaf_entries ? num_leaves_size : n_internal;
      const Elem* src = reinterpret_cast<const Elem*>(p + prefix.offset);
      dst_vec->assign(src, src + entries);
    };
    scatter_assign(CUDATreePooledLayout::kThreshold, &threshold_);
    scatter_assign(CUDATreePooledLayout::kInternalWeight, &internal_weight_);
    scatter_assign(CUDATreePooledLayout::kInternalValue, &internal_value_);
    scatter_assign(CUDATreePooledLayout::kLeafValue, &leaf_value_);
    scatter_assign(CUDATreePooledLayout::kLeafWeight, &leaf_weight_);
    scatter_assign(CUDATreePooledLayout::kLeftChild, &left_child_);
    scatter_assign(CUDATreePooledLayout::kRightChild, &right_child_);
    scatter_assign(CUDATreePooledLayout::kSplitFeatureInner, &split_feature_inner_);
    scatter_assign(CUDATreePooledLayout::kSplitFeature, &split_feature_);
    scatter_assign(CUDATreePooledLayout::kLeafDepth, &leaf_depth_);
    scatter_assign(CUDATreePooledLayout::kLeafParent, &leaf_parent_);
    scatter_assign(CUDATreePooledLayout::kThresholdInBin, &threshold_in_bin_);
    scatter_assign(CUDATreePooledLayout::kLeafCount, &leaf_count_);
    scatter_assign(CUDATreePooledLayout::kInternalCount, &internal_count_);
    scatter_assign(CUDATreePooledLayout::kSplitGain, &split_gain_);
    scatter_assign(CUDATreePooledLayout::kDecisionType, &decision_type_);
  } else {
    left_child_.resize(n_internal);
    right_child_.resize(n_internal);
    split_feature_inner_.resize(n_internal);
    split_feature_.resize(n_internal);
    threshold_in_bin_.resize(n_internal);
    threshold_.resize(n_internal);
    decision_type_.resize(n_internal, 0);
    split_gain_.resize(n_internal);
    leaf_parent_.resize(num_leaves_size);
    leaf_value_.resize(num_leaves_size);
    leaf_weight_.resize(num_leaves_size);
    leaf_count_.resize(num_leaves_size);
    internal_value_.resize(n_internal);
    internal_weight_.resize(n_internal);
    internal_count_.resize(n_internal);
    leaf_depth_.resize(num_leaves_size);
    if (n_internal > 0) {
      CopyFromCUDADeviceToHost<int>(left_child_.data(), cuda_left_child_.RawData(), n_internal, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<int>(right_child_.data(), cuda_right_child_.RawData(), n_internal, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<int>(split_feature_inner_.data(), cuda_split_feature_inner_.RawData(), n_internal, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<int>(split_feature_.data(), cuda_split_feature_.RawData(), n_internal, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<uint32_t>(threshold_in_bin_.data(), cuda_threshold_in_bin_.RawData(), n_internal, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<double>(threshold_.data(), cuda_threshold_.RawData(), n_internal, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<int8_t>(decision_type_.data(), cuda_decision_type_.RawData(), n_internal, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<float>(split_gain_.data(), cuda_split_gain_.RawData(), n_internal, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<double>(internal_value_.data(), cuda_internal_value_.RawData(), n_internal, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<double>(internal_weight_.data(), cuda_internal_weight_.RawData(), n_internal, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<data_size_t>(internal_count_.data(), cuda_internal_count_.RawData(), n_internal, __FILE__, __LINE__);
    }
    if (num_leaves_size > 0) {
      CopyFromCUDADeviceToHost<int>(leaf_parent_.data(), cuda_leaf_parent_.RawData(), num_leaves_size, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<double>(leaf_value_.data(), cuda_leaf_value_.RawData(), num_leaves_size, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<double>(leaf_weight_.data(), cuda_leaf_weight_.RawData(), num_leaves_size, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<data_size_t>(leaf_count_.data(), cuda_leaf_count_.RawData(), num_leaves_size, __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<int>(leaf_depth_.data(), cuda_leaf_depth_.RawData(), num_leaves_size, __FILE__, __LINE__);
    }
    if (after_device_copy) after_device_copy();
  }

  if (num_cat_ > 0) {
    // the first num_cat_ + 1 boundaries are the record (the buffers hold max_leaves or more)
    const size_t num_boundaries = static_cast<size_t>(num_cat_) + 1;
    if (!ReadCategoricalRecordOneCopy()) {
      SettleCategoricalRecordSizes();
      cat_boundaries_.resize(num_boundaries);
      cat_boundaries_inner_.resize(num_boundaries);
      CopyFromCUDADeviceToHost<int>(cat_boundaries_.data(), cuda_cat_boundaries_.RawData(), num_boundaries,
                                    __FILE__, __LINE__);
      CopyFromCUDADeviceToHost<int>(cat_boundaries_inner_.data(), cuda_cat_boundaries_inner_.RawData(),
                                    num_boundaries, __FILE__, __LINE__);
      cat_threshold_ = cuda_bitset_.ToHost();
      cat_threshold_inner_ = cuda_bitset_inner_.ToHost();
    }
    if (cat_arena_ == nullptr) {
      // no arena: shrink to the record as before (the retained inner copies are what prediction reads)
      cuda_cat_boundaries_inner_.Resize(num_boundaries);
      cuda_cat_boundaries_.Resize(num_boundaries);
    }
  }

  if (leaf_value_dim_ > 1) {
    const size_t vec_elems = num_leaves_size * static_cast<size_t>(leaf_value_dim_);
    leaf_values_vec_.resize(vec_elems);
    if (vec_elems > 0) {
      CopyFromCUDADeviceToHost<double>(leaf_values_vec_.data(), cuda_leaf_values_vec_.RawData(),
        vec_elems, __FILE__, __LINE__);
    }
  }

  ShrinkHostVectorsAndReleaseDevice();
}

bool CUDATree::ReadCategoricalRecordOneCopy() {
  // the categorical level record: the boundaries hold the word counts, so one pack kernel gathers both boundary
  // arrays and both bitsets (each up to its bound) into the arena's staging and one copy brings them over, instead
  // of four copies of which two wait for the other two's counts. When the bound passes kMaxWords (high-cardinality
  // features) the four copies stay: the bound may then be far above the record.
  constexpr size_t kMaxWords = static_cast<size_t>(1) << 16;
  if (cat_level_launches_ == 0 || cat_arena_ == nullptr || num_cat_ == 0) {
    return false;
  }
  const size_t num_boundaries = static_cast<size_t>(num_cat_) + 1;
  const size_t words_bound = std::min(cat_words_bound_, cuda_bitset_.Capacity());
  const size_t words_inner_bound = std::min(cat_words_inner_bound_, cuda_bitset_inner_.Capacity());
  const size_t total = 2 * num_boundaries + words_bound + words_inner_bound;
  if (total > kMaxWords) {
    return false;
  }
  CUDAVector<uint32_t>& staging = cat_arena_->readback;
  if (staging.Size() < total) {
    staging.ResizeDiscard(std::max(total, 2 * staging.Size()));
  }
  LaunchPackCategoricalRecordKernel(staging.RawData(), words_bound, words_inner_bound);
  // host staging kept by the learner's arena (no allocation per tree once it has grown)
  std::vector<uint32_t>& packed = cat_arena_->readback_host;
  if (packed.size() < total) {
    packed.resize(total);
  }
  // legacy-stream copy: waits for the pack kernel on the (blocking) tree stream
  CopyFromCUDADeviceToHost<uint32_t>(packed.data(), staging.RawData(), total, __FILE__, __LINE__);
  cat_boundaries_.resize(num_boundaries);
  cat_boundaries_inner_.resize(num_boundaries);
  std::memcpy(cat_boundaries_.data(), packed.data(), num_boundaries * sizeof(int));
  std::memcpy(cat_boundaries_inner_.data(), packed.data() + num_boundaries, num_boundaries * sizeof(int));
  const size_t words = static_cast<size_t>(cat_boundaries_[num_cat_]);
  const size_t words_inner = static_cast<size_t>(cat_boundaries_inner_[num_cat_]);
  CHECK_LE(words, words_bound);
  CHECK_LE(words_inner, words_inner_bound);
  const uint32_t* words_begin = packed.data() + 2 * num_boundaries;
  cat_threshold_.assign(words_begin, words_begin + words);
  cat_threshold_inner_.assign(words_begin + words_bound, words_begin + words_bound + words_inner);
  cuda_bitset_.SetSizeWithinCapacity(words);
  cuda_bitset_inner_.SetSizeWithinCapacity(words_inner);
  cat_record_on_device_ = false;
  cat_record_one_copy_ = true;
  return true;
}

void CUDATree::ShrinkHostVectorsAndReleaseDevice() {
  // Trim any host vector that still exceeds its exact size (ToHost and
  // RebuildFromHostSplits size all 16 directly to num_leaves_, so the common
  // path skips all 16 shrink_to_fit() reallocations).
  if (num_leaves_ > 0) {
    const size_t n_internal = static_cast<size_t>(num_leaves_) - 1;
    const size_t n_leaves = static_cast<size_t>(num_leaves_);
    const auto trim_if_needed = [](auto& vec, const size_t target) {
      if (vec.size() != target) {
        vec.resize(target);
      }
      if (vec.capacity() > target * 2 + 16) {
        vec.shrink_to_fit();
      }
    };
    trim_if_needed(left_child_, n_internal);
    trim_if_needed(right_child_, n_internal);
    trim_if_needed(split_feature_inner_, n_internal);
    trim_if_needed(split_feature_, n_internal);
    trim_if_needed(threshold_in_bin_, n_internal);
    trim_if_needed(threshold_, n_internal);
    trim_if_needed(decision_type_, n_internal);
    trim_if_needed(split_gain_, n_internal);
    trim_if_needed(leaf_parent_, n_leaves);
    trim_if_needed(internal_value_, n_internal);
    trim_if_needed(internal_weight_, n_internal);
    trim_if_needed(internal_count_, n_internal);
    trim_if_needed(leaf_value_, n_leaves);
    trim_if_needed(leaf_weight_, n_leaves);
    trim_if_needed(leaf_count_, n_leaves);
    trim_if_needed(leaf_depth_, n_leaves);
    if (leaf_value_dim_ > 1) {
      trim_if_needed(leaf_values_vec_, n_leaves * static_cast<size_t>(leaf_value_dim_));
    }
  }

  // cuda_plan key final_readback_first: a pooled tree's device arrays are views of the learner's slab, so nothing
  // below frees device memory (the retained leaf values are copied out on the default stream, ahead of the next
  // tree's writes to the slab): the host need not wait for the device, e.g. for a leaf-map pass queued after the
  // tree's copy (CUDASingleGPUTreeLearner::FlushEarlyLeafMap)
  if (!(pooled_ && FalcataPlan::Get().final_readback_first)) {
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }

  // Free per-tree GPU buffers no longer needed after ToHost. Without this,
  // accumulated per-tree GPU memory OOMs a 32GB device after ~6k trees on
  // the Numerai prod config. cuda_leaf_value_ is the only field needed
  // post-train (AddPredictionToScore / Shrinkage / AddBias read/write it);
  // sub-allocate from the learner's chunked refcounted slab arena when given
  // (1 cudaMalloc per ~128-1000 trees + async D2D copy on stream 0) so there
  // is zero per-tree cudaMalloc / sync cudaMemcpy and zero dangling pointers
  // if this tree outlives the learner.
  if (prepared_leaf_value_ != nullptr) {
    if (cuda_leaf_value_.IsView() && num_leaves_ > 0 && num_leaves_ <= max_leaves_ &&
        cuda_leaf_value_.Size() >= static_cast<size_t>(num_leaves_) &&
        prepared_leaf_value_size_ == static_cast<size_t>(num_leaves_)) {
      // cuda_plan key tree_end_prealloc: the bytes Resize would copy, into the buffer allocated at the final level,
      // on the same legacy default stream without the host wait (the readers are later work on that stream)
      CopyFromCUDADeviceToCUDADeviceAsync<double>(prepared_leaf_value_, cuda_leaf_value_.RawData(),
                                                  static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
      cuda_leaf_value_.Adopt(prepared_leaf_value_, static_cast<size_t>(num_leaves_));
    } else {
      DeallocateCUDAMemory<double>(&prepared_leaf_value_, __FILE__, __LINE__);
    }
    prepared_leaf_value_ = nullptr;
    prepared_leaf_value_size_ = 0;
  }
  if (leaf_arena_ != nullptr && cuda_leaf_value_.IsView() && num_leaves_ > 0 &&
      cuda_leaf_value_.Size() >= static_cast<size_t>(num_leaves_)) {
    const size_t n_leaves = static_cast<size_t>(num_leaves_);
    auto alloc = leaf_arena_->Allocate(n_leaves);
    CopyFromCUDADeviceToCUDADeviceAsync<double>(alloc.first, cuda_leaf_value_.RawData(),
                                                n_leaves, __FILE__, __LINE__);
    retained_leaf_slab_ = std::move(alloc.second);
    cuda_leaf_value_.SetView(alloc.first, n_leaves);
  }
  leaf_arena_ = nullptr;
  if (retained_leaf_slab_ == nullptr) {
    if (num_leaves_ > 0 && num_leaves_ < max_leaves_ && cuda_leaf_value_.Size() > 0) {
      cuda_leaf_value_.Resize(static_cast<size_t>(num_leaves_));
    }
    cuda_leaf_value_.Materialize();
  }
  // the vector-leaf outputs are retained for the same reasons (score updates,
  // shrinkage); already owned memory, just shrunk
  if (leaf_value_dim_ > 1 && num_leaves_ > 0 && num_leaves_ < max_leaves_ &&
      cuda_leaf_values_vec_.Size() > 0) {
    cuda_leaf_values_vec_.Resize(static_cast<size_t>(num_leaves_) * leaf_value_dim_);
  }
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
  ReturnCategoricalArena();

  // Release the per-tree CUDA stream — only destroy it if this tree created it.
  if (cuda_stream_ != nullptr) {
    if (owns_cuda_stream_) {
      gpuAssert(cudaStreamDestroy(cuda_stream_), __FILE__, __LINE__);
    }
    cuda_stream_ = nullptr;
    owns_cuda_stream_ = false;
  }
}

void CUDATree::ReturnCategoricalArena() {
  if (cat_arena_ == nullptr) {
    return;
  }
  // a record SplitLevel left pending (no ToHost in between) gets its exact sizes first
  SettleCategoricalRecordSizes();
  // AddPredictionToScore traverses this tree's categorical nodes through the inner bitsets for the tree's whole
  // lifetime: copy the record into refcounted arena slices on stream 0 (async D2D, no per-tree cudaMalloc/cudaFree).
  // The copies are ordered on the default stream after the blocking tree stream and before the next tree's writes to
  // the arena. A tree whose device record is empty (the selective flow builds categorical splits on the host) keeps
  // none, as before: prediction uploads them from the host vectors.
  CUDAVector<uint32_t> kept_bitset_inner;
  CUDAVector<int> kept_boundaries_inner;
  if (num_cat_ > 0 && cuda_bitset_inner_.Size() > 0) {
    const size_t num_words = cuda_bitset_inner_.Size();
    const size_t num_boundaries = static_cast<size_t>(num_cat_) + 1;
    auto words_alloc = cat_arena_->kept_bitset_inner_arena.Allocate(num_words);
    CopyFromCUDADeviceToCUDADeviceAsync<uint32_t>(words_alloc.first, cuda_bitset_inner_.RawData(),
                                                  num_words, __FILE__, __LINE__);
    retained_bitset_inner_slab_ = std::move(words_alloc.second);
    kept_bitset_inner.SetView(words_alloc.first, num_words);
    auto boundaries_alloc = cat_arena_->kept_boundaries_inner_arena.Allocate(num_boundaries);
    CopyFromCUDADeviceToCUDADeviceAsync<int>(boundaries_alloc.first, cuda_cat_boundaries_inner_.RawData(),
                                             num_boundaries, __FILE__, __LINE__);
    retained_boundaries_inner_slab_ = std::move(boundaries_alloc.second);
    kept_boundaries_inner.SetView(boundaries_alloc.first, num_boundaries);
  }
  cuda_bitset_.Swap(&cat_arena_->bitset);
  cuda_bitset_inner_.Swap(&cat_arena_->bitset_inner);
  cuda_cat_boundaries_.Swap(&cat_arena_->boundaries);
  cuda_cat_boundaries_inner_.Swap(&cat_arena_->boundaries_inner);
  // what the arena held is empty unless another tree took and returned it meanwhile; never keep it
  cuda_bitset_.Clear();
  cuda_bitset_inner_.Clear();
  cuda_cat_boundaries_.Clear();
  cuda_cat_boundaries_inner_.Clear();
  cuda_bitset_inner_.Swap(&kept_bitset_inner);
  cuda_cat_boundaries_inner_.Swap(&kept_boundaries_inner);
  cat_arena_ = nullptr;
}

void CUDATree::RebuildFromHostSplits(const std::vector<CUDATreeHostSplit>& splits) {
  // Host replay of SplitKernel, one split at a time in classic (greedy) order.
  // Size all 16 host vectors once directly to the exact final split/leaf counts.
  const size_t n_internal = splits.size();
  const size_t n_leaves = n_internal + 1;
  const double root_value = leaf_value_.empty() ? 0.0 : leaf_value_[0];
  left_child_.resize(n_internal);
  right_child_.resize(n_internal);
  split_feature_inner_.resize(n_internal);
  split_feature_.resize(n_internal);
  threshold_in_bin_.resize(n_internal);
  threshold_.resize(n_internal);
  decision_type_.resize(n_internal, 0);
  split_gain_.resize(n_internal);
  internal_value_.resize(n_internal);
  internal_weight_.resize(n_internal);
  internal_count_.resize(n_internal);
  leaf_parent_.resize(n_leaves);
  leaf_value_.resize(n_leaves);
  leaf_weight_.resize(n_leaves, 0.0);
  leaf_count_.resize(n_leaves, 0);
  leaf_depth_.resize(n_leaves);
  if (leaf_value_dim_ > 1) {
    leaf_values_vec_.resize(n_leaves * static_cast<size_t>(leaf_value_dim_));
  }
  leaf_parent_[0] = -1;
  leaf_depth_[0] = 0;
  leaf_value_[0] = root_value;
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
    if (leaf_value_dim_ > 1) {
      // host replay of SetVectorLeafValuesFromSplitKernel over the split's
      // captured vector payload
      CHECK_EQ(static_cast<int>(s.vec_left_values.size()), leaf_value_dim_);
      CHECK_EQ(static_cast<int>(s.vec_right_values.size()), leaf_value_dim_);
      const size_t left_base = static_cast<size_t>(leaf_index) * leaf_value_dim_;
      const size_t right_base = static_cast<size_t>(num_leaves_) * leaf_value_dim_;
      for (int target = 0; target < leaf_value_dim_; ++target) {
        const double left_value = s.vec_left_values[target];
        const double right_value = s.vec_right_values[target];
        leaf_values_vec_[left_base + target] = std::isnan(left_value) ? 0.0 : left_value;
        leaf_values_vec_[right_base + target] = std::isnan(right_value) ? 0.0 : right_value;
      }
    }
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
  // upload the final leaf values directly into the chunked leaf arena when
  // available (skipping an intermediate pooled-slab copy); AddPredictionToScore,
  // Shrinkage and the leaf renewal kernels read them.
  if (leaf_arena_ != nullptr && num_leaves_ > 0) {
    const size_t n_leaves_u = static_cast<size_t>(num_leaves_);
    auto alloc = leaf_arena_->Allocate(n_leaves_u);
    CopyFromHostToCUDADevice<double>(alloc.first, leaf_value_.data(), n_leaves_u, __FILE__, __LINE__);
    retained_leaf_slab_ = std::move(alloc.second);
    cuda_leaf_value_.SetView(alloc.first, n_leaves_u);
    leaf_arena_ = nullptr;
  } else {
    CopyFromHostToCUDADevice<double>(cuda_leaf_value_.RawData(), leaf_value_.data(),
      static_cast<size_t>(num_leaves_), __FILE__, __LINE__);
  }
  if (leaf_value_dim_ > 1) {
    CopyFromHostToCUDADevice<double>(cuda_leaf_values_vec_.RawData(), leaf_values_vec_.data(),
      static_cast<size_t>(num_leaves_) * leaf_value_dim_, __FILE__, __LINE__);
  }
  ShrinkHostVectorsAndReleaseDevice();
}

void CUDATree::SyncLeafOutputFromHostToCUDA() {
  EnsureRootInitializedOnDevice();
  CopyFromHostToCUDADevice<double>(cuda_leaf_value_.RawData(), leaf_value_.data(), leaf_value_.size(), __FILE__, __LINE__);
  if (leaf_value_dim_ > 1) {
    const size_t count = std::min(leaf_values_vec_.size(), cuda_leaf_values_vec_.Size());
    CopyFromHostToCUDADevice<double>(cuda_leaf_values_vec_.RawData(), leaf_values_vec_.data(), count, __FILE__, __LINE__);
  }
}

void CUDATree::SyncLeafOutputFromCUDAToHost() {
  CopyFromCUDADeviceToHost<double>(leaf_value_.data(), cuda_leaf_value_.RawData(), leaf_value_.size(), __FILE__, __LINE__);
  if (leaf_value_dim_ > 1) {
    const size_t count = std::min(leaf_values_vec_.size(), cuda_leaf_values_vec_.Size());
    CopyFromCUDADeviceToHost<double>(leaf_values_vec_.data(), cuda_leaf_values_vec_.RawData(), count, __FILE__, __LINE__);
  }
}

void CUDATree::AsConstantTree(double val, int count) {
  if (leaf_value_.empty()) {
    leaf_value_.resize(1);
  }
  if (leaf_count_.empty()) {
    leaf_count_.resize(1);
  }
  Tree::AsConstantTree(val, count);
  // After ToHost, cuda_leaf_value_ may have been Resize()d to a smaller size
  // and cuda_leaf_count_ Clear()ed. GBDT calls AsConstantTree on 1-leaf trees,
  // so realloc cuda_leaf_value_ at size 1 if needed and skip the count write
  // when its storage was freed.
  if (cuda_leaf_value_.Size() == 0) {
    cuda_leaf_value_.Resize(1);
  }
  CopyFromHostToCUDADevice<double>(cuda_leaf_value_.RawData(), &val, 1, __FILE__, __LINE__);
  if (leaf_value_dim_ > 1) {
    if (cuda_leaf_values_vec_.Size() < static_cast<size_t>(leaf_value_dim_)) {
      cuda_leaf_values_vec_.Resize(static_cast<size_t>(leaf_value_dim_));
    }
    CopyFromHostToCUDADevice<double>(cuda_leaf_values_vec_.RawData(), leaf_values_vec_.data(),
      static_cast<size_t>(leaf_value_dim_), __FILE__, __LINE__);
  }
  if (cuda_leaf_count_.Size() > 0) {
    CopyFromHostToCUDADevice<int>(cuda_leaf_count_.RawData(), &count, 1, __FILE__, __LINE__);
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
