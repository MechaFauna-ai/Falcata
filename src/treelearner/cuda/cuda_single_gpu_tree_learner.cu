/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include "cuda_single_gpu_tree_learner.hpp"

#include <Falcata/cuda/cuda_algorithms.hpp>
#include <Falcata/falcata_plan.h>
#include <Falcata/cuda/cuda_rocm_interop.h>

#include <algorithm>
#include <vector>

namespace Falcata {

// Vector-leaf plane fan-out: refresh the per-plane leaf-splits structs of both
// children from the primary structs the partition split just wrote. Plane t
// copies the primary struct of its role, then overrides the per-target
// gradient sum and mirror output from the split's vector payload and offsets
// the histogram pointer to plane t. The role's side (left vs right child) is
// identified by comparing the primary struct's leaf index against the split
// leaf (== the left child's index).
// Quantized training additionally re-packs the plane's exact integer parent
// total: the target-t gradient half comes from the payload, the hessian half
// from the role's primary struct (the hessian stream every target shares).
__device__ __forceinline__ int64_t VectorPlanePackedSum(
  const double* payload, const int num_targets, const int t, const bool is_left,
  const int64_t primary_packed) {
  const int32_t grad_int = static_cast<int32_t>(
    payload[(is_left ? kVecLeftGradInt : kVecRightGradInt) * num_targets + t]);
  return (static_cast<int64_t>(grad_int) << 32) |
         (primary_packed & 0x00000000ffffffffLL);
}

__global__ void VectorPlaneFanOutKernel(
  const CUDALeafSplitsStruct* smaller_struct,
  const CUDALeafSplitsStruct* larger_struct,
  CUDALeafSplitsStruct* planes,
  const int num_targets,
  const size_t plane_stride_entries,
  const CUDASplitInfo* split_info,
  const int left_leaf_index,
  const bool use_quantized_grad) {
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (i < 2 * num_targets) {
    const int role = i / num_targets;
    const int t = i % num_targets;
    const CUDALeafSplitsStruct* src = role == 0 ? smaller_struct : larger_struct;
    CUDALeafSplitsStruct s = *src;
    const bool is_left = (s.leaf_index == left_leaf_index);
    const double* payload = split_info->vec_payload;
    s.sum_of_gradients = payload[(is_left ? kVecLeftSumGradients : kVecRightSumGradients) * num_targets + t];
    s.leaf_value = payload[(is_left ? kVecLeftValue : kVecRightValue) * num_targets + t];
    if (use_quantized_grad) {
      s.sum_of_gradients_hessians = VectorPlanePackedSum(
        payload, num_targets, t, is_left, src->sum_of_gradients_hessians);
    }
    // the vector finder recomputes the parent gain from the plane sums, so the
    // scalar (target-0) gain field would be stale for t > 0; zero it
    s.gain = 0.0;
    s.hist_in_leaf = src->hist_in_leaf + static_cast<size_t>(t) * plane_stride_entries;
    planes[i] = s;
  }
}

void CUDASingleGPUTreeLearner::LaunchVectorPlaneFanOutKernel(
    const CUDASplitInfo* best_split_info, const int left_leaf_index) {
  const int num_slots = 2 * vec_num_targets_;
  VectorPlaneFanOutKernel<<<1, ((num_slots + 31) / 32) * 32>>>(
    cuda_smaller_leaf_splits_->GetCUDAStruct(),
    cuda_larger_leaf_splits_->GetCUDAStruct(),
    vec_plane_structs_.RawData(),
    vec_num_targets_,
    static_cast<size_t>(2) * num_total_bin_,
    best_split_info,
    left_leaf_index,
    config_->use_quantized_grad);
}

// Level-batched plane fan-out: the hybrid prefix's counterpart of
// VectorPlaneFanOutKernel. Thread i covers plane t of primary slot
// s = 2 * pair + role (role 0 = smaller, 1 = larger) and writes plane slab
// entry s * T + t, the layout the batched vector find kernel reads.
__global__ void VectorLevelPlaneFanOutKernel(
  const CUDALeafSplitsStruct* slots,
  const CUDASplitInfo* const* split_infos,
  const int* left_leaf_indices,
  CUDALeafSplitsStruct* plane_slab,
  const int num_targets,
  const size_t plane_stride_entries,
  const int num_pairs,
  const bool use_quantized_grad) {
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (i >= num_pairs * 2 * num_targets) {
    return;
  }
  const int slot = i / num_targets;
  const int t = i - slot * num_targets;
  const int pair = slot >> 1;
  const CUDALeafSplitsStruct* src = slots + slot;
  CUDALeafSplitsStruct s = *src;
  const bool is_left = (s.leaf_index == left_leaf_indices[pair]);
  const double* payload = split_infos[pair]->vec_payload;
  s.sum_of_gradients = payload[(is_left ? kVecLeftSumGradients : kVecRightSumGradients) * num_targets + t];
  s.leaf_value = payload[(is_left ? kVecLeftValue : kVecRightValue) * num_targets + t];
  if (use_quantized_grad) {
    s.sum_of_gradients_hessians = VectorPlanePackedSum(
      payload, num_targets, t, is_left, src->sum_of_gradients_hessians);
  }
  // the vector finder recomputes the parent gain from the plane sums, so the
  // scalar (target-0) gain field would be stale for t > 0; zero it
  s.gain = 0.0;
  s.hist_in_leaf = src->hist_in_leaf + static_cast<size_t>(t) * plane_stride_entries;
  plane_slab[i] = s;
}

void CUDASingleGPUTreeLearner::LaunchVectorLevelPlaneFanOutKernel(const int num_pairs) {
  const int num_slots = num_pairs * 2 * vec_num_targets_;
  const int num_threads_per_block = 128;
  const int num_blocks = (num_slots + num_threads_per_block - 1) / num_threads_per_block;
  VectorLevelPlaneFanOutKernel<<<num_blocks, num_threads_per_block>>>(
    hybrid_pair_slots_.RawDataReadOnly(),
    cuda_vec_fanout_infos_.RawDataReadOnly(),
    cuda_vec_fanout_left_leaves_.RawDataReadOnly(),
    vec_level_plane_slots_.RawData(),
    vec_num_targets_,
    static_cast<size_t>(2) * num_total_bin_,
    num_pairs,
    config_->use_quantized_grad);
}

__global__ void ReduceLeafStatKernel_SharedMemory(
  const score_t* gradients,
  const score_t* hessians,
  const int num_leaves,
  const data_size_t num_data,
  const int* data_index_to_leaf_index,
  double* leaf_grad_stat_buffer,
  double* leaf_hess_stat_buffer) {
  extern __shared__ double shared_mem[];
  double* shared_grad_sum = shared_mem;
  double* shared_hess_sum = shared_mem + num_leaves;
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  for (int leaf_index = static_cast<int>(threadIdx.x); leaf_index < num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    shared_grad_sum[leaf_index] = 0.0f;
    shared_hess_sum[leaf_index] = 0.0f;
  }
  __syncthreads();
  if (data_index < num_data) {
    const int leaf_index = data_index_to_leaf_index[data_index];
    atomicAdd_block(shared_grad_sum + leaf_index, gradients[data_index]);
    atomicAdd_block(shared_hess_sum + leaf_index, hessians[data_index]);
  }
  __syncthreads();
  for (int leaf_index = static_cast<int>(threadIdx.x); leaf_index < num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    atomicAdd_system(leaf_grad_stat_buffer + leaf_index, shared_grad_sum[leaf_index]);
    atomicAdd_system(leaf_hess_stat_buffer + leaf_index, shared_hess_sum[leaf_index]);
  }
}

__global__ void ReduceLeafStatKernel_GlobalMemory(
  const score_t* gradients,
  const score_t* hessians,
  const int num_leaves,
  const data_size_t num_data,
  const int* data_index_to_leaf_index,
  double* leaf_grad_stat_buffer,
  double* leaf_hess_stat_buffer) {
  const size_t offset = static_cast<size_t>(num_leaves) * (blockIdx.x + 1);
  double* grad_sum = leaf_grad_stat_buffer + offset;
  double* hess_sum = leaf_hess_stat_buffer + offset;
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  for (int leaf_index = static_cast<int>(threadIdx.x); leaf_index < num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    grad_sum[leaf_index] = 0.0f;
    hess_sum[leaf_index] = 0.0f;
  }
  __syncthreads();
  if (data_index < num_data) {
    const int leaf_index = data_index_to_leaf_index[data_index];
    atomicAdd_block(grad_sum + leaf_index, gradients[data_index]);
    atomicAdd_block(hess_sum + leaf_index, hessians[data_index]);
  }
  __syncthreads();
  for (int leaf_index = static_cast<int>(threadIdx.x); leaf_index < num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    atomicAdd_system(leaf_grad_stat_buffer + leaf_index, grad_sum[leaf_index]);
    atomicAdd_system(leaf_hess_stat_buffer + leaf_index, hess_sum[leaf_index]);
  }
}

template <bool USE_L1, bool USE_SMOOTHING>
__global__ void CalcRefitLeafOutputKernel(
  const int num_leaves,
  const double* leaf_grad_stat_buffer,
  const double* leaf_hess_stat_buffer,
  const data_size_t* num_data_in_leaf,
  const int* leaf_parent,
  const int* left_child,
  const int* right_child,
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const double max_delta_step,
  const double shrinkage_rate,
  const double refit_decay_rate,
  double* leaf_value) {
  const int leaf_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (leaf_index < num_leaves) {
    const double sum_gradients = leaf_grad_stat_buffer[leaf_index];
    const double sum_hessians = leaf_hess_stat_buffer[leaf_index];
    const data_size_t num_data = num_data_in_leaf[leaf_index];
    const double old_leaf_value = leaf_value[leaf_index];
    double new_leaf_value = 0.0f;
    if (!USE_SMOOTHING) {
      new_leaf_value = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(sum_gradients, sum_hessians, lambda_l1, lambda_l2, 0.0, max_delta_step, 0, 0.0);
    } else {
      const int parent = leaf_parent[leaf_index];
      if (parent >= 0) {
        const int sibliing = left_child[parent] == leaf_index ? right_child[parent] : left_child[parent];
        const double sum_gradients_of_parent = sum_gradients + leaf_grad_stat_buffer[sibliing];
        const double sum_hessians_of_parent = sum_hessians + leaf_hess_stat_buffer[sibliing];
        const data_size_t num_data_in_parent = num_data + num_data_in_leaf[sibliing];
        const double parent_output =
          CUDALeafSplits::CalculateSplittedLeafOutput<false, true>(
            sum_gradients_of_parent, sum_hessians_of_parent, lambda_l1, lambda_l2, 0.0, max_delta_step, 0, 0.0);
          new_leaf_value = CUDALeafSplits::CalculateSplittedLeafOutput<false, true>(
          sum_gradients, sum_hessians, lambda_l1, lambda_l2, path_smooth, max_delta_step, num_data_in_parent, parent_output);
      } else {
        new_leaf_value = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(sum_gradients, sum_hessians, lambda_l1, lambda_l2, 0.0, max_delta_step, 0, 0.0);
      }
    }
    if (isnan(new_leaf_value)) {
      new_leaf_value = 0.0f;
    } else {
      new_leaf_value *= shrinkage_rate;
    }
    leaf_value[leaf_index] = refit_decay_rate * old_leaf_value + (1.0f - refit_decay_rate) * new_leaf_value;
  }
}

void CUDASingleGPUTreeLearner::LaunchReduceLeafStatKernel(
  const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf,
  const int* leaf_parent, const int* left_child, const int* right_child, const int num_leaves,
  const data_size_t num_data, double* cuda_leaf_value, const double shrinkage_rate) const {
  int num_block = (num_data + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
  if (num_leaves <= 2048) {
    ReduceLeafStatKernel_SharedMemory<<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE, 2 * num_leaves * sizeof(double)>>>(
      gradients, hessians, num_leaves, num_data, cuda_data_partition_->cuda_data_index_to_leaf_index(),
      cuda_leaf_gradient_stat_buffer_.RawData(), cuda_leaf_hessian_stat_buffer_.RawData());
  } else {
    ReduceLeafStatKernel_GlobalMemory<<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(
      gradients, hessians, num_leaves, num_data, cuda_data_partition_->cuda_data_index_to_leaf_index(),
      cuda_leaf_gradient_stat_buffer_.RawData(), cuda_leaf_hessian_stat_buffer_.RawData());
  }
  const bool use_l1 = config_->lambda_l1 > 0.0f;
  const bool use_smoothing = config_->path_smooth > 0.0f;
  num_block = (num_leaves + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;

  #define CalcRefitLeafOutputKernel_ARGS \
    num_leaves, cuda_leaf_gradient_stat_buffer_.RawData(), cuda_leaf_hessian_stat_buffer_.RawData(), num_data_in_leaf, \
    leaf_parent, left_child, right_child, \
    config_->lambda_l1, config_->lambda_l2, config_->path_smooth, config_->max_delta_step, \
    shrinkage_rate, config_->refit_decay_rate, cuda_leaf_value

  if (!use_l1) {
    if (!use_smoothing) {
      CalcRefitLeafOutputKernel<false, false>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    } else {
      CalcRefitLeafOutputKernel<false, true>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    }
  } else {
    if (!use_smoothing) {
      CalcRefitLeafOutputKernel<true, false>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    } else {
      CalcRefitLeafOutputKernel<true, true>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    }
  }
  #undef CalcRefitLeafOutputKernel_ARGS
}

template <typename T, bool IS_INNER>
__global__ void CalcBitsetLenKernel(const CUDASplitInfo* best_split_info, size_t* out_len_buffer) {
  __shared__ size_t shared_mem_buffer[WARPSIZE];
  const T* vals = nullptr;
  if (IS_INNER) {
    vals = reinterpret_cast<const T*>(best_split_info->cat_threshold);
  } else {
    vals = reinterpret_cast<const T*>(best_split_info->cat_threshold_real);
  }
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  size_t len = 0;
  if (i < best_split_info->num_cat_threshold) {
    const T val = vals[i];
    len = (val / WARPSIZE) + 1;
  }
  const size_t block_max_len = ShuffleReduceMax<size_t>(len, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    out_len_buffer[blockIdx.x] = block_max_len;
  }
}

__global__ void ReduceBlockMaxLen(size_t* out_len_buffer, const int num_blocks) {
  __shared__ size_t shared_mem_buffer[WARPSIZE];
  size_t max_len = 0;
  for (int i = static_cast<int>(threadIdx.x); i < num_blocks; i += static_cast<int>(blockDim.x)) {
    max_len = max(out_len_buffer[i], max_len);
  }
  const size_t all_max_len = ShuffleReduceMax<size_t>(max_len, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    out_len_buffer[0] = max_len;
  }
}

template <typename T, bool IS_INNER>
__global__ void CUDAConstructBitsetKernel(const CUDASplitInfo* best_split_info, uint32_t* out, size_t cuda_bitset_len) {
  const T* vals = nullptr;
  if (IS_INNER) {
    vals = reinterpret_cast<const T*>(best_split_info->cat_threshold);
  } else {
    vals = reinterpret_cast<const T*>(best_split_info->cat_threshold_real);
  }
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (i < best_split_info->num_cat_threshold) {
    const T val = vals[i];
    // can use add instead of or here, because each bit will only be added once
    atomicAdd_system(out + (val / WARPSIZE), (0x1 << (val % WARPSIZE)));
  }
}

__global__ void SetRealThresholdKernel(
  const CUDASplitInfo* best_split_info,
  const int* categorical_bin_to_value,
  const int* categorical_bin_offsets) {
  const int num_cat_threshold = best_split_info->num_cat_threshold;
  const int* categorical_bin_to_value_ptr = categorical_bin_to_value + categorical_bin_offsets[best_split_info->inner_feature_index];
  int* cat_threshold_real = best_split_info->cat_threshold_real;
  const uint32_t* cat_threshold = best_split_info->cat_threshold;
  const int index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (index < num_cat_threshold) {
    cat_threshold_real[index] = categorical_bin_to_value_ptr[cat_threshold[index]];
  }
}

template <typename T, bool IS_INNER>
void CUDAConstructBitset(const CUDASplitInfo* best_split_info, const int num_cat_threshold, uint32_t* out, size_t bitset_len) {
  const int num_blocks = (num_cat_threshold + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
  // clear the bitset vector first
  SetCUDAMemory<uint32_t>(out, 0, bitset_len, __FILE__, __LINE__);
  CUDAConstructBitsetKernel<T, IS_INNER><<<num_blocks, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(best_split_info, out, bitset_len);
}

template <typename T, bool IS_INNER>
size_t CUDABitsetLen(const CUDASplitInfo* best_split_info, const int num_cat_threshold, size_t* out_len_buffer) {
  const int num_blocks = (num_cat_threshold + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
  CalcBitsetLenKernel<T, IS_INNER><<<num_blocks, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(best_split_info, out_len_buffer);
  ReduceBlockMaxLen<<<1, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(out_len_buffer, num_blocks);
  size_t host_max_len = 0;
  CopyFromCUDADeviceToHost<size_t>(&host_max_len, out_len_buffer, 1, __FILE__, __LINE__);
  return host_max_len;
}

// Level-batched categorical bitset construction: one block per categorical
// split builds the real-value and inner-bin bitsets into slab regions and
// writes both word lengths (the block body of BatchConstructCatBitsetsKernel and
// of the complete device tree's DeviceLevelCatBitsetsKernel).
__device__ __forceinline__ void ConstructCatBitsetsBlock(
  const CUDASplitInfo* info,
  const int inner_feature,
  const int c,
  const int* categorical_bin_to_value,
  const int* categorical_bin_offsets,
  uint32_t* bitset_slab, const int bitset_stride,
  uint32_t* bitset_inner_slab, const int bitset_inner_stride,
  uint64_t* lens_out) {
  const int n = info->num_cat_threshold;
  const uint32_t* cat_threshold = info->cat_threshold;
  int* cat_threshold_real = info->cat_threshold_real;
  const int* bin_to_value = categorical_bin_to_value + categorical_bin_offsets[inner_feature];
  uint32_t* bitset = bitset_slab + static_cast<size_t>(c) * bitset_stride;
  uint32_t* bitset_inner = bitset_inner_slab + static_cast<size_t>(c) * bitset_inner_stride;
  __shared__ uint32_t max_inner_shared;
  __shared__ uint32_t max_real_shared;
  if (threadIdx.x == 0) {
    max_inner_shared = 0;
    max_real_shared = 0;
  }
  // zero the slab regions
  for (int w = static_cast<int>(threadIdx.x); w < bitset_stride; w += static_cast<int>(blockDim.x)) {
    bitset[w] = 0;
  }
  for (int w = static_cast<int>(threadIdx.x); w < bitset_inner_stride; w += static_cast<int>(blockDim.x)) {
    bitset_inner[w] = 0;
  }
  __syncthreads();
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    const uint32_t inner = cat_threshold[i];
    const uint32_t real = static_cast<uint32_t>(bin_to_value[inner]);
    cat_threshold_real[i] = static_cast<int>(real);
    atomicOr(bitset_inner + (inner >> 5), 1u << (inner & 31));
    atomicOr(bitset + (real >> 5), 1u << (real & 31));
    atomicMax(&max_inner_shared, inner);
    atomicMax(&max_real_shared, real);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    lens_out[2 * c] = static_cast<uint64_t>(max_real_shared / 32 + 1);
    lens_out[2 * c + 1] = static_cast<uint64_t>(max_inner_shared / 32 + 1);
  }
}

// The host level loop's categorical splits of one level (ApplyLevelBatched): replaces the per-split
// SetRealThreshold + 2x(CalcBitsetLen + ReduceBlockMaxLen + sync D2H) + 2x(memset + ConstructBitset) sequence --
// ~six launches and two blocking readbacks PER SPLIT -- with one launch per level.
__global__ void BatchConstructCatBitsetsKernel(
  const CUDASplitInfo* const* infos,
  const int* inner_features,
  const int* categorical_bin_to_value,
  const int* categorical_bin_offsets,
  uint32_t* bitset_slab, const int bitset_stride,
  uint32_t* bitset_inner_slab, const int bitset_inner_stride,
  uint64_t* lens_out) {
  const int c = static_cast<int>(blockIdx.x);
  ConstructCatBitsetsBlock(infos[c], inner_features[c], c, categorical_bin_to_value, categorical_bin_offsets,
                           bitset_slab, bitset_stride, bitset_inner_slab, bitset_inner_stride, lens_out);
}

// The complete device tree's categorical splits of one level: the select listed them (CUDADeviceLevelCatConfig);
// the grid is the level's worst case, blocks past the level's categorical splits exit. After the bitsets, the split's
// apply descriptor gets its default directions from the inner bitset, exactly as HybridBuildCatBitsetArenaKernel
// patches SplitLevelBatched's descriptors: the most frequent bin's membership (bin 0 never), read over the
// descriptor's bin-range cap (the arena region's length there; the slab's words past it are not read)
__global__ void DeviceLevelCatBitsetsKernel(const CUDADeviceLevelConfig cfg) {
  const int c = static_cast<int>(blockIdx.x);
  if (c >= cfg.state->live_cat_splits) {
    return;
  }
  const CUDADeviceLevelCatConfig& cat = cfg.cat;
  const int inner_feature = cat.inner_features[c];
  ConstructCatBitsetsBlock(cat.infos[c], inner_feature, c, cat.bin_to_value, cat.bin_offsets, cat.bitset_slab,
                           cat.bitset_stride, cat.bitset_inner_slab, cat.bitset_inner_stride, cat.lens);
  __syncthreads();
  if (threadIdx.x == 0) {
    CUDAHybridApplyDescriptor* d = cfg.apply_descs + cat.split_index[c];
    const uint32_t* bits = d->cat_bitset;
    const uint32_t mfb = cfg.feature_meta[inner_feature].most_freq_bin;
    const uint32_t word = mfb / 32;
    const uint8_t to_left = (mfb > 0 && word < static_cast<uint32_t>(d->cat_bitset_len) &&
                             ((bits[word] >> (mfb % 32)) & 1u) != 0) ? 1 : 0;
    d->split_default_to_left = to_left;
    d->split_missing_default_to_left = to_left;
    d->default_leaf_index = to_left ? d->left_leaf_index : d->right_leaf_index;
    d->missing_default_leaf_index = d->default_leaf_index;
  }
}

void LaunchDeviceLevelCatBitsets(const CUDADeviceLevelConfig& config, const int max_splits, cudaStream_t stream) {
  DeviceLevelCatBitsetsKernel<<<max_splits, 128, 0, stream>>>(config);
}

void CUDASingleGPUTreeLearner::LaunchBatchConstructCatBitsetsKernel(const int num_cat_splits) {
  BatchConstructCatBitsetsKernel<<<num_cat_splits, 128>>>(
    cuda_batch_cat_infos_.RawDataReadOnly(),
    cuda_batch_cat_feats_.RawDataReadOnly(),
    cuda_categorical_bin_to_value_,
    cuda_categorical_bin_offsets_,
    cuda_batch_cat_bitset_.RawData(), static_cast<int>(batch_cat_bitset_stride_),
    cuda_batch_cat_bitset_inner_.RawData(), static_cast<int>(batch_cat_bitset_inner_stride_),
    cuda_batch_cat_lens_.RawData());
}

void CUDASingleGPUTreeLearner::LaunchConstructBitsetForCategoricalSplitKernel(
  const CUDASplitInfo* best_split_info) {
  const int num_blocks = (num_cat_threshold_ + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
  SetRealThresholdKernel<<<num_blocks, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>
    (best_split_info, cuda_categorical_bin_to_value_, cuda_categorical_bin_offsets_);
  cuda_bitset_inner_len_ = CUDABitsetLen<uint32_t, true>(best_split_info, num_cat_threshold_, cuda_block_bitset_len_buffer_);
  CUDAConstructBitset<uint32_t, true>(best_split_info, num_cat_threshold_, cuda_bitset_inner_, cuda_bitset_inner_len_);
  cuda_bitset_len_ = CUDABitsetLen<int, false>(best_split_info, num_cat_threshold_, cuda_block_bitset_len_buffer_);
  CUDAConstructBitset<int, false>(best_split_info, num_cat_threshold_, cuda_bitset_, cuda_bitset_len_);
}

void CUDASingleGPUTreeLearner::LaunchCalcLeafValuesGivenGradStat(
  CUDATree* cuda_tree, const data_size_t* num_data_in_leaf) {
  #define CalcRefitLeafOutputKernel_ARGS \
    cuda_tree->num_leaves(), cuda_leaf_gradient_stat_buffer_.RawData(), cuda_leaf_hessian_stat_buffer_.RawData(), num_data_in_leaf, \
    cuda_tree->cuda_leaf_parent(), cuda_tree->cuda_left_child(), cuda_tree->cuda_right_child(), \
    config_->lambda_l1, config_->lambda_l2, config_->path_smooth, config_->max_delta_step, \
    1.0f, config_->refit_decay_rate, cuda_tree->cuda_leaf_value_ref()
  const bool use_l1 = config_->lambda_l1 > 0.0f;
  const bool use_smoothing = config_->path_smooth > 0.0f;
  const int num_block = (cuda_tree->num_leaves() + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
  if (!use_l1) {
    if (!use_smoothing) {
      CalcRefitLeafOutputKernel<false, false>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    } else {
      CalcRefitLeafOutputKernel<false, true>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    }
  } else {
    if (!use_smoothing) {
      CalcRefitLeafOutputKernel<true, false>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    } else {
      CalcRefitLeafOutputKernel<true, true>
        <<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(CalcRefitLeafOutputKernel_ARGS);
    }
  }

  #undef CalcRefitLeafOutputKernel_ARGS
}

// ---- linear tree ----
// Per leaf, the h-weighted normal equations XtHX (upper triangle, row-major) and Xtg of its rows. Mirrors the CPU
// inner loop in LinearTreeLearner::CalculateLinear (float row, double accumulation, g applied before the hessian
// scaling). NaN rows are skipped (and not counted), matching the CPU HAS_NAN branch; for NaN-free data every leaf
// row is counted == leaf_count.
//
// Fixed-order reduction, so the sums (and the fitted coefficients) repeat bit for bit run to run: the rows of a leaf
// are its window of the partition's index list (cuda_leaf_data_start / cuda_leaf_num_data, contiguous per leaf after
// the tree is grown; under bagging the list holds only the bagged rows), cut into chunks of LINEAR_GRAM_CHUNK
// positions (a multiple of it for a leaf of more than LINEAR_GRAM_WIDE entries). Pass 1 (one block per chunk):
// thread (g, e) sums entry e over the chunk positions p with p % G == g in increasing p, serially in a register;
// the G group sums are then added in g order. Pass 2 (one thread per leaf entry) adds the chunk sums in chunk order.
// G depends only on the leaf's entry count; the batch size of the row staging changes nothing. A per-row double atomicAdd (the earlier kernel) summed in scheduler order instead.
//
// Entry e of a leaf with nf features (m = nf + 1 columns, the last the constant 1): e < m is Xtg[e], e >= m is XtHX
// upper-triangle element e - m. Each term is rounded as the CPU rounds it: x1 * g and (x1 * h) * x2 in double (the
// first products of two floats are exact), no FMA contraction.
#define LINEAR_GRAM_CHUNK 2048
#define LINEAR_GRAM_BLOCK 256
#define LINEAR_GRAM_KMAX 8
#define LINEAR_GRAM_WIDE 1024
// per-leaf meta (int, stride LINEAR_GRAM_META): nf, feature offset, entries, partial offset, window start, window
// rows, first chunk, chunks, gram offset, chunk rows
#define LINEAR_GRAM_META 10

__device__ __forceinline__ void LinearGramDecodeEntry(const int e, const int m, int* f1, int* f2) {
  if (e < m) {
    *f1 = -1;
    *f2 = e;
    return;
  }
  int j = e - m;
  int r = 0;
  while (j >= m - r) {
    j -= m - r;
    ++r;
  }
  *f1 = r;
  *f2 = r + j;
}

__global__ void __launch_bounds__(LINEAR_GRAM_BLOCK) CalcLinearGramChunkKernel(
    const int* __restrict__ leaf_meta,
    const int* __restrict__ chunk_table,
    const data_size_t* __restrict__ data_indices,
    const float* __restrict__ raw_data,
    const data_size_t num_data_stride,
    const score_t* __restrict__ gradients,
    const score_t* __restrict__ hessians,
    const int* __restrict__ leaf_feat_col,
    const int stage_bytes,
    double* __restrict__ partial,
    int* __restrict__ chunk_count) {
  extern __shared__ double linear_gram_shared[];
  double* group_sum = linear_gram_shared;  // [LINEAR_GRAM_BLOCK]
  int* count_buffer = reinterpret_cast<int*>(group_sum + LINEAR_GRAM_BLOCK);  // [32]
  uint8_t* stage = reinterpret_cast<uint8_t*>(count_buffer + 32);
  const int leaf = chunk_table[2 * blockIdx.x];
  const int chunk = chunk_table[2 * blockIdx.x + 1];
  const int* meta = leaf_meta + LINEAR_GRAM_META * leaf;
  const int nf = meta[0];
  const int foff = meta[1];
  const int num_entries = meta[2];
  const int64_t poff = static_cast<int64_t>(meta[3]);
  const data_size_t leaf_start = meta[4];
  const data_size_t leaf_num = meta[5];
  const int m = nf + 1;
  const int row_stride = m | 1;  // odd: the staging stores and the entry reads hit distinct banks
  const data_size_t chunk_rows = meta[9];
  const data_size_t chunk_begin = static_cast<data_size_t>(chunk) * chunk_rows;
  const int rows = static_cast<int>(min(chunk_rows, leaf_num - chunk_begin));
  // staging batch: row ids, g, h, the NaN flag and the row's m columns
  int batch = stage_bytes / static_cast<int>(sizeof(float) * row_stride + 4 * sizeof(int));
  batch = min(batch, LINEAR_GRAM_BLOCK);
  int* row_id = reinterpret_cast<int*>(stage);
  float* g_stage = reinterpret_cast<float*>(row_id + batch);
  float* h_stage = g_stage + batch;
  int* bad = reinterpret_cast<int*>(h_stage + batch);
  float* x_stage = reinterpret_cast<float*>(bad + batch);
  const int tid = static_cast<int>(threadIdx.x);
  int num_groups;
  int slots;
  if (2 * num_entries <= LINEAR_GRAM_BLOCK) {
    num_groups = LINEAR_GRAM_BLOCK / num_entries;
    slots = num_entries;
  } else {
    num_groups = 1;
    slots = LINEAR_GRAM_BLOCK;
  }
  const int group = tid / slots;
  const int slot = tid % slots;
  const bool active = group < num_groups;
  int valid_rows = 0;
  const data_size_t* window = data_indices + leaf_start + chunk_begin;
  for (int e_base = 0; e_base < num_entries; e_base += slots * LINEAR_GRAM_KMAX) {
    double acc[LINEAR_GRAM_KMAX];
    int f1[LINEAR_GRAM_KMAX];
    int f2[LINEAR_GRAM_KMAX];
    #pragma unroll
    for (int k = 0; k < LINEAR_GRAM_KMAX; ++k) {
      acc[k] = 0.0;
      const int e = e_base + slot + k * slots;
      if (active && e < num_entries) {
        LinearGramDecodeEntry(e, m, &f1[k], &f2[k]);
      } else {
        f1[k] = -2;  // unused
        f2[k] = 0;
      }
    }
    for (int b = 0; b < rows; b += batch) {
      const int nb = min(batch, rows - b);
      __syncthreads();  // the previous batch's readers are done
      for (int r = tid; r < nb; r += LINEAR_GRAM_BLOCK) {
        const data_size_t i = window[b + r];
        row_id[r] = i;
        g_stage[r] = static_cast<float>(gradients[i]);
        h_stage[r] = static_cast<float>(hessians[i]);
        x_stage[r * row_stride + nf] = 1.0f;
      }
      __syncthreads();
      for (int idx = tid; idx < nb * nf; idx += LINEAR_GRAM_BLOCK) {
        const int f = idx / nb;
        const int r = idx - f * nb;
        const int col = leaf_feat_col[foff + f];
        x_stage[r * row_stride + f] = raw_data[static_cast<size_t>(col) * num_data_stride + row_id[r]];
      }
      __syncthreads();
      for (int r = tid; r < nb; r += LINEAR_GRAM_BLOCK) {
        int is_bad = 0;
        for (int f = 0; f < nf; ++f) {
          is_bad |= isnan(x_stage[r * row_stride + f]) ? 1 : 0;
        }
        bad[r] = is_bad;
        if (e_base == 0) {
          valid_rows += 1 - is_bad;
        }
      }
      __syncthreads();
      if (active) {
        // first batch row of this group: chunk position b + r with (b + r) % num_groups == group
        const int r0 = (group - b % num_groups + num_groups) % num_groups;
        for (int r = r0; r < nb; r += num_groups) {
          if (bad[r]) {
            continue;
          }
          const float* x = x_stage + r * row_stride;
          const double g = static_cast<double>(g_stage[r]);
          const double h = static_cast<double>(h_stage[r]);
          #pragma unroll
          for (int k = 0; k < LINEAR_GRAM_KMAX; ++k) {
            if (f1[k] == -1) {
              acc[k] = __dadd_rn(acc[k], __dmul_rn(static_cast<double>(x[f2[k]]), g));
            } else if (f1[k] >= 0) {
              acc[k] = __dadd_rn(acc[k], __dmul_rn(__dmul_rn(static_cast<double>(x[f1[k]]), h),
                                                   static_cast<double>(x[f2[k]])));
            }
          }
        }
      }
    }
    double* out = partial + poff + static_cast<int64_t>(chunk) * num_entries;
    if (num_groups > 1) {
      // one entry per thread (2 * num_entries <= block): add the group sums in group order
      __syncthreads();
      if (active) {
        group_sum[group * slots + slot] = acc[0];
      }
      __syncthreads();
      if (tid < num_entries) {
        double s = 0.0;
        for (int gr = 0; gr < num_groups; ++gr) {
          s = __dadd_rn(s, group_sum[gr * slots + tid]);
        }
        out[tid] = s;
      }
    } else {
      #pragma unroll
      for (int k = 0; k < LINEAR_GRAM_KMAX; ++k) {
        const int e = e_base + slot + k * slots;
        if (e < num_entries) {
          out[e] = acc[k];
        }
      }
    }
  }
  // integer count: any order gives the same sum
  valid_rows = ShuffleReduceSum<int>(valid_rows, count_buffer, LINEAR_GRAM_BLOCK);
  if (tid == 0) {
    chunk_count[blockIdx.x] = valid_rows;
  }
}

// Pass 2: entry e of leaf blockIdx.y = the sum of its chunk sums in chunk order; the leaf's row count likewise.
__global__ void CalcLinearGramFinalKernel(
    const int* __restrict__ leaf_meta,
    const double* __restrict__ partial,
    const int* __restrict__ chunk_count,
    double* __restrict__ gram,
    int* __restrict__ leaf_nonzero) {
  const int leaf = static_cast<int>(blockIdx.y);
  const int* meta = leaf_meta + LINEAR_GRAM_META * leaf;
  const int num_entries = meta[2];
  const int64_t poff = static_cast<int64_t>(meta[3]);
  const int first_chunk = meta[6];
  const int num_chunks = meta[7];
  const int e = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  if (e == 0) {
    int n = 0;
    for (int c = 0; c < num_chunks; ++c) {
      n += chunk_count[first_chunk + c];
    }
    leaf_nonzero[leaf] = n;
  }
  if (e >= num_entries) {
    return;
  }
  double s = 0.0;
  for (int c = 0; c < num_chunks; ++c) {
    s = __dadd_rn(s, partial[poff + static_cast<int64_t>(c) * num_entries + e]);
  }
  gram[meta[8] + e] = s;
}

// score += leaf_const + sum(coeff * raw_feat); if any used feature is NaN, fall back
// to the constant leaf output. Mirrors LinearTreeLearner::AddPredictionToScoreInner.
template <bool USE_BAGGING>
__global__ void LinearAddScoreKernel(
    const data_size_t num_data,
    const data_size_t* __restrict__ data_indices,
    const int* __restrict__ data_index_to_leaf_index,
    const float* __restrict__ raw_data,
    const data_size_t num_data_stride,
    const double* __restrict__ leaf_const,
    const double* __restrict__ leaf_coeff,
    const int* __restrict__ leaf_coeff_offset,
    const int* __restrict__ leaf_coeff_col,
    const int* __restrict__ leaf_num_coeff,
    const double* __restrict__ leaf_output,
    double* __restrict__ score) {
  const data_size_t local_data_index = static_cast<data_size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_data_index >= num_data) {
    return;
  }
  // Under bagging num_data is the BAGGED count and the rows to score are the
  // partition's index list -- exactly what the non-linear
  // CUDADataPartition::AddPredictionToScoreKernel does. Walking 0..num_data_
  // instead scored the out-of-bag rows here too, and GBDT::UpdateScore scored
  // them a second time through its own out-of-bag pass: their score carried
  // twice the tree's contribution, so the training metric stopped matching
  // what the model predicts.
  const data_size_t i = USE_BAGGING ? data_indices[local_data_index] : local_data_index;
  const int leaf = data_index_to_leaf_index[i];
  if (leaf < 0) {
    return;
  }
  double out = leaf_const[leaf];
  const int nc = leaf_num_coeff[leaf];
  const int off = leaf_coeff_offset[leaf];
  bool nan_found = false;
  for (int k = 0; k < nc; ++k) {
    const int col = leaf_coeff_col[off + k];
    const float val = raw_data[static_cast<size_t>(col) * num_data_stride + i];
    if (isnan(val)) {
      nan_found = true;
      break;
    }
    out += leaf_coeff[off + k] * static_cast<double>(val);
  }
  score[i] += nan_found ? leaf_output[leaf] : out;
}

void CUDASingleGPUTreeLearner::LaunchCalcLinearGramKernel(
    const std::vector<int>& leaf_num_feat, const std::vector<int>& leaf_feat_offset,
    const std::vector<int>& leaf_feat_col, const score_t* gradients, const score_t* hessians,
    std::vector<double>* gram, std::vector<int>* gram_offset, std::vector<int>* leaf_nonzero) {
  const int num_leaves = static_cast<int>(leaf_num_feat.size());
  gram_offset->assign(num_leaves + 1, 0);
  leaf_nonzero->assign(num_leaves, 0);
  // the leaf windows of the partition's index list. Both map-only final-level gates exclude linear trees (the host
  // level loop's map_only and the device tree's prefetch_tree_end), so nothing is pending here today; the call
  // only keeps the windows valid should a later change lift that exclusion (a no-op otherwise)
  cuda_data_partition_->EnsureLevelPartitioned();
  std::vector<data_size_t> leaf_start(num_leaves), leaf_num(num_leaves);
  CopyFromCUDADeviceToHost<data_size_t>(leaf_start.data(), cuda_data_partition_->cuda_leaf_data_start(),
                                        num_leaves, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<data_size_t>(leaf_num.data(), cuda_data_partition_->cuda_leaf_num_data(),
                                        num_leaves, __FILE__, __LINE__);
  const data_size_t root_rows = cuda_data_partition_->root_num_data();
  if (FalcataVerifyEnabled()) {
    // FALCATA_VERIFY=1: the windows hold every partition row once, each in the leaf the row -> leaf map names
    std::vector<data_size_t> indices(root_rows > 0 ? root_rows : 1);
    std::vector<int> map(num_data_ > 0 ? num_data_ : 1);
    if (root_rows > 0) {
      CopyFromCUDADeviceToHost<data_size_t>(indices.data(), cuda_data_partition_->cuda_data_indices(),
                                            root_rows, __FILE__, __LINE__);
    }
    CopyFromCUDADeviceToHost<int>(map.data(), cuda_data_partition_->cuda_data_index_to_leaf_index(), num_data_,
                                  __FILE__, __LINE__);
    std::vector<uint8_t> seen(num_data_ > 0 ? num_data_ : 1, 0);
    int64_t total = 0;
    for (int leaf = 0; leaf < num_leaves; ++leaf) {
      CHECK_GE(leaf_num[leaf], 0);
      CHECK_GE(leaf_start[leaf], 0);
      CHECK_LE(static_cast<int64_t>(leaf_start[leaf]) + leaf_num[leaf], static_cast<int64_t>(root_rows));
      for (data_size_t j = 0; j < leaf_num[leaf]; ++j) {
        const data_size_t row = indices[leaf_start[leaf] + j];
        CHECK_GE(row, 0);
        CHECK_LT(row, num_data_);
        CHECK_EQ(map[row], leaf);
        CHECK_EQ(seen[row], 0);
        seen[row] = 1;
      }
      total += leaf_num[leaf];
    }
    CHECK_EQ(total, static_cast<int64_t>(root_rows));
  }
  // per-leaf meta + chunk table (see CalcLinearGramChunkKernel)
  std::vector<int> meta(static_cast<size_t>(LINEAR_GRAM_META) * num_leaves, 0);
  std::vector<int> chunks;
  int64_t partial_size = 0;
  int max_entries = 0;
  int max_row_stride = 1;
  for (int leaf = 0; leaf < num_leaves; ++leaf) {
    const int nf = leaf_num_feat[leaf];
    const int m = nf + 1;
    const int entries = m + m * (m + 1) / 2;
    // wide leaves take proportionally longer chunks: the partial buffer stays about 4 bytes per row
    const int chunk_rows = LINEAR_GRAM_CHUNK * std::max(1, (entries + LINEAR_GRAM_WIDE - 1) / LINEAR_GRAM_WIDE);
    const int num_chunks = static_cast<int>((static_cast<int64_t>(leaf_num[leaf]) + chunk_rows - 1) / chunk_rows);
    int* lm = meta.data() + static_cast<size_t>(LINEAR_GRAM_META) * leaf;
    lm[0] = nf;
    lm[1] = leaf_feat_offset[leaf];
    lm[2] = entries;
    lm[3] = static_cast<int>(partial_size);
    lm[4] = leaf_start[leaf];
    lm[5] = leaf_num[leaf];
    lm[6] = static_cast<int>(chunks.size() / 2);
    lm[7] = num_chunks;
    lm[8] = (*gram_offset)[leaf];
    lm[9] = chunk_rows;
    for (int c = 0; c < num_chunks; ++c) {
      chunks.push_back(leaf);
      chunks.push_back(c);
    }
    partial_size += static_cast<int64_t>(num_chunks) * entries;
    CHECK_LT(partial_size, static_cast<int64_t>(INT32_MAX));
    (*gram_offset)[leaf + 1] = (*gram_offset)[leaf] + entries;
    max_entries = std::max(max_entries, entries);
    max_row_stride = std::max(max_row_stride, m | 1);
  }
  const int total_gram = (*gram_offset)[num_leaves];
  const int num_chunk_blocks = static_cast<int>(chunks.size() / 2);
  // grow-only device scratch, sized before any launch of the tree
  auto reserve = [](auto* vec, size_t n) {
    if (vec->Size() < n) {
      vec->ResizeDiscard(n);
    }
  };
  reserve(&cuda_linear_meta_, meta.size());
  reserve(&cuda_linear_feat_col_, std::max<size_t>(leaf_feat_col.size(), 1));
  reserve(&cuda_linear_chunks_, std::max<size_t>(chunks.size(), 2));
  reserve(&cuda_linear_partial_, std::max<size_t>(static_cast<size_t>(partial_size), 1));
  reserve(&cuda_linear_chunk_count_, std::max<size_t>(static_cast<size_t>(num_chunk_blocks), 1));
  reserve(&cuda_linear_gram_, std::max<size_t>(static_cast<size_t>(total_gram), 1));
  reserve(&cuda_linear_nonzero_, std::max<size_t>(static_cast<size_t>(num_leaves), 1));
  CopyFromHostToCUDADevice<int>(cuda_linear_meta_.RawData(), meta.data(), meta.size(), __FILE__, __LINE__);
  if (!leaf_feat_col.empty()) {
    CopyFromHostToCUDADevice<int>(cuda_linear_feat_col_.RawData(), leaf_feat_col.data(), leaf_feat_col.size(),
                                  __FILE__, __LINE__);
  }
  if (num_chunk_blocks > 0) {
    CopyFromHostToCUDADevice<int>(cuda_linear_chunks_.RawData(), chunks.data(), chunks.size(), __FILE__, __LINE__);
    // staging sized for a full batch of the widest leaf, capped so the block fits the default 48 KB
    const int fixed_bytes = LINEAR_GRAM_BLOCK * static_cast<int>(sizeof(double)) + 32 * static_cast<int>(sizeof(int));
    const int stage_bytes = std::min(48 * 1024 - fixed_bytes,
      LINEAR_GRAM_BLOCK * (static_cast<int>(sizeof(float)) * max_row_stride + 4 * static_cast<int>(sizeof(int))));
    CHECK_GE(stage_bytes, static_cast<int>(sizeof(float)) * max_row_stride + 4 * static_cast<int>(sizeof(int)));
    CalcLinearGramChunkKernel<<<num_chunk_blocks, LINEAR_GRAM_BLOCK, fixed_bytes + stage_bytes>>>(
      cuda_linear_meta_.RawData(), cuda_linear_chunks_.RawData(), cuda_data_partition_->cuda_data_indices(),
      cuda_raw_data_.RawData(), num_data_, gradients, hessians, cuda_linear_feat_col_.RawData(), stage_bytes,
      cuda_linear_partial_.RawData(), cuda_linear_chunk_count_.RawData());
  }
  if (num_leaves > 0) {
    const dim3 grid((std::max(max_entries, 1) + LINEAR_GRAM_BLOCK - 1) / LINEAR_GRAM_BLOCK, num_leaves);
    CalcLinearGramFinalKernel<<<grid, LINEAR_GRAM_BLOCK>>>(
      cuda_linear_meta_.RawData(), cuda_linear_partial_.RawData(), cuda_linear_chunk_count_.RawData(),
      cuda_linear_gram_.RawData(), cuda_linear_nonzero_.RawData());
  }
  gram->resize(std::max(total_gram, 1));
  CopyFromCUDADeviceToHost<double>(gram->data(), cuda_linear_gram_.RawData(), total_gram, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(leaf_nonzero->data(), cuda_linear_nonzero_.RawData(), num_leaves,
                                __FILE__, __LINE__);
}

void CUDASingleGPUTreeLearner::LaunchLinearAddScoreKernel(
    const int* cuda_data_index_to_leaf_index, const double* cuda_leaf_const,
    const double* cuda_leaf_coeff, const int* cuda_leaf_coeff_offset, const int* cuda_leaf_coeff_col,
    const int* leaf_num_coeff, const double* cuda_leaf_output, double* score) const {
  const data_size_t num_score_data = cuda_data_partition_->root_num_data();
  if (num_score_data <= 0) {
    return;
  }
  const int block = 256;
  const int grid = (num_score_data + block - 1) / block;
  if (cuda_data_partition_->use_bagging()) {
    LinearAddScoreKernel<true><<<grid, block>>>(
      num_score_data, cuda_data_partition_->cuda_data_indices(),
      cuda_data_index_to_leaf_index, cuda_raw_data_.RawData(), num_data_,
      cuda_leaf_const, cuda_leaf_coeff, cuda_leaf_coeff_offset, cuda_leaf_coeff_col,
      leaf_num_coeff, cuda_leaf_output, score);
  } else {
    LinearAddScoreKernel<false><<<grid, block>>>(
      num_score_data, nullptr,
      cuda_data_index_to_leaf_index, cuda_raw_data_.RawData(), num_data_,
      cuda_leaf_const, cuda_leaf_coeff, cuda_leaf_coeff_offset, cuda_leaf_coeff_col,
      leaf_num_coeff, cuda_leaf_output, score);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

}  // namespace Falcata

#endif  // USE_CUDA

namespace Falcata {

// ---- NCCL compacted histogram reduce -------------------------------------
// With feature_fraction < 1 the per-leaf histogram still spans EVERY feature's
// bins, but only the sampled features' bins are ever read. All-reducing the
// whole array therefore ships (1 - feature_fraction) of the message for
// nothing -- 90% of it at numerai's feature_fraction=0.1. These two kernels
// pack the readable bins into a contiguous staging buffer so one collective
// moves only the live data, and unpack afterwards.

__global__ void NCCLGatherBinsKernel(const hist_t* __restrict__ hist,
                                     const int* __restrict__ idx,
                                     double* __restrict__ out,
                                     const int num_bins) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_bins) return;
  const int bin = idx[i];
  out[2 * i] = hist[2 * bin];
  out[2 * i + 1] = hist[2 * bin + 1];
}

__global__ void NCCLScatterBinsKernel(const double* __restrict__ in,
                                      const int* __restrict__ idx,
                                      hist_t* __restrict__ hist,
                                      const int num_bins) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_bins) return;
  const int bin = idx[i];
  hist[2 * bin] = in[2 * i];
  hist[2 * bin + 1] = in[2 * i + 1];
}

void CUDASingleGPUTreeLearner::LaunchNCCLGatherBins(const hist_t* hist, double* out,
                                                    int num_bins, cudaStream_t stream) {
  const int block = 256;
  const int grid = (num_bins + block - 1) / block;
  NCCLGatherBinsKernel<<<grid, block, 0, stream>>>(
      hist, cuda_nccl_reduce_bin_idx_.RawData(), out, num_bins);
}

void CUDASingleGPUTreeLearner::LaunchNCCLScatterBins(const double* in, hist_t* hist,
                                                     int num_bins, cudaStream_t stream) {
  const int block = 256;
  const int grid = (num_bins + block - 1) / block;
  NCCLScatterBinsKernel<<<grid, block, 0, stream>>>(
      in, cuda_nccl_reduce_bin_idx_.RawData(), hist, num_bins);
}

}  // namespace Falcata

namespace Falcata {

// ---- level-batched NCCL histogram reduce ---------------------------------
// The classic path all-reduces ONE leaf histogram per split, so a 255-leaf
// tree pays ~254 collectives. Measured on 2x RTX 3090 that is ~190 us of
// latency each and dominates everything else -- shrinking the messages (the
// colsample-aware compaction) removed only ~20 ms of ~68 ms. Reducing a whole
// LEVEL in one collective attacks the term that actually binds.
//
// Every pair's smaller-leaf histogram is gathered into one contiguous buffer
// (optionally through the per-tree used-bin index), reduced once, scattered
// back. Grid is (bins, pairs). Histogram locations come from the pair
// descriptors' smaller_struct->hist_in_leaf -- written by the batched apply,
// and the same pointer the construct kernel writes through -- because the
// level flow does not maintain the host leaf_to_hist_index_map_. A pair
// invalid for construct (construct_valid == 0) has a stale hist pointer, so
// it is skipped on both sides; its buffer region still participates in the
// collective (ranks must stay symmetric), so the gather zero-fills it.

__global__ void NCCLGatherLevelKernel(const CUDAHybridPairDescriptor* __restrict__ descs,
                                      const int* __restrict__ idx,
                                      double* __restrict__ out,
                                      const int num_bins) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_bins) return;
  const int pair = blockIdx.y;
  double* dst = out + static_cast<size_t>(pair) * num_bins * 2;
  if (!descs[pair].construct_valid) {
    dst[2 * i] = 0.0;
    dst[2 * i + 1] = 0.0;
    return;
  }
  const hist_t* hist = descs[pair].smaller_struct->hist_in_leaf;
  const int bin = (idx == nullptr) ? i : idx[i];
  dst[2 * i] = hist[2 * bin];
  dst[2 * i + 1] = hist[2 * bin + 1];
}

__global__ void NCCLScatterLevelKernel(const double* __restrict__ in,
                                       const int* __restrict__ idx,
                                       const CUDAHybridPairDescriptor* __restrict__ descs,
                                       const int num_bins) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_bins) return;
  const int pair = blockIdx.y;
  if (!descs[pair].construct_valid) return;
  hist_t* hist = descs[pair].smaller_struct->hist_in_leaf;
  const int bin = (idx == nullptr) ? i : idx[i];
  const double* src = in + static_cast<size_t>(pair) * num_bins * 2;
  hist[2 * bin] = src[2 * i];
  hist[2 * bin + 1] = src[2 * i + 1];
}

void CUDASingleGPUTreeLearner::LaunchNCCLGatherLevel(const CUDAHybridPairDescriptor* descs,
                                                     const int* idx,
                                                     double* out, int num_bins, int num_pairs,
                                                     cudaStream_t stream) {
  const int block = 256;
  dim3 grid((num_bins + block - 1) / block, num_pairs);
  NCCLGatherLevelKernel<<<grid, block, 0, stream>>>(descs, idx, out, num_bins);
}

void CUDASingleGPUTreeLearner::LaunchNCCLScatterLevel(const double* in, const int* idx,
                                                      const CUDAHybridPairDescriptor* descs,
                                                      int num_bins,
                                                      int num_pairs, cudaStream_t stream) {
  const int block = 256;
  dim3 grid((num_bins + block - 1) / block, num_pairs);
  NCCLScatterLevelKernel<<<grid, block, 0, stream>>>(in, idx, descs, num_bins);
}

// ---- quantized level reduce ------------------------------------------------
// Integer histograms: 16-bit mode packs (grad16,hess16) into one int32 per
// bin, 32-bit mode packs (grad32,hess32) into one int64 per bin; the per-pair
// width is desc->smaller_num_bits (the GLOBAL table under NCCL, so every rank
// picks the same lane format AND the summed histogram is guaranteed to fit
// the width). The gather widens 16-bit fields to the int64 (grad32,hess32)
// lane so a single ncclInt64 sum serves every pair regardless of width --
// field carries are impossible because the final sums fit the ORIGINAL
// width (16/32 bits) with the widened lane giving 16 spare bits per field.
// Payload: one int64 per bin, HALF the fp64-pair layout of the non-quantized
// reduce, and the sums are exact integers (bit-stable across topologies).

__global__ void NCCLGatherLevelQuantKernel(const CUDAHybridPairDescriptor* __restrict__ descs,
                                           const int* __restrict__ idx,
                                           int64_t* __restrict__ out,
                                           const int num_bins) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_bins) return;
  const int pair = blockIdx.y;
  int64_t* dst = out + static_cast<size_t>(pair) * num_bins;
  if (!descs[pair].construct_valid) {
    dst[i] = 0;
    return;
  }
  const int bin = (idx == nullptr) ? i : idx[i];
  const hist_t* hist = descs[pair].smaller_struct->hist_in_leaf;
  if (descs[pair].smaller_num_bits <= 16) {
    const int32_t v = reinterpret_cast<const int32_t*>(hist)[bin];
    dst[i] = (static_cast<int64_t>(static_cast<int16_t>(v >> 16)) << 32) |
             static_cast<int64_t>(static_cast<uint32_t>(v & 0x0000ffff));
  } else {
    dst[i] = reinterpret_cast<const int64_t*>(hist)[bin];
  }
}

__global__ void NCCLScatterLevelQuantKernel(const int64_t* __restrict__ in,
                                            const int* __restrict__ idx,
                                            const CUDAHybridPairDescriptor* __restrict__ descs,
                                            const int num_bins) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_bins) return;
  const int pair = blockIdx.y;
  if (!descs[pair].construct_valid) return;
  const int bin = (idx == nullptr) ? i : idx[i];
  hist_t* hist = descs[pair].smaller_struct->hist_in_leaf;
  const int64_t lane = in[static_cast<size_t>(pair) * num_bins + i];
  if (descs[pair].smaller_num_bits <= 16) {
    const int32_t grad = static_cast<int32_t>(lane >> 32);
    const int32_t hess = static_cast<int32_t>(lane & 0x00000000ffffffff);
    reinterpret_cast<int32_t*>(hist)[bin] =
        (static_cast<int32_t>(static_cast<uint32_t>(grad) << 16)) |
        (hess & 0x0000ffff);
  } else {
    reinterpret_cast<int64_t*>(hist)[bin] = lane;
  }
}

void CUDASingleGPUTreeLearner::LaunchNCCLGatherLevelQuant(const CUDAHybridPairDescriptor* descs,
                                                          const int* idx, int64_t* out,
                                                          int num_bins, int num_pairs,
                                                          cudaStream_t stream) {
  const int block = 256;
  dim3 grid((num_bins + block - 1) / block, num_pairs);
  NCCLGatherLevelQuantKernel<<<grid, block, 0, stream>>>(descs, idx, out, num_bins);
}

void CUDASingleGPUTreeLearner::LaunchNCCLScatterLevelQuant(const int64_t* in, const int* idx,
                                                           const CUDAHybridPairDescriptor* descs,
                                                           int num_bins, int num_pairs,
                                                           cudaStream_t stream) {
  const int block = 256;
  dim3 grid((num_bins + block - 1) / block, num_pairs);
  NCCLScatterLevelQuantKernel<<<grid, block, 0, stream>>>(in, idx, descs, num_bins);
}

}  // namespace Falcata
