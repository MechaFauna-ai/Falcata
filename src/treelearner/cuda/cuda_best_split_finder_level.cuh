/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
// The batched per-level quantized split finder and its launcher, shared by cuda_best_split_finder.cu (the variant
// without L1 and path smoothing) and cuda_best_split_finder_level_l1_smoothing.cu (the other three): each module
// holds only the variants it instantiates, so a run loads only the code it uses.
#ifndef FALCATA_TREELEARNER_CUDA_CUDA_BEST_SPLIT_FINDER_LEVEL_CUH_
#define FALCATA_TREELEARNER_CUDA_CUDA_BEST_SPLIT_FINDER_LEVEL_CUH_

#ifdef USE_CUDA

#include "cuda_best_split_finder_kernels.cuh"
#include "cuda_device_levels.hpp"

namespace Falcata {

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, typename GAIN_T>
__global__ void FindBestSplitsDiscretizedForLevelKernel(
  const int8_t* is_feature_used_bytree,
  const int8_t* const* node_masks,
  const int* node_tasks,
  const int num_tasks,
  const SplitFindTask* tasks,
  const int* used_task_indices,
  CUDARandom* cuda_randoms,
  const CUDAHybridPairDescriptor* pair_descs,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l1,
  const double lambda_l2_in,
  const double path_smooth,
  const double max_delta_step,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  const score_t* grad_scale,
  const score_t* hess_scale,
  const double hessian_ridge_quanta,
  CUDASplitInfo* cuda_best_split_info,
  const CUDADeviceLevelState* dls) {
  PdlWaitForProducer();  // cuda_plan key pdl_graph_chain: fix / subtract / copy histograms are complete and visible
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next level-chain kernel may get resident
  // cuda_plan key device_levels: a worst-case launch's pair past the level's live pairs exits before any read
  // (stale descriptors)
  if (DeviceLevelBeyondLivePairs(dls, blockIdx.y)) {
    return;
  }
  // the tree learner's hessian-quantum ridge (see the per-leaf discretized
  // kernel for the noise argument)
  const double lambda_l2 = hessian_ridge_quanta > 0.0 ?
    lambda_l2_in + hessian_ridge_quanta * static_cast<double>(*hess_scale) : lambda_l2_in;
  const unsigned int pair_index = blockIdx.y;
  const bool is_larger = (blockIdx.z == 1);
  const CUDAHybridPairDescriptor* desc = pair_descs + pair_index;
  if (is_larger ? !desc->larger_valid : !desc->smaller_valid) {
    return;
  }
  const CUDALeafSplitsStruct* leaf_splits = is_larger ? desc->larger_struct : desc->smaller_struct;
  // the descriptor (host-written or the device level's) carries the exact leaf size, and the validity flags
  // already include the min_data / min_hessian gates
  const data_size_t num_data = is_larger ? desc->num_data_in_larger_leaf : desc->num_data_in_smaller_leaf;
  unsigned int task_index;
  if (node_tasks != nullptr) {
    // cuda_plan key level_bynode_masks: the leaf's own task list (see the non-quantized level kernel)
    if (leaf_splits->leaf_index < 0) {
      return;
    }
    const int* range = node_tasks + 2 * leaf_splits->leaf_index;
    if (static_cast<int>(blockIdx.x) >= range[1]) {
      return;
    }
    task_index = static_cast<unsigned int>(node_tasks[range[0] + static_cast<int>(blockIdx.x)]);
  } else {
    task_index = used_task_indices == nullptr ? blockIdx.x : static_cast<unsigned int>(used_task_indices[blockIdx.x]);
  }
  const SplitFindTask* task = tasks + task_index;
  const double parent_gain = leaf_splits->gain;
  const int64_t sum_gradients_hessians = leaf_splits->sum_of_gradients_hessians;
  const double parent_output = leaf_splits->leaf_value;
  const unsigned int output_offset = pair_index * (2 * static_cast<unsigned int>(num_tasks)) +
    (is_larger ? task_index + num_tasks : task_index);
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  CUDARandom* cuda_random = USE_RAND ?
    (is_larger ? cuda_randoms + task_index * 2 + 1 : cuda_randoms + task_index * 2) : nullptr;
  // per-leaf histogram bit width (the descriptor's)
  const uint8_t leaf_num_bits = (is_larger ? desc->larger_num_bits : desc->smaller_num_bits);
  const bool use_16bit_bin = leaf_num_bits <= 16;
  // cuda_plan key level_bynode_masks: the leaf's own per-node mask row (a subset of the tree's); null otherwise
  const int8_t* is_feature_used = node_masks == nullptr ? is_feature_used_bytree :
    (leaf_splits->leaf_index < 0 ? nullptr : node_masks[leaf_splits->leaf_index]);
  if (is_feature_used != nullptr && is_feature_used[task->inner_feature_index]) {
    if (task->is_categorical) {
      // mirror of the per-pair discretized categorical dispatch: unpack the
      // integer bins per bin through a reader; the shared categorical body
      // serves both pipelines
      const double sum_gradients = static_cast<double>(
        static_cast<int32_t>((sum_gradients_hessians & 0xffffffff00000000) >> 32)) * (*grad_scale);
      const double sum_hessians = static_cast<double>(
        static_cast<int32_t>(sum_gradients_hessians & 0x00000000ffffffff)) * (*hess_scale) + 2 * kEpsilon;
      const int8_t* hist_base = reinterpret_cast<const int8_t*>(leaf_splits->hist_in_leaf);
      if (use_16bit_bin) {
        const CatHistReaderQuant16 hist_reader{
          reinterpret_cast<const int32_t*>(hist_base) + task->hist_offset,
          static_cast<double>(*grad_scale), static_cast<double>(*hess_scale)};
        FindBestSplitsForLeafKernelCategoricalInner<USE_RAND, USE_L1, USE_SMOOTHING, CatHistReaderQuant16>(
          hist_reader, task, cuda_random,
          lambda_l1, lambda_l2, path_smooth, max_delta_step,
          min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
          cat_smooth, cat_l2, max_cat_threshold, min_data_per_group,
          // see the non-quantized level kernel: SupportsBatchedLevel excludes
          // cat_random_search, whose draws would race across the level's pairs
          /*cat_random_search=*/0,
          parent_gain, sum_gradients, sum_hessians, num_data, parent_output,
          sum_gradients_hessians, out);
      } else {
        const CatHistReaderQuant32 hist_reader{
          reinterpret_cast<const int64_t*>(hist_base) + task->hist_offset,
          static_cast<double>(*grad_scale), static_cast<double>(*hess_scale)};
        FindBestSplitsForLeafKernelCategoricalInner<USE_RAND, USE_L1, USE_SMOOTHING, CatHistReaderQuant32>(
          hist_reader, task, cuda_random,
          lambda_l1, lambda_l2, path_smooth, max_delta_step,
          min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
          cat_smooth, cat_l2, max_cat_threshold, min_data_per_group,
          /*cat_random_search=*/0,
          parent_gain, sum_gradients, sum_hessians, num_data, parent_output,
          sum_gradients_hessians, out);
      }
    } else if (!task->reverse) {
      if (use_16bit_bin) {
        const int32_t* hist_ptr =
          reinterpret_cast<const int32_t*>(leaf_splits->hist_in_leaf) + task->hist_offset;
        FindBestSplitsDiscretizedForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, false, int32_t, int32_t, true, true, GAIN_T>(
          hist_ptr, task, cuda_random,
          lambda_l1, lambda_l2, path_smooth, max_delta_step,
          min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
          parent_gain, sum_gradients_hessians, num_data, parent_output,
          *grad_scale, *hess_scale, out);
      } else {
        // 32-bit histogram is int64-per-bin; read as int64 (see the per-pair kernel)
        const int64_t* hist_ptr =
          reinterpret_cast<const int64_t*>(leaf_splits->hist_in_leaf) + task->hist_offset;
        FindBestSplitsDiscretizedForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, false, int64_t, int64_t, false, false, GAIN_T>(
          hist_ptr, task, cuda_random,
          lambda_l1, lambda_l2, path_smooth, max_delta_step,
          min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
          parent_gain, sum_gradients_hessians, num_data, parent_output,
          *grad_scale, *hess_scale, out);
      }
    } else {
      if (use_16bit_bin) {
        const int32_t* hist_ptr =
          reinterpret_cast<const int32_t*>(leaf_splits->hist_in_leaf) + task->hist_offset;
        FindBestSplitsDiscretizedForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, true, int32_t, int32_t, true, true, GAIN_T>(
          hist_ptr, task, cuda_random,
          lambda_l1, lambda_l2, path_smooth, max_delta_step,
          min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
          parent_gain, sum_gradients_hessians, num_data, parent_output,
          *grad_scale, *hess_scale, out);
      } else {
        const int64_t* hist_ptr =
          reinterpret_cast<const int64_t*>(leaf_splits->hist_in_leaf) + task->hist_offset;
        FindBestSplitsDiscretizedForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, true, int64_t, int64_t, false, false, GAIN_T>(
          hist_ptr, task, cuda_random,
          lambda_l1, lambda_l2, path_smooth, max_delta_step,
          min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
          parent_gain, sum_gradients_hessians, num_data, parent_output,
          *grad_scale, *hess_scale, out);
      }
    }
  } else {
    out->is_valid = false;
  }
}

template <bool USE_L1, bool USE_SMOOTHING>
void CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLevelKernelInner(
  const dim3 grid_dim, const bool compact_tasks, const CUDAHybridPairDescriptor* pair_descs,
  const score_t* grad_scale, const score_t* hess_scale) {
  #define FindBestSplitsDiscretizedForLevelKernel_ARGS \
      cuda_is_feature_used_bytree_.RawData(), \
      LevelNodeMaskTable(), \
      LevelNodeTaskLists(), \
      num_tasks_, \
      cuda_split_find_tasks_.RawData(), \
      compact_tasks ? cuda_used_task_indices_.RawDataReadOnly() : nullptr, \
      cuda_randoms_.RawData(), \
      pair_descs, \
      min_data_in_leaf_, \
      min_sum_hessian_in_leaf_, \
      min_gain_to_split_, \
      lambda_l1_, \
      lambda_l2_, \
      path_smooth_, \
      max_delta_step_, \
      cat_smooth_, \
      cat_l2_, \
      max_cat_threshold_, \
      min_data_per_group_, \
      grad_scale, \
      hess_scale, \
      hessian_ridge_quanta_, \
      cuda_best_split_info_.RawData(), \
      \
      device_level_state_
  const bool pdl = PdlGraphChainEnabled();
  if (FalcataFP32GainEnabled()) {
    LaunchLevelChainKernel(pdl, FindBestSplitsDiscretizedForLevelKernel<false, USE_L1, USE_SMOOTHING, float>,
      grid_dim, dim3(NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER), 0, cuda_streams_[0],
      FindBestSplitsDiscretizedForLevelKernel_ARGS);
  } else {
    LaunchLevelChainKernel(pdl, FindBestSplitsDiscretizedForLevelKernel<false, USE_L1, USE_SMOOTHING, double>,
      grid_dim, dim3(NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER), 0, cuda_streams_[0],
      FindBestSplitsDiscretizedForLevelKernel_ARGS);
  }
  #undef FindBestSplitsDiscretizedForLevelKernel_ARGS
}

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_TREELEARNER_CUDA_CUDA_BEST_SPLIT_FINDER_LEVEL_CUH_
