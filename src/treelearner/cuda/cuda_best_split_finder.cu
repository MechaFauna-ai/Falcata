/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include "cuda_best_split_finder.hpp"

#include <LightGBM/cuda/cuda_rocm_interop.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include <LightGBM/cuda/cuda_algorithms.hpp>

namespace LightGBM {

// Match CPU's Common::RoundInt(x) = static_cast<int>(x + 0.5f) (round half up) for
// the histogram-count estimate count = round(hessian * cnt_factor). CUDA previously
// used __double2int_rn (round half to even), which disagrees with CPU at exact .5
// ties (e.g. 2.5 -> CPU 3, __double2int_rn 2), flipping per-leaf counts and
// min_data_in_leaf / cat_smooth validity for non-constant-hessian objectives and
// causing CPU/CUDA split divergence. (L2 is unaffected: cnt_factor == 1 so the
// product is an exact integer.)
__device__ __forceinline__ int CUDARoundInt(double x) {
  return static_cast<int>(x + 0.5);
}

__device__ __forceinline__ int CUDARoundInt(float x) {
  return static_cast<int>(x + 0.5f);
}

/*! \brief histogram entry read of the non-quantized find kernels: the storage
 *  is float pairs in the fp32 histogram mode (EXABOOST_FP32_HIST) and hist_t
 *  (double) pairs otherwise; the pointer is pre-offset in the right units by
 *  the caller. */
template <typename GAIN_T>
__device__ __forceinline__ GAIN_T ReadHistEntry(const hist_t* ptr, const unsigned int index, const bool fp32_hist) {
  return fp32_hist ? static_cast<GAIN_T>(reinterpret_cast<const float*>(ptr)[index]) :
    static_cast<GAIN_T>(ptr[index]);
}

/*! \brief per-task feature histogram pointer: hist_in_leaf is the leaf slot
 *  base in either layout; the task offset advances in the storage's own
 *  element width. */
__device__ __forceinline__ const hist_t* FeatureHistPtr(const hist_t* hist_in_leaf, const uint32_t hist_offset, const bool fp32_hist) {
  return fp32_hist ?
    reinterpret_cast<const hist_t*>(reinterpret_cast<const float*>(hist_in_leaf) + (hist_offset << 1)) :
    hist_in_leaf + (hist_offset << 1);
}

// Per-type machine epsilon used to size the gain-tie tolerance. Spelled out as
// device-safe constants (rather than std::numeric_limits, which is awkward in
// device code) so the reductions can run in either fp64 (default) or fp32
// (EXABOOST_FP32_GAIN) gain arithmetic.
template <typename GAIN_T>
__device__ __forceinline__ constexpr GAIN_T GainTieEpsilon();
template <>
__device__ __forceinline__ constexpr double GainTieEpsilon<double>() { return 2.2204460492503131e-16; }
template <>
__device__ __forceinline__ constexpr float GainTieEpsilon<float>() { return 1.19209290e-07f; }

// Relative tolerance for gain-tie detection in the best-split reductions.
// Two gains within this relative band are treated as a plateau; the reduction
// tie-breaks to the lower thread index, which corresponds to the lower bin index
// (for REVERSE tasks, lower thread index is earlier in CPU's reverse scan) --
// matching CPU's "first candidate in scan order wins" on a strict-'>' scan.
// Sized at ~5000x the GAIN type's epsilon, well above the reduction-order FP
// noise (~1e-15 fp64 / ~1e-7 fp32) but well below any genuine gain difference
// seen in practice. Type-dependent so that under EXABOOST_FP32_GAIN the band is
// float-appropriate (~6e-4) instead of the fixed 1e-12, which is below float
// epsilon and would degenerate to exact equality, re-exposing the plateau bug.
// Note the comparison is intentionally non-transitive: a chain of near-ties
// spanning more than one tolerance band can still be pairing-dependent in a tree
// reduction. With this ~5000x-epsilon band that is theoretical, not observed.
//
// The same helper flows through ReduceBestGain (warp + block reductions) into
// every path -- the classic per-leaf kernels, the batched level kernels, the
// graph-captured level bodies, and the cross-task SyncBestSplitForLeaf/Level
// reductions -- so one definition fixes all of them. In the cross-task reduction
// "lower thread index" is the lower TASK index; tasks are built feature-major /
// direction-minor (InitCUDAFeatureMetaInfo), so a cross-feature plateau
// tie-breaks to the lower feature index, matching the order CPU scans features.
template <typename GAIN_T>
__device__ __forceinline__ bool OtherIsBetterWithTieBreak(
    GAIN_T other_gain, GAIN_T gain, uint32_t other_thread_index, uint32_t thread_index) {
  constexpr GAIN_T kRelTol = static_cast<GAIN_T>(5.0e3) * GainTieEpsilon<GAIN_T>();
  const GAIN_T tol = fmax(fabs(gain), fabs(other_gain)) * kRelTol;
  if (other_gain > gain + tol) {
    return true;
  }
  if (fabs(other_gain - gain) <= tol && other_thread_index < thread_index) {
    return true;
  }
  return false;
}

template <typename GAIN_T>
__device__ void ReduceBestGainWarp(GAIN_T gain, bool found, uint32_t thread_index, GAIN_T* out_gain, bool* out_found, uint32_t* out_thread_index) {
  const uint32_t mask = 0xffffffff;
  const uint32_t warpLane = threadIdx.x % warpSize;
  for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    const bool other_found = __shfl_down_sync(mask, found, offset);
    const GAIN_T other_gain = __shfl_down_sync(mask, gain, offset);
    const uint32_t other_thread_index = __shfl_down_sync(mask, thread_index, offset);
    const bool other_better = OtherIsBetterWithTieBreak(other_gain, gain, other_thread_index, thread_index);
    if ((other_found && found && other_better) || (!found && other_found)) {
      found = other_found;
      gain = other_gain;
      thread_index = other_thread_index;
    }
  }
  if (warpLane == 0) {
    *out_gain = gain;
    *out_found = found;
    *out_thread_index = thread_index;
  }
}

template <typename GAIN_T>
__device__ uint32_t ReduceBestGainBlock(GAIN_T gain, bool found, uint32_t thread_index) {
  const uint32_t mask = 0xffffffff;
  for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    const bool other_found = __shfl_down_sync(mask, found, offset);
    const GAIN_T other_gain = __shfl_down_sync(mask, gain, offset);
    const uint32_t other_thread_index = __shfl_down_sync(mask, thread_index, offset);
    const bool other_better = OtherIsBetterWithTieBreak(other_gain, gain, other_thread_index, thread_index);
    if ((other_found && found && other_better) || (!found && other_found)) {
      found = other_found;
      gain = other_gain;
      thread_index = other_thread_index;
    }
  }
  return thread_index;
}

template <typename GAIN_T>
__device__ uint32_t ReduceBestGain(GAIN_T gain, bool found, uint32_t thread_index,
    GAIN_T* shared_gain_buffer, bool* shared_found_buffer, uint32_t* shared_thread_index_buffer) {
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t num_warp = blockDim.x / warpSize;
  ReduceBestGainWarp(gain, found, thread_index, shared_gain_buffer + warpID, shared_found_buffer + warpID, shared_thread_index_buffer + warpID);
  __syncthreads();
  if (warpID == 0) {
    gain = warpLane < num_warp ? shared_gain_buffer[warpLane] : static_cast<GAIN_T>(kMinScore);
    found = warpLane < num_warp ? shared_found_buffer[warpLane] : false;
    thread_index = warpLane < num_warp ? shared_thread_index_buffer[warpLane] : 0;
    thread_index = ReduceBestGainBlock(gain, found, thread_index);
  }
  return thread_index;
}

__device__ void ReduceBestGainForLeaves(double* gain, int* leaves, int cuda_cur_num_leaves) {
  const unsigned int tid = threadIdx.x;
  for (unsigned int s = 1; s < cuda_cur_num_leaves; s *= 2) {
    if (tid % (2 * s) == 0 && (tid + s) < cuda_cur_num_leaves) {
      const uint32_t tid_s = tid + s;
      if ((leaves[tid] == -1 && leaves[tid_s] != -1) || (leaves[tid] != -1 && leaves[tid_s] != -1 && gain[tid_s] > gain[tid])) {
        gain[tid] = gain[tid_s];
        leaves[tid] = leaves[tid_s];
      }
    }
    __syncthreads();
  }
}

__device__ void ReduceBestGainForLeavesWarp(double gain, int leaf_index, double* out_gain, int* out_leaf_index) {
  const uint32_t mask = 0xffffffff;
  const uint32_t warpLane = threadIdx.x % warpSize;
  for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    const int other_leaf_index = __shfl_down_sync(mask, leaf_index, offset);
    const double other_gain = __shfl_down_sync(mask, gain, offset);
    if ((leaf_index != -1 && other_leaf_index != -1 && other_gain > gain) || (leaf_index == -1 && other_leaf_index != -1)) {
      gain = other_gain;
      leaf_index = other_leaf_index;
    }
  }
  if (warpLane == 0) {
    *out_gain = gain;
    *out_leaf_index = leaf_index;
  }
}

__device__ int ReduceBestGainForLeavesBlock(double gain, int leaf_index) {
  const uint32_t mask = 0xffffffff;
  for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    const int other_leaf_index = __shfl_down_sync(mask, leaf_index, offset);
    const double other_gain = __shfl_down_sync(mask, gain, offset);
    if ((leaf_index != -1 && other_leaf_index != -1 && other_gain > gain) || (leaf_index == -1 && other_leaf_index != -1)) {
      gain = other_gain;
      leaf_index = other_leaf_index;
    }
  }
  return leaf_index;
}

__device__ int ReduceBestGainForLeaves(double gain, int leaf_index, double* shared_gain_buffer, int* shared_leaf_index_buffer) {
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t num_warp = blockDim.x / warpSize;
  ReduceBestGainForLeavesWarp(gain, leaf_index, shared_gain_buffer + warpID, shared_leaf_index_buffer + warpID);
  __syncthreads();
  if (warpID == 0) {
    gain = warpLane < num_warp ? shared_gain_buffer[warpLane] : kMinScore;
    leaf_index = warpLane < num_warp ? shared_leaf_index_buffer[warpLane] : -1;
    leaf_index = ReduceBestGainForLeavesBlock(gain, leaf_index);
  }
  return leaf_index;
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, bool REVERSE, typename GAIN_T>
__device__ void FindBestSplitsForLeafKernelInner(
  // input feature information
  const hist_t* feature_hist_ptr,
  const bool fp32_hist,
  // input task information
  const SplitFindTask* task,
  CUDARandom* cuda_random,
  // input config parameter values
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  // input parent node information
  const double parent_gain,
  const double sum_gradients,
  const double sum_hessians,
  const data_size_t num_data,
  const double parent_output,
  // output parameters
  CUDASplitInfo* cuda_best_split_info) {
  // leaf-level inputs stay double; GAIN_T = float (EXABOOST_FP32_GAIN) converts
  // them ONCE per task here, so all per-bin arithmetic below runs in fp32
  const GAIN_T cnt_factor = static_cast<GAIN_T>(num_data / sum_hessians);
  const GAIN_T min_gain_shift = static_cast<GAIN_T>(parent_gain + min_gain_to_split);
  const GAIN_T sum_gradients_acc = static_cast<GAIN_T>(sum_gradients);
  const GAIN_T sum_hessians_acc = static_cast<GAIN_T>(sum_hessians);
  const GAIN_T min_sum_hessian_acc = static_cast<GAIN_T>(min_sum_hessian_in_leaf);
  const GAIN_T lambda_l1_acc = static_cast<GAIN_T>(lambda_l1);
  const GAIN_T lambda_l2_acc = static_cast<GAIN_T>(lambda_l2);
  const GAIN_T path_smooth_acc = static_cast<GAIN_T>(path_smooth);
  const GAIN_T parent_output_acc = static_cast<GAIN_T>(parent_output);

  cuda_best_split_info->is_valid = false;

  GAIN_T local_grad_hist = 0.0f;
  GAIN_T local_hess_hist = 0.0f;
  GAIN_T local_gain = 0.0f;
  bool threshold_found = false;
  uint32_t threshold_value = 0;
  __shared__ int rand_threshold;
  if (USE_RAND && threadIdx.x == 0) {
    if (task->num_bin - 2 > 0) {
      rand_threshold = cuda_random->NextInt(0, task->num_bin - 2);
    }
  }
  __shared__ uint32_t best_thread_index;
  __shared__ GAIN_T shared_gain_buffer[WARPSIZE];
  __shared__ bool shared_bool_buffer[WARPSIZE];
  __shared__ uint32_t shared_int_buffer[WARPSIZE];
  const unsigned int threadIdx_x = threadIdx.x;
  const bool skip_sum = REVERSE ?
    (task->skip_default_bin && (task->num_bin - 1 - threadIdx_x) == static_cast<int>(task->default_bin)) :
    (task->skip_default_bin && (threadIdx_x + task->mfb_offset) == static_cast<int>(task->default_bin));
  const uint32_t feature_num_bin_minus_offset = task->num_bin - task->mfb_offset;
  if (!REVERSE) {
    if (task->na_as_missing && task->mfb_offset == 1) {
      if (threadIdx_x < static_cast<uint32_t>(task->num_bin) && threadIdx_x > 0) {
        const unsigned int bin_offset = (threadIdx_x - 1) << 1;
        local_grad_hist = ReadHistEntry<GAIN_T>(feature_hist_ptr, bin_offset, fp32_hist);
        local_hess_hist = ReadHistEntry<GAIN_T>(feature_hist_ptr, bin_offset + 1, fp32_hist);
      }
    } else {
      if (threadIdx_x < feature_num_bin_minus_offset && !skip_sum) {
        const unsigned int bin_offset = threadIdx_x << 1;
        local_grad_hist = ReadHistEntry<GAIN_T>(feature_hist_ptr, bin_offset, fp32_hist);
        local_hess_hist = ReadHistEntry<GAIN_T>(feature_hist_ptr, bin_offset + 1, fp32_hist);
      }
    }
  } else {
    if (threadIdx_x >= static_cast<unsigned int>(task->na_as_missing) &&
      threadIdx_x < feature_num_bin_minus_offset && !skip_sum) {
      const unsigned int read_index = feature_num_bin_minus_offset - 1 - threadIdx_x;
      const unsigned int bin_offset = read_index << 1;
      local_grad_hist = ReadHistEntry<GAIN_T>(feature_hist_ptr, bin_offset, fp32_hist);
      local_hess_hist = ReadHistEntry<GAIN_T>(feature_hist_ptr, bin_offset + 1, fp32_hist);
    }
  }
  __syncthreads();
  if (!REVERSE && task->na_as_missing && task->mfb_offset == 1) {
    const GAIN_T sum_gradients_non_default = ShuffleReduceSum<GAIN_T>(local_grad_hist, shared_gain_buffer, blockDim.x);
    __syncthreads();
    const GAIN_T sum_hessians_non_default = ShuffleReduceSum<GAIN_T>(local_hess_hist, shared_gain_buffer, blockDim.x);
    if (threadIdx_x == 0) {
      local_grad_hist += (sum_gradients_acc - sum_gradients_non_default);
      local_hess_hist += (sum_hessians_acc - sum_hessians_non_default);
    }
  }
  if (threadIdx_x == 0) {
    local_hess_hist += kEpsilon;
  }
  local_gain = kMinScore;
  local_grad_hist = ShufflePrefixSum(local_grad_hist, shared_gain_buffer);
  __syncthreads();
  local_hess_hist = ShufflePrefixSum(local_hess_hist, shared_gain_buffer);
  if (REVERSE) {
    if (threadIdx_x >= static_cast<unsigned int>(task->na_as_missing) && threadIdx_x <= task->num_bin - 2 && !skip_sum) {
      const GAIN_T sum_right_gradient = local_grad_hist;
      const GAIN_T sum_right_hessian = local_hess_hist;
      const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      const GAIN_T sum_left_gradient = sum_gradients_acc - sum_right_gradient;
      const GAIN_T sum_left_hessian = sum_hessians_acc - sum_right_hessian;
      const data_size_t left_count = num_data - right_count;
      if (sum_left_hessian >= min_sum_hessian_acc && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_acc && right_count >= min_data_in_leaf &&
        (!USE_RAND || static_cast<int>(task->num_bin - 2 - threadIdx_x) == rand_threshold)) {
        GAIN_T current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING, GAIN_T>(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1_acc,
          lambda_l2_acc, path_smooth_acc, left_count, right_count, parent_output_acc);
        // gain with split is worse than without split
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_value = static_cast<uint32_t>(task->num_bin - 2 - threadIdx_x);
          threshold_found = true;
        }
      }
    }
  } else {
    const uint32_t end = (task->na_as_missing && task->mfb_offset == 1) ? static_cast<uint32_t>(task->num_bin - 2) : feature_num_bin_minus_offset - 2;
    if (threadIdx_x <= end && !skip_sum) {
      const GAIN_T sum_left_gradient = local_grad_hist;
      const GAIN_T sum_left_hessian = local_hess_hist;
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const GAIN_T sum_right_gradient = sum_gradients_acc - sum_left_gradient;
      const GAIN_T sum_right_hessian = sum_hessians_acc - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_acc && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_acc && right_count >= min_data_in_leaf &&
        (!USE_RAND || static_cast<int>(threadIdx_x + task->mfb_offset) == rand_threshold)) {
        GAIN_T current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING, GAIN_T>(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1_acc,
          lambda_l2_acc, path_smooth_acc, left_count, right_count, parent_output_acc);
        // gain with split is worse than without split
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_value = (task->na_as_missing && task->mfb_offset == 1) ?
            static_cast<uint32_t>(threadIdx_x) :
            static_cast<uint32_t>(threadIdx_x + task->mfb_offset);
          threshold_found = true;
        }
      }
    }
  }
  __syncthreads();
  const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_bool_buffer, shared_int_buffer);
  if (threadIdx_x == 0) {
    best_thread_index = result;
  }
  __syncthreads();
  if (threshold_found && threadIdx_x == best_thread_index) {
    cuda_best_split_info->is_valid = true;
    cuda_best_split_info->threshold = threshold_value;
    cuda_best_split_info->gain = local_gain;
    cuda_best_split_info->default_left = task->assume_out_default_left;
    // the once-per-task output block stays in double (negligible cost)
    if (REVERSE) {
      const double sum_right_gradient = static_cast<double>(local_grad_hist);
      const double sum_right_hessian = static_cast<double>(local_hess_hist) - kEpsilon;
      const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * static_cast<double>(cnt_factor)));
      const double sum_left_gradient = sum_gradients - sum_right_gradient;
      const double sum_left_hessian = sum_hessians - sum_right_hessian - kEpsilon;
      const data_size_t left_count = num_data - right_count;
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
        sum_left_hessian, lambda_l1, lambda_l2, path_smooth, left_count, parent_output);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
        sum_right_hessian, lambda_l1, lambda_l2, path_smooth, right_count, parent_output);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient,
        sum_left_hessian, lambda_l1, lambda_l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient,
        sum_right_hessian, lambda_l1, lambda_l2, right_output);
    } else {
      const double sum_left_gradient = static_cast<double>(local_grad_hist);
      const double sum_left_hessian = static_cast<double>(local_hess_hist) - kEpsilon;
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * static_cast<double>(cnt_factor)));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian - kEpsilon;
      const data_size_t right_count = num_data - left_count;
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
        sum_left_hessian, lambda_l1, lambda_l2, path_smooth, left_count, parent_output);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
        sum_right_hessian, lambda_l1, lambda_l2, path_smooth, right_count, parent_output);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient,
        sum_left_hessian, lambda_l1, lambda_l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient,
        sum_right_hessian, lambda_l1, lambda_l2, right_output);
    }
  }
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, bool REVERSE, typename BIN_HIST_TYPE, typename ACC_HIST_TYPE, bool USE_16BIT_BIN_HIST, bool USE_16BIT_ACC_HIST, typename GAIN_T>
__device__ void FindBestSplitsDiscretizedForLeafKernelInner(
  // input feature information
  const BIN_HIST_TYPE* feature_hist_ptr,
  // input task information
  const SplitFindTask* task,
  CUDARandom* cuda_random,
  // input config parameter values
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  // input parent node information
  const double parent_gain,
  const int64_t sum_gradients_hessians,
  const data_size_t num_data,
  const double parent_output,
  // gradient scale
  const double grad_scale,
  const double hess_scale,
  // output parameters
  CUDASplitInfo* cuda_best_split_info) {
  const double sum_hessians = static_cast<double>(sum_gradients_hessians & 0x00000000ffffffff) * hess_scale;
  // leaf-level inputs converted once per task; per-bin math below runs in
  // GAIN_T (float under EXABOOST_FP32_GAIN, double otherwise)
  const GAIN_T cnt_factor = static_cast<GAIN_T>(num_data / sum_hessians);
  const GAIN_T min_gain_shift = static_cast<GAIN_T>(parent_gain + min_gain_to_split);
  const GAIN_T grad_scale_acc = static_cast<GAIN_T>(grad_scale);
  const GAIN_T hess_scale_acc = static_cast<GAIN_T>(hess_scale);
  const GAIN_T min_sum_hessian_acc = static_cast<GAIN_T>(min_sum_hessian_in_leaf);
  const GAIN_T lambda_l1_acc = static_cast<GAIN_T>(lambda_l1);
  const GAIN_T lambda_l2_acc = static_cast<GAIN_T>(lambda_l2);
  const GAIN_T path_smooth_acc = static_cast<GAIN_T>(path_smooth);
  const GAIN_T parent_output_acc = static_cast<GAIN_T>(parent_output);

  cuda_best_split_info->is_valid = false;

  ACC_HIST_TYPE local_grad_hess_hist = 0;
  GAIN_T local_gain = 0.0f;
  bool threshold_found = false;
  uint32_t threshold_value = 0;
  __shared__ int rand_threshold;
  if (USE_RAND && threadIdx.x == 0) {
    if (task->num_bin - 2 > 0) {
      rand_threshold = cuda_random->NextInt(0, task->num_bin - 2);
    }
  }
  __shared__ uint32_t best_thread_index;
  __shared__ GAIN_T shared_gain_buffer[WARPSIZE];
  __shared__ bool shared_bool_buffer[WARPSIZE];
  __shared__ uint32_t shared_int_buffer[2 * WARPSIZE];  // need 2 * WARPSIZE since the actual ACC_HIST_TYPE could be long int
  const unsigned int threadIdx_x = threadIdx.x;
  const bool skip_sum = REVERSE ?
    (task->skip_default_bin && (task->num_bin - 1 - threadIdx_x) == static_cast<int>(task->default_bin)) :
    (task->skip_default_bin && (threadIdx_x + task->mfb_offset) == static_cast<int>(task->default_bin));
  const uint32_t feature_num_bin_minus_offset = task->num_bin - task->mfb_offset;
  if (!REVERSE) {
    if (threadIdx_x < feature_num_bin_minus_offset && !skip_sum) {
      const unsigned int bin_offset = threadIdx_x;
      if (USE_16BIT_BIN_HIST && !USE_16BIT_ACC_HIST) {
        const int32_t local_grad_hess_hist_int32 = feature_hist_ptr[bin_offset];
        local_grad_hess_hist = (static_cast<int64_t>(static_cast<int16_t>(local_grad_hess_hist_int32 >> 16)) << 32) | (static_cast<int64_t>(local_grad_hess_hist_int32 & 0x0000ffff));
      } else {
        local_grad_hess_hist = feature_hist_ptr[bin_offset];
      }
    }
  } else {
    if (threadIdx_x >= static_cast<unsigned int>(task->na_as_missing) &&
      threadIdx_x < feature_num_bin_minus_offset && !skip_sum) {
      const unsigned int read_index = feature_num_bin_minus_offset - 1 - threadIdx_x;
      if (USE_16BIT_BIN_HIST && !USE_16BIT_ACC_HIST) {
        const int32_t local_grad_hess_hist_int32 = feature_hist_ptr[read_index];
        local_grad_hess_hist = (static_cast<int64_t>(static_cast<int16_t>(local_grad_hess_hist_int32 >> 16)) << 32) | (static_cast<int64_t>(local_grad_hess_hist_int32 & 0x0000ffff));
      } else {
        local_grad_hess_hist = feature_hist_ptr[read_index];
      }
    }
  }
  __syncthreads();
  local_gain = kMinScore;
  local_grad_hess_hist = ShufflePrefixSum<ACC_HIST_TYPE>(local_grad_hess_hist, reinterpret_cast<ACC_HIST_TYPE*>(shared_int_buffer));
  GAIN_T sum_left_gradient = 0.0f;
  GAIN_T sum_left_hessian = 0.0f;
  GAIN_T sum_right_gradient = 0.0f;
  GAIN_T sum_right_hessian = 0.0f;
  data_size_t left_count = 0;
  data_size_t right_count = 0;
  int64_t sum_left_gradient_hessian = 0;
  int64_t sum_right_gradient_hessian = 0;
  if (REVERSE) {
    if (threadIdx_x >= static_cast<unsigned int>(task->na_as_missing) && threadIdx_x <= task->num_bin - 2 && !skip_sum) {
      sum_right_gradient_hessian = USE_16BIT_ACC_HIST ?
        (static_cast<int64_t>(static_cast<int16_t>(local_grad_hess_hist >> 16)) << 32) | static_cast<int64_t>(local_grad_hess_hist & 0x0000ffff) :
        local_grad_hess_hist;
      sum_right_gradient = static_cast<GAIN_T>(static_cast<int32_t>((sum_right_gradient_hessian & 0xffffffff00000000) >> 32)) * grad_scale_acc;
      sum_right_hessian = static_cast<GAIN_T>(static_cast<int32_t>(sum_right_gradient_hessian & 0x00000000ffffffff)) * hess_scale_acc;
      right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      sum_left_gradient_hessian = sum_gradients_hessians - sum_right_gradient_hessian;
      sum_left_gradient = static_cast<GAIN_T>(static_cast<int32_t>((sum_left_gradient_hessian & 0xffffffff00000000)>> 32)) * grad_scale_acc;
      sum_left_hessian = static_cast<GAIN_T>(static_cast<int32_t>(sum_left_gradient_hessian & 0x00000000ffffffff)) * hess_scale_acc;
      left_count = num_data - right_count;
      if (sum_left_hessian >= min_sum_hessian_acc && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_acc && right_count >= min_data_in_leaf &&
        (!USE_RAND || static_cast<int>(task->num_bin - 2 - threadIdx_x) == rand_threshold)) {
        GAIN_T current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING, GAIN_T>(
          sum_left_gradient, sum_left_hessian + kEpsilon, sum_right_gradient,
          sum_right_hessian + kEpsilon, lambda_l1_acc,
          lambda_l2_acc, path_smooth_acc, left_count, right_count, parent_output_acc);
        // gain with split is worse than without split
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_value = static_cast<uint32_t>(task->num_bin - 2 - threadIdx_x);
          threshold_found = true;
        }
      }
    }
  } else {
    if (threadIdx_x <= feature_num_bin_minus_offset - 2 && !skip_sum) {
      sum_left_gradient_hessian = USE_16BIT_ACC_HIST ?
        (static_cast<int64_t>(static_cast<int16_t>(local_grad_hess_hist >> 16)) << 32) | static_cast<int64_t>(local_grad_hess_hist & 0x0000ffff) :
        local_grad_hess_hist;
      sum_left_gradient = static_cast<GAIN_T>(static_cast<int32_t>((sum_left_gradient_hessian & 0xffffffff00000000) >> 32)) * grad_scale_acc;
      sum_left_hessian = static_cast<GAIN_T>(static_cast<int32_t>(sum_left_gradient_hessian & 0x00000000ffffffff)) * hess_scale_acc;
      left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      sum_right_gradient_hessian = sum_gradients_hessians - sum_left_gradient_hessian;
      sum_right_gradient = static_cast<GAIN_T>(static_cast<int32_t>((sum_right_gradient_hessian & 0xffffffff00000000) >> 32)) * grad_scale_acc;
      sum_right_hessian = static_cast<GAIN_T>(static_cast<int32_t>(sum_right_gradient_hessian & 0x00000000ffffffff)) * hess_scale_acc;
      right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_acc && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_acc && right_count >= min_data_in_leaf &&
        (!USE_RAND || static_cast<int>(threadIdx_x + task->mfb_offset) == rand_threshold)) {
        GAIN_T current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING, GAIN_T>(
          sum_left_gradient, sum_left_hessian + kEpsilon, sum_right_gradient,
          sum_right_hessian + kEpsilon, lambda_l1_acc,
          lambda_l2_acc, path_smooth_acc, left_count, right_count, parent_output_acc);
        // gain with split is worse than without split
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_value = static_cast<uint32_t>(threadIdx_x + task->mfb_offset);
          threshold_found = true;
        }
      }
    }
  }
  __syncthreads();
  const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_bool_buffer, shared_int_buffer);
  if (threadIdx_x == 0) {
    best_thread_index = result;
  }
  __syncthreads();
  if (threshold_found && threadIdx_x == best_thread_index) {
    cuda_best_split_info->is_valid = true;
    cuda_best_split_info->threshold = threshold_value;
    cuda_best_split_info->gain = local_gain;
    cuda_best_split_info->default_left = task->assume_out_default_left;
    // the once-per-task output block stays in double (negligible cost)
    const double sum_left_gradient_dbl = static_cast<double>(sum_left_gradient);
    const double sum_left_hessian_dbl = static_cast<double>(sum_left_hessian);
    const double sum_right_gradient_dbl = static_cast<double>(sum_right_gradient);
    const double sum_right_hessian_dbl = static_cast<double>(sum_right_hessian);
    const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient_dbl,
      sum_left_hessian_dbl, lambda_l1, lambda_l2, path_smooth, left_count, parent_output);
    const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient_dbl,
      sum_right_hessian_dbl, lambda_l1, lambda_l2, path_smooth, right_count, parent_output);
    cuda_best_split_info->left_sum_gradients = sum_left_gradient_dbl;
    cuda_best_split_info->left_sum_hessians = sum_left_hessian_dbl;
    cuda_best_split_info->left_sum_of_gradients_hessians = sum_left_gradient_hessian;
    cuda_best_split_info->left_count = left_count;
    cuda_best_split_info->right_sum_gradients = sum_right_gradient_dbl;
    cuda_best_split_info->right_sum_hessians = sum_right_hessian_dbl;
    cuda_best_split_info->right_sum_of_gradients_hessians = sum_right_gradient_hessian;
    cuda_best_split_info->right_count = right_count;
    cuda_best_split_info->left_value = left_output;
    cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient_dbl,
      sum_left_hessian_dbl, lambda_l1, lambda_l2, left_output);
    cuda_best_split_info->right_value = right_output;
    cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient_dbl,
      sum_right_hessian_dbl, lambda_l1, lambda_l2, right_output);
  }
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING>
__device__ void FindBestSplitsForLeafKernelCategoricalInner(
  // input feature information
  const hist_t* feature_hist_ptr,
  // input task information
  const SplitFindTask* task,
  CUDARandom* cuda_random,
  // input config parameter values
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  // input parent node information
  const double parent_gain,
  const double sum_gradients,
  const double sum_hessians,
  const data_size_t num_data,
  const double parent_output,
  // output parameters
  CUDASplitInfo* cuda_best_split_info) {
  __shared__ double shared_gain_buffer[WARPSIZE];
  __shared__ bool shared_found_buffer[WARPSIZE];
  __shared__ uint32_t shared_thread_index_buffer[WARPSIZE];
  __shared__ uint32_t best_thread_index;
  const double cnt_factor = num_data / sum_hessians;
  const double min_gain_shift = parent_gain + min_gain_to_split;
  double l2 = lambda_l2;

  double local_gain = min_gain_shift;
  bool threshold_found = false;

  cuda_best_split_info->is_valid = false;

  const int bin_start = 1 - task->mfb_offset;
  const int bin_end = task->num_bin - task->mfb_offset;
  const int threadIdx_x = static_cast<int>(threadIdx.x);

  __shared__ int rand_threshold;

  if (task->is_one_hot) {
    if (USE_RAND && threadIdx.x == 0) {
      rand_threshold = 0;
      if (bin_end > bin_start) {
        rand_threshold = cuda_random->NextInt(bin_start, bin_end);
      }
    }
    __syncthreads();
    if (threadIdx_x >= bin_start && threadIdx_x < bin_end) {
      const int bin_offset = (threadIdx_x << 1);
      const hist_t grad = feature_hist_ptr[bin_offset];
      const hist_t hess = feature_hist_ptr[bin_offset + 1];
      data_size_t cnt =
            static_cast<data_size_t>(CUDARoundInt(hess * cnt_factor));
      if (cnt >= min_data_in_leaf && hess >= min_sum_hessian_in_leaf) {
        const data_size_t other_count = num_data - cnt;
        if (other_count >= min_data_in_leaf) {
          const double sum_other_hessian = sum_hessians - hess - kEpsilon;
          if (sum_other_hessian >= min_sum_hessian_in_leaf && (!USE_RAND || static_cast<int>(threadIdx_x) == rand_threshold)) {
            const double sum_other_gradient = sum_gradients - grad;
            double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
              sum_other_gradient, sum_other_hessian, grad,
              hess + kEpsilon, lambda_l1,
              l2, path_smooth, other_count, cnt, parent_output);
            if (current_gain > min_gain_shift) {
              local_gain = current_gain;
              threshold_found = true;
            }
          }
        }
      }
    }
    __syncthreads();
    const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
    if (threadIdx_x == 0) {
      best_thread_index = result;
    }
    __syncthreads();
    if (threshold_found && threadIdx_x == best_thread_index) {
      cuda_best_split_info->is_valid = true;
      cuda_best_split_info->num_cat_threshold = 1;
      cuda_best_split_info->gain = local_gain - min_gain_shift;
      *(cuda_best_split_info->cat_threshold) = static_cast<uint32_t>(threadIdx_x + task->mfb_offset);
      cuda_best_split_info->default_left = false;
      const int bin_offset = (threadIdx_x << 1);
      const hist_t sum_left_gradient = feature_hist_ptr[bin_offset];
      const hist_t sum_left_hessian = feature_hist_ptr[bin_offset + 1];
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
        sum_left_hessian, lambda_l1, l2, path_smooth, left_count, parent_output);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
        sum_right_hessian, lambda_l1, l2, path_smooth, right_count, parent_output);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient,
        sum_left_hessian, lambda_l1, l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient,
        sum_right_hessian, lambda_l1, l2, right_output);
    }
  } else {
    __shared__ double shared_value_buffer[NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER];
    __shared__ int16_t shared_index_buffer[NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER];
    __shared__ uint16_t shared_mem_buffer_uint16[WARPSIZE];
    __shared__ double shared_mem_buffer_double[WARPSIZE];
    __shared__ int used_bin;
    l2 += cat_l2;
    uint16_t is_valid_bin = 0;
    int best_dir = 0;
    double best_sum_left_gradient = 0.0f;
    double best_sum_left_hessian = 0.0f;
    if (threadIdx_x >= bin_start && threadIdx_x < bin_end) {
      const int bin_offset = (threadIdx_x << 1);
      const double hess = feature_hist_ptr[bin_offset + 1];
      if (CUDARoundInt(hess * cnt_factor) >= cat_smooth) {
        const double grad = feature_hist_ptr[bin_offset];
        shared_value_buffer[threadIdx_x] = grad / (hess + cat_smooth);
        is_valid_bin = 1;
      } else {
        shared_value_buffer[threadIdx_x] = kMaxScore;
      }
    } else {
      shared_value_buffer[threadIdx_x] = kMaxScore;
    }
    shared_index_buffer[threadIdx_x] = threadIdx_x;
    __syncthreads();
    const int local_used_bin = ShuffleReduceSum<uint16_t>(is_valid_bin, shared_mem_buffer_uint16, blockDim.x);
    if (threadIdx_x == 0) {
      used_bin = local_used_bin;
    }
    __syncthreads();
    BitonicArgSort_1024<double, int16_t, true>(shared_value_buffer, shared_index_buffer, bin_end);
    __syncthreads();
    const int max_num_cat = min(max_cat_threshold, (used_bin + 1) / 2);

    if (USE_RAND) {
      rand_threshold = 0;
      const int max_threshold = max(min(max_num_cat, used_bin) - 1, 0);
      if (max_threshold > 0) {
        rand_threshold = cuda_random->NextInt(0, max_threshold);
      }
    }

    // left to right
    double grad = 0.0f;
    double hess = 0.0f;
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const int bin_offset = (shared_index_buffer[threadIdx_x] << 1);
      grad = feature_hist_ptr[bin_offset];
      hess = feature_hist_ptr[bin_offset + 1];
    }
    if (threadIdx_x == 0) {
      hess += kEpsilon;
    }
    __syncthreads();
    double sum_left_gradient = ShufflePrefixSum<double>(grad, shared_mem_buffer_double);
    __syncthreads();
    double sum_left_hessian = ShufflePrefixSum<double>(hess, shared_mem_buffer_double);
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf &&
        left_count >= min_data_per_group && right_count >= min_data_per_group &&
        (!USE_RAND || threadIdx_x == static_cast<int>(rand_threshold))) {
        double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1,
          l2, path_smooth, left_count, right_count, parent_output);
        // gain with split is worse than without split
        if (current_gain > local_gain) {
          local_gain = current_gain;
          threshold_found = true;
          best_dir = 1;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
        }
      }
    }
    __syncthreads();

    // right to left
    grad = 0.0f;
    hess = 0.0f;
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const int bin_offset = (shared_index_buffer[used_bin - 1 - threadIdx_x] << 1);
      grad = feature_hist_ptr[bin_offset];
      hess = feature_hist_ptr[bin_offset + 1];
    }
    if (threadIdx_x == 0) {
      hess += kEpsilon;
    }
    __syncthreads();
    sum_left_gradient = ShufflePrefixSum<double>(grad, shared_mem_buffer_double);
    __syncthreads();
    sum_left_hessian = ShufflePrefixSum<double>(hess, shared_mem_buffer_double);
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf &&
        left_count >= min_data_per_group && right_count >= min_data_per_group &&
        (!USE_RAND || threadIdx_x == static_cast<int>(rand_threshold))) {
        double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1,
          l2, path_smooth, left_count, right_count, parent_output);
        // gain with split is worse than without split
        if (current_gain > local_gain) {
          local_gain = current_gain;
          threshold_found = true;
          best_dir = -1;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
        }
      }
    }
    __syncthreads();

    const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
    if (threadIdx_x == 0) {
      best_thread_index = result;
    }
    __syncthreads();
    if (threshold_found && threadIdx_x == best_thread_index) {
      cuda_best_split_info->is_valid = true;
      cuda_best_split_info->num_cat_threshold = threadIdx_x + 1;
      cuda_best_split_info->gain = local_gain - min_gain_shift;
      if (best_dir == 1) {
        for (int i = 0; i < threadIdx_x + 1; ++i) {
          (cuda_best_split_info->cat_threshold)[i] = shared_index_buffer[i] + task->mfb_offset;
        }
      } else {
        for (int i = 0; i < threadIdx_x + 1; ++i) {
          (cuda_best_split_info->cat_threshold)[i] = shared_index_buffer[used_bin - 1 - i] + task->mfb_offset;
        }
      }
      cuda_best_split_info->default_left = false;
      const hist_t sum_left_gradient = best_sum_left_gradient;
      const hist_t sum_left_hessian = best_sum_left_hessian;
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
        sum_left_hessian, lambda_l1, l2, path_smooth, left_count, parent_output);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
        sum_right_hessian, lambda_l1, l2, path_smooth, right_count, parent_output);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient,
        sum_left_hessian, lambda_l1, l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient,
        sum_right_hessian, lambda_l1, l2, right_output);
    }
  }
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, bool IS_LARGER, typename GAIN_T>
__global__ void FindBestSplitsForLeafKernel(
  // input feature information
  const int8_t* is_feature_used_bytree,
  const bool fp32_hist,
  // input task information
  const int num_tasks,
  const SplitFindTask* tasks,
  CUDARandom* cuda_randoms,
  // input leaf information
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const CUDALeafSplitsStruct* larger_leaf_splits,
  // input config parameter values
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  // output
  CUDASplitInfo* cuda_best_split_info,
  // global num data in leaf
  const data_size_t global_num_data_in_smaller_leaf,
  const data_size_t global_num_data_in_larger_leaf) {
  const unsigned int task_index = blockIdx.x;
  const SplitFindTask* task = tasks + task_index;
  const int inner_feature_index = task->inner_feature_index;
  const double parent_gain = IS_LARGER ? larger_leaf_splits->gain : smaller_leaf_splits->gain;
  const double sum_gradients = IS_LARGER ? larger_leaf_splits->sum_of_gradients : smaller_leaf_splits->sum_of_gradients;
  const double sum_hessians = (IS_LARGER ? larger_leaf_splits->sum_of_hessians : smaller_leaf_splits->sum_of_hessians) + 2 * kEpsilon;
  const data_size_t num_data = IS_LARGER ? global_num_data_in_larger_leaf : global_num_data_in_smaller_leaf;
  const double parent_output = IS_LARGER ? larger_leaf_splits->leaf_value : smaller_leaf_splits->leaf_value;
  const unsigned int output_offset = IS_LARGER ? (task_index + num_tasks) : task_index;
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  CUDARandom* cuda_random = USE_RAND ?
    (IS_LARGER ? cuda_randoms + task_index * 2 + 1 : cuda_randoms + task_index * 2) : nullptr;
  if (is_feature_used_bytree[inner_feature_index]) {
    const hist_t* hist_ptr = FeatureHistPtr(
      IS_LARGER ? larger_leaf_splits->hist_in_leaf : smaller_leaf_splits->hist_in_leaf, task->hist_offset, fp32_hist);
    if (task->is_categorical) {
      // fp32_hist is globally disabled for datasets with categorical features
      FindBestSplitsForLeafKernelCategoricalInner<USE_RAND, USE_L1, USE_SMOOTHING>(
        // input feature information
        hist_ptr,
        // input task information
        task,
        cuda_random,
        // input config parameter values
        lambda_l1,
        lambda_l2,
        path_smooth,
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        min_gain_to_split,
        cat_smooth,
        cat_l2,
        max_cat_threshold,
        min_data_per_group,
        // input parent node information
        parent_gain,
        sum_gradients,
        sum_hessians,
        num_data,
        parent_output,
        // output parameters
        out);
    } else {
      if (!task->reverse) {
        FindBestSplitsForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, false, GAIN_T>(
          // input feature information
          hist_ptr,
          fp32_hist,
          // input task information
          task,
          cuda_random,
          // input config parameter values
          lambda_l1,
          lambda_l2,
          path_smooth,
          min_data_in_leaf,
          min_sum_hessian_in_leaf,
          min_gain_to_split,
          // input parent node information
          parent_gain,
          sum_gradients,
          sum_hessians,
          num_data,
          parent_output,
          // output parameters
          out);
      } else {
        FindBestSplitsForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, true, GAIN_T>(
          // input feature information
          hist_ptr,
          fp32_hist,
          // input task information
          task,
          cuda_random,
          // input config parameter values
          lambda_l1,
          lambda_l2,
          path_smooth,
          min_data_in_leaf,
          min_sum_hessian_in_leaf,
          min_gain_to_split,
          // input parent node information
          parent_gain,
          sum_gradients,
          sum_hessians,
          num_data,
          parent_output,
          // output parameters
          out);
      }
    }
  } else {
    out->is_valid = false;
  }
}


template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, bool IS_LARGER, typename GAIN_T>
__global__ void FindBestSplitsDiscretizedForLeafKernel(
  // input feature information
  const int8_t* is_feature_used_bytree,
  // input task information
  const int num_tasks,
  const SplitFindTask* tasks,
  CUDARandom* cuda_randoms,
  // input leaf information
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const CUDALeafSplitsStruct* larger_leaf_splits,
  const uint8_t smaller_leaf_num_bits_in_histogram_bin,
  const uint8_t larger_leaf_num_bits_in_histogram_bin,
  // input config parameter values
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  const int max_cat_to_onehot,
  // gradient scale
  const score_t* grad_scale,
  const score_t* hess_scale,
  // output
  CUDASplitInfo* cuda_best_split_info,
  // global num data in leaf
  const data_size_t global_num_data_in_smaller_leaf,
  const data_size_t global_num_data_in_larger_leaf) {
  const unsigned int task_index = blockIdx.x;
  const SplitFindTask* task = tasks + task_index;
  const int inner_feature_index = task->inner_feature_index;
  const double parent_gain = IS_LARGER ? larger_leaf_splits->gain : smaller_leaf_splits->gain;
  const int64_t sum_gradients_hessians = IS_LARGER ? larger_leaf_splits->sum_of_gradients_hessians : smaller_leaf_splits->sum_of_gradients_hessians;
  const data_size_t num_data = IS_LARGER ? global_num_data_in_larger_leaf : global_num_data_in_smaller_leaf;
  const double parent_output = IS_LARGER ? larger_leaf_splits->leaf_value : smaller_leaf_splits->leaf_value;
  const unsigned int output_offset = IS_LARGER ? (task_index + num_tasks) : task_index;
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  CUDARandom* cuda_random = USE_RAND ?
    (IS_LARGER ? cuda_randoms + task_index * 2 + 1 : cuda_randoms + task_index * 2) : nullptr;
  const bool use_16bit_bin = IS_LARGER ? (larger_leaf_num_bits_in_histogram_bin <= 16) : (smaller_leaf_num_bits_in_histogram_bin <= 16);
  if (is_feature_used_bytree[inner_feature_index]) {
    if (task->is_categorical) {
      __threadfence();  // ensure store issued before trap
#if defined(USE_ROCM)
      __builtin_trap();
#else
      asm("trap;");
#endif
    } else {
      if (!task->reverse) {
        if (use_16bit_bin) {
          const int32_t* hist_ptr =
            reinterpret_cast<const int32_t*>(IS_LARGER ? larger_leaf_splits->hist_in_leaf : smaller_leaf_splits->hist_in_leaf) + task->hist_offset;
          FindBestSplitsDiscretizedForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, false, int32_t, int32_t, true, true, GAIN_T>(
            // input feature information
            hist_ptr,
            // input task information
            task,
            cuda_random,
            // input config parameter values
            lambda_l1,
            lambda_l2,
            path_smooth,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            min_gain_to_split,
            // input parent node information
            parent_gain,
            sum_gradients_hessians,
            num_data,
            parent_output,
            // gradient scale
            *grad_scale,
            *hess_scale,
            // output parameters
            out);
        } else {
          // 32-bit histogram: each bin is an int64 (grad32 << 32 | hess32), written by
          // the construct kernel via int64 atomics. It must be read as int64 (offset in
          // bin units). Reading it as int32 read 4-byte half-bins at the wrong stride and
          // produced garbage splits whenever a leaf needed 32-bit bins (large leaves).
          const int64_t* hist_ptr =
            reinterpret_cast<const int64_t*>(IS_LARGER ? larger_leaf_splits->hist_in_leaf : smaller_leaf_splits->hist_in_leaf) + task->hist_offset;
          FindBestSplitsDiscretizedForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, false, int64_t, int64_t, false, false, GAIN_T>(
            // input feature information
            hist_ptr,
            // input task information
            task,
            cuda_random,
            // input config parameter values
            lambda_l1,
            lambda_l2,
            path_smooth,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            min_gain_to_split,
            // input parent node information
            parent_gain,
            sum_gradients_hessians,
            num_data,
            parent_output,
            // gradient scale
            *grad_scale,
            *hess_scale,
            // output parameters
            out);
        }
      } else {
        if (use_16bit_bin) {
          const int32_t* hist_ptr =
            reinterpret_cast<const int32_t*>(IS_LARGER ? larger_leaf_splits->hist_in_leaf : smaller_leaf_splits->hist_in_leaf) + task->hist_offset;
          FindBestSplitsDiscretizedForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, true, int32_t, int32_t, true, true, GAIN_T>(
            // input feature information
            hist_ptr,
            // input task information
            task,
            cuda_random,
            // input config parameter values
            lambda_l1,
            lambda_l2,
            path_smooth,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            min_gain_to_split,
            // input parent node information
            parent_gain,
            sum_gradients_hessians,
            num_data,
            parent_output,
            // gradient scale
            *grad_scale,
            *hess_scale,
            // output parameters
            out);
        } else {
          // 32-bit histogram is int64-per-bin; read as int64 (see forward branch above).
          const int64_t* hist_ptr =
            reinterpret_cast<const int64_t*>(IS_LARGER ? larger_leaf_splits->hist_in_leaf : smaller_leaf_splits->hist_in_leaf) + task->hist_offset;
          FindBestSplitsDiscretizedForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, true, int64_t, int64_t, false, false, GAIN_T>(
            // input feature information
            hist_ptr,
            // input task information
            task,
            cuda_random,
            // input config parameter values
            lambda_l1,
            lambda_l2,
            path_smooth,
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            min_gain_to_split,
            // input parent node information
            parent_gain,
            sum_gradients_hessians,
            num_data,
            parent_output,
            // gradient scale
            *grad_scale,
            *hess_scale,
            // output parameters
            out);
        }
      }
    }
  } else {
    out->is_valid = false;
  }
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, bool REVERSE>
__device__ void FindBestSplitsForLeafKernelInner_GlobalMemory(
  // input feature information
  const hist_t* feature_hist_ptr,
  // input task information
  const SplitFindTask* task,
  CUDARandom* cuda_random,
  // input config parameter values
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  // input parent node information
  const double parent_gain,
  const double sum_gradients,
  const double sum_hessians,
  const data_size_t num_data,
  const double parent_output,
  // output parameters
  CUDASplitInfo* cuda_best_split_info,
  // buffer
  hist_t* hist_grad_buffer_ptr,
  hist_t* hist_hess_buffer_ptr) {
  const double cnt_factor = num_data / sum_hessians;
  const double min_gain_shift = parent_gain + min_gain_to_split;

  cuda_best_split_info->is_valid = false;
  double local_gain = 0.0f;
  bool threshold_found = false;
  uint32_t threshold_value = 0;
  __shared__ int rand_threshold;
  if (USE_RAND && threadIdx.x == 0) {
    if (task->num_bin - 2 > 0) {
      rand_threshold = cuda_random->NextInt(0, task->num_bin - 2);
    }
  }
  __shared__ uint32_t best_thread_index;
  __shared__ double shared_double_buffer[WARPSIZE];
  __shared__ bool shared_found_buffer[WARPSIZE];
  __shared__ uint32_t shared_thread_index_buffer[WARPSIZE];
  const unsigned int threadIdx_x = threadIdx.x;
  const uint32_t feature_num_bin_minus_offset = task->num_bin - task->mfb_offset;
  if (!REVERSE) {
    if (task->na_as_missing && task->mfb_offset == 1) {
      uint32_t bin_start = threadIdx_x > 0 ? threadIdx_x : blockDim.x;
      hist_t thread_sum_gradients = 0.0f;
      hist_t thread_sum_hessians = 0.0f;
      for (unsigned int bin = bin_start; bin < static_cast<uint32_t>(task->num_bin); bin += blockDim.x) {
        const unsigned int bin_offset = (bin - 1) << 1;
        const hist_t grad = feature_hist_ptr[bin_offset];
        const hist_t hess = feature_hist_ptr[bin_offset + 1];
        hist_grad_buffer_ptr[bin] = grad;
        hist_hess_buffer_ptr[bin] = hess;
        thread_sum_gradients += grad;
        thread_sum_hessians += hess;
      }
      const hist_t sum_gradients_non_default = ShuffleReduceSum<double>(thread_sum_gradients, shared_double_buffer, blockDim.x);
      __syncthreads();
      const hist_t sum_hessians_non_default = ShuffleReduceSum<double>(thread_sum_hessians, shared_double_buffer, blockDim.x);
      if (threadIdx_x == 0) {
        hist_grad_buffer_ptr[0] = sum_gradients - sum_gradients_non_default;
        hist_hess_buffer_ptr[0] = sum_hessians - sum_hessians_non_default;
      }
    } else {
      for (unsigned int bin = threadIdx_x; bin < feature_num_bin_minus_offset; bin += blockDim.x) {
        const bool skip_sum =
          (task->skip_default_bin && (bin + task->mfb_offset) == static_cast<int>(task->default_bin));
        if (!skip_sum) {
          const unsigned int bin_offset = bin << 1;
          hist_grad_buffer_ptr[bin] = feature_hist_ptr[bin_offset];
          hist_hess_buffer_ptr[bin] = feature_hist_ptr[bin_offset + 1];
        } else {
          hist_grad_buffer_ptr[bin] = 0.0f;
          hist_hess_buffer_ptr[bin] = 0.0f;
        }
      }
    }
  } else {
    for (unsigned int bin = threadIdx_x; bin < feature_num_bin_minus_offset; bin += blockDim.x) {
      const bool skip_sum = bin >= static_cast<unsigned int>(task->na_as_missing) &&
        (task->skip_default_bin && (task->num_bin - 1 - bin) == static_cast<int>(task->default_bin));
      if (!skip_sum) {
        const unsigned int read_index = feature_num_bin_minus_offset - 1 - bin;
        const unsigned int bin_offset = read_index << 1;
        hist_grad_buffer_ptr[bin] = feature_hist_ptr[bin_offset];
        hist_hess_buffer_ptr[bin] = feature_hist_ptr[bin_offset + 1];
      } else {
        hist_grad_buffer_ptr[bin] = 0.0f;
        hist_hess_buffer_ptr[bin] = 0.0f;
      }
    }
  }
  __syncthreads();
  if (threadIdx_x == 0) {
    hist_hess_buffer_ptr[0] += kEpsilon;
  }
  local_gain = kMinScore;
  GlobalMemoryPrefixSum(hist_grad_buffer_ptr, static_cast<size_t>(feature_num_bin_minus_offset));
  __syncthreads();
  GlobalMemoryPrefixSum(hist_hess_buffer_ptr, static_cast<size_t>(feature_num_bin_minus_offset));
  if (REVERSE) {
    for (unsigned int bin = threadIdx_x; bin < feature_num_bin_minus_offset; bin += blockDim.x) {
      const bool skip_sum = (bin >= static_cast<unsigned int>(task->na_as_missing) &&
        (task->skip_default_bin && (task->num_bin - 1 - bin) == static_cast<int>(task->default_bin)));
      if (!skip_sum) {
        const double sum_right_gradient = hist_grad_buffer_ptr[bin];
        const double sum_right_hessian = hist_hess_buffer_ptr[bin];
        const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
        const double sum_left_gradient = sum_gradients - sum_right_gradient;
        const double sum_left_hessian = sum_hessians - sum_right_hessian;
        const data_size_t left_count = num_data - right_count;
        if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
          sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf &&
          (!USE_RAND || static_cast<int>(task->num_bin - 2 - bin) == rand_threshold)) {
          double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, lambda_l1,
            lambda_l2, path_smooth, left_count, right_count, parent_output);
          // gain with split is worse than without split
          if (current_gain > min_gain_shift) {
            local_gain = current_gain - min_gain_shift;
            threshold_value = static_cast<uint32_t>(task->num_bin - 2 - bin);
            threshold_found = true;
          }
        }
      }
    }
  } else {
    const uint32_t end = (task->na_as_missing && task->mfb_offset == 1) ? static_cast<uint32_t>(task->num_bin - 2) : feature_num_bin_minus_offset - 2;
    for (unsigned int bin = threadIdx_x; bin <= end; bin += blockDim.x) {
      const bool skip_sum =
        (task->skip_default_bin && (bin + task->mfb_offset) == static_cast<int>(task->default_bin));
      if (!skip_sum) {
        const double sum_left_gradient = hist_grad_buffer_ptr[bin];
        const double sum_left_hessian = hist_hess_buffer_ptr[bin];
        const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
        const double sum_right_gradient = sum_gradients - sum_left_gradient;
        const double sum_right_hessian = sum_hessians - sum_left_hessian;
        const data_size_t right_count = num_data - left_count;
        if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
          sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf &&
          (!USE_RAND || static_cast<int>(bin + task->mfb_offset) == rand_threshold)) {
          double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, lambda_l1,
            lambda_l2, path_smooth, left_count, right_count, parent_output);
          // gain with split is worse than without split
          if (current_gain > min_gain_shift) {
            local_gain = current_gain - min_gain_shift;
            threshold_value = (task->na_as_missing && task->mfb_offset == 1) ?
              bin : static_cast<uint32_t>(bin + task->mfb_offset);
            threshold_found = true;
          }
        }
      }
    }
  }
  __syncthreads();
  const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_double_buffer, shared_found_buffer, shared_thread_index_buffer);
  if (threadIdx_x == 0) {
    best_thread_index = result;
  }
  __syncthreads();
  if (threshold_found && threadIdx_x == best_thread_index) {
    cuda_best_split_info->is_valid = true;
    cuda_best_split_info->threshold = threshold_value;
    cuda_best_split_info->gain = local_gain;
    cuda_best_split_info->default_left = task->assume_out_default_left;
    if (REVERSE) {
      const unsigned int best_bin = static_cast<uint32_t>(task->num_bin - 2 - threshold_value);
      const double sum_right_gradient = hist_grad_buffer_ptr[best_bin];
      const double sum_right_hessian = hist_hess_buffer_ptr[best_bin] - kEpsilon;
      const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      const double sum_left_gradient = sum_gradients - sum_right_gradient;
      const double sum_left_hessian = sum_hessians - sum_right_hessian - kEpsilon;
      const data_size_t left_count = num_data - right_count;
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
        sum_left_hessian, lambda_l1, lambda_l2, path_smooth, left_count, parent_output);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
        sum_right_hessian, lambda_l1, lambda_l2, path_smooth, right_count, parent_output);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient,
        sum_left_hessian, lambda_l1, lambda_l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient,
        sum_right_hessian, lambda_l1, lambda_l2, right_output);
    } else {
      const unsigned int best_bin = (task->na_as_missing && task->mfb_offset == 1) ?
        threshold_value : static_cast<uint32_t>(threshold_value - task->mfb_offset);
      const double sum_left_gradient = hist_grad_buffer_ptr[best_bin];
      const double sum_left_hessian = hist_hess_buffer_ptr[best_bin] - kEpsilon;
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian - kEpsilon;
      const data_size_t right_count = num_data - left_count;
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
        sum_left_hessian, lambda_l1, lambda_l2, path_smooth, left_count, parent_output);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
        sum_right_hessian, lambda_l1, lambda_l2, path_smooth, right_count, parent_output);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient,
        sum_left_hessian, lambda_l1, lambda_l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient,
        sum_right_hessian, lambda_l1, lambda_l2, right_output);
    }
  }
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING>
__device__ void FindBestSplitsForLeafKernelCategoricalInner_GlobalMemory(
  // input feature information
  const hist_t* feature_hist_ptr,
  // input task information
  const SplitFindTask* task,
  CUDARandom* cuda_random,
  // input config parameter values
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  // input parent node information
  const double parent_gain,
  const double sum_gradients,
  const double sum_hessians,
  const data_size_t num_data,
  const double parent_output,
  // buffer
  hist_t* hist_grad_buffer_ptr,
  hist_t* hist_hess_buffer_ptr,
  hist_t* hist_stat_buffer_ptr,
  data_size_t* hist_index_buffer_ptr,
  // output parameters
  CUDASplitInfo* cuda_best_split_info) {
  __shared__ double shared_gain_buffer[WARPSIZE];
  __shared__ bool shared_found_buffer[WARPSIZE];
  __shared__ uint32_t shared_thread_index_buffer[WARPSIZE];
  __shared__ uint32_t best_thread_index;
  const double cnt_factor = num_data / sum_hessians;
  const double min_gain_shift = parent_gain + min_gain_to_split;
  double l2 = lambda_l2;

  double local_gain = kMinScore;
  bool threshold_found = false;

  cuda_best_split_info->is_valid = false;

  __shared__ int rand_threshold;

  const int bin_start = 1 - task->mfb_offset;
  const int bin_end = task->num_bin - task->mfb_offset;
  int best_threshold = -1;
  const int threadIdx_x = static_cast<int>(threadIdx.x);
  if (task->is_one_hot) {
    if (USE_RAND && threadIdx.x == 0) {
      rand_threshold = 0;
      if (bin_end > bin_start) {
        rand_threshold = cuda_random->NextInt(bin_start, bin_end);
      }
    }
    __syncthreads();
    for (int bin = bin_start + threadIdx_x; bin < bin_end; bin += static_cast<int>(blockDim.x)) {
      const int bin_offset = (bin << 1);
      const hist_t grad = feature_hist_ptr[bin_offset];
      const hist_t hess = feature_hist_ptr[bin_offset + 1];
      data_size_t cnt =
            static_cast<data_size_t>(CUDARoundInt(hess * cnt_factor));
      if (cnt >= min_data_in_leaf && hess >= min_sum_hessian_in_leaf) {
        const data_size_t other_count = num_data - cnt;
        if (other_count >= min_data_in_leaf) {
          const double sum_other_hessian = sum_hessians - hess - kEpsilon;
          if (sum_other_hessian >= min_sum_hessian_in_leaf && (!USE_RAND || bin == rand_threshold)) {
            const double sum_other_gradient = sum_gradients - grad;
            double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
              sum_other_gradient, sum_other_hessian, grad,
              hess + kEpsilon, lambda_l1,
              l2, path_smooth, other_count, cnt, parent_output);
            if (current_gain > min_gain_shift) {
              best_threshold = bin;
              local_gain = current_gain - min_gain_shift;
              threshold_found = true;
            }
          }
        }
      }
    }
    __syncthreads();
    const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
    if (threadIdx_x == 0) {
      best_thread_index = result;
    }
    __syncthreads();
    if (threshold_found && threadIdx_x == best_thread_index) {
      cuda_best_split_info->is_valid = true;
      cuda_best_split_info->num_cat_threshold = 1;
      cuda_best_split_info->cat_threshold = new uint32_t[1];
      *(cuda_best_split_info->cat_threshold) = static_cast<uint32_t>(best_threshold);
      cuda_best_split_info->default_left = false;
      const int bin_offset = (best_threshold << 1);
      const hist_t sum_left_gradient = feature_hist_ptr[bin_offset];
      const hist_t sum_left_hessian = feature_hist_ptr[bin_offset + 1];
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
        sum_left_hessian, lambda_l1, l2, path_smooth, left_count, parent_output);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
        sum_right_hessian, lambda_l1, l2, path_smooth, right_count, parent_output);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient,
        sum_left_hessian, lambda_l1, l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient,
        sum_right_hessian, lambda_l1, l2, right_output);
    }
  } else {
    __shared__ uint16_t shared_mem_buffer_uint16[WARPSIZE];
    __shared__ int used_bin;
    l2 += cat_l2;
    uint16_t is_valid_bin = 0;
    int best_dir = 0;
    double best_sum_left_gradient = 0.0f;
    double best_sum_left_hessian = 0.0f;
    for (int bin = 0; bin < bin_end; bin += static_cast<int>(blockDim.x)) {
      if (bin >= bin_start) {
        const int bin_offset = (bin << 1);
        const double hess = feature_hist_ptr[bin_offset + 1];
        if (CUDARoundInt(hess * cnt_factor) >= cat_smooth) {
          const double grad = feature_hist_ptr[bin_offset];
          hist_stat_buffer_ptr[bin] = grad / (hess + cat_smooth);
          hist_index_buffer_ptr[bin] = threadIdx_x;
          is_valid_bin = 1;
        } else {
          hist_stat_buffer_ptr[bin] = kMaxScore;
          hist_index_buffer_ptr[bin] = -1;
        }
      }
    }
    __syncthreads();
    const int local_used_bin = ShuffleReduceSum<uint16_t>(is_valid_bin, shared_mem_buffer_uint16, blockDim.x);
    if (threadIdx_x == 0) {
      used_bin = local_used_bin;
    }
    __syncthreads();
    BitonicArgSortDevice<double, data_size_t, true, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 11>(
      hist_stat_buffer_ptr, hist_index_buffer_ptr, task->num_bin - task->mfb_offset);
    const int max_num_cat = min(max_cat_threshold, (used_bin + 1) / 2);
    if (USE_RAND) {
      rand_threshold = 0;
      const int max_threshold = max(min(max_num_cat, used_bin) - 1, 0);
      if (max_threshold > 0) {
        rand_threshold = cuda_random->NextInt(0, max_threshold);
      }
    }
    __syncthreads();

    // left to right
    for (int bin = static_cast<int>(threadIdx_x); bin < used_bin && bin < max_num_cat; bin += static_cast<int>(blockDim.x)) {
      const int bin_offset = (hist_index_buffer_ptr[bin] << 1);
      hist_grad_buffer_ptr[bin] = feature_hist_ptr[bin_offset];
      hist_hess_buffer_ptr[bin] = feature_hist_ptr[bin_offset + 1];
    }
    if (threadIdx_x == 0) {
      hist_hess_buffer_ptr[0] += kEpsilon;
    }
    __syncthreads();
    GlobalMemoryPrefixSum<double>(hist_grad_buffer_ptr, static_cast<size_t>(bin_end));
    __syncthreads();
    GlobalMemoryPrefixSum<double>(hist_hess_buffer_ptr, static_cast<size_t>(bin_end));
    for (int bin = static_cast<int>(threadIdx_x); bin < used_bin && bin < max_num_cat; bin += static_cast<int>(blockDim.x)) {
      const double sum_left_gradient = hist_grad_buffer_ptr[bin];
      const double sum_left_hessian = hist_hess_buffer_ptr[bin];
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf &&
        left_count >= min_data_per_group && right_count >= min_data_per_group) {
        double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1,
          l2, path_smooth, left_count, right_count, parent_output);
        // gain with split is worse than without split
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_found = true;
          best_dir = 1;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
          best_threshold = bin;
        }
      }
    }
    __syncthreads();

    // right to left
    for (int bin = static_cast<int>(threadIdx_x); bin < used_bin && bin < max_num_cat; bin += static_cast<int>(blockDim.x)) {
      const int bin_offset = (hist_index_buffer_ptr[used_bin - 1 - bin] << 1);
      hist_grad_buffer_ptr[bin] = feature_hist_ptr[bin_offset];
      hist_hess_buffer_ptr[bin] = feature_hist_ptr[bin_offset + 1];
    }
    if (threadIdx_x == 0) {
      hist_hess_buffer_ptr[0] += kEpsilon;
    }
    __syncthreads();
    GlobalMemoryPrefixSum<double>(hist_grad_buffer_ptr, static_cast<size_t>(bin_end));
    __syncthreads();
    GlobalMemoryPrefixSum<double>(hist_hess_buffer_ptr, static_cast<size_t>(bin_end));
    for (int bin = static_cast<int>(threadIdx_x); bin < used_bin && bin < max_num_cat; bin += static_cast<int>(blockDim.x)) {
      const double sum_left_gradient = hist_grad_buffer_ptr[bin];
      const double sum_left_hessian = hist_hess_buffer_ptr[bin];
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf &&
        left_count >= min_data_per_group && right_count >= min_data_per_group) {
        double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1,
          l2, path_smooth, left_count, right_count, parent_output);
        // gain with split is worse than without split
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_found = true;
          best_dir = -1;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
          best_threshold = bin;
        }
      }
    }
    __syncthreads();

    const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
    if (threadIdx_x == 0) {
      best_thread_index = result;
    }
    __syncthreads();
    if (threshold_found && threadIdx_x == best_thread_index) {
      cuda_best_split_info->is_valid = true;
      cuda_best_split_info->num_cat_threshold = best_threshold + 1;
      cuda_best_split_info->cat_threshold = new uint32_t[best_threshold + 1];
      cuda_best_split_info->gain = local_gain;
      if (best_dir == 1) {
        for (int i = 0; i < best_threshold + 1; ++i) {
          (cuda_best_split_info->cat_threshold)[i] = hist_index_buffer_ptr[i] + task->mfb_offset;
        }
      } else {
        for (int i = 0; i < best_threshold + 1; ++i) {
          (cuda_best_split_info->cat_threshold)[i] = hist_index_buffer_ptr[used_bin - 1 - i] + task->mfb_offset;
        }
      }
      cuda_best_split_info->default_left = false;
      const hist_t sum_left_gradient = best_sum_left_gradient;
      const hist_t sum_left_hessian = best_sum_left_hessian;
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
        sum_left_hessian, lambda_l1, l2, path_smooth, left_count, parent_output);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
        sum_right_hessian, lambda_l1, l2, path_smooth, right_count, parent_output);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient,
        sum_left_hessian, lambda_l1, l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient,
        sum_right_hessian, lambda_l1, l2, right_output);
    }
  }
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, bool IS_LARGER>
__global__ void FindBestSplitsForLeafKernel_GlobalMemory(
  // input feature information
  const int8_t* is_feature_used_bytree,
  // input task information
  const int num_tasks,
  const SplitFindTask* tasks,
  CUDARandom* cuda_randoms,
  // input leaf information
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const CUDALeafSplitsStruct* larger_leaf_splits,
  // input config parameter values
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  // output
  CUDASplitInfo* cuda_best_split_info,
  // global num data in leaf
  const data_size_t global_num_data_in_smaller_leaf,
  const data_size_t global_num_data_in_larger_leaf,
  // buffer
  hist_t* feature_hist_grad_buffer,
  hist_t* feature_hist_hess_buffer,
  hist_t* feature_hist_stat_buffer,
  data_size_t* feature_hist_index_buffer) {
  const unsigned int task_index = blockIdx.x;
  const SplitFindTask* task = tasks + task_index;
  const double parent_gain = IS_LARGER ? larger_leaf_splits->gain : smaller_leaf_splits->gain;
  const double sum_gradients = IS_LARGER ? larger_leaf_splits->sum_of_gradients : smaller_leaf_splits->sum_of_gradients;
  const double sum_hessians = (IS_LARGER ? larger_leaf_splits->sum_of_hessians : smaller_leaf_splits->sum_of_hessians) + 2 * kEpsilon;
  const data_size_t num_data = IS_LARGER ? global_num_data_in_larger_leaf : global_num_data_in_smaller_leaf;
  const double parent_output = IS_LARGER ? larger_leaf_splits->leaf_value : smaller_leaf_splits->leaf_value;
  const unsigned int output_offset = IS_LARGER ? (task_index + num_tasks) : task_index;
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  CUDARandom* cuda_random = USE_RAND ?
    (IS_LARGER ? cuda_randoms + task_index * 2 + 1: cuda_randoms + task_index * 2) : nullptr;
  if (is_feature_used_bytree[task->inner_feature_index]) {
    const uint32_t hist_offset = task->hist_offset;
    const hist_t* hist_ptr = (IS_LARGER ? larger_leaf_splits->hist_in_leaf : smaller_leaf_splits->hist_in_leaf) + hist_offset * 2;
    hist_t* hist_grad_buffer_ptr = feature_hist_grad_buffer + hist_offset * 2;
    hist_t* hist_hess_buffer_ptr = feature_hist_hess_buffer + hist_offset * 2;
    hist_t* hist_stat_buffer_ptr = feature_hist_stat_buffer + hist_offset * 2;
    data_size_t* hist_index_buffer_ptr = feature_hist_index_buffer + hist_offset * 2;
    if (task->is_categorical) {
      FindBestSplitsForLeafKernelCategoricalInner_GlobalMemory<USE_RAND, USE_L1, USE_SMOOTHING>(
        // input feature information
        hist_ptr,
        // input task information
        task,
        cuda_random,
        // input config parameter values
        lambda_l1,
        lambda_l2,
        path_smooth,
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        min_gain_to_split,
        cat_smooth,
        cat_l2,
        max_cat_threshold,
        min_data_per_group,
        // input parent node information
        parent_gain,
        sum_gradients,
        sum_hessians,
        num_data,
        parent_output,
        // buffer
        hist_grad_buffer_ptr,
        hist_hess_buffer_ptr,
        hist_stat_buffer_ptr,
        hist_index_buffer_ptr,
        // output parameters
        out);
    } else {
      if (!task->reverse) {
        FindBestSplitsForLeafKernelInner_GlobalMemory<USE_RAND, USE_L1, USE_SMOOTHING, false>(
          // input feature information
          hist_ptr,
          // input task information
          task,
          cuda_random,
          // input config parameter values
          lambda_l1,
          lambda_l2,
          path_smooth,
          min_data_in_leaf,
          min_sum_hessian_in_leaf,
          min_gain_to_split,
          // input parent node information
          parent_gain,
          sum_gradients,
          sum_hessians,
          num_data,
          parent_output,
          // output parameters
          out,
          // buffer
          hist_grad_buffer_ptr,
          hist_hess_buffer_ptr);
      } else {
        FindBestSplitsForLeafKernelInner_GlobalMemory<USE_RAND, USE_L1, USE_SMOOTHING, true>(
          // input feature information
          hist_ptr,
          // input task information
          task,
          cuda_random,
          // input config parameter values
          lambda_l1,
          lambda_l2,
          path_smooth,
          min_data_in_leaf,
          min_sum_hessian_in_leaf,
          min_gain_to_split,
          // input parent node information
          parent_gain,
          sum_gradients,
          sum_hessians,
          num_data,
          parent_output,
          // output parameters
          out,
          // buffer
          hist_grad_buffer_ptr,
          hist_hess_buffer_ptr);
      }
    }
  } else {
    out->is_valid = false;
  }
}

#define LaunchFindBestSplitsForLeafKernel_PARAMS \
  const CUDALeafSplitsStruct* smaller_leaf_splits, \
  const CUDALeafSplitsStruct* larger_leaf_splits, \
  const int smaller_leaf_index, \
  const int larger_leaf_index, \
  const bool is_smaller_leaf_valid, \
  const bool is_larger_leaf_valid, \
  const data_size_t global_num_data_in_smaller_leaf, \
  const data_size_t global_num_data_in_larger_leaf

#define LaunchFindBestSplitsForLeafKernel_ARGS \
  smaller_leaf_splits, \
  larger_leaf_splits, \
  smaller_leaf_index, \
  larger_leaf_index, \
  is_smaller_leaf_valid, \
  is_larger_leaf_valid, \
  global_num_data_in_smaller_leaf, \
  global_num_data_in_larger_leaf

#define FindBestSplitsForLeafKernel_ARGS \
    num_tasks_, \
    cuda_split_find_tasks_.RawData(), \
    cuda_randoms_.RawData(), \
    smaller_leaf_splits, \
    larger_leaf_splits, \
    min_data_in_leaf_, \
    min_sum_hessian_in_leaf_, \
    min_gain_to_split_, \
    lambda_l1_, \
    lambda_l2_, \
    path_smooth_, \
    cat_smooth_, \
    cat_l2_, \
    max_cat_threshold_, \
    min_data_per_group_, \
    cuda_best_split_info_.RawData(), \
    global_num_data_in_smaller_leaf, \
    global_num_data_in_larger_leaf

#define GlobalMemory_Buffer_ARGS \
  cuda_feature_hist_grad_buffer_.RawData(), \
  cuda_feature_hist_hess_buffer_.RawData(), \
  cuda_feature_hist_stat_buffer_.RawData(), \
  cuda_feature_hist_index_buffer_.RawData()

void CUDABestSplitFinder::LaunchFindBestSplitsForLeafKernel(LaunchFindBestSplitsForLeafKernel_PARAMS) {
  if (!is_smaller_leaf_valid && !is_larger_leaf_valid) {
    return;
  }
  if (!extra_trees_) {
    LaunchFindBestSplitsForLeafKernelInner0<false>(LaunchFindBestSplitsForLeafKernel_ARGS);
  } else {
    LaunchFindBestSplitsForLeafKernelInner0<true>(LaunchFindBestSplitsForLeafKernel_ARGS);
  }
}

template <bool USE_RAND>
void CUDABestSplitFinder::LaunchFindBestSplitsForLeafKernelInner0(LaunchFindBestSplitsForLeafKernel_PARAMS) {
  if (lambda_l1_ <= 0.0f) {
    LaunchFindBestSplitsForLeafKernelInner1<USE_RAND, false>(LaunchFindBestSplitsForLeafKernel_ARGS);
  } else {
    LaunchFindBestSplitsForLeafKernelInner1<USE_RAND, true>(LaunchFindBestSplitsForLeafKernel_ARGS);
  }
}

template <bool USE_RAND, bool USE_L1>
void CUDABestSplitFinder::LaunchFindBestSplitsForLeafKernelInner1(LaunchFindBestSplitsForLeafKernel_PARAMS) {
  if (!use_smoothing_) {
    LaunchFindBestSplitsForLeafKernelInner2<USE_RAND, USE_L1, false>(LaunchFindBestSplitsForLeafKernel_ARGS);
  } else {
    LaunchFindBestSplitsForLeafKernelInner2<USE_RAND, USE_L1, true>(LaunchFindBestSplitsForLeafKernel_ARGS);
  }
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING>
void CUDABestSplitFinder::LaunchFindBestSplitsForLeafKernelInner2(LaunchFindBestSplitsForLeafKernel_PARAMS) {
  if (ExaboostFP32GainEnabled()) {
    LaunchFindBestSplitsForLeafKernelInner3<USE_RAND, USE_L1, USE_SMOOTHING, float>(LaunchFindBestSplitsForLeafKernel_ARGS);
  } else {
    LaunchFindBestSplitsForLeafKernelInner3<USE_RAND, USE_L1, USE_SMOOTHING, double>(LaunchFindBestSplitsForLeafKernel_ARGS);
  }
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, typename GAIN_T>
void CUDABestSplitFinder::LaunchFindBestSplitsForLeafKernelInner3(LaunchFindBestSplitsForLeafKernel_PARAMS) {
  const int8_t* is_feature_used_by_smaller_node = cuda_is_feature_used_bytree_.RawData();
  const int8_t* is_feature_used_by_larger_node = cuda_is_feature_used_bytree_.RawData();
  if (select_features_by_node_) {
    is_feature_used_by_smaller_node = is_feature_used_by_smaller_node_.RawData();
    is_feature_used_by_larger_node = is_feature_used_by_larger_node_.RawData();
  }
  // Order each leaf's FindBestSplits after the histogram it reads, on its own stream,
  // instead of a device-wide sync: cudaStreamWaitEvent keeps the host from stalling and
  // lets the two leaves run concurrently. BOTH leaves wait on the subtract-done event
  // (recorded after fix + subtract on the constructor's stream): the smaller leaf's
  // histogram is only complete after FixHistogramKernel writes its most-frequent bin,
  // and that runs AFTER the construct-done event is recorded, so waiting on
  // construct_done alone raced with the fix kernel (nondeterministic splits when the
  // per-pair pipelines overlap).
  if (is_smaller_leaf_valid) {
    CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[0], hist_subtract_done_events_[active_hist_pipeline_], 0));
  }
  if (is_larger_leaf_valid) {
    CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[1], hist_subtract_done_events_[active_hist_pipeline_], 0));
  }
  if (!use_global_memory_) {
    if (is_smaller_leaf_valid) {
      FindBestSplitsForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, false, GAIN_T>
        <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>
        (is_feature_used_by_smaller_node, hist_fp32_, FindBestSplitsForLeafKernel_ARGS);
    }
    // No device sync here: the larger-leaf launch below waits on subtract_done_event via
    // its stream, and the smaller/larger leaves write disjoint cuda_best_split_info_ slots.
    if (is_larger_leaf_valid) {
      FindBestSplitsForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, true, GAIN_T>
        <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[1]>>>
        (is_feature_used_by_larger_node, hist_fp32_, FindBestSplitsForLeafKernel_ARGS);
    }
  } else {
    // Global-memory path (large-dataset fallback, not covered by the shared-memory
    // benchmarks): the construct/subtract waits above provide histogram visibility, but
    // keep the device sync because the smaller and larger launches share
    // cuda_feature_hist_grad/hess_buffer_ and must not run concurrently.
    if (is_smaller_leaf_valid) {
      FindBestSplitsForLeafKernel_GlobalMemory<USE_RAND, USE_L1, USE_SMOOTHING, false>
        <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>
        (is_feature_used_by_smaller_node, FindBestSplitsForLeafKernel_ARGS, GlobalMemory_Buffer_ARGS);
    }
    SynchronizeCUDADevice(__FILE__, __LINE__);
    if (is_larger_leaf_valid) {
      FindBestSplitsForLeafKernel_GlobalMemory<USE_RAND, USE_L1, USE_SMOOTHING, true>
        <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[1]>>>
        (is_feature_used_by_larger_node, FindBestSplitsForLeafKernel_ARGS, GlobalMemory_Buffer_ARGS);
    }
  }
}

#undef LaunchFindBestSplitsForLeafKernel_PARAMS
#undef FindBestSplitsForLeafKernel_ARGS
#undef GlobalMemory_Buffer_ARGS


#define LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS \
  const CUDALeafSplitsStruct* smaller_leaf_splits, \
  const CUDALeafSplitsStruct* larger_leaf_splits, \
  const int smaller_leaf_index, \
  const int larger_leaf_index, \
  const bool is_smaller_leaf_valid, \
  const bool is_larger_leaf_valid, \
  const score_t* grad_scale, \
  const score_t* hess_scale, \
  const uint8_t smaller_num_bits_in_histogram_bins, \
  const uint8_t larger_num_bits_in_histogram_bins, \
  const data_size_t global_num_data_in_smaller_leaf, \
  const data_size_t global_num_data_in_larger_leaf

#define LaunchFindBestSplitsDiscretizedForLeafKernel_ARGS \
  smaller_leaf_splits, \
  larger_leaf_splits, \
  smaller_leaf_index, \
  larger_leaf_index, \
  is_smaller_leaf_valid, \
  is_larger_leaf_valid, \
  grad_scale, \
  hess_scale, \
  smaller_num_bits_in_histogram_bins, \
  larger_num_bits_in_histogram_bins, \
  global_num_data_in_smaller_leaf, \
  global_num_data_in_larger_leaf

#define FindBestSplitsDiscretizedForLeafKernel_ARGS \
    cuda_is_feature_used_bytree_.RawData(), \
    num_tasks_, \
    cuda_split_find_tasks_.RawData(), \
    cuda_randoms_.RawData(), \
    smaller_leaf_splits, \
    larger_leaf_splits, \
    smaller_num_bits_in_histogram_bins, \
    larger_num_bits_in_histogram_bins, \
    min_data_in_leaf_, \
    min_sum_hessian_in_leaf_, \
    min_gain_to_split_, \
    lambda_l1_, \
    lambda_l2_, \
    path_smooth_, \
    cat_smooth_, \
    cat_l2_, \
    max_cat_threshold_, \
    min_data_per_group_, \
    max_cat_to_onehot_, \
    grad_scale, \
    hess_scale, \
    cuda_best_split_info_.RawData(), \
    global_num_data_in_smaller_leaf, \
    global_num_data_in_larger_leaf

void CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLeafKernel(LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS) {
  if (!is_smaller_leaf_valid && !is_larger_leaf_valid) {
    return;
  }
  if (!extra_trees_) {
    LaunchFindBestSplitsDiscretizedForLeafKernelInner0<false>(LaunchFindBestSplitsDiscretizedForLeafKernel_ARGS);
  } else {
    LaunchFindBestSplitsDiscretizedForLeafKernelInner0<true>(LaunchFindBestSplitsDiscretizedForLeafKernel_ARGS);
  }
}

template <bool USE_RAND>
void CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLeafKernelInner0(LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS) {
  if (lambda_l1_ <= 0.0f) {
    LaunchFindBestSplitsDiscretizedForLeafKernelInner1<USE_RAND, false>(LaunchFindBestSplitsDiscretizedForLeafKernel_ARGS);
  } else {
    LaunchFindBestSplitsDiscretizedForLeafKernelInner1<USE_RAND, true>(LaunchFindBestSplitsDiscretizedForLeafKernel_ARGS);
  }
}

template <bool USE_RAND, bool USE_L1>
void CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLeafKernelInner1(LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS) {
  if (!use_smoothing_) {
    LaunchFindBestSplitsDiscretizedForLeafKernelInner2<USE_RAND, USE_L1, false>(LaunchFindBestSplitsDiscretizedForLeafKernel_ARGS);
  } else {
    LaunchFindBestSplitsDiscretizedForLeafKernelInner2<USE_RAND, USE_L1, true>(LaunchFindBestSplitsDiscretizedForLeafKernel_ARGS);
  }
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING>
void CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLeafKernelInner2(LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS) {
  if (ExaboostFP32GainEnabled()) {
    LaunchFindBestSplitsDiscretizedForLeafKernelInner3<USE_RAND, USE_L1, USE_SMOOTHING, float>(LaunchFindBestSplitsDiscretizedForLeafKernel_ARGS);
  } else {
    LaunchFindBestSplitsDiscretizedForLeafKernelInner3<USE_RAND, USE_L1, USE_SMOOTHING, double>(LaunchFindBestSplitsDiscretizedForLeafKernel_ARGS);
  }
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, typename GAIN_T>
void CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLeafKernelInner3(LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS) {
  if (!use_global_memory_) {
    // Order each leaf after the histogram it reads via its own stream (see the
    // non-discretized path for the rationale — both leaves wait on subtract_done,
    // which is recorded after the fix kernel completes the smaller histogram);
    // no device sync between the two leaves.
    if (is_smaller_leaf_valid) {
      CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[0], hist_subtract_done_events_[active_hist_pipeline_], 0));
      FindBestSplitsDiscretizedForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, false, GAIN_T>
        <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>
        (FindBestSplitsDiscretizedForLeafKernel_ARGS);
    }
    if (is_larger_leaf_valid) {
      CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[1], hist_subtract_done_events_[active_hist_pipeline_], 0));
      FindBestSplitsDiscretizedForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, true, GAIN_T>
        <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[1]>>>
        (FindBestSplitsDiscretizedForLeafKernel_ARGS);
    }
  } else {
    // TODO(shiyu1994)
  }
}

#undef LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS
#undef LaunchFindBestSplitsDiscretizedForLeafKernel_ARGS
#undef FindBestSplitsDiscretizedForLeafKernel_ARGS


// ----- Batched per-level find + sync (hybrid growth) -------------------------------
// One find launch covers every (pair, leaf, task) of a level: blockIdx.x = task,
// blockIdx.y = pair, blockIdx.z = 0 (smaller leaf) / 1 (larger leaf). Each pair
// writes its own region of cuda_best_split_info (2 * num_tasks slots at offset
// pair * 2 * num_tasks) so pairs never conflict. Only the template combination the
// benchmarks exercise is instantiated (no rand / L1 / smoothing); other configs
// fall back to the per-pair path (see SupportsBatchedLevel).

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, typename GAIN_T>
__global__ void FindBestSplitsForLevelKernel(
  const int8_t* is_feature_used_bytree,
  const bool fp32_hist,
  const int num_tasks,
  const SplitFindTask* tasks,
  const int* used_task_indices,
  CUDARandom* cuda_randoms,
  const CUDAHybridPairDescriptor* pair_descs,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  CUDASplitInfo* cuda_best_split_info,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2: the graph-frozen grid is a pow2 bucket of the live pair count;
  // blocks beyond the live range exit before any read (stale descriptors)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  const unsigned int pair_index = blockIdx.y;
  const bool is_larger = (blockIdx.z == 1);
  const CUDAHybridPairDescriptor* desc = pair_descs + pair_index;
  // desc->{smaller,larger}_valid carries the HOST-known gates (max_depth); the
  // min_data/min_hessian gates are evaluated on-device from the leaf struct so
  // the speculative single-sync flow can enqueue this kernel before the child
  // statistics are read back. In the classic flow the host flags already include
  // the identical data/hessian conditions, so the device check is a no-op.
  if (is_larger ? !desc->larger_valid : !desc->smaller_valid) {
    return;
  }
  const CUDALeafSplitsStruct* leaf_splits = is_larger ? desc->larger_struct : desc->smaller_struct;
  const data_size_t num_data = leaf_splits->num_data_in_leaf;
  const double leaf_sum_hessians = leaf_splits->sum_of_hessians;
  if (leaf_splits->leaf_index < 0 || num_data <= min_data_in_leaf ||
      leaf_sum_hessians <= min_sum_hessian_in_leaf) {
    return;
  }
  // feature-sampled trees launch one block per USED task only; the output slot
  // keeps the ORIGINAL task index (unused slots are masked out by the sync)
  const unsigned int task_index = used_task_indices == nullptr ?
    blockIdx.x : static_cast<unsigned int>(used_task_indices[blockIdx.x]);
  const SplitFindTask* task = tasks + task_index;
  const double parent_gain = leaf_splits->gain;
  const double sum_gradients = leaf_splits->sum_of_gradients;
  const double sum_hessians = leaf_sum_hessians + 2 * kEpsilon;
  const double parent_output = leaf_splits->leaf_value;
  const unsigned int output_offset = pair_index * (2 * static_cast<unsigned int>(num_tasks)) +
    (is_larger ? task_index + num_tasks : task_index);
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  CUDARandom* cuda_random = USE_RAND ?
    (is_larger ? cuda_randoms + task_index * 2 + 1 : cuda_randoms + task_index * 2) : nullptr;
  if (is_feature_used_bytree[task->inner_feature_index]) {
    // no categorical branch: the batched path is gated on !has_categorical_feature_
    const hist_t* hist_ptr = FeatureHistPtr(leaf_splits->hist_in_leaf, task->hist_offset, fp32_hist);
    if (!task->reverse) {
      FindBestSplitsForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, false, GAIN_T>(
        hist_ptr, fp32_hist, task, cuda_random,
        lambda_l1, lambda_l2, path_smooth,
        min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
        parent_gain, sum_gradients, sum_hessians, num_data, parent_output,
        out);
    } else {
      FindBestSplitsForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, true, GAIN_T>(
        hist_ptr, fp32_hist, task, cuda_random,
        lambda_l1, lambda_l2, path_smooth,
        min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
        parent_gain, sum_gradients, sum_hessians, num_data, parent_output,
        out);
    }
  } else {
    out->is_valid = false;
  }
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, typename GAIN_T>
__global__ void FindBestSplitsDiscretizedForLevelKernel(
  const int8_t* is_feature_used_bytree,
  const int num_tasks,
  const SplitFindTask* tasks,
  const int* used_task_indices,
  CUDARandom* cuda_randoms,
  const CUDAHybridPairDescriptor* pair_descs,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const score_t* grad_scale,
  const score_t* hess_scale,
  CUDASplitInfo* cuda_best_split_info,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2: the graph-frozen grid is a pow2 bucket of the live pair count;
  // blocks beyond the live range exit before any read (stale descriptors)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  const unsigned int pair_index = blockIdx.y;
  const bool is_larger = (blockIdx.z == 1);
  const CUDAHybridPairDescriptor* desc = pair_descs + pair_index;
  if (is_larger ? !desc->larger_valid : !desc->smaller_valid) {
    return;
  }
  const CUDALeafSplitsStruct* leaf_splits = is_larger ? desc->larger_struct : desc->smaller_struct;
  // host-launched (two-sync) path: the host-written descriptor carries the
  // exact leaf size and the validity flags already include the min_data /
  // min_hessian gates (bit-for-bit the previous behavior). Graph loop: the
  // controller-written flags carry only the max_depth gate, so the size comes
  // from the child struct the batched apply kernels wrote and the data /
  // hessian gates are evaluated here (mirror of the non-quantized level kernel)
  const data_size_t num_data = HybridGraphActive(gstate) ?
    leaf_splits->num_data_in_leaf :
    (is_larger ? desc->num_data_in_larger_leaf : desc->num_data_in_smaller_leaf);
  if (HybridGraphActive(gstate) &&
      (leaf_splits->leaf_index < 0 || num_data <= min_data_in_leaf ||
       leaf_splits->sum_of_hessians <= min_sum_hessian_in_leaf)) {
    return;
  }
  const unsigned int task_index = used_task_indices == nullptr ?
    blockIdx.x : static_cast<unsigned int>(used_task_indices[blockIdx.x]);
  const SplitFindTask* task = tasks + task_index;
  const double parent_gain = leaf_splits->gain;
  const int64_t sum_gradients_hessians = leaf_splits->sum_of_gradients_hessians;
  const double parent_output = leaf_splits->leaf_value;
  const unsigned int output_offset = pair_index * (2 * static_cast<unsigned int>(num_tasks)) +
    (is_larger ? task_index + num_tasks : task_index);
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  CUDARandom* cuda_random = USE_RAND ?
    (is_larger ? cuda_randoms + task_index * 2 + 1 : cuda_randoms + task_index * 2) : nullptr;
  // per-leaf histogram bit width: host descriptor on the host-launched path,
  // device-derived from the exact leaf count (host thresholds) inside the graph
  const uint8_t leaf_num_bits = HybridGraphActive(gstate) ?
    HybridGraphQuantHistBits(gstate, num_data) :
    (is_larger ? desc->larger_num_bits : desc->smaller_num_bits);
  const bool use_16bit_bin = leaf_num_bits <= 16;
  if (is_feature_used_bytree[task->inner_feature_index]) {
    // no categorical branch: quantized training rejects categorical features
    if (!task->reverse) {
      if (use_16bit_bin) {
        const int32_t* hist_ptr =
          reinterpret_cast<const int32_t*>(leaf_splits->hist_in_leaf) + task->hist_offset;
        FindBestSplitsDiscretizedForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, false, int32_t, int32_t, true, true, GAIN_T>(
          hist_ptr, task, cuda_random,
          lambda_l1, lambda_l2, path_smooth,
          min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
          parent_gain, sum_gradients_hessians, num_data, parent_output,
          *grad_scale, *hess_scale, out);
      } else {
        // 32-bit histogram is int64-per-bin; read as int64 (see the per-pair kernel)
        const int64_t* hist_ptr =
          reinterpret_cast<const int64_t*>(leaf_splits->hist_in_leaf) + task->hist_offset;
        FindBestSplitsDiscretizedForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, false, int64_t, int64_t, false, false, GAIN_T>(
          hist_ptr, task, cuda_random,
          lambda_l1, lambda_l2, path_smooth,
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
          lambda_l1, lambda_l2, path_smooth,
          min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
          parent_gain, sum_gradients_hessians, num_data, parent_output,
          *grad_scale, *hess_scale, out);
      } else {
        const int64_t* hist_ptr =
          reinterpret_cast<const int64_t*>(leaf_splits->hist_in_leaf) + task->hist_offset;
        FindBestSplitsDiscretizedForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, true, int64_t, int64_t, false, false, GAIN_T>(
          hist_ptr, task, cuda_random,
          lambda_l1, lambda_l2, path_smooth,
          min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
          parent_gain, sum_gradients_hessians, num_data, parent_output,
          *grad_scale, *hess_scale, out);
      }
    }
  } else {
    out->is_valid = false;
  }
}

// Batched reduction over the per-pair task regions: blockIdx.x = 0 (smaller) / 1
// (larger), blockIdx.y = pair, blockIdx.z = task block (num_tasks may exceed one
// sync block for wide datasets). Each block reduces its slice of one leaf's
// 2 * num_tasks region exactly like SyncBestSplitForLeafKernel (same block size,
// same task slicing, same reduction order, hence bit-identical results) and
// writes that leaf's slot of the per-leaf best-split cache (block z writes slot
// leaf + z * num_leaves, exactly like the per-pair kernel); leaves of different
// pairs are disjoint, so no conflicts. Multi-block leaves are merged afterwards
// by SyncBestSplitForLevelKernelAllBlocks (the per-pair AllBlocks replica).
__global__ void SyncBestSplitForLevelKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  CUDASplitInfo* cuda_leaf_best_split_info,
  const SplitFindTask* tasks,
  const int8_t* is_feature_used_bytree,
  const CUDASplitInfo* cuda_best_split_info,
  const int num_tasks,
  const int num_leaves,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2 idle-block guard (pow2-frozen grid; see the find kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  __shared__ double shared_gain_buffer[WARPSIZE];
  __shared__ bool shared_found_buffer[WARPSIZE];
  __shared__ uint32_t shared_thread_index_buffer[WARPSIZE];
  const unsigned int pair_index = blockIdx.y;
  const bool is_larger = (blockIdx.x == 1);
  const unsigned int leaf_block_index = blockIdx.z;
  const CUDAHybridPairDescriptor* desc = pair_descs + pair_index;
  // leaf index and validity mirror the find kernel: the leaf index comes from
  // the struct (written by the batched apply kernels; equal to the host value in
  // the classic flow) and the host validity flags are combined (bitwise AND) with the on-device
  // min_data/min_hessian gates so the speculative single-sync flow needs no
  // host-side child statistics.
  const CUDALeafSplitsStruct* leaf_splits = is_larger ? desc->larger_struct : desc->smaller_struct;
  const int leaf_index = leaf_splits->leaf_index;
  if (leaf_index < 0) {
    return;
  }
  bool leaf_valid = is_larger ? (desc->larger_valid != 0) : (desc->smaller_valid != 0);
  leaf_valid = leaf_valid && leaf_splits->num_data_in_leaf > min_data_in_leaf &&
    leaf_splits->sum_of_hessians > min_sum_hessian_in_leaf;
  if (!leaf_valid) {
    // mirror SetInvalidLeafSplitInfoKernel of the per-pair path (block-slot
    // copies for z >= 1 stay stale, so the AllBlocks merge skips invalid leaves)
    if (threadIdx.x == 0 && leaf_block_index == 0) {
      cuda_leaf_best_split_info[leaf_index].is_valid = false;
    }
    return;
  }
  const uint32_t threadIdx_x = threadIdx.x;
  const CUDASplitInfo* pair_split_info = cuda_best_split_info +
    static_cast<size_t>(pair_index) * (2 * static_cast<size_t>(num_tasks));
  bool best_found = false;
  double best_gain = kMinScore;
  uint32_t shared_read_index = 0;
  const int task_index = static_cast<int>(leaf_block_index * blockDim.x + threadIdx_x);
  const uint32_t read_index = is_larger ? static_cast<uint32_t>(task_index + num_tasks) : static_cast<uint32_t>(task_index);
  // Feature-sampled trees launch the batched find over USED tasks only, leaving
  // unused tasks' output slots stale; mask them to not-found here. This is
  // bit-identical to the per-pair reduction, where those lanes hold the
  // is_valid=false slots the (full-grid) find would have written: a not-found
  // lane never propagates its gain/read-index through ReduceBestGain.
  if (task_index < num_tasks &&
      (is_feature_used_bytree == nullptr ||
       is_feature_used_bytree[tasks[task_index].inner_feature_index])) {
    best_found = pair_split_info[read_index].is_valid;
    best_gain = pair_split_info[read_index].gain;
    shared_read_index = read_index;
  }
  __syncthreads();
  const uint32_t best_read_index = ReduceBestGain(best_gain, best_found, shared_read_index,
      shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
  if (threadIdx.x == 0) {
    CUDASplitInfo* cuda_split_info = cuda_leaf_best_split_info +
      static_cast<unsigned int>(leaf_index) + leaf_block_index * static_cast<unsigned int>(num_leaves);
    const int best_task_index = best_read_index >= static_cast<uint32_t>(num_tasks) ?
      static_cast<int>(best_read_index) - num_tasks : static_cast<int>(best_read_index);
    // when nothing was found the reduction returns thread 0's own lane, whose
    // slot may be stale under task compaction: re-apply the mask before reading
    const bool best_task_used = is_feature_used_bytree == nullptr ||
      is_feature_used_bytree[tasks[best_task_index].inner_feature_index];
    const CUDASplitInfo* best_split_info = pair_split_info + best_read_index;
    if (best_task_used && best_split_info->is_valid) {
      *cuda_split_info = *best_split_info;
      cuda_split_info->inner_feature_index = tasks[best_task_index].inner_feature_index;
      cuda_split_info->is_valid = true;
    } else {
      cuda_split_info->gain = kMinScore;
      cuda_split_info->is_valid = false;
    }
  }
}

// Cross-block merge for wide datasets (num_tasks > one sync block): one block per
// (leaf role, pair) sequentially folds block slots 1..num_blocks_per_leaf-1 into
// slot 0 with the exact comparison (and hence tie-break order: the LOWEST task
// block wins ties) of the per-pair SyncBestSplitForLeafKernelAllBlocks.
__global__ void SyncBestSplitForLevelKernelAllBlocks(
  const CUDAHybridPairDescriptor* pair_descs,
  CUDASplitInfo* cuda_leaf_best_split_info,
  const unsigned int num_blocks_per_leaf,
  const int num_leaves,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2 idle-block guard (pow2-frozen grid; see the find kernel)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  const unsigned int pair_index = blockIdx.y;
  const bool is_larger = (blockIdx.x == 1);
  const CUDAHybridPairDescriptor* desc = pair_descs + pair_index;
  const CUDALeafSplitsStruct* leaf_splits = is_larger ? desc->larger_struct : desc->smaller_struct;
  const int leaf_index = leaf_splits->leaf_index;
  if (leaf_index < 0) {
    return;
  }
  bool leaf_valid = is_larger ? (desc->larger_valid != 0) : (desc->smaller_valid != 0);
  leaf_valid = leaf_valid && leaf_splits->num_data_in_leaf > min_data_in_leaf &&
    leaf_splits->sum_of_hessians > min_sum_hessian_in_leaf;
  if (!leaf_valid) {
    // the sync kernel's block-slot copies were skipped for this leaf
    return;
  }
  CUDASplitInfo* leaf_split_info = cuda_leaf_best_split_info + leaf_index;
  for (unsigned int block_index = 1; block_index < num_blocks_per_leaf; ++block_index) {
    const unsigned int leaf_read_pos = static_cast<unsigned int>(leaf_index) +
      block_index * static_cast<unsigned int>(num_leaves);
    const CUDASplitInfo* other_split_info = cuda_leaf_best_split_info + leaf_read_pos;
    if ((other_split_info->is_valid && leaf_split_info->is_valid &&
      other_split_info->gain > leaf_split_info->gain) ||
        (!leaf_split_info->is_valid && other_split_info->is_valid)) {
      *leaf_split_info = *other_split_info;
    }
  }
}

void CUDABestSplitFinder::LaunchFindBestSplitsForLevelKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const CUDAHybridGraphLoopStateOpt gstate) {
  const bool compact_tasks = num_used_tasks_ > 0 && num_used_tasks_ < num_tasks_;
  if (num_used_tasks_ == 0) {
    return;  // no usable feature this tree; the sync masks every lane not-found
  }
  dim3 grid_dim(compact_tasks ? num_used_tasks_ : num_tasks_, num_pairs, 2);
  #define FindBestSplitsForLevelKernel_ARGS \
      cuda_is_feature_used_bytree_.RawData(), \
      hist_fp32_, \
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
      cuda_best_split_info_.RawData(), \
      gstate
  if (ExaboostFP32GainEnabled()) {
    FindBestSplitsForLevelKernel<false, false, false, float>
      <<<grid_dim, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>(
        FindBestSplitsForLevelKernel_ARGS);
  } else {
    FindBestSplitsForLevelKernel<false, false, false, double>
      <<<grid_dim, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>(
        FindBestSplitsForLevelKernel_ARGS);
  }
  #undef FindBestSplitsForLevelKernel_ARGS
}

void CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLevelKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const score_t* grad_scale,
  const score_t* hess_scale,
  const CUDAHybridGraphLoopStateOpt gstate) {
  const bool compact_tasks = num_used_tasks_ > 0 && num_used_tasks_ < num_tasks_;
  if (num_used_tasks_ == 0) {
    return;  // no usable feature this tree; the sync masks every lane not-found
  }
  dim3 grid_dim(compact_tasks ? num_used_tasks_ : num_tasks_, num_pairs, 2);
  #define FindBestSplitsDiscretizedForLevelKernel_ARGS \
      cuda_is_feature_used_bytree_.RawData(), \
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
      grad_scale, \
      hess_scale, \
      cuda_best_split_info_.RawData(), \
      gstate
  if (ExaboostFP32GainEnabled()) {
    FindBestSplitsDiscretizedForLevelKernel<false, false, false, float>
      <<<grid_dim, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>(
        FindBestSplitsDiscretizedForLevelKernel_ARGS);
  } else {
    FindBestSplitsDiscretizedForLevelKernel<false, false, false, double>
      <<<grid_dim, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>(
        FindBestSplitsDiscretizedForLevelKernel_ARGS);
  }
  #undef FindBestSplitsDiscretizedForLevelKernel_ARGS
}

#ifdef EXABOOST_HYBRID_GRAPH_SUPPORTED
void CUDABestSplitFinder::CaptureHybridGraphFindKernels(
    const CUDAHybridPairDescriptor* pair_descs,
    const score_t* grad_scale,
    const score_t* hess_scale,
    const CUDAHybridGraphLoopState* gstate,
    std::vector<cudaGraphNode_t>* nodes,
    std::vector<int>* roles,
    std::vector<int>* role_static_x) {
  // graphs L1 body capture: find + sync with placeholder pair counts; the wait
  // on the histogram phase's subtract event becomes a graph edge, exactly
  // mirroring the host flow's ordering. Quantized training (grad_scale !=
  // nullptr) captures the discretized find variant, mirroring
  // FindBestSplitsForLevel's host dispatch.
  CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[0], hist_subtract_done_events_[0], 0));
  if (grad_scale != nullptr && hess_scale != nullptr) {
    LaunchFindBestSplitsDiscretizedForLevelKernel(pair_descs, 1, grad_scale, hess_scale, gstate);
  } else {
    LaunchFindBestSplitsForLevelKernel(pair_descs, 1, gstate);
  }
  if (!AppendCapturedNode(cuda_streams_[0], nodes)) return;
  roles->push_back(kHybridGraphNodeFind);
  role_static_x->push_back(0);
  // mirror of LaunchSyncBestSplitForLevelKernel, one collected node per launch
  const int num_blocks_per_leaf = (num_tasks_ + NUM_TASKS_PER_SYNC_BLOCK - 1) / NUM_TASKS_PER_SYNC_BLOCK;
  const bool compact_tasks = num_used_tasks_ < num_tasks_;
  dim3 grid_dim(2, 1, num_blocks_per_leaf);
  SyncBestSplitForLevelKernel<<<grid_dim, NUM_TASKS_PER_SYNC_BLOCK, 0, cuda_streams_[0]>>>(
    pair_descs,
    cuda_leaf_best_split_info_.RawData(),
    cuda_split_find_tasks_.RawData(),
    compact_tasks ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
    cuda_best_split_info_.RawData(),
    num_tasks_,
    num_leaves_,
    min_data_in_leaf_,
    min_sum_hessian_in_leaf_,
    gstate);
  if (!AppendCapturedNode(cuda_streams_[0], nodes)) return;
  roles->push_back(kHybridGraphNodeSyncLevel);
  role_static_x->push_back(num_blocks_per_leaf);
  if (num_blocks_per_leaf > 1) {
    dim3 merge_grid_dim(2, 1);
    SyncBestSplitForLevelKernelAllBlocks<<<merge_grid_dim, 1, 0, cuda_streams_[0]>>>(
      pair_descs,
      cuda_leaf_best_split_info_.RawData(),
      static_cast<unsigned int>(num_blocks_per_leaf),
      num_leaves_,
      min_data_in_leaf_,
      min_sum_hessian_in_leaf_,
      gstate);
    if (!AppendCapturedNode(cuda_streams_[0], nodes)) return;
    roles->push_back(kHybridGraphNodeSyncAllBlocks);
    role_static_x->push_back(0);
  }
}
#endif  // EXABOOST_HYBRID_GRAPH_SUPPORTED

void CUDABestSplitFinder::LaunchSyncBestSplitForLevelKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs) {
  const int num_blocks_per_leaf = (num_tasks_ + NUM_TASKS_PER_SYNC_BLOCK - 1) / NUM_TASKS_PER_SYNC_BLOCK;
  const bool compact_tasks = num_used_tasks_ < num_tasks_;
  dim3 grid_dim(2, num_pairs, num_blocks_per_leaf);
  SyncBestSplitForLevelKernel<<<grid_dim, NUM_TASKS_PER_SYNC_BLOCK, 0, cuda_streams_[0]>>>(
    pair_descs,
    cuda_leaf_best_split_info_.RawData(),
    cuda_split_find_tasks_.RawData(),
    compact_tasks ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr,
    cuda_best_split_info_.RawData(),
    num_tasks_,
    num_leaves_,
    min_data_in_leaf_,
    min_sum_hessian_in_leaf_,
    nullptr);
  if (num_blocks_per_leaf > 1) {
    // stream-ordered after the sync kernel above; same stream as the batched
    // find, so no extra synchronization is needed
    dim3 merge_grid_dim(2, num_pairs);
    SyncBestSplitForLevelKernelAllBlocks<<<merge_grid_dim, 1, 0, cuda_streams_[0]>>>(
      pair_descs,
      cuda_leaf_best_split_info_.RawData(),
      static_cast<unsigned int>(num_blocks_per_leaf),
      num_leaves_,
      min_data_in_leaf_,
      min_sum_hessian_in_leaf_,
      nullptr);
  }
}


// ----- Forced-split support -------------------------------------------------------
// Mirrors FeatureHistogram::GatherInfoForThresholdNumericalInner. Single-threaded so
// the bin accumulation order matches the CPU loop and results are bit-identical.
__global__ void ComputeForcedSplitKernel(
  const CUDALeafSplitsStruct* leaf_splits,
  const SplitFindTask* task,
  const uint32_t threshold,
  const data_size_t num_data,
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const double min_gain_to_split,
  CUDASplitInfo* out) {
  const double sum_gradients = leaf_splits->sum_of_gradients;
  const double sum_hessians = leaf_splits->sum_of_hessians + 2 * kEpsilon;
  const double parent_output = leaf_splits->leaf_value;
  const hist_t* feature_hist_ptr = leaf_splits->hist_in_leaf + task->hist_offset * 2;
  out->num_cat_threshold = 0;
  const double gain_shift = CUDALeafSplits::GetLeafGainGivenOutput<true>(
      sum_gradients, sum_hessians, lambda_l1, lambda_l2, parent_output);
  const double min_gain_shift = gain_shift + min_gain_to_split;
  const int8_t offset = static_cast<int8_t>(task->mfb_offset);
  const bool use_na_as_missing = task->na_as_missing;
  const bool skip_default_bin = task->skip_default_bin;

  double sum_right_gradient = 0.0f;
  double sum_right_hessian = kEpsilon;
  data_size_t right_count = 0;
  int t = task->num_bin - 1 - offset - (use_na_as_missing ? 1 : 0);
  const int t_end = 1 - offset;
  const double cnt_factor = num_data / sum_hessians;
  for (; t >= t_end; --t) {
    if (static_cast<uint32_t>(t + offset) <= threshold) {
      break;
    }
    if (skip_default_bin && (t + offset) == static_cast<int>(task->default_bin)) {
      continue;
    }
    const unsigned int bin_offset = static_cast<unsigned int>(t) << 1;
    const double grad = feature_hist_ptr[bin_offset];
    const double hess = feature_hist_ptr[bin_offset + 1];
    const data_size_t cnt = static_cast<data_size_t>(CUDARoundInt(hess * cnt_factor));
    sum_right_gradient += grad;
    sum_right_hessian += hess;
    right_count += cnt;
  }
  const double sum_left_gradient = sum_gradients - sum_right_gradient;
  const double sum_left_hessian = sum_hessians - sum_right_hessian;
  const data_size_t left_count = num_data - right_count;
  const double current_gain =
    CUDALeafSplits::GetLeafGain<true, false>(sum_left_gradient, sum_left_hessian,
      lambda_l1, lambda_l2, path_smooth, left_count, parent_output) +
    CUDALeafSplits::GetLeafGain<true, false>(sum_right_gradient, sum_right_hessian,
      lambda_l1, lambda_l2, path_smooth, right_count, parent_output);
  if (isnan(current_gain) || current_gain <= min_gain_shift) {
    out->is_valid = false;
    out->gain = kMinScore;
    return;
  }
  out->is_valid = true;
  out->threshold = threshold;
  out->default_left = true;
  out->gain = current_gain - min_gain_shift;
  out->left_sum_gradients = sum_left_gradient;
  out->left_sum_hessians = sum_left_hessian - kEpsilon;
  out->left_count = left_count;
  out->right_sum_gradients = sum_right_gradient;
  out->right_sum_hessians = sum_right_hessian - kEpsilon;
  out->right_count = num_data - left_count;
  const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(
      sum_left_gradient, sum_left_hessian, lambda_l1, lambda_l2, path_smooth, left_count, parent_output);
  const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(
      sum_right_gradient, sum_right_hessian, lambda_l1, lambda_l2, path_smooth, right_count, parent_output);
  out->left_value = left_output;
  out->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<true>(
      sum_left_gradient, sum_left_hessian - kEpsilon, lambda_l1, lambda_l2, left_output);
  out->right_value = right_output;
  out->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<true>(
      sum_right_gradient, sum_right_hessian - kEpsilon, lambda_l1, lambda_l2, right_output);
}

__global__ void InvalidateLeafBestSplitKernel(
  const int left_leaf_index, const int right_leaf_index,
  CUDASplitInfo* cuda_leaf_best_split_info) {
  if (threadIdx.x == 0) {
    if (left_leaf_index >= 0) {
      cuda_leaf_best_split_info[left_leaf_index].is_valid = false;
    }
    if (right_leaf_index >= 0) {
      cuda_leaf_best_split_info[right_leaf_index].is_valid = false;
    }
  }
}

void CUDABestSplitFinder::LaunchComputeForcedSplitKernel(
  const CUDALeafSplitsStruct* leaf_splits,
  const SplitFindTask* device_task,
  const uint32_t threshold,
  const data_size_t num_data_in_leaf,
  CUDASplitInfo* out) {
  ComputeForcedSplitKernel<<<1, 1>>>(
    leaf_splits, device_task, threshold, num_data_in_leaf,
    lambda_l1_, lambda_l2_, path_smooth_, min_gain_to_split_, out);
}

void CUDABestSplitFinder::LaunchInvalidateLeafBestSplitKernel(const int left_leaf_index, const int right_leaf_index) {
  InvalidateLeafBestSplitKernel<<<1, 1>>>(left_leaf_index, right_leaf_index, cuda_leaf_best_split_info_.RawData());
}

__device__ void ReduceBestSplit(bool* found, double* gain, uint32_t* shared_read_index,
  uint32_t num_features_aligned) {
  const uint32_t threadIdx_x = threadIdx.x;
  for (unsigned int s = 1; s < num_features_aligned; s <<= 1) {
    if (threadIdx_x % (2 * s) == 0 && (threadIdx_x + s) < num_features_aligned) {
      const uint32_t pos_to_compare = threadIdx_x + s;
      if ((!found[threadIdx_x] && found[pos_to_compare]) ||
        (found[threadIdx_x] && found[pos_to_compare] && gain[threadIdx_x] < gain[pos_to_compare])) {
        found[threadIdx_x] = found[pos_to_compare];
        gain[threadIdx_x] = gain[pos_to_compare];
        shared_read_index[threadIdx_x] = shared_read_index[pos_to_compare];
      }
    }
    __syncthreads();
  }
}

__global__ void SyncBestSplitForLeafKernel(const int smaller_leaf_index, const int larger_leaf_index,
  CUDASplitInfo* cuda_leaf_best_split_info,
  // input parameters
  const SplitFindTask* tasks,
  const CUDASplitInfo* cuda_best_split_info,
  const int num_tasks,
  const int num_tasks_aligned,
  const int num_blocks_per_leaf,
  const bool larger_only,
  const int num_leaves) {
  __shared__ double shared_gain_buffer[WARPSIZE];
  __shared__ bool shared_found_buffer[WARPSIZE];
  __shared__ uint32_t shared_thread_index_buffer[WARPSIZE];
  const uint32_t threadIdx_x = threadIdx.x;
  const uint32_t blockIdx_x = blockIdx.x;

  bool best_found = false;
  double best_gain = kMinScore;
  uint32_t shared_read_index = 0;

  const bool is_smaller = (blockIdx_x < static_cast<unsigned int>(num_blocks_per_leaf) && !larger_only);
  const uint32_t leaf_block_index = (is_smaller || larger_only) ? blockIdx_x : (blockIdx_x - static_cast<unsigned int>(num_blocks_per_leaf));
  const int task_index = static_cast<int>(leaf_block_index * blockDim.x + threadIdx_x);
  const uint32_t read_index = is_smaller ? static_cast<uint32_t>(task_index) : static_cast<uint32_t>(task_index + num_tasks);
  if (task_index < num_tasks) {
    best_found = cuda_best_split_info[read_index].is_valid;
    best_gain = cuda_best_split_info[read_index].gain;
    shared_read_index = read_index;
  } else {
    best_found = false;
  }

  __syncthreads();
  const uint32_t best_read_index = ReduceBestGain(best_gain, best_found, shared_read_index,
      shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
  if (threadIdx.x == 0) {
    const int leaf_index_ref = is_smaller ? smaller_leaf_index : larger_leaf_index;
    const unsigned buffer_write_pos = static_cast<unsigned int>(leaf_index_ref) + leaf_block_index * num_leaves;
    CUDASplitInfo* cuda_split_info = cuda_leaf_best_split_info + buffer_write_pos;
    const CUDASplitInfo* best_split_info = cuda_best_split_info + best_read_index;
    if (best_split_info->is_valid) {
      *cuda_split_info = *best_split_info;
      cuda_split_info->inner_feature_index = is_smaller ? tasks[best_read_index].inner_feature_index :
        tasks[static_cast<int>(best_read_index) - num_tasks].inner_feature_index;
      cuda_split_info->is_valid = true;
    } else {
      cuda_split_info->gain = kMinScore;
      cuda_split_info->is_valid = false;
    }
  }
}

__global__ void SyncBestSplitForLeafKernelAllBlocks(
  const int smaller_leaf_index,
  const int larger_leaf_index,
  const unsigned int num_blocks_per_leaf,
  const int num_leaves,
  CUDASplitInfo* cuda_leaf_best_split_info,
  const bool larger_only) {
  if (!larger_only) {
    if (blockIdx.x == 0) {
      CUDASplitInfo* smaller_leaf_split_info = cuda_leaf_best_split_info + smaller_leaf_index;
      for (unsigned int block_index = 1; block_index < num_blocks_per_leaf; ++block_index) {
        const unsigned int leaf_read_pos = static_cast<unsigned int>(smaller_leaf_index) + block_index * static_cast<unsigned int>(num_leaves);
        const CUDASplitInfo* other_split_info = cuda_leaf_best_split_info + leaf_read_pos;
        if ((other_split_info->is_valid && smaller_leaf_split_info->is_valid &&
          other_split_info->gain > smaller_leaf_split_info->gain) ||
            (!smaller_leaf_split_info->is_valid && other_split_info->is_valid)) {
          *smaller_leaf_split_info = *other_split_info;
        }
      }
    }
  }
  if (larger_leaf_index >= 0) {
    if (blockIdx.x == 1 || larger_only) {
      CUDASplitInfo* larger_leaf_split_info = cuda_leaf_best_split_info + larger_leaf_index;
      for (unsigned int block_index = 1; block_index < num_blocks_per_leaf; ++block_index) {
        const unsigned int leaf_read_pos = static_cast<unsigned int>(larger_leaf_index) + block_index * static_cast<unsigned int>(num_leaves);
        const CUDASplitInfo* other_split_info = cuda_leaf_best_split_info + leaf_read_pos;
        if ((other_split_info->is_valid && larger_leaf_split_info->is_valid &&
          other_split_info->gain > larger_leaf_split_info->gain) ||
            (!larger_leaf_split_info->is_valid && other_split_info->is_valid)) {
            *larger_leaf_split_info = *other_split_info;
        }
      }
    }
  }
}

__global__ void SetInvalidLeafSplitInfoKernel(
  CUDASplitInfo* cuda_leaf_best_split_info,
  const bool is_smaller_leaf_valid,
  const bool is_larger_leaf_valid,
  const int smaller_leaf_index,
  const int larger_leaf_index) {
  if (!is_smaller_leaf_valid) {
    cuda_leaf_best_split_info[smaller_leaf_index].is_valid = false;
  }
  if (!is_larger_leaf_valid && larger_leaf_index >= 0) {
    cuda_leaf_best_split_info[larger_leaf_index].is_valid = false;
  }
}

void CUDABestSplitFinder::LaunchSyncBestSplitForLeafKernel(
  const int host_smaller_leaf_index,
  const int host_larger_leaf_index,
  const bool is_smaller_leaf_valid,
  const bool is_larger_leaf_valid) {
  if (!is_smaller_leaf_valid || !is_larger_leaf_valid) {
    SetInvalidLeafSplitInfoKernel<<<1, 1>>>(
      cuda_leaf_best_split_info_.RawData(),
      is_smaller_leaf_valid, is_larger_leaf_valid,
      host_smaller_leaf_index, host_larger_leaf_index);
  }
  if (!is_smaller_leaf_valid && !is_larger_leaf_valid) {
    return;
  }
  int num_tasks = num_tasks_;
  int num_tasks_aligned = 1;
  num_tasks -= 1;
  while (num_tasks > 0) {
    num_tasks_aligned <<= 1;
    num_tasks >>= 1;
  }
  const int num_blocks_per_leaf = (num_tasks_ + NUM_TASKS_PER_SYNC_BLOCK - 1) / NUM_TASKS_PER_SYNC_BLOCK;
  if (host_larger_leaf_index >= 0 && is_smaller_leaf_valid && is_larger_leaf_valid) {
    // Reduce the two child leaves concurrently on separate streams. The smaller leaf
    // is reduced on stream 0 and the larger on stream 1; each reads only the region of
    // cuda_best_split_info_ that FindBestSplitsForLeafKernel wrote on that same stream,
    // so per-stream ordering guarantees correctness with no device sync between them.
    // This drops the full SynchronizeCUDADevice that previously separated the two
    // launches (a host<->GPU round trip per split that also serialized the two leaves
    // onto a single SM at a time). SyncBestSplitForLeafKernel was ~19% of training GPU
    // time. Output is bit-identical to the synced version (verified vs the deterministic
    // quantized path). NOTE: a single merged launch over 2*num_blocks_per_leaf blocks on
    // the default stream is NOT equivalent here -- it races with the stream-0/1 writes of
    // FindBestSplits and yields non-deterministic trees; keep the two leaves stream-local.
    SyncBestSplitForLeafKernel<<<num_blocks_per_leaf, NUM_TASKS_PER_SYNC_BLOCK, 0, cuda_streams_[0]>>>(
      host_smaller_leaf_index,
      host_larger_leaf_index,
      cuda_leaf_best_split_info_.RawData(),
      cuda_split_find_tasks_.RawData(),
      cuda_best_split_info_.RawData(),
      num_tasks_,
      num_tasks_aligned,
      num_blocks_per_leaf,
      false,
      num_leaves_);
    if (num_blocks_per_leaf > 1) {
      SyncBestSplitForLeafKernelAllBlocks<<<1, 1, 0, cuda_streams_[0]>>>(
        host_smaller_leaf_index, host_larger_leaf_index,
        num_blocks_per_leaf, num_leaves_, cuda_leaf_best_split_info_.RawData(), false);
    }
    SyncBestSplitForLeafKernel<<<num_blocks_per_leaf, NUM_TASKS_PER_SYNC_BLOCK, 0, cuda_streams_[1]>>>(
      host_smaller_leaf_index,
      host_larger_leaf_index,
      cuda_leaf_best_split_info_.RawData(),
      cuda_split_find_tasks_.RawData(),
      cuda_best_split_info_.RawData(),
      num_tasks_,
      num_tasks_aligned,
      num_blocks_per_leaf,
      true,
      num_leaves_);
    if (num_blocks_per_leaf > 1) {
      SyncBestSplitForLeafKernelAllBlocks<<<1, 1, 0, cuda_streams_[1]>>>(
        host_smaller_leaf_index, host_larger_leaf_index,
        num_blocks_per_leaf, num_leaves_, cuda_leaf_best_split_info_.RawData(), true);
    }
  } else {
    const bool larger_only = (!is_smaller_leaf_valid && is_larger_leaf_valid);
    SyncBestSplitForLeafKernel<<<num_blocks_per_leaf, NUM_TASKS_PER_SYNC_BLOCK>>>(
      host_smaller_leaf_index,
      host_larger_leaf_index,
      cuda_leaf_best_split_info_.RawData(),
      cuda_split_find_tasks_.RawData(),
      cuda_best_split_info_.RawData(),
      num_tasks_,
      num_tasks_aligned,
      num_blocks_per_leaf,
      larger_only,
      num_leaves_);
    if (num_blocks_per_leaf > 1) {
      SynchronizeCUDADevice(__FILE__, __LINE__);
      SyncBestSplitForLeafKernelAllBlocks<<<1, 1>>>(
        host_smaller_leaf_index,
        host_larger_leaf_index,
        num_blocks_per_leaf,
        num_leaves_,
        cuda_leaf_best_split_info_.RawData(),
        larger_only);
    }
  }
}

__global__ void FindBestFromAllSplitsKernel(const int cur_num_leaves,
  CUDASplitInfo* cuda_leaf_best_split_info,
  int* cuda_best_split_info_buffer) {
  __shared__ double gain_shared_buffer[WARPSIZE];
  __shared__ int leaf_index_shared_buffer[WARPSIZE];
  double thread_best_gain = kMinScore;
  int thread_best_leaf_index = -1;
  const int threadIdx_x = static_cast<int>(threadIdx.x);
  for (int leaf_index = threadIdx_x; leaf_index < cur_num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    const double leaf_best_gain = cuda_leaf_best_split_info[leaf_index].gain;
    if (cuda_leaf_best_split_info[leaf_index].is_valid && leaf_best_gain > thread_best_gain) {
      thread_best_gain = leaf_best_gain;
      thread_best_leaf_index = leaf_index;
    }
  }
  const int best_leaf_index = ReduceBestGainForLeaves(thread_best_gain, thread_best_leaf_index, gain_shared_buffer, leaf_index_shared_buffer);
  if (threadIdx_x == 0) {
    cuda_best_split_info_buffer[6] = best_leaf_index;
    if (best_leaf_index != -1) {
      cuda_leaf_best_split_info[best_leaf_index].is_valid = false;
      cuda_leaf_best_split_info[cur_num_leaves].is_valid = false;
      cuda_best_split_info_buffer[7] = cuda_leaf_best_split_info[best_leaf_index].num_cat_threshold;
    }
  }
}

__global__ void PrepareLeafBestSplitInfo(const int smaller_leaf_index, const int larger_leaf_index,
  int* cuda_best_split_info_buffer,
  const CUDASplitInfo* cuda_leaf_best_split_info) {
  const unsigned int threadIdx_x = blockIdx.x;
  if (threadIdx_x == 0) {
    cuda_best_split_info_buffer[0] = cuda_leaf_best_split_info[smaller_leaf_index].inner_feature_index;
  } else if (threadIdx_x == 1) {
    cuda_best_split_info_buffer[1] = cuda_leaf_best_split_info[smaller_leaf_index].threshold;
  } else if (threadIdx_x == 2) {
    cuda_best_split_info_buffer[2] = cuda_leaf_best_split_info[smaller_leaf_index].default_left;
  }
  if (larger_leaf_index >= 0) {
    if (threadIdx_x == 3) {
      cuda_best_split_info_buffer[3] = cuda_leaf_best_split_info[larger_leaf_index].inner_feature_index;
    } else if (threadIdx_x == 4) {
      cuda_best_split_info_buffer[4] = cuda_leaf_best_split_info[larger_leaf_index].threshold;
    } else if (threadIdx_x == 5) {
      cuda_best_split_info_buffer[5] = cuda_leaf_best_split_info[larger_leaf_index].default_left;
    }
  }
}

void CUDABestSplitFinder::SyncLeafBestSplitToHost(
  const int smaller_leaf_index,
  const int larger_leaf_index,
  int* smaller_leaf_best_split_feature,
  uint32_t* smaller_leaf_best_split_threshold,
  uint8_t* smaller_leaf_best_split_default_left,
  int* larger_leaf_best_split_feature,
  uint32_t* larger_leaf_best_split_threshold,
  uint8_t* larger_leaf_best_split_default_left) {
  PrepareLeafBestSplitInfo<<<6, 1, 0, cuda_streams_[0]>>>(smaller_leaf_index, larger_leaf_index,
    cuda_best_split_info_buffer_.RawData(),
    cuda_leaf_best_split_info_.RawData());
  std::vector<int> host_leaf_best_split_info_buffer(8, 0);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(host_leaf_best_split_info_buffer.data(), cuda_best_split_info_buffer_.RawData(), 8, __FILE__, __LINE__);
  *smaller_leaf_best_split_feature = host_leaf_best_split_info_buffer[0];
  *smaller_leaf_best_split_threshold = static_cast<uint32_t>(host_leaf_best_split_info_buffer[1]);
  *smaller_leaf_best_split_default_left = static_cast<uint8_t>(host_leaf_best_split_info_buffer[2]);
  if (larger_leaf_index >= 0) {
    *larger_leaf_best_split_feature = host_leaf_best_split_info_buffer[3];
    *larger_leaf_best_split_threshold = static_cast<uint32_t>(host_leaf_best_split_info_buffer[4]);
    *larger_leaf_best_split_default_left = static_cast<uint8_t>(host_leaf_best_split_info_buffer[5]);
  }
}

void CUDABestSplitFinder::EnsurePinnedLeafBestSplitCapacity(const int num_leaves) const {
  if (pinned_leaf_best_split_info_size_ >= static_cast<size_t>(num_leaves)) {
    return;
  }
  if (pinned_leaf_best_split_info_ != nullptr) {
    CUDASUCCESS_OR_FATAL(cudaFreeHost(pinned_leaf_best_split_info_));
  }
  pinned_leaf_best_split_info_size_ = static_cast<size_t>(num_leaves_ > num_leaves ? num_leaves_ : num_leaves);
  CUDASUCCESS_OR_FATAL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_leaf_best_split_info_),
    pinned_leaf_best_split_info_size_ * sizeof(CUDASplitInfo), cudaHostAllocDefault));
}

void CUDABestSplitFinder::ReadPrefetchedLeafBestSplits(const int num_leaves, std::vector<CUDASplitInfo>* out) const {
  out->resize(static_cast<size_t>(num_leaves));
  std::memcpy(reinterpret_cast<void*>(out->data()), pinned_leaf_best_split_info_,
    static_cast<size_t>(num_leaves) * sizeof(CUDASplitInfo));
  // the raw copy brings over device categorical-threshold pointers; scrub them so the
  // host-side destructor never frees device memory (hybrid growth is numerical-only)
  for (CUDASplitInfo& info : *out) {
    info.num_cat_threshold = 0;
    info.cat_threshold = nullptr;
    info.cat_threshold_real = nullptr;
  }
}

void CUDABestSplitFinder::PrefetchLeafBestSplitsAsync(const int num_leaves, cudaStream_t stream) const {
  EnsurePinnedLeafBestSplitCapacity(num_leaves);
  CopyFromCUDADeviceToHostAsync<CUDASplitInfo>(pinned_leaf_best_split_info_,
    cuda_leaf_best_split_info_.RawDataReadOnly(), static_cast<size_t>(num_leaves),
    stream, __FILE__, __LINE__);
}

void CUDABestSplitFinder::SyncAllLeafBestSplitsToHost(const int num_leaves, std::vector<CUDASplitInfo>* out) const {
  // synchronous D2H on the legacy default stream: implicitly waits for all
  // preceding work on the (blocking) streams, so no explicit device sync needed.
  // Staged through a PINNED buffer: this copy runs once per level on the
  // hybrid critical path, and a pageable sync D2H pays an extra driver staging
  // round trip; the host-to-host memcpy of a few KB afterwards is negligible.
  EnsurePinnedLeafBestSplitCapacity(num_leaves);
  CopyFromCUDADeviceToHost<CUDASplitInfo>(pinned_leaf_best_split_info_, cuda_leaf_best_split_info_.RawDataReadOnly(),
    static_cast<size_t>(num_leaves), __FILE__, __LINE__);
  ReadPrefetchedLeafBestSplits(num_leaves, out);
}

void CUDABestSplitFinder::LaunchFindBestFromAllSplitsKernel(
  const int cur_num_leaves,
  const int smaller_leaf_index, const int larger_leaf_index,
  int* smaller_leaf_best_split_feature,
  uint32_t* smaller_leaf_best_split_threshold,
  uint8_t* smaller_leaf_best_split_default_left,
  int* larger_leaf_best_split_feature,
  uint32_t* larger_leaf_best_split_threshold,
  uint8_t* larger_leaf_best_split_default_left,
  int* best_leaf_index,
  int* num_cat_threshold) {
  FindBestFromAllSplitsKernel<<<1, NUM_THREADS_FIND_BEST_LEAF, 0, cuda_streams_[1]>>>(cur_num_leaves,
    cuda_leaf_best_split_info_.RawData(),
    cuda_best_split_info_buffer_.RawData());
  PrepareLeafBestSplitInfo<<<6, 1, 0, cuda_streams_[0]>>>(smaller_leaf_index, larger_leaf_index,
    cuda_best_split_info_buffer_.RawData(),
    cuda_leaf_best_split_info_.RawData());
  std::vector<int> host_leaf_best_split_info_buffer(8, 0);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(host_leaf_best_split_info_buffer.data(), cuda_best_split_info_buffer_.RawData(), 8, __FILE__, __LINE__);
  *smaller_leaf_best_split_feature = host_leaf_best_split_info_buffer[0];
  *smaller_leaf_best_split_threshold = static_cast<uint32_t>(host_leaf_best_split_info_buffer[1]);
  *smaller_leaf_best_split_default_left = static_cast<uint8_t>(host_leaf_best_split_info_buffer[2]);
  if (larger_leaf_index >= 0) {
    *larger_leaf_best_split_feature = host_leaf_best_split_info_buffer[3];
    *larger_leaf_best_split_threshold = static_cast<uint32_t>(host_leaf_best_split_info_buffer[4]);
    *larger_leaf_best_split_default_left = static_cast<uint8_t>(host_leaf_best_split_info_buffer[5]);
  }
  *best_leaf_index = host_leaf_best_split_info_buffer[6];
  *num_cat_threshold = host_leaf_best_split_info_buffer[7];
}

__global__ void AllocateCatVectorsKernel(
  CUDASplitInfo* cuda_split_infos, size_t len,
  const int max_num_categories_in_split,
  const bool has_categorical_feature,
  uint32_t* cat_threshold_vec,
  int* cat_threshold_real_vec) {
  const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < len) {
    if (has_categorical_feature) {
      cuda_split_infos[i].cat_threshold = cat_threshold_vec + i * max_num_categories_in_split;
      cuda_split_infos[i].cat_threshold_real = cat_threshold_real_vec + i * max_num_categories_in_split;
      cuda_split_infos[i].num_cat_threshold = 0;
    } else {
      cuda_split_infos[i].cat_threshold = nullptr;
      cuda_split_infos[i].cat_threshold_real = nullptr;
      cuda_split_infos[i].num_cat_threshold = 0;
    }
  }
}

void CUDABestSplitFinder::LaunchAllocateCatVectorsKernel(
  CUDASplitInfo* cuda_split_infos, uint32_t* cat_threshold_vec, int* cat_threshold_real_vec, size_t len) {
  const int num_blocks = (static_cast<int>(len) + NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER - 1) / NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER;
  AllocateCatVectorsKernel<<<num_blocks, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER>>>(
    cuda_split_infos, len, max_num_categories_in_split_, has_categorical_feature_, cat_threshold_vec, cat_threshold_real_vec);
}

__global__ void InitCUDARandomKernel(
  const int seed,
  const int num_tasks,
  CUDARandom* cuda_randoms) {
  const int task_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (task_index < num_tasks) {
    cuda_randoms[task_index].SetSeed(seed + task_index);
  }
}

void CUDABestSplitFinder::LaunchInitCUDARandomKernel() {
  const int num_blocks = (static_cast<int>(cuda_randoms_.Size()) +
    NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER - 1) / NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER;
  InitCUDARandomKernel<<<num_blocks, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER>>>(extra_seed_,
    static_cast<int>(cuda_randoms_.Size()), cuda_randoms_.RawData());
}

}  // namespace LightGBM

#endif  // USE_CUDA
