/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include "cuda_best_split_finder.hpp"

#include <Falcata/cuda/cuda_rocm_interop.h>

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <type_traits>
#include <vector>

#include <Falcata/cuda/cuda_algorithms.hpp>

namespace Falcata {

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
 *  is float pairs in the fp32 histogram mode (FALCATA_FP32_HIST) and hist_t
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

// Definition of CPU's sequential categorical-threshold acceptance rule. The
// reference implementation is FeatureHistogram::FindBestThresholdCategoricalInner
// in src/treelearner/feature_histogram.cpp; this replays it for a single
// threshold index `i` and returns whether CPU would evaluate a split there.
//
// Rule. Categories are sorted by their gradient/hessian ratio and scanned in
// that fixed order. Groups form *sequentially* over the scan: a running counter
// accumulates category counts, a threshold is a candidate iff its accumulated
// group has reached `min_data_per_group`, and the counter resets to zero at
// every accepted candidate. Acceptance therefore depends on which earlier
// thresholds were accepted, so it cannot be expressed as an independent
// per-thread predicate -- hence the O(i) replay (i < max_cat_threshold, default
// 32, so this is cheaper than materialising a mask in shared memory).
//
// This rule *is* the specification: min_data_per_group has no external canon
// (the sorted-category scan is Fisher-1958-canonical, but the per-group minimum
// is Falcata's own regularization heuristic), so the CPU routine above is the
// reference, not merely "whatever CPU happens to do". The previous CUDA
// approximation `left_count >= min_data_per_group && right_count >=
// min_data_per_group` is a different rule and picked a different categorical
// split than CPU whenever min_data_per_group landed near the per-category counts.
//
// Preconditions, each matching the CPU scan exactly:
//  * `sum_left_hessian_prefix` is the inclusive prefix sum of the sorted
//    per-category hessians with kEpsilon folded into element 0 -- CPU starts
//    `sum_left_hessian = kEpsilon`. Both call sites satisfy this by construction:
//    they add kEpsilon to the first entry before the prefix scan and hand this
//    routine the very same prefix buffer the gain computation reads.
//  * Residual count-rounding caveat: CPU accumulates left_count as the sum of
//    per-category roundings (`left_count += RoundInt(hess * cnt_factor)`), while
//    here -- as everywhere else in the CUDA finder -- left_count is the rounding
//    of the summed hessian (`RoundInt(sum_left_hessian * cnt_factor)`). The two
//    agree for unit-hessian objectives and can differ by a category otherwise.
//    This is pre-existing CUDA behaviour, not introduced by the group check.
template <typename PREFIX_PTR_T>
__device__ __forceinline__ bool SequentialCategoricalGroupAccepted(
  const PREFIX_PTR_T sum_left_hessian_prefix,
  const data_size_t* left_count_prefix,
  const int i,
  const int num_thresholds,
  const data_size_t num_data,
  const double sum_hessians,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const int min_data_per_group) {
  data_size_t last_accepted_left_count = 0;
  for (int j = 0; j < num_thresholds; ++j) {
    const double sum_left_hessian = sum_left_hessian_prefix[j];
    const data_size_t left_count = left_count_prefix[j];
    // CPU: `continue` -- the group counter keeps accumulating across skipped
    // thresholds, which the left_count difference below accounts for.
    if (left_count < min_data_in_leaf || sum_left_hessian < min_sum_hessian_in_leaf) {
      continue;
    }
    const data_size_t right_count = num_data - left_count;
    // CPU: `break` -- every threshold from here on is rejected, including i.
    if (right_count < min_data_in_leaf || right_count < min_data_per_group) {
      return false;
    }
    const double sum_right_hessian = sum_hessians - sum_left_hessian;
    if (sum_right_hessian < min_sum_hessian_in_leaf) {
      return false;
    }
    // CPU: cnt_cur_group == left_count - (left_count at last accepted threshold).
    if (left_count - last_accepted_left_count < min_data_per_group) {
      continue;
    }
    last_accepted_left_count = left_count;
    if (j == i) {
      return true;
    }
  }
  return false;
}

// ----- RANDOM categorical search (cat_random_search) -------------------------
// Draws for the RANDOM search come from a SplitMix64 stream, not from
// CUDARandom: CUDARandom is an LCG whose low bits carry short periods, and the
// trials run one per thread, so a single shared LCG could not feed them
// independently. The block's CUDARandom is drawn ONCE, by thread 0, to seed the
// node; every trial stream is then mixed from that seed and its own trial
// index, which makes a trial's subset a pure function of (node seed, trial) and
// so re-drawable by the winning thread at write-out.
__device__ __forceinline__ uint64_t CatSplitMix64(uint64_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

__device__ __forceinline__ uint64_t CatTrialSeed(const unsigned int node_seed, const int trial) {
  return (static_cast<uint64_t>(node_seed) << 32) ^
         (static_cast<uint64_t>(static_cast<uint32_t>(trial)) * 0x9E3779B97F4A7C15ULL);
}

// Subset size, uniform over [1, max_num_cat] -- max_num_cat is the same cap the
// exact search puts on its prefix length. Including each category with
// probability 1/2 instead would put every subset of a high-cardinality feature
// over that cap, and the feature would never split at all.
__device__ __forceinline__ int CatTrialSubsetSize(uint64_t* state, const int max_num_cat) {
  return 1 + static_cast<int>(CatSplitMix64(state) % static_cast<uint64_t>(max_num_cat));
}

// One step of Knuth's selection sampling (Algorithm S): take the next candidate
// with probability remaining_needed / remaining_cand. A whole pass draws a
// uniformly random subset of exactly the requested size -- the same
// distribution as CPU's partial Fisher-Yates -- in ascending candidate order
// and with no per-thread scratch, which a shuffled pool copy would need (the
// candidate list runs to thousands of categories, far past a thread's budget).
__device__ __forceinline__ bool CatCandidateSelected(uint64_t* state,
                                                     const int remaining_cand,
                                                     const int remaining_needed) {
  return CatSplitMix64(state) % static_cast<uint64_t>(remaining_cand) <
         static_cast<uint64_t>(remaining_needed);
}

// Per-type machine epsilon used to size the gain-tie tolerance. Spelled out as
// device-safe constants (rather than std::numeric_limits, which is awkward in
// device code) so the reductions can run in either fp64 (default) or fp32
// (FALCATA_FP32_GAIN) gain arithmetic.
template <typename GAIN_T>
__device__ __forceinline__ constexpr GAIN_T GainTieEpsilon();
template <>
__device__ __forceinline__ constexpr double GainTieEpsilon<double>() { return 2.2204460492503131e-16; }
template <>
__device__ __forceinline__ constexpr float GainTieEpsilon<float>() { return 1.19209290e-07f; }

// Gain-tie handling in the best-split reductions: ties break to the lower
// thread index, which corresponds to the lower bin index (for REVERSE tasks,
// lower thread index is earlier in CPU's reverse scan) -- matching CPU's
// "first candidate in scan order wins" on a strict-'>' scan.
//
// In fp64 gain arithmetic the tie test is EXACT equality: the deterministic
// constructs make both devices' histogram sums bit-equal, so equal candidates
// produce bit-equal gains and (gain desc, index asc) is a TOTAL ORDER -- the
// tree reduction is pairing-independent and lands on the same candidate CPU's
// sequential scan does. A tolerance band here would break transitivity: a
// chain of near-ties spanning more than one band lets a higher-index
// candidate win without ever comparing against the true (lower-index) winner
// -- observed as the #308-class plateau divergence, where a 2^-42 noise-gain
// tie went to feature 2 while CPU took feature 0.
//
// Under FALCATA_FP32_GAIN the gains themselves are float-noisy, so exact
// equality would degenerate to "never ties" and re-expose the plateau bug;
// the fp32 specialization keeps a ~5000-epsilon band (~6e-4 relative) --
// non-transitive in theory, but fp32 gain mode holds no bit-parity contract.
//
// The same helper flows through ReduceBestGain (warp + block reductions) into
// every path -- the classic per-leaf kernels, the batched level kernels, the
// graph-captured level bodies, and the cross-task SyncBestSplitForLeaf/Level
// reductions -- so one definition fixes all of them. In the cross-task reduction
// "lower thread index" is the lower TASK index; tasks are built feature-major /
// direction-minor (InitCUDAFeatureMetaInfo), so a cross-feature plateau
// tie-breaks to the lower feature index, matching the order CPU scans features.
template <typename GAIN_T>
__device__ __forceinline__ constexpr GAIN_T GainTieRelTol();
template <>
__device__ __forceinline__ constexpr double GainTieRelTol<double>() { return 0.0; }
template <>
__device__ __forceinline__ constexpr float GainTieRelTol<float>() {
  return 5.0e3f * GainTieEpsilon<float>();
}

// CPU-order inclusive prefix sum: one thread accumulates lanes 0..i
// sequentially, bit-matching the CPU finder's threshold scan. The shuffle
// scan's tree-shaped addition order is not bit-equal to sequential
// accumulation, and on plateau (cancellation-noise) gains those low bits
// decide the argmax -- device-vs-host parity needs the same bits, not the
// same tolerance class. Serving the fp64 gain path only: quantized prefixes
// are integer (order-invariant) and fp32 gain mode holds no parity contract,
// so both keep the parallel scan.
template <typename T>
__device__ __forceinline__ T SequentialPrefixSum(T value, T* row_buffer) {
  __syncthreads();  // previous users of row_buffer may still be reading
  row_buffer[threadIdx.x] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    T acc = row_buffer[0];
    for (unsigned int i = 1; i < blockDim.x; ++i) {
      acc += row_buffer[i];
      row_buffer[i] = acc;
    }
  }
  __syncthreads();
  return row_buffer[threadIdx.x];
}

// Two CPU-order scans in one pass: the a/b accumulator chains are
// independent, so interleaving them hides each other's add latency and the
// pair costs about as much as one scan. len bounds the fold at the lanes the
// caller actually reads (trailing lanes then keep their own deposit, which
// the callers' bin-range guards never read); row_buffer holds 2 * blockDim.x
// elements.
template <typename T>
__device__ __forceinline__ void SequentialPrefixSumPair(
    T* value_a, T* value_b, T* row_buffer, const unsigned int len) {
  __syncthreads();  // previous users of row_buffer may still be reading
  row_buffer[threadIdx.x] = *value_a;
  row_buffer[blockDim.x + threadIdx.x] = *value_b;
  __syncthreads();
  if (threadIdx.x == 0) {
    T acc_a = row_buffer[0];
    T acc_b = row_buffer[blockDim.x];
    const unsigned int bound = len < blockDim.x ? len : blockDim.x;
    for (unsigned int i = 1; i < bound; ++i) {
      acc_a += row_buffer[i];
      row_buffer[i] = acc_a;
      acc_b += row_buffer[blockDim.x + i];
      row_buffer[blockDim.x + i] = acc_b;
    }
  }
  __syncthreads();
  *value_a = row_buffer[threadIdx.x];
  *value_b = row_buffer[blockDim.x + threadIdx.x];
}

// CPU-order in-place inclusive scan over a global-memory row (the
// GlobalMemory finder's fp64 buffers, features wider than one block): chunks
// are staged through shared memory (coalesced), one thread folds each chunk
// carrying the running sum, so the fold order is exactly CPU's sequential
// accumulation at staging speed. Integer count prefixes stay on the parallel
// GlobalMemoryPrefixSum (order-invariant).
template <typename T>
__device__ __forceinline__ void GlobalMemorySequentialPrefixSum(T* array, const size_t len) {
  constexpr unsigned int kSeqScanChunk = 256;
  __shared__ T seq_scan_stage[kSeqScanChunk];
  __shared__ T seq_scan_carry;
  if (threadIdx.x == 0) {
    seq_scan_carry = 0;
  }
  for (size_t chunk_start = 0; chunk_start < len; chunk_start += kSeqScanChunk) {
    const unsigned int chunk = static_cast<unsigned int>(
      min(static_cast<size_t>(kSeqScanChunk), len - chunk_start));
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < chunk; i += blockDim.x) {
      seq_scan_stage[i] = array[chunk_start + i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      T acc = seq_scan_carry;
      for (unsigned int i = 0; i < chunk; ++i) {
        acc += seq_scan_stage[i];
        seq_scan_stage[i] = acc;
      }
      seq_scan_carry = acc;
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < chunk; i += blockDim.x) {
      array[chunk_start + i] = seq_scan_stage[i];
    }
  }
  __syncthreads();
}

// CPU-order sequential subtraction of a feature's raw histogram out of the
// leaf totals (the GlobalMemory finder's NA_AS_MISSING head). CPU seeds the
// "missing" pseudo-bin with the totals and subtracts every real bin in
// ASCENDING bin order (feature_histogram.hpp, NA_AS_MISSING && offset == 1),
// so a tree-shaped reduce of the bins followed by one subtraction lands on
// different low bits. Chunks are staged through shared memory (coalesced) and
// one thread folds each chunk, the same shape as
// GlobalMemorySequentialPrefixSum.
__device__ __forceinline__ void GlobalMemorySequentialSubtractPair(
    const hist_t* hist, const uint32_t num_bins, double* grad, double* hess) {
  constexpr unsigned int kSeqSubChunk = 256;
  __shared__ double seq_sub_stage[2 * kSeqSubChunk];
  __shared__ double seq_sub_acc_grad;
  __shared__ double seq_sub_acc_hess;
  if (threadIdx.x == 0) {
    seq_sub_acc_grad = *grad;
    seq_sub_acc_hess = *hess;
  }
  for (uint32_t chunk_start = 0; chunk_start < num_bins; chunk_start += kSeqSubChunk) {
    const unsigned int chunk = static_cast<unsigned int>(
      min(static_cast<uint32_t>(kSeqSubChunk), num_bins - chunk_start));
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < chunk; i += blockDim.x) {
      const unsigned int bin_offset = (chunk_start + i) << 1;
      seq_sub_stage[i] = hist[bin_offset];
      seq_sub_stage[kSeqSubChunk + i] = hist[bin_offset + 1];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      double acc_g = seq_sub_acc_grad;
      double acc_h = seq_sub_acc_hess;
      for (unsigned int i = 0; i < chunk; ++i) {
        acc_g -= seq_sub_stage[i];
        acc_h -= seq_sub_stage[kSeqSubChunk + i];
      }
      seq_sub_acc_grad = acc_g;
      seq_sub_acc_hess = acc_h;
    }
  }
  __syncthreads();
  *grad = seq_sub_acc_grad;
  *hess = seq_sub_acc_hess;
}

template <typename GAIN_T>
__device__ __forceinline__ bool OtherIsBetterWithTieBreak(
    GAIN_T other_gain, GAIN_T gain, uint32_t other_thread_index, uint32_t thread_index) {
  const GAIN_T tol = fmax(fabs(gain), fabs(other_gain)) * GainTieRelTol<GAIN_T>();
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
      if ((leaves[tid] == -1 && leaves[tid_s] != -1) ||
          (leaves[tid] != -1 && leaves[tid_s] != -1 &&
           (gain[tid_s] > gain[tid] || (gain[tid_s] == gain[tid] && leaves[tid_s] < leaves[tid])))) {
        gain[tid] = gain[tid_s];
        leaves[tid] = leaves[tid_s];
      }
    }
    __syncthreads();
  }
}

// Gain ties break to the LOWER leaf index: CPU's ArgMax over
// best_split_per_leaf_ keeps the first (lowest-index) leaf on a strict-'>'
// scan, and (gain desc, leaf asc) is a total order, so the shuffle reduction
// is pairing-independent. Without the index tie-break, "keep own on tie"
// crowns whichever tied leaf happens to travel up the reduction lanes --
// observed as the #308-class cross-leaf plateau divergence.
__device__ __forceinline__ bool OtherLeafIsBetter(
    double other_gain, double gain, int other_leaf_index, int leaf_index) {
  if (leaf_index == -1) {
    return other_leaf_index != -1;
  }
  if (other_leaf_index == -1) {
    return false;
  }
  return other_gain > gain || (other_gain == gain && other_leaf_index < leaf_index);
}

__device__ void ReduceBestGainForLeavesWarp(double gain, int leaf_index, double* out_gain, int* out_leaf_index) {
  const uint32_t mask = 0xffffffff;
  const uint32_t warpLane = threadIdx.x % warpSize;
  for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    const int other_leaf_index = __shfl_down_sync(mask, leaf_index, offset);
    const double other_gain = __shfl_down_sync(mask, gain, offset);
    if (OtherLeafIsBetter(other_gain, gain, other_leaf_index, leaf_index)) {
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
    if (OtherLeafIsBetter(other_gain, gain, other_leaf_index, leaf_index)) {
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

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, bool REVERSE, typename GAIN_T, bool USE_MC>
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
  const double max_delta_step,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  // input parent node information
  const double parent_gain,
  const double sum_gradients,
  const double sum_hessians,
  const data_size_t num_data,
  const double parent_output,
  // monotone constraint information for this leaf
  const double leaf_constraint_min,
  const double leaf_constraint_max,
  // output parameters
  CUDASplitInfo* cuda_best_split_info) {
  // leaf-level inputs stay double; GAIN_T = float (FALCATA_FP32_GAIN) converts
  // them ONCE per task here, so all per-bin arithmetic below runs in fp32
  const int8_t monotone_constraint = task->monotone_type;
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
  const bool na_missing_head = (!REVERSE && task->na_as_missing && task->mfb_offset == 1);
  local_gain = kMinScore;
  data_size_t local_cnt_prefix = 0;
  if (sizeof(GAIN_T) == sizeof(double)) {
    // fp64 gains: CPU-order scans (see SequentialPrefixSum) -- plateau
    // (cancellation-noise) gains rank by their low bits, so the prefix must
    // carry CPU's exact fold order. The count prefix sums PER-BIN roundings
    // like CPU's scan (right_count += RoundInt(hess * cnt_factor) each bin,
    // feature_histogram.hpp) -- rounding the summed hessian instead can
    // differ by a count, and the count feeds min_data_in_leaf gating and the
    // stored child counts; integer addends are order-invariant, so the
    // shuffle scan serves them bit-exactly. (Bin 0's kEpsilon shifts its
    // product by ~1e-15, which never moves RoundInt in practice.)
    __shared__ GAIN_T seq_prefix_buffer[2 * NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER];
    data_size_t local_bin_cnt = static_cast<data_size_t>(
      CUDARoundInt(static_cast<double>(local_hess_hist) * static_cast<double>(cnt_factor)));
    if (na_missing_head) {
      // CPU's NaN-missing head (feature_histogram.hpp NA_AS_MISSING,
      // offset == 1): the lane-0 "missing" mass is the totals with every bin
      // SUBTRACTED in scan order -- gradient seeded from the total, hessian
      // from total - kEpsilon (no bin-0 epsilon add in this mode) -- and its
      // count is num_data minus the sum of per-bin roundings.
      const data_size_t cnt_non_default = ShuffleReduceSum<data_size_t>(
        local_bin_cnt, reinterpret_cast<data_size_t*>(shared_gain_buffer), blockDim.x);
      if (threadIdx_x == 0) {
        local_bin_cnt = num_data - cnt_non_default;
      }
      seq_prefix_buffer[threadIdx_x] = local_grad_hist;
      seq_prefix_buffer[blockDim.x + threadIdx_x] = local_hess_hist;
      __syncthreads();
      if (threadIdx_x == 0) {
        GAIN_T acc_g = static_cast<GAIN_T>(sum_gradients);
        GAIN_T acc_h = static_cast<GAIN_T>(sum_hessians) - static_cast<GAIN_T>(kEpsilon);
        for (unsigned int i = 1; i < blockDim.x; ++i) {
          acc_g -= seq_prefix_buffer[i];
          acc_h -= seq_prefix_buffer[blockDim.x + i];
        }
        local_grad_hist = acc_g;
        local_hess_hist = acc_h;
      }
    } else if (threadIdx_x == 0) {
      local_hess_hist += kEpsilon;
    }
    local_cnt_prefix = ShufflePrefixSum<data_size_t>(
      local_bin_cnt, reinterpret_cast<data_size_t*>(shared_gain_buffer));
    SequentialPrefixSumPair<GAIN_T>(&local_grad_hist, &local_hess_hist,
      seq_prefix_buffer, static_cast<unsigned int>(task->num_bin));
  } else {
    // fp32 gain mode: the parallel missing-mass reduce and shuffle scans
    // (no bit-parity contract).
    if (na_missing_head) {
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
    local_grad_hist = ShufflePrefixSum(local_grad_hist, shared_gain_buffer);
    __syncthreads();
    local_hess_hist = ShufflePrefixSum(local_hess_hist, shared_gain_buffer);
  }
  if (REVERSE) {
    if (threadIdx_x >= static_cast<unsigned int>(task->na_as_missing) && threadIdx_x <= task->num_bin - 2 && !skip_sum) {
      const GAIN_T sum_right_gradient = local_grad_hist;
      const GAIN_T sum_right_hessian = local_hess_hist;
      const data_size_t right_count = sizeof(GAIN_T) == sizeof(double) ?
        local_cnt_prefix :
        static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      const GAIN_T sum_left_gradient = sum_gradients_acc - sum_right_gradient;
      const GAIN_T sum_left_hessian = sum_hessians_acc - sum_right_hessian;
      const data_size_t left_count = num_data - right_count;
      if (sum_left_hessian >= min_sum_hessian_acc && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_acc && right_count >= min_data_in_leaf &&
        (!USE_RAND || static_cast<int>(task->num_bin - 2 - threadIdx_x) == rand_threshold)) {
        GAIN_T current_gain = USE_MC ?
          static_cast<GAIN_T>(CUDALeafSplits::GetSplitGainsMC<USE_L1, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, lambda_l1_acc,
            lambda_l2_acc, path_smooth_acc, static_cast<double>(max_delta_step), left_count, right_count, parent_output_acc,
            leaf_constraint_min, leaf_constraint_max, monotone_constraint)) :
          CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING, GAIN_T>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, lambda_l1_acc,
            lambda_l2_acc, path_smooth_acc, static_cast<GAIN_T>(max_delta_step), left_count, right_count, parent_output_acc);
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
      const data_size_t left_count = sizeof(GAIN_T) == sizeof(double) ?
        local_cnt_prefix :
        static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const GAIN_T sum_right_gradient = sum_gradients_acc - sum_left_gradient;
      const GAIN_T sum_right_hessian = sum_hessians_acc - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_acc && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_acc && right_count >= min_data_in_leaf &&
        (!USE_RAND || static_cast<int>(threadIdx_x + task->mfb_offset) == rand_threshold)) {
        GAIN_T current_gain = USE_MC ?
          static_cast<GAIN_T>(CUDALeafSplits::GetSplitGainsMC<USE_L1, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, lambda_l1_acc,
            lambda_l2_acc, path_smooth_acc, static_cast<double>(max_delta_step), left_count, right_count, parent_output_acc,
            leaf_constraint_min, leaf_constraint_max, monotone_constraint)) :
          CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING, GAIN_T>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, lambda_l1_acc,
            lambda_l2_acc, path_smooth_acc, static_cast<GAIN_T>(max_delta_step), left_count, right_count, parent_output_acc);
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
    cuda_best_split_info->gain = local_gain * task->penalty;
    cuda_best_split_info->default_left = task->assume_out_default_left;
    // the once-per-task output block stays in double (negligible cost)
    if (REVERSE) {
      // CPU's output chain (feature_histogram.hpp ~1015): left-basis sums
      // carry the kEpsilon seed; the stored fields subtract kEpsilon through
      // EXACTLY the CPU expression sequence, because a different rounding
      // chain leaves ulp differences in the child sums that snowball into
      // divergent trees. In fp64 the count is the per-bin-rounding prefix.
      const double best_sum_left_gradient = sum_gradients - static_cast<double>(local_grad_hist);
      const double best_sum_left_hessian = sum_hessians - static_cast<double>(local_hess_hist);
      const data_size_t left_count = sizeof(GAIN_T) == sizeof(double) ?
        (num_data - local_cnt_prefix) :
        (num_data - static_cast<data_size_t>(CUDARoundInt(
          (static_cast<double>(local_hess_hist) - kEpsilon) * static_cast<double>(cnt_factor))));
      const double sum_left_gradient = best_sum_left_gradient;
      const double sum_left_hessian = best_sum_left_hessian;
      const double sum_right_gradient = sum_gradients - best_sum_left_gradient;
      const double sum_right_hessian = sum_hessians - best_sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      // Unconstrained outputs first (max_delta_step cap included). The leaf VALUE
      // is clamped into the constraint range (MC), but the leaf GAIN stored for
      // future splits must stay unconstrained: it becomes the child's parent_gain
      // / min_gain_shift baseline, and the CPU reference (BeforeNumerical)
      // recomputes that baseline as the unconstrained GetLeafGain of the leaf's
      // sums. The max_delta_step cap is part of the analytic (unconstrained)
      // output, applied before the MC clamp, matching the CPU ordering.
      const double left_output_unconstrained =
        CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
          sum_left_hessian, lambda_l1, lambda_l2, path_smooth, max_delta_step, left_count, parent_output);
      const double right_output_unconstrained =
        CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
          sum_right_hessian, lambda_l1, lambda_l2, path_smooth, max_delta_step, right_count, parent_output);
      const double left_output = USE_MC ?
        (left_output_unconstrained < leaf_constraint_min ? leaf_constraint_min :
          (left_output_unconstrained > leaf_constraint_max ? leaf_constraint_max : left_output_unconstrained)) :
        left_output_unconstrained;
      const double right_output = USE_MC ?
        (right_output_unconstrained < leaf_constraint_min ? leaf_constraint_min :
          (right_output_unconstrained > leaf_constraint_max ? leaf_constraint_max : right_output_unconstrained)) :
        right_output_unconstrained;
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian - kEpsilon;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_hessians - sum_left_hessian - kEpsilon;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient,
        sum_left_hessian, lambda_l1, lambda_l2, left_output_unconstrained);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient,
        sum_right_hessian, lambda_l1, lambda_l2, right_output_unconstrained);
    } else {
      // CPU's output chain; see the REVERSE branch.
      const double best_sum_left_gradient = static_cast<double>(local_grad_hist);
      const double best_sum_left_hessian = static_cast<double>(local_hess_hist);
      const data_size_t left_count = sizeof(GAIN_T) == sizeof(double) ?
        local_cnt_prefix :
        static_cast<data_size_t>(CUDARoundInt(
          (static_cast<double>(local_hess_hist) - kEpsilon) * static_cast<double>(cnt_factor)));
      const double sum_left_gradient = best_sum_left_gradient;
      const double sum_left_hessian = best_sum_left_hessian;
      const double sum_right_gradient = sum_gradients - best_sum_left_gradient;
      const double sum_right_hessian = sum_hessians - best_sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      // Unconstrained outputs first (max_delta_step cap included). The leaf VALUE
      // is clamped into the constraint range (MC), but the leaf GAIN stored for
      // future splits must stay unconstrained: it becomes the child's parent_gain
      // / min_gain_shift baseline, and the CPU reference (BeforeNumerical)
      // recomputes that baseline as the unconstrained GetLeafGain of the leaf's
      // sums. The max_delta_step cap is part of the analytic (unconstrained)
      // output, applied before the MC clamp, matching the CPU ordering.
      const double left_output_unconstrained =
        CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
          sum_left_hessian, lambda_l1, lambda_l2, path_smooth, max_delta_step, left_count, parent_output);
      const double right_output_unconstrained =
        CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
          sum_right_hessian, lambda_l1, lambda_l2, path_smooth, max_delta_step, right_count, parent_output);
      const double left_output = USE_MC ?
        (left_output_unconstrained < leaf_constraint_min ? leaf_constraint_min :
          (left_output_unconstrained > leaf_constraint_max ? leaf_constraint_max : left_output_unconstrained)) :
        left_output_unconstrained;
      const double right_output = USE_MC ?
        (right_output_unconstrained < leaf_constraint_min ? leaf_constraint_min :
          (right_output_unconstrained > leaf_constraint_max ? leaf_constraint_max : right_output_unconstrained)) :
        right_output_unconstrained;
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian - kEpsilon;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_hessians - sum_left_hessian - kEpsilon;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient,
        sum_left_hessian, lambda_l1, lambda_l2, left_output_unconstrained);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient,
        sum_right_hessian, lambda_l1, lambda_l2, right_output_unconstrained);
    }
  }
}

// Warp-partial scratch for the discretized split kernels. One block-wide scan
// over packed ACC_HIST_TYPE accumulators and one best-gain reduction over
// uint32_t thread indices run at disjoint times, so they share the storage. The
// union is what makes the sharing legal: it carries the size AND the alignment
// of the widest member, whereas a uint32_t array over-sized to the right byte
// count still only guarantees 4-byte alignment, which an 8-byte ACC_HIST_TYPE
// store may not use.
template <typename ACC_HIST_TYPE>
union DiscretizedScanScratch {
  ACC_HIST_TYPE acc[WARPSIZE];
  uint32_t thread_index[WARPSIZE];
};

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
  const double max_delta_step,
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
  // GAIN_T (float under FALCATA_FP32_GAIN, double otherwise)
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
  __shared__ DiscretizedScanScratch<ACC_HIST_TYPE> shared_scan_scratch;
  const unsigned int threadIdx_x = threadIdx.x;
  const bool skip_sum = REVERSE ?
    (task->skip_default_bin && (task->num_bin - 1 - threadIdx_x) == static_cast<int>(task->default_bin)) :
    (task->skip_default_bin && (threadIdx_x + task->mfb_offset) == static_cast<int>(task->default_bin));
  const uint32_t feature_num_bin_minus_offset = task->num_bin - task->mfb_offset;
  if (!REVERSE) {
    if (task->na_as_missing && task->mfb_offset == 1) {
      // NaN is its own bin AND the most-frequent bin 0 is not stored in the
      // histogram, so the forward scan must read shifted by one and synthesise
      // bin 0 from the leaf total (done right after the sync below) -- exactly
      // what the non-quantized kernel does. Without this the quantized scan
      // drops every most-frequent-bin row from the left sums and reports
      // thresholds one bin off from what the partitioner then applies.
      if (threadIdx_x < static_cast<uint32_t>(task->num_bin) && threadIdx_x > 0) {
        const unsigned int bin_offset = threadIdx_x - 1;
        if (USE_16BIT_BIN_HIST && !USE_16BIT_ACC_HIST) {
          const int32_t local_grad_hess_hist_int32 = feature_hist_ptr[bin_offset];
          local_grad_hess_hist = (static_cast<int64_t>(static_cast<int16_t>(local_grad_hess_hist_int32 >> 16)) << 32) | (static_cast<int64_t>(local_grad_hess_hist_int32 & 0x0000ffff));
        } else {
          local_grad_hess_hist = feature_hist_ptr[bin_offset];
        }
      }
    } else if (threadIdx_x < feature_num_bin_minus_offset && !skip_sum) {
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
  if (!REVERSE && task->na_as_missing && task->mfb_offset == 1) {
    // bin 0 (the unstored most-frequent bin) = leaf total - sum of stored bins.
    // Packed accumulators add field-wise, so the reduction works directly on the
    // packed representation; the int64 detour below is only to reuse the leaf
    // total, which is always packed grad32<<32 | hess32.
    const ACC_HIST_TYPE sum_non_default = ShuffleReduceSum<ACC_HIST_TYPE>(
      local_grad_hess_hist, shared_scan_scratch.acc, blockDim.x);
    if (threadIdx_x == 0) {
      const int64_t non_default_packed = USE_16BIT_ACC_HIST ?
        ((static_cast<int64_t>(static_cast<int16_t>(sum_non_default >> 16)) << 32) |
         static_cast<int64_t>(sum_non_default & 0x0000ffff)) :
        static_cast<int64_t>(sum_non_default);
      const int64_t default_bin_packed = sum_gradients_hessians - non_default_packed;
      local_grad_hess_hist = USE_16BIT_ACC_HIST ?
        static_cast<ACC_HIST_TYPE>(static_cast<int32_t>(
          (static_cast<uint32_t>(static_cast<int32_t>(default_bin_packed >> 32)) << 16) |
          (static_cast<uint32_t>(default_bin_packed) & 0x0000ffffu))) :
        static_cast<ACC_HIST_TYPE>(default_bin_packed);
    }
    __syncthreads();
  }
  local_gain = kMinScore;
  local_grad_hess_hist = ShufflePrefixSum<ACC_HIST_TYPE>(local_grad_hess_hist, shared_scan_scratch.acc);
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
          lambda_l2_acc, path_smooth_acc, static_cast<GAIN_T>(max_delta_step), left_count, right_count, parent_output_acc);
        // gain with split is worse than without split
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_value = static_cast<uint32_t>(task->num_bin - 2 - threadIdx_x);
          threshold_found = true;
        }
      }
    }
  } else {
    const uint32_t end = (task->na_as_missing && task->mfb_offset == 1) ?
      static_cast<uint32_t>(task->num_bin - 2) : feature_num_bin_minus_offset - 2;
    if (threadIdx_x <= end && !skip_sum) {
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
          lambda_l2_acc, path_smooth_acc, static_cast<GAIN_T>(max_delta_step), left_count, right_count, parent_output_acc);
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
  const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_bool_buffer, shared_scan_scratch.thread_index);
  if (threadIdx_x == 0) {
    best_thread_index = result;
  }
  __syncthreads();
  if (threshold_found && threadIdx_x == best_thread_index) {
    cuda_best_split_info->is_valid = true;
    cuda_best_split_info->threshold = threshold_value;
    cuda_best_split_info->gain = local_gain * task->penalty;
    cuda_best_split_info->default_left = task->assume_out_default_left;
    // the once-per-task output block stays in double (negligible cost)
    const double sum_left_gradient_dbl = static_cast<double>(sum_left_gradient);
    const double sum_left_hessian_dbl = static_cast<double>(sum_left_hessian);
    const double sum_right_gradient_dbl = static_cast<double>(sum_right_gradient);
    const double sum_right_hessian_dbl = static_cast<double>(sum_right_hessian);
    const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient_dbl,
      sum_left_hessian_dbl, lambda_l1, lambda_l2, path_smooth, max_delta_step, left_count, parent_output);
    const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient_dbl,
      sum_right_hessian_dbl, lambda_l1, lambda_l2, path_smooth, max_delta_step, right_count, parent_output);
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

// Histogram readers for the categorical finder: the categorical search runs
// its per-bin math in double either way, so quantized histograms are unpacked
// per bin (exact integer sums times the scales) and the one body serves both
// pipelines. Packed() returns the canonical int64 (grad32 << 32 | hess32)
// accumulator the quantized pipeline needs for child-leaf seeding.
struct CatHistReaderF64 {
  static constexpr bool kQuant = false;
  const hist_t* p;
  __device__ __forceinline__ double Grad(int bin) const { return p[bin << 1]; }
  __device__ __forceinline__ double Hess(int bin) const { return p[(bin << 1) + 1]; }
  __device__ __forceinline__ int64_t Packed(int) const { return 0; }
};
struct CatHistReaderQuant16 {
  static constexpr bool kQuant = true;
  const int32_t* p;
  double grad_scale;
  double hess_scale;
  __device__ __forceinline__ int64_t Packed(int bin) const {
    const int32_t v = p[bin];
    return (static_cast<int64_t>(static_cast<int16_t>(v >> 16)) << 32) |
           static_cast<int64_t>(v & 0x0000ffff);
  }
  __device__ __forceinline__ double Grad(int bin) const {
    return static_cast<double>(static_cast<int16_t>(p[bin] >> 16)) * grad_scale;
  }
  __device__ __forceinline__ double Hess(int bin) const {
    return static_cast<double>(p[bin] & 0x0000ffff) * hess_scale;
  }
};
struct CatHistReaderQuant32 {
  static constexpr bool kQuant = true;
  const int64_t* p;
  double grad_scale;
  double hess_scale;
  __device__ __forceinline__ int64_t Packed(int bin) const { return p[bin]; }
  __device__ __forceinline__ double Grad(int bin) const {
    return static_cast<double>(static_cast<int32_t>((p[bin] & 0xffffffff00000000) >> 32)) * grad_scale;
  }
  __device__ __forceinline__ double Hess(int bin) const {
    return static_cast<double>(static_cast<int32_t>(p[bin] & 0x00000000ffffffff)) * hess_scale;
  }
};

// RANDOM categorical search (YDF's categorical_algorithm=RANDOM, Breiman 2001
// 5.1), the cat_random_search > 0 replacement for the sorted exact search. The
// reference implementation is FeatureHistogram::FindBestThresholdCategoricalInner
// in src/treelearner/feature_histogram.cpp; the constraint set and the gain
// formula are the exact search's.
//
// The parallel mapping INVERTS: the exact search puts one thread on each
// candidate THRESHOLD, which only works because the thresholds are prefixes of
// one sorted order and so share a prefix scan. Random subsets have no such
// structure, so a thread owns a whole TRIAL -- it draws its own subset and sums
// it serially -- and the sort, the prefix scans and the group-acceptance replay
// all drop out. Trials beyond blockDim.x are grid-strided.
//
// `candidate_bins` is the ascending, compacted list of the bins that pass
// cat_smooth (CPU's sorted_idx before its sort); both callers build it from
// their own scratch, so the search itself needs no scratch beyond the reduce.
// `l2` already carries cat_l2, and `min_gain_shift` the parent gain, as in the
// callers' exact branches.
template <bool USE_L1, bool USE_SMOOTHING, typename HIST_READER, typename INDEX_T>
__device__ void FindBestSplitsForLeafKernelCategoricalRandom(
  const HIST_READER hist_reader,
  const SplitFindTask* task,
  CUDARandom* cuda_random,
  const INDEX_T* candidate_bins,
  const int num_cand,
  // input config parameter values
  const int cat_random_search,
  const double lambda_l1,
  const double l2,
  const double path_smooth,
  const double max_delta_step,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const int max_cat_threshold,
  const int min_data_per_group,
  // input parent node information
  const double min_gain_shift,
  const double cnt_factor,
  const double sum_gradients,
  const double sum_hessians,
  const data_size_t num_data,
  const double parent_output,
  // canonical int64 leaf total (quant readers only; 0 for f64)
  const int64_t sum_gradients_hessians_total,
  // output parameters
  CUDASplitInfo* cuda_best_split_info) {
  __shared__ double shared_rand_gain_buffer[WARPSIZE];
  __shared__ bool shared_rand_found_buffer[WARPSIZE];
  __shared__ uint32_t shared_rand_trial_buffer[WARPSIZE];
  __shared__ uint32_t best_trial;
  __shared__ unsigned int node_seed;
  // One draw off the block's CUDARandom per node evaluation: the trial streams
  // fan out from it, so the shared LCG advances exactly once and its state
  // stays a function of how many nodes this task has been evaluated for.
  if (threadIdx.x == 0) {
    node_seed = static_cast<unsigned int>(cuda_random->NextInt(0, 0x7FFFFFFF));
  }
  __syncthreads();

  const int max_num_cat = min(max_cat_threshold, (num_cand + 1) / 2);
  double local_gain = min_gain_shift;
  bool trial_found = false;
  uint32_t local_best_trial = 0;
  double best_sum_left_gradient = 0.0;
  double best_sum_left_hessian = 0.0;
  data_size_t best_left_count = 0;
  int best_subset_size = 0;

  for (int trial = static_cast<int>(threadIdx.x);
       trial < cat_random_search && max_num_cat > 0;
       trial += static_cast<int>(blockDim.x)) {
    uint64_t state = CatTrialSeed(node_seed, trial);
    const int subset_size = CatTrialSubsetSize(&state, max_num_cat);
    double sum_left_gradient = 0.0;
    double sum_left_hessian = kEpsilon;
    data_size_t left_count = 0;
    int remaining_needed = subset_size;
    for (int k = 0; k < num_cand && remaining_needed > 0; ++k) {
      if (!CatCandidateSelected(&state, num_cand - k, remaining_needed)) {
        continue;
      }
      --remaining_needed;
      const int bin = static_cast<int>(candidate_bins[k]);
      const double hess = hist_reader.Hess(bin);
      sum_left_gradient += hist_reader.Grad(bin);
      sum_left_hessian += hess;
      // CPU rounds each category's count separately and sums the roundings.
      left_count += static_cast<data_size_t>(CUDARoundInt(hess * cnt_factor));
    }
    // categories filtered by cat_smooth, and categories unseen in this node,
    // always go right
    const data_size_t right_count = num_data - left_count;
    if (left_count < min_data_in_leaf || left_count < min_data_per_group ||
        sum_left_hessian < min_sum_hessian_in_leaf) {
      continue;
    }
    if (right_count < min_data_in_leaf || right_count < min_data_per_group) {
      continue;
    }
    const double sum_right_hessian = sum_hessians - sum_left_hessian;
    if (sum_right_hessian < min_sum_hessian_in_leaf) {
      continue;
    }
    const double sum_right_gradient = sum_gradients - sum_left_gradient;
    const double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
      sum_left_gradient, sum_left_hessian, sum_right_gradient, sum_right_hessian,
      lambda_l1, l2, path_smooth, max_delta_step, left_count, right_count, parent_output);
    // Running MAXIMUM over the grid-stride trials this thread owns; strict '>'
    // keeps the lowest trial index on an exact tie, matching CPU's scan over
    // its trial sequence.
    if (current_gain > local_gain) {
      local_gain = current_gain;
      trial_found = true;
      local_best_trial = static_cast<uint32_t>(trial);
      best_sum_left_gradient = sum_left_gradient;
      best_sum_left_hessian = sum_left_hessian;
      best_left_count = left_count;
      best_subset_size = subset_size;
    }
  }
  __syncthreads();
  // Reduce on the TRIAL index, not the thread index: above blockDim.x trials a
  // thread owns several, and trials across the stride boundary do not order
  // like thread indices. Trials are disjoint across threads (trial mod
  // blockDim.x == threadIdx.x), so exactly one writer matches below.
  const uint32_t result = ReduceBestGain(local_gain, trial_found, local_best_trial,
    shared_rand_gain_buffer, shared_rand_found_buffer, shared_rand_trial_buffer);
  if (threadIdx.x == 0) {
    best_trial = result;
  }
  __syncthreads();
  if (trial_found && local_best_trial == best_trial) {
    cuda_best_split_info->is_valid = true;
    cuda_best_split_info->num_cat_threshold = best_subset_size;
    cuda_best_split_info->gain = (local_gain - min_gain_shift) * task->penalty;
    cuda_best_split_info->default_left = false;
    // The winning subset is a draw, not a function of the thread index, so the
    // winner re-runs its own trial stream to emit it (one pass over the
    // candidates by one thread). Caching every trial's mask in shared memory
    // instead would cost blockDim.x * max_cat_threshold slots. Candidates are
    // ascending, so the emitted set is ascending, like CPU's sorted left set.
    uint64_t state = CatTrialSeed(node_seed, static_cast<int>(local_best_trial));
    const int subset_size = CatTrialSubsetSize(&state, max_num_cat);
    int remaining_needed = subset_size;
    int num_emitted = 0;
    int64_t left_packed = 0;
    for (int k = 0; k < num_cand && remaining_needed > 0; ++k) {
      if (!CatCandidateSelected(&state, num_cand - k, remaining_needed)) {
        continue;
      }
      --remaining_needed;
      const int bin = static_cast<int>(candidate_bins[k]);
      // slab slot pre-assigned by AllocateCatVectorsKernel; capacity is
      // max_num_categories_in_split, which bounds subset_size <= max_num_cat
      // exactly as it bounds the exact search's threshold + 1
      (cuda_best_split_info->cat_threshold)[num_emitted] =
        static_cast<uint32_t>(bin + task->mfb_offset);
      ++num_emitted;
      if (HIST_READER::kQuant) {
        left_packed += hist_reader.Packed(bin);
      }
    }
    if (HIST_READER::kQuant) {
      // exact int64 left sums over the selected categories: the quantized
      // pipeline seeds the child leaf structs from these packed fields
      cuda_best_split_info->left_sum_of_gradients_hessians = left_packed;
      cuda_best_split_info->right_sum_of_gradients_hessians =
        sum_gradients_hessians_total - left_packed;
    }
    const double sum_left_gradient = best_sum_left_gradient;
    const double sum_left_hessian = best_sum_left_hessian;
    // CPU stores the sum of per-category roundings; the child leaf inherits
    // this count, so rounding the summed hessian here would diverge by a
    // category and shift every child decision.
    const data_size_t left_count = best_left_count;
    const double sum_right_gradient = sum_gradients - sum_left_gradient;
    const double sum_right_hessian = sum_hessians - sum_left_hessian;
    const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
    const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
      sum_left_hessian, lambda_l1, l2, path_smooth, max_delta_step, left_count, parent_output);
    const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
      sum_right_hessian, lambda_l1, l2, path_smooth, max_delta_step, right_count, parent_output);
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

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, typename HIST_READER>
__device__ void FindBestSplitsForLeafKernelCategoricalInner(
  // input feature information
  const HIST_READER hist_reader,
  // input task information
  const SplitFindTask* task,
  CUDARandom* cuda_random,
  // input config parameter values
  const double lambda_l1,
  const double lambda_l2,
  const double path_smooth,
  const double max_delta_step,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  const int cat_random_search,
  // input parent node information
  const double parent_gain,
  const double sum_gradients,
  const double sum_hessians,
  const data_size_t num_data,
  const double parent_output,
  // canonical int64 leaf total (quant readers only; 0 for f64)
  const int64_t sum_gradients_hessians_total,
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
      const double grad = hist_reader.Grad(threadIdx_x);
      const double hess = hist_reader.Hess(threadIdx_x);
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
              l2, path_smooth, max_delta_step, other_count, cnt, parent_output);
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
      cuda_best_split_info->gain = (local_gain - min_gain_shift) * task->penalty;
      *(cuda_best_split_info->cat_threshold) = static_cast<uint32_t>(threadIdx_x + task->mfb_offset);
      cuda_best_split_info->default_left = false;
      if (HIST_READER::kQuant) {
        const int64_t left_packed = hist_reader.Packed(threadIdx_x);
        cuda_best_split_info->left_sum_of_gradients_hessians = left_packed;
        cuda_best_split_info->right_sum_of_gradients_hessians =
          sum_gradients_hessians_total - left_packed;
      }
      const double sum_left_gradient = hist_reader.Grad(threadIdx_x);
      const double sum_left_hessian = hist_reader.Hess(threadIdx_x);
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
        sum_left_hessian, lambda_l1, l2, path_smooth, max_delta_step, left_count, parent_output);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
        sum_right_hessian, lambda_l1, l2, path_smooth, max_delta_step, right_count, parent_output);
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
    __shared__ data_size_t shared_mem_buffer_cnt[WARPSIZE];
    __shared__ data_size_t shared_count_buffer[NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER];
    __shared__ int used_bin;
    l2 += cat_l2;
    uint16_t is_valid_bin = 0;
    int best_dir = 0;
    double best_sum_left_gradient = 0.0f;
    double best_sum_left_hessian = 0.0f;
    // Registered per thread: the count buffers hold only the LAST pass's
    // prefixes at write-out time, so the winning pass's count must ride along
    // with the other best_* registers.
    data_size_t best_left_count = 0;
    if (cat_random_search > 0) {
      // Compact the cat_smooth-passing bins into shared_index_buffer -- the
      // RANDOM search needs the candidate list, not the sorted order, so the
      // bitonic sort below is skipped entirely. Same candidate rule and same
      // <= blockDim.x bin window as the exact branch.
      if (threadIdx_x >= bin_start && threadIdx_x < bin_end &&
          CUDARoundInt(hist_reader.Hess(threadIdx_x) * cnt_factor) >= cat_smooth) {
        is_valid_bin = 1;
      }
      const data_size_t candidate_prefix =
        ShufflePrefixSum<data_size_t>(static_cast<data_size_t>(is_valid_bin), shared_mem_buffer_cnt);
      if (threadIdx.x == blockDim.x - 1) {
        used_bin = static_cast<int>(candidate_prefix);
      }
      if (is_valid_bin) {
        shared_index_buffer[candidate_prefix - 1] = static_cast<int16_t>(threadIdx_x);
      }
      __syncthreads();
      FindBestSplitsForLeafKernelCategoricalRandom<USE_L1, USE_SMOOTHING, HIST_READER, int16_t>(
        hist_reader, task, cuda_random, shared_index_buffer, used_bin,
        cat_random_search, lambda_l1, l2, path_smooth, max_delta_step,
        min_data_in_leaf, min_sum_hessian_in_leaf, max_cat_threshold, min_data_per_group,
        min_gain_shift, cnt_factor, sum_gradients, sum_hessians, num_data, parent_output,
        sum_gradients_hessians_total, cuda_best_split_info);
      return;
    }
    if (threadIdx_x >= bin_start && threadIdx_x < bin_end) {
      const double hess = hist_reader.Hess(threadIdx_x);
      if (CUDARoundInt(hess * cnt_factor) >= cat_smooth) {
        const double grad = hist_reader.Grad(threadIdx_x);
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
    // The shared sort buffers hold one slot per thread; a categorical feature
    // with more bins than the block has threads (cardinality > 256 -- bin
    // counts for categoricals are NOT capped by max_bin) previously made this
    // sort index past the buffers, corrupting adjacent shared state and
    // exploding training. Bins beyond the block are excluded from the
    // many-vs-many candidate set instead (categorical bins are ordered by
    // descending category frequency, so these are the rarest categories;
    // their mass routes to the right child). Exact full-cardinality support
    // is a tracked follow-up; the Init-time warning names affected features.
    const int sortable_bin_end = min(bin_end, static_cast<int>(blockDim.x));
    BitonicArgSort_1024<double, int16_t, true>(shared_value_buffer, shared_index_buffer, static_cast<int16_t>(sortable_bin_end));
    __syncthreads();
    const int max_num_cat = min(max_cat_threshold, (used_bin + 1) / 2);
    // Number of candidate thresholds CPU would walk in each direction.
    const int num_cat_thresholds = min(used_bin, max_num_cat);
    // shared_value_buffer held the sort keys; the sort is done with it, so both
    // passes below reuse it to publish their prefix-hessian scan.

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
    data_size_t cat_cnt = 0;
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const int sorted_bin = shared_index_buffer[threadIdx_x];
      grad = hist_reader.Grad(sorted_bin);
      hess = hist_reader.Hess(sorted_bin);
      // CPU rounds each category's count separately and sums the roundings;
      // rounding the summed hessian disagrees with that by a category
      // whenever the two roundings differ.
      cat_cnt = static_cast<data_size_t>(CUDARoundInt(hess * cnt_factor));
    }
    if (threadIdx_x == 0) {
      hess += kEpsilon;
    }
    __syncthreads();
    // CPU-order scans (see SequentialPrefixSum); shared_value_buffer is free
    // scratch here -- its prefix contents are (re)stored right below. The
    // count prefix stays a shuffle scan: integer sums are order-invariant.
    double sum_left_gradient = SequentialPrefixSum<double>(grad, shared_value_buffer);
    double sum_left_hessian = SequentialPrefixSum<double>(hess, shared_value_buffer);
    data_size_t left_count_prefix =
      ShufflePrefixSum<data_size_t>(cat_cnt, shared_mem_buffer_cnt);
    __syncthreads();
    if (threadIdx_x < num_cat_thresholds) {
      shared_value_buffer[threadIdx_x] = sum_left_hessian;
      shared_count_buffer[threadIdx_x] = left_count_prefix;
    }
    __syncthreads();
    if (threadIdx_x < num_cat_thresholds) {
      const data_size_t left_count = left_count_prefix;
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (SequentialCategoricalGroupAccepted(shared_value_buffer, shared_count_buffer,
          threadIdx_x, num_cat_thresholds,
          num_data, sum_hessians, min_data_in_leaf, min_sum_hessian_in_leaf,
          min_data_per_group) &&
        (!USE_RAND || threadIdx_x == static_cast<int>(rand_threshold))) {
        double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1,
          l2, path_smooth, max_delta_step, left_count, right_count, parent_output);
        // gain with split is worse than without split
        if (current_gain > local_gain) {
          local_gain = current_gain;
          threshold_found = true;
          best_dir = 1;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
          best_left_count = left_count;
        }
      }
    }
    __syncthreads();

    // right to left
    grad = 0.0f;
    hess = 0.0f;
    cat_cnt = 0;
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const int sorted_bin = shared_index_buffer[used_bin - 1 - threadIdx_x];
      grad = hist_reader.Grad(sorted_bin);
      hess = hist_reader.Hess(sorted_bin);
      cat_cnt = static_cast<data_size_t>(CUDARoundInt(hess * cnt_factor));
    }
    if (threadIdx_x == 0) {
      hess += kEpsilon;
    }
    __syncthreads();
    sum_left_gradient = SequentialPrefixSum<double>(grad, shared_value_buffer);
    sum_left_hessian = SequentialPrefixSum<double>(hess, shared_value_buffer);
    left_count_prefix = ShufflePrefixSum<data_size_t>(cat_cnt, shared_mem_buffer_cnt);
    __syncthreads();
    if (threadIdx_x < num_cat_thresholds) {
      shared_value_buffer[threadIdx_x] = sum_left_hessian;
      shared_count_buffer[threadIdx_x] = left_count_prefix;
    }
    __syncthreads();
    if (threadIdx_x < num_cat_thresholds) {
      const data_size_t left_count = left_count_prefix;
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (SequentialCategoricalGroupAccepted(shared_value_buffer, shared_count_buffer,
          threadIdx_x, num_cat_thresholds,
          num_data, sum_hessians, min_data_in_leaf, min_sum_hessian_in_leaf,
          min_data_per_group) &&
        (!USE_RAND || threadIdx_x == static_cast<int>(rand_threshold))) {
        double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1,
          l2, path_smooth, max_delta_step, left_count, right_count, parent_output);
        // gain with split is worse than without split
        if (current_gain > local_gain) {
          local_gain = current_gain;
          threshold_found = true;
          best_dir = -1;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
          best_left_count = left_count;
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
      cuda_best_split_info->gain = (local_gain - min_gain_shift) * task->penalty;
      if (best_dir == 1) {
        for (int i = 0; i < threadIdx_x + 1; ++i) {
          (cuda_best_split_info->cat_threshold)[i] = shared_index_buffer[i] + task->mfb_offset;
        }
      } else {
        for (int i = 0; i < threadIdx_x + 1; ++i) {
          (cuda_best_split_info->cat_threshold)[i] = shared_index_buffer[used_bin - 1 - i] + task->mfb_offset;
        }
      }
      if (HIST_READER::kQuant) {
        // exact int64 left sums over the selected categories (<= max_cat_threshold
        // iterations by one thread): the quantized pipeline seeds the child leaf
        // structs from these packed fields
        int64_t left_packed = 0;
        if (best_dir == 1) {
          for (int i = 0; i < threadIdx_x + 1; ++i) {
            left_packed += hist_reader.Packed(shared_index_buffer[i]);
          }
        } else {
          for (int i = 0; i < threadIdx_x + 1; ++i) {
            left_packed += hist_reader.Packed(shared_index_buffer[used_bin - 1 - i]);
          }
        }
        cuda_best_split_info->left_sum_of_gradients_hessians = left_packed;
        cuda_best_split_info->right_sum_of_gradients_hessians =
          sum_gradients_hessians_total - left_packed;
      }
      cuda_best_split_info->default_left = false;
      const double sum_left_gradient = best_sum_left_gradient;
      const double sum_left_hessian = best_sum_left_hessian;
      // CPU stores the sum of per-category roundings; the child leaf inherits
      // this count, so rounding the summed hessian here diverges from CPU by
      // a category and shifts every child decision. The count comes from the
      // winning pass's register: shared_count_buffer holds only the
      // right-to-left prefixes by now.
      const data_size_t left_count = best_left_count;
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
        sum_left_hessian, lambda_l1, l2, path_smooth, max_delta_step, left_count, parent_output);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
        sum_right_hessian, lambda_l1, l2, path_smooth, max_delta_step, right_count, parent_output);
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

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, bool IS_LARGER, typename GAIN_T, bool USE_MC>
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
  const double max_delta_step,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  const int cat_random_search,
  // output
  CUDASplitInfo* cuda_best_split_info,
  // global num data in leaf
  const data_size_t global_num_data_in_smaller_leaf,
  const data_size_t global_num_data_in_larger_leaf,
  // CEGB (passed via FindBestSplitsForLeafKernel_ARGS, before CONSTRAINT_ARGS)
  const double* cuda_task_cegb_penalty,
  const double cegb_tradeoff_times_penalty_split,
  // monotone leaf constraints (passed via FindBestSplitsForLeafKernel_CONSTRAINT_ARGS)
  const double smaller_leaf_constraint_min,
  const double smaller_leaf_constraint_max,
  const double larger_leaf_constraint_min,
  const double larger_leaf_constraint_max) {
  const double leaf_constraint_min = IS_LARGER ? larger_leaf_constraint_min : smaller_leaf_constraint_min;
  const double leaf_constraint_max = IS_LARGER ? larger_leaf_constraint_max : smaller_leaf_constraint_max;
  const unsigned int task_index = blockIdx.x;
  const SplitFindTask* task = tasks + task_index;
  const int inner_feature_index = task->inner_feature_index;
  const double sum_gradients = IS_LARGER ? larger_leaf_splits->sum_of_gradients : smaller_leaf_splits->sum_of_gradients;
  const double sum_hessians = (IS_LARGER ? larger_leaf_splits->sum_of_hessians : smaller_leaf_splits->sum_of_hessians) + 2 * kEpsilon;
  const data_size_t num_data = IS_LARGER ? global_num_data_in_larger_leaf : global_num_data_in_smaller_leaf;
  const double parent_output = IS_LARGER ? larger_leaf_splits->leaf_value : smaller_leaf_splits->leaf_value;
  // CPU recomputes the parent's gain from the leaf sums at find time
  // (feature_histogram.hpp gain_shift); the leaf struct's stored gain is the
  // same quantity through GetLeafGainGivenOutput, whose low bits differ -- and
  // a min_gain_shift off by one noise quantum flips split VALIDITY and
  // ranking on plateau gains. Same formula from the same sums, same bits.
  // (The discretized kernels keep the stored gain: the quantized flow has its
  // own shift semantics and no CPU bit-parity contract.)
  const double parent_gain = CUDALeafSplits::GetLeafGain<USE_L1, USE_SMOOTHING>(
      sum_gradients, sum_hessians, lambda_l1, lambda_l2, path_smooth,
      max_delta_step, num_data, parent_output);
  const unsigned int output_offset = IS_LARGER ? (task_index + num_tasks) : task_index;
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  // Both extra_trees (USE_RAND) and cat_random_search consume the per-task RNG
  // state, and the host allocates it for either, so the pointer must not be
  // gated on USE_RAND alone.
  CUDARandom* cuda_random = (USE_RAND || cat_random_search > 0) ?
    (IS_LARGER ? cuda_randoms + task_index * 2 + 1 : cuda_randoms + task_index * 2) : nullptr;
  if (is_feature_used_bytree[inner_feature_index]) {
    const hist_t* hist_ptr = FeatureHistPtr(
      IS_LARGER ? larger_leaf_splits->hist_in_leaf : smaller_leaf_splits->hist_in_leaf, task->hist_offset, fp32_hist);
    if (task->is_categorical) {
      // fp32_hist is globally disabled for datasets with categorical features
      const CatHistReaderF64 hist_reader{hist_ptr};
      FindBestSplitsForLeafKernelCategoricalInner<USE_RAND, USE_L1, USE_SMOOTHING, CatHistReaderF64>(
        // input feature information
        hist_reader,
        // input task information
        task,
        cuda_random,
        // input config parameter values
        lambda_l1,
        lambda_l2,
        path_smooth,
        max_delta_step,
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        min_gain_to_split,
        cat_smooth,
        cat_l2,
        max_cat_threshold,
        min_data_per_group,
        cat_random_search,
        // input parent node information
        parent_gain,
        sum_gradients,
        sum_hessians,
        num_data,
        parent_output,
        /*sum_gradients_hessians_total=*/0,
        // output parameters
        out);
    } else {
      if (!task->reverse) {
        FindBestSplitsForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, false, GAIN_T, USE_MC>(
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
          max_delta_step,
          min_data_in_leaf,
          min_sum_hessian_in_leaf,
          min_gain_to_split,
          // input parent node information
          parent_gain,
          sum_gradients,
          sum_hessians,
          num_data,
          parent_output,
          // monotone constraint information
          leaf_constraint_min,
          leaf_constraint_max,
          // output parameters
          out);
      } else {
        FindBestSplitsForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, true, GAIN_T, USE_MC>(
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
          max_delta_step,
          min_data_in_leaf,
          min_sum_hessian_in_leaf,
          min_gain_to_split,
          // input parent node information
          parent_gain,
          sum_gradients,
          sum_hessians,
          num_data,
          parent_output,
          // monotone constraint information
          leaf_constraint_min,
          leaf_constraint_max,
          // output parameters
          out);
      }
    }
    // CEGB: subtract the cost penalty from this task's gain (block-leader only).
    // Matches the CPU path: new_split.gain -= DeltaGain(...). is_valid is NOT
    // modified, so a split made negative by the penalty stays "valid" and simply
    // loses the cross-feature / cross-leaf gain comparison, exactly as on CPU.
    // The inner kernel finalizes `out` from a single (not necessarily thread 0)
    // thread and does not sync afterwards, so sync before reading it here.
    __syncthreads();
    if (threadIdx.x == 0 && out->is_valid) {
      double delta = cegb_tradeoff_times_penalty_split * static_cast<double>(num_data);
      if (cuda_task_cegb_penalty != nullptr) {
        delta += cuda_task_cegb_penalty[task_index];
      }
      out->gain -= delta;
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
  const double lambda_l2_in,
  const double path_smooth,
  const double max_delta_step,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  const int cat_random_search,
  const int max_cat_to_onehot,
  // gradient scale
  const score_t* grad_scale,
  const score_t* hess_scale,
  const bool quant_bagging_ridge,
  // output
  CUDASplitInfo* cuda_best_split_info,
  // global num data in leaf
  const data_size_t global_num_data_in_smaller_leaf,
  const data_size_t global_num_data_in_larger_leaf,
  // CEGB
  const double* cuda_task_cegb_penalty,
  const double cegb_tradeoff_times_penalty_split) {
  // Bagged quantized training: a child's integer hessian sum carries O(1)
  // quanta of rounding noise, and bagging redraws that noise every iteration.
  // The gain argmax then harvests children whose noisy |G| is large while
  // noisy H is near zero; with lambda_l2 == 0 their gains and outputs explode
  // far beyond what the leaf's true gradient mass supports (the compounding
  // multiclass-deep-tree quality collapse). One hessian quantum of ridge
  // bounds 1/H against exactly that noise. Unbagged training keeps the exact
  // math: its rounding noise is drawn once and fitted once, which stays
  // within quantization tolerance -- and keeps those models bit-identical.
  const double lambda_l2 = quant_bagging_ridge ?
    lambda_l2_in + static_cast<double>(*hess_scale) : lambda_l2_in;
  const unsigned int task_index = blockIdx.x;
  const SplitFindTask* task = tasks + task_index;
  const int inner_feature_index = task->inner_feature_index;
  const double parent_gain = IS_LARGER ? larger_leaf_splits->gain : smaller_leaf_splits->gain;
  const int64_t sum_gradients_hessians = IS_LARGER ? larger_leaf_splits->sum_of_gradients_hessians : smaller_leaf_splits->sum_of_gradients_hessians;
  const data_size_t num_data = IS_LARGER ? global_num_data_in_larger_leaf : global_num_data_in_smaller_leaf;
  const double parent_output = IS_LARGER ? larger_leaf_splits->leaf_value : smaller_leaf_splits->leaf_value;
  const unsigned int output_offset = IS_LARGER ? (task_index + num_tasks) : task_index;
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  // Both extra_trees (USE_RAND) and cat_random_search consume the per-task RNG
  // state, and the host allocates it for either, so the pointer must not be
  // gated on USE_RAND alone.
  CUDARandom* cuda_random = (USE_RAND || cat_random_search > 0) ?
    (IS_LARGER ? cuda_randoms + task_index * 2 + 1 : cuda_randoms + task_index * 2) : nullptr;
  const bool use_16bit_bin = IS_LARGER ? (larger_leaf_num_bits_in_histogram_bin <= 16) : (smaller_leaf_num_bits_in_histogram_bin <= 16);
  if (is_feature_used_bytree[inner_feature_index]) {
    if (task->is_categorical) {
      // categorical search: per-bin math runs in double either way, so the
      // quantized histogram is unpacked per bin through a reader and the
      // shared categorical body serves both pipelines; the packed int64 leaf
      // totals seed the child leaf structs (writer blocks)
      const double sum_gradients = static_cast<double>(
        static_cast<int32_t>((sum_gradients_hessians & 0xffffffff00000000) >> 32)) * (*grad_scale);
      const double sum_hessians = static_cast<double>(
        static_cast<int32_t>(sum_gradients_hessians & 0x00000000ffffffff)) * (*hess_scale) + 2 * kEpsilon;
      const int8_t* hist_base = reinterpret_cast<const int8_t*>(
        IS_LARGER ? larger_leaf_splits->hist_in_leaf : smaller_leaf_splits->hist_in_leaf);
      if (use_16bit_bin) {
        const CatHistReaderQuant16 hist_reader{
          reinterpret_cast<const int32_t*>(hist_base) + task->hist_offset,
          static_cast<double>(*grad_scale), static_cast<double>(*hess_scale)};
        FindBestSplitsForLeafKernelCategoricalInner<USE_RAND, USE_L1, USE_SMOOTHING, CatHistReaderQuant16>(
          hist_reader, task, cuda_random,
          lambda_l1, lambda_l2, path_smooth, max_delta_step,
          min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
          cat_smooth, cat_l2, max_cat_threshold, min_data_per_group, cat_random_search,
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
          cat_smooth, cat_l2, max_cat_threshold, min_data_per_group, cat_random_search,
          parent_gain, sum_gradients, sum_hessians, num_data, parent_output,
          sum_gradients_hessians, out);
      }
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
            max_delta_step,
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
            max_delta_step,
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
            max_delta_step,
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
            max_delta_step,
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
    // CEGB: subtract cost penalty (see numerical kernel for rationale).
    __syncthreads();
    if (threadIdx.x == 0 && out->is_valid) {
      double delta = cegb_tradeoff_times_penalty_split * static_cast<double>(num_data);
      if (cuda_task_cegb_penalty != nullptr) {
        delta += cuda_task_cegb_penalty[task_index];
      }
      out->gain -= delta;
    }
  } else {
    out->is_valid = false;
  }
}

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, bool REVERSE, bool USE_MC>
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
  const double max_delta_step,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  // input parent node information
  const double parent_gain,
  const double sum_gradients,
  const double sum_hessians,
  const data_size_t num_data,
  const double parent_output,
  // monotone constraint information for this leaf
  const double leaf_constraint_min,
  const double leaf_constraint_max,
  // output parameters
  CUDASplitInfo* cuda_best_split_info,
  // buffer
  hist_t* hist_grad_buffer_ptr,
  hist_t* hist_hess_buffer_ptr,
  data_size_t* hist_cnt_buffer_ptr) {
  const int8_t monotone_constraint = task->monotone_type;
  const double cnt_factor = num_data / sum_hessians;
  const double min_gain_shift = parent_gain + min_gain_to_split;

  cuda_best_split_info->is_valid = false;
  double local_gain = 0.0f;
  bool threshold_found = false;
  uint32_t threshold_value = 0;
  // Scan position (the shared-memory kernel's threadIdx_x) of this thread's
  // best candidate. Above blockDim.x bins a thread owns SEVERAL positions
  // (bin, bin + blockDim.x, ...), so the block reduce must tie-break on the
  // position, not the thread index: CPU's strict-'>' scan keeps the EARLIEST
  // position on an exact gain tie, and positions congruent mod blockDim.x
  // do not order like thread indices across the stride boundary.
  uint32_t local_best_pos = 0;
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
  // NA_AS_MISSING head (CPU: feature_histogram.hpp, offset == 1): slot 0 is a
  // "missing" pseudo-bin holding the leaf totals minus every real bin, and its
  // COUNT is num_data minus the sum of the per-bin roundings -- not a rounding
  // of the pseudo-bin's own hessian, which differs by whole rows and so flips
  // min_data_in_leaf gating. Carried out of the staging branch because the
  // count fix-up lands after the shared per-bin count loop below.
  const bool na_missing_head = (!REVERSE && task->na_as_missing && task->mfb_offset == 1);
  data_size_t na_cnt_non_default = 0;
  if (!REVERSE) {
    if (na_missing_head) {
      uint32_t bin_start = threadIdx_x > 0 ? threadIdx_x : blockDim.x;
      data_size_t thread_cnt = 0;
      for (unsigned int bin = bin_start; bin < static_cast<uint32_t>(task->num_bin); bin += blockDim.x) {
        const unsigned int bin_offset = (bin - 1) << 1;
        const hist_t grad = feature_hist_ptr[bin_offset];
        const hist_t hess = feature_hist_ptr[bin_offset + 1];
        // This feature's scratch region holds num_bin - 1 slots (mfb_offset
        // == 1 here); bin == num_bin - 1 is only ever summed, never scanned
        // (the scan stops at num_bin - 2), and storing it would land one past
        // the region -- in the NEXT feature's slot 0, which another block of
        // this launch stages and scans concurrently.
        if (bin + 1 < static_cast<uint32_t>(task->num_bin)) {
          hist_grad_buffer_ptr[bin] = grad;
          hist_hess_buffer_ptr[bin] = hess;
        }
        // Per-bin rounding, summed over every real bin (the last one has no
        // scratch slot but still belongs to the non-default mass). Integer
        // addends are order-invariant, so the parallel reduce is exact.
        thread_cnt += static_cast<data_size_t>(CUDARoundInt(hess * cnt_factor));
      }
      na_cnt_non_default = ShuffleReduceSum<data_size_t>(
        thread_cnt, reinterpret_cast<data_size_t*>(shared_double_buffer), blockDim.x);
      // Totals minus every real bin, subtracted in CPU's ascending bin order.
      // The hessian seed is sum_hessians - kEpsilon (CPU's), which is why the
      // generic bin-0 kEpsilon ADD below is skipped for this head.
      double missing_grad = sum_gradients;
      double missing_hess = sum_hessians - kEpsilon;
      GlobalMemorySequentialSubtractPair(feature_hist_ptr, feature_num_bin_minus_offset,
                                         &missing_grad, &missing_hess);
      if (threadIdx_x == 0) {
        hist_grad_buffer_ptr[0] = missing_grad;
        hist_hess_buffer_ptr[0] = missing_hess;
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
      // The na_as_missing gate is INDEPENDENT of skip_default_bin (see the
      // shared-memory kernel): a NaN feature's reverse scan leaves position 0
      // empty because that slot is the missing mass, which the reverse
      // direction assigns to the left side wholesale. Folding the gate into
      // skip_sum -- `bin >= na_as_missing && (skip_default_bin && ...)` --
      // made it dead, because NaN tasks carry skip_default_bin == false: the
      // scan then staged and scored a bin the CPU never considers.
      const bool skip_sum =
        (task->skip_default_bin && (task->num_bin - 1 - bin) == static_cast<int>(task->default_bin));
      if (bin >= static_cast<unsigned int>(task->na_as_missing) && !skip_sum) {
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
  // Per-bin count roundings summed in scan order, like CPU's
  // right_count += RoundInt(hess * cnt_factor) per bin (see the shared-memory
  // variant); rounding the summed hessian instead can differ by a count.
  // Bin 0's kEpsilon (added below, after this loop reads raw values) shifts
  // nothing at RoundInt granularity.
  for (unsigned int bin = threadIdx_x; bin < feature_num_bin_minus_offset; bin += blockDim.x) {
    hist_cnt_buffer_ptr[bin] = static_cast<data_size_t>(
      CUDARoundInt(hist_hess_buffer_ptr[bin] * cnt_factor));
  }
  __syncthreads();
  if (threadIdx_x == 0) {
    if (na_missing_head) {
      // CPU's left_count seed for the missing pseudo-bin; the loop above
      // rounded the pseudo-bin's own hessian instead, which is a different
      // integer. The hessian already carries CPU's -kEpsilon seed.
      hist_cnt_buffer_ptr[0] = num_data - na_cnt_non_default;
    } else {
      hist_hess_buffer_ptr[0] += kEpsilon;
    }
  }
  local_gain = kMinScore;
  // CPU-order fp64 scans; the integer count prefix is order-invariant and
  // keeps the parallel scan.
  GlobalMemorySequentialPrefixSum(hist_grad_buffer_ptr, static_cast<size_t>(feature_num_bin_minus_offset));
  GlobalMemorySequentialPrefixSum(hist_hess_buffer_ptr, static_cast<size_t>(feature_num_bin_minus_offset));
  GlobalMemoryPrefixSum<data_size_t>(hist_cnt_buffer_ptr, static_cast<size_t>(feature_num_bin_minus_offset));
  if (REVERSE) {
    for (unsigned int bin = threadIdx_x; bin < feature_num_bin_minus_offset; bin += blockDim.x) {
      // na_as_missing gates the candidate independently of skip_default_bin;
      // see the staging loop above for why conflating the two made it dead.
      const bool skip_sum =
        (task->skip_default_bin && (task->num_bin - 1 - bin) == static_cast<int>(task->default_bin));
      // bin == num_bin - 1 (reachable when mfb_offset == 0) would encode
      // threshold = num_bin - 2 - bin = -1, a split the CPU reverse scan never
      // considers; without this bound it escaped as threshold 0xFFFFFFFF and the
      // host indexed bin_upper_bound with it. The shared-memory kernel has the
      // same bound (threadIdx_x <= task->num_bin - 2).
      if (bin >= static_cast<unsigned int>(task->na_as_missing) &&
          !skip_sum && static_cast<int>(bin) <= task->num_bin - 2) {
        const double sum_right_gradient = hist_grad_buffer_ptr[bin];
        const double sum_right_hessian = hist_hess_buffer_ptr[bin];
        const data_size_t right_count = hist_cnt_buffer_ptr[bin];
        const double sum_left_gradient = sum_gradients - sum_right_gradient;
        const double sum_left_hessian = sum_hessians - sum_right_hessian;
        const data_size_t left_count = num_data - right_count;
        if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
          sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf &&
          (!USE_RAND || static_cast<int>(task->num_bin - 2 - bin) == rand_threshold)) {
          double current_gain = USE_MC ?
            CUDALeafSplits::GetSplitGainsMC<USE_L1, USE_SMOOTHING>(
              sum_left_gradient, sum_left_hessian, sum_right_gradient,
              sum_right_hessian, lambda_l1,
              lambda_l2, path_smooth, max_delta_step, left_count, right_count, parent_output,
              leaf_constraint_min, leaf_constraint_max, monotone_constraint) :
            CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
              sum_left_gradient, sum_left_hessian, sum_right_gradient,
              sum_right_hessian, lambda_l1,
              lambda_l2, path_smooth, max_delta_step, left_count, right_count, parent_output);
          // gain with split is worse than without split. Running MAXIMUM over
          // the grid-stride bins this thread owns (strict '>' keeps the
          // earliest scan position on an exact tie, like CPU's sequential
          // scan): a bare `> min_gain_shift` overwrite let a later, worse bin
          // clobber the thread's best candidate whenever the feature has more
          // than blockDim.x bins.
          if (current_gain > min_gain_shift &&
              current_gain - min_gain_shift > local_gain) {
            local_gain = current_gain - min_gain_shift;
            threshold_value = static_cast<uint32_t>(task->num_bin - 2 - bin);
            local_best_pos = bin;
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
        const data_size_t left_count = hist_cnt_buffer_ptr[bin];
        const double sum_right_gradient = sum_gradients - sum_left_gradient;
        const double sum_right_hessian = sum_hessians - sum_left_hessian;
        const data_size_t right_count = num_data - left_count;
        if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
          sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf &&
          (!USE_RAND || static_cast<int>(bin + task->mfb_offset) == rand_threshold)) {
          double current_gain = USE_MC ?
            CUDALeafSplits::GetSplitGainsMC<USE_L1, USE_SMOOTHING>(
              sum_left_gradient, sum_left_hessian, sum_right_gradient,
              sum_right_hessian, lambda_l1,
              lambda_l2, path_smooth, max_delta_step, left_count, right_count, parent_output,
              leaf_constraint_min, leaf_constraint_max, monotone_constraint) :
            CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
              sum_left_gradient, sum_left_hessian, sum_right_gradient,
              sum_right_hessian, lambda_l1,
              lambda_l2, path_smooth, max_delta_step, left_count, right_count, parent_output);
          // gain with split is worse than without split. Running MAXIMUM over
          // the thread's grid-stride bins; see the REVERSE branch.
          if (current_gain > min_gain_shift &&
              current_gain - min_gain_shift > local_gain) {
            local_gain = current_gain - min_gain_shift;
            threshold_value = (task->na_as_missing && task->mfb_offset == 1) ?
              bin : static_cast<uint32_t>(bin + task->mfb_offset);
            local_best_pos = bin;
            threshold_found = true;
          }
        }
      }
    }
  }
  __syncthreads();
  // Reduce on the scan POSITION of each thread's best candidate, not the
  // thread index: with more bins than threads the earliest tied position is
  // what CPU's strict-'>' scan keeps, and positions across the stride
  // boundary do not order like thread indices. Positions are disjoint across
  // threads (position mod blockDim.x == threadIdx.x), so the winner check
  // below selects exactly one writer.
  const uint32_t result = ReduceBestGain(local_gain, threshold_found, local_best_pos, shared_double_buffer, shared_found_buffer, shared_thread_index_buffer);
  if (threadIdx_x == 0) {
    best_thread_index = result;
  }
  __syncthreads();
  if (threshold_found && local_best_pos == best_thread_index) {
    cuda_best_split_info->is_valid = true;
    cuda_best_split_info->threshold = threshold_value;
    cuda_best_split_info->gain = local_gain * task->penalty;
    cuda_best_split_info->default_left = task->assume_out_default_left;
    if (REVERSE) {
      // CPU's output chain (see the shared-memory variant): left-basis sums
      // carry the kEpsilon seed; stored fields subtract kEpsilon through
      // EXACTLY the CPU expression sequence; the count is the per-bin-
      // rounding prefix.
      const unsigned int best_bin = static_cast<uint32_t>(task->num_bin - 2 - threshold_value);
      const double sum_left_gradient = sum_gradients - hist_grad_buffer_ptr[best_bin];
      const double sum_left_hessian = sum_hessians - hist_hess_buffer_ptr[best_bin];
      const data_size_t left_count = num_data - hist_cnt_buffer_ptr[best_bin];
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      // Unconstrained outputs first (max_delta_step cap included). The leaf VALUE
      // is clamped into the constraint range (MC), but the leaf GAIN stored for
      // future splits must stay unconstrained: it becomes the child's parent_gain
      // / min_gain_shift baseline, and the CPU reference (BeforeNumerical)
      // recomputes that baseline as the unconstrained GetLeafGain of the leaf's
      // sums. The max_delta_step cap is part of the analytic (unconstrained)
      // output, applied before the MC clamp, matching the CPU ordering.
      const double left_output_unconstrained =
        CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
          sum_left_hessian, lambda_l1, lambda_l2, path_smooth, max_delta_step, left_count, parent_output);
      const double right_output_unconstrained =
        CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
          sum_right_hessian, lambda_l1, lambda_l2, path_smooth, max_delta_step, right_count, parent_output);
      const double left_output = USE_MC ?
        (left_output_unconstrained < leaf_constraint_min ? leaf_constraint_min :
          (left_output_unconstrained > leaf_constraint_max ? leaf_constraint_max : left_output_unconstrained)) :
        left_output_unconstrained;
      const double right_output = USE_MC ?
        (right_output_unconstrained < leaf_constraint_min ? leaf_constraint_min :
          (right_output_unconstrained > leaf_constraint_max ? leaf_constraint_max : right_output_unconstrained)) :
        right_output_unconstrained;
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian - kEpsilon;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_hessians - sum_left_hessian - kEpsilon;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient,
        sum_left_hessian, lambda_l1, lambda_l2, left_output_unconstrained);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient,
        sum_right_hessian, lambda_l1, lambda_l2, right_output_unconstrained);
    } else {
      // CPU's output chain; see the REVERSE branch.
      const unsigned int best_bin = (task->na_as_missing && task->mfb_offset == 1) ?
        threshold_value : static_cast<uint32_t>(threshold_value - task->mfb_offset);
      const double sum_left_gradient = hist_grad_buffer_ptr[best_bin];
      const double sum_left_hessian = hist_hess_buffer_ptr[best_bin];
      const data_size_t left_count = hist_cnt_buffer_ptr[best_bin];
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      // Unconstrained outputs first (max_delta_step cap included). The leaf VALUE
      // is clamped into the constraint range (MC), but the leaf GAIN stored for
      // future splits must stay unconstrained: it becomes the child's parent_gain
      // / min_gain_shift baseline, and the CPU reference (BeforeNumerical)
      // recomputes that baseline as the unconstrained GetLeafGain of the leaf's
      // sums. The max_delta_step cap is part of the analytic (unconstrained)
      // output, applied before the MC clamp, matching the CPU ordering.
      const double left_output_unconstrained =
        CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
          sum_left_hessian, lambda_l1, lambda_l2, path_smooth, max_delta_step, left_count, parent_output);
      const double right_output_unconstrained =
        CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
          sum_right_hessian, lambda_l1, lambda_l2, path_smooth, max_delta_step, right_count, parent_output);
      const double left_output = USE_MC ?
        (left_output_unconstrained < leaf_constraint_min ? leaf_constraint_min :
          (left_output_unconstrained > leaf_constraint_max ? leaf_constraint_max : left_output_unconstrained)) :
        left_output_unconstrained;
      const double right_output = USE_MC ?
        (right_output_unconstrained < leaf_constraint_min ? leaf_constraint_min :
          (right_output_unconstrained > leaf_constraint_max ? leaf_constraint_max : right_output_unconstrained)) :
        right_output_unconstrained;
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_left_gradient,
        sum_left_hessian, lambda_l1, lambda_l2, left_output_unconstrained);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<USE_L1>(sum_right_gradient,
        sum_right_hessian, lambda_l1, lambda_l2, right_output_unconstrained);
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
  const double max_delta_step,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  const int cat_random_search,
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
  data_size_t* hist_cnt_buffer_ptr,
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
              l2, path_smooth, max_delta_step, other_count, cnt, parent_output);
            if (current_gain > min_gain_shift && current_gain - min_gain_shift > local_gain) {
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
      // write into the slab slot pre-assigned by AllocateCatVectorsKernel;
      // the kernel must never allocate (the slab is the only storage whose
      // lifetime the host controls)
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
        sum_left_hessian, lambda_l1, l2, path_smooth, max_delta_step, left_count, parent_output);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
        sum_right_hessian, lambda_l1, l2, path_smooth, max_delta_step, right_count, parent_output);
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
    // Registered per thread: the count buffer holds only the LAST pass's
    // prefixes at write-out time, so the winning pass's count must ride along
    // with the other best_* registers.
    data_size_t best_left_count = 0;
    if (cat_random_search > 0) {
      // Compact the cat_smooth-passing bins into hist_index_buffer_ptr -- the
      // RANDOM search needs the candidate list, not the sorted order, so the
      // bitonic sort and the prefix scans below are skipped entirely. The count
      // buffer carries the compaction scan; the index buffer's -1 sentinel for
      // filtered bins has no reader on this path.
      for (int bin = threadIdx_x; bin < bin_end; bin += static_cast<int>(blockDim.x)) {
        data_size_t is_candidate = 0;
        if (bin >= bin_start &&
            CUDARoundInt(feature_hist_ptr[(bin << 1) + 1] * cnt_factor) >= cat_smooth) {
          is_candidate = 1;
        }
        hist_cnt_buffer_ptr[bin] = is_candidate;
      }
      __syncthreads();
      GlobalMemoryPrefixSum<data_size_t>(hist_cnt_buffer_ptr, static_cast<size_t>(bin_end));
      const int num_cand = bin_end > 0 ? static_cast<int>(hist_cnt_buffer_ptr[bin_end - 1]) : 0;
      for (int bin = threadIdx_x; bin < bin_end; bin += static_cast<int>(blockDim.x)) {
        const data_size_t inclusive = hist_cnt_buffer_ptr[bin];
        const data_size_t exclusive = bin > 0 ? hist_cnt_buffer_ptr[bin - 1] : 0;
        if (inclusive > exclusive) {
          hist_index_buffer_ptr[inclusive - 1] = bin;
        }
      }
      __syncthreads();
      const CatHistReaderF64 hist_reader{feature_hist_ptr};
      FindBestSplitsForLeafKernelCategoricalRandom<USE_L1, USE_SMOOTHING, CatHistReaderF64, data_size_t>(
        hist_reader, task, cuda_random, hist_index_buffer_ptr, num_cand,
        cat_random_search, lambda_l1, l2, path_smooth, max_delta_step,
        min_data_in_leaf, min_sum_hessian_in_leaf, max_cat_threshold, min_data_per_group,
        min_gain_shift, cnt_factor, sum_gradients, sum_hessians, num_data, parent_output,
        /*sum_gradients_hessians_total=*/0, cuda_best_split_info);
      return;
    }
    // Grid-stride over every bin: thread t owns bins t, t + blockDim, ... The
    // original loop started at bin = 0 for every thread, so it only ever touched
    // bins that are multiples of blockDim and raced on the writes -- leaving most
    // of hist_stat_buffer_ptr unfilled. Two further consequences of that bug:
    //  * hist_index_buffer_ptr[bin] must record each bin's own index (it is read
    //    back after the sort to map a sorted position to its histogram bin);
    //    storing threadIdx_x was only correct for the first block.
    //  * is_valid_bin must accumulate across the stride so used_bin counts every
    //    valid category, not one-per-thread.
    // Bins below bin_start are written with the kMaxScore sentinel so they sort to
    // the end, matching the shared-memory categorical kernel.
    for (int bin = static_cast<int>(threadIdx_x); bin < bin_end; bin += static_cast<int>(blockDim.x)) {
      if (bin >= bin_start) {
        const int bin_offset = (bin << 1);
        const double hess = feature_hist_ptr[bin_offset + 1];
        if (CUDARoundInt(hess * cnt_factor) >= cat_smooth) {
          const double grad = feature_hist_ptr[bin_offset];
          hist_stat_buffer_ptr[bin] = grad / (hess + cat_smooth);
          hist_index_buffer_ptr[bin] = bin;
          ++is_valid_bin;
        } else {
          hist_stat_buffer_ptr[bin] = kMaxScore;
          hist_index_buffer_ptr[bin] = -1;
        }
      } else {
        hist_stat_buffer_ptr[bin] = kMaxScore;
        hist_index_buffer_ptr[bin] = -1;
      }
    }
    __syncthreads();
    const int local_used_bin = ShuffleReduceSum<uint16_t>(is_valid_bin, shared_mem_buffer_uint16, blockDim.x);
    if (threadIdx_x == 0) {
      used_bin = local_used_bin;
    }
    __syncthreads();
    // BitonicArgSortDevice's MAX_DEPTH must be log2(BLOCK_DIM) + 1 (see the
    // BITONIC_SORT_NUM_ELEMENTS/2 -> 10 and /4 -> 9 dispatch in cuda_algorithms.hpp):
    // it selects the boundary between the in-block phase and the cross-block merge
    // phase. NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER is 256, so the correct depth is
    // 9, not 11. Passing 11 (the value for a 1024-wide block) let the in-block phase
    // run two levels too deep -- reading shared memory out of bounds and skipping the
    // cross-block merge -- which produced a wrong order for categoricals above 256 bins
    // (wrong split vs CPU) and read past the sort buffers (illegal memory access at
    // larger category counts).
    // STABLE = true: CPU sorts the categories with std::stable_sort
    // (feature_histogram.cpp), so tied gradient/hessian ratios keep ascending
    // category order. Without a stable device sort the tied categories get an
    // arbitrary order, which selects a different categorical threshold set than
    // CPU whenever ratios tie (a data-dependent divergence at > 256 categories).
    BitonicArgSortDevice<double, data_size_t, true, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 9, /*STABLE=*/true>(
      hist_stat_buffer_ptr, hist_index_buffer_ptr, task->num_bin - task->mfb_offset);
    const int max_num_cat = min(max_cat_threshold, (used_bin + 1) / 2);
    // Number of candidate thresholds CPU would walk in each direction. The
    // prefix-hessian scan the group check needs is already materialised in
    // hist_hess_buffer_ptr, so each thread can replay it directly.
    const int num_cat_thresholds = min(used_bin, max_num_cat);
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
      // CPU rounds each category's count separately and sums the roundings.
      hist_cnt_buffer_ptr[bin] = static_cast<data_size_t>(
        CUDARoundInt(feature_hist_ptr[bin_offset + 1] * cnt_factor));
    }
    if (threadIdx_x == 0) {
      hist_hess_buffer_ptr[0] += kEpsilon;
    }
    __syncthreads();
    GlobalMemoryPrefixSum<double>(hist_grad_buffer_ptr, static_cast<size_t>(bin_end));
    __syncthreads();
    GlobalMemoryPrefixSum<double>(hist_hess_buffer_ptr, static_cast<size_t>(bin_end));
    __syncthreads();
    GlobalMemoryPrefixSum<data_size_t>(hist_cnt_buffer_ptr, static_cast<size_t>(bin_end));
    for (int bin = static_cast<int>(threadIdx_x); bin < used_bin && bin < max_num_cat; bin += static_cast<int>(blockDim.x)) {
      const double sum_left_gradient = hist_grad_buffer_ptr[bin];
      const double sum_left_hessian = hist_hess_buffer_ptr[bin];
      const data_size_t left_count = hist_cnt_buffer_ptr[bin];
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (SequentialCategoricalGroupAccepted(hist_hess_buffer_ptr, hist_cnt_buffer_ptr, bin, num_cat_thresholds,
          num_data, sum_hessians, min_data_in_leaf, min_sum_hessian_in_leaf,
          min_data_per_group)) {
        double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1,
          l2, path_smooth, max_delta_step, left_count, right_count, parent_output);
        // Keep the RUNNING MAXIMUM across BOTH the left-to-right and
        // right-to-left passes (and, above 256 bins, across the grid-stride
        // bins a thread owns): a bare `> min_gain_shift` test would overwrite
        // local_gain on every qualifying threshold, letting the right-to-left
        // pass clobber a better left-to-right split.
        if (current_gain > min_gain_shift && current_gain - min_gain_shift > local_gain) {
          local_gain = current_gain - min_gain_shift;
          threshold_found = true;
          best_dir = 1;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
          best_threshold = bin;
          best_left_count = left_count;
        }
      }
    }
    __syncthreads();

    // right to left
    for (int bin = static_cast<int>(threadIdx_x); bin < used_bin && bin < max_num_cat; bin += static_cast<int>(blockDim.x)) {
      const int bin_offset = (hist_index_buffer_ptr[used_bin - 1 - bin] << 1);
      hist_grad_buffer_ptr[bin] = feature_hist_ptr[bin_offset];
      hist_hess_buffer_ptr[bin] = feature_hist_ptr[bin_offset + 1];
      hist_cnt_buffer_ptr[bin] = static_cast<data_size_t>(
        CUDARoundInt(feature_hist_ptr[bin_offset + 1] * cnt_factor));
    }
    if (threadIdx_x == 0) {
      hist_hess_buffer_ptr[0] += kEpsilon;
    }
    __syncthreads();
    GlobalMemoryPrefixSum<double>(hist_grad_buffer_ptr, static_cast<size_t>(bin_end));
    __syncthreads();
    GlobalMemoryPrefixSum<double>(hist_hess_buffer_ptr, static_cast<size_t>(bin_end));
    __syncthreads();
    GlobalMemoryPrefixSum<data_size_t>(hist_cnt_buffer_ptr, static_cast<size_t>(bin_end));
    for (int bin = static_cast<int>(threadIdx_x); bin < used_bin && bin < max_num_cat; bin += static_cast<int>(blockDim.x)) {
      const double sum_left_gradient = hist_grad_buffer_ptr[bin];
      const double sum_left_hessian = hist_hess_buffer_ptr[bin];
      const data_size_t left_count = hist_cnt_buffer_ptr[bin];
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (SequentialCategoricalGroupAccepted(hist_hess_buffer_ptr, hist_cnt_buffer_ptr, bin, num_cat_thresholds,
          num_data, sum_hessians, min_data_in_leaf, min_sum_hessian_in_leaf,
          min_data_per_group)) {
        double current_gain = CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING>(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1,
          l2, path_smooth, max_delta_step, left_count, right_count, parent_output);
        // Keep the RUNNING MAXIMUM across BOTH the left-to-right and
        // right-to-left passes (and, above 256 bins, across the grid-stride
        // bins a thread owns): a bare `> min_gain_shift` test would overwrite
        // local_gain on every qualifying threshold, letting the right-to-left
        // pass clobber a better left-to-right split.
        if (current_gain > min_gain_shift && current_gain - min_gain_shift > local_gain) {
          local_gain = current_gain - min_gain_shift;
          threshold_found = true;
          best_dir = -1;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
          best_threshold = bin;
          best_left_count = left_count;
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
      // slab slot pre-assigned by AllocateCatVectorsKernel; capacity is
      // max_num_categories_in_split, which bounds best_threshold + 1 exactly
      // as it bounds the shared-memory twin's threadIdx_x + 1
      cuda_best_split_info->gain = local_gain * task->penalty;
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
      // CPU stores the sum of per-category roundings (see the shared variant).
      // The count comes from the winning pass's register: the cnt buffer holds
      // only the right-to-left prefixes by now.
      const data_size_t left_count = best_left_count;
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_left_gradient,
        sum_left_hessian, lambda_l1, l2, path_smooth, max_delta_step, left_count, parent_output);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(sum_right_gradient,
        sum_right_hessian, lambda_l1, l2, path_smooth, max_delta_step, right_count, parent_output);
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

template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING, bool IS_LARGER, bool USE_MC>
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
  const double max_delta_step,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  const int cat_random_search,
  // output
  CUDASplitInfo* cuda_best_split_info,
  // global num data in leaf
  const data_size_t global_num_data_in_smaller_leaf,
  const data_size_t global_num_data_in_larger_leaf,
  // CEGB (passed via FindBestSplitsForLeafKernel_ARGS, before CONSTRAINT_ARGS)
  const double* cuda_task_cegb_penalty,
  const double cegb_tradeoff_times_penalty_split,
  // monotone leaf constraints (passed via FindBestSplitsForLeafKernel_CONSTRAINT_ARGS)
  const double smaller_leaf_constraint_min,
  const double smaller_leaf_constraint_max,
  const double larger_leaf_constraint_min,
  const double larger_leaf_constraint_max,
  // buffer
  hist_t* feature_hist_grad_buffer,
  hist_t* feature_hist_hess_buffer,
  hist_t* feature_hist_stat_buffer,
  data_size_t* feature_hist_index_buffer,
  data_size_t* feature_hist_cnt_buffer,
  const uint32_t hist_buffer_stride) {
  const double leaf_constraint_min = IS_LARGER ? larger_leaf_constraint_min : smaller_leaf_constraint_min;
  const double leaf_constraint_max = IS_LARGER ? larger_leaf_constraint_max : smaller_leaf_constraint_max;
  const unsigned int task_index = blockIdx.x;
  const SplitFindTask* task = tasks + task_index;
  const double sum_gradients = IS_LARGER ? larger_leaf_splits->sum_of_gradients : smaller_leaf_splits->sum_of_gradients;
  const double sum_hessians = (IS_LARGER ? larger_leaf_splits->sum_of_hessians : smaller_leaf_splits->sum_of_hessians) + 2 * kEpsilon;
  const data_size_t num_data = IS_LARGER ? global_num_data_in_larger_leaf : global_num_data_in_smaller_leaf;
  const double parent_output = IS_LARGER ? larger_leaf_splits->leaf_value : smaller_leaf_splits->leaf_value;
  // CPU recomputes the parent's gain from the leaf sums at find time
  // (feature_histogram.hpp gain_shift); the leaf struct's stored gain is the
  // same quantity through GetLeafGainGivenOutput, whose low bits differ -- and
  // a min_gain_shift off by one noise quantum flips split VALIDITY and
  // ranking on plateau gains. Same formula from the same sums, same bits.
  // (The discretized kernels keep the stored gain: the quantized flow has its
  // own shift semantics and no CPU bit-parity contract.)
  const double parent_gain = CUDALeafSplits::GetLeafGain<USE_L1, USE_SMOOTHING>(
      sum_gradients, sum_hessians, lambda_l1, lambda_l2, path_smooth,
      max_delta_step, num_data, parent_output);
  const unsigned int output_offset = IS_LARGER ? (task_index + num_tasks) : task_index;
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  // Both extra_trees (USE_RAND) and cat_random_search consume the per-task RNG
  // state, and the host allocates it for either, so the pointer must not be
  // gated on USE_RAND alone.
  CUDARandom* cuda_random = (USE_RAND || cat_random_search > 0) ?
    (IS_LARGER ? cuda_randoms + task_index * 2 + 1: cuda_randoms + task_index * 2) : nullptr;
  if (is_feature_used_bytree[task->inner_feature_index]) {
    const uint32_t hist_offset = task->hist_offset;
    const hist_t* hist_ptr = (IS_LARGER ? larger_leaf_splits->hist_in_leaf : smaller_leaf_splits->hist_in_leaf) + hist_offset * 2;
    // The scratch buffers below hold one element per histogram bin and are sized
    // num_total_bin_ (= feature_hist_offsets.back()), so a feature's region begins
    // at its bin offset hist_offset -- NOT hist_offset * 2. The * 2 is correct only
    // for hist_ptr above, whose entries are (grad, hess) pairs. With * 2 the sort and
    // the prefix scans ran off the end of these buffers for any feature with
    // hist_offset >= 1, reading adjacent (stale) memory and producing data-dependent
    // wrong splits.
    // The forward and reverse tasks of one numerical feature share the same
    // hist_offset but run as DIFFERENT BLOCKS of this launch: without a
    // per-direction region they race on the staging buffers, silently
    // corrupting every split decision (any run with a >256-bin feature took
    // this path and trained garbage). Reverse tasks use the second half.
    const uint32_t dir_offset = task->reverse ? hist_buffer_stride : 0;
    hist_t* hist_grad_buffer_ptr = feature_hist_grad_buffer + dir_offset + hist_offset;
    hist_t* hist_hess_buffer_ptr = feature_hist_hess_buffer + dir_offset + hist_offset;
    hist_t* hist_stat_buffer_ptr = feature_hist_stat_buffer + hist_offset;
    data_size_t* hist_index_buffer_ptr = feature_hist_index_buffer + hist_offset;
    data_size_t* hist_cnt_buffer_ptr = feature_hist_cnt_buffer + dir_offset + hist_offset;
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
        max_delta_step,
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        min_gain_to_split,
        cat_smooth,
        cat_l2,
        max_cat_threshold,
        min_data_per_group,
        cat_random_search,
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
        hist_cnt_buffer_ptr,
        // output parameters
        out);
    } else {
      if (!task->reverse) {
        FindBestSplitsForLeafKernelInner_GlobalMemory<USE_RAND, USE_L1, USE_SMOOTHING, false, USE_MC>(
          // input feature information
          hist_ptr,
          // input task information
          task,
          cuda_random,
          // input config parameter values
          lambda_l1,
          lambda_l2,
          path_smooth,
          max_delta_step,
          min_data_in_leaf,
          min_sum_hessian_in_leaf,
          min_gain_to_split,
          // input parent node information
          parent_gain,
          sum_gradients,
          sum_hessians,
          num_data,
          parent_output,
          // monotone constraint information
          leaf_constraint_min,
          leaf_constraint_max,
          // output parameters
          out,
          // buffer
          hist_grad_buffer_ptr,
          hist_hess_buffer_ptr,
          hist_cnt_buffer_ptr);
      } else {
        FindBestSplitsForLeafKernelInner_GlobalMemory<USE_RAND, USE_L1, USE_SMOOTHING, true, USE_MC>(
          // input feature information
          hist_ptr,
          // input task information
          task,
          cuda_random,
          // input config parameter values
          lambda_l1,
          lambda_l2,
          path_smooth,
          max_delta_step,
          min_data_in_leaf,
          min_sum_hessian_in_leaf,
          min_gain_to_split,
          // input parent node information
          parent_gain,
          sum_gradients,
          sum_hessians,
          num_data,
          parent_output,
          // monotone constraint information
          leaf_constraint_min,
          leaf_constraint_max,
          // output parameters
          out,
          // buffer
          hist_grad_buffer_ptr,
          hist_hess_buffer_ptr,
          hist_cnt_buffer_ptr);
      }
    }
    // CEGB: subtract cost penalty (see numerical kernel for rationale).
    __syncthreads();
    if (threadIdx.x == 0 && out->is_valid) {
      double delta = cegb_tradeoff_times_penalty_split * static_cast<double>(num_data);
      if (cuda_task_cegb_penalty != nullptr) {
        delta += cuda_task_cegb_penalty[task_index];
      }
      out->gain -= delta;
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
  const data_size_t global_num_data_in_larger_leaf, \
  const double smaller_leaf_constraint_min, \
  const double smaller_leaf_constraint_max, \
  const double larger_leaf_constraint_min, \
  const double larger_leaf_constraint_max

#define LaunchFindBestSplitsForLeafKernel_ARGS \
  smaller_leaf_splits, \
  larger_leaf_splits, \
  smaller_leaf_index, \
  larger_leaf_index, \
  is_smaller_leaf_valid, \
  is_larger_leaf_valid, \
  global_num_data_in_smaller_leaf, \
  global_num_data_in_larger_leaf, \
  smaller_leaf_constraint_min, \
  smaller_leaf_constraint_max, \
  larger_leaf_constraint_min, \
  larger_leaf_constraint_max

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
    max_delta_step_, \
    cat_smooth_, \
    cat_l2_, \
    max_cat_threshold_, \
    min_data_per_group_, \
    cat_random_search_, \
    cuda_best_split_info_.RawData(), \
    global_num_data_in_smaller_leaf, \
    global_num_data_in_larger_leaf, \
    cegb_use_ ? cuda_task_cegb_penalty_.RawData() : nullptr, \
    cegb_tradeoff_times_penalty_split_

#define FindBestSplitsForLeafKernel_CONSTRAINT_ARGS \
    smaller_leaf_constraint_min, \
    smaller_leaf_constraint_max, \
    larger_leaf_constraint_min, \
    larger_leaf_constraint_max

#define GlobalMemory_Buffer_ARGS \
  cuda_feature_hist_grad_buffer_.RawData(), \
  cuda_feature_hist_hess_buffer_.RawData(), \
  cuda_feature_hist_stat_buffer_.RawData(), \
  cuda_feature_hist_index_buffer_.RawData(), \
  cuda_feature_hist_cnt_buffer_.RawData(), \
  static_cast<uint32_t>(num_total_bin_)

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
  if (FalcataFP32GainEnabled()) {
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
      if (use_monotone_constraints_) {
        FindBestSplitsForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, false, GAIN_T, true>
          <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>
          (is_feature_used_by_smaller_node, hist_fp32_, FindBestSplitsForLeafKernel_ARGS, FindBestSplitsForLeafKernel_CONSTRAINT_ARGS);
      } else {
        FindBestSplitsForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, false, GAIN_T, false>
          <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>
          (is_feature_used_by_smaller_node, hist_fp32_, FindBestSplitsForLeafKernel_ARGS, FindBestSplitsForLeafKernel_CONSTRAINT_ARGS);
      }
    }
    // No device sync here: the larger-leaf launch below waits on subtract_done_event via
    // its stream, and the smaller/larger leaves write disjoint cuda_best_split_info_ slots.
    if (is_larger_leaf_valid) {
      if (use_monotone_constraints_) {
        FindBestSplitsForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, true, GAIN_T, true>
          <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[1]>>>
          (is_feature_used_by_larger_node, hist_fp32_, FindBestSplitsForLeafKernel_ARGS, FindBestSplitsForLeafKernel_CONSTRAINT_ARGS);
      } else {
        FindBestSplitsForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, true, GAIN_T, false>
          <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[1]>>>
          (is_feature_used_by_larger_node, hist_fp32_, FindBestSplitsForLeafKernel_ARGS, FindBestSplitsForLeafKernel_CONSTRAINT_ARGS);
      }
    }
  } else {
    // Global-memory path (large-dataset fallback, not covered by the shared-memory
    // benchmarks): the construct/subtract waits above provide histogram visibility, but
    // keep the device sync because the smaller and larger launches share
    // cuda_feature_hist_grad/hess_buffer_ and must not run concurrently.
    if (is_smaller_leaf_valid) {
      if (use_monotone_constraints_) {
        FindBestSplitsForLeafKernel_GlobalMemory<USE_RAND, USE_L1, USE_SMOOTHING, false, true>
          <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>
          (is_feature_used_by_smaller_node, FindBestSplitsForLeafKernel_ARGS, FindBestSplitsForLeafKernel_CONSTRAINT_ARGS, GlobalMemory_Buffer_ARGS);
      } else {
        FindBestSplitsForLeafKernel_GlobalMemory<USE_RAND, USE_L1, USE_SMOOTHING, false, false>
          <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>
          (is_feature_used_by_smaller_node, FindBestSplitsForLeafKernel_ARGS, FindBestSplitsForLeafKernel_CONSTRAINT_ARGS, GlobalMemory_Buffer_ARGS);
      }
    }
    SynchronizeCUDADevice(__FILE__, __LINE__);
    if (is_larger_leaf_valid) {
      if (use_monotone_constraints_) {
        FindBestSplitsForLeafKernel_GlobalMemory<USE_RAND, USE_L1, USE_SMOOTHING, true, true>
          <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[1]>>>
          (is_feature_used_by_larger_node, FindBestSplitsForLeafKernel_ARGS, FindBestSplitsForLeafKernel_CONSTRAINT_ARGS, GlobalMemory_Buffer_ARGS);
      } else {
        FindBestSplitsForLeafKernel_GlobalMemory<USE_RAND, USE_L1, USE_SMOOTHING, true, false>
          <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[1]>>>
          (is_feature_used_by_larger_node, FindBestSplitsForLeafKernel_ARGS, FindBestSplitsForLeafKernel_CONSTRAINT_ARGS, GlobalMemory_Buffer_ARGS);
      }
    }
  }
}

#undef LaunchFindBestSplitsForLeafKernel_PARAMS
#undef FindBestSplitsForLeafKernel_ARGS
#undef FindBestSplitsForLeafKernel_CONSTRAINT_ARGS
#undef GlobalMemory_Buffer_ARGS

// ---- vector-leaf (multi-target) per-pair find -------------------------------
//
// One block per task, one leaf role per launch (mirroring the scalar
// FindBestSplitsForLeafKernel). The leaf is described by num_targets contiguous
// CUDALeafSplitsStructs: plane t carries the target-t gradient sums and the
// target-t histogram plane pointer; the hessian stream (and the row counts
// derived from it) is plane 0's, shared by every target. Split gain is the sum
// of the per-target gains, all arithmetic fp64 with the scalar kernel's exact
// CPU-order prefix folds, so a duplicated-target problem reproduces the scalar
// scan's candidate ranking. Supported task shapes are numerical only, with
// L1 / path smoothing / max_delta_step / extra_trees / monotone constraints /
// CEGB all gated off by the tree learner. The winner records target-0 values in
// the scalar fields (leaf_value_ mirrors target 0 everywhere in vector mode)
// and every target's child sums/outputs in the slot's vector payload.

/*! \brief hard cap on vector-leaf targets.
 *
 *  The binding constraint is the fp64 kernel's (T+1)-row dynamic shared prefix
 *  buffer: (T+1) * 256 lanes * 8B, which at T=16 is 34.8KB. A block also carries
 *  ~4.6KB of static shared (the categorical inner's sort scratch dominates), and
 *  a kernel that never calls cudaFuncSetAttribute is capped at 48KB of
 *  static+dynamic together -- so the fp64 kernels stop working at T = 21, and 16
 *  is the round number below that. Raising the cap therefore means opting the
 *  fp64 vector kernels into the larger per-block shared limit
 *  (cudaFuncAttributeMaxDynamicSharedMemorySize, ~99KB on sm_120, good to about
 *  T = 47) or restructuring the prefix to scan targets in chunks.
 *
 *  The quantized vector kernels need no prefix slab at all -- integer scans are
 *  exact and associative, so they reuse one warp-shuffle scratch per plane --
 *  and are bounded only by the per-thread arrays below (T int64 + T doubles),
 *  which spill to local memory and cost bandwidth rather than correctness. */
static constexpr int kMaxVectorTargets = 16;

// Vector-leaf categorical split search: the scalar categorical inner
// (one-hot + max_cat_threshold-sorted many-vs-many) over T gradient streams
// with the shared plane-0 hessian stream. Gains sum the per-target gains.
// The many-vs-many sort key is the summed-over-targets gradient over the
// shared smoothed hessian, (sum_t g_t) / (h + cat_smooth) -- the 1-D
// output-ordering heuristic applied to the summed objective; for duplicated
// targets it is a positive multiple of the scalar key, so the sorted order
// (and hence the greedy structure) matches scalar training exactly.
// USE_RAND (extra_trees) and cat_random_search are fenced off in vector mode.
__device__ void FindBestSplitsForLeafKernelVectorCategoricalInner(
  const CUDALeafSplitsStruct* leaf_splits_planes,
  const int num_targets,
  const SplitFindTask* task,
  const double lambda_l2,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  const double min_gain_shift,
  const double cnt_factor,
  const double sum_hessians,
  const data_size_t num_data,
  CUDASplitInfo* out) {
  __shared__ double shared_gain_buffer[WARPSIZE];
  __shared__ bool shared_found_buffer[WARPSIZE];
  __shared__ uint32_t shared_thread_index_buffer[WARPSIZE];
  __shared__ uint32_t best_thread_index;
  const int threadIdx_x = static_cast<int>(threadIdx.x);
  const int bin_start = 1 - task->mfb_offset;
  const int bin_end = task->num_bin - task->mfb_offset;
  // per-plane feature histogram bases: plane t's gradients, plane 0's hessians
  const hist_t* plane_hist[kMaxVectorTargets];
  for (int t = 0; t < num_targets; ++t) {
    plane_hist[t] = leaf_splits_planes[t].hist_in_leaf + (task->hist_offset << 1);
  }

  double local_gain = min_gain_shift;
  bool threshold_found = false;

  if (threadIdx_x == 0) {
    out->is_valid = false;
  }

  if (task->is_one_hot) {
    if (threadIdx_x >= bin_start && threadIdx_x < bin_end) {
      const double hess = plane_hist[0][(threadIdx_x << 1) + 1];
      const data_size_t cnt = static_cast<data_size_t>(CUDARoundInt(hess * cnt_factor));
      if (cnt >= min_data_in_leaf && hess >= min_sum_hessian_in_leaf) {
        const data_size_t other_count = num_data - cnt;
        if (other_count >= min_data_in_leaf) {
          const double sum_other_hessian = sum_hessians - hess - kEpsilon;
          if (sum_other_hessian >= min_sum_hessian_in_leaf) {
            double current_gain = 0.0;
            for (int t = 0; t < num_targets; ++t) {
              const double grad = plane_hist[t][threadIdx_x << 1];
              const double sum_other_gradient = leaf_splits_planes[t].sum_of_gradients - grad;
              current_gain += CUDALeafSplits::GetSplitGains<false, false>(
                sum_other_gradient, sum_other_hessian, grad,
                hess + kEpsilon, 0.0, lambda_l2, 0.0, 0.0, other_count, cnt, 0.0);
            }
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
    if (threshold_found && threadIdx_x == static_cast<int>(best_thread_index)) {
      out->is_valid = true;
      out->num_cat_threshold = 1;
      out->gain = (local_gain - min_gain_shift) * task->penalty;
      *(out->cat_threshold) = static_cast<uint32_t>(threadIdx_x + task->mfb_offset);
      out->default_left = false;
      out->num_vec_targets = num_targets;
      const double sum_left_hessian = plane_hist[0][(threadIdx_x << 1) + 1];
      const data_size_t left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      out->left_sum_hessians = sum_left_hessian;
      out->right_sum_hessians = sum_right_hessian;
      out->left_count = left_count;
      out->right_count = right_count;
      double* payload = out->vec_payload;
      for (int t = 0; t < num_targets; ++t) {
        const double sum_left_gradient = plane_hist[t][threadIdx_x << 1];
        const double sum_right_gradient = leaf_splits_planes[t].sum_of_gradients - sum_left_gradient;
        const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(
          sum_left_gradient, sum_left_hessian, 0.0, lambda_l2, 0.0, 0.0, left_count, 0.0);
        const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(
          sum_right_gradient, sum_right_hessian, 0.0, lambda_l2, 0.0, 0.0, right_count, 0.0);
        payload[kVecLeftSumGradients * num_targets + t] = sum_left_gradient;
        payload[kVecRightSumGradients * num_targets + t] = sum_right_gradient;
        payload[kVecLeftValue * num_targets + t] = left_output;
        payload[kVecRightValue * num_targets + t] = right_output;
        if (t == 0) {
          out->left_sum_gradients = sum_left_gradient;
          out->right_sum_gradients = sum_right_gradient;
          out->left_value = left_output;
          out->right_value = right_output;
          out->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(
            sum_left_gradient, sum_left_hessian, 0.0, lambda_l2, left_output);
          out->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(
            sum_right_gradient, sum_right_hessian, 0.0, lambda_l2, right_output);
        }
      }
    }
    return;
  }

  // many-vs-many over the sorted categories, mirroring the scalar inner
  __shared__ double shared_value_buffer[NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER];
  __shared__ int16_t shared_index_buffer[NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER];
  __shared__ uint16_t shared_mem_buffer_uint16[WARPSIZE];
  __shared__ data_size_t shared_mem_buffer_cnt[WARPSIZE];
  __shared__ data_size_t shared_count_buffer[NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER];
  __shared__ int used_bin;
  const double l2 = lambda_l2 + cat_l2;
  uint16_t is_valid_bin = 0;
  int best_dir = 0;
  double best_sum_left_hessian = 0.0;
  data_size_t best_left_count = 0;
  double best_sum_left_gradient[kMaxVectorTargets];
  if (threadIdx_x >= bin_start && threadIdx_x < bin_end) {
    const double hess = plane_hist[0][(threadIdx_x << 1) + 1];
    if (CUDARoundInt(hess * cnt_factor) >= cat_smooth) {
      double grad_sum = 0.0;
      for (int t = 0; t < num_targets; ++t) {
        grad_sum += plane_hist[t][threadIdx_x << 1];
      }
      shared_value_buffer[threadIdx_x] = grad_sum / (hess + cat_smooth);
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
  // bins beyond the block are excluded, as in the scalar inner (rarest
  // categories; their mass routes to the right child)
  const int sortable_bin_end = min(bin_end, static_cast<int>(blockDim.x));
  BitonicArgSort_1024<double, int16_t, true>(shared_value_buffer, shared_index_buffer, static_cast<int16_t>(sortable_bin_end));
  __syncthreads();
  const int max_num_cat = min(max_cat_threshold, (used_bin + 1) / 2);
  const int num_cat_thresholds = min(used_bin, max_num_cat);

  for (int dir = 0; dir < 2; ++dir) {
    double grad[kMaxVectorTargets];
    for (int t = 0; t < num_targets; ++t) {
      grad[t] = 0.0;
    }
    double hess = 0.0;
    data_size_t cat_cnt = 0;
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const int sorted_bin = dir == 0 ?
        shared_index_buffer[threadIdx_x] :
        shared_index_buffer[used_bin - 1 - threadIdx_x];
      for (int t = 0; t < num_targets; ++t) {
        grad[t] = plane_hist[t][sorted_bin << 1];
      }
      hess = plane_hist[0][(sorted_bin << 1) + 1];
      cat_cnt = static_cast<data_size_t>(CUDARoundInt(hess * cnt_factor));
    }
    if (threadIdx_x == 0) {
      hess += kEpsilon;
    }
    __syncthreads();
    // CPU-order scans per stream (see SequentialPrefixSum); the hessian scan
    // runs LAST so shared_value_buffer can be re-stored with its prefix below
    double sum_left_gradient[kMaxVectorTargets];
    for (int t = 0; t < num_targets; ++t) {
      sum_left_gradient[t] = SequentialPrefixSum<double>(grad[t], shared_value_buffer);
    }
    const double sum_left_hessian = SequentialPrefixSum<double>(hess, shared_value_buffer);
    const data_size_t left_count_prefix =
      ShufflePrefixSum<data_size_t>(cat_cnt, shared_mem_buffer_cnt);
    __syncthreads();
    if (threadIdx_x < num_cat_thresholds) {
      shared_value_buffer[threadIdx_x] = sum_left_hessian;
      shared_count_buffer[threadIdx_x] = left_count_prefix;
    }
    __syncthreads();
    if (threadIdx_x < num_cat_thresholds) {
      const data_size_t left_count = left_count_prefix;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (SequentialCategoricalGroupAccepted(shared_value_buffer, shared_count_buffer,
          threadIdx_x, num_cat_thresholds,
          num_data, sum_hessians, min_data_in_leaf, min_sum_hessian_in_leaf,
          min_data_per_group)) {
        double current_gain = 0.0;
        for (int t = 0; t < num_targets; ++t) {
          const double sum_right_gradient = leaf_splits_planes[t].sum_of_gradients - sum_left_gradient[t];
          current_gain += CUDALeafSplits::GetSplitGains<false, false>(
            sum_left_gradient[t], sum_left_hessian, sum_right_gradient,
            sum_right_hessian, 0.0, l2, 0.0, 0.0, left_count, right_count, 0.0);
        }
        if (current_gain > local_gain) {
          local_gain = current_gain;
          threshold_found = true;
          best_dir = dir == 0 ? 1 : -1;
          best_sum_left_hessian = sum_left_hessian;
          best_left_count = left_count;
          for (int t = 0; t < num_targets; ++t) {
            best_sum_left_gradient[t] = sum_left_gradient[t];
          }
        }
      }
    }
    __syncthreads();
  }

  const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
  if (threadIdx_x == 0) {
    best_thread_index = result;
  }
  __syncthreads();
  if (threshold_found && threadIdx_x == static_cast<int>(best_thread_index)) {
    out->is_valid = true;
    out->num_cat_threshold = threadIdx_x + 1;
    out->gain = (local_gain - min_gain_shift) * task->penalty;
    if (best_dir == 1) {
      for (int i = 0; i < threadIdx_x + 1; ++i) {
        (out->cat_threshold)[i] = shared_index_buffer[i] + task->mfb_offset;
      }
    } else {
      for (int i = 0; i < threadIdx_x + 1; ++i) {
        (out->cat_threshold)[i] = shared_index_buffer[used_bin - 1 - i] + task->mfb_offset;
      }
    }
    out->default_left = false;
    out->num_vec_targets = num_targets;
    const double sum_left_hessian = best_sum_left_hessian;
    const data_size_t left_count = best_left_count;
    const double sum_right_hessian = sum_hessians - sum_left_hessian;
    const data_size_t right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
    out->left_sum_hessians = sum_left_hessian;
    out->right_sum_hessians = sum_right_hessian;
    out->left_count = left_count;
    out->right_count = right_count;
    double* payload = out->vec_payload;
    for (int t = 0; t < num_targets; ++t) {
      const double sum_left_gradient = best_sum_left_gradient[t];
      const double sum_right_gradient = leaf_splits_planes[t].sum_of_gradients - sum_left_gradient;
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(
        sum_left_gradient, sum_left_hessian, 0.0, l2, 0.0, 0.0, left_count, 0.0);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(
        sum_right_gradient, sum_right_hessian, 0.0, l2, 0.0, 0.0, right_count, 0.0);
      payload[kVecLeftSumGradients * num_targets + t] = sum_left_gradient;
      payload[kVecRightSumGradients * num_targets + t] = sum_right_gradient;
      payload[kVecLeftValue * num_targets + t] = left_output;
      payload[kVecRightValue * num_targets + t] = right_output;
      if (t == 0) {
        out->left_sum_gradients = sum_left_gradient;
        out->right_sum_gradients = sum_right_gradient;
        out->left_value = left_output;
        out->right_value = right_output;
        out->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(
          sum_left_gradient, sum_left_hessian, 0.0, l2, left_output);
        out->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(
          sum_right_gradient, sum_right_hessian, 0.0, l2, right_output);
      }
    }
  }
}

// One (leaf, feature task) vector-leaf best-split search over T histogram
// planes. Shared by the per-pair kernel and the batched per-level kernel: both
// resolve their own leaf planes / output slot and then run this body, so the
// two flows produce identical split decisions.
__device__ void FindBestSplitsVectorInner(
  const CUDALeafSplitsStruct* leaf_splits_planes,
  const int num_targets,
  const SplitFindTask* task,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l2,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  const data_size_t num_data_in_leaf,
  CUDASplitInfo* out) {
  // dynamic shared prefix rows: row 0 = hessian stream, rows 1..T = per-target
  // gradient streams; blockDim.x lanes each
  extern __shared__ double vec_prefix_rows[];
  __shared__ double shared_gain_buffer[WARPSIZE];
  __shared__ bool shared_bool_buffer[WARPSIZE];
  __shared__ uint32_t shared_int_buffer[WARPSIZE];
  __shared__ uint32_t best_thread_index;

  const double sum_hessians = leaf_splits_planes[0].sum_of_hessians + 2 * kEpsilon;
  const data_size_t num_data = num_data_in_leaf;
  const double cnt_factor = static_cast<double>(num_data) / sum_hessians;
  // parent gain recomputed from the plane sums at find time, exactly like the
  // scalar kernel (same formula from the same sums, same bits)
  double parent_gain = 0.0;
  for (int t = 0; t < num_targets; ++t) {
    parent_gain += CUDALeafSplits::GetLeafGain<false, false>(
      leaf_splits_planes[t].sum_of_gradients, sum_hessians, 0.0, lambda_l2, 0.0,
      0.0, num_data, 0.0);
  }
  const double min_gain_shift = parent_gain + min_gain_to_split;

  if (task->is_categorical) {
    FindBestSplitsForLeafKernelVectorCategoricalInner(
      leaf_splits_planes, num_targets, task,
      lambda_l2, min_data_in_leaf, min_sum_hessian_in_leaf,
      cat_smooth, cat_l2, max_cat_threshold, min_data_per_group,
      min_gain_shift, cnt_factor, sum_hessians, num_data, out);
    return;
  }

  if (threadIdx.x == 0) {
    out->is_valid = false;
  }

  double local_grad[kMaxVectorTargets];
  for (int t = 0; t < num_targets; ++t) {
    local_grad[t] = 0.0;
  }
  double local_hess = 0.0;
  double local_gain = kMinScore;
  bool threshold_found = false;
  uint32_t threshold_value = 0;
  const unsigned int threadIdx_x = threadIdx.x;
  const bool skip_sum = task->reverse ?
    (task->skip_default_bin && (task->num_bin - 1 - threadIdx_x) == static_cast<int>(task->default_bin)) :
    (task->skip_default_bin && (threadIdx_x + task->mfb_offset) == static_cast<int>(task->default_bin));
  const uint32_t feature_num_bin_minus_offset = task->num_bin - task->mfb_offset;
  const bool na_missing_head = (!task->reverse && task->na_as_missing && task->mfb_offset == 1);
  if (!task->reverse) {
    if (na_missing_head) {
      if (threadIdx_x < static_cast<uint32_t>(task->num_bin) && threadIdx_x > 0) {
        const unsigned int bin_offset = (threadIdx_x - 1) << 1;
        local_hess = leaf_splits_planes[0].hist_in_leaf[(task->hist_offset << 1) + bin_offset + 1];
        for (int t = 0; t < num_targets; ++t) {
          local_grad[t] = leaf_splits_planes[t].hist_in_leaf[(task->hist_offset << 1) + bin_offset];
        }
      }
    } else {
      if (threadIdx_x < feature_num_bin_minus_offset && !skip_sum) {
        const unsigned int bin_offset = threadIdx_x << 1;
        local_hess = leaf_splits_planes[0].hist_in_leaf[(task->hist_offset << 1) + bin_offset + 1];
        for (int t = 0; t < num_targets; ++t) {
          local_grad[t] = leaf_splits_planes[t].hist_in_leaf[(task->hist_offset << 1) + bin_offset];
        }
      }
    }
  } else {
    if (threadIdx_x >= static_cast<unsigned int>(task->na_as_missing) &&
      threadIdx_x < feature_num_bin_minus_offset && !skip_sum) {
      const unsigned int read_index = feature_num_bin_minus_offset - 1 - threadIdx_x;
      const unsigned int bin_offset = read_index << 1;
      local_hess = leaf_splits_planes[0].hist_in_leaf[(task->hist_offset << 1) + bin_offset + 1];
      for (int t = 0; t < num_targets; ++t) {
        local_grad[t] = leaf_splits_planes[t].hist_in_leaf[(task->hist_offset << 1) + bin_offset];
      }
    }
  }
  __syncthreads();
  // fp64 CPU-order scans (see the scalar kernel's fp64 branch): the count
  // prefix sums per-bin roundings; the value prefixes fold sequentially.
  data_size_t local_bin_cnt = static_cast<data_size_t>(
    CUDARoundInt(local_hess * cnt_factor));
  if (na_missing_head) {
    const data_size_t cnt_non_default = ShuffleReduceSum<data_size_t>(
      local_bin_cnt, reinterpret_cast<data_size_t*>(shared_gain_buffer), blockDim.x);
    if (threadIdx_x == 0) {
      local_bin_cnt = num_data - cnt_non_default;
    }
    __syncthreads();
    vec_prefix_rows[threadIdx_x] = local_hess;
    for (int t = 0; t < num_targets; ++t) {
      vec_prefix_rows[(1 + t) * blockDim.x + threadIdx_x] = local_grad[t];
    }
    __syncthreads();
    if (threadIdx_x == 0) {
      // the lane-0 "missing" mass: totals with every bin SUBTRACTED in scan
      // order (gradient seeded from the plane total, hessian from
      // total - kEpsilon), matching the scalar kernel's chain per stream
      double acc_h = sum_hessians - kEpsilon;
      for (unsigned int i = 1; i < blockDim.x; ++i) {
        acc_h -= vec_prefix_rows[i];
      }
      local_hess = acc_h;
      for (int t = 0; t < num_targets; ++t) {
        double acc_g = leaf_splits_planes[t].sum_of_gradients;
        const double* row = vec_prefix_rows + (1 + t) * blockDim.x;
        for (unsigned int i = 1; i < blockDim.x; ++i) {
          acc_g -= row[i];
        }
        local_grad[t] = acc_g;
      }
    }
  } else if (threadIdx_x == 0) {
    local_hess += kEpsilon;
  }
  data_size_t local_cnt_prefix = ShufflePrefixSum<data_size_t>(
    local_bin_cnt, reinterpret_cast<data_size_t*>(shared_gain_buffer));
  // multi-stream CPU-order inclusive prefix (SequentialPrefixSumPair's fold,
  // one row per stream)
  {
    __syncthreads();
    vec_prefix_rows[threadIdx_x] = local_hess;
    for (int t = 0; t < num_targets; ++t) {
      vec_prefix_rows[(1 + t) * blockDim.x + threadIdx_x] = local_grad[t];
    }
    __syncthreads();
    if (threadIdx_x == 0) {
      const unsigned int len = static_cast<unsigned int>(task->num_bin);
      const unsigned int bound = len < blockDim.x ? len : blockDim.x;
      for (int row = 0; row <= num_targets; ++row) {
        double* row_buffer = vec_prefix_rows + row * blockDim.x;
        double acc = row_buffer[0];
        for (unsigned int i = 1; i < bound; ++i) {
          acc += row_buffer[i];
          row_buffer[i] = acc;
        }
      }
    }
    __syncthreads();
    local_hess = vec_prefix_rows[threadIdx_x];
    for (int t = 0; t < num_targets; ++t) {
      local_grad[t] = vec_prefix_rows[(1 + t) * blockDim.x + threadIdx_x];
    }
  }
  local_gain = kMinScore;
  if (task->reverse) {
    if (threadIdx_x >= static_cast<unsigned int>(task->na_as_missing) && threadIdx_x <= task->num_bin - 2 && !skip_sum) {
      const double sum_right_hessian = local_hess;
      const data_size_t right_count = local_cnt_prefix;
      const double sum_left_hessian = sum_hessians - sum_right_hessian;
      const data_size_t left_count = num_data - right_count;
      if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf) {
        double current_gain = 0.0;
        for (int t = 0; t < num_targets; ++t) {
          const double sum_right_gradient = local_grad[t];
          const double sum_left_gradient = leaf_splits_planes[t].sum_of_gradients - sum_right_gradient;
          current_gain += CUDALeafSplits::GetSplitGains<false, false, double>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, 0.0, lambda_l2, 0.0, 0.0, left_count, right_count, 0.0);
        }
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_value = static_cast<uint32_t>(task->num_bin - 2 - threadIdx_x);
          threshold_found = true;
        }
      }
    }
  } else {
    const uint32_t end = na_missing_head ? static_cast<uint32_t>(task->num_bin - 2) : feature_num_bin_minus_offset - 2;
    if (threadIdx_x <= end && !skip_sum) {
      const double sum_left_hessian = local_hess;
      const data_size_t left_count = local_cnt_prefix;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf) {
        double current_gain = 0.0;
        for (int t = 0; t < num_targets; ++t) {
          const double sum_left_gradient = local_grad[t];
          const double sum_right_gradient = leaf_splits_planes[t].sum_of_gradients - sum_left_gradient;
          current_gain += CUDALeafSplits::GetSplitGains<false, false, double>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, 0.0, lambda_l2, 0.0, 0.0, left_count, right_count, 0.0);
        }
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_value = na_missing_head ?
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
    out->is_valid = true;
    out->threshold = threshold_value;
    out->gain = local_gain * task->penalty;
    out->default_left = task->assume_out_default_left;
    out->num_vec_targets = num_targets;
    double* payload = out->vec_payload;
    if (task->reverse) {
      // the scalar REVERSE output chain, hessians shared across targets
      const double best_sum_left_hessian = sum_hessians - local_hess;
      const data_size_t left_count = num_data - local_cnt_prefix;
      const data_size_t right_count = num_data - left_count;
      const double sum_left_hessian = best_sum_left_hessian;
      const double sum_right_hessian = sum_hessians - best_sum_left_hessian;
      out->left_sum_hessians = sum_left_hessian - kEpsilon;
      out->right_sum_hessians = sum_hessians - sum_left_hessian - kEpsilon;
      out->left_count = left_count;
      out->right_count = right_count;
      for (int t = 0; t < num_targets; ++t) {
        const double sum_gradients_t = leaf_splits_planes[t].sum_of_gradients;
        const double sum_left_gradient = sum_gradients_t - local_grad[t];
        const double sum_right_gradient = sum_gradients_t - sum_left_gradient;
        const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(
          sum_left_gradient, sum_left_hessian, 0.0, lambda_l2, 0.0, 0.0, left_count, 0.0);
        const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(
          sum_right_gradient, sum_right_hessian, 0.0, lambda_l2, 0.0, 0.0, right_count, 0.0);
        payload[kVecLeftSumGradients * num_targets + t] = sum_left_gradient;
        payload[kVecRightSumGradients * num_targets + t] = sum_right_gradient;
        payload[kVecLeftValue * num_targets + t] = left_output;
        payload[kVecRightValue * num_targets + t] = right_output;
        if (t == 0) {
          out->left_sum_gradients = sum_left_gradient;
          out->right_sum_gradients = sum_right_gradient;
          out->left_value = left_output;
          out->right_value = right_output;
          out->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(
            sum_left_gradient, sum_left_hessian, 0.0, lambda_l2, left_output);
          out->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(
            sum_right_gradient, sum_right_hessian, 0.0, lambda_l2, right_output);
        }
      }
    } else {
      const double best_sum_left_hessian = local_hess;
      const data_size_t left_count = local_cnt_prefix;
      const data_size_t right_count = num_data - left_count;
      const double sum_left_hessian = best_sum_left_hessian;
      const double sum_right_hessian = sum_hessians - best_sum_left_hessian;
      out->left_sum_hessians = sum_left_hessian - kEpsilon;
      out->right_sum_hessians = sum_hessians - sum_left_hessian - kEpsilon;
      out->left_count = left_count;
      out->right_count = right_count;
      for (int t = 0; t < num_targets; ++t) {
        const double sum_gradients_t = leaf_splits_planes[t].sum_of_gradients;
        const double sum_left_gradient = local_grad[t];
        const double sum_right_gradient = sum_gradients_t - sum_left_gradient;
        const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(
          sum_left_gradient, sum_left_hessian, 0.0, lambda_l2, 0.0, 0.0, left_count, 0.0);
        const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(
          sum_right_gradient, sum_right_hessian, 0.0, lambda_l2, 0.0, 0.0, right_count, 0.0);
        payload[kVecLeftSumGradients * num_targets + t] = sum_left_gradient;
        payload[kVecRightSumGradients * num_targets + t] = sum_right_gradient;
        payload[kVecLeftValue * num_targets + t] = left_output;
        payload[kVecRightValue * num_targets + t] = right_output;
        if (t == 0) {
          out->left_sum_gradients = sum_left_gradient;
          out->right_sum_gradients = sum_right_gradient;
          out->left_value = left_output;
          out->right_value = right_output;
          out->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(
            sum_left_gradient, sum_left_hessian, 0.0, lambda_l2, left_output);
          out->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(
            sum_right_gradient, sum_right_hessian, 0.0, lambda_l2, right_output);
        }
      }
    }
  }
}

// ---- vector-leaf x quantized (discretized) find ------------------------------
//
// The quantized twin of FindBestSplitsVectorInner. Plane t's histogram holds
// target t's gradients packed against the SHARED hessian stream, at target t's
// own gradient scale (CUDAGradientDiscretizer::DiscretizeGradientsForPlane).
// Every scan therefore runs in exact packed integers -- one ShufflePrefixSum
// per plane, no fp64 slab -- and only the gain evaluation dequantizes, with
// grad_scales[t] for target t and the one hess_scale for the hessian every
// target shares. Numerical tasks only; categorical is fenced off for the
// quantized vector configuration.
template <bool USE_16BIT_ACC_HIST, typename ACC_HIST_TYPE>
__device__ __forceinline__ int64_t VectorUnpackAcc(const ACC_HIST_TYPE v) {
  return USE_16BIT_ACC_HIST ?
    ((static_cast<int64_t>(static_cast<int16_t>(v >> 16)) << 32) |
      static_cast<int64_t>(v & 0x0000ffff)) :
    static_cast<int64_t>(v);
}

template <bool REVERSE, typename ACC_HIST_TYPE, bool USE_16BIT_ACC_HIST>
__device__ void FindBestSplitsDiscretizedVectorInner(
  const CUDALeafSplitsStruct* leaf_splits_planes,
  const int num_targets,
  const SplitFindTask* task,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l2_in,
  const data_size_t num_data,
  const score_t* grad_scales,
  const score_t* hess_scale_ptr,
  const bool quant_bagging_ridge,
  CUDASplitInfo* out) {
  __shared__ double shared_gain_buffer[WARPSIZE];
  __shared__ bool shared_bool_buffer[WARPSIZE];
  __shared__ DiscretizedScanScratch<ACC_HIST_TYPE> shared_scan_scratch;
  __shared__ uint32_t best_thread_index;

  const double hess_scale = static_cast<double>(*hess_scale_ptr);
  // one hessian quantum of ridge under bagging: bagged quantized training
  // redraws the hessian rounding noise every iteration, and the gain argmax
  // harvests children whose noisy |G| is large while noisy H is near zero.
  // Vector mode sums T such gains, so the same noise enters T times over.
  const double lambda_l2 = quant_bagging_ridge ? lambda_l2_in + hess_scale : lambda_l2_in;
  const int64_t parent_gh_hess = leaf_splits_planes[0].sum_of_gradients_hessians;
  const double sum_hessians =
    static_cast<double>(static_cast<int32_t>(parent_gh_hess & 0x00000000ffffffff)) * hess_scale;
  const double cnt_factor = static_cast<double>(num_data) / sum_hessians;

  int64_t parent_gh[kMaxVectorTargets];
  double grad_scale[kMaxVectorTargets];
  double parent_gain = 0.0;
  for (int t = 0; t < num_targets; ++t) {
    parent_gh[t] = leaf_splits_planes[t].sum_of_gradients_hessians;
    grad_scale[t] = static_cast<double>(grad_scales[t]);
    const double sum_gradients_t = static_cast<double>(
      static_cast<int32_t>((parent_gh[t] & 0xffffffff00000000) >> 32)) * grad_scale[t];
    parent_gain += CUDALeafSplits::GetLeafGain<false, false>(
      sum_gradients_t, sum_hessians, 0.0, lambda_l2, 0.0, 0.0, num_data, 0.0);
  }
  const double min_gain_shift = parent_gain + min_gain_to_split;

  if (threadIdx.x == 0) {
    out->is_valid = false;
  }

  ACC_HIST_TYPE local_hist[kMaxVectorTargets];
  for (int t = 0; t < num_targets; ++t) {
    local_hist[t] = 0;
  }
  double local_gain = kMinScore;
  bool threshold_found = false;
  uint32_t threshold_value = 0;
  const unsigned int threadIdx_x = threadIdx.x;
  const bool skip_sum = REVERSE ?
    (task->skip_default_bin && (task->num_bin - 1 - threadIdx_x) == static_cast<int>(task->default_bin)) :
    (task->skip_default_bin && (threadIdx_x + task->mfb_offset) == static_cast<int>(task->default_bin));
  const uint32_t feature_num_bin_minus_offset = task->num_bin - task->mfb_offset;
  const bool na_missing_head = (!REVERSE && task->na_as_missing && task->mfb_offset == 1);
  int read_bin = -1;
  if (!REVERSE) {
    if (na_missing_head) {
      if (threadIdx_x < static_cast<uint32_t>(task->num_bin) && threadIdx_x > 0) {
        read_bin = static_cast<int>(threadIdx_x) - 1;
      }
    } else if (threadIdx_x < feature_num_bin_minus_offset && !skip_sum) {
      read_bin = static_cast<int>(threadIdx_x);
    }
  } else {
    if (threadIdx_x >= static_cast<unsigned int>(task->na_as_missing) &&
        threadIdx_x < feature_num_bin_minus_offset && !skip_sum) {
      read_bin = static_cast<int>(feature_num_bin_minus_offset - 1 - threadIdx_x);
    }
  }
  if (read_bin >= 0) {
    for (int t = 0; t < num_targets; ++t) {
      local_hist[t] = reinterpret_cast<const ACC_HIST_TYPE*>(
        leaf_splits_planes[t].hist_in_leaf)[task->hist_offset + read_bin];
    }
  }
  __syncthreads();
  if (na_missing_head) {
    // bin 0 (the unstored most-frequent bin) = plane total - sum of stored bins,
    // per plane. Packed accumulators add field-wise, so the reduction runs
    // directly on the packed representation.
    for (int t = 0; t < num_targets; ++t) {
      const ACC_HIST_TYPE sum_non_default = ShuffleReduceSum<ACC_HIST_TYPE>(
        local_hist[t], shared_scan_scratch.acc, blockDim.x);
      if (threadIdx_x == 0) {
        const int64_t default_bin_packed = parent_gh[t] -
          VectorUnpackAcc<USE_16BIT_ACC_HIST, ACC_HIST_TYPE>(sum_non_default);
        local_hist[t] = USE_16BIT_ACC_HIST ?
          static_cast<ACC_HIST_TYPE>(static_cast<int32_t>(
            (static_cast<uint32_t>(static_cast<int32_t>(default_bin_packed >> 32)) << 16) |
            (static_cast<uint32_t>(default_bin_packed) & 0x0000ffffu))) :
          static_cast<ACC_HIST_TYPE>(default_bin_packed);
      }
      __syncthreads();
    }
  }
  for (int t = 0; t < num_targets; ++t) {
    local_hist[t] = ShufflePrefixSum<ACC_HIST_TYPE>(
      local_hist[t], shared_scan_scratch.acc);
  }
  // the packed prefix of plane 0 carries the shared hessian side
  int64_t scan_gh0 = VectorUnpackAcc<USE_16BIT_ACC_HIST, ACC_HIST_TYPE>(local_hist[0]);
  double sum_left_hessian = 0.0;
  double sum_right_hessian = 0.0;
  data_size_t left_count = 0;
  data_size_t right_count = 0;
  const bool in_range = REVERSE ?
    (threadIdx_x >= static_cast<unsigned int>(task->na_as_missing) &&
     threadIdx_x <= static_cast<unsigned int>(task->num_bin - 2) && !skip_sum) :
    (threadIdx_x <= (na_missing_head ? static_cast<uint32_t>(task->num_bin - 2)
                                     : feature_num_bin_minus_offset - 2) && !skip_sum);
  if (in_range) {
    if (REVERSE) {
      sum_right_hessian =
        static_cast<double>(static_cast<int32_t>(scan_gh0 & 0x00000000ffffffff)) * hess_scale;
      right_count = static_cast<data_size_t>(CUDARoundInt(sum_right_hessian * cnt_factor));
      const int64_t left_gh0 = parent_gh_hess - scan_gh0;
      sum_left_hessian =
        static_cast<double>(static_cast<int32_t>(left_gh0 & 0x00000000ffffffff)) * hess_scale;
      left_count = num_data - right_count;
    } else {
      sum_left_hessian =
        static_cast<double>(static_cast<int32_t>(scan_gh0 & 0x00000000ffffffff)) * hess_scale;
      left_count = static_cast<data_size_t>(CUDARoundInt(sum_left_hessian * cnt_factor));
      const int64_t right_gh0 = parent_gh_hess - scan_gh0;
      sum_right_hessian =
        static_cast<double>(static_cast<int32_t>(right_gh0 & 0x00000000ffffffff)) * hess_scale;
      right_count = num_data - left_count;
    }
    if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf) {
      double current_gain = 0.0;
      for (int t = 0; t < num_targets; ++t) {
        const int64_t scan_gh_t =
          VectorUnpackAcc<USE_16BIT_ACC_HIST, ACC_HIST_TYPE>(local_hist[t]);
        const int64_t other_gh_t = parent_gh[t] - scan_gh_t;
        const double scan_grad = static_cast<double>(
          static_cast<int32_t>((scan_gh_t & 0xffffffff00000000) >> 32)) * grad_scale[t];
        const double other_grad = static_cast<double>(
          static_cast<int32_t>((other_gh_t & 0xffffffff00000000) >> 32)) * grad_scale[t];
        const double sum_left_gradient = REVERSE ? other_grad : scan_grad;
        const double sum_right_gradient = REVERSE ? scan_grad : other_grad;
        current_gain += CUDALeafSplits::GetSplitGains<false, false, double>(
          sum_left_gradient, sum_left_hessian + kEpsilon,
          sum_right_gradient, sum_right_hessian + kEpsilon,
          0.0, lambda_l2, 0.0, 0.0, left_count, right_count, 0.0);
      }
      if (current_gain > min_gain_shift) {
        local_gain = current_gain - min_gain_shift;
        threshold_value = REVERSE ?
          static_cast<uint32_t>(task->num_bin - 2 - threadIdx_x) :
          (na_missing_head ? static_cast<uint32_t>(threadIdx_x) :
                             static_cast<uint32_t>(threadIdx_x + task->mfb_offset));
        threshold_found = true;
      }
    }
  }
  __syncthreads();
  const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x,
    shared_gain_buffer, shared_bool_buffer, shared_scan_scratch.thread_index);
  if (threadIdx_x == 0) {
    best_thread_index = result;
  }
  __syncthreads();
  if (threshold_found && threadIdx_x == best_thread_index) {
    out->is_valid = true;
    out->threshold = threshold_value;
    out->gain = local_gain * task->penalty;
    out->default_left = task->assume_out_default_left;
    out->num_vec_targets = num_targets;
    const int64_t left_gh0 = REVERSE ? (parent_gh_hess - scan_gh0) : scan_gh0;
    const int64_t right_gh0 = parent_gh_hess - left_gh0;
    out->left_sum_hessians = sum_left_hessian;
    out->right_sum_hessians = sum_right_hessian;
    out->left_count = left_count;
    out->right_count = right_count;
    out->left_sum_of_gradients_hessians = left_gh0;
    out->right_sum_of_gradients_hessians = right_gh0;
    double* payload = out->vec_payload;
    for (int t = 0; t < num_targets; ++t) {
      const int64_t scan_gh_t =
        VectorUnpackAcc<USE_16BIT_ACC_HIST, ACC_HIST_TYPE>(local_hist[t]);
      const int64_t left_gh_t = REVERSE ? (parent_gh[t] - scan_gh_t) : scan_gh_t;
      const int32_t left_grad_int = static_cast<int32_t>((left_gh_t & 0xffffffff00000000) >> 32);
      const int32_t right_grad_int = static_cast<int32_t>(
        ((parent_gh[t] - left_gh_t) & 0xffffffff00000000) >> 32);
      const double sum_left_gradient = static_cast<double>(left_grad_int) * grad_scale[t];
      const double sum_right_gradient = static_cast<double>(right_grad_int) * grad_scale[t];
      const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(
        sum_left_gradient, sum_left_hessian, 0.0, lambda_l2, 0.0, 0.0, left_count, 0.0);
      const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(
        sum_right_gradient, sum_right_hessian, 0.0, lambda_l2, 0.0, 0.0, right_count, 0.0);
      payload[kVecLeftSumGradients * num_targets + t] = sum_left_gradient;
      payload[kVecRightSumGradients * num_targets + t] = sum_right_gradient;
      payload[kVecLeftValue * num_targets + t] = left_output;
      payload[kVecRightValue * num_targets + t] = right_output;
      payload[kVecLeftGradInt * num_targets + t] = static_cast<double>(left_grad_int);
      payload[kVecRightGradInt * num_targets + t] = static_cast<double>(right_grad_int);
      if (t == 0) {
        out->left_sum_gradients = sum_left_gradient;
        out->right_sum_gradients = sum_right_gradient;
        out->left_value = left_output;
        out->right_value = right_output;
        out->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(
          sum_left_gradient, sum_left_hessian, 0.0, lambda_l2, left_output);
        out->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(
          sum_right_gradient, sum_right_hessian, 0.0, lambda_l2, right_output);
      }
    }
  }
}

// Dispatch on the leaf's histogram bit width, then on scan direction.
__device__ void FindBestSplitsDiscretizedVectorDispatch(
  const CUDALeafSplitsStruct* leaf_splits_planes,
  const int num_targets,
  const SplitFindTask* task,
  const bool use_16bit_bin,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l2,
  const data_size_t num_data,
  const score_t* grad_scales,
  const score_t* hess_scale,
  const bool quant_bagging_ridge,
  CUDASplitInfo* out) {
  if (use_16bit_bin) {
    if (task->reverse) {
      FindBestSplitsDiscretizedVectorInner<true, int32_t, true>(
        leaf_splits_planes, num_targets, task, min_data_in_leaf, min_sum_hessian_in_leaf,
        min_gain_to_split, lambda_l2, num_data, grad_scales, hess_scale,
        quant_bagging_ridge, out);
    } else {
      FindBestSplitsDiscretizedVectorInner<false, int32_t, true>(
        leaf_splits_planes, num_targets, task, min_data_in_leaf, min_sum_hessian_in_leaf,
        min_gain_to_split, lambda_l2, num_data, grad_scales, hess_scale,
        quant_bagging_ridge, out);
    }
  } else {
    if (task->reverse) {
      FindBestSplitsDiscretizedVectorInner<true, int64_t, false>(
        leaf_splits_planes, num_targets, task, min_data_in_leaf, min_sum_hessian_in_leaf,
        min_gain_to_split, lambda_l2, num_data, grad_scales, hess_scale,
        quant_bagging_ridge, out);
    } else {
      FindBestSplitsDiscretizedVectorInner<false, int64_t, false>(
        leaf_splits_planes, num_targets, task, min_data_in_leaf, min_sum_hessian_in_leaf,
        min_gain_to_split, lambda_l2, num_data, grad_scales, hess_scale,
        quant_bagging_ridge, out);
    }
  }
}

__global__ void FindBestSplitsDiscretizedForLeafKernelVector(
  const int8_t* is_feature_used_bytree,
  const int num_tasks,
  const SplitFindTask* tasks,
  const CUDALeafSplitsStruct* leaf_splits_planes,
  const int num_targets,
  const bool is_larger,
  const uint8_t num_bits_in_histogram_bin,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l2,
  const score_t* grad_scales,
  const score_t* hess_scale,
  const bool quant_bagging_ridge,
  CUDASplitInfo* cuda_best_split_info,
  const data_size_t global_num_data_in_leaf) {
  const unsigned int task_index = blockIdx.x;
  const SplitFindTask* task = tasks + task_index;
  const unsigned int output_offset = is_larger ? (task_index + num_tasks) : task_index;
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  if (!is_feature_used_bytree[task->inner_feature_index]) {
    if (threadIdx.x == 0) {
      out->is_valid = false;
    }
    return;
  }
  FindBestSplitsDiscretizedVectorDispatch(leaf_splits_planes, num_targets, task,
    num_bits_in_histogram_bin <= 16, min_data_in_leaf, min_sum_hessian_in_leaf,
    min_gain_to_split, lambda_l2, global_num_data_in_leaf, grad_scales, hess_scale,
    quant_bagging_ridge, out);
}

// Batched per-level quantized vector find: grid and output layout mirror
// FindBestSplitsForLevelKernelVector.
__global__ void FindBestSplitsDiscretizedForLevelKernelVector(
  const int8_t* is_feature_used_bytree,
  const int num_tasks,
  const SplitFindTask* tasks,
  const int* used_task_indices,
  const CUDAHybridPairDescriptor* pair_descs,
  const CUDALeafSplitsStruct* plane_slab,
  const int num_targets,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l2,
  const score_t* grad_scales,
  const score_t* hess_scale,
  const bool quant_bagging_ridge,
  CUDASplitInfo* cuda_best_split_info) {
  const unsigned int pair_index = blockIdx.y;
  const bool is_larger = (blockIdx.z == 1);
  const CUDAHybridPairDescriptor* desc = pair_descs + pair_index;
  if (is_larger ? !desc->larger_valid : !desc->smaller_valid) {
    return;
  }
  const CUDALeafSplitsStruct* leaf_splits_planes = plane_slab +
    (static_cast<size_t>(2 * pair_index) + (is_larger ? 1 : 0)) * static_cast<size_t>(num_targets);
  const data_size_t num_data = leaf_splits_planes[0].num_data_in_leaf;
  if (leaf_splits_planes[0].leaf_index < 0 || num_data <= min_data_in_leaf ||
      leaf_splits_planes[0].sum_of_hessians <= min_sum_hessian_in_leaf) {
    return;
  }
  const unsigned int task_index = used_task_indices == nullptr ?
    blockIdx.x : static_cast<unsigned int>(used_task_indices[blockIdx.x]);
  const SplitFindTask* task = tasks + task_index;
  CUDASplitInfo* out = cuda_best_split_info +
    pair_index * (2 * static_cast<unsigned int>(num_tasks)) +
    (is_larger ? task_index + num_tasks : task_index);
  if (!is_feature_used_bytree[task->inner_feature_index]) {
    if (threadIdx.x == 0) {
      out->is_valid = false;
    }
    return;
  }
  const uint8_t num_bits = is_larger ? desc->larger_num_bits : desc->smaller_num_bits;
  FindBestSplitsDiscretizedVectorDispatch(leaf_splits_planes, num_targets, task,
    num_bits <= 16, min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
    lambda_l2, num_data, grad_scales, hess_scale, quant_bagging_ridge, out);
}

__global__ void FindBestSplitsForLeafKernelVector(
  const int8_t* is_feature_used_bytree,
  const int num_tasks,
  const SplitFindTask* tasks,
  const CUDALeafSplitsStruct* leaf_splits_planes,
  const int num_targets,
  const bool is_larger,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l2,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  CUDASplitInfo* cuda_best_split_info,
  const data_size_t global_num_data_in_leaf) {
  const unsigned int task_index = blockIdx.x;
  const SplitFindTask* task = tasks + task_index;
  const unsigned int output_offset = is_larger ? (task_index + num_tasks) : task_index;
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  if (!is_feature_used_bytree[task->inner_feature_index]) {
    if (threadIdx.x == 0) {
      out->is_valid = false;
    }
    return;
  }
  FindBestSplitsVectorInner(leaf_splits_planes, num_targets, task,
    min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split, lambda_l2,
    cat_smooth, cat_l2, max_cat_threshold, min_data_per_group,
    global_num_data_in_leaf, out);
}

// Batched per-level vector-leaf find: blockIdx.x = task, blockIdx.y = pair,
// blockIdx.z = 0 (smaller leaf) / 1 (larger leaf), mirroring the scalar level
// kernel's grid and output layout. Each (pair, role) reads its own T
// contiguous leaf-splits planes from the level plane slab
// (slab[(2 * pair + role) * T + t] carries target t's gradient sums and
// histogram plane) and writes its own region of the task output buffer.
__global__ void FindBestSplitsForLevelKernelVector(
  const int8_t* is_feature_used_bytree,
  const int num_tasks,
  const SplitFindTask* tasks,
  const int* used_task_indices,
  const CUDAHybridPairDescriptor* pair_descs,
  const CUDALeafSplitsStruct* plane_slab,
  const int num_targets,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l2,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  CUDASplitInfo* cuda_best_split_info) {
  const unsigned int pair_index = blockIdx.y;
  const bool is_larger = (blockIdx.z == 1);
  const CUDAHybridPairDescriptor* desc = pair_descs + pair_index;
  if (is_larger ? !desc->larger_valid : !desc->smaller_valid) {
    return;
  }
  const CUDALeafSplitsStruct* leaf_splits_planes = plane_slab +
    (static_cast<size_t>(2 * pair_index) + (is_larger ? 1 : 0)) * static_cast<size_t>(num_targets);
  // plane 0 mirrors the primary leaf-splits struct of this role (leaf index,
  // row count and the shared hessian sum), so the gates read it
  const data_size_t num_data = leaf_splits_planes[0].num_data_in_leaf;
  if (leaf_splits_planes[0].leaf_index < 0 || num_data <= min_data_in_leaf ||
      leaf_splits_planes[0].sum_of_hessians <= min_sum_hessian_in_leaf) {
    return;
  }
  // feature-sampled trees launch one block per USED task only; the output slot
  // keeps the ORIGINAL task index (unused slots are masked out by the sync)
  const unsigned int task_index = used_task_indices == nullptr ?
    blockIdx.x : static_cast<unsigned int>(used_task_indices[blockIdx.x]);
  const SplitFindTask* task = tasks + task_index;
  CUDASplitInfo* out = cuda_best_split_info +
    pair_index * (2 * static_cast<unsigned int>(num_tasks)) +
    (is_larger ? task_index + num_tasks : task_index);
  if (!is_feature_used_bytree[task->inner_feature_index]) {
    if (threadIdx.x == 0) {
      out->is_valid = false;
    }
    return;
  }
  FindBestSplitsVectorInner(leaf_splits_planes, num_targets, task,
    min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split, lambda_l2,
    cat_smooth, cat_l2, max_cat_threshold, min_data_per_group,
    num_data, out);
}

void CUDABestSplitFinder::LaunchFindBestSplitsForLevelKernelVector(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const CUDALeafSplitsStruct* plane_slab,
  const VectorQuantArgs& quant) {
  const bool compact_tasks = num_used_tasks_ > 0 && num_used_tasks_ < num_tasks_;
  if (num_used_tasks_ == 0) {
    return;  // no usable feature this tree; the sync masks every lane not-found
  }
  const dim3 grid_dim(compact_tasks ? num_used_tasks_ : num_tasks_, num_pairs, 2);
  if (quant.active()) {
    // integer prefix scans are exact and associative, so the quantized vector
    // kernel needs no fp64 prefix slab
    FindBestSplitsDiscretizedForLevelKernelVector
      <<<grid_dim, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>(
        cuda_is_feature_used_bytree_.RawData(),
        num_tasks_,
        cuda_split_find_tasks_.RawData(),
        compact_tasks ? cuda_used_task_indices_.RawDataReadOnly() : nullptr,
        pair_descs,
        plane_slab,
        vec_num_targets_,
        min_data_in_leaf_,
        min_sum_hessian_in_leaf_,
        min_gain_to_split_,
        lambda_l2_,
        quant.grad_scales,
        quant.hess_scale,
        quant_bagging_ridge_,
        cuda_best_split_info_.RawData());
    return;
  }
  const size_t shared_bytes = static_cast<size_t>(vec_num_targets_ + 1) *
    NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER * sizeof(double);
  FindBestSplitsForLevelKernelVector
    <<<grid_dim, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, shared_bytes, cuda_streams_[0]>>>(
      cuda_is_feature_used_bytree_.RawData(),
      num_tasks_,
      cuda_split_find_tasks_.RawData(),
      compact_tasks ? cuda_used_task_indices_.RawDataReadOnly() : nullptr,
      pair_descs,
      plane_slab,
      vec_num_targets_,
      min_data_in_leaf_,
      min_sum_hessian_in_leaf_,
      min_gain_to_split_,
      lambda_l2_,
      cat_smooth_,
      cat_l2_,
      max_cat_threshold_,
      min_data_per_group_,
      cuda_best_split_info_.RawData());
}

void CUDABestSplitFinder::LaunchFindBestSplitsForLeafKernelVector(
  const CUDALeafSplitsStruct* smaller_leaf_splits_planes,
  const CUDALeafSplitsStruct* larger_leaf_splits_planes,
  const bool is_smaller_leaf_valid,
  const bool is_larger_leaf_valid,
  const data_size_t global_num_data_in_smaller_leaf,
  const data_size_t global_num_data_in_larger_leaf,
  const VectorQuantArgs& quant) {
  if (!is_smaller_leaf_valid && !is_larger_leaf_valid) {
    return;
  }
  if (quant.active()) {
    if (is_smaller_leaf_valid) {
      CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[0], hist_subtract_done_events_[active_hist_pipeline_], 0));
      FindBestSplitsDiscretizedForLeafKernelVector
        <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>(
        cuda_is_feature_used_bytree_.RawData(), num_tasks_, cuda_split_find_tasks_.RawData(),
        smaller_leaf_splits_planes, vec_num_targets_, false, quant.smaller_num_bits,
        min_data_in_leaf_, min_sum_hessian_in_leaf_, min_gain_to_split_, lambda_l2_,
        quant.grad_scales, quant.hess_scale, quant_bagging_ridge_,
        cuda_best_split_info_.RawData(), global_num_data_in_smaller_leaf);
    }
    if (is_larger_leaf_valid) {
      CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[1], hist_subtract_done_events_[active_hist_pipeline_], 0));
      FindBestSplitsDiscretizedForLeafKernelVector
        <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[1]>>>(
        cuda_is_feature_used_bytree_.RawData(), num_tasks_, cuda_split_find_tasks_.RawData(),
        larger_leaf_splits_planes, vec_num_targets_, true, quant.larger_num_bits,
        min_data_in_leaf_, min_sum_hessian_in_leaf_, min_gain_to_split_, lambda_l2_,
        quant.grad_scales, quant.hess_scale, quant_bagging_ridge_,
        cuda_best_split_info_.RawData(), global_num_data_in_larger_leaf);
    }
    return;
  }
  const size_t shared_bytes = static_cast<size_t>(vec_num_targets_ + 1) *
    NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER * sizeof(double);
  if (is_smaller_leaf_valid) {
    CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[0], hist_subtract_done_events_[active_hist_pipeline_], 0));
    FindBestSplitsForLeafKernelVector
      <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, shared_bytes, cuda_streams_[0]>>>(
      cuda_is_feature_used_bytree_.RawData(), num_tasks_, cuda_split_find_tasks_.RawData(),
      smaller_leaf_splits_planes, vec_num_targets_, false,
      min_data_in_leaf_, min_sum_hessian_in_leaf_, min_gain_to_split_, lambda_l2_,
      cat_smooth_, cat_l2_, max_cat_threshold_, min_data_per_group_,
      cuda_best_split_info_.RawData(), global_num_data_in_smaller_leaf);
  }
  if (is_larger_leaf_valid) {
    CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[1], hist_subtract_done_events_[active_hist_pipeline_], 0));
    FindBestSplitsForLeafKernelVector
      <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, shared_bytes, cuda_streams_[1]>>>(
      cuda_is_feature_used_bytree_.RawData(), num_tasks_, cuda_split_find_tasks_.RawData(),
      larger_leaf_splits_planes, vec_num_targets_, true,
      min_data_in_leaf_, min_sum_hessian_in_leaf_, min_gain_to_split_, lambda_l2_,
      cat_smooth_, cat_l2_, max_cat_threshold_, min_data_per_group_,
      cuda_best_split_info_.RawData(), global_num_data_in_larger_leaf);
  }
}

__global__ void AssignVecPayloadKernel(
  CUDASplitInfo* cuda_split_infos, size_t len,
  const int num_targets, double* payload_slab) {
  const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < len) {
    cuda_split_infos[i].vec_payload =
      payload_slab + i * static_cast<size_t>(kNumVecPayloadFields) * num_targets;
    cuda_split_infos[i].num_vec_targets = 0;
  }
}

void CUDABestSplitFinder::LaunchAssignVecPayloadKernel(
  CUDASplitInfo* cuda_split_infos, double* payload_slab, size_t len) {
  if (len == 0) {
    return;
  }
  const int num_blocks = static_cast<int>((len + NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER - 1) / NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER);
  AssignVecPayloadKernel<<<num_blocks, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER>>>(
    cuda_split_infos, len, vec_num_targets_, payload_slab);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}


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
    max_delta_step_, \
    cat_smooth_, \
    cat_l2_, \
    max_cat_threshold_, \
    min_data_per_group_, \
    cat_random_search_, \
    max_cat_to_onehot_, \
    grad_scale, \
    hess_scale, \
    quant_bagging_ridge_, \
    cuda_best_split_info_.RawData(), \
    global_num_data_in_smaller_leaf, \
    global_num_data_in_larger_leaf, \
    cegb_use_ ? cuda_task_cegb_penalty_.RawData() : nullptr, \
    cegb_tradeoff_times_penalty_split_

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
  if (FalcataFP32GainEnabled()) {
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
  const double max_delta_step,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  CUDASplitInfo* cuda_best_split_info,
  const bool use_desc_counts,
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
  // Under NCCL (use_desc_counts) the struct count is the rank-LOCAL row count
  // while every histogram sum is GLOBAL: deriving cnt_factor from it halves
  // every split's left/right counts (min_data gates then reject globally
  // splittable leaves, and the halved counts poison slots 16/17 -> the next
  // level's global bookkeeping). The descriptor counts are host-written GLOBAL
  // counts there, mirroring the discretized level kernel's host-launched path.
  const data_size_t num_data = use_desc_counts ?
    (is_larger ? desc->num_data_in_larger_leaf : desc->num_data_in_smaller_leaf) :
    leaf_splits->num_data_in_leaf;
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
  const double sum_gradients = leaf_splits->sum_of_gradients;
  const double sum_hessians = leaf_sum_hessians + 2 * kEpsilon;
  const double parent_output = leaf_splits->leaf_value;
  // Recompute the parent's gain from the leaf sums, matching CPU's gain_shift
  // (see FindBestSplitsForLeafKernel for the full rationale).
  const double parent_gain = CUDALeafSplits::GetLeafGain<USE_L1, USE_SMOOTHING>(
      sum_gradients, sum_hessians, lambda_l1, lambda_l2, path_smooth,
      max_delta_step, num_data, parent_output);
  const unsigned int output_offset = pair_index * (2 * static_cast<unsigned int>(num_tasks)) +
    (is_larger ? task_index + num_tasks : task_index);
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  CUDARandom* cuda_random = USE_RAND ?
    (is_larger ? cuda_randoms + task_index * 2 + 1 : cuda_randoms + task_index * 2) : nullptr;
  if (is_feature_used_bytree[task->inner_feature_index]) {
    const hist_t* hist_ptr = FeatureHistPtr(leaf_splits->hist_in_leaf, task->hist_offset, fp32_hist);
    if (task->is_categorical) {
      // fp32_hist is globally disabled for datasets with categorical features
      const CatHistReaderF64 hist_reader{hist_ptr};
      FindBestSplitsForLeafKernelCategoricalInner<USE_RAND, USE_L1, USE_SMOOTHING, CatHistReaderF64>(
        hist_reader, task, cuda_random,
        lambda_l1, lambda_l2, path_smooth, max_delta_step,
        min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
        cat_smooth, cat_l2, max_cat_threshold, min_data_per_group,
        // SupportsBatchedLevel excludes cat_random_search: this launch covers
        // every sibling pair of the level, so its blocks would share (and race
        // on) the per-task CUDARandom the RANDOM search draws from.
        /*cat_random_search=*/0,
        parent_gain, sum_gradients, sum_hessians, num_data, parent_output,
        /*sum_gradients_hessians_total=*/0, out);
    } else if (!task->reverse) {
      FindBestSplitsForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, false, GAIN_T, /*USE_MC=*/false>(
        hist_ptr, fp32_hist, task, cuda_random,
        lambda_l1, lambda_l2, path_smooth, max_delta_step,
        min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
        parent_gain, sum_gradients, sum_hessians, num_data, parent_output,
        // hybrid growth is gated off whenever monotone_constraints is set, so the
        // batched path never sees a binding constraint: pass identity bounds.
        -DBL_MAX, DBL_MAX,
        out);
    } else {
      FindBestSplitsForLeafKernelInner<USE_RAND, USE_L1, USE_SMOOTHING, true, GAIN_T, /*USE_MC=*/false>(
        hist_ptr, fp32_hist, task, cuda_random,
        lambda_l1, lambda_l2, path_smooth, max_delta_step,
        min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split,
        parent_gain, sum_gradients, sum_hessians, num_data, parent_output,
        // hybrid growth is gated off whenever monotone_constraints is set, so the
        // batched path never sees a binding constraint: pass identity bounds.
        -DBL_MAX, DBL_MAX,
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
  const double lambda_l2_in,
  const double path_smooth,
  const double max_delta_step,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  const score_t* grad_scale,
  const score_t* hess_scale,
  const bool quant_bagging_ridge,
  CUDASplitInfo* cuda_best_split_info,
  const CUDAHybridGraphLoopStateOpt gstate) {
  // graphs A2: the graph-frozen grid is a pow2 bucket of the live pair count;
  // blocks beyond the live range exit before any read (stale descriptors)
  if (HybridGraphBeyondLiveSplits(gstate, blockIdx.y)) {
    return;
  }
  // one hessian quantum of ridge under bagging (see the per-leaf discretized
  // kernel for the noise argument; unbagged models stay bit-identical)
  const double lambda_l2 = quant_bagging_ridge ?
    lambda_l2_in + static_cast<double>(*hess_scale) : lambda_l2_in;
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

// CUDASplitInfo::operator= with every field it copies unconditionally loaded before the first store (cuda_plan key
// sync_copy_batched). The operator's field-by-field order compiles to load, store, load, store, ...: a store to dst
// may alias the next field of src, so each load waits for the previous store's data, one global round trip per
// field (~23 in a row). Here the fields are loaded into registers first (one round trip), then stored; the
// categorical and vector-leaf slab copies run exactly as operator= runs them, from the same counts. Callers pass
// structs of two distinct split-info arrays (no overlap), so the stored values are operator='s.
struct SplitInfoFields {
  bool is_valid;
  int leaf_index;
  double gain;
  int inner_feature_index;
  uint32_t threshold;
  bool default_left;
  double left_sum_gradients;
  double left_sum_hessians;
  int64_t left_sum_of_gradients_hessians;
  data_size_t left_count;
  double left_gain;
  double left_value;
  double right_sum_gradients;
  double right_sum_hessians;
  int64_t right_sum_of_gradients_hessians;
  data_size_t right_count;
  double right_gain;
  double right_value;
  int num_cat_threshold;
  int num_vec_targets;
};

__device__ __forceinline__ SplitInfoFields LoadSplitInfoFields(const CUDASplitInfo* src) {
  SplitInfoFields f;
  f.is_valid = src->is_valid;
  f.leaf_index = src->leaf_index;
  f.gain = src->gain;
  f.inner_feature_index = src->inner_feature_index;
  f.threshold = src->threshold;
  f.default_left = src->default_left;
  f.left_sum_gradients = src->left_sum_gradients;
  f.left_sum_hessians = src->left_sum_hessians;
  f.left_sum_of_gradients_hessians = src->left_sum_of_gradients_hessians;
  f.left_count = src->left_count;
  f.left_gain = src->left_gain;
  f.left_value = src->left_value;
  f.right_sum_gradients = src->right_sum_gradients;
  f.right_sum_hessians = src->right_sum_hessians;
  f.right_sum_of_gradients_hessians = src->right_sum_of_gradients_hessians;
  f.right_count = src->right_count;
  f.right_gain = src->right_gain;
  f.right_value = src->right_value;
  f.num_cat_threshold = src->num_cat_threshold;
  f.num_vec_targets = src->num_vec_targets;
  return f;
}

// *dst = *src given f = LoadSplitInfoFields(src)
__device__ __forceinline__ void StoreSplitInfoFields(CUDASplitInfo* dst, const SplitInfoFields& f,
                                                     const CUDASplitInfo* src) {
  dst->is_valid = f.is_valid;
  dst->leaf_index = f.leaf_index;
  dst->gain = f.gain;
  dst->inner_feature_index = f.inner_feature_index;
  dst->threshold = f.threshold;
  dst->default_left = f.default_left;
  dst->left_sum_gradients = f.left_sum_gradients;
  dst->left_sum_hessians = f.left_sum_hessians;
  dst->left_sum_of_gradients_hessians = f.left_sum_of_gradients_hessians;
  dst->left_count = f.left_count;
  dst->left_gain = f.left_gain;
  dst->left_value = f.left_value;
  dst->right_sum_gradients = f.right_sum_gradients;
  dst->right_sum_hessians = f.right_sum_hessians;
  dst->right_sum_of_gradients_hessians = f.right_sum_of_gradients_hessians;
  dst->right_count = f.right_count;
  dst->right_gain = f.right_gain;
  dst->right_value = f.right_value;
  dst->num_cat_threshold = f.num_cat_threshold;
  if (f.num_cat_threshold > 0 && dst->cat_threshold != nullptr) {
    if (src->cat_threshold != nullptr) {
      for (int i = 0; i < f.num_cat_threshold; ++i) {
        dst->cat_threshold[i] = src->cat_threshold[i];
      }
      if (dst->cat_threshold_real != nullptr && src->cat_threshold_real != nullptr) {
        for (int i = 0; i < f.num_cat_threshold; ++i) {
          dst->cat_threshold_real[i] = src->cat_threshold_real[i];
        }
      }
    } else {
      dst->num_cat_threshold = 0;
    }
  }
  dst->num_vec_targets = f.num_vec_targets;
  if (f.num_vec_targets > 0 && dst->vec_payload != nullptr) {
    if (src->vec_payload != nullptr) {
      for (int i = 0; i < kNumVecPayloadFields * f.num_vec_targets; ++i) {
        dst->vec_payload[i] = src->vec_payload[i];
      }
    } else {
      dst->num_vec_targets = 0;
    }
  }
}

// sync_copy_batched's block reduction for the used-task sync when every found gain is finite: there
// OtherIsBetterWithTieBreak<double> (tolerance fmax(|a|, |b|) * 0 = 0) is "higher gain, or equal gain and lower
// index", and found beats not found. The key maps a finite gain to an unsigned integer of the same order (-0 shares
// +0's key: they compare equal), not found to 0, below every finite gain's key; the reduction is the maximum of
// (key, then lower index) in integer compares instead of a chain of fp64 compares. Found entries have distinct
// indices (distinct tasks), so the order is strict and the maximum is the original's winner; not-found entries
// all carry the same index (position 0's slot), as in the original.
__device__ __forceinline__ uint64_t FiniteGainOrderKey(const bool found, const double gain) {
  uint64_t bits = static_cast<uint64_t>(__double_as_longlong(gain));
  if (bits == 0x8000000000000000ull) bits = 0;
  const uint64_t key = (bits >> 63) != 0 ? ~bits : (bits | 0x8000000000000000ull);
  return found ? key : 0ull;
}

// blockDim.x a multiple of 32, at most 32 warps; the result is valid in thread 0
__device__ __forceinline__ uint32_t ReduceBestKeyBlock(uint64_t key, uint32_t index, uint64_t* shared_key,
                                                       uint32_t* shared_index) {
  const uint32_t lane = threadIdx.x % WARPSIZE;
  const uint32_t warp_id = threadIdx.x / WARPSIZE;
  const uint32_t num_warps = blockDim.x / WARPSIZE;
#pragma unroll
  for (uint32_t offset = WARPSIZE / 2; offset > 0; offset >>= 1) {
    const uint64_t other_key = __shfl_down_sync(0xffffffffu, key, offset);
    const uint32_t other_index = __shfl_down_sync(0xffffffffu, index, offset);
    if (other_key > key || (other_key == key && other_index < index)) {
      key = other_key;
      index = other_index;
    }
  }
  if (lane == 0) {
    shared_key[warp_id] = key;
    shared_index[warp_id] = index;
  }
  __syncthreads();
  if (warp_id == 0) {
    key = lane < num_warps ? shared_key[lane] : 0ull;
    index = lane < num_warps ? shared_index[lane] : 0xffffffffu;
#pragma unroll
    for (uint32_t offset = WARPSIZE / 2; offset > 0; offset >>= 1) {
      const uint64_t other_key = __shfl_down_sync(0xffffffffu, key, offset);
      const uint32_t other_index = __shfl_down_sync(0xffffffffu, index, offset);
      if (other_key > key || (other_key == key && other_index < index)) {
        key = other_key;
        index = other_index;
      }
    }
  }
  return index;
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
// BATCHED (cuda_plan key sync_copy_batched, host-launched flow): when every found gain of the block is finite, the
// block reduction is ReduceBestKeyBlock (the same winner: found entries have distinct slots; not-found entries tie on
// key 0 and the lowest index, thread 0's own slot, as ReduceBestGain keeps it) and thread 0 copies the winner with
// LoadSplitInfoFields / StoreSplitInfoFields; otherwise ReduceBestGain as before.
template <bool BATCHED = false>
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
  const bool gate_on_desc_counts,
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
  // Under NCCL (gate_on_desc_counts) the struct's num_data_in_leaf is the
  // rank-LOCAL row count -- gating on it re-imposes min_data_in_leaf per rank
  // and silently prunes leaves that are globally splittable (a 2-rank run cut
  // trees at half size). The descriptor counts are host-written GLOBAL counts
  // there; the struct hessian sums are global either way.
  const data_size_t gate_num_data = gate_on_desc_counts ?
    (is_larger ? desc->num_data_in_larger_leaf : desc->num_data_in_smaller_leaf) :
    leaf_splits->num_data_in_leaf;
  leaf_valid = leaf_valid && gate_num_data > min_data_in_leaf &&
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
  const int task_index = static_cast<int>(leaf_block_index * blockDim.x + threadIdx_x);
  const uint32_t read_index = is_larger ? static_cast<uint32_t>(task_index + num_tasks) : static_cast<uint32_t>(task_index);
  // The nothing-found default MUST name a slot of THIS role (in bounds).
  // It used to be plain 0 == (smaller role, task 0): when this role's fresh
  // search found no valid split, thread 0 then consulted the SMALLER role's
  // task-0 slot -- which is stale whenever the smaller sibling was skipped
  // (e.g. it sits at exactly min_data_in_leaf), because the find kernel
  // early-returns for invalid roles and never rewrites their slots. The
  // is_valid flag left there by an EARLIER level passed the final guard, and
  // an ancestor's candidate (tagged with tasks[0]'s feature, the lowest real
  // feature index) was resurrected as this leaf's best split: min_data
  // violations, leaf_weight != leaf_count, and unreachable leaves in the
  // serialized model. An own-role default is always safe: the slot is fresh
  // whenever this (valid) role was searched, and the mask re-check below
  // rejects tasks the compacted find skipped.
  const int safe_task_index = task_index < num_tasks ? task_index : num_tasks - 1;
  uint32_t shared_read_index = is_larger ? static_cast<uint32_t>(safe_task_index + num_tasks)
                                         : static_cast<uint32_t>(safe_task_index);
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
  bool finite_keys = false;
  if (BATCHED) {
    finite_keys = !__syncthreads_or(best_found && !isfinite(best_gain));
  } else {
    __syncthreads();
  }
  uint32_t best_read_index;
  if (BATCHED && finite_keys) {
    __shared__ uint64_t shared_key_buffer[BATCHED ? WARPSIZE : 1];
    best_read_index = ReduceBestKeyBlock(FiniteGainOrderKey(best_found, best_gain), shared_read_index,
                                         shared_key_buffer, shared_thread_index_buffer);
  } else {
    best_read_index = ReduceBestGain(best_gain, best_found, shared_read_index,
        shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
  }
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
    if (BATCHED) {
      // best_task_index is a task index whatever the slot holds: both loads in bounds
      const int best_feature = tasks[best_task_index].inner_feature_index;
      const SplitInfoFields best = LoadSplitInfoFields(best_split_info);
      if (best_task_used && best.is_valid) {
        StoreSplitInfoFields(cuda_split_info, best, best_split_info);
        cuda_split_info->inner_feature_index = best_feature;
        cuda_split_info->is_valid = true;
      } else {
        cuda_split_info->gain = kMinScore;
        cuda_split_info->is_valid = false;
      }
    } else if (best_task_used && best_split_info->is_valid) {
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
  const bool gate_on_desc_counts,
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
  // same NCCL local-vs-global count distinction as SyncBestSplitForLevelKernel
  const data_size_t gate_num_data = gate_on_desc_counts ?
    (is_larger ? desc->num_data_in_larger_leaf : desc->num_data_in_smaller_leaf) :
    leaf_splits->num_data_in_leaf;
  leaf_valid = leaf_valid && gate_num_data > min_data_in_leaf &&
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

// Used-task level sync (cuda_plan key sync_used_tasks): one block per (leaf role, pair) replaces
// SyncBestSplitForLevelKernel's blocks of NUM_TASKS_PER_SYNC_BLOCK tasks over every task plus the AllBlocks merge.
// Each thread folds the used tasks at positions threadIdx.x, threadIdx.x + blockDim.x, ... (ascending task order;
// used_task_indices == nullptr means every task) keeping the first strict maximum, then the block reduces through
// ReduceBestGain. With finite gains, OtherIsBetterWithTieBreak (fp64: no tolerance) and the merge's strict ">"
// with the lower block holding both order (gain, then lower task) strictly, and unused or not-found lanes never
// win, so the original's winner is the unique maximum of the found used tasks, which any fold order returns. A
// non-finite found gain breaks the total order, so then the block replays the original exactly: every
// 1024-task block's 32 warps (run as virtual warps, same lanes, same shuffle tree), its 32-warp reduction and
// mask, then the merge's sequential fold over blocks. Only slot leaf_index is written, with the value the
// original leaves there; the per-block slots leaf + z * num_leaves are scratch of the merge.
// The used-task reads do not depend on the leaf, so they are issued before the leaf's validity loads.
// BATCHED (cuda_plan key sync_copy_batched): the winner's slot, its validity and its task's feature index are
// loaded in one batch before any store (LoadSplitInfoFields / StoreSplitInfoFields), the same values written.
// blockDim.x: a multiple of 32, at most NUM_TASKS_PER_SYNC_BLOCK.
template <bool BATCHED>
__global__ void SyncBestSplitForLevelUsedTasksKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  CUDASplitInfo* cuda_leaf_best_split_info,
  const SplitFindTask* tasks,
  const int8_t* is_feature_used_bytree,
  const int* used_task_indices,
  const int num_used_tasks,
  const CUDASplitInfo* cuda_best_split_info,
  const int num_tasks,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const bool gate_on_desc_counts) {
  constexpr int kReplayWarps = NUM_TASKS_PER_SYNC_BLOCK / WARPSIZE;
  __shared__ double shared_gain_buffer[kReplayWarps];
  __shared__ bool shared_found_buffer[kReplayWarps];
  __shared__ uint32_t shared_thread_index_buffer[kReplayWarps];
  const unsigned int pair_index = blockIdx.y;
  const bool is_larger = (blockIdx.x == 1);
  const CUDASplitInfo* pair_split_info = cuda_best_split_info +
    static_cast<size_t>(pair_index) * (2 * static_cast<size_t>(num_tasks));
  const uint32_t role_offset = is_larger ? static_cast<uint32_t>(num_tasks) : 0u;
  bool found = false;
  bool non_finite = false;
  double gain = kMinScore;
  // a thread that finds nothing keeps the slot of position 0, which thread 0 reads and finds invalid, so a
  // reduction that finds nothing returns an invalid slot
  uint32_t read_index = role_offset + static_cast<uint32_t>(used_task_indices != nullptr ? used_task_indices[0] : 0);
  const CUDAHybridPairDescriptor* desc = pair_descs + pair_index;
  const CUDALeafSplitsStruct* leaf_splits;
  int leaf_index;
  bool leaf_valid;
  if (BATCHED) {
    // the fold in chunks of kFold positions per thread (pos, pos + blockDim.x, ...: the same ascending order):
    // a chunk's task-index loads, then its validity and gain loads (the gain is loaded whatever the validity, so it
    // does not wait for it), then its compares; the leaf descriptor's loads go out with the first chunk's
    constexpr int kFold = 4;
    const int stride = static_cast<int>(blockDim.x);
    uint32_t idx[kFold];
    bool valid[kFold];
    double task_gain[kFold];
    auto load_indices = [&](const int first) {
#pragma unroll
      for (int k = 0; k < kFold; ++k) {
        const int pos = first + k * stride;
        idx[k] = role_offset + static_cast<uint32_t>(
          pos < num_used_tasks ? (used_task_indices != nullptr ? used_task_indices[pos] : pos) : 0);
      }
    };
    // a leaf the descriptor marks invalid is written invalid below whatever the fold finds: its slots are not read
    // (the flag arrives with the task indices, so the slot loads wait no longer)
    bool fold_slots = true;
    auto load_slots = [&](const int first) {
#pragma unroll
      for (int k = 0; k < kFold; ++k) {
        valid[k] = false;
        task_gain[k] = 0.0;
        if (fold_slots && first + k * stride < num_used_tasks) {
          valid[k] = pair_split_info[idx[k]].is_valid;
          task_gain[k] = pair_split_info[idx[k]].gain;
        }
      }
    };
    auto fold = [&]() {
#pragma unroll
      for (int k = 0; k < kFold; ++k) {
        if (valid[k]) {
          non_finite = non_finite || !isfinite(task_gain[k]);
          if (!found || task_gain[k] > gain) {
            found = true;
            gain = task_gain[k];
            read_index = idx[k];
          }
        }
      }
    };
    const int first = static_cast<int>(threadIdx.x);
    load_indices(first);
    leaf_splits = is_larger ? desc->larger_struct : desc->smaller_struct;
    const bool desc_valid = is_larger ? (desc->larger_valid != 0) : (desc->smaller_valid != 0);
    const data_size_t desc_num_data = is_larger ? desc->num_data_in_larger_leaf : desc->num_data_in_smaller_leaf;
    fold_slots = desc_valid;
    load_slots(first);
    leaf_index = leaf_splits->leaf_index;
    // same NCCL local-vs-global count distinction as SyncBestSplitForLevelKernel
    const data_size_t gate_num_data = gate_on_desc_counts ? desc_num_data : leaf_splits->num_data_in_leaf;
    const double leaf_sum_hessians = leaf_splits->sum_of_hessians;
    fold();
    for (int next = first + kFold * stride; next < num_used_tasks; next += kFold * stride) {
      load_indices(next);
      load_slots(next);
      fold();
    }
    if (leaf_index < 0) {
      return;
    }
    leaf_valid = desc_valid && gate_num_data > min_data_in_leaf && leaf_sum_hessians > min_sum_hessian_in_leaf;
  } else {
    for (int pos = static_cast<int>(threadIdx.x); pos < num_used_tasks; pos += static_cast<int>(blockDim.x)) {
      const uint32_t task_read_index = role_offset +
        static_cast<uint32_t>(used_task_indices != nullptr ? used_task_indices[pos] : pos);
      if (pair_split_info[task_read_index].is_valid) {
        const double task_gain = pair_split_info[task_read_index].gain;
        non_finite = non_finite || !isfinite(task_gain);
        if (!found || task_gain > gain) {
          found = true;
          gain = task_gain;
          read_index = task_read_index;
        }
      }
    }
    leaf_splits = is_larger ? desc->larger_struct : desc->smaller_struct;
    leaf_index = leaf_splits->leaf_index;
    if (leaf_index < 0) {
      return;
    }
    leaf_valid = is_larger ? (desc->larger_valid != 0) : (desc->smaller_valid != 0);
    // same NCCL local-vs-global count distinction as SyncBestSplitForLevelKernel
    const data_size_t gate_num_data = gate_on_desc_counts ?
      (is_larger ? desc->num_data_in_larger_leaf : desc->num_data_in_smaller_leaf) :
      leaf_splits->num_data_in_leaf;
    leaf_valid = leaf_valid && gate_num_data > min_data_in_leaf &&
      leaf_splits->sum_of_hessians > min_sum_hessian_in_leaf;
  }
  CUDASplitInfo* cuda_split_info = cuda_leaf_best_split_info + leaf_index;
  if (!leaf_valid) {
    if (threadIdx.x == 0) {
      cuda_split_info->is_valid = false;
    }
    return;
  }
  if (!__syncthreads_or(non_finite)) {
    __shared__ uint64_t shared_key_buffer[BATCHED ? kReplayWarps : 1];
    const uint32_t best_read_index = BATCHED ?
      ReduceBestKeyBlock(FiniteGainOrderKey(found, gain), read_index, shared_key_buffer, shared_thread_index_buffer) :
      ReduceBestGain(gain, found, read_index, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
    if (threadIdx.x == 0) {
      const CUDASplitInfo* best_split_info = pair_split_info + best_read_index;
      if (BATCHED) {
        // best_read_index - role_offset is always a task index (a used task, or position 0's), so both loads are
        // in bounds whatever the slot holds
        const int best_feature = tasks[best_read_index - role_offset].inner_feature_index;
        const SplitInfoFields best = LoadSplitInfoFields(best_split_info);
        if (best.is_valid) {
          StoreSplitInfoFields(cuda_split_info, best, best_split_info);
          cuda_split_info->inner_feature_index = best_feature;
          cuda_split_info->is_valid = true;
        } else {
          cuda_split_info->gain = kMinScore;
          cuda_split_info->is_valid = false;
        }
      } else if (best_split_info->is_valid) {
        *cuda_split_info = *best_split_info;
        cuda_split_info->inner_feature_index = tasks[best_read_index - role_offset].inner_feature_index;
        cuda_split_info->is_valid = true;
      } else {
        cuda_split_info->gain = kMinScore;
        cuda_split_info->is_valid = false;
      }
    }
    return;
  }
  // exact replay of SyncBestSplitForLevelKernel (blocks of NUM_TASKS_PER_SYNC_BLOCK threads: warp w, lane l holds
  // task z * NUM_TASKS_PER_SYNC_BLOCK + 32 w + l) and the AllBlocks merge; only thread 0's merge state is used
  const uint32_t lane = threadIdx.x % WARPSIZE;
  const int warp_id = static_cast<int>(threadIdx.x / WARPSIZE);
  const int num_warps = static_cast<int>(blockDim.x / WARPSIZE);
  bool merged_valid = false;
  double merged_gain = kMinScore;
  uint32_t merged_read_index = 0;
  const int num_blocks_per_leaf = (num_tasks + NUM_TASKS_PER_SYNC_BLOCK - 1) / NUM_TASKS_PER_SYNC_BLOCK;
  for (int block_index = 0; block_index < num_blocks_per_leaf; ++block_index) {
    for (int virtual_warp = warp_id; virtual_warp < kReplayWarps; virtual_warp += num_warps) {
      const int task_index = block_index * NUM_TASKS_PER_SYNC_BLOCK + virtual_warp * WARPSIZE + static_cast<int>(lane);
      const int safe_task_index = task_index < num_tasks ? task_index : num_tasks - 1;
      bool best_found = false;
      double best_gain = kMinScore;
      uint32_t shared_read_index = role_offset + static_cast<uint32_t>(safe_task_index);
      if (task_index < num_tasks &&
          (is_feature_used_bytree == nullptr ||
           is_feature_used_bytree[tasks[task_index].inner_feature_index])) {
        shared_read_index = role_offset + static_cast<uint32_t>(task_index);
        best_found = pair_split_info[shared_read_index].is_valid;
        best_gain = pair_split_info[shared_read_index].gain;
      }
      ReduceBestGainWarp(best_gain, best_found, shared_read_index, shared_gain_buffer + virtual_warp,
                         shared_found_buffer + virtual_warp, shared_thread_index_buffer + virtual_warp);
    }
    __syncthreads();
    if (warp_id == 0) {
      // ReduceBestGain's second stage with its block of kReplayWarps warps
      const bool has_warp = lane < static_cast<uint32_t>(kReplayWarps);
      const uint32_t best_read_index = ReduceBestGainBlock(
        has_warp ? shared_gain_buffer[lane] : static_cast<double>(kMinScore),
        has_warp ? shared_found_buffer[lane] : false,
        has_warp ? shared_thread_index_buffer[lane] : 0u);
      if (lane == 0) {
        const int best_task_index = static_cast<int>(best_read_index - role_offset);
        const bool best_task_used = is_feature_used_bytree == nullptr ||
          is_feature_used_bytree[tasks[best_task_index].inner_feature_index];
        const CUDASplitInfo* best_split_info = pair_split_info + best_read_index;
        const bool block_valid = best_task_used && best_split_info->is_valid;
        const double block_gain = block_valid ? best_split_info->gain : static_cast<double>(kMinScore);
        if (block_index == 0 || (block_valid && merged_valid && block_gain > merged_gain) ||
            (!merged_valid && block_valid)) {
          merged_valid = block_valid;
          merged_gain = block_gain;
          merged_read_index = best_read_index;
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (merged_valid) {
      *cuda_split_info = pair_split_info[merged_read_index];
      cuda_split_info->inner_feature_index = tasks[merged_read_index - role_offset].inner_feature_index;
      cuda_split_info->is_valid = true;
    } else {
      cuda_split_info->gain = kMinScore;
      cuda_split_info->is_valid = false;
    }
  }
}

void CUDABestSplitFinder::LaunchFindBestSplitsForLevelKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const CUDAHybridGraphLoopStateOpt gstate,
  const bool use_desc_counts) {
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
      max_delta_step_, \
      cat_smooth_, \
      cat_l2_, \
      max_cat_threshold_, \
      min_data_per_group_, \
      cuda_best_split_info_.RawData(), \
      use_desc_counts, \
      gstate
  if (FalcataFP32GainEnabled()) {
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

#if !defined(__HIP_PLATFORM_AMD__)
// Warp-per-task quantized level finder (cuda_plan key warp_find). CUDA only: it assumes 32-lane warps.
struct WarpFindCandidate {
  double sl_g, sl_h, sr_g, sr_h, gain;
  data_size_t lc, rc;
  int64_t slgh, srgh;
};

// One warp scans one (task, leaf): each lane owns 8 consecutive scan positions. The prefix sum is a per-lane
// register scan plus a warp shuffle scan of the lane totals (the block kernel's wrapping packed-integer adds);
// the per-threshold unpack, gates and gain are the block kernel's expressions; the best threshold is a strict
// ascending scan per lane followed by a shuffle reduction through OtherIsBetterWithTieBreak.
// PPL = 1 (cuda_plan key warp_find_spread, scans of at most 32 positions): each lane owns one position, so the
// fp64 unpack and gain math of a few-bin feature runs once across lanes instead of serially in lane 0. The
// selection then reproduces the 8-per-lane result exactly: an exact first-maximum within each group of 8 lanes
// (what one lane's strict ascending scan of those 8 positions yields), then the same tolerance tie-break between
// the groups in the same pairing order as between 8-position lanes 0..3 (lanes 4..31 own no position there).
// G = 4 (with PPL = 2, every task of at most 8 positions): one group of 4 lanes per (task, leaf), eight per warp;
// at most 8 positions all sit in 8-position lane 0, whose strict ascending scan is the exact first maximum and
// meets no other candidate in the tolerance reduction, so each lane's strict scan of its 2 positions plus an
// exact first-maximum across the group's lanes is the same winner.
// G = 8 (with PPL = 1, cuda_plan key warp_find_narrow, every task of at most 8 bins): one group of 8 lanes per
// (task, leaf), four per warp, one position per lane; the same exact first maximum across the group's lanes.
// STRIDED (G = 4, cuda_plan key warp_find_strided): lane l's slot i holds position i * G + l instead of l * PPL + i,
// so the (at most 4) candidates of a 5-bin task all sit in slot 0 and a warp whose items all fit skips slot 1's
// fp64 work. Each slot's positions are scanned across the group and offset by the earlier slots' totals (the same
// wrapping integer prefixes); a lane's slots stay in ascending position order and the group reduction is the
// exact first maximum of (gain, then lower position), so which lane holds a candidate does not change the winner.
template <bool REVERSE, bool B16, typename ACC_T, int PPL, int G, bool STRIDED = false>
__device__ __forceinline__ void WarpFindBest(
    const ACC_T* hist, const SplitFindTask* task,
    const double lambda_l2, const double max_delta_step, const data_size_t min_data_in_leaf,
    const double min_sum_hessian_in_leaf, const double min_gain_to_split,
    const double parent_gain, const int64_t sum_gradients_hessians, const data_size_t num_data,
    const double parent_output, const double grad_scale, const double hess_scale,
    const bool prune_fp32, const bool select_int, CUDASplitInfo* out) {
  using UT = std::make_unsigned_t<ACC_T>;
  constexpr uint32_t kNoThreshold = 0xffffffffu;
  static_assert(G == 32 || (G == 4 && PPL == 2) || (G == 8 && PPL == 1),
                "4-lane groups own 2 positions per lane, 8-lane groups one");
  static_assert(!STRIDED || G == 4, "strided positions: 4-lane groups only");
  const uint32_t lane = threadIdx.x & static_cast<uint32_t>(G - 1);
  const unsigned int mask = G == 32 ? 0xffffffffu : (((1u << G) - 1u) << ((threadIdx.x & 31u) & ~(G - 1u)));
  // the level kernel only runs without L1 and path smoothing (SupportsBatchedLevel)
  const double lambda_l1 = 0.0;
  const double path_smooth = 0.0;
  const double sum_hessians = static_cast<double>(sum_gradients_hessians & 0x00000000ffffffff) * hess_scale;
  const double cnt_factor = num_data / sum_hessians;
  const double min_gain_shift = parent_gain + min_gain_to_split;

  const uint32_t fnb = task->num_bin - task->mfb_offset;
  const uint32_t na = static_cast<uint32_t>(task->na_as_missing);
  const uint32_t base = lane * static_cast<uint32_t>(PPL);
  // scan position of the lane's slot i
  auto position = [&](const int i) -> uint32_t {
    return STRIDED ? static_cast<uint32_t>(i) * static_cast<uint32_t>(G) + lane : base + static_cast<uint32_t>(i);
  };
  // Forward scan with a NaN bin AND an unstored most-frequent bin 0 (the block kernel's na_as_missing &&
  // mfb_offset == 1 case): position t holds bin t, read from hist[t - 1] for t >= 1, and bin 0 is synthesised
  // from the leaf total below.
  const uint32_t shift = (!REVERSE && task->na_as_missing && task->mfb_offset == 1) ? 1u : 0u;

  UT v[PPL];
#pragma unroll
  for (int i = 0; i < PPL; ++i) {
    const uint32_t t = position(i);
    UT x = 0;
    if (REVERSE ? (t >= na && t < fnb) : (t >= shift && t - shift < fnb)) {
      x = static_cast<UT>(hist[REVERSE ? (fnb - 1 - t) : (t - shift)]);
    }
    v[i] = x;
  }
  UT excl = 0;
  UT total = 0;  // sum of every position of the group
  if (STRIDED) {
    // slot i: inclusive scan of positions i * G .. i * G + G - 1 across the group, plus every earlier slot's total
#pragma unroll
    for (int i = 0; i < PPL; ++i) {
      UT incl_i = v[i];
#pragma unroll
      for (uint32_t d = 1; d < static_cast<uint32_t>(G); d <<= 1) {
        const UT y = __shfl_up_sync(mask, incl_i, d, G);
        if (lane >= d) incl_i += y;
      }
      v[i] = total + incl_i;
      total += __shfl_sync(mask, incl_i, G - 1, G);
    }
  } else {
    UT run = 0;
#pragma unroll
    for (int i = 0; i < PPL; ++i) {
      run += v[i];
      v[i] = run;
    }
    UT incl = run;
#pragma unroll
    for (uint32_t d = 1; d < static_cast<uint32_t>(G); d <<= 1) {
      const UT y = __shfl_up_sync(mask, incl, d, G);
      if (lane >= d) incl += y;
    }
    excl = incl - run;
    if (shift != 0) total = __shfl_sync(mask, incl, G - 1, G);
  }
  if (shift != 0) {
    // bin 0 = leaf total - sum of the stored bins, through the block kernel's packed conversions; it sits at
    // position 0, so it adds to every prefix
    const ACC_T sum_non_default = static_cast<ACC_T>(total);
    const int64_t non_default_packed = B16 ?
      ((static_cast<int64_t>(static_cast<int16_t>(sum_non_default >> 16)) << 32) |
       static_cast<int64_t>(sum_non_default & 0x0000ffff)) :
      static_cast<int64_t>(sum_non_default);
    const int64_t default_bin_packed = sum_gradients_hessians - non_default_packed;
    const ACC_T bin0 = B16 ?
      static_cast<ACC_T>(static_cast<int32_t>(
        (static_cast<uint32_t>(static_cast<int32_t>(default_bin_packed >> 32)) << 16) |
        (static_cast<uint32_t>(default_bin_packed) & 0x0000ffffu))) :
      static_cast<ACC_T>(default_bin_packed);
    excl += static_cast<UT>(bin0);
  }

  // the block kernel's unpack and validity gates for one threshold
  auto prep = [&](UT accu, WarpFindCandidate& o) -> bool {
    const ACC_T acc = static_cast<ACC_T>(accu);
    const int64_t packed = B16 ?
      ((static_cast<int64_t>(static_cast<int16_t>(acc >> 16)) << 32) | static_cast<int64_t>(acc & 0x0000ffff)) :
      static_cast<int64_t>(acc);
    if (REVERSE) {
      o.srgh = packed;
      o.sr_g = static_cast<double>(static_cast<int32_t>((o.srgh & 0xffffffff00000000) >> 32)) * grad_scale;
      o.sr_h = static_cast<double>(static_cast<int32_t>(o.srgh & 0x00000000ffffffff)) * hess_scale;
      o.rc = static_cast<data_size_t>(CUDARoundInt(o.sr_h * cnt_factor));
      o.slgh = sum_gradients_hessians - o.srgh;
      o.sl_g = static_cast<double>(static_cast<int32_t>((o.slgh & 0xffffffff00000000) >> 32)) * grad_scale;
      o.sl_h = static_cast<double>(static_cast<int32_t>(o.slgh & 0x00000000ffffffff)) * hess_scale;
      o.lc = num_data - o.rc;
    } else {
      o.slgh = packed;
      o.sl_g = static_cast<double>(static_cast<int32_t>((o.slgh & 0xffffffff00000000) >> 32)) * grad_scale;
      o.sl_h = static_cast<double>(static_cast<int32_t>(o.slgh & 0x00000000ffffffff)) * hess_scale;
      o.lc = static_cast<data_size_t>(CUDARoundInt(o.sl_h * cnt_factor));
      o.srgh = sum_gradients_hessians - o.slgh;
      o.sr_g = static_cast<double>(static_cast<int32_t>((o.srgh & 0xffffffff00000000) >> 32)) * grad_scale;
      o.sr_h = static_cast<double>(static_cast<int32_t>(o.srgh & 0x00000000ffffffff)) * hess_scale;
      o.rc = num_data - o.lc;
    }
    return o.sl_h >= min_sum_hessian_in_leaf && o.lc >= min_data_in_leaf &&
           o.sr_h >= min_sum_hessian_in_leaf && o.rc >= min_data_in_leaf;
  };

  // Pruning (only without max_delta_step, where each candidate's gain is the closed form
  // g_l^2/(h_l+eps+l2) + g_r^2/(h_r+eps+l2)): an fp32 estimate widened by 2e-4 relative plus 1e-30 absolute
  // bounds each candidate. A candidate is skipped only when its upper bound is below another valid candidate's
  // lower bound or below the min-gain cutoff, so it can neither win nor tie. Candidates whose squared gradient
  // sums underflow fp32 are never pruned and set no lower bound.
  const bool prune = max_delta_step <= 0.0 && lambda_l2 >= 0.0;
  constexpr float kRel = 2.0e-4f;
  constexpr float kAbs = 1.0e-30f;
  float up[PPL];
  // STRIDED: the prune's unpack of each slot, reused by the gain loop (prep is a pure function of the prefix)
  WarpFindCandidate pre[STRIDED ? PPL : 1];  // NOLINT(runtime/arrays): a compile-time size
  uint32_t vmask = 0;
#pragma unroll
  for (int i = 0; i < PPL; ++i) {
    const uint32_t t = position(i);
    const bool candidate = REVERSE ? (t >= na && t <= task->num_bin - 2) : (t + 2 <= fnb + shift);
    if (candidate) vmask |= (1u << i);
    up[i] = INFINITY;
  }
  if (prune) {
    const float l2e = static_cast<float>(lambda_l2 + kEpsilon);
    const float mg = static_cast<float>(min_gain_shift);
    float threshold = mg - fabsf(mg) * kRel - kAbs;
    // (float)hess_scale normal and |int32| * it finite in fp32, so every fp32 hessian below is within a few 1e-7
    // relative of prep's (a too-small or too-large scale keeps the exact per-position prune below)
    const float hs_f32 = static_cast<float>(hess_scale);
    if (G == 32 && PPL >= 2 && !STRIDED && prune_fp32 && min_sum_hessian_in_leaf > 0.0 && hs_f32 >= FLT_MIN &&
        hs_f32 <= FLT_MAX / 4.3e9f) {
      // cuda_plan key find_prune_fp32: the bounds from the int32 halves in fp32 (the int->float converts and the
      // product round three times instead of once: a few 1e-7 relative, far inside kRel; a valid side's hessian
      // is >= min_sum_hessian_in_leaf > 0, so both denominators are positive and the terms do not cancel). A
      // position failing the exact gates keeps its bound here and is rejected by the gain loop's exact prep; the
      // lower bound must come from a valid position, so each lane offers one: the largest lower bound among its
      // positions passing approximate gates (a choice only), verified with the exact prep.
      const float gs_f = static_cast<float>(grad_scale);
      const float hs_f = hs_f32;
      const float cf_f = static_cast<float>(cnt_factor);
      const float msh_f = static_cast<float>(min_sum_hessian_in_leaf);
      const float md_f = static_cast<float>(min_data_in_leaf) - 0.5f;
      // Exact-safe screen: a position whose fp32 sums miss a gate by more than 1e-5 relative (the fp32 values are
      // within ~3e-7 relative of prep's fp64 ones when every factor is a positive normal float) fails prep's gate
      // too, and leaves vmask here instead of costing the gain loop an exact prep. The gates are
      // sl_h, sr_h >= min_sum_hessian_in_leaf and, with x = (rounded side's hessian) * cnt_factor and that side's
      // count (int)(x + 0.5): x >= min_data - 0.5 and x < num_data - min_data + 0.5 (the other side's count is
      // num_data minus it); both count bounds are positive where the screen is on, so the margins point inward.
      const bool screen = msh_f >= 1.0e-30f;  // hess_scale > 0: (float) of it >= FLT_MIN above
      const float scr_h = msh_f * (1.0f - 1.0e-5f);
      const bool screen_cnt = screen && min_data_in_leaf >= 1 && num_data - min_data_in_leaf >= min_data_in_leaf &&
                              cf_f >= FLT_MIN && cf_f <= FLT_MAX;
      const float scr_lo = md_f * (1.0f - 1.0e-5f);
      const float scr_hi = (static_cast<float>(num_data - min_data_in_leaf) + 0.5f) * (1.0f + 1.0e-5f);
      // Exact dominance: a position whose prefix equals the previous position's (an empty bin) has the same sums,
      // gates and gain, and loses the exact (gain, then lower position) order to it whenever that position is a
      // candidate too (a pruned one implies this one's identical bound is pruned as well).
      const uint32_t cand_mask = vmask;
#pragma unroll
      for (int i = 1; i < PPL; ++i) {
        if (((cand_mask >> (i - 1)) & 1u) && v[i] == v[i - 1]) vmask &= ~(1u << i);
      }
      float lane_lo = -INFINITY;
      UT lane_v = 0;
#pragma unroll
      for (int i = 0; i < PPL; ++i) {
        if ((vmask >> i) & 1u) {
          // prep's packed sums, left = prefix (forward) or leaf total - prefix (reverse)
          const ACC_T acc = static_cast<ACC_T>(v[i] + excl);
          const int64_t packed = B16 ?
            ((static_cast<int64_t>(static_cast<int16_t>(acc >> 16)) << 32) | static_cast<int64_t>(acc & 0x0000ffff)) :
            static_cast<int64_t>(acc);
          const int64_t other = sum_gradients_hessians - packed;
          const int64_t left = REVERSE ? other : packed;
          const int64_t right = REVERSE ? packed : other;
          const float gl = static_cast<float>(static_cast<int32_t>(left >> 32)) * gs_f;
          const float hl = static_cast<float>(static_cast<int32_t>(left & 0x00000000ffffffff)) * hs_f;
          const float gr = static_cast<float>(static_cast<int32_t>(right >> 32)) * gs_f;
          const float hr = static_cast<float>(static_cast<int32_t>(right & 0x00000000ffffffff)) * hs_f;
          const float xc = (REVERSE ? hr : hl) * cf_f;
          if ((screen && (hl < scr_h || hr < scr_h)) || (screen_cnt && (xc < scr_lo || xc > scr_hi))) {
            vmask &= ~(1u << i);
            continue;
          }
          const float gl2 = gl * gl;
          const float gr2 = gr * gr;
          if (gl2 < FLT_MIN || gr2 < FLT_MIN) continue;  // underflow: keep, set no lower bound
          const float a = gl2 / (hl + l2e) + gr2 / (hr + l2e);
          up[i] = a * (1.0f + kRel) + kAbs;
          const float lo = a * (1.0f - kRel) - kAbs;
          if (lo < 1.0e30f && lo > lane_lo && hl >= msh_f && hr >= msh_f && hl * cf_f >= md_f && hr * cf_f >= md_f) {
            lane_lo = lo;
            lane_v = v[i];
          }
        }
      }
      if (lane_lo > threshold) {
        WarpFindCandidate o;
        if (prep(lane_v + excl, o)) threshold = lane_lo;
      }
    } else {
#pragma unroll
    for (int i = 0; i < PPL; ++i) {
      if ((vmask >> i) & 1u) {
        WarpFindCandidate o_local;
        WarpFindCandidate& o = STRIDED ? pre[STRIDED ? i : 0] : o_local;
        if (!prep(v[i] + excl, o)) {
          vmask &= ~(1u << i);
          continue;
        }
        const float gl = static_cast<float>(o.sl_g);
        const float gr = static_cast<float>(o.sr_g);
        const float gl2 = gl * gl;
        const float gr2 = gr * gr;
        if (gl2 < FLT_MIN || gr2 < FLT_MIN) continue;  // underflow: keep, set no lower bound
        const float a = gl2 / (static_cast<float>(o.sl_h) + l2e) + gr2 / (static_cast<float>(o.sr_h) + l2e);
        up[i] = a * (1.0f + kRel) + kAbs;
        const float lo = a * (1.0f - kRel) - kAbs;
        if (lo < 1.0e30f) threshold = fmaxf(threshold, lo);
      }
    }
    }
#pragma unroll
    for (uint32_t off = G / 2; off > 0; off >>= 1) {
      threshold = fmaxf(threshold, __shfl_xor_sync(mask, threshold, off));
    }
#pragma unroll
    for (int i = 0; i < PPL; ++i) {
      if (up[i] < threshold) vmask &= ~(1u << i);
    }
  }

  double best_gain = kMinScore;
  uint32_t best_t = kNoThreshold;
  WarpFindCandidate best{};
  if (G == 32 && PPL >= 2 && !STRIDED && prune_fp32) {
    // find_prune_fp32: the lane's surviving positions in ascending order as a work list, so the warp runs as many
    // exact prep + gain rounds as its busiest lane has survivors instead of one per slot any lane holds
    uint32_t rem = vmask;
    while (rem != 0u) {
      const int i = __ffs(rem) - 1;
      rem &= rem - 1u;
      UT val = v[0];
#pragma unroll
      for (int j = 1; j < PPL; ++j) {
        if (i == j) val = v[j];
      }
      WarpFindCandidate o;
      if (!prep(val + excl, o)) continue;
      const double current_gain = CUDALeafSplits::GetSplitGains<false, false, double>(
          o.sl_g, o.sl_h + kEpsilon, o.sr_g, o.sr_h + kEpsilon, lambda_l1,
          lambda_l2, path_smooth, max_delta_step, o.lc, o.rc, parent_output);
      if (current_gain > min_gain_shift && (best_t == kNoThreshold || current_gain - min_gain_shift > best_gain)) {
        best_gain = current_gain - min_gain_shift;
        best_t = base + static_cast<uint32_t>(i);
        best = o;
      }
    }
  } else {
#pragma unroll
  for (int i = 0; i < PPL; ++i) {
    if ((vmask >> i) & 1u) {
      WarpFindCandidate o;
      if (STRIDED && prune) {
        o = pre[STRIDED ? i : 0];  // passed prep in the prune (a failing slot left vmask there)
      } else if (!prep(v[i] + excl, o)) {
        continue;
      }
      const double current_gain = CUDALeafSplits::GetSplitGains<false, false, double>(
          o.sl_g, o.sl_h + kEpsilon, o.sr_g, o.sr_h + kEpsilon, lambda_l1,
          lambda_l2, path_smooth, max_delta_step, o.lc, o.rc, parent_output);
      if (current_gain > min_gain_shift && (best_t == kNoThreshold || current_gain - min_gain_shift > best_gain)) {
        best_gain = current_gain - min_gain_shift;
        best_t = position(i);
        best = o;
      }
    }
  }
  }
  if (STRIDED) {
    // the same exact first maximum in integer compares: a lane holding a threshold has best_gain =
    // current_gain - min_gain_shift with current_gain > min_gain_shift, so best_gain > 0 and not NaN (x > y gives
    // x - y > 0 with gradual underflow; +inf only from overflow), and positive doubles order and compare equal
    // exactly as their bit patterns do as unsigned integers; lanes without one are excluded by kNoThreshold
    uint64_t best_bits = static_cast<uint64_t>(__double_as_longlong(best_gain));
#pragma unroll
    for (uint32_t off = G / 2; off > 0; off >>= 1) {
      const uint64_t other_bits = __shfl_xor_sync(mask, best_bits, off);
      const uint32_t other_t = __shfl_xor_sync(mask, best_t, off);
      if (other_t != kNoThreshold &&
          (best_t == kNoThreshold || other_bits > best_bits || (other_bits == best_bits && other_t < best_t))) {
        best_bits = other_bits;
        best_t = other_t;
      }
    }
    best_gain = __longlong_as_double(static_cast<long long>(best_bits));  // NOLINT(runtime/int): the intrinsic's type
  } else if (G == 32 && select_int) {
    // cuda_plan key find_select_int: the two reductions below in integer compares (see the STRIDED branch: a lane
    // holding a threshold has a positive non-NaN gain). OtherIsBetterWithTieBreak's tolerance is
    // fmax(|a|, |b|) * 0, NaN when either gain is +inf, which makes it return false: kept as the kInfBits test.
    constexpr uint64_t kInfBits = 0x7ff0000000000000ull;
    uint64_t best_bits = static_cast<uint64_t>(__double_as_longlong(best_gain));
#pragma unroll
    for (uint32_t off = 4u; PPL == 1 && off > 0; off >>= 1) {
      const uint64_t other_bits = __shfl_xor_sync(mask, best_bits, off);
      const uint32_t other_t = __shfl_xor_sync(mask, best_t, off);
      if (other_t != kNoThreshold &&
          (best_t == kNoThreshold || other_bits > best_bits || (other_bits == best_bits && other_t < best_t))) {
        best_bits = other_bits;
        best_t = other_t;
      }
    }
#pragma unroll
    for (uint32_t off = 16; off >= (PPL == 1 ? 8u : 1u); off >>= 1) {
      const uint64_t other_bits = __shfl_xor_sync(mask, best_bits, off);
      const uint32_t other_t = __shfl_xor_sync(mask, best_t, off);
      if (other_t != kNoThreshold &&
          (best_t == kNoThreshold || (other_bits != kInfBits && best_bits != kInfBits &&
                                      (other_bits > best_bits || (other_bits == best_bits && other_t < best_t))))) {
        best_bits = other_bits;
        best_t = other_t;
      }
    }
    best_gain = __longlong_as_double(static_cast<long long>(best_bits));  // NOLINT(runtime/int): the intrinsic's type
  } else if (G < 32 || PPL == 1) {
    // exact first maximum of (gain, then lowest threshold) within each group of 8 positions
#pragma unroll
    for (uint32_t off = (G < 32 ? G / 2 : 4u); off > 0; off >>= 1) {
      const double other_gain = __shfl_xor_sync(mask, best_gain, off);
      const uint32_t other_t = __shfl_xor_sync(mask, best_t, off);
      if (other_t != kNoThreshold &&
          (best_t == kNoThreshold || other_gain > best_gain || (other_gain == best_gain && other_t < best_t))) {
        best_gain = other_gain;
        best_t = other_t;
      }
    }
  }
  // PPL == 1: groups g and g ^ 2, then g and g ^ 1 -- the 8-per-lane order with its empty lanes 4..31 skipped
#pragma unroll
  for (uint32_t off = 16; G == 32 && !select_int && off >= (PPL == 1 ? 8u : 1u); off >>= 1) {
    const double other_gain = __shfl_xor_sync(mask, best_gain, off);
    const uint32_t other_t = __shfl_xor_sync(mask, best_t, off);
    if (other_t != kNoThreshold &&
        (best_t == kNoThreshold || OtherIsBetterWithTieBreak<double>(other_gain, best_gain, other_t, best_t))) {
      best_gain = other_gain;
      best_t = other_t;
    }
  }
  if (best_t == kNoThreshold) {
    if (lane == 0) out->is_valid = false;
    return;
  }
  if (STRIDED || (G == 32 && select_int)) {
    // the winner lane (every lane of the group holds best_t) computes the left child's output and leaf gain and
    // its partner lane the right child's from the winner's candidate values: the same expressions on the same
    // inputs, issued once for both children instead of one after the other in one lane (G == 32: cuda_plan key
    // find_select_int, the winner is the lane owning best_t's slot)
    const uint32_t winner = STRIDED ? best_t % static_cast<uint32_t>(G) : best_t / static_cast<uint32_t>(PPL);
    const double right_g = __shfl_sync(mask, best.sr_g, winner, G);
    const double right_h = __shfl_sync(mask, best.sr_h, winner, G);
    const data_size_t right_c = __shfl_sync(mask, best.rc, winner, G);
    const bool is_winner = lane == winner;
    if (is_winner || lane == (winner ^ 1u)) {
      const double child_g = is_winner ? best.sl_g : right_g;
      const double child_h = is_winner ? best.sl_h : right_h;
      const data_size_t child_c = is_winner ? best.lc : right_c;
      const double child_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(child_g,
        child_h, lambda_l1, lambda_l2, path_smooth, max_delta_step, child_c, parent_output);
      const double child_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(child_g, child_h, lambda_l1,
                                                                             lambda_l2, child_output);
      if (is_winner) {
        out->is_valid = true;
        out->threshold = REVERSE ? static_cast<uint32_t>(task->num_bin - 2 - best_t) :
                                   static_cast<uint32_t>(best_t + task->mfb_offset - shift);
        out->gain = best_gain * task->penalty;
        out->default_left = task->assume_out_default_left;
        out->left_sum_gradients = best.sl_g;
        out->left_sum_hessians = best.sl_h;
        out->left_sum_of_gradients_hessians = best.slgh;
        out->left_count = best.lc;
        out->right_sum_gradients = best.sr_g;
        out->right_sum_hessians = best.sr_h;
        out->right_sum_of_gradients_hessians = best.srgh;
        out->right_count = best.rc;
        out->left_value = child_output;
        out->left_gain = child_gain;
      } else {
        out->right_value = child_output;
        out->right_gain = child_gain;
      }
    }
    return;
  }
  if (lane == best_t / static_cast<uint32_t>(PPL)) {
    // this lane's local best is the global best: its candidate values are in `best`
    out->is_valid = true;
    out->threshold = REVERSE ? static_cast<uint32_t>(task->num_bin - 2 - best_t) :
                               static_cast<uint32_t>(best_t + task->mfb_offset - shift);
    out->gain = best_gain * task->penalty;
    out->default_left = task->assume_out_default_left;
    const double left_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(best.sl_g,
      best.sl_h, lambda_l1, lambda_l2, path_smooth, max_delta_step, best.lc, parent_output);
    const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(best.sr_g,
      best.sr_h, lambda_l1, lambda_l2, path_smooth, max_delta_step, best.rc, parent_output);
    out->left_sum_gradients = best.sl_g;
    out->left_sum_hessians = best.sl_h;
    out->left_sum_of_gradients_hessians = best.slgh;
    out->left_count = best.lc;
    out->right_sum_gradients = best.sr_g;
    out->right_sum_hessians = best.sr_h;
    out->right_sum_of_gradients_hessians = best.srgh;
    out->right_count = best.rc;
    out->left_value = left_output;
    out->left_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(best.sl_g, best.sl_h, lambda_l1, lambda_l2,
                                                                  left_output);
    out->right_value = right_output;
    out->right_gain = CUDALeafSplits::GetLeafGainGivenOutput<false>(best.sr_g, best.sr_h, lambda_l1, lambda_l2,
                                                                   right_output);
  }
}

// G = 32: one warp per (task, leaf); G = 4: one 4-lane group per (task, leaf), every task at most 8 positions;
// G = 8 (warp_find_narrow without warp_find_spread): one 8-lane group per (task, leaf), every task at most 8 bins
// BATCHED (cuda_plan key find_loads_batched): the item's loads go out in dependency levels -- the pair descriptor,
// the used-task index and both scales together, then the task and the leaf's sums together (the leaf's only when
// it is valid), then the feature-used flag -- instead of one round trip per load behind each early exit (leaf
// validity, then the task index, the task, the flag, the scales, the leaf). Same values, same exits and writes.
template <int G, bool STRIDED = false, bool BATCHED = false>
__global__ void __launch_bounds__(128) FindBestSplitsDiscretizedForLevelWarpKernel(
  const int8_t* is_feature_used_bytree,
  const int num_tasks,
  const int num_used_tasks,
  const SplitFindTask* tasks,
  const int* used_task_indices,
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l2_in,
  const double max_delta_step,
  const score_t* grad_scale,
  const score_t* hess_scale,
  const bool quant_bagging_ridge,
  const bool spread,
  const bool narrow_bins,
  const bool mid_ppl,
  const bool prune_fp32,
  const bool select_int,
  CUDASplitInfo* cuda_best_split_info) {
  const unsigned int item = blockIdx.x * (128u / G) + threadIdx.x / G;
  const unsigned int nt = static_cast<unsigned int>(num_used_tasks);
  if (item >= nt * static_cast<unsigned int>(num_pairs) * 2u) return;
  const unsigned int slot = item % nt;
  const unsigned int rest = item / nt;
  const unsigned int pair_index = rest >> 1;
  const bool is_larger = (rest & 1u) != 0;
  const double lambda_l2 = quant_bagging_ridge ? lambda_l2_in + static_cast<double>(*hess_scale) : lambda_l2_in;
  const CUDAHybridPairDescriptor* desc = pair_descs + pair_index;
  const CUDALeafSplitsStruct* leaf_splits;
  data_size_t num_data;
  unsigned int task_index;
  const SplitFindTask* task;
  CUDASplitInfo* out;
  double parent_gain;
  int64_t sum_gradients_hessians;
  double parent_output;
  uint8_t leaf_num_bits;
  double gscale;
  double hscale;
  const hist_t* hist_in_leaf;
  if (BATCHED) {
    // level 1: the descriptor, the used-task index (slot < num_used_tasks), both scales
    const bool leaf_valid = is_larger ? desc->larger_valid : desc->smaller_valid;
    leaf_splits = is_larger ? desc->larger_struct : desc->smaller_struct;
    num_data = is_larger ? desc->num_data_in_larger_leaf : desc->num_data_in_smaller_leaf;
    leaf_num_bits = is_larger ? desc->larger_num_bits : desc->smaller_num_bits;
    task_index = used_task_indices == nullptr ? slot : static_cast<unsigned int>(used_task_indices[slot]);
    gscale = *grad_scale;
    hscale = *hess_scale;
    // level 2: the task and, for a valid leaf only, its sums and histogram
    task = tasks + task_index;
    const int feature = task->inner_feature_index;
    parent_gain = 0.0;
    sum_gradients_hessians = 0;
    parent_output = 0.0;
    hist_in_leaf = nullptr;
    if (leaf_valid) {
      parent_gain = leaf_splits->gain;
      sum_gradients_hessians = leaf_splits->sum_of_gradients_hessians;
      parent_output = leaf_splits->leaf_value;
      hist_in_leaf = leaf_splits->hist_in_leaf;
    }
    if (!leaf_valid) {
      return;
    }
    // level 3: the feature-used flag
    const unsigned int output_offset = pair_index * (2 * static_cast<unsigned int>(num_tasks)) +
      (is_larger ? task_index + num_tasks : task_index);
    out = cuda_best_split_info + output_offset;
    if (!is_feature_used_bytree[feature]) {
      if ((threadIdx.x & (G - 1u)) == 0) out->is_valid = false;
      return;
    }
  } else {
    if (is_larger ? !desc->larger_valid : !desc->smaller_valid) {
      return;
    }
    leaf_splits = is_larger ? desc->larger_struct : desc->smaller_struct;
    num_data = is_larger ? desc->num_data_in_larger_leaf : desc->num_data_in_smaller_leaf;
    task_index = used_task_indices == nullptr ? slot : static_cast<unsigned int>(used_task_indices[slot]);
    task = tasks + task_index;
    const unsigned int output_offset = pair_index * (2 * static_cast<unsigned int>(num_tasks)) +
      (is_larger ? task_index + num_tasks : task_index);
    out = cuda_best_split_info + output_offset;
    if (!is_feature_used_bytree[task->inner_feature_index]) {
      if ((threadIdx.x & (G - 1u)) == 0) out->is_valid = false;
      return;
    }
    parent_gain = leaf_splits->gain;
    sum_gradients_hessians = leaf_splits->sum_of_gradients_hessians;
    parent_output = leaf_splits->leaf_value;
    leaf_num_bits = is_larger ? desc->larger_num_bits : desc->smaller_num_bits;
    gscale = *grad_scale;
    hscale = *hess_scale;
    hist_in_leaf = leaf_splits->hist_in_leaf;
  }
  // scan positions 0 .. fnb + shift - 1 (see WarpFindBest); at most 32 fit one per lane
  const uint32_t num_positions = static_cast<uint32_t>(task->num_bin - task->mfb_offset) +
    ((!task->reverse && task->na_as_missing && task->mfb_offset == 1) ? 1u : 0u);
  // warp_find_narrow: a task with at most 8 bins has at most 8 positions
  const bool one_per_lane = (spread && num_positions <= 32u) || (narrow_bins && task->num_bin <= 8);
  // warp_find_mid_ppl (G == 32 only; the G == 4 / G == 8 templates always force PPL to 2 / 1 below regardless of
  // which tier is picked here): a task does not need the full 8-slot unrolled loop PPL=8 reaches for every width
  // up to 256: one tier per 32 positions, PPL = ceil(num_positions / 32), at least 2 (num_bin <= 256 keeps it <= 8).
  const uint32_t mid_tier = (G == 32 && mid_ppl) ? (num_positions + 31u) / 32u : 8u;
#define FALCATA_WARP_FIND_PPL(REV, B16, ACC_T, PPL) \
  WarpFindBest<REV, B16, ACC_T, (G == 4 ? 2 : G == 8 ? 1 : PPL), G, STRIDED && G == 4>(reinterpret_cast<const ACC_T*>(hist_in_leaf) + task->hist_offset, \
    task, lambda_l2, max_delta_step, min_data_in_leaf, min_sum_hessian_in_leaf, min_gain_to_split, parent_gain, \
    sum_gradients_hessians, num_data, parent_output, gscale, hscale, prune_fp32, select_int, out)
#define FALCATA_WARP_FIND(REV, B16, ACC_T) \
  if (G < 32 || one_per_lane) { \
    FALCATA_WARP_FIND_PPL(REV, B16, ACC_T, 1); \
  } else if (mid_tier <= 2u) { \
    FALCATA_WARP_FIND_PPL(REV, B16, ACC_T, 2); \
  } else if (mid_tier == 3u) { \
    FALCATA_WARP_FIND_PPL(REV, B16, ACC_T, 3); \
  } else if (mid_tier == 4u) { \
    FALCATA_WARP_FIND_PPL(REV, B16, ACC_T, 4); \
  } else if (mid_tier == 5u) { \
    FALCATA_WARP_FIND_PPL(REV, B16, ACC_T, 5); \
  } else if (mid_tier == 6u) { \
    FALCATA_WARP_FIND_PPL(REV, B16, ACC_T, 6); \
  } else if (mid_tier == 7u) { \
    FALCATA_WARP_FIND_PPL(REV, B16, ACC_T, 7); \
  } else { \
    FALCATA_WARP_FIND_PPL(REV, B16, ACC_T, 8); \
  }
  if (leaf_num_bits <= 16) {
    if (task->reverse) {
      FALCATA_WARP_FIND(true, true, int32_t);
    } else {
      FALCATA_WARP_FIND(false, true, int32_t);
    }
  } else {
    if (task->reverse) {
      FALCATA_WARP_FIND(true, false, int64_t);
    } else {
      FALCATA_WARP_FIND(false, false, int64_t);
    }
  }
#undef FALCATA_WARP_FIND
#undef FALCATA_WARP_FIND_PPL
}
#endif  // !defined(__HIP_PLATFORM_AMD__)

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
  const bool use_warp = warp_find_eligible_ && gstate == nullptr && !FalcataFP32GainEnabled();
  if (FalcataDebug().diag && num_pairs == 1) {
    // single-pair levels: each tree logs at least its root
    Log::Info("split finder: %s kernel for this tree's quantized levels", use_warp ? "warp" : "block");
  }
#if !defined(__HIP_PLATFORM_AMD__)
  if (use_warp) {
    const int num_launch_tasks = compact_tasks ? num_used_tasks_ : num_tasks_;
    const uint64_t items = static_cast<uint64_t>(num_launch_tasks) * static_cast<uint64_t>(num_pairs) * 2;
    // warp_find_spread with every task at most 8 scan positions: eight (task, leaf) items per warp; else
    // warp_find_narrow with every task at most 8 bins: four
    const bool spread = FalcataPlan::Get().warp_find_spread;
    const bool narrow = FalcataPlan::Get().warp_find_narrow;
    const bool mid_ppl = FalcataPlan::Get().warp_find_mid_ppl;
    const bool prune_fp32 = FalcataPlan::Get().find_prune_fp32;
    const bool select_int = FalcataPlan::Get().find_select_int;
    #define FindBestSplitsDiscretizedForLevelWarpKernel_ARGS \
      cuda_is_feature_used_bytree_.RawData(), num_tasks_, num_launch_tasks, cuda_split_find_tasks_.RawData(), \
      compact_tasks ? cuda_used_task_indices_.RawDataReadOnly() : nullptr, pair_descs, num_pairs, min_data_in_leaf_, \
      min_sum_hessian_in_leaf_, min_gain_to_split_, lambda_l2_, max_delta_step_, grad_scale, hess_scale, \
      quant_bagging_ridge_, spread, narrow, mid_ppl, prune_fp32, select_int, cuda_best_split_info_.RawData()
    if (spread && warp_find_max_positions_ <= 8 && FalcataPlan::Get().warp_find_strided &&
        FalcataPlan::Get().find_loads_batched) {
      // cuda_plan keys warp_find_strided (the 4-lane groups' positions strided over the lanes) and
      // find_loads_batched (the item's loads in dependency levels)
      FindBestSplitsDiscretizedForLevelWarpKernel<4, true, true><<<static_cast<unsigned int>((items + 31) / 32), 128,
                                                                   0, cuda_streams_[0]>>>(
        FindBestSplitsDiscretizedForLevelWarpKernel_ARGS);
    } else if (spread && warp_find_max_positions_ <= 8 && FalcataPlan::Get().warp_find_strided) {
      // cuda_plan key warp_find_strided: the 4-lane groups' positions strided over the lanes
      FindBestSplitsDiscretizedForLevelWarpKernel<4, true><<<static_cast<unsigned int>((items + 31) / 32), 128, 0,
                                                             cuda_streams_[0]>>>(
        FindBestSplitsDiscretizedForLevelWarpKernel_ARGS);
    } else if (spread && warp_find_max_positions_ <= 8) {
      FindBestSplitsDiscretizedForLevelWarpKernel<4><<<static_cast<unsigned int>((items + 31) / 32), 128, 0,
                                                       cuda_streams_[0]>>>(
        FindBestSplitsDiscretizedForLevelWarpKernel_ARGS);
    } else if (narrow && warp_find_all_narrow_) {
      FindBestSplitsDiscretizedForLevelWarpKernel<8><<<static_cast<unsigned int>((items + 15) / 16), 128, 0,
                                                       cuda_streams_[0]>>>(
        FindBestSplitsDiscretizedForLevelWarpKernel_ARGS);
    } else {
      FindBestSplitsDiscretizedForLevelWarpKernel<32><<<static_cast<unsigned int>((items + 3) / 4), 128, 0,
                                                        cuda_streams_[0]>>>(
        FindBestSplitsDiscretizedForLevelWarpKernel_ARGS);
    }
    #undef FindBestSplitsDiscretizedForLevelWarpKernel_ARGS
    return;
  }
#endif
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
      max_delta_step_, \
      cat_smooth_, \
      cat_l2_, \
      max_cat_threshold_, \
      min_data_per_group_, \
      grad_scale, \
      hess_scale, \
      quant_bagging_ridge_, \
      cuda_best_split_info_.RawData(), \
      gstate
  if (FalcataFP32GainEnabled()) {
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

#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
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
    /*gate_on_desc_counts=*/false,  // graph flow is single-GPU only
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
      /*gate_on_desc_counts=*/false,  // graph flow is single-GPU only
      gstate);
    if (!AppendCapturedNode(cuda_streams_[0], nodes)) return;
    roles->push_back(kHybridGraphNodeSyncAllBlocks);
    role_static_x->push_back(0);
  }
}
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED

void CUDABestSplitFinder::LaunchSyncBestSplitForLevelKernel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const bool gate_on_desc_counts) {
  const int num_blocks_per_leaf = (num_tasks_ + NUM_TASKS_PER_SYNC_BLOCK - 1) / NUM_TASKS_PER_SYNC_BLOCK;
  const bool compact_tasks = num_used_tasks_ < num_tasks_;
  // sync_used_tasks: a feature-sampled tree (the used-task list is on the device) or a task list wider than one
  // sync block takes one block per leaf over the used tasks only, with no merge kernel
  const bool used_list = compact_tasks && num_used_tasks_ > 0;
  if (FalcataPlan::Get().sync_used_tasks && (used_list || (!compact_tasks && num_blocks_per_leaf > 1))) {
    dim3 used_grid_dim(2, num_pairs);
    // a few hundred used tasks per leaf: 256 threads fold about two each (measured 10 us vs 14 us at 1024)
    constexpr int kSyncUsedTasksThreads = 256;
    #define SyncBestSplitForLevelUsedTasksKernel_ARGS \
      pair_descs, cuda_leaf_best_split_info_.RawData(), cuda_split_find_tasks_.RawData(), \
      compact_tasks ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr, \
      used_list ? cuda_used_task_indices_.RawDataReadOnly() : nullptr, used_list ? num_used_tasks_ : num_tasks_, \
      cuda_best_split_info_.RawData(), num_tasks_, min_data_in_leaf_, min_sum_hessian_in_leaf_, gate_on_desc_counts
#if !defined(__HIP_PLATFORM_AMD__)
    const bool sync_copy_batched = FalcataPlan::Get().sync_copy_batched;
#else
    const bool sync_copy_batched = false;  // the integer-key reduction assumes 32-lane warps
#endif
    if (sync_copy_batched) {
      SyncBestSplitForLevelUsedTasksKernel<true><<<used_grid_dim, kSyncUsedTasksThreads, 0, cuda_streams_[0]>>>(
        SyncBestSplitForLevelUsedTasksKernel_ARGS);
    } else {
      SyncBestSplitForLevelUsedTasksKernel<false><<<used_grid_dim, kSyncUsedTasksThreads, 0, cuda_streams_[0]>>>(
        SyncBestSplitForLevelUsedTasksKernel_ARGS);
    }
    #undef SyncBestSplitForLevelUsedTasksKernel_ARGS
    return;
  }
  dim3 grid_dim(2, num_pairs, num_blocks_per_leaf);
#if !defined(__HIP_PLATFORM_AMD__)
  const bool sync_copy_batched = FalcataPlan::Get().sync_copy_batched;
#else
  const bool sync_copy_batched = false;  // the integer-key reduction assumes 32-lane warps
#endif
  if (sync_copy_batched) {
    SyncBestSplitForLevelKernel<true><<<grid_dim, NUM_TASKS_PER_SYNC_BLOCK, 0, cuda_streams_[0]>>>(
      pair_descs, cuda_leaf_best_split_info_.RawData(), cuda_split_find_tasks_.RawData(),
      compact_tasks ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr, cuda_best_split_info_.RawData(),
      num_tasks_, num_leaves_, min_data_in_leaf_, min_sum_hessian_in_leaf_, gate_on_desc_counts, nullptr);
  } else {
    SyncBestSplitForLevelKernel<<<grid_dim, NUM_TASKS_PER_SYNC_BLOCK, 0, cuda_streams_[0]>>>(
      pair_descs, cuda_leaf_best_split_info_.RawData(), cuda_split_find_tasks_.RawData(),
      compact_tasks ? cuda_is_feature_used_bytree_.RawDataReadOnly() : nullptr, cuda_best_split_info_.RawData(),
      num_tasks_, num_leaves_, min_data_in_leaf_, min_sum_hessian_in_leaf_, gate_on_desc_counts, nullptr);
  }
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
      gate_on_desc_counts,
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
  const double max_delta_step,
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
      lambda_l1, lambda_l2, path_smooth, max_delta_step, left_count, parent_output) +
    CUDALeafSplits::GetLeafGain<true, false>(sum_right_gradient, sum_right_hessian,
      lambda_l1, lambda_l2, path_smooth, max_delta_step, right_count, parent_output);
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
      sum_left_gradient, sum_left_hessian, lambda_l1, lambda_l2, path_smooth, max_delta_step,
      left_count, parent_output);
  const double right_output = CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(
      sum_right_gradient, sum_right_hessian, lambda_l1, lambda_l2, path_smooth, max_delta_step,
      right_count, parent_output);
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
    lambda_l1_, lambda_l2_, path_smooth_, max_delta_step_, min_gain_to_split_, out);
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

  const bool is_smaller = (blockIdx_x < static_cast<unsigned int>(num_blocks_per_leaf) && !larger_only);
  const uint32_t leaf_block_index = (is_smaller || larger_only) ? blockIdx_x : (blockIdx_x - static_cast<unsigned int>(num_blocks_per_leaf));
  const int task_index = static_cast<int>(leaf_block_index * blockDim.x + threadIdx_x);
  const uint32_t read_index = is_smaller ? static_cast<uint32_t>(task_index) : static_cast<uint32_t>(task_index + num_tasks);
  // Same stale-slot hazard as SyncBestSplitForLevelKernel (see the comment
  // there): the nothing-found default used to be plain 0 == (smaller role,
  // task 0). When this kernel syncs the LARGER leaf and its search found no
  // valid split, a stale-valid smaller-role task-0 slot from an earlier
  // search was resurrected as the larger leaf's best split. Default to the
  // lane's OWN role and block instead: those slots are always freshly
  // written when this role was searched, and this kernel is only launched
  // over searched roles.
  const int safe_task_index = task_index < num_tasks ? task_index : num_tasks - 1;
  uint32_t shared_read_index = is_smaller ? static_cast<uint32_t>(safe_task_index)
                                          : static_cast<uint32_t>(safe_task_index + num_tasks);
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

__global__ void InvalidateLeafCandidatesKernel(
  const int* leaves, const int num, CUDASplitInfo* cuda_leaf_best_split_info) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < num) {
    cuda_leaf_best_split_info[leaves[i]].is_valid = false;
    cuda_leaf_best_split_info[leaves[i]].gain = kMinScore;
  }
}

void CUDABestSplitFinder::InvalidateLeafCandidates(const std::vector<int>& leaves) {
  if (leaves.empty()) {
    return;
  }
  if (cuda_invalidate_leaves_.Size() < leaves.size()) {
    cuda_invalidate_leaves_.Resize(leaves.size());
  }
  // cuda_plan key invalidate_async: no host wait (see the key); the upload and the kernel stay on the default stream
  const bool async = FalcataPlan::Get().invalidate_async && AsyncTreeStart();
  if (async) {
    // the caller's list is local to it, and a pageable source is promised to be staged on return only by the
    // synchronous cudaMemcpy: the copy reads a member instead, next rewritten at the next tree's final level, after
    // Train()'s tree-end device sync
    invalidate_leaves_host_.assign(leaves.begin(), leaves.end());
    CopyFromHostToCUDADeviceAsync<int>(cuda_invalidate_leaves_.RawData(), invalidate_leaves_host_.data(),
                                       invalidate_leaves_host_.size(), 0, __FILE__, __LINE__);
  } else {
    CopyFromHostToCUDADevice<int>(cuda_invalidate_leaves_.RawData(), leaves.data(),
                                  leaves.size(), __FILE__, __LINE__);
  }
  const int num = static_cast<int>(leaves.size());
  InvalidateLeafCandidatesKernel<<<(num + 255) / 256, 256>>>(
    cuda_invalidate_leaves_.RawDataReadOnly(), num, cuda_leaf_best_split_info_.RawData());
  if (async) {
    CUDASUCCESS_OR_FATAL(cudaGetLastError());
  } else {
    SynchronizeCUDADevice(__FILE__, __LINE__);
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
    // leaf_best_gain > 0.0 mirrors the CPU stop condition (serial_tree_learner.cpp:
    // "if (best_leaf_SplitInfo.gain <= 0.0) break"). Without CEGB this is a no-op
    // (valid splits always have positive gain); with CEGB the cost penalty can push
    // gains negative and such splits must not be taken.
    if (cuda_leaf_best_split_info[leaf_index].is_valid && leaf_best_gain > 0.0 && leaf_best_gain > thread_best_gain) {
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
  AllocatePinnedMappedHost(reinterpret_cast<void**>(&pinned_leaf_best_split_info_),
    pinned_leaf_best_split_info_size_ * sizeof(CUDASplitInfo), __FILE__, __LINE__);
  pinned_leaf_best_split_info_device_ =
    static_cast<CUDASplitInfo*>(MappedHostDevicePointer(pinned_leaf_best_split_info_));
}

void CUDABestSplitFinder::ReadPrefetchedLeafBestSplits(const int num_leaves, std::vector<CUDASplitInfo>* out) const {
  out->resize(static_cast<size_t>(num_leaves));
  std::memcpy(reinterpret_cast<void*>(out->data()), pinned_leaf_best_split_info_,
    static_cast<size_t>(num_leaves) * sizeof(CUDASplitInfo));
  // the raw copy brings over device categorical-threshold pointers; scrub the
  // POINTERS so the host-side destructor never frees device memory, but KEEP
  // num_cat_threshold: the hybrid categorical apply uses the
  // count to route splits, and both ~CUDASplitInfo and operator= are
  // null-pointer-safe with a positive count
  for (CUDASplitInfo& info : *out) {
    info.cat_threshold = nullptr;
    info.cat_threshold_real = nullptr;
    info.vec_payload = nullptr;
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
  if (FalcataPlan::Get().readback_kernel && pinned_leaf_best_split_info_device_ != nullptr) {
    // cuda_plan key readback_kernel: a copy kernel on the same default stream, then that stream's sync (see the key)
    CopyFromCUDADeviceToMappedHost(pinned_leaf_best_split_info_device_, pinned_leaf_best_split_info_,
                                   cuda_leaf_best_split_info_.RawDataReadOnly(),
                                   static_cast<size_t>(num_leaves) * sizeof(CUDASplitInfo), __FILE__, __LINE__);
  } else {
    CopyFromCUDADeviceToHost<CUDASplitInfo>(pinned_leaf_best_split_info_, cuda_leaf_best_split_info_.RawDataReadOnly(),
      static_cast<size_t>(num_leaves), __FILE__, __LINE__);
  }
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
    // the vector-leaf payload fields must never be garbage: the sync kernels'
    // wholesale struct copies read them in every mode. Vector mode reassigns
    // real slab pointers afterwards (AssignVecPayloadKernel).
    cuda_split_infos[i].vec_payload = nullptr;
    cuda_split_infos[i].num_vec_targets = 0;
  }
}

void CUDABestSplitFinder::LaunchAllocateCatVectorsKernel(
  CUDASplitInfo* cuda_split_infos, uint32_t* cat_threshold_vec, int* cat_threshold_real_vec, size_t len) {
  // A dataset whose features are all constant has no split-find tasks, so len
  // is 0 and the grid would be too. CUDA rejects a 0-block launch outright,
  // and the resulting error is sticky: it surfaces at whatever unrelated call
  // reads it next, which makes it look like that caller's bug.
  if (len == 0) {
    return;
  }
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
  // no split-find tasks means no per-task RNG state; see
  // LaunchAllocateCatVectorsKernel for why a 0-block launch must not happen
  if (cuda_randoms_.Size() == 0) {
    return;
  }
  const int num_blocks = (static_cast<int>(cuda_randoms_.Size()) +
    NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER - 1) / NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER;
  InitCUDARandomKernel<<<num_blocks, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER>>>(extra_seed_,
    static_cast<int>(cuda_randoms_.Size()), cuda_randoms_.RawData());
}

}  // namespace Falcata

#endif  // USE_CUDA
