/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_TREELEARNER_CUDA_CUDA_BEST_SPLIT_FINDER_KERNELS_CUH_
#define FALCATA_TREELEARNER_CUDA_CUDA_BEST_SPLIT_FINDER_KERNELS_CUH_

#ifdef USE_CUDA

#include "cuda_best_split_finder.hpp"
#include "cuda_device_levels.hpp"
#include "cuda_device_levels.cuh"

#include <Falcata/cuda/cuda_rocm_interop.h>

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <type_traits>
#include <vector>

#include <Falcata/cuda/cuda_algorithms.hpp>

#include "cuda_pdl_launch.hpp"

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

// Two CPU-order scans in one pass, one chain per lane: lane 0 folds a and
// lane 1 folds b, so every fold step is ONE warp instruction serving both
// chains. A reduced-rate fp64 pipe (1/64 of fp32 on consumer parts) charges a
// warp instruction its full issue cost however few lanes are active, so the
// pair costs what one chain does. The staging interleaves the chains (a at
// [2i], b at [2i + 1]) so the two lanes' accesses of a step fall in distinct
// banks. len bounds the fold at the lanes the caller actually reads (trailing
// lanes then keep their own deposit, which the callers' bin-range guards
// never read); row_buffer holds 2 * blockDim.x elements.
template <typename T>
__device__ __forceinline__ void SequentialPrefixSumPair(
    T* value_a, T* value_b, T* row_buffer, const unsigned int len) {
  __syncthreads();  // previous users of row_buffer may still be reading
  row_buffer[2 * threadIdx.x] = *value_a;
  row_buffer[2 * threadIdx.x + 1] = *value_b;
  __syncthreads();
  if (threadIdx.x < 2) {
    T* chain = row_buffer + threadIdx.x;
    T acc = chain[0];
    const unsigned int bound = len < blockDim.x ? len : blockDim.x;
    #pragma unroll 4
    for (unsigned int i = 1; i < bound; ++i) {
      acc += chain[2 * i];
      chain[2 * i] = acc;
    }
  }
  __syncthreads();
  *value_a = row_buffer[2 * threadIdx.x];
  *value_b = row_buffer[2 * threadIdx.x + 1];
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

// ReduceBestGain for the numerical threshold scan. A found gain there is
// current_gain - min_gain_shift with current_gain > min_gain_shift, so it is
// positive. When every found gain of the block is also finite, the fp64
// reduction's zero tie tolerance makes its result the exact maximum of (found,
// gain, then lower thread index), and positive finite doubles order as their
// bit patterns: the fp64 specialization reduces 64-bit integer keys of the same
// order instead of issuing fp64 compares, fmax and the tolerance multiply at
// every shuffle step (fp64 pipe work on parts with 1/64-rate fp64). A block
// holding a non-finite found gain keeps the fp64 reduction and its NaN-tolerance
// semantics. The fp32 gain mode keeps its tolerance band through ReduceBestGain.
// Like ReduceBestGain, the result is valid in thread 0; the caller keeps the
// leading barrier.
template <typename GAIN_T>
__device__ __forceinline__ uint32_t ReduceBestThresholdGain(GAIN_T gain, bool found, uint32_t thread_index,
    GAIN_T* shared_gain_buffer, bool* shared_found_buffer, uint32_t* shared_thread_index_buffer) {
  return ReduceBestGain(gain, found, thread_index, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
}

template <>
__device__ __forceinline__ uint32_t ReduceBestThresholdGain<double>(double gain, bool found, uint32_t thread_index,
    double* shared_gain_buffer, bool* shared_found_buffer, uint32_t* shared_thread_index_buffer) {
  constexpr uint64_t kExponentMask = 0x7ff0000000000000ULL;
  const uint64_t bits = static_cast<uint64_t>(__double_as_longlong(gain));
  if (__syncthreads_or(found && (bits & kExponentMask) == kExponentMask)) {
    return ReduceBestGain(gain, found, thread_index, shared_gain_buffer, shared_found_buffer,
                          shared_thread_index_buffer);
  }
  // not found -> key 0, below every positive gain's bit pattern
  uint64_t key = found ? bits : 0ULL;
  uint32_t index = thread_index;
  const uint32_t mask = 0xffffffff;
  for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    const uint64_t other_key = __shfl_down_sync(mask, key, offset);
    const uint32_t other_index = __shfl_down_sync(mask, index, offset);
    if (other_key > key || (other_key == key && other_index < index)) {
      key = other_key;
      index = other_index;
    }
  }
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t warpLane = threadIdx.x % warpSize;
  uint64_t* shared_key_buffer = reinterpret_cast<uint64_t*>(shared_gain_buffer);
  if (warpLane == 0) {
    shared_key_buffer[warpID] = key;
    shared_thread_index_buffer[warpID] = index;
  }
  __syncthreads();
  if (warpID == 0) {
    const uint32_t num_warp = blockDim.x / warpSize;
    key = warpLane < num_warp ? shared_key_buffer[warpLane] : 0ULL;
    index = warpLane < num_warp ? shared_thread_index_buffer[warpLane] : 0;
    for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
      const uint64_t other_key = __shfl_down_sync(mask, key, offset);
      const uint32_t other_index = __shfl_down_sync(mask, index, offset);
      if (other_key > key || (other_key == key && other_index < index)) {
        key = other_key;
        index = other_index;
      }
    }
  }
  return index;
}

static __device__ void ReduceBestGainForLeaves(double* gain, int* leaves, int cuda_cur_num_leaves) {
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

static __device__ void ReduceBestGainForLeavesWarp(double gain, int leaf_index, double* out_gain, int* out_leaf_index) {
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

static __device__ int ReduceBestGainForLeavesBlock(double gain, int leaf_index) {
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

static __device__ int ReduceBestGainForLeaves(double gain, int leaf_index, double* shared_gain_buffer, int* shared_leaf_index_buffer) {
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

// fp32 bounds of the fp64 split gain sl_g^2 / (sl_h + l2) + sr_g^2 / (sr_h + l2)
// (GetSplitGains without L1, smoothing or max_delta_step): lo <= the fp64 gain
// <= hi. Every input is rounded to fp32 once and the gain is evaluated in fp32
// with IEEE operations; within the guarded range (each gradient sum zero or of
// magnitude in [2^-40, 2^40], each hessian sum and l2 + hessian sum in
// [2^-40, 2^40]) every intermediate is a normal fp32 value, so the fp32 gain is
// within ~9 ulp (relative 2^-20.8) of the exact one and the fp64 gain within
// 2^-50 of it; the bounds take a 2^-16 relative margin. Outside the range the
// bounds stay (-inf, inf): no prune.
__device__ __forceinline__ bool GainBoundOperandF32(const double g, float* g32) {
  const uint64_t magnitude = static_cast<uint64_t>(__double_as_longlong(g)) & 0x7fffffffffffffffULL;
  *g32 = __double2float_rn(g);
  const float a = fabsf(*g32);
  return magnitude == 0 || (a >= 0x1p-40f && a <= 0x1p40f);
}

__device__ __forceinline__ bool GainBoundDenominatorF32(const double h, const float l2, float* d32) {
  const float h32 = __double2float_rn(h);
  *d32 = __fadd_rn(h32, l2);
  return h32 >= 0x1p-40f && h32 <= 0x1p40f && *d32 <= 0x1p40f;
}

__device__ __forceinline__ void SplitGainBoundsF32(
    const double sl_g, const double sl_h, const double sr_g, const double sr_h, const double l2,
    float* lo, float* hi) {
  constexpr float kRelMargin = 0x1p-16f;
  const float l2_32 = __double2float_rn(l2);
  float gl, gr, dl, dr;
  if (!(l2 >= 0.0) || l2_32 > 0x1p40f ||
      !GainBoundOperandF32(sl_g, &gl) || !GainBoundOperandF32(sr_g, &gr) ||
      !GainBoundDenominatorF32(sl_h, l2_32, &dl) || !GainBoundDenominatorF32(sr_h, l2_32, &dr)) {
    return;
  }
  const float gain = __fadd_rn(__fdiv_rn(__fmul_rn(gl, gl), dl), __fdiv_rn(__fmul_rn(gr, gr), dr));
  const float margin = __fmul_rn(gain, kRelMargin);
  *lo = __fsub_rn(gain, margin);
  *hi = __fadd_rn(gain, margin);
}

// Relative cut below the best lower bound: a pruned threshold's fp64 gain is
// below (1 - 2^-21) times the maximum's, which keeps the two distinct after
// both subtract min_gain_shift (each subtraction rounds by at most 2^-53 of
// the maximum).
constexpr float kSplitGainCutMargin = 0x1p-20f;

// Block-wide fp32 maximum, broadcast to every thread; block-uniform call.
__device__ __forceinline__ float BlockMaxF32(float value, float* shared_buffer) {
  const uint32_t mask = 0xffffffff;
  for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, offset));
  }
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t warpLane = threadIdx.x % warpSize;
  __syncthreads();  // previous readers of shared_buffer
  if (warpLane == 0) {
    shared_buffer[warpID] = value;
  }
  __syncthreads();
  const uint32_t num_warp = blockDim.x / warpSize;
  value = warpLane < num_warp ? shared_buffer[warpLane] : -INFINITY;
  for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, offset));
  }
  return value;
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
  __shared__ float shared_prune_buffer[WARPSIZE];
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
      // one chain per lane, interleaved staging (see SequentialPrefixSumPair):
      // lane 0 folds the gradient, lane 1 the hessian
      seq_prefix_buffer[2 * threadIdx_x] = local_grad_hist;
      seq_prefix_buffer[2 * threadIdx_x + 1] = local_hess_hist;
      __syncthreads();
      if (threadIdx_x < 2) {
        GAIN_T acc = threadIdx_x == 0 ?
          static_cast<GAIN_T>(sum_gradients) :
          static_cast<GAIN_T>(sum_hessians) - static_cast<GAIN_T>(kEpsilon);
        for (unsigned int i = 1; i < blockDim.x; ++i) {
          acc -= seq_prefix_buffer[2 * i + threadIdx_x];
        }
        const GAIN_T acc_h = __shfl_down_sync(0x3u, acc, 1);
        if (threadIdx_x == 0) {
          local_grad_hist = acc;
          local_hess_hist = acc_h;
        }
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
  // Each thread's threshold: its child sums and counts and whether it passes
  // the split gates (min_sum_hessian, min_data, the extra_trees draw).
  bool candidate = false;
  GAIN_T cand_sum_left_gradient = 0.0f;
  GAIN_T cand_sum_left_hessian = 0.0f;
  GAIN_T cand_sum_right_gradient = 0.0f;
  GAIN_T cand_sum_right_hessian = 0.0f;
  data_size_t cand_left_count = 0;
  data_size_t cand_right_count = 0;
  uint32_t candidate_threshold = 0;
  if (REVERSE) {
    if (threadIdx_x >= static_cast<unsigned int>(task->na_as_missing) && threadIdx_x <= task->num_bin - 2 && !skip_sum) {
      cand_sum_right_gradient = local_grad_hist;
      cand_sum_right_hessian = local_hess_hist;
      cand_right_count = sizeof(GAIN_T) == sizeof(double) ?
        local_cnt_prefix :
        static_cast<data_size_t>(CUDARoundInt(cand_sum_right_hessian * cnt_factor));
      cand_sum_left_gradient = sum_gradients_acc - cand_sum_right_gradient;
      cand_sum_left_hessian = sum_hessians_acc - cand_sum_right_hessian;
      cand_left_count = num_data - cand_right_count;
      candidate = cand_sum_left_hessian >= min_sum_hessian_acc && cand_left_count >= min_data_in_leaf &&
        cand_sum_right_hessian >= min_sum_hessian_acc && cand_right_count >= min_data_in_leaf &&
        (!USE_RAND || static_cast<int>(task->num_bin - 2 - threadIdx_x) == rand_threshold);
      candidate_threshold = static_cast<uint32_t>(task->num_bin - 2 - threadIdx_x);
    }
  } else {
    const uint32_t end = (task->na_as_missing && task->mfb_offset == 1) ? static_cast<uint32_t>(task->num_bin - 2) : feature_num_bin_minus_offset - 2;
    if (threadIdx_x <= end && !skip_sum) {
      cand_sum_left_gradient = local_grad_hist;
      cand_sum_left_hessian = local_hess_hist;
      cand_left_count = sizeof(GAIN_T) == sizeof(double) ?
        local_cnt_prefix :
        static_cast<data_size_t>(CUDARoundInt(cand_sum_left_hessian * cnt_factor));
      cand_sum_right_gradient = sum_gradients_acc - cand_sum_left_gradient;
      cand_sum_right_hessian = sum_hessians_acc - cand_sum_left_hessian;
      cand_right_count = num_data - cand_left_count;
      candidate = cand_sum_left_hessian >= min_sum_hessian_acc && cand_left_count >= min_data_in_leaf &&
        cand_sum_right_hessian >= min_sum_hessian_acc && cand_right_count >= min_data_in_leaf &&
        (!USE_RAND || static_cast<int>(threadIdx_x + task->mfb_offset) == rand_threshold);
      candidate_threshold = (task->na_as_missing && task->mfb_offset == 1) ?
        static_cast<uint32_t>(threadIdx_x) :
        static_cast<uint32_t>(threadIdx_x + task->mfb_offset);
    }
  }
  if (sizeof(GAIN_T) == sizeof(double) && !USE_RAND && !USE_MC && !USE_L1 && !USE_SMOOTHING &&
      !(max_delta_step > 0.0) && min_gain_shift >= static_cast<GAIN_T>(0)) {
    // The fp64 gain below costs dozens of fp64 instructions per threshold
    // (two divisions), issued by every warp that holds a candidate. fp32
    // bounds rule out the thresholds that cannot be the block's winner, so the
    // exact gain runs only for the rest (see SplitGainBoundsF32): a threshold
    // whose upper bound lies below the best lower bound by more than the cut's
    // margin has an exact gain strictly below the maximum's, also after both
    // subtract min_gain_shift, so it is neither the winner nor tied with it.
    // Same survivors' expressions, same winner: bit-identical.
    float bound_lo = -INFINITY;
    float bound_hi = INFINITY;
    if (candidate) {
      SplitGainBoundsF32(static_cast<double>(cand_sum_left_gradient), static_cast<double>(cand_sum_left_hessian),
                         static_cast<double>(cand_sum_right_gradient), static_cast<double>(cand_sum_right_hessian),
                         lambda_l2, &bound_lo, &bound_hi);
    }
    const float best_lo = BlockMaxF32(candidate ? bound_lo : -INFINITY, shared_prune_buffer);
    if (best_lo > 0.0f && bound_hi < best_lo - best_lo * kSplitGainCutMargin) {
      candidate = false;
    }
  }
  if (candidate) {
    GAIN_T current_gain = USE_MC ?
      static_cast<GAIN_T>(CUDALeafSplits::GetSplitGainsMC<USE_L1, USE_SMOOTHING>(
        cand_sum_left_gradient, cand_sum_left_hessian, cand_sum_right_gradient,
        cand_sum_right_hessian, lambda_l1_acc,
        lambda_l2_acc, path_smooth_acc, static_cast<double>(max_delta_step), cand_left_count, cand_right_count, parent_output_acc,
        leaf_constraint_min, leaf_constraint_max, monotone_constraint)) :
      CUDALeafSplits::GetSplitGains<USE_L1, USE_SMOOTHING, GAIN_T>(
        cand_sum_left_gradient, cand_sum_left_hessian, cand_sum_right_gradient,
        cand_sum_right_hessian, lambda_l1_acc,
        lambda_l2_acc, path_smooth_acc, static_cast<GAIN_T>(max_delta_step), cand_left_count, cand_right_count, parent_output_acc);
    // gain with split is worse than without split
    if (current_gain > min_gain_shift) {
      local_gain = current_gain - min_gain_shift;
      threshold_value = candidate_threshold;
      threshold_found = true;
    }
  }
  __syncthreads();
  const uint32_t result = ReduceBestThresholdGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_bool_buffer, shared_int_buffer);
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
  const double hessian_ridge_quanta,
  // output
  CUDASplitInfo* cuda_best_split_info,
  // global num data in leaf
  const data_size_t global_num_data_in_smaller_leaf,
  const data_size_t global_num_data_in_larger_leaf,
  // CEGB
  const double* cuda_task_cegb_penalty,
  const double cegb_tradeoff_times_penalty_split) {
  // Hessian-quantum ridge. A child's integer hessian sum is known only to O(1)
  // quanta: a row whose hessian is a fraction of one quantum contributes it as
  // an occasional whole quantum (fixedpoint error feedback, stochastic
  // rounding) or as nothing at all (round to nearest), so a child made of
  // such rows -- well-fit rows, and softmax/logistic rows saturated on the
  // wrong side, whose |g| is near 1 while h is near 0 -- can record a few
  // quanta of H against many quanta of true hessian mass. The gain argmax
  // harvests exactly those children, and with lambda_l2 == 0 their gains and
  // Newton outputs -G/H explode far beyond what the leaf's true gradient mass
  // supports; the overshoot saturates more rows and the run collapses. A
  // ridge of a few hessian quanta bounds 1/H by the hessian's resolution. The
  // tree learner sizes it (CUDASingleGPUTreeLearner::BeforeTrain): fixedpoint
  // training whose hessians vary per row, bagged training, or none -- a
  // constant hessian quantizes exactly and unbagged stochastic training keeps
  // the unridged math, so both stay bit-identical.
  const double lambda_l2 = hessian_ridge_quanta > 0.0 ?
    lambda_l2_in + hessian_ridge_quanta * static_cast<double>(*hess_scale) : lambda_l2_in;
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
    hessian_ridge_quanta_, \
    cuda_best_split_info_.RawData(), \
    global_num_data_in_smaller_leaf, \
    global_num_data_in_larger_leaf, \
    cegb_use_ ? cuda_task_cegb_penalty_.RawData() : nullptr, \
    cegb_tradeoff_times_penalty_split_

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
  // Per-node feature masks (feature_fraction_bynode < 1 or interaction
  // constraints): the same selection as the non-discretized path. Without
  // select_features_by_node_ both leaves read the per-tree mask, as before.
  const int8_t* is_feature_used_by_smaller_node = cuda_is_feature_used_bytree_.RawData();
  const int8_t* is_feature_used_by_larger_node = cuda_is_feature_used_bytree_.RawData();
  if (select_features_by_node_) {
    is_feature_used_by_smaller_node = is_feature_used_by_smaller_node_.RawData();
    is_feature_used_by_larger_node = is_feature_used_by_larger_node_.RawData();
  }
  if (!use_global_memory_) {
    // Order each leaf after the histogram it reads via its own stream (see the
    // non-discretized path for the rationale — both leaves wait on subtract_done,
    // which is recorded after the fix kernel completes the smaller histogram);
    // no device sync between the two leaves.
    if (is_smaller_leaf_valid) {
      CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[0], hist_subtract_done_events_[active_hist_pipeline_], 0));
      FindBestSplitsDiscretizedForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, false, GAIN_T>
        <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>
        (is_feature_used_by_smaller_node, FindBestSplitsDiscretizedForLeafKernel_ARGS);
    }
    if (is_larger_leaf_valid) {
      CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[1], hist_subtract_done_events_[active_hist_pipeline_], 0));
      FindBestSplitsDiscretizedForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, true, GAIN_T>
        <<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[1]>>>
        (is_feature_used_by_larger_node, FindBestSplitsDiscretizedForLeafKernel_ARGS);
    }
  } else {
    // TODO(shiyu1994)
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_TREELEARNER_CUDA_CUDA_BEST_SPLIT_FINDER_KERNELS_CUH_
