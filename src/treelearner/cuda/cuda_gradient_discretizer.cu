/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <Falcata/falcata_plan.h>
#include <Falcata/objective_function.h>

#include <algorithm>
#include <atomic>

#include <Falcata/cuda/cuda_algorithms.hpp>

#include "cuda_gradient_discretizer.hpp"
#include "cuda_pdl_launch.hpp"

namespace Falcata {

static_assert(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE == NUM_THREADS_PER_BLOCK_LEAF_SPLITS,
              "discretize_root_sums requires matching 1024-row chunks");
static_assert(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE == ObjectiveFunction::CUDAPrecomputedGradMinMax::kBlockSize,
              "grad_minmax_fuse requires matching 1024-row blocks");

// Prevent ptxas from contracting (v * scale) + residual into FMA when scale lives in a uniform register
// across unrolled rows, keeping DiscretizeGradientsKernel and DiscretizeGradientsChunk256Kernel
// instruction-for-instruction identical under both float and SCORE_T_USE_DOUBLE builds.
__device__ __forceinline__ score_t MulNoFMA(const score_t a, const score_t b) {
#ifdef SCORE_T_USE_DOUBLE
  return __dmul_rn(a, b);
#else
  return __fmul_rn(a, b);
#endif
}

static int CachedDeviceSMCount() {
  constexpr int kMaxDevices = 64;
  static std::atomic<int> sm_counts[kMaxDevices] = {};
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kMaxDevices) {
    return 128;
  }
  int count = sm_counts[device].load(std::memory_order_relaxed);
  if (count == 0) {
    int attr = 0;
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrMultiProcessorCount, device) == cudaSuccess && attr > 0) {
      count = attr;
    } else {
      count = 128;
    }
    sm_counts[device].store(count, std::memory_order_relaxed);
  }
  return count;
}

__global__ void ReduceMinMaxKernel(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians,
  score_t* grad_min_block_buffer,
  score_t* grad_max_block_buffer,
  score_t* hess_min_block_buffer,
  score_t* hess_max_block_buffer) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the kernel before it on the stream is complete and visible
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel of the chain may get resident
  __shared__ score_t shared_mem_buffer[WARPSIZE];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  score_t grad_max_val = kMinScore;
  score_t grad_min_val = kMaxScore;
  score_t hess_max_val = kMinScore;
  score_t hess_min_val = kMaxScore;
  if (index < num_data) {
    grad_max_val = input_gradients[index];
    grad_min_val = input_gradients[index];
    hess_max_val = input_hessians[index];
    hess_min_val = input_hessians[index];
  }
  grad_min_val = ShuffleReduceMin<score_t>(grad_min_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  grad_max_val = ShuffleReduceMax<score_t>(grad_max_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  hess_min_val = ShuffleReduceMin<score_t>(hess_min_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  hess_max_val = ShuffleReduceMax<score_t>(hess_max_val, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    grad_min_block_buffer[blockIdx.x] = grad_min_val;
    grad_max_block_buffer[blockIdx.x] = grad_max_val;
    hess_min_block_buffer[blockIdx.x] = hess_min_val;
    hess_max_block_buffer[blockIdx.x] = hess_max_val;
  }
}

#if !defined(__HIP_PLATFORM_AMD__)
// ReduceMinMaxKernel with one warp per chunk of CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE rows (cuda_plan key
// minmax_warp). The block version's thread 32 w + l holds row 32 w + l of the chunk (or the +-inf fill past
// num_data); ShuffleReduce{Min,Max} run every warp's shuffle tree over its 32 rows, then warp 0's tree over the
// 32 warp results. Here the warp runs warp w's trees on rows 32 w .. 32 w + 31 (same lanes, same shuffles, same
// operand order), lane w keeps their lane-0 results, and the warp then runs the cross-warp tree, so every
// partial is the same value, NaN and signed zero included. CUDA only.
// CONST_HESS (cuda_plan key const_hess_reads): every row's hessian equals row 0's (see
// CUDAGradientDiscretizer::SetHessiansConstant), so each valid row takes that value instead of loading its own. In
// a chunk of only valid rows (every chunk but a partial last one) the hessian trees would reduce copies of that one
// value, and fmaxf / fminf of equal non-NaN operands return that operand: such a warp runs only the gradient trees
// and stores that value as both hessian partials (a NaN hessian or the partial chunk runs every tree as before).
template <bool CONST_HESS>
__global__ void ReduceMinMaxChunkWarpKernel(
  const data_size_t num_data,
  const int num_chunks,
  const score_t* input_gradients,
  const score_t* input_hessians,
  score_t* grad_min_block_buffer,
  score_t* grad_max_block_buffer,
  score_t* hess_min_block_buffer,
  score_t* hess_max_block_buffer) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the kernel before it on the stream is complete and visible
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel of the chain may get resident
  constexpr int kRowsPerWarpTree = 32;
  static_assert(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE == kRowsPerWarpTree * kRowsPerWarpTree,
                "the chunk is 32 warps of 32 rows");
  const int chunk = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / WARPSIZE);
  if (chunk >= num_chunks) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x % WARPSIZE);
  const data_size_t chunk_start = static_cast<data_size_t>(chunk) * CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE;
  const score_t hess_row0 = CONST_HESS ? input_hessians[0] : 0.0f;
  if (CONST_HESS && chunk_start + CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE <= num_data && hess_row0 == hess_row0) {
    score_t grad_min_known = 0.0f;
    score_t grad_max_known = 0.0f;
#pragma unroll 4
    for (int w = 0; w < kRowsPerWarpTree; ++w) {
      const score_t grad = input_gradients[chunk_start + static_cast<data_size_t>(w * kRowsPerWarpTree + lane)];
      const score_t grad_min_val = __shfl_sync(0xffffffff, ShuffleReduceMinWarp<score_t>(grad, kRowsPerWarpTree), 0);
      const score_t grad_max_val = __shfl_sync(0xffffffff, ShuffleReduceMaxWarp<score_t>(grad, kRowsPerWarpTree), 0);
      if (lane == w) {
        grad_min_known = grad_min_val;
        grad_max_known = grad_max_val;
      }
    }
    grad_min_known = ShuffleReduceMinWarp<score_t>(grad_min_known, kRowsPerWarpTree);
    grad_max_known = ShuffleReduceMaxWarp<score_t>(grad_max_known, kRowsPerWarpTree);
    if (lane == 0) {
      grad_min_block_buffer[chunk] = grad_min_known;
      grad_max_block_buffer[chunk] = grad_max_known;
      hess_min_block_buffer[chunk] = hess_row0;
      hess_max_block_buffer[chunk] = hess_row0;
    }
    return;
  }
  score_t grad_min_warp = 0.0f;
  score_t grad_max_warp = 0.0f;
  score_t hess_min_warp = 0.0f;
  score_t hess_max_warp = 0.0f;
#pragma unroll 4
  for (int w = 0; w < kRowsPerWarpTree; ++w) {
    const data_size_t index = chunk_start + static_cast<data_size_t>(w * kRowsPerWarpTree + lane);
    score_t grad_max_val = kMinScore;
    score_t grad_min_val = kMaxScore;
    score_t hess_max_val = kMinScore;
    score_t hess_min_val = kMaxScore;
    if (index < num_data) {
      grad_max_val = input_gradients[index];
      grad_min_val = input_gradients[index];
      hess_max_val = CONST_HESS ? hess_row0 : input_hessians[index];
      hess_min_val = hess_max_val;
    }
    grad_min_val = __shfl_sync(0xffffffff, ShuffleReduceMinWarp<score_t>(grad_min_val, kRowsPerWarpTree), 0);
    grad_max_val = __shfl_sync(0xffffffff, ShuffleReduceMaxWarp<score_t>(grad_max_val, kRowsPerWarpTree), 0);
    hess_min_val = __shfl_sync(0xffffffff, ShuffleReduceMinWarp<score_t>(hess_min_val, kRowsPerWarpTree), 0);
    hess_max_val = __shfl_sync(0xffffffff, ShuffleReduceMaxWarp<score_t>(hess_max_val, kRowsPerWarpTree), 0);
    if (lane == w) {
      grad_min_warp = grad_min_val;
      grad_max_warp = grad_max_val;
      hess_min_warp = hess_min_val;
      hess_max_warp = hess_max_val;
    }
  }
  grad_min_warp = ShuffleReduceMinWarp<score_t>(grad_min_warp, kRowsPerWarpTree);
  grad_max_warp = ShuffleReduceMaxWarp<score_t>(grad_max_warp, kRowsPerWarpTree);
  hess_min_warp = ShuffleReduceMinWarp<score_t>(hess_min_warp, kRowsPerWarpTree);
  hess_max_warp = ShuffleReduceMaxWarp<score_t>(hess_max_warp, kRowsPerWarpTree);
  if (lane == 0) {
    grad_min_block_buffer[chunk] = grad_min_warp;
    grad_max_block_buffer[chunk] = grad_max_warp;
    hess_min_block_buffer[chunk] = hess_min_warp;
    hess_max_block_buffer[chunk] = hess_max_warp;
  }
}
#endif  // !defined(__HIP_PLATFORM_AMD__)

__global__ void ReduceBlockMinMaxKernel(
  const int num_blocks,
  const int grad_discretize_bins,
  const score_t* in_grad_min_block_buffer,
  const score_t* in_grad_max_block_buffer,
  const score_t* in_hess_min_block_buffer,
  const score_t* in_hess_max_block_buffer,
  score_t* out_grad_min_block_buffer,
  score_t* out_grad_max_block_buffer,
  score_t* out_hess_min_block_buffer,
  score_t* out_hess_max_block_buffer) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the kernel before it on the stream is complete and visible
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel of the chain may get resident
  __shared__ score_t shared_mem_buffer[WARPSIZE];
  score_t grad_max_val = kMinScore;
  score_t grad_min_val = kMaxScore;
  score_t hess_max_val = kMinScore;
  score_t hess_min_val = kMaxScore;
  for (int block_index = static_cast<int>(threadIdx.x); block_index < num_blocks; block_index += static_cast<int>(blockDim.x)) {
    grad_min_val = min(grad_min_val, in_grad_min_block_buffer[block_index]);
    grad_max_val = max(grad_max_val, in_grad_max_block_buffer[block_index]);
    hess_min_val = min(hess_min_val, in_hess_min_block_buffer[block_index]);
    hess_max_val = max(hess_max_val, in_hess_max_block_buffer[block_index]);
  }
  grad_min_val = ShuffleReduceMin<score_t>(grad_min_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  grad_max_val = ShuffleReduceMax<score_t>(grad_max_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  hess_min_val = ShuffleReduceMin<score_t>(hess_min_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  hess_max_val = ShuffleReduceMax<score_t>(hess_max_val, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    const score_t grad_abs_max = max(fabs(grad_min_val), fabs(grad_max_val));
    const score_t hess_abs_max = max(fabs(hess_min_val), fabs(hess_max_val));
    out_grad_min_block_buffer[0] = 1.0f / (grad_abs_max / (grad_discretize_bins / 2));
    out_grad_max_block_buffer[0] = (grad_abs_max / (grad_discretize_bins / 2));
    out_hess_min_block_buffer[0] = 1.0f / (hess_abs_max / (grad_discretize_bins));
    out_hess_max_block_buffer[0] = (hess_abs_max / (grad_discretize_bins));
  }
}

// -----------------------------------------------------------------------------
// Outlier-robust grad scale (fixed-point mode). On heavily-imbalanced data the
// rare-class gradients are extreme high-magnitude outliers separated from the
// low-magnitude bulk by a wide EMPTY magnitude gap; a global max|grad| scale then
// crushes the bulk to near-zero quant resolution (rounds to 0). We characterise
// the |grad| distribution with a cheap log-magnitude histogram (no sort), detect
// that gap (see ComputeRobustGradScaleKernel), and, only when a genuine gap is
// present, re-anchor the scale to the bulk so it gets the full quant range while
// the outliers saturate. Balanced (continuous, gap-free) distributions are left
// on the exact global-max scale -- bit-for-bit unchanged, zero regression.
//
// Bucket mapping (num_buckets log2-spaced buckets): for |g| in (0, grad_abs_max],
//   frac = |g| / grad_abs_max in (0, 1]
//   bucket = clamp( floor(-log2(frac) * kMagBucketsPerOctave), 0, num_buckets-1 )
// bucket 0 = the top octave [grad_abs_max/2, grad_abs_max]; larger bucket index
// = smaller magnitude. Resolution is concentrated near the top of the range where
// the outlier cluster and the gap live.
#define kMagBucketsPerOctave (128.0f)

// Per-block shared-memory histogram: threads accumulate into shared bins first
// (cheap intra-block atomics), then one merge atomicAdd per non-empty bin to
// global. This keeps the magnitude histogram cheap even on large data where a
// direct global-atomic pass would contend heavily on the few populated buckets.
// num_buckets (kNumMagBuckets) must fit in shared memory: 2048*4B = 8KB.
__global__ void ComputeGradMagHistogramKernel(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* grad_dequant_ptr,  // = grad_max_block_buffer_[0] = grad_abs_max / (bins/2)
  const int grad_discretize_bins,
  const int num_buckets,
  unsigned int* mag_hist) {
  extern __shared__ unsigned int shared_hist[];
  for (int b = static_cast<int>(threadIdx.x); b < num_buckets; b += static_cast<int>(blockDim.x)) {
    shared_hist[b] = 0u;
  }
  __syncthreads();
  // grad_max_block_buffer_[0] holds the dequant scale grad_abs_max/(bins/2); recover raw max.
  const score_t grad_abs_max = (*grad_dequant_ptr) * static_cast<score_t>(grad_discretize_bins / 2);
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (index < num_data && grad_abs_max > 0.0f) {
    const score_t g = fabs(input_gradients[index]);
    if (g > 0.0f) {
      const score_t frac = g / grad_abs_max;  // in (0, 1]
      // float math: -log2f of a positive finite float lies in about [-128, 149], so multiplying by
      // 128 is exact in float as it was in double, and the bucket is the one the former double
      // multiply and floor gave (checked on sm_120 for all 2^32 float bit patterns of frac)
      int bucket = static_cast<int>(floorf(-log2f(frac) * kMagBucketsPerOctave));
      if (bucket < 0) bucket = 0;
      if (bucket >= num_buckets) bucket = num_buckets - 1;
      atomicAdd(&shared_hist[bucket], 1u);
    }
  }
  __syncthreads();
  for (int b = static_cast<int>(threadIdx.x); b < num_buckets; b += static_cast<int>(blockDim.x)) {
    const unsigned int v = shared_hist[b];
    if (v != 0u) {
      atomicAdd(&mag_hist[b], v);
    }
  }
}

// cuda_plan key robust_scale_async: grid-stride shared-memory log-magnitude histogram with PDL chaining.
// Zeroing shared_hist runs before PdlWaitForProducer so it overlaps the tail of ReduceBlockMinMaxKernel;
// unsigned atomicAdd is commutative/associative, so every bucket count is bit-identical to
// ComputeGradMagHistogramKernel.
__global__ void ComputeGradMagHistogramAsyncKernel(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* grad_dequant_ptr,
  const int grad_discretize_bins,
  const int num_buckets,
  unsigned int* mag_hist) {
  extern __shared__ unsigned int shared_hist[];
  for (int b = static_cast<int>(threadIdx.x); b < num_buckets; b += static_cast<int>(blockDim.x)) {
    shared_hist[b] = 0u;
  }
  __syncthreads();
  PdlWaitForProducer();
  const score_t grad_abs_max = (*grad_dequant_ptr) * static_cast<score_t>(grad_discretize_bins / 2);
  if (grad_abs_max > 0.0f) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    const int64_t limit = static_cast<int64_t>(num_data);
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
                         static_cast<int64_t>(threadIdx.x);
         index < limit; index += stride) {
      const score_t g = fabs(input_gradients[index]);
      if (g > 0.0f) {
        const score_t frac = g / grad_abs_max;
        int bucket = static_cast<int>(floorf(-log2f(frac) * kMagBucketsPerOctave));
        if (bucket < 0) bucket = 0;
        if (bucket >= num_buckets) bucket = num_buckets - 1;
        atomicAdd(&shared_hist[bucket], 1u);
      }
    }
  }
  __syncthreads();
  PdlLaunchDependents();
  for (int b = static_cast<int>(threadIdx.x); b < num_buckets; b += static_cast<int>(blockDim.x)) {
    const unsigned int v = shared_hist[b];
    if (v != 0u) {
      atomicAdd(&mag_hist[b], v);
    }
  }
}

// Gap-gated robust grad scale. Walking the log-magnitude histogram from the top
// (largest |grad|) downward, a heavily-imbalanced gradient distribution shows a
// distinctive shape: a small cluster of high-magnitude outliers (the rare-class
// gradients), then a wide EMPTY magnitude gap, then the low-magnitude bulk. A
// well-behaved (balanced) distribution has no such gap -- it is continuous from
// the max down. We only intervene when a genuine gap is present, so balanced
// datasets (year/higgs/epsilon) keep the exact global-max scale (zero regression)
// while the imbalanced case (fraud) gets the bulk anchored for full resolution.
//
// Rule: scan down; accumulate the outlier count. Once we have seen at least one
// non-empty bucket, count consecutive EMPTY buckets. If that empty run reaches
// kMinGapBuckets (a real magnitude gap) AND the outliers so far are a small
// fraction (<= kMaxOutlierFraction), anchor the scale at the top edge of the bulk
// (the first non-empty bucket below the gap): its |grad| maps to bins/2, the
// outliers saturate (clamped in the discretize kernel). If no qualifying gap is
// found, leave the global-max scale untouched.
#define kMinGapBuckets (128)          // >= 1 octave of empty magnitude space
#define kMaxOutlierFraction (0.05)    // outliers must be a small minority

__global__ void ComputeRobustGradScaleKernel(
  const int num_buckets,
  const int grad_discretize_bins,
  const data_size_t num_data,
  const unsigned int* mag_hist,
  score_t* grad_min_block_buffer,   // written: 1 / robust_scale  (quant multiplier)
  score_t* grad_max_block_buffer) {  // read: grad_abs_max ; written: robust dequant scale
  if (threadIdx.x == 0) {
    // grad_max_block_buffer[0] holds the dequant scale grad_abs_max/(bins/2); recover raw max.
    const score_t grad_abs_max = grad_max_block_buffer[0] * static_cast<score_t>(grad_discretize_bins / 2);
    const unsigned int max_outliers =
      static_cast<unsigned int>(kMaxOutlierFraction * static_cast<double>(num_data));

    unsigned int outlier_cum = 0;
    bool seen_nonempty = false;
    int gap_run = 0;
    int bulk_top_bucket = -1;  // first non-empty bucket below a qualifying gap
    for (int b = 0; b < num_buckets; ++b) {
      const unsigned int c = mag_hist[b];
      if (c == 0u) {
        if (seen_nonempty) {
          ++gap_run;
        }
      } else {
        if (seen_nonempty && gap_run >= kMinGapBuckets && outlier_cum <= max_outliers) {
          // Found the bulk just below a real gap.
          bulk_top_bucket = b;
          break;
        }
        seen_nonempty = true;
        gap_run = 0;
        outlier_cum += c;
        if (outlier_cum > max_outliers) {
          // Too much mass at high magnitude to call it an outlier cluster;
          // this is not the imbalanced case -- do not intervene.
          break;
        }
      }
    }

    if (bulk_top_bucket >= 0) {
      // Anchor at the UPPER magnitude edge of the bulk's top bucket so the bulk
      // gets the full quant range: grad_abs_max * 2^(-bulk_top_bucket/perOctave).
      const score_t anchor = grad_abs_max *
        exp2f(-static_cast<float>(bulk_top_bucket) / static_cast<float>(kMagBucketsPerOctave));
      if (anchor > 0.0f && anchor <= grad_abs_max) {
        const score_t dequant = anchor / (grad_discretize_bins / 2);
        grad_max_block_buffer[0] = dequant;
        grad_min_block_buffer[0] = 1.0f / dequant;
      }
    }
    // else: leave the global-max scale in place (balanced case, no regression).
  }
}

// cuda_plan key robust_scale_async: PDL-chained bucket scan that also zeroes mag_hist across the block
// right after thread 0 finishes scanning, keeping the next tree's histogram pass free of a cudaMemset.
__global__ void ComputeRobustGradScaleAsyncKernel(
  const int num_buckets,
  const int grad_discretize_bins,
  const data_size_t num_data,
  unsigned int* mag_hist,
  score_t* grad_min_block_buffer,
  score_t* grad_max_block_buffer) {
  PdlWaitForProducer();
  if (threadIdx.x == 0) {
    const score_t grad_abs_max = grad_max_block_buffer[0] * static_cast<score_t>(grad_discretize_bins / 2);
    const unsigned int max_outliers =
      static_cast<unsigned int>(kMaxOutlierFraction * static_cast<double>(num_data));

    unsigned int outlier_cum = 0;
    bool seen_nonempty = false;
    int gap_run = 0;
    int bulk_top_bucket = -1;
    for (int b = 0; b < num_buckets; ++b) {
      const unsigned int c = mag_hist[b];
      if (c == 0u) {
        if (seen_nonempty) {
          ++gap_run;
        }
      } else {
        if (seen_nonempty && gap_run >= kMinGapBuckets && outlier_cum <= max_outliers) {
          bulk_top_bucket = b;
          break;
        }
        seen_nonempty = true;
        gap_run = 0;
        outlier_cum += c;
        if (outlier_cum > max_outliers) {
          break;
        }
      }
    }

    if (bulk_top_bucket >= 0) {
      const score_t anchor = grad_abs_max *
        exp2f(-static_cast<float>(bulk_top_bucket) / static_cast<float>(kMagBucketsPerOctave));
      if (anchor > 0.0f && anchor <= grad_abs_max) {
        const score_t dequant = anchor / (grad_discretize_bins / 2);
        grad_max_block_buffer[0] = dequant;
        grad_min_block_buffer[0] = 1.0f / dequant;
      }
    }
  }
  __syncthreads();
  PdlLaunchDependents();
  for (int b = static_cast<int>(threadIdx.x); b < num_buckets; b += static_cast<int>(blockDim.x)) {
    mag_hist[b] = 0u;
  }
}


// ---------------------------------------------------------------------------
// Philox4x32-10 (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3",
// SC'11): counter-based RNG. The rounding noise for row r of tree t is a pure
// function of (seed, t, r) -- no tables, no memory traffic, and identical on
// any machine/thread-count/GPU by construction. (Idea borrowed from XGBoost
// 3.3's move to Philox sampling, dmlc/xgboost#12223.)
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint4 Philox4x32Round(const uint4 ctr, const uint2 key) {
  const uint32_t hi0 = __umulhi(0xD2511F53u, ctr.x);
  const uint32_t lo0 = 0xD2511F53u * ctr.x;
  const uint32_t hi1 = __umulhi(0xCD9E8D57u, ctr.z);
  const uint32_t lo1 = 0xCD9E8D57u * ctr.z;
  return make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}
__device__ __forceinline__ uint4 Philox4x32_10(uint4 ctr, uint2 key) {
  #pragma unroll
  for (int r = 0; r < 10; ++r) {
    ctr = Philox4x32Round(ctr, key);
    key.x += 0x9E3779B9u;
    key.y += 0xBB67AE85u;
  }
  return ctr;
}
// uniform in [0, 1): top 24 bits -> exact float
__device__ __forceinline__ float PhiloxUniform(const uint32_t x) {
  return static_cast<float>(x >> 8) * 5.9604644775390625e-08f;  // 2^-24
}

__global__ void BuildInbagMaskKernel(
  const data_size_t inbag_count,
  const data_size_t* inbag_indices,
  uint8_t* mask) {
  const data_size_t i = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (i < inbag_count) {
    mask[inbag_indices[i]] = 1;
  }
}

// COPY_HESS: vector-leaf plane t >= 1. The hessian is target-independent and
// every consumer reads it from plane 0, so the plane copies plane 0's already
// quantized hessian instead of re-rounding it under an independent dither.
// CONST_HESS (cuda_plan key const_hess_reads): every row's hessian equals row 0's,
// which each row takes instead of loading its own.
template <bool STOCHASTIC_ROUNDING, bool CLAMP, bool COPY_HESS = false, bool CONST_HESS = false, bool FUSE_ROOT_SUMS = false>
__global__ void DiscretizeGradientsKernel(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians,
  const score_t* grad_scale_ptr,
  const score_t* hess_scale_ptr,
  const int iter,
  const uint32_t philox_seed,
  const int grad_discretize_bins,
  int8_t* ef_residuals,
  const uint8_t* ef_inbag_mask,
  int8_t* output_gradients_and_hessians,
  const int8_t* plane0_gradients_and_hessians = nullptr,
  const score_t* grad_dequant_ptr = nullptr,
  const score_t* hess_dequant_ptr = nullptr,
  double* out_block_sum_grad = nullptr,
  double* out_block_sum_hess = nullptr,
  int64_t* out_block_sum_grad_hess = nullptr,
  score_t* plane_grad_scale = nullptr,
  score_t* plane_hess_scale = nullptr) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the kernel before it on the stream is complete and visible
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel of the chain may get resident
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const score_t grad_scale = *grad_scale_ptr;
  const score_t hess_scale = *hess_scale_ptr;
  // With the robust scale, outliers quantize past +/-(bins/2); clamp them so every
  // magnitude stays <= bins/2, matching the global-max guarantee that the packed-
  // histogram bit-width promotion (num_data*bins bound) relies on.
  const int16_t clamp_max = static_cast<int16_t>(grad_discretize_bins / 2);
  uint32_t* output_packed_ptr = reinterpret_cast<uint32_t*>(output_gradients_and_hessians);
  int32_t thread_q_grad = 0;
  int32_t thread_q_hess = 0;
  if (index < num_data) {
    int16_t q = 0;
    int16_t qh = 0;
    if (STOCHASTIC_ROUNDING) {
      const score_t gradient = input_gradients[index];
      const score_t hessian = input_hessians[CONST_HESS ? 0 : index];
      const uint4 rnd = Philox4x32_10(
        make_uint4(static_cast<uint32_t>(index), static_cast<uint32_t>(iter),
                   0x9E3779B9u, 0xBB67AE85u),
        make_uint2(philox_seed, philox_seed ^ 0x85EBCA6Bu));
      const score_t gradient_random_value = PhiloxUniform(rnd.x);
      const score_t hessian_random_value = PhiloxUniform(rnd.y);
      // Explicit fused multiply-add: ptxas contracts a*b+c differently per
      // GPU architecture, and at quant_bins > 4 the one-ULP difference flips
      // rounding buckets often enough to diverge models across GPU models.
      // Pinning the instruction makes the quantization arch-independent. (The
      // fixedpoint branch below adds a double 0.5 -- unfusable, already
      // arch-stable -- and is left untouched.)
      q = gradient > 0.0f ?
        static_cast<int16_t>(__fmaf_rn(gradient, grad_scale, gradient_random_value)) :
        static_cast<int16_t>(__fmaf_rn(gradient, grad_scale, -gradient_random_value));
      qh = COPY_HESS ?
        reinterpret_cast<const int16_t*>(plane0_gradients_and_hessians)[2 * index] :
        static_cast<int16_t>(__fmaf_rn(hessian, hess_scale, hessian_random_value));
    } else {
      // the double 0.5 is deliberate: a double add cannot be contracted with
      // the float multiply, keeping the rounding sequence arch-stable; MulNoFMA
      // pins the separate multiply before adding the error-feedback residual so
      // DiscretizeGradientsKernel and DiscretizeGradientsChunk256Kernel stay bit-identical
      const score_t gv = MulNoFMA(input_gradients[index], grad_scale);
      const score_t hv = MulNoFMA(input_hessians[CONST_HESS ? 0 : index], hess_scale);
      // out-of-bag residuals must not move: nothing consumes those values
      const bool ef_active = ef_residuals != nullptr &&
          (ef_inbag_mask == nullptr || ef_inbag_mask[index] != 0);
      // residuals are stored as int8 in units of 1/128 quant bin (see Init)
      score_t ge = 0.0f;
      score_t he = 0.0f;
      // SUBTRACTIVE DITHER on the rounding threshold. Error feedback is a
      // sigma-delta loop, and rows that carry the SAME value drive identical
      // loops: without a dither they cross the rounding threshold on the same
      // round and emit their whole-quantum steps in lockstep, so a group of N
      // identical rows contributes N quanta at once instead of N*value. Real
      // data supplies such groups by the thousand (every row that is zero in
      // every feature is one), and a leaf holding one is handed a gradient sum
      // that is pure quantization step. A fixed per-row phase spreads the
      // group's crossings uniformly across rounds; because every row in the
      // group rotates by the same amount each round, the spread is preserved
      // for the whole run. It shifts only WHEN a quantum is emitted: the
      // residual below still accumulates the undithered error, so the loop
      // stays exactly conservative and the mode stays unbiased.
      //
      // The phase is a pure function of (seed, row) and of NOTHING ELSE --
      // deterministic and machine-independent like the stochastic path's noise,
      // and, critically, a property of the ROW rather than of the stream's
      // position. Decorrelating rows is the whole job; two gradient streams
      // that carry identical values owe identical quantization, which is the
      // contract multi-target training is built on (duplicating a target must
      // reproduce the scalar model exactly). Keying on the class/plane slot
      // would hand those two streams different dithers and split them apart.
      // Streams that carry DIFFERENT values need no help staying independent:
      // their own residual trajectories already diverge.
      score_t gphase = 0.0f;
      score_t hphase = 0.0f;
      if (ef_active) {
        he = static_cast<score_t>(ef_residuals[2 * index]) * 0.0078125f;      // 1/128
        ge = static_cast<score_t>(ef_residuals[2 * index + 1]) * 0.0078125f;
        const uint4 ph = Philox4x32_10(
          make_uint4(static_cast<uint32_t>(index), 0x243F6A88u,
                     0x85A308D3u, 0x13198A2Eu),
          make_uint2(philox_seed, philox_seed ^ 0xC2B2AE35u));
        gphase = PhiloxUniform(ph.x) - 0.5f;
        hphase = PhiloxUniform(ph.y) - 0.5f;
      }
      q = (gv + ge + gphase) > 0.0f ?
        static_cast<int16_t>((gv + ge + gphase) + 0.5) :
        static_cast<int16_t>((gv + ge + gphase) - 0.5);
      // hessians are non-negative by construction and the packed histogram
      // reads their field unsigned, so the dither may shift a crossing but
      // never below zero
      const score_t hx = hv + he + hphase;
      qh = COPY_HESS ?
        reinterpret_cast<const int16_t*>(plane0_gradients_and_hessians)[2 * index] :
        (hx > 0.0f ? static_cast<int16_t>(hx + 0.5) : static_cast<int16_t>(0));
      if (ef_active) {
        // rounding error only: the clamp below is intentional saturation and
        // must not feed back, or a saturated row accumulates unbounded carry.
        // A copied-hessian plane rounds no hessian, so it carries no hessian
        // residual (plane 0 owns that stream).
        int rg = __float2int_rn(((gv + ge) - static_cast<score_t>(q)) * 128.0f);
        rg = rg > 127 ? 127 : (rg < -127 ? -127 : rg);
        if (!COPY_HESS) {
          int rh = __float2int_rn(((hv + he) - static_cast<score_t>(qh)) * 128.0f);
          rh = rh > 127 ? 127 : (rh < -127 ? -127 : rh);
          ef_residuals[2 * index] = static_cast<int8_t>(rh);
        }
        ef_residuals[2 * index + 1] = static_cast<int8_t>(rg);
      }
      if (CLAMP) {
        if (q > clamp_max) q = clamp_max;
        else if (q < static_cast<int16_t>(-clamp_max)) q = static_cast<int16_t>(-clamp_max);
      }
    }
    const uint32_t packed = static_cast<uint32_t>(static_cast<uint16_t>(qh)) |
                            (static_cast<uint32_t>(static_cast<uint16_t>(q)) << 16);
    output_packed_ptr[index] = packed;
    if (FUSE_ROOT_SUMS) {
      thread_q_grad = static_cast<int32_t>(q);
      thread_q_hess = static_cast<int32_t>(qh);
    }
  }
  if (FUSE_ROOT_SUMS) {
    // 1024 int16 values sum within +-33.6M (< 2^25): exact in int32, then widened to int64
    // and converted with the exact same (int64 * score_t) expressions as CUDAInitValuesKernel3 / ChunkWarp.
    __shared__ int32_t shared_grad[WARPSIZE];
    __shared__ int32_t shared_hess[WARPSIZE];
    const uint32_t lane = threadIdx.x % warpSize;
    const uint32_t warp_id = threadIdx.x / warpSize;
    int32_t g_sum = ShuffleReduceSumWarp<int32_t>(thread_q_grad, warpSize);
    int32_t h_sum = ShuffleReduceSumWarp<int32_t>(thread_q_hess, warpSize);
    if (lane == 0) {
      shared_grad[warp_id] = g_sum;
      shared_hess[warp_id] = h_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
      const uint32_t num_warps = blockDim.x / warpSize;
      g_sum = lane < num_warps ? shared_grad[lane] : 0;
      h_sum = lane < num_warps ? shared_hess[lane] : 0;
      g_sum = ShuffleReduceSumWarp<int32_t>(g_sum, static_cast<data_size_t>(num_warps));
      h_sum = ShuffleReduceSumWarp<int32_t>(h_sum, static_cast<data_size_t>(num_warps));
      if (lane == 0) {
        const int64_t block_sum_gradient = static_cast<int64_t>(g_sum);
        const int64_t block_sum_hessian = static_cast<int64_t>(h_sum);
        const score_t grad_dequant = *grad_dequant_ptr;
        const score_t hess_dequant = *hess_dequant_ptr;
        out_block_sum_grad[blockIdx.x] = block_sum_gradient * grad_dequant;
        out_block_sum_hess[blockIdx.x] = block_sum_hessian * hess_dequant;
        out_block_sum_grad_hess[blockIdx.x] = ((block_sum_gradient << 32) | block_sum_hessian);
        if (blockIdx.x == 0 && plane_grad_scale != nullptr) {
          *plane_grad_scale = grad_dequant;
          if (plane_hess_scale != nullptr) {
            *plane_hess_scale = hess_dequant;
          }
        }
      }
    }
  }
}

// Snapshot the reduced dequantization scales out of the shared reduce scratch,
// which the next plane's reduction overwrites.
#if !defined(__HIP_PLATFORM_AMD__)
// cuda_plan key discretize_root_sums: one 256-thread block (8 warps, 4 rows/thread) per 1024-row chunk.
// Launches num_chunks blocks so even 285k-580k datasets saturate all 170 SMs, while 256-thread blocks keep
// 5-6 blocks resident per SM on sm_120 (83-100% occupancy vs 1 block / 66% at 1024 threads) so the tiny
// 8-warp shared reduction never stalls the SM. Bit-identical.
template <bool STOCHASTIC_ROUNDING, bool CLAMP, bool CONST_HESS>
__global__ void DiscretizeGradientsChunk256Kernel(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians,
  const score_t* grad_scale_ptr,
  const score_t* hess_scale_ptr,
  const int iter,
  const uint32_t philox_seed,
  const int grad_discretize_bins,
  int8_t* ef_residuals,
  const uint8_t* ef_inbag_mask,
  int8_t* output_gradients_and_hessians,
  const score_t* grad_dequant_ptr,
  const score_t* hess_dequant_ptr,
  double* out_block_sum_grad,
  double* out_block_sum_hess,
  int64_t* out_block_sum_grad_hess,
  score_t* plane_grad_scale,
  score_t* plane_hess_scale) {
  PdlWaitForProducer();
  PdlLaunchDependents();
  constexpr int kThreads = 256;
  constexpr int kRowsPerThread = CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE / kThreads;  // 4
  constexpr int kNumWarps = kThreads / WARPSIZE;  // 8
  const int chunk = static_cast<int>(blockIdx.x);
  const data_size_t chunk_start = static_cast<data_size_t>(chunk) * CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE;
  const score_t grad_scale = *grad_scale_ptr;
  const score_t hess_scale = *hess_scale_ptr;
  const score_t const_hess_val = (CONST_HESS && num_data > 0) ? input_hessians[0] : 0.0f;
  const int16_t clamp_max = static_cast<int16_t>(grad_discretize_bins / 2);
  uint32_t* output_packed_ptr = reinterpret_cast<uint32_t*>(output_gradients_and_hessians);
  int32_t sum_g = 0;
  int32_t sum_h = 0;
  const bool full_chunk = chunk_start + CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE <= num_data;
#pragma unroll
  for (int k = 0; k < kRowsPerThread; ++k) {
    const data_size_t index = chunk_start + static_cast<data_size_t>(k * kThreads + threadIdx.x);
    if (full_chunk || index < num_data) {
      int16_t q = 0;
      int16_t qh = 0;
      if (STOCHASTIC_ROUNDING) {
        const score_t gradient = input_gradients[index];
        const score_t hessian = CONST_HESS ? const_hess_val : input_hessians[index];
        const uint4 rnd = Philox4x32_10(
          make_uint4(static_cast<uint32_t>(index), static_cast<uint32_t>(iter),
                     0x9E3779B9u, 0xBB67AE85u),
          make_uint2(philox_seed, philox_seed ^ 0x85EBCA6Bu));
        const score_t gradient_random_value = PhiloxUniform(rnd.x);
        const score_t hessian_random_value = PhiloxUniform(rnd.y);
        q = gradient > 0.0f ?
          static_cast<int16_t>(__fmaf_rn(gradient, grad_scale, gradient_random_value)) :
          static_cast<int16_t>(__fmaf_rn(gradient, grad_scale, -gradient_random_value));
        qh = static_cast<int16_t>(__fmaf_rn(hessian, hess_scale, hessian_random_value));
      } else {
        // MulNoFMA matches DiscretizeGradientsKernel's separate FMUL + FADD (and DMUL + DADD under SCORE_T_USE_DOUBLE)
        const score_t gv = MulNoFMA(input_gradients[index], grad_scale);
        const score_t hv = MulNoFMA(CONST_HESS ? const_hess_val : input_hessians[index], hess_scale);
        const bool ef_active = ef_residuals != nullptr &&
            (ef_inbag_mask == nullptr || ef_inbag_mask[index] != 0);
        score_t ge = 0.0f;
        score_t he = 0.0f;
        score_t gphase = 0.0f;
        score_t hphase = 0.0f;
        if (ef_active) {
          he = static_cast<score_t>(ef_residuals[2 * index]) * 0.0078125f;
          ge = static_cast<score_t>(ef_residuals[2 * index + 1]) * 0.0078125f;
          const uint4 ph = Philox4x32_10(
            make_uint4(static_cast<uint32_t>(index), 0x243F6A88u,
                       0x85A308D3u, 0x13198A2Eu),
            make_uint2(philox_seed, philox_seed ^ 0xC2B2AE35u));
          gphase = PhiloxUniform(ph.x) - 0.5f;
          hphase = PhiloxUniform(ph.y) - 0.5f;
        }
        q = (gv + ge + gphase) > 0.0f ?
          static_cast<int16_t>((gv + ge + gphase) + 0.5) :
          static_cast<int16_t>((gv + ge + gphase) - 0.5);
        const score_t hx = hv + he + hphase;
        qh = hx > 0.0f ? static_cast<int16_t>(hx + 0.5) : static_cast<int16_t>(0);
        if (ef_active) {
          int rg = __float2int_rn(((gv + ge) - static_cast<score_t>(q)) * 128.0f);
          rg = rg > 127 ? 127 : (rg < -127 ? -127 : rg);
          int rh = __float2int_rn(((hv + he) - static_cast<score_t>(qh)) * 128.0f);
          rh = rh > 127 ? 127 : (rh < -127 ? -127 : rh);
          ef_residuals[2 * index] = static_cast<int8_t>(rh);
          ef_residuals[2 * index + 1] = static_cast<int8_t>(rg);
        }
        if (CLAMP) {
          if (q > clamp_max) q = clamp_max;
          else if (q < static_cast<int16_t>(-clamp_max)) q = static_cast<int16_t>(-clamp_max);
        }
      }
      const uint32_t packed = static_cast<uint32_t>(static_cast<uint16_t>(qh)) |
                              (static_cast<uint32_t>(static_cast<uint16_t>(q)) << 16);
      output_packed_ptr[index] = packed;
      sum_g += static_cast<int32_t>(q);
      sum_h += static_cast<int32_t>(qh);
    }
  }
  __shared__ int32_t s_grad[kNumWarps];
  __shared__ int32_t s_hess[kNumWarps];
  const uint32_t lane = threadIdx.x % WARPSIZE;
  const uint32_t warp_id = threadIdx.x / WARPSIZE;
  sum_g = ShuffleReduceSumWarp<int32_t>(sum_g, WARPSIZE);
  sum_h = ShuffleReduceSumWarp<int32_t>(sum_h, WARPSIZE);
  if (lane == 0) {
    s_grad[warp_id] = sum_g;
    s_hess[warp_id] = sum_h;
  }
  __syncthreads();
  if (warp_id == 0) {
    sum_g = lane < kNumWarps ? s_grad[lane] : 0;
    sum_h = lane < kNumWarps ? s_hess[lane] : 0;
    for (int offset = kNumWarps / 2; offset > 0; offset >>= 1) {
      sum_g += __shfl_down_sync(0xffffffff, sum_g, offset);
      sum_h += __shfl_down_sync(0xffffffff, sum_h, offset);
    }
    if (lane == 0) {
      const int64_t block_sum_gradient = static_cast<int64_t>(sum_g);
      const int64_t block_sum_hessian = static_cast<int64_t>(sum_h);
      const score_t grad_dequant = *grad_dequant_ptr;
      const score_t hess_dequant = *hess_dequant_ptr;
      out_block_sum_grad[chunk] = block_sum_gradient * grad_dequant;
      out_block_sum_hess[chunk] = block_sum_hessian * hess_dequant;
      out_block_sum_grad_hess[chunk] = ((block_sum_gradient << 32) | block_sum_hessian);
      if (chunk == 0 && plane_grad_scale != nullptr) {
        *plane_grad_scale = grad_dequant;
        if (plane_hess_scale != nullptr) {
          *plane_hess_scale = hess_dequant;
        }
      }
    }
  }
}
#endif  // !defined(__HIP_PLATFORM_AMD__)

__global__ void SnapshotPlaneScalesKernel(
  const score_t* grad_scale_src,
  const score_t* hess_scale_src,
  score_t* plane_grad_scale,
  score_t* plane_hess_scale) {
  PdlWaitForProducer();  // cuda_plan key pdl_level_chain: the kernel before it on the stream is complete and visible
  if (threadIdx.x == 0) {
    *plane_grad_scale = *grad_scale_src;
    if (plane_hess_scale != nullptr) {
      *plane_hess_scale = *hess_scale_src;
    }
  }
}

void CUDAGradientDiscretizer::DiscretizeGradients(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians) {
  DiscretizeGradientsForPlaneImpl(0, num_data, input_gradients, input_hessians, nullptr, nullptr, nullptr);
}

void CUDAGradientDiscretizer::DiscretizeGradientsWithRootSums(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians,
  double* out_block_sum_grad,
  double* out_block_sum_hess,
  int64_t* out_block_sum_grad_hess) {
  DiscretizeGradientsForPlaneImpl(0, num_data, input_gradients, input_hessians,
                                  out_block_sum_grad, out_block_sum_hess, out_block_sum_grad_hess);
}

void CUDAGradientDiscretizer::DiscretizeGradientsForPlane(
  const int plane,
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians) {
  DiscretizeGradientsForPlaneImpl(plane, num_data, input_gradients, input_hessians, nullptr, nullptr, nullptr);
}

void CUDAGradientDiscretizer::DiscretizeGradientsForPlaneImpl(
  const int plane,
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians,
  double* out_block_sum_grad,
  double* out_block_sum_hess,
  int64_t* out_block_sum_grad_hess) {
  // cuda_plan key const_hess_reads: see SetHessiansConstant
  const bool const_hess = hessians_constant_ && FalcataPlan::Get().const_hess_reads && num_data > 0;
#if !defined(__HIP_PLATFORM_AMD__)
  const bool chunk_warp = FalcataPlan::Get().minmax_warp;
#else
  const bool chunk_warp = false;  // the warp-per-chunk kernel is CUDA only
#endif
  // cuda_plan key grad_minmax_fuse: reuse the per-1024-row block min/max partials already produced by
  // the objective's GetGradients kernel (plane 0 only, consumed once so stale pointers can never survive)
  const bool use_precomputed_minmax =
      plane == 0 &&
      precomputed_grad_min_ != nullptr &&
      precomputed_for_gradients_ == input_gradients &&
      precomputed_for_hessians_ == input_hessians &&
      precomputed_num_data_ == num_data &&
      precomputed_num_blocks_ == num_reduce_blocks_ &&
      FalcataPlan::Get().grad_minmax_fuse;
  const score_t* reduce_in_grad_min =
      use_precomputed_minmax ? precomputed_grad_min_ : grad_min_block_buffer_.RawDataReadOnly();
  const score_t* reduce_in_grad_max =
      use_precomputed_minmax ? precomputed_grad_max_ : grad_max_block_buffer_.RawDataReadOnly();
  const score_t* reduce_in_hess_min =
      use_precomputed_minmax ? precomputed_hess_min_ : hess_min_block_buffer_.RawDataReadOnly();
  const score_t* reduce_in_hess_max =
      use_precomputed_minmax ? precomputed_hess_max_ : hess_max_block_buffer_.RawDataReadOnly();
  precomputed_grad_min_ = nullptr;
  precomputed_grad_max_ = nullptr;
  precomputed_hess_min_ = nullptr;
  precomputed_hess_max_ = nullptr;
  precomputed_for_gradients_ = nullptr;
  precomputed_for_hessians_ = nullptr;
  precomputed_num_data_ = 0;
  precomputed_num_blocks_ = 0;

  // cuda_plan key async_tree_start: these kernels share the legacy default stream and the host reads none of
  // their outputs in between, so the device syncs between them only add launch gaps
  const bool device_syncs = !AsyncTreeStart();
  // cuda_plan key pdl_level_chain: a programmatic dependent of the kernel before it (the tree start's leaf window init)
  const bool pdl_reduce = PdlLevelChainEnabled();
  // cuda_plan key pdl_level_chain: the tree-start chain (block min/max, discretize, scale snapshot) launched as
  // programmatic dependents of the kernel before each on the legacy default stream, where no device sync separates them
  const bool pdl = !device_syncs && pdl_reduce;
  if (!use_precomputed_minmax) {
    if (chunk_warp) {
#if !defined(__HIP_PLATFORM_AMD__)
      // one warp per chunk, eight chunks per 256-thread block
      constexpr int kChunkWarpThreads = 256;
      const int chunk_grid = (num_reduce_blocks_ * WARPSIZE + kChunkWarpThreads - 1) / kChunkWarpThreads;
      if (const_hess) {
        LaunchLevelChainKernel(pdl_reduce, ReduceMinMaxChunkWarpKernel<true>, dim3(chunk_grid), dim3(kChunkWarpThreads), 0, nullptr,
          num_data, num_reduce_blocks_, input_gradients, input_hessians,
          grad_min_block_buffer_.RawData(),
          grad_max_block_buffer_.RawData(),
          hess_min_block_buffer_.RawData(),
          hess_max_block_buffer_.RawData());
      } else {
        LaunchLevelChainKernel(pdl_reduce, ReduceMinMaxChunkWarpKernel<false>, dim3(chunk_grid), dim3(kChunkWarpThreads), 0, nullptr,
          num_data, num_reduce_blocks_, input_gradients, input_hessians,
          grad_min_block_buffer_.RawData(),
          grad_max_block_buffer_.RawData(),
          hess_min_block_buffer_.RawData(),
          hess_max_block_buffer_.RawData());
      }
#endif
    } else {
      LaunchLevelChainKernel(pdl_reduce, ReduceMinMaxKernel, dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0,
        nullptr,
        num_data, input_gradients, input_hessians,
        grad_min_block_buffer_.RawData(),
        grad_max_block_buffer_.RawData(),
        hess_min_block_buffer_.RawData(),
        hess_max_block_buffer_.RawData());
    }
    if (device_syncs) {
      SynchronizeCUDADevice(__FILE__, __LINE__);
    }
  }
  LaunchLevelChainKernel(use_precomputed_minmax ? pdl_reduce : pdl,
    ReduceBlockMinMaxKernel, dim3(1), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
    num_reduce_blocks_,
    num_grad_quant_bins_,
    reduce_in_grad_min,
    reduce_in_grad_max,
    reduce_in_hess_min,
    reduce_in_hess_max,
    grad_min_block_buffer_.RawData(),
    grad_max_block_buffer_.RawData(),
    hess_min_block_buffer_.RawData(),
    hess_max_block_buffer_.RawData());
  if (device_syncs) {
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }

  if (nccl_communicator_ != nullptr) {
    SynchronizeCUDADevice(__FILE__, __LINE__);
    cudaStream_t cuda_stream = CUDAStreamCreate();
    NCCLGroupStart();
    NCCLAllReduce<score_t>(grad_min_block_buffer_.RawDataReadOnly(), grad_min_block_buffer_.RawData(), 1, ncclFloat32, ncclMin, nccl_communicator_, cuda_stream);
    NCCLAllReduce<score_t>(hess_min_block_buffer_.RawDataReadOnly(), hess_min_block_buffer_.RawData(), 1, ncclFloat32, ncclMin, nccl_communicator_, cuda_stream);
    NCCLAllReduce<score_t>(grad_max_block_buffer_.RawDataReadOnly(), grad_max_block_buffer_.RawData(), 1, ncclFloat32, ncclMax, nccl_communicator_, cuda_stream);
    NCCLAllReduce<score_t>(hess_max_block_buffer_.RawDataReadOnly(), hess_max_block_buffer_.RawData(), 1, ncclFloat32, ncclMax, nccl_communicator_, cuda_stream);
    NCCLGroupEnd();
    SynchronizeCUDAStream(cuda_stream, __FILE__, __LINE__);
    CUDAStreamDestroy(cuda_stream);
  }

  if (robust_scale_) {
    // Replace the global-max grad scale with a gap-gated outlier-robust scale.
    // Build a log-magnitude histogram of |grad|, then (only if a genuine outlier
    // gap is present) re-anchor the scale to the bulk. Cheap: one shared-memory
    // histogram pass over the data + a single-thread bucket scan. hess scale is
    // untouched (hessians are not the imbalanced-outlier problem). Under NCCL each
    // rank uses its local histogram, which is acceptable (the scale is only a
    // resolution knob and these buffers are not all-reduced here); single-GPU is
    // the validated path.
    const size_t mag_hist_shared_bytes = static_cast<size_t>(kNumMagBuckets) * sizeof(unsigned int);
    if (FalcataPlan::Get().robust_scale_async) {
      if (!mag_hist_zeroed_) {
        if (device_syncs) {
          SetCUDAMemory<unsigned int>(grad_mag_hist_buffer_.RawData(), 0, kNumMagBuckets, __FILE__, __LINE__);
        } else {
          CUDASUCCESS_OR_FATAL(cudaMemsetAsync(grad_mag_hist_buffer_.RawData(), 0,
            static_cast<size_t>(kNumMagBuckets) * sizeof(unsigned int), nullptr));
        }
        mag_hist_zeroed_ = true;
      }
      constexpr int kMagHistBlockSize = 256;
      const int min_blocks = CUDABlockCount(num_data, kMagHistBlockSize * 4);
      const int max_blocks = std::max(1, 2 * CachedDeviceSMCount());
      const int mag_hist_grid = std::max(1, std::min(min_blocks, max_blocks));
      const bool pdl_hist = pdl && nccl_communicator_ == nullptr;
      LaunchLevelChainKernel(pdl_hist, ComputeGradMagHistogramAsyncKernel,
        dim3(mag_hist_grid), dim3(kMagHistBlockSize), mag_hist_shared_bytes, nullptr,
        num_data, input_gradients,
        grad_max_block_buffer_.RawDataReadOnly(),
        num_grad_quant_bins_,
        kNumMagBuckets,
        grad_mag_hist_buffer_.RawData());
      if (device_syncs) {
        SynchronizeCUDADevice(__FILE__, __LINE__);
      }
      LaunchLevelChainKernel(pdl, ComputeRobustGradScaleAsyncKernel,
        dim3(1), dim3(kMagHistBlockSize), 0, nullptr,
        kNumMagBuckets,
        num_grad_quant_bins_,
        num_data,
        grad_mag_hist_buffer_.RawData(),
        grad_min_block_buffer_.RawData(),
        grad_max_block_buffer_.RawData());
      if (device_syncs) {
        SynchronizeCUDADevice(__FILE__, __LINE__);
      }
    } else {
      SetCUDAMemory<unsigned int>(grad_mag_hist_buffer_.RawData(), 0, kNumMagBuckets, __FILE__, __LINE__);
      mag_hist_zeroed_ = false;
      ComputeGradMagHistogramKernel<<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE, mag_hist_shared_bytes>>>(
        num_data, input_gradients,
        grad_max_block_buffer_.RawDataReadOnly(),
        num_grad_quant_bins_,
        kNumMagBuckets,
        grad_mag_hist_buffer_.RawData());
      SynchronizeCUDADevice(__FILE__, __LINE__);
      ComputeRobustGradScaleKernel<<<1, 1>>>(
        kNumMagBuckets,
        num_grad_quant_bins_,
        num_data,
        grad_mag_hist_buffer_.RawDataReadOnly(),
        grad_min_block_buffer_.RawData(),
        grad_max_block_buffer_.RawData());
      SynchronizeCUDADevice(__FILE__, __LINE__);
    }
  }

  // iter_ advances once per tree and classes train in fixed order, so
  // iter_ % ef_num_slots_ is this tree's class slot
  int8_t* ef_slot = error_feedback_
      ? ef_residuals_.RawData() +
            static_cast<size_t>(iter_ % ef_num_slots_) * static_cast<size_t>(num_data) * 2
      : nullptr;
  const uint8_t* ef_mask = nullptr;
  if (ef_slot != nullptr && ef_inbag_indices_ != nullptr && ef_inbag_count_ < num_data) {
    if (ef_inbag_mask_.Size() < static_cast<size_t>(num_data)) {
      ef_inbag_mask_.Resize(static_cast<size_t>(num_data));
    }
    SetCUDAMemory<uint8_t>(ef_inbag_mask_.RawData(), 0, static_cast<size_t>(num_data), __FILE__, __LINE__);
    const int mask_blocks = CUDABlockCount(ef_inbag_count_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE);
    BuildInbagMaskKernel<<<mask_blocks, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(
      ef_inbag_count_, ef_inbag_indices_, ef_inbag_mask_.RawData());
    ef_mask = ef_inbag_mask_.RawData();
  }

  // Plane 0 owns the hessian stream; planes 1..T-1 copy its quantized hessians
  // (see the COPY_HESS branch of DiscretizeGradientsKernel).
  const bool copy_hess = plane > 0;
  const int8_t* plane0 = discretized_gradients_and_hessians(0);
  const bool fuse_root_sums =
      !copy_hess &&
      out_block_sum_grad != nullptr &&
      out_block_sum_hess != nullptr &&
      out_block_sum_grad_hess != nullptr &&
      num_reduce_blocks_ > 0;

  #define DiscretizeGradientsKernel_COMMON_ARGS \
    num_data, \
    input_gradients, \
    input_hessians, \
    grad_min_block_buffer_.RawData(), \
    hess_min_block_buffer_.RawData(), \
    iter_, /* Philox counter component: any 32-bit value is a valid stream */ \
    static_cast<uint32_t>(random_seed_), \
    num_grad_quant_bins_, \
    ef_slot, \
    ef_mask, \
    discretized_gradients_and_hessians_.RawData() + \
      static_cast<size_t>(plane) * static_cast<size_t>(num_data_planes_) * 4, \
    plane0

  #define DiscretizeGradientsKernel_ARGS \
    DiscretizeGradientsKernel_COMMON_ARGS, \
    static_cast<const score_t*>(nullptr), \
    static_cast<const score_t*>(nullptr), \
    static_cast<double*>(nullptr), \
    static_cast<double*>(nullptr), \
    static_cast<int64_t*>(nullptr), \
    static_cast<score_t*>(nullptr), \
    static_cast<score_t*>(nullptr)

  #define DiscretizeGradientsKernel_FUSED_ARGS \
    DiscretizeGradientsKernel_COMMON_ARGS, \
    grad_max_block_buffer_.RawDataReadOnly(), \
    hess_max_block_buffer_.RawDataReadOnly(), \
    out_block_sum_grad, \
    out_block_sum_hess, \
    out_block_sum_grad_hess, \
    plane_grad_scale_.RawData() + plane, \
    plane_hess_scale_.RawData()

  if (fuse_root_sums) {
#if !defined(__HIP_PLATFORM_AMD__)
    constexpr int kChunk256Threads = 256;
    #define DiscretizeChunk256_ARGS \
      num_data, \
      input_gradients, \
      input_hessians, \
      grad_min_block_buffer_.RawData(), \
      hess_min_block_buffer_.RawData(), \
      iter_, \
      static_cast<uint32_t>(random_seed_), \
      num_grad_quant_bins_, \
      ef_slot, \
      ef_mask, \
      discretized_gradients_and_hessians_.RawData() + \
        static_cast<size_t>(plane) * static_cast<size_t>(num_data_planes_) * 4, \
      grad_max_block_buffer_.RawDataReadOnly(), \
      hess_max_block_buffer_.RawDataReadOnly(), \
      out_block_sum_grad, \
      out_block_sum_hess, \
      out_block_sum_grad_hess, \
      plane_grad_scale_.RawData() + plane, \
      plane_hess_scale_.RawData()
    if (stochastic_rounding_) {
      if (const_hess) {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsChunk256Kernel<true, false, true>,
          dim3(num_reduce_blocks_), dim3(kChunk256Threads), 0, nullptr, DiscretizeChunk256_ARGS);
      } else {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsChunk256Kernel<true, false, false>,
          dim3(num_reduce_blocks_), dim3(kChunk256Threads), 0, nullptr, DiscretizeChunk256_ARGS);
      }
    } else if (robust_scale_) {
      if (const_hess) {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsChunk256Kernel<false, true, true>,
          dim3(num_reduce_blocks_), dim3(kChunk256Threads), 0, nullptr, DiscretizeChunk256_ARGS);
      } else {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsChunk256Kernel<false, true, false>,
          dim3(num_reduce_blocks_), dim3(kChunk256Threads), 0, nullptr, DiscretizeChunk256_ARGS);
      }
    } else {
      if (const_hess) {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsChunk256Kernel<false, false, true>,
          dim3(num_reduce_blocks_), dim3(kChunk256Threads), 0, nullptr, DiscretizeChunk256_ARGS);
      } else {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsChunk256Kernel<false, false, false>,
          dim3(num_reduce_blocks_), dim3(kChunk256Threads), 0, nullptr, DiscretizeChunk256_ARGS);
      }
    }
    #undef DiscretizeChunk256_ARGS
#else
    if (stochastic_rounding_) {
      if (const_hess) {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<true, false, false, true, true>,
          dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_FUSED_ARGS);
      } else {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<true, false, false, false, true>,
          dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_FUSED_ARGS);
      }
    } else if (robust_scale_) {
      if (const_hess) {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<false, true, false, true, true>,
          dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_FUSED_ARGS);
      } else {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<false, true, false, false, true>,
          dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_FUSED_ARGS);
      }
    } else {
      if (const_hess) {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<false, false, false, true, true>,
          dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_FUSED_ARGS);
      } else {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<false, false, false, false, true>,
          dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_FUSED_ARGS);
      }
    }
#endif
  } else {
    if (stochastic_rounding_) {
      // Stochastic path never uses the robust scale (it is a fixed-point-only mode).
      if (copy_hess) {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<true, false, true>, dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_ARGS);
      } else if (const_hess) {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<true, false, false, true>, dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_ARGS);
      } else {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<true, false, false>, dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_ARGS);
      }
    } else if (robust_scale_) {
      if (copy_hess) {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<false, true, true>, dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_ARGS);
      } else if (const_hess) {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<false, true, false, true>, dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_ARGS);
      } else {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<false, true, false>, dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_ARGS);
      }
    } else {
      if (copy_hess) {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<false, false, true>, dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_ARGS);
      } else if (const_hess) {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<false, false, false, true>, dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_ARGS);
      } else {
        LaunchLevelChainKernel(pdl, DiscretizeGradientsKernel<false, false, false>, dim3(num_reduce_blocks_), dim3(CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE), 0, nullptr,
          DiscretizeGradientsKernel_ARGS);
      }
    }
    // the scales the find kernels dequantize with: per-plane for the gradient,
    // plane 0's for the shared hessian (fused path snapshots these in block 0 of DiscretizeGradientsKernel)
    LaunchLevelChainKernel(pdl, SnapshotPlaneScalesKernel, dim3(1), dim3(1), 0, nullptr,
      grad_max_block_buffer_.RawDataReadOnly(),
      hess_max_block_buffer_.RawDataReadOnly(),
      plane_grad_scale_.RawData() + plane,
      copy_hess ? nullptr : plane_hess_scale_.RawData());
  }
  #undef DiscretizeGradientsKernel_FUSED_ARGS
  #undef DiscretizeGradientsKernel_ARGS
  if (device_syncs) {
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }
  ++iter_;
}

}  // namespace Falcata

#endif  // USE_CUDA
