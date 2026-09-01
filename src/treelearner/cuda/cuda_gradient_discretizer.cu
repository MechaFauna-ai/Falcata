/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <algorithm>

#include <Falcata/cuda/cuda_algorithms.hpp>

#include "cuda_gradient_discretizer.hpp"

namespace Falcata {

__global__ void ReduceMinMaxKernel(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians,
  score_t* grad_min_block_buffer,
  score_t* grad_max_block_buffer,
  score_t* hess_min_block_buffer,
  score_t* hess_max_block_buffer) {
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

__global__ void ReduceBlockMinMaxKernel(
  const int num_blocks,
  const int grad_discretize_bins,
  score_t* grad_min_block_buffer,
  score_t* grad_max_block_buffer,
  score_t* hess_min_block_buffer,
  score_t* hess_max_block_buffer) {
  __shared__ score_t shared_mem_buffer[WARPSIZE];
  score_t grad_max_val = kMinScore;
  score_t grad_min_val = kMaxScore;
  score_t hess_max_val = kMinScore;
  score_t hess_min_val = kMaxScore;
  for (int block_index = static_cast<int>(threadIdx.x); block_index < num_blocks; block_index += static_cast<int>(blockDim.x)) {
    grad_min_val = min(grad_min_val, grad_min_block_buffer[block_index]);
    grad_max_val = max(grad_max_val, grad_max_block_buffer[block_index]);
    hess_min_val = min(hess_min_val, hess_min_block_buffer[block_index]);
    hess_max_val = max(hess_max_val, hess_max_block_buffer[block_index]);
  }
  grad_min_val = ShuffleReduceMin<score_t>(grad_min_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  grad_max_val = ShuffleReduceMax<score_t>(grad_max_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  hess_max_val = ShuffleReduceMax<score_t>(hess_max_val, shared_mem_buffer, blockDim.x);
  __syncthreads();
  hess_max_val = ShuffleReduceMax<score_t>(hess_max_val, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    const score_t grad_abs_max = max(fabs(grad_min_val), fabs(grad_max_val));
    const score_t hess_abs_max = max(fabs(hess_min_val), fabs(hess_max_val));
    grad_min_block_buffer[0] = 1.0f / (grad_abs_max / (grad_discretize_bins / 2));
    grad_max_block_buffer[0] = (grad_abs_max / (grad_discretize_bins / 2));
    hess_min_block_buffer[0] = 1.0f / (hess_abs_max / (grad_discretize_bins));
    hess_max_block_buffer[0] = (hess_abs_max / (grad_discretize_bins));
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
#define kMagBucketsPerOctave (128.0)

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
      int bucket = static_cast<int>(floor(-log2f(frac) * kMagBucketsPerOctave));
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
template <bool STOCHASTIC_ROUNDING, bool CLAMP, bool COPY_HESS = false>
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
  const int8_t* plane0_gradients_and_hessians = nullptr) {
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const score_t grad_scale = *grad_scale_ptr;
  const score_t hess_scale = *hess_scale_ptr;
  // With the robust scale, outliers quantize past +/-(bins/2); clamp them so every
  // magnitude stays <= bins/2, matching the global-max guarantee that the packed-
  // histogram bit-width promotion (num_data*bins bound) relies on.
  const int16_t clamp_max = static_cast<int16_t>(grad_discretize_bins / 2);
  int16_t* output_gradients_and_hessians_ptr = reinterpret_cast<int16_t*>(output_gradients_and_hessians);
  if (index < num_data) {
    if (STOCHASTIC_ROUNDING) {
      const score_t gradient = input_gradients[index];
      const score_t hessian = input_hessians[index];
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
      output_gradients_and_hessians_ptr[2 * index + 1] = gradient > 0.0f ?
        static_cast<int16_t>(__fmaf_rn(gradient, grad_scale, gradient_random_value)) :
        static_cast<int16_t>(__fmaf_rn(gradient, grad_scale, -gradient_random_value));
      output_gradients_and_hessians_ptr[2 * index] = COPY_HESS ?
        reinterpret_cast<const int16_t*>(plane0_gradients_and_hessians)[2 * index] :
        static_cast<int16_t>(__fmaf_rn(hessian, hess_scale, hessian_random_value));
    } else {
      // the double 0.5 is deliberate: a double add cannot be contracted with
      // the float multiply, keeping the rounding sequence arch-stable
      const score_t gv = input_gradients[index] * grad_scale;
      const score_t hv = input_hessians[index] * hess_scale;
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
      int16_t q = (gv + ge + gphase) > 0.0f ?
        static_cast<int16_t>((gv + ge + gphase) + 0.5) :
        static_cast<int16_t>((gv + ge + gphase) - 0.5);
      // hessians are non-negative by construction and the packed histogram
      // reads their field unsigned, so the dither may shift a crossing but
      // never below zero
      const score_t hx = hv + he + hphase;
      const int16_t qh = COPY_HESS ?
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
      output_gradients_and_hessians_ptr[2 * index + 1] = q;
      output_gradients_and_hessians_ptr[2 * index] = qh;
    }
  }
}

// Snapshot the reduced dequantization scales out of the shared reduce scratch,
// which the next plane's reduction overwrites.
__global__ void SnapshotPlaneScalesKernel(
  const score_t* grad_scale_src,
  const score_t* hess_scale_src,
  score_t* plane_grad_scale,
  score_t* plane_hess_scale) {
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
  DiscretizeGradientsForPlane(0, num_data, input_gradients, input_hessians);
}

void CUDAGradientDiscretizer::DiscretizeGradientsForPlane(
  const int plane,
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians) {
  ReduceMinMaxKernel<<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(
    num_data, input_gradients, input_hessians,
    grad_min_block_buffer_.RawData(),
    grad_max_block_buffer_.RawData(),
    hess_min_block_buffer_.RawData(),
    hess_max_block_buffer_.RawData());
    SynchronizeCUDADevice(__FILE__, __LINE__);
  ReduceBlockMinMaxKernel<<<1, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(
    num_reduce_blocks_,
    num_grad_quant_bins_,
    grad_min_block_buffer_.RawData(),
    grad_max_block_buffer_.RawData(),
    hess_min_block_buffer_.RawData(),
    hess_max_block_buffer_.RawData());
    SynchronizeCUDADevice(__FILE__, __LINE__);

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
    SetCUDAMemory<unsigned int>(grad_mag_hist_buffer_.RawData(), 0, kNumMagBuckets, __FILE__, __LINE__);
    const size_t mag_hist_shared_bytes = static_cast<size_t>(kNumMagBuckets) * sizeof(unsigned int);
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
    const int mask_blocks = (ef_inbag_count_ + CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE - 1) / CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE;
    BuildInbagMaskKernel<<<mask_blocks, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(
      ef_inbag_count_, ef_inbag_indices_, ef_inbag_mask_.RawData());
    ef_mask = ef_inbag_mask_.RawData();
  }

  // Plane 0 owns the hessian stream; planes 1..T-1 copy its quantized hessians
  // (see the COPY_HESS branch of DiscretizeGradientsKernel).
  const bool copy_hess = plane > 0;
  const int8_t* plane0 = discretized_gradients_and_hessians(0);

  #define DiscretizeGradientsKernel_ARGS \
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

  if (stochastic_rounding_) {
    // Stochastic path never uses the robust scale (it is a fixed-point-only mode).
    if (copy_hess) {
      DiscretizeGradientsKernel<true, false, true><<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(DiscretizeGradientsKernel_ARGS);
    } else {
      DiscretizeGradientsKernel<true, false, false><<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(DiscretizeGradientsKernel_ARGS);
    }
  } else if (robust_scale_) {
    if (copy_hess) {
      DiscretizeGradientsKernel<false, true, true><<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(DiscretizeGradientsKernel_ARGS);
    } else {
      DiscretizeGradientsKernel<false, true, false><<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(DiscretizeGradientsKernel_ARGS);
    }
  } else {
    if (copy_hess) {
      DiscretizeGradientsKernel<false, false, true><<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(DiscretizeGradientsKernel_ARGS);
    } else {
      DiscretizeGradientsKernel<false, false, false><<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(DiscretizeGradientsKernel_ARGS);
    }
  }
  // the scales the find kernels dequantize with: per-plane for the gradient,
  // plane 0's for the shared hessian
  SnapshotPlaneScalesKernel<<<1, 1>>>(
    grad_max_block_buffer_.RawDataReadOnly(),
    hess_max_block_buffer_.RawDataReadOnly(),
    plane_grad_scale_.RawData() + plane,
    copy_hess ? nullptr : plane_hess_scale_.RawData());
  SynchronizeCUDADevice(__FILE__, __LINE__);
  ++iter_;
}

}  // namespace Falcata

#endif  // USE_CUDA
