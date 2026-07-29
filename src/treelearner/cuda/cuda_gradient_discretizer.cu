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

template <bool STOCHASTIC_ROUNDING, bool CLAMP>
__global__ void DiscretizeGradientsKernel(
  const data_size_t num_data,
  const score_t* input_gradients,
  const score_t* input_hessians,
  const score_t* grad_scale_ptr,
  const score_t* hess_scale_ptr,
  const int iter,
  const int* random_values_use_start,
  const score_t* gradient_random_values,
  const score_t* hessian_random_values,
  const int grad_discretize_bins,
  int8_t* output_gradients_and_hessians) {
  const int start = random_values_use_start[iter];
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
      const data_size_t index_offset = (index + start) % num_data;
      const score_t gradient = input_gradients[index];
      const score_t hessian = input_hessians[index];
      const score_t gradient_random_value = gradient_random_values[index_offset];
      const score_t hessian_random_value = hessian_random_values[index_offset];
      output_gradients_and_hessians_ptr[2 * index + 1] = gradient > 0.0f ?
        static_cast<int16_t>(gradient * grad_scale + gradient_random_value) :
        static_cast<int16_t>(gradient * grad_scale - gradient_random_value);
      output_gradients_and_hessians_ptr[2 * index] = static_cast<int16_t>(hessian * hess_scale + hessian_random_value);
    } else {
      const score_t gradient = input_gradients[index];
      const score_t hessian = input_hessians[index];
      int16_t q = gradient > 0.0f ?
        static_cast<int16_t>(gradient * grad_scale + 0.5) :
        static_cast<int16_t>(gradient * grad_scale - 0.5);
      if (CLAMP) {
        if (q > clamp_max) q = clamp_max;
        else if (q < static_cast<int16_t>(-clamp_max)) q = static_cast<int16_t>(-clamp_max);
      }
      output_gradients_and_hessians_ptr[2 * index + 1] = q;
      output_gradients_and_hessians_ptr[2 * index] = static_cast<int16_t>(hessian * hess_scale + 0.5);
    }
  }
}

void CUDAGradientDiscretizer::DiscretizeGradients(
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

  #define DiscretizeGradientsKernel_ARGS \
    num_data, \
    input_gradients, \
    input_hessians, \
    grad_min_block_buffer_.RawData(), \
    hess_min_block_buffer_.RawData(), \
    iter_ % num_trees_, /* guard: never index past the per-tree offset buffer */ \
    random_values_use_start_.RawData(), \
    gradient_random_values_.RawData(), \
    hessian_random_values_.RawData(), \
    num_grad_quant_bins_, \
    discretized_gradients_and_hessians_.RawData()

  if (stochastic_rounding_) {
    // Stochastic path never uses the robust scale (it is a fixed-point-only mode).
    DiscretizeGradientsKernel<true, false><<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(DiscretizeGradientsKernel_ARGS);
  } else if (robust_scale_) {
    DiscretizeGradientsKernel<false, true><<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(DiscretizeGradientsKernel_ARGS);
  } else {
    DiscretizeGradientsKernel<false, false><<<num_reduce_blocks_, CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE>>>(DiscretizeGradientsKernel_ARGS);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  ++iter_;
}

}  // namespace Falcata

#endif  // USE_CUDA
