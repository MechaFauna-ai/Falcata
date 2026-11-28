/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <Falcata/falcata_plan.h>

#include <algorithm>

#include "cuda_multiclass_objective.hpp"

#include "../../treelearner/cuda/cuda_pdl_launch.hpp"

namespace Falcata {

// Softmax over a row's num_class values in 2 global scratch passes over cuda_softmax_buffer (1 write + 1 read) plus 2
// reads of cuda_scores, down from 7 scratch accesses + 1 score read: pass 1 reads cuda_scores directly for wmax, pass
// 2 reads cuda_scores and writes exp(s - wmax) once to scratch while accumulating wsum in class order, and pass 3
// reads scratch once to divide by wsum and write grad/hess with label and weight hoisted outside the class loop.
// Class-major layout keeps consecutive rows coalesced across a warp, and double math in class order preserves
// bit-identity with the CPU objective and the register kernel.
template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_MulticlassSoftmax(
  const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights,
  const double factor, const int num_class, const data_size_t num_data,
  double* cuda_softmax_buffer, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    const size_t stride = static_cast<size_t>(num_data);
    double* softmax_result = cuda_softmax_buffer + data_index;
    double wmax = cuda_scores[data_index];
    for (int k = 1; k < num_class; ++k) {
      wmax = max(cuda_scores[stride * k + data_index], wmax);
    }
    double wsum = 0.0f;
    for (int k = 0; k < num_class; ++k) {
      const double e = exp(cuda_scores[stride * k + data_index] - wmax);
      softmax_result[k * stride] = e;
      wsum += e;
    }
    const int label = static_cast<int>(cuda_labels[data_index]);
    const double weight = USE_WEIGHT ? static_cast<double>(cuda_weights[data_index]) : 1.0;
    for (int k = 0; k < num_class; ++k) {
      const double p = softmax_result[k * stride] / static_cast<double>(wsum);
      const size_t idx = stride * k + data_index;
      if (!USE_WEIGHT) {
        cuda_out_gradients[idx] = static_cast<score_t>(label == k ? p - 1.0f : p);
        cuda_out_hessians[idx] = static_cast<score_t>(factor * p * (1.0f - p));
      } else {
        cuda_out_gradients[idx] = static_cast<score_t>(label == k ? (p - 1.0f) * weight : p * weight);
        cuda_out_hessians[idx] = static_cast<score_t>((factor * p * (1.0f - p)) * weight);
      }
    }
  }
}

// MAX_CLASS=32 holds 32 doubles (64 32-bit registers) per thread; at __launch_bounds__(1024) the 64-register cap forced
// 120 B of local-memory stack spills per thread. Using 256-thread blocks for MAX_CLASS >= 32 lifts the cap so the array
// stays in registers with zero spills.
template <int MAX_CLASS>
constexpr int SoftmaxRegistersBlockSize() {
  return MAX_CLASS >= 32 ? 256 : GET_GRADIENTS_BLOCK_SIZE_MULTICLASS;
}

// cuda_plan key softmax_registers: GetGradientsKernel_MulticlassSoftmax with the row's class scores in registers
// instead of the global scratch. MAX_CLASS is a capacity: classes k >= num_class are never read, and the loops
// (unrolled over the capacity) run the softmax operations in class order: the max from class 0 up, the exponentials
// summed from 0.0 in class order, one division per class. The same doubles: bit-identical.
template <bool USE_WEIGHT, int MAX_CLASS>
__global__ void __launch_bounds__(SoftmaxRegistersBlockSize<MAX_CLASS>()) GetGradientsKernel_MulticlassSoftmaxRegisters(
  const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights,
  const double factor, const int num_class, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index >= num_data) {
    return;
  }
  double p[MAX_CLASS];
#pragma unroll
  for (int k = 0; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      p[k] = cuda_scores[static_cast<size_t>(num_data) * k + data_index];
    }
  }
  double wmax = p[0];
#pragma unroll
  for (int k = 1; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      wmax = max(p[k], wmax);
    }
  }
  double wsum = 0.0f;
#pragma unroll
  for (int k = 0; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      p[k] = exp(p[k] - wmax);
      wsum += p[k];
    }
  }
  const int label = static_cast<int>(cuda_labels[data_index]);
  const double weight = USE_WEIGHT ? static_cast<double>(cuda_weights[data_index]) : 1.0;
#pragma unroll
  for (int k = 0; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      const double pk = p[k] / static_cast<double>(wsum);
      const size_t idx = static_cast<size_t>(num_data) * k + data_index;
      if (!USE_WEIGHT) {
        cuda_out_gradients[idx] = static_cast<score_t>(label == k ? pk - 1.0f : pk);
        cuda_out_hessians[idx] = static_cast<score_t>(factor * pk * (1.0f - pk));
      } else {
        cuda_out_gradients[idx] = static_cast<score_t>(label == k ? (pk - 1.0f) * weight : pk * weight);
        cuda_out_hessians[idx] = static_cast<score_t>((factor * pk * (1.0f - pk)) * weight);
      }
    }
  }
}

template <bool USE_WEIGHT, int MAX_CLASS>
static bool LaunchMulticlassSoftmaxRegisters(const double* scores, const label_t* labels,
                                             const label_t* weights, const double factor, const int num_class,
                                             const data_size_t num_data, score_t* gradients, score_t* hessians) {
  if (num_class > MAX_CLASS) {
    return false;
  }
  constexpr int kBlockSize = SoftmaxRegistersBlockSize<MAX_CLASS>();
  const int num_blocks = CUDABlockCount(num_data, kBlockSize);
  GetGradientsKernel_MulticlassSoftmaxRegisters<USE_WEIGHT, MAX_CLASS><<<num_blocks, kBlockSize>>>(
    scores, labels, weights, factor, num_class, num_data, gradients, hessians);
  return true;
}

void CUDAMulticlassSoftmax::LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const {
  if (num_data_ <= 0) {
    return;
  }
  if (FalcataPlan::Get().softmax_registers) {
    // class capacities 8 / 16 (1024 threads) and 32 (256 threads, zero spills); wider class counts take the 2-pass scratch kernel
    const bool launched = cuda_weights_ == nullptr ?
      (LaunchMulticlassSoftmaxRegisters<false, 8>(scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                  num_data_, gradients, hessians) ||
       LaunchMulticlassSoftmaxRegisters<false, 16>(scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                   num_data_, gradients, hessians) ||
       LaunchMulticlassSoftmaxRegisters<false, 32>(scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                   num_data_, gradients, hessians)) :
      (LaunchMulticlassSoftmaxRegisters<true, 8>(scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                 num_data_, gradients, hessians) ||
       LaunchMulticlassSoftmaxRegisters<true, 16>(scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                  num_data_, gradients, hessians) ||
       LaunchMulticlassSoftmaxRegisters<true, 32>(scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                  num_data_, gradients, hessians));
    if (launched) {
      return;
    }
  }
  double* softmax_buffer = EnsureSoftmaxBuffer(num_data_);
  const int num_blocks = CUDABlockCount(num_data_, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS);
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_MulticlassSoftmax<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
      scores, cuda_labels_, cuda_weights_, factor_, num_class_, num_data_,
      softmax_buffer, gradients, hessians);
  } else {
    GetGradientsKernel_MulticlassSoftmax<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
      scores, cuda_labels_, cuda_weights_, factor_, num_class_, num_data_,
      softmax_buffer, gradients, hessians);
  }
}

__global__ void ConvertOutputCUDAKernel_MulticlassSoftmax(
  const int num_class, const data_size_t num_data, const double* input, double* cuda_softmax_buffer, double* output) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    // 2-pass class-major scratch (aliasing-safe even if input == output)
    const size_t stride = static_cast<size_t>(num_data);
    double* softmax_ptr = cuda_softmax_buffer + data_index;
    double wmax = input[data_index];
    for (int class_index = 1; class_index < num_class; ++class_index) {
      wmax = max(input[stride * class_index + data_index], wmax);
    }
    double wsum = 0.0f;
    for (int class_index = 0; class_index < num_class; ++class_index) {
      const double e = exp(input[stride * class_index + data_index] - wmax);
      softmax_ptr[class_index * stride] = e;
      wsum += e;
    }
    for (int class_index = 0; class_index < num_class; ++class_index) {
      output[stride * class_index + data_index] = softmax_ptr[class_index * stride] / static_cast<double>(wsum);
    }
  }
}

const double* CUDAMulticlassSoftmax::LaunchConvertOutputCUDAKernel(
  const data_size_t num_data, const double* input, double* output) const {
  if (num_data <= 0) {
    return output;
  }
  double* softmax_buffer = EnsureSoftmaxBuffer(num_data);
  const int num_blocks = CUDABlockCount(num_data, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS);
  ConvertOutputCUDAKernel_MulticlassSoftmax<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
    num_class_, num_data, input, softmax_buffer, output);
  return output;
}

}  // namespace Falcata

#endif  // USE_CUDA
