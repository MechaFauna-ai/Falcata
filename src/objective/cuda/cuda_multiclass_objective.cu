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

__device__ void SoftmaxCUDA(double* softmax_buffer, int len) {
  double wmax = softmax_buffer[0];
  for (int i = 1; i < len; ++i) {
    wmax = max(softmax_buffer[i], wmax);
  }
  double wsum = 0.0f;
  for (int i = 0; i < len; ++i) {
    softmax_buffer[i] = exp(softmax_buffer[i] - wmax);
    wsum += softmax_buffer[i];
  }
  for (int i = 0; i < len; ++i) {
    softmax_buffer[i] /= static_cast<double>(wsum);
  }
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_MulticlassSoftmax(
  const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights,
  const double factor, const int num_class, const data_size_t num_data,
  double* cuda_softmax_buffer, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    const size_t offset = static_cast<size_t>(data_index) * num_class;
    double* softmax_result = cuda_softmax_buffer + offset;
    for (int k = 0; k < num_class; ++k) {
      softmax_result[k] = cuda_scores[static_cast<size_t>(num_data) * k + data_index];
    }
    SoftmaxCUDA(softmax_result, num_class);
    if (!USE_WEIGHT) {
      for (int k = 0; k < num_class; ++k) {
        const double p = softmax_result[k];
        size_t idx = static_cast<size_t>(num_data) * k + data_index;
        if (static_cast<int>(cuda_labels[data_index]) == k) {
          cuda_out_gradients[idx] = static_cast<score_t>(p - 1.0f);
        } else {
          cuda_out_gradients[idx] = static_cast<score_t>(p);
        }
        cuda_out_hessians[idx] = static_cast<score_t>(factor * p * (1.0f - p));
      }
    } else {
      for (int k = 0; k < num_class; ++k) {
        const double p = softmax_result[k];
        const double weight = cuda_weights[data_index];
        size_t idx = static_cast<size_t>(num_data) * k + data_index;
        if (static_cast<int>(cuda_labels[data_index]) == k) {
          cuda_out_gradients[idx] = static_cast<score_t>((p - 1.0f) * weight);
        } else {
          cuda_out_gradients[idx] = static_cast<score_t>(p * weight);
        }
        cuda_out_hessians[idx] = static_cast<score_t>((factor * p * (1.0f - p)) * weight);
      }
    }
  }
}

// cuda_plan key softmax_registers: GetGradientsKernel_MulticlassSoftmax with the row's class scores in registers
// instead of the row-major global scratch (num_class doubles per thread, 6 passes over it). MAX_CLASS is a capacity:
// classes k >= num_class are never read, and the loops (unrolled over the capacity) run SoftmaxCUDA's operations in
// its order: the max from class 0 up, the exponentials summed from 0.0 in class order, one division per class. The
// same doubles: bit-identical.
template <bool USE_WEIGHT, int MAX_CLASS>
__global__ void __launch_bounds__(GET_GRADIENTS_BLOCK_SIZE_MULTICLASS) GetGradientsKernel_MulticlassSoftmaxRegisters(
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
static bool LaunchMulticlassSoftmaxRegisters(const int num_blocks, const double* scores, const label_t* labels,
                                             const label_t* weights, const double factor, const int num_class,
                                             const data_size_t num_data, score_t* gradients, score_t* hessians) {
  if (num_class > MAX_CLASS) {
    return false;
  }
  GetGradientsKernel_MulticlassSoftmaxRegisters<USE_WEIGHT, MAX_CLASS><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
    scores, labels, weights, factor, num_class, num_data, gradients, hessians);
  return true;
}

void CUDAMulticlassSoftmax::LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_MULTICLASS - 1) / GET_GRADIENTS_BLOCK_SIZE_MULTICLASS;
  if (FalcataPlan::Get().softmax_registers) {
    // class capacities 8 / 16 / 32 (at 32 the 64-register cap of __launch_bounds__(1024) spills part of the array to
    // thread-interleaved local memory, still coalesced); wider class counts keep the scratch kernel
    const bool launched = cuda_weights_ == nullptr ?
      (LaunchMulticlassSoftmaxRegisters<false, 8>(num_blocks, scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                  num_data_, gradients, hessians) ||
       LaunchMulticlassSoftmaxRegisters<false, 16>(num_blocks, scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                   num_data_, gradients, hessians) ||
       LaunchMulticlassSoftmaxRegisters<false, 32>(num_blocks, scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                   num_data_, gradients, hessians)) :
      (LaunchMulticlassSoftmaxRegisters<true, 8>(num_blocks, scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                 num_data_, gradients, hessians) ||
       LaunchMulticlassSoftmaxRegisters<true, 16>(num_blocks, scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                  num_data_, gradients, hessians) ||
       LaunchMulticlassSoftmaxRegisters<true, 32>(num_blocks, scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                  num_data_, gradients, hessians));
    if (launched) {
      return;
    }
  }
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_MulticlassSoftmax<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
      scores, cuda_labels_, cuda_weights_, factor_, num_class_, num_data_,
      cuda_softmax_buffer_.RawData(), gradients, hessians);
  } else {
    GetGradientsKernel_MulticlassSoftmax<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
      scores, cuda_labels_, cuda_weights_, factor_, num_class_, num_data_,
      cuda_softmax_buffer_.RawData(), gradients, hessians);
  }
}

__global__ void ConvertOutputCUDAKernel_MulticlassSoftmax(
  const int num_class, const data_size_t num_data, const double* input, double* cuda_softmax_buffer, double* output) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    const size_t offset = static_cast<size_t>(data_index) * num_class;
    double* cuda_softmax_buffer_ptr = cuda_softmax_buffer + offset;
    for (int class_index = 0; class_index < num_class; ++class_index) {
      cuda_softmax_buffer_ptr[class_index] = input[static_cast<size_t>(num_data) * class_index + data_index];
    }
    SoftmaxCUDA(cuda_softmax_buffer_ptr, num_class);
    for (int class_index = 0; class_index < num_class; ++class_index) {
      output[static_cast<size_t>(num_data) * class_index + data_index] = cuda_softmax_buffer_ptr[class_index];
    }
  }
}

const double* CUDAMulticlassSoftmax::LaunchConvertOutputCUDAKernel(
  const data_size_t num_data, const double* input, double* output) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_MULTICLASS - 1) / GET_GRADIENTS_BLOCK_SIZE_MULTICLASS;
  ConvertOutputCUDAKernel_MulticlassSoftmax<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
    num_class_, num_data, input, cuda_softmax_buffer_.RawData(), output);
  return output;
}

}  // namespace Falcata

#endif  // USE_CUDA
