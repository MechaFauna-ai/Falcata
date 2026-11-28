/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_algorithms.hpp>
#include <Falcata/cuda/cuda_rocm_interop.h>

#include "cuda_multiclass_metric.hpp"

namespace Falcata {

namespace {

// Matches std::max(val, cur) on CPU: (val < cur) ? cur : val.
__device__ __forceinline__ double StdMaxDouble(double val, double cur) {
  return (val < cur) ? cur : val;
}

__global__ void MulticlassReduceBlockSumKernel(const double* block_buffer, const int num_blocks, double* out_sum) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  double thread_sum = 0.0;
  for (int block_index = static_cast<int>(threadIdx.x); block_index < num_blocks;
       block_index += static_cast<int>(blockDim.x)) {
    thread_sum += block_buffer[block_index];
  }
  const double total = ShuffleReduceSum<double>(thread_sum, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    out_sum[0] = total;
  }
}

void FinalizeMulticlassReduction(const int num_blocks, double* reduce_block_buffer,
                                 double* reduce_block_buffer_inner, double* sum_loss) {
  if (num_blocks == 1) {
    CopyFromCUDADeviceToHost<double>(sum_loss, reduce_block_buffer, 1, __FILE__, __LINE__);
  } else {
    MulticlassReduceBlockSumKernel<<<1, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      reduce_block_buffer, num_blocks, reduce_block_buffer_inner);
    CopyFromCUDADeviceToHost<double>(sum_loss, reduce_block_buffer_inner, 1, __FILE__, __LINE__);
  }
}

template <bool USE_WEIGHTS, int MAX_CLASS>
__global__ void __launch_bounds__(MULTICLASS_EVAL_BLOCK_SIZE) EvalMultiLoglossSoftmaxRegistersKernel(
    const data_size_t num_data, const int num_class, const label_t* labels, const label_t* weights,
    const double* scores, double* reduce_block_buffer) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_metric = 0.0;
  if (index < num_data) {
    const size_t stride = static_cast<size_t>(num_data);
    double s[MAX_CLASS];
    #pragma unroll
    for (int k = 0; k < MAX_CLASS; ++k) {
      if (k < num_class) {
        s[k] = scores[static_cast<size_t>(k) * stride + static_cast<size_t>(index)];
      }
    }
    double wmax = s[0];
    #pragma unroll
    for (int k = 1; k < MAX_CLASS; ++k) {
      if (k < num_class) {
        wmax = StdMaxDouble(s[k], wmax);
      }
    }
    const int true_class = static_cast<int>(labels[index]);
    double wsum = 0.0;
    double exp_true = 0.0;
    #pragma unroll
    for (int k = 0; k < MAX_CLASS; ++k) {
      if (k < num_class) {
        const double e = exp(s[k] - wmax);
        wsum += e;
        if (k == true_class) {
          exp_true = e;
        }
      }
    }
    const double prob = exp_true / wsum;
    point_metric = (prob > kEpsilon) ? -log(prob) : -log(kEpsilon);
    if (USE_WEIGHTS) {
      point_metric *= static_cast<double>(weights[index]);
    }
  }
  const double block_sum = ShuffleReduceSum<double>(point_metric, shared_mem_buffer, MULTICLASS_EVAL_BLOCK_SIZE);
  if (threadIdx.x == 0) {
    reduce_block_buffer[blockIdx.x] = block_sum;
  }
}

template <bool USE_WEIGHTS>
__global__ void __launch_bounds__(MULTICLASS_EVAL_BLOCK_SIZE) EvalMultiLoglossSoftmaxStreamingKernel(
    const data_size_t num_data, const int num_class, const label_t* labels, const label_t* weights,
    const double* scores, double* reduce_block_buffer) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_metric = 0.0;
  if (index < num_data) {
    const size_t stride = static_cast<size_t>(num_data);
    double wmax = scores[static_cast<size_t>(index)];
    for (int k = 1; k < num_class; ++k) {
      wmax = StdMaxDouble(scores[static_cast<size_t>(k) * stride + static_cast<size_t>(index)], wmax);
    }
    const int true_class = static_cast<int>(labels[index]);
    double wsum = 0.0;
    double exp_true = 0.0;
    for (int k = 0; k < num_class; ++k) {
      const double e = exp(scores[static_cast<size_t>(k) * stride + static_cast<size_t>(index)] - wmax);
      wsum += e;
      if (k == true_class) {
        exp_true = e;
      }
    }
    const double prob = exp_true / wsum;
    point_metric = (prob > kEpsilon) ? -log(prob) : -log(kEpsilon);
    if (USE_WEIGHTS) {
      point_metric *= static_cast<double>(weights[index]);
    }
  }
  const double block_sum = ShuffleReduceSum<double>(point_metric, shared_mem_buffer, MULTICLASS_EVAL_BLOCK_SIZE);
  if (threadIdx.x == 0) {
    reduce_block_buffer[blockIdx.x] = block_sum;
  }
}

template <bool USE_WEIGHTS, bool USE_SIGMOID>
__global__ void __launch_bounds__(MULTICLASS_EVAL_BLOCK_SIZE) EvalMultiLoglossPointKernel(
    const data_size_t num_data, const label_t* labels, const label_t* weights,
    const double* scores, const double sigmoid, double* reduce_block_buffer) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_metric = 0.0;
  if (index < num_data) {
    const size_t stride = static_cast<size_t>(num_data);
    const int true_class = static_cast<int>(labels[index]);
    double prob = scores[static_cast<size_t>(true_class) * stride + static_cast<size_t>(index)];
    if (USE_SIGMOID) {
      prob = 1.0f / (1.0f + exp(-sigmoid * prob));
    }
    point_metric = (prob > kEpsilon) ? -log(prob) : -log(kEpsilon);
    if (USE_WEIGHTS) {
      point_metric *= static_cast<double>(weights[index]);
    }
  }
  const double block_sum = ShuffleReduceSum<double>(point_metric, shared_mem_buffer, MULTICLASS_EVAL_BLOCK_SIZE);
  if (threadIdx.x == 0) {
    reduce_block_buffer[blockIdx.x] = block_sum;
  }
}

template <bool USE_WEIGHTS, bool USE_SIGMOID>
__global__ void __launch_bounds__(MULTICLASS_EVAL_BLOCK_SIZE) EvalMultiErrorRawKernel(
    const data_size_t num_data, const int num_class, const int top_k,
    const label_t* labels, const label_t* weights, const double* scores,
    const double sigmoid, double* reduce_block_buffer) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_metric = 0.0;
  if (index < num_data) {
    const size_t stride = static_cast<size_t>(num_data);
    const int true_class = static_cast<int>(labels[index]);
    double true_val = scores[static_cast<size_t>(true_class) * stride + static_cast<size_t>(index)];
    if (USE_SIGMOID) {
      true_val = 1.0f / (1.0f + exp(-sigmoid * true_val));
    }
    int num_larger = 0;
    double row_loss = 0.0;
    for (int k = 0; k < num_class; ++k) {
      double val = scores[static_cast<size_t>(k) * stride + static_cast<size_t>(index)];
      if (USE_SIGMOID) {
        val = 1.0f / (1.0f + exp(-sigmoid * val));
      }
      if (val >= true_val) {
        ++num_larger;
        if (num_larger > top_k) {
          row_loss = 1.0;
          break;
        }
      }
    }
    point_metric = USE_WEIGHTS ? (row_loss * static_cast<double>(weights[index])) : row_loss;
  }
  const double block_sum = ShuffleReduceSum<double>(point_metric, shared_mem_buffer, MULTICLASS_EVAL_BLOCK_SIZE);
  if (threadIdx.x == 0) {
    reduce_block_buffer[blockIdx.x] = block_sum;
  }
}

template <bool USE_WEIGHTS, int MAX_CLASS>
__global__ void __launch_bounds__(MULTICLASS_EVAL_BLOCK_SIZE) EvalMultiErrorSoftmaxRegistersKernel(
    const data_size_t num_data, const int num_class, const int top_k,
    const label_t* labels, const label_t* weights, const double* scores,
    double* reduce_block_buffer) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_metric = 0.0;
  if (index < num_data) {
    const size_t stride = static_cast<size_t>(num_data);
    double p[MAX_CLASS];
    #pragma unroll
    for (int k = 0; k < MAX_CLASS; ++k) {
      if (k < num_class) {
        p[k] = scores[static_cast<size_t>(k) * stride + static_cast<size_t>(index)];
      }
    }
    double wmax = p[0];
    #pragma unroll
    for (int k = 1; k < MAX_CLASS; ++k) {
      if (k < num_class) {
        wmax = StdMaxDouble(p[k], wmax);
      }
    }
    double wsum = 0.0;
    #pragma unroll
    for (int k = 0; k < MAX_CLASS; ++k) {
      if (k < num_class) {
        p[k] = exp(p[k] - wmax);
        wsum += p[k];
      }
    }
    const int true_class = static_cast<int>(labels[index]);
    double true_prob = 0.0;
    #pragma unroll
    for (int k = 0; k < MAX_CLASS; ++k) {
      if (k < num_class) {
        p[k] /= wsum;
        if (k == true_class) {
          true_prob = p[k];
        }
      }
    }
    int num_larger = 0;
    #pragma unroll
    for (int k = 0; k < MAX_CLASS; ++k) {
      if (k < num_class && p[k] >= true_prob) {
        ++num_larger;
      }
    }
    const double row_loss = (num_larger > top_k) ? 1.0 : 0.0;
    point_metric = USE_WEIGHTS ? (row_loss * static_cast<double>(weights[index])) : row_loss;
  }
  const double block_sum = ShuffleReduceSum<double>(point_metric, shared_mem_buffer, MULTICLASS_EVAL_BLOCK_SIZE);
  if (threadIdx.x == 0) {
    reduce_block_buffer[blockIdx.x] = block_sum;
  }
}

template <bool USE_WEIGHTS>
__global__ void __launch_bounds__(MULTICLASS_EVAL_BLOCK_SIZE) EvalMultiErrorSoftmaxStreamingKernel(
    const data_size_t num_data, const int num_class, const int top_k,
    const label_t* labels, const label_t* weights, const double* scores,
    double* reduce_block_buffer) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_metric = 0.0;
  if (index < num_data) {
    const size_t stride = static_cast<size_t>(num_data);
    double wmax = scores[static_cast<size_t>(index)];
    for (int k = 1; k < num_class; ++k) {
      wmax = StdMaxDouble(scores[static_cast<size_t>(k) * stride + static_cast<size_t>(index)], wmax);
    }
    const int true_class = static_cast<int>(labels[index]);
    double wsum = 0.0;
    double exp_true = 0.0;
    for (int k = 0; k < num_class; ++k) {
      const double e = exp(scores[static_cast<size_t>(k) * stride + static_cast<size_t>(index)] - wmax);
      wsum += e;
      if (k == true_class) {
        exp_true = e;
      }
    }
    const double true_prob = exp_true / wsum;
    int num_larger = 0;
    double row_loss = 0.0;
    for (int k = 0; k < num_class; ++k) {
      const double pk = exp(scores[static_cast<size_t>(k) * stride + static_cast<size_t>(index)] - wmax) / wsum;
      if (pk >= true_prob) {
        ++num_larger;
        if (num_larger > top_k) {
          row_loss = 1.0;
          break;
        }
      }
    }
    point_metric = USE_WEIGHTS ? (row_loss * static_cast<double>(weights[index])) : row_loss;
  }
  const double block_sum = ShuffleReduceSum<double>(point_metric, shared_mem_buffer, MULTICLASS_EVAL_BLOCK_SIZE);
  if (threadIdx.x == 0) {
    reduce_block_buffer[blockIdx.x] = block_sum;
  }
}

template <bool USE_WEIGHTS>
__global__ void __launch_bounds__(MULTICLASS_EVAL_BLOCK_SIZE) EvalMultiRMSEKernel(
    const data_size_t num_data, const int num_class, const label_t* labels, const label_t* weights,
    const double* scores, double* reduce_block_buffer) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_metric = 0.0;
  if (index < num_data) {
    const size_t stride = static_cast<size_t>(num_data);
    const double weight = USE_WEIGHTS ? static_cast<double>(weights[index]) : 1.0;
    double row_loss = 0.0;
    for (int k = 0; k < num_class; ++k) {
      const size_t idx = static_cast<size_t>(k) * stride + static_cast<size_t>(index);
      const double d = scores[idx] - static_cast<double>(labels[idx]);
      row_loss += USE_WEIGHTS ? (d * d * weight) : (d * d);
    }
    point_metric = row_loss;
  }
  const double block_sum = ShuffleReduceSum<double>(point_metric, shared_mem_buffer, MULTICLASS_EVAL_BLOCK_SIZE);
  if (threadIdx.x == 0) {
    reduce_block_buffer[blockIdx.x] = block_sum;
  }
}

template <bool USE_WEIGHTS>
void DispatchMultiLoglossSoftmax(const int num_blocks, const data_size_t num_data, const int num_class,
                                 const label_t* labels, const label_t* weights, const double* scores,
                                 double* reduce_block_buffer) {
  if (num_class <= 8) {
    EvalMultiLoglossSoftmaxRegistersKernel<USE_WEIGHTS, 8><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      num_data, num_class, labels, weights, scores, reduce_block_buffer);
  } else if (num_class <= 16) {
    EvalMultiLoglossSoftmaxRegistersKernel<USE_WEIGHTS, 16><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      num_data, num_class, labels, weights, scores, reduce_block_buffer);
  } else if (num_class <= 32) {
    EvalMultiLoglossSoftmaxRegistersKernel<USE_WEIGHTS, 32><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      num_data, num_class, labels, weights, scores, reduce_block_buffer);
  } else {
    EvalMultiLoglossSoftmaxStreamingKernel<USE_WEIGHTS><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      num_data, num_class, labels, weights, scores, reduce_block_buffer);
  }
}

template <bool USE_WEIGHTS>
void DispatchMultiErrorSoftmax(const int num_blocks, const data_size_t num_data, const int num_class,
                               const int top_k, const label_t* labels, const label_t* weights,
                               const double* scores, double* reduce_block_buffer) {
  if (num_class <= 8) {
    EvalMultiErrorSoftmaxRegistersKernel<USE_WEIGHTS, 8><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      num_data, num_class, top_k, labels, weights, scores, reduce_block_buffer);
  } else if (num_class <= 16) {
    EvalMultiErrorSoftmaxRegistersKernel<USE_WEIGHTS, 16><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      num_data, num_class, top_k, labels, weights, scores, reduce_block_buffer);
  } else if (num_class <= 32) {
    EvalMultiErrorSoftmaxRegistersKernel<USE_WEIGHTS, 32><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      num_data, num_class, top_k, labels, weights, scores, reduce_block_buffer);
  } else {
    EvalMultiErrorSoftmaxStreamingKernel<USE_WEIGHTS><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      num_data, num_class, top_k, labels, weights, scores, reduce_block_buffer);
  }
}

}  // namespace

void CUDAMultiSoftmaxLoglossMetric::LaunchEvalSoftmaxKernel(const double* score, double* sum_loss) const {
  if (this->num_data_ <= 0) {
    *sum_loss = 0.0;
    return;
  }
  const int num_blocks = CUDABlockCount(this->num_data_, MULTICLASS_EVAL_BLOCK_SIZE);
  if (this->cuda_weights_ != nullptr) {
    DispatchMultiLoglossSoftmax<true>(
      num_blocks, this->num_data_, this->num_class_, this->cuda_labels_, this->cuda_weights_,
      score, reduce_block_buffer_.RawData());
  } else {
    DispatchMultiLoglossSoftmax<false>(
      num_blocks, this->num_data_, this->num_class_, this->cuda_labels_, this->cuda_weights_,
      score, reduce_block_buffer_.RawData());
  }
  FinalizeMulticlassReduction(num_blocks, reduce_block_buffer_.RawData(), reduce_block_buffer_inner_.RawData(), sum_loss);
}

void CUDAMultiSoftmaxLoglossMetric::LaunchEvalSigmoidKernel(const double* score, double sigmoid, double* sum_loss) const {
  if (this->num_data_ <= 0) {
    *sum_loss = 0.0;
    return;
  }
  const int num_blocks = CUDABlockCount(this->num_data_, MULTICLASS_EVAL_BLOCK_SIZE);
  if (this->cuda_weights_ != nullptr) {
    EvalMultiLoglossPointKernel<true, true><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      this->num_data_, this->cuda_labels_, this->cuda_weights_, score, sigmoid, reduce_block_buffer_.RawData());
  } else {
    EvalMultiLoglossPointKernel<false, true><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      this->num_data_, this->cuda_labels_, this->cuda_weights_, score, sigmoid, reduce_block_buffer_.RawData());
  }
  FinalizeMulticlassReduction(num_blocks, reduce_block_buffer_.RawData(), reduce_block_buffer_inner_.RawData(), sum_loss);
}

void CUDAMultiSoftmaxLoglossMetric::LaunchEvalIdentityKernel(const double* score, double* sum_loss) const {
  if (this->num_data_ <= 0) {
    *sum_loss = 0.0;
    return;
  }
  const int num_blocks = CUDABlockCount(this->num_data_, MULTICLASS_EVAL_BLOCK_SIZE);
  if (this->cuda_weights_ != nullptr) {
    EvalMultiLoglossPointKernel<true, false><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      this->num_data_, this->cuda_labels_, this->cuda_weights_, score, 0.0, reduce_block_buffer_.RawData());
  } else {
    EvalMultiLoglossPointKernel<false, false><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      this->num_data_, this->cuda_labels_, this->cuda_weights_, score, 0.0, reduce_block_buffer_.RawData());
  }
  FinalizeMulticlassReduction(num_blocks, reduce_block_buffer_.RawData(), reduce_block_buffer_inner_.RawData(), sum_loss);
}

void CUDAMultiErrorMetric::LaunchEvalIdentityKernel(const double* score, double* sum_loss) const {
  if (this->num_data_ <= 0) {
    *sum_loss = 0.0;
    return;
  }
  const int num_blocks = CUDABlockCount(this->num_data_, MULTICLASS_EVAL_BLOCK_SIZE);
  if (this->cuda_weights_ != nullptr) {
    EvalMultiErrorRawKernel<true, false><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      this->num_data_, this->num_class_, multi_error_top_k_, this->cuda_labels_, this->cuda_weights_,
      score, 0.0, reduce_block_buffer_.RawData());
  } else {
    EvalMultiErrorRawKernel<false, false><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      this->num_data_, this->num_class_, multi_error_top_k_, this->cuda_labels_, this->cuda_weights_,
      score, 0.0, reduce_block_buffer_.RawData());
  }
  FinalizeMulticlassReduction(num_blocks, reduce_block_buffer_.RawData(), reduce_block_buffer_inner_.RawData(), sum_loss);
}

void CUDAMultiErrorMetric::LaunchEvalSoftmaxTopKKernel(const double* score, double* sum_loss) const {
  if (this->num_data_ <= 0) {
    *sum_loss = 0.0;
    return;
  }
  const int num_blocks = CUDABlockCount(this->num_data_, MULTICLASS_EVAL_BLOCK_SIZE);
  if (this->cuda_weights_ != nullptr) {
    DispatchMultiErrorSoftmax<true>(
      num_blocks, this->num_data_, this->num_class_, multi_error_top_k_, this->cuda_labels_,
      this->cuda_weights_, score, reduce_block_buffer_.RawData());
  } else {
    DispatchMultiErrorSoftmax<false>(
      num_blocks, this->num_data_, this->num_class_, multi_error_top_k_, this->cuda_labels_,
      this->cuda_weights_, score, reduce_block_buffer_.RawData());
  }
  FinalizeMulticlassReduction(num_blocks, reduce_block_buffer_.RawData(), reduce_block_buffer_inner_.RawData(), sum_loss);
}

void CUDAMultiErrorMetric::LaunchEvalSigmoidKernel(const double* score, double sigmoid, double* sum_loss) const {
  if (this->num_data_ <= 0) {
    *sum_loss = 0.0;
    return;
  }
  const int num_blocks = CUDABlockCount(this->num_data_, MULTICLASS_EVAL_BLOCK_SIZE);
  if (this->cuda_weights_ != nullptr) {
    EvalMultiErrorRawKernel<true, true><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      this->num_data_, this->num_class_, multi_error_top_k_, this->cuda_labels_, this->cuda_weights_,
      score, sigmoid, reduce_block_buffer_.RawData());
  } else {
    EvalMultiErrorRawKernel<false, true><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      this->num_data_, this->num_class_, multi_error_top_k_, this->cuda_labels_, this->cuda_weights_,
      score, sigmoid, reduce_block_buffer_.RawData());
  }
  FinalizeMulticlassReduction(num_blocks, reduce_block_buffer_.RawData(), reduce_block_buffer_inner_.RawData(), sum_loss);
}

void CUDAMultiRMSEMetric::LaunchEvalKernel(const double* score, double* sum_loss) const {
  if (this->num_data_ <= 0) {
    *sum_loss = 0.0;
    return;
  }
  const int num_blocks = CUDABlockCount(this->num_data_, MULTICLASS_EVAL_BLOCK_SIZE);
  if (this->cuda_weights_ != nullptr) {
    EvalMultiRMSEKernel<true><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      this->num_data_, this->num_class_, this->cuda_labels_, this->cuda_weights_,
      score, reduce_block_buffer_.RawData());
  } else {
    EvalMultiRMSEKernel<false><<<num_blocks, MULTICLASS_EVAL_BLOCK_SIZE>>>(
      this->num_data_, this->num_class_, this->cuda_labels_, this->cuda_weights_,
      score, reduce_block_buffer_.RawData());
  }
  FinalizeMulticlassReduction(num_blocks, reduce_block_buffer_.RawData(), reduce_block_buffer_inner_.RawData(), sum_loss);
}

}  // namespace Falcata

#endif  // USE_CUDA
