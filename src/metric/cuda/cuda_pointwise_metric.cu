/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_algorithms.hpp>
#include <Falcata/cuda/cuda_rocm_interop.h>

#include "cuda_binary_metric.hpp"
#include "cuda_pointwise_metric.hpp"
#include "cuda_regression_metric.hpp"

namespace Falcata {

namespace {

enum class CUDAPointwiseScoreTransform {
  kNone = 0,
  kSigmoid = 1,
  kExp = 2,
};

__global__ void EvalReduceBlockSumKernel(const double* block_buffer, const int num_blocks, double* out_sum) {
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

template <typename CUDA_METRIC, bool USE_WEIGHTS, CUDAPointwiseScoreTransform TRANSFORM>
__global__ void EvalKernel(const data_size_t num_data, const label_t* labels, const label_t* weights,
                           const double* scores, double* reduce_block_buffer, const double param,
                           const double transform_param) {
  __shared__ double shared_mem_buffer[WARPSIZE];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_metric = 0.0;
  if (index < num_data) {
    double score = scores[index];
    if (TRANSFORM == CUDAPointwiseScoreTransform::kSigmoid) {
      score = 1.0f / (1.0f + exp(-transform_param * score));
    } else if (TRANSFORM == CUDAPointwiseScoreTransform::kExp) {
      score = exp(score);
    }
    point_metric = USE_WEIGHTS ?
      CUDA_METRIC::MetricOnPointCUDA(labels[index], score, param) * weights[index] :
      CUDA_METRIC::MetricOnPointCUDA(labels[index], score, param);
  }
  const double block_sum_point_metric = ShuffleReduceSum<double>(point_metric, shared_mem_buffer, NUM_DATA_PER_EVAL_THREAD);
  if (threadIdx.x == 0) {
    reduce_block_buffer[blockIdx.x] = block_sum_point_metric;
  }
}

template <typename CUDA_METRIC, CUDAPointwiseScoreTransform TRANSFORM>
void LaunchEvalKernelImpl(const data_size_t num_data, const label_t* cuda_labels, const label_t* cuda_weights,
                          const double* scores, const double param, const double transform_param,
                          double* reduce_block_buffer, double* reduce_block_buffer_inner,
                          double* sum_loss) {
  if (num_data <= 0) {
    *sum_loss = 0.0;
    return;
  }
  const int num_blocks = CUDABlockCount(num_data, NUM_DATA_PER_EVAL_THREAD);
  if (cuda_weights != nullptr) {
    EvalKernel<CUDA_METRIC, true, TRANSFORM><<<num_blocks, NUM_DATA_PER_EVAL_THREAD>>>(
      num_data, cuda_labels, cuda_weights, scores, reduce_block_buffer, param, transform_param);
  } else {
    EvalKernel<CUDA_METRIC, false, TRANSFORM><<<num_blocks, NUM_DATA_PER_EVAL_THREAD>>>(
      num_data, cuda_labels, cuda_weights, scores, reduce_block_buffer, param, transform_param);
  }
  if (num_blocks == 1) {
    CopyFromCUDADeviceToHost<double>(sum_loss, reduce_block_buffer, 1, __FILE__, __LINE__);
  } else {
    EvalReduceBlockSumKernel<<<1, NUM_DATA_PER_EVAL_THREAD>>>(
      reduce_block_buffer, num_blocks, reduce_block_buffer_inner);
    CopyFromCUDADeviceToHost<double>(sum_loss, reduce_block_buffer_inner, 1, __FILE__, __LINE__);
  }
}

}  // namespace

template <typename HOST_METRIC, typename CUDA_METRIC>
void CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC>::LaunchEvalKernel(
    const double* score, double* sum_loss, double* sum_weight) const {
  *sum_weight = this->sum_weights_;
  LaunchEvalKernelImpl<CUDA_METRIC, CUDAPointwiseScoreTransform::kNone>(
    this->num_data_, this->cuda_labels_, this->cuda_weights_, score, GetParamFromConfig(), 0.0,
    reduce_block_buffer_.RawData(), reduce_block_buffer_inner_.RawData(), sum_loss);
}

template <typename HOST_METRIC, typename CUDA_METRIC>
void CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC>::LaunchEvalKernelFusedSigmoid(
    const double* raw_score, double sigmoid, double* sum_loss, double* sum_weight) const {
  *sum_weight = this->sum_weights_;
  LaunchEvalKernelImpl<CUDA_METRIC, CUDAPointwiseScoreTransform::kSigmoid>(
    this->num_data_, this->cuda_labels_, this->cuda_weights_, raw_score, GetParamFromConfig(), sigmoid,
    reduce_block_buffer_.RawData(), reduce_block_buffer_inner_.RawData(), sum_loss);
}

template <typename HOST_METRIC, typename CUDA_METRIC>
void CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC>::LaunchEvalKernelFusedExp(
    const double* raw_score, double* sum_loss, double* sum_weight) const {
  *sum_weight = this->sum_weights_;
  LaunchEvalKernelImpl<CUDA_METRIC, CUDAPointwiseScoreTransform::kExp>(
    this->num_data_, this->cuda_labels_, this->cuda_weights_, raw_score, GetParamFromConfig(), 0.0,
    reduce_block_buffer_.RawData(), reduce_block_buffer_inner_.RawData(), sum_loss);
}

// Regression metrics
template void CUDAPointwiseMetricInterface<RMSEMetric, CUDARMSEMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<L2Metric, CUDAL2Metric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<QuantileMetric, CUDAQuantileMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<L1Metric, CUDAL1Metric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<HuberLossMetric, CUDAHuberLossMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<FairLossMetric, CUDAFairLossMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<PoissonMetric, CUDAPoissonMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<PoissonMetric, CUDAPoissonMetric>::LaunchEvalKernelFusedExp(const double* raw_score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<MAPEMetric, CUDAMAPEMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<GammaMetric, CUDAGammaMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<GammaDevianceMetric, CUDAGammaDevianceMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<TweedieMetric, CUDATweedieMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;

// Binary metrics
template void CUDAPointwiseMetricInterface<BinaryLoglossMetric, CUDABinaryLoglossMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<BinaryLoglossMetric, CUDABinaryLoglossMetric>::LaunchEvalKernelFusedSigmoid(const double* raw_score, double sigmoid, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<BinaryErrorMetric, CUDABinaryErrorMetric>::LaunchEvalKernel(const double* score, double* sum_loss, double* sum_weight) const;
template void CUDAPointwiseMetricInterface<BinaryErrorMetric, CUDABinaryErrorMetric>::LaunchEvalKernelFusedSigmoid(const double* raw_score, double sigmoid, double* sum_loss, double* sum_weight) const;

}  // namespace Falcata

#endif  // USE_CUDA
