/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_multiclass_metric.hpp"

#include <cmath>
#include <vector>

namespace Falcata {

CUDAMultiSoftmaxLoglossMetric::CUDAMultiSoftmaxLoglossMetric(const Config& config)
    : CUDAMetricInterface<MultiSoftmaxLoglossMetric>(config) {}

void CUDAMultiSoftmaxLoglossMetric::Init(const Metadata& metadata, data_size_t num_data) {
  CUDAMetricInterface<MultiSoftmaxLoglossMetric>::Init(metadata, num_data);
  const int max_num_reduce_blocks = CUDABlockCount(this->num_data_, MULTICLASS_EVAL_BLOCK_SIZE);
  reduce_block_buffer_.Resize(static_cast<size_t>(max_num_reduce_blocks));
  reduce_block_buffer_inner_.Resize(1);
}

std::vector<double> CUDAMultiSoftmaxLoglossMetric::Eval(const double* score, const ObjectiveFunction* objective) const {
  double sum_loss = 0.0;
  if (objective != nullptr) {
    double transform_param = 0.0;
    const ObjectiveFunction::CUDAOutputTransform transform = objective->GetCUDAOutputTransform(&transform_param);
    if (transform == ObjectiveFunction::CUDAOutputTransform::kSoftmax) {
      LaunchEvalSoftmaxKernel(score, &sum_loss);
    } else if (transform == ObjectiveFunction::CUDAOutputTransform::kSigmoid) {
      LaunchEvalSigmoidKernel(score, transform_param, &sum_loss);
    } else if (transform == ObjectiveFunction::CUDAOutputTransform::kIdentity) {
      LaunchEvalIdentityKernel(score, &sum_loss);
    } else {
      this->score_convert_buffer_.Resize(static_cast<size_t>(this->num_data_) * static_cast<size_t>(this->num_class_));
      const double* score_convert = objective->ConvertOutputCUDA(this->num_data_, score, this->score_convert_buffer_.RawData());
      LaunchEvalIdentityKernel(score_convert, &sum_loss);
    }
  } else {
    LaunchEvalIdentityKernel(score, &sum_loss);
  }
  return std::vector<double>{sum_loss / this->sum_weights_};
}

CUDAMultiErrorMetric::CUDAMultiErrorMetric(const Config& config)
    : CUDAMetricInterface<MultiErrorMetric>(config), multi_error_top_k_(config.multi_error_top_k) {}

void CUDAMultiErrorMetric::Init(const Metadata& metadata, data_size_t num_data) {
  CUDAMetricInterface<MultiErrorMetric>::Init(metadata, num_data);
  const int max_num_reduce_blocks = CUDABlockCount(this->num_data_, MULTICLASS_EVAL_BLOCK_SIZE);
  reduce_block_buffer_.Resize(static_cast<size_t>(max_num_reduce_blocks));
  reduce_block_buffer_inner_.Resize(1);
}

std::vector<double> CUDAMultiErrorMetric::Eval(const double* score, const ObjectiveFunction* objective) const {
  double sum_loss = 0.0;
  if (objective != nullptr) {
    double transform_param = 0.0;
    const ObjectiveFunction::CUDAOutputTransform transform = objective->GetCUDAOutputTransform(&transform_param);
    if (transform == ObjectiveFunction::CUDAOutputTransform::kSoftmax) {
      if (multi_error_top_k_ == 1) {
        LaunchEvalIdentityKernel(score, &sum_loss);
      } else {
        LaunchEvalSoftmaxTopKKernel(score, &sum_loss);
      }
    } else if (transform == ObjectiveFunction::CUDAOutputTransform::kSigmoid) {
      LaunchEvalSigmoidKernel(score, transform_param, &sum_loss);
    } else if (transform == ObjectiveFunction::CUDAOutputTransform::kIdentity) {
      LaunchEvalIdentityKernel(score, &sum_loss);
    } else {
      this->score_convert_buffer_.Resize(static_cast<size_t>(this->num_data_) * static_cast<size_t>(this->num_class_));
      const double* score_convert = objective->ConvertOutputCUDA(this->num_data_, score, this->score_convert_buffer_.RawData());
      LaunchEvalIdentityKernel(score_convert, &sum_loss);
    }
  } else {
    LaunchEvalIdentityKernel(score, &sum_loss);
  }
  return std::vector<double>{sum_loss / this->sum_weights_};
}

CUDAMultiRMSEMetric::CUDAMultiRMSEMetric(const Config& config)
    : CUDAMetricInterface<MultiRMSEMetric>(config) {}

void CUDAMultiRMSEMetric::Init(const Metadata& metadata, data_size_t num_data) {
  CUDAMetricInterface<MultiRMSEMetric>::Init(metadata, num_data);
  const int max_num_reduce_blocks = CUDABlockCount(this->num_data_, MULTICLASS_EVAL_BLOCK_SIZE);
  reduce_block_buffer_.Resize(static_cast<size_t>(max_num_reduce_blocks));
  reduce_block_buffer_inner_.Resize(1);
}

std::vector<double> CUDAMultiRMSEMetric::Eval(const double* score, const ObjectiveFunction* /*objective*/) const {
  double sum_loss = 0.0;
  LaunchEvalKernel(score, &sum_loss);
  return std::vector<double>{std::sqrt(sum_loss / this->sum_weights_)};
}

}  // namespace Falcata

#endif  // USE_CUDA
