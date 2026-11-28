/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_binary_metric.hpp"

#include <vector>

namespace Falcata {

template <typename HOST_METRIC, typename CUDA_METRIC>
std::vector<double> CUDABinaryMetricInterface<HOST_METRIC, CUDA_METRIC>::Eval(const double* score, const ObjectiveFunction* objective) const {
  double sum_loss = 0.0, sum_weight = 0.0;
  if (objective != nullptr) {
    double transform_param = 0.0;
    const ObjectiveFunction::CUDAOutputTransform transform = objective->GetCUDAOutputTransform(&transform_param);
    if (transform == ObjectiveFunction::CUDAOutputTransform::kSigmoid) {
      this->LaunchEvalKernelFusedSigmoid(score, transform_param, &sum_loss, &sum_weight);
      return std::vector<double>{sum_loss / sum_weight};
    } else if (transform != ObjectiveFunction::CUDAOutputTransform::kIdentity) {
      this->score_convert_buffer_.Resize(static_cast<size_t>(this->num_data_) * static_cast<size_t>(this->num_class_));
      const double* score_convert = objective->ConvertOutputCUDA(this->num_data_, score, this->score_convert_buffer_.RawData());
      this->LaunchEvalKernel(score_convert, &sum_loss, &sum_weight);
      return std::vector<double>{sum_loss / sum_weight};
    }
  }
  this->LaunchEvalKernel(score, &sum_loss, &sum_weight);
  const double eval_score = sum_loss / sum_weight;
  return std::vector<double>{eval_score};
}

CUDABinaryLoglossMetric::CUDABinaryLoglossMetric(const Config& config):CUDABinaryMetricInterface<BinaryLoglossMetric, CUDABinaryLoglossMetric>(config) {}

CUDABinaryErrorMetric::CUDABinaryErrorMetric(const Config& config):CUDABinaryMetricInterface<BinaryErrorMetric, CUDABinaryErrorMetric>(config) {}

}  // namespace Falcata

#endif  // USE_CUDA
