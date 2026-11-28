/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef FALCATA_SRC_METRIC_CUDA_CUDA_MULTICLASS_METRIC_HPP_
#define FALCATA_SRC_METRIC_CUDA_CUDA_MULTICLASS_METRIC_HPP_

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_metric.hpp>
#include <Falcata/cuda/cuda_utils.hu>

#include <vector>

#include "../multiclass_metric.hpp"

#define MULTICLASS_EVAL_BLOCK_SIZE (256)

namespace Falcata {

class CUDAMultiSoftmaxLoglossMetric : public CUDAMetricInterface<MultiSoftmaxLoglossMetric> {
 public:
  explicit CUDAMultiSoftmaxLoglossMetric(const Config& config);

  virtual ~CUDAMultiSoftmaxLoglossMetric() {}

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;

 private:
  void LaunchEvalSoftmaxKernel(const double* score, double* sum_loss) const;

  void LaunchEvalSigmoidKernel(const double* score, double sigmoid, double* sum_loss) const;

  void LaunchEvalIdentityKernel(const double* score, double* sum_loss) const;

  mutable CUDAVector<double> score_convert_buffer_;
  CUDAVector<double> reduce_block_buffer_;
  CUDAVector<double> reduce_block_buffer_inner_;
};

class CUDAMultiErrorMetric : public CUDAMetricInterface<MultiErrorMetric> {
 public:
  explicit CUDAMultiErrorMetric(const Config& config);

  virtual ~CUDAMultiErrorMetric() {}

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;

 private:
  void LaunchEvalIdentityKernel(const double* score, double* sum_loss) const;

  void LaunchEvalSoftmaxTopKKernel(const double* score, double* sum_loss) const;

  void LaunchEvalSigmoidKernel(const double* score, double sigmoid, double* sum_loss) const;

  const int multi_error_top_k_;
  mutable CUDAVector<double> score_convert_buffer_;
  CUDAVector<double> reduce_block_buffer_;
  CUDAVector<double> reduce_block_buffer_inner_;
};

class CUDAMultiRMSEMetric : public CUDAMetricInterface<MultiRMSEMetric> {
 public:
  explicit CUDAMultiRMSEMetric(const Config& config);

  virtual ~CUDAMultiRMSEMetric() {}

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;

 private:
  void LaunchEvalKernel(const double* score, double* sum_loss) const;

  CUDAVector<double> reduce_block_buffer_;
  CUDAVector<double> reduce_block_buffer_inner_;
};

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_SRC_METRIC_CUDA_CUDA_MULTICLASS_METRIC_HPP_
