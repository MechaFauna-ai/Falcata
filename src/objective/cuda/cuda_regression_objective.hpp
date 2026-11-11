/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef FALCATA_SRC_OBJECTIVE_CUDA_CUDA_REGRESSION_OBJECTIVE_HPP_
#define FALCATA_SRC_OBJECTIVE_CUDA_CUDA_REGRESSION_OBJECTIVE_HPP_

#ifdef USE_CUDA

#define GET_GRADIENTS_BLOCK_SIZE_REGRESSION (1024)

#include <Falcata/cuda/cuda_objective_function.hpp>

#include <string>
#include <vector>

#include "../regression_objective.hpp"
#include "../multi_regression_objective.hpp"

namespace Falcata {

/*! \brief multi-target objectives: the column-major offset of target k's element of row, k * num_data + row. In 64
 *  bits: num_data * num_target exceeds INT32_MAX for large multi-target datasets. */
__host__ __device__ inline size_t MultiTargetElementIndex(const int k, const data_size_t num_data,
                                                          const data_size_t row) {
  return static_cast<size_t>(k) * static_cast<size_t>(num_data) + static_cast<size_t>(row);
}

/*! \brief multi-target objectives: blocks of block_size threads covering num_data rows (one thread per row) */
__host__ __device__ inline int MultiTargetRowBlocks(const data_size_t num_data, const int block_size) {
  return static_cast<int>((static_cast<int64_t>(num_data) + block_size - 1) / block_size);
}

template <typename HOST_OBJECTIVE>
class CUDARegressionObjectiveInterface: public CUDAObjectiveInterface<HOST_OBJECTIVE> {
 public:
  explicit CUDARegressionObjectiveInterface(const Config& config): CUDAObjectiveInterface<HOST_OBJECTIVE>(config) {}

  explicit CUDARegressionObjectiveInterface(const std::vector<std::string>& strs): CUDAObjectiveInterface<HOST_OBJECTIVE>(strs) {}

  void Init(const Metadata& metadata, data_size_t num_data) override;

 protected:
  double LaunchCalcInitScoreKernel(const int class_id) const override;

  CUDAVector<double> cuda_block_buffer_;
  CUDAVector<label_t> cuda_trans_label_;
};

/*! \brief L1 and quantile losses: both seed the init score from a global percentile of the labels and renew leaf
 *  outputs from per-leaf percentiles of the residuals, through the same device buffers, sized here for both. */
template <typename HOST_OBJECTIVE>
class CUDAPercentileRegressionObjective : public CUDARegressionObjectiveInterface<HOST_OBJECTIVE> {
 public:
  explicit CUDAPercentileRegressionObjective(const Config& config)
      : CUDARegressionObjectiveInterface<HOST_OBJECTIVE>(config) {}

  explicit CUDAPercentileRegressionObjective(const std::vector<std::string>& strs)
      : CUDARegressionObjectiveInterface<HOST_OBJECTIVE>(strs) {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    CUDARegressionObjectiveInterface<HOST_OBJECTIVE>::Init(metadata, num_data);
    cuda_data_indices_buffer_.Resize(static_cast<size_t>(num_data));
    cuda_percentile_result_.Resize(1);
    if (this->cuda_weights_ != nullptr) {
      cuda_weights_prefix_sum_.Resize(static_cast<size_t>(num_data));
      // one partial sum per prefix-sum block, plus one
      cuda_weights_prefix_sum_buffer_.Resize(
        static_cast<size_t>(CUDABlockCount(num_data, GET_GRADIENTS_BLOCK_SIZE_REGRESSION)) + 1);
      cuda_weight_by_leaf_buffer_.Resize(static_cast<size_t>(num_data));
    }
    cuda_residual_buffer_.Resize(static_cast<size_t>(num_data));
  }

 protected:
  CUDAVector<data_size_t> cuda_data_indices_buffer_;
  CUDAVector<double> cuda_weights_prefix_sum_;
  CUDAVector<double> cuda_weights_prefix_sum_buffer_;
  CUDAVector<double> cuda_residual_buffer_;
  CUDAVector<label_t> cuda_weight_by_leaf_buffer_;
  CUDAVector<label_t> cuda_percentile_result_;
};

class CUDARegressionL2loss : public CUDARegressionObjectiveInterface<RegressionL2loss> {
 public:
  explicit CUDARegressionL2loss(const Config& config);

  explicit CUDARegressionL2loss(const std::vector<std::string>& strs);

  ~CUDARegressionL2loss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

 protected:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  const double* LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const override;

  bool NeedConvertOutputCUDA() const override { return sqrt_; }
};


// Round-robin multi-target L2: one elementwise kernel over
// num_target * num_data (score/label/gradient layouts are all column-major
// [target * num_data + i]); per-row weights broadcast across targets.
// Boost-from-average stays on host (per-target label means, host label copy).
class CUDAMultiRegressionL2 : public CUDAObjectiveInterface<MultiRegressionL2> {
 public:
  explicit CUDAMultiRegressionL2(const Config& config);

  explicit CUDAMultiRegressionL2(const std::vector<std::string>& strs);

  ~CUDAMultiRegressionL2();

 protected:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;
};


class CUDARegressionL1loss : public CUDAPercentileRegressionObjective<RegressionL1loss> {
 public:
  explicit CUDARegressionL1loss(const Config& config);

  explicit CUDARegressionL1loss(const std::vector<std::string>& strs);

  ~CUDARegressionL1loss();

 protected:
  double LaunchCalcInitScoreKernel(const int class_id) const override;

  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  void LaunchRenewTreeOutputCUDAKernel(
    const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const override;
};


class CUDARegressionHuberLoss : public CUDARegressionObjectiveInterface<RegressionHuberLoss> {
 public:
  explicit CUDARegressionHuberLoss(const Config& config);

  explicit CUDARegressionHuberLoss(const std::vector<std::string>& strs);

  ~CUDARegressionHuberLoss();

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;
};


// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
class CUDARegressionFairLoss : public CUDARegressionObjectiveInterface<RegressionFairLoss> {
 public:
  explicit CUDARegressionFairLoss(const Config& config);

  explicit CUDARegressionFairLoss(const std::vector<std::string>& strs);

  ~CUDARegressionFairLoss();

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;
};


class CUDARegressionPoissonLoss : public CUDARegressionObjectiveInterface<RegressionPoissonLoss> {
 public:
  explicit CUDARegressionPoissonLoss(const Config& config);

  explicit CUDARegressionPoissonLoss(const std::vector<std::string>& strs);

  ~CUDARegressionPoissonLoss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  const double* LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const override;

  bool NeedConvertOutputCUDA() const override { return true; }

  double LaunchCalcInitScoreKernel(const int class_id) const override;

  void LaunchCheckLabelKernel() const;
};


class CUDARegressionQuantileloss : public CUDAPercentileRegressionObjective<RegressionQuantileloss> {
 public:
  explicit CUDARegressionQuantileloss(const Config& config);

  explicit CUDARegressionQuantileloss(const std::vector<std::string>& strs);

  ~CUDARegressionQuantileloss();

 protected:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  double LaunchCalcInitScoreKernel(const int class_id) const override;

  void LaunchRenewTreeOutputCUDAKernel(
    const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const override;
};


}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_SRC_OBJECTIVE_CUDA_CUDA_REGRESSION_OBJECTIVE_HPP_
