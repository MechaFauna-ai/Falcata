/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_regression_objective.hpp"

#include <Falcata/cuda/cuda_algorithms.hpp>
#include <Falcata/falcata_plan.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "../../treelearner/cuda/cuda_pdl_launch.hpp"

namespace Falcata {

template <typename HOST_OBJECTIVE>
void CUDARegressionObjectiveInterface<HOST_OBJECTIVE>::Init(const Metadata& metadata, data_size_t num_data) {
  CUDAObjectiveInterface<HOST_OBJECTIVE>::Init(metadata, num_data);
  const data_size_t num_get_gradients_blocks = CUDABlockCount(this->num_data_, GET_GRADIENTS_BLOCK_SIZE_REGRESSION);
  cuda_block_buffer_.Resize(static_cast<size_t>(num_get_gradients_blocks));
  if (this->sqrt_) {
    cuda_trans_label_.Resize(this->trans_label_.size());
    CopyFromHostToCUDADevice<label_t>(cuda_trans_label_.RawData(), this->trans_label_.data(), this->trans_label_.size(), __FILE__, __LINE__);
    this->cuda_labels_ = cuda_trans_label_.RawData();
  }
}

template void CUDARegressionObjectiveInterface<RegressionL2loss>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDARegressionObjectiveInterface<RegressionL1loss>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDARegressionObjectiveInterface<RegressionHuberLoss>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDARegressionObjectiveInterface<RegressionFairLoss>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDARegressionObjectiveInterface<RegressionPoissonLoss>::Init(const Metadata& metadata, data_size_t num_data);
template void CUDARegressionObjectiveInterface<RegressionQuantileloss>::Init(const Metadata& metadata, data_size_t num_data);

template <typename HOST_OBJECTIVE>
double CUDARegressionObjectiveInterface<HOST_OBJECTIVE>::LaunchCalcInitScoreKernel(const int /*class_id*/) const {
  double label_sum = 0.0f, weight_sum = 0.0f;
  if (this->cuda_weights_ == nullptr) {
    ShuffleReduceSumGlobal<label_t, double>(this->cuda_labels_,
      static_cast<size_t>(this->num_data_), cuda_block_buffer_.RawData());
    CopyFromCUDADeviceToHost<double>(&label_sum, cuda_block_buffer_.RawData(), 1, __FILE__, __LINE__);
    weight_sum = static_cast<double>(this->num_data_);
  } else {
    ShuffleReduceDotProdGlobal<label_t, double>(this->cuda_labels_,
      this->cuda_weights_, static_cast<size_t>(this->num_data_), cuda_block_buffer_.RawData());
    CopyFromCUDADeviceToHost<double>(&label_sum, cuda_block_buffer_.RawData(), 1, __FILE__, __LINE__);
    ShuffleReduceSumGlobal<label_t, double>(this->cuda_weights_,
      static_cast<size_t>(this->num_data_), cuda_block_buffer_.RawData());
    CopyFromCUDADeviceToHost<double>(&weight_sum, cuda_block_buffer_.RawData(), 1, __FILE__, __LINE__);
  }
  return label_sum / weight_sum;
}

template double CUDARegressionObjectiveInterface<RegressionL2loss>::LaunchCalcInitScoreKernel(const int class_id) const;
template double CUDARegressionObjectiveInterface<RegressionL1loss>::LaunchCalcInitScoreKernel(const int class_id) const;
template double CUDARegressionObjectiveInterface<RegressionHuberLoss>::LaunchCalcInitScoreKernel(const int class_id) const;
template double CUDARegressionObjectiveInterface<RegressionFairLoss>::LaunchCalcInitScoreKernel(const int class_id) const;
template double CUDARegressionObjectiveInterface<RegressionPoissonLoss>::LaunchCalcInitScoreKernel(const int class_id) const;
template double CUDARegressionObjectiveInterface<RegressionQuantileloss>::LaunchCalcInitScoreKernel(const int class_id) const;

__global__ void ConvertOutputCUDAKernel_Regression(const bool sqrt, const data_size_t num_data, const double* input, double* output) {
  const int data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (data_index < num_data) {
    if (sqrt) {
      const double sign = input[data_index] >= 0.0f ? 1 : -1;
      output[data_index] = sign * input[data_index] * input[data_index];
    } else {
      output[data_index] = input[data_index];
    }
  }
}

const double* CUDARegressionL2loss::LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const {
  const int num_blocks = CUDABlockCount(num_data, GET_GRADIENTS_BLOCK_SIZE_REGRESSION);
  if (sqrt_) {
    ConvertOutputCUDAKernel_Regression<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(sqrt_, num_data, input, output);
    return output;
  } else {
    return input;
  }
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_RegressionL2(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      cuda_out_gradients[data_index] = static_cast<score_t>(cuda_scores[data_index] - cuda_labels[data_index]);
      cuda_out_hessians[data_index] = 1.0f;
    } else {
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>(cuda_scores[data_index] - cuda_labels[data_index]) * weight;
      cuda_out_hessians[data_index] = weight;
    }
  }
}

void CUDARegressionL2loss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = CUDABlockCount(num_data_, GET_GRADIENTS_BLOCK_SIZE_REGRESSION);
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_RegressionL2<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, gradients, hessians);
  } else {
    GetGradientsKernel_RegressionL2<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, gradients, hessians);
  }
}


// One thread per row, looping over the targets on the grid's y axis: the column-major element of target k is at
// k * num_data + row (64-bit, since num_data * num_target can exceed INT32_MAX), and the row's weight is read once
// instead of recovered with a per-element modulo.
template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_MultiRegressionL2(const double* cuda_scores, const label_t* cuda_labels,
  const label_t* cuda_weights, const data_size_t num_data, const int num_target,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t row = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (row >= num_data) {
    return;
  }
  // per-row weights broadcast over the column-major target dimension
  const score_t weight = USE_WEIGHT ? static_cast<score_t>(cuda_weights[row]) : 1.0f;
  for (int k = static_cast<int>(blockIdx.y); k < num_target; k += static_cast<int>(gridDim.y)) {
    const size_t index = MultiTargetElementIndex(k, num_data, row);
    if (!USE_WEIGHT) {
      cuda_out_gradients[index] = static_cast<score_t>(cuda_scores[index] - cuda_labels[index]);
      cuda_out_hessians[index] = 1.0f;
    } else {
      cuda_out_gradients[index] = static_cast<score_t>(cuda_scores[index] - cuda_labels[index]) * weight;
      cuda_out_hessians[index] = weight;
    }
  }
}

void CUDAMultiRegressionL2::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  if (num_data_ <= 0 || num_target_ <= 0) {
    return;
  }
  const int num_blocks = MultiTargetRowBlocks(num_data_, GET_GRADIENTS_BLOCK_SIZE_REGRESSION);
  const dim3 grid(static_cast<unsigned int>(num_blocks), static_cast<unsigned int>(std::min(num_target_, 65535)));
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_MultiRegressionL2<false><<<grid, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(
      score, cuda_labels_, nullptr, num_data_, num_target_, gradients, hessians);
  } else {
    GetGradientsKernel_MultiRegressionL2<true><<<grid, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(
      score, cuda_labels_, cuda_weights_, num_data_, num_target_, gradients, hessians);
  }
}


double CUDARegressionL1loss::LaunchCalcInitScoreKernel(const int /*class_id*/) const {
  const double alpha = 0.5f;
  if (cuda_weights_ == nullptr) {
    PercentileGlobal<label_t, data_size_t, label_t, double, false, false>(
      cuda_labels_, nullptr, cuda_data_indices_buffer_.RawData(), nullptr, nullptr, alpha, num_data_, cuda_percentile_result_.RawData());
  } else {
    PercentileGlobal<label_t, data_size_t, label_t, double, true, true>(
      cuda_labels_, cuda_weights_, cuda_data_indices_buffer_.RawData(), cuda_weights_prefix_sum_.RawData(),
      cuda_weights_prefix_sum_buffer_.RawData(), alpha, num_data_, cuda_percentile_result_.RawData());
  }
  label_t percentile_result = 0.0f;
  CopyFromCUDADeviceToHost<label_t>(&percentile_result, cuda_percentile_result_.RawData(), 1, __FILE__, __LINE__);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  return static_cast<label_t>(percentile_result);
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_RegressionL1(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>((diff > 0.0f) - (diff < 0.0f));
      cuda_out_hessians[data_index] = 1.0f;
    } else {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>((diff > 0.0f) - (diff < 0.0f)) * weight;
      cuda_out_hessians[data_index] = weight;
    }
  }
}

void CUDARegressionL1loss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = CUDABlockCount(num_data_, GET_GRADIENTS_BLOCK_SIZE_REGRESSION);
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_RegressionL1<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, gradients, hessians);
  } else {
    GetGradientsKernel_RegressionL1<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, gradients, hessians);
  }
}

template <bool USE_WEIGHT>
__global__ void RenewTreeOutputCUDAKernel_RegressionL1(
  const double* score,
  const label_t* label,
  const label_t* weight,
  double* residual_buffer,
  label_t* weight_by_leaf,
  double* weight_prefix_sum_buffer,
  const data_size_t* data_indices_in_leaf,
  const data_size_t* num_data_in_leaf,
  const data_size_t* data_start_in_leaf,
  data_size_t* data_indices_buffer,
  double* leaf_value) {
  const int leaf_index = static_cast<int>(blockIdx.x);
  const data_size_t data_start = data_start_in_leaf[leaf_index];
  const data_size_t num_data = num_data_in_leaf[leaf_index];
  data_size_t* data_indices_buffer_pointer = data_indices_buffer + data_start;
  const label_t* weight_by_leaf_pointer = weight_by_leaf + data_start;
  double* weight_prefix_sum_buffer_pointer = weight_prefix_sum_buffer + data_start;
  const double* residual_buffer_pointer = residual_buffer + data_start;
  const double alpha = 0.5f;
  for (data_size_t inner_data_index = data_start + static_cast<data_size_t>(threadIdx.x);
    inner_data_index < data_start + num_data; inner_data_index += static_cast<data_size_t>(blockDim.x)) {
    const data_size_t data_index = data_indices_in_leaf[inner_data_index];
    const label_t data_label = label[data_index];
    const double data_score = score[data_index];
    residual_buffer[inner_data_index] = static_cast<double>(data_label) - data_score;
    if (USE_WEIGHT) {
      weight_by_leaf[inner_data_index] = weight[data_index];
    }
  }
  __syncthreads();
  const double renew_leaf_value = PercentileDevice<double, data_size_t, label_t, double, USE_WEIGHT, USE_WEIGHT>(
    residual_buffer_pointer, weight_by_leaf_pointer, data_indices_buffer_pointer,
    weight_prefix_sum_buffer_pointer, alpha, num_data);
  if (threadIdx.x == 0) {
    leaf_value[leaf_index] = renew_leaf_value;
  }
}

void CUDARegressionL1loss::LaunchRenewTreeOutputCUDAKernel(
  const double* score,
  const data_size_t* data_indices_in_leaf,
  const data_size_t* num_data_in_leaf,
  const data_size_t* data_start_in_leaf,
  const int num_leaves,
  double* leaf_value) const {
  if (cuda_weights_ == nullptr) {
    RenewTreeOutputCUDAKernel_RegressionL1<false><<<num_leaves, GET_GRADIENTS_BLOCK_SIZE_REGRESSION / 2>>>(
      score,
      cuda_labels_,
      cuda_weights_,
      cuda_residual_buffer_.RawData(),
      cuda_weight_by_leaf_buffer_.RawData(),
      cuda_weights_prefix_sum_.RawData(),
      data_indices_in_leaf,
      num_data_in_leaf,
      data_start_in_leaf,
      cuda_data_indices_buffer_.RawData(),
      leaf_value);
  } else {
    RenewTreeOutputCUDAKernel_RegressionL1<true><<<num_leaves, GET_GRADIENTS_BLOCK_SIZE_REGRESSION / 4>>>(
      score,
      cuda_labels_,
      cuda_weights_,
      cuda_residual_buffer_.RawData(),
      cuda_weight_by_leaf_buffer_.RawData(),
      cuda_weights_prefix_sum_.RawData(),
      data_indices_in_leaf,
      num_data_in_leaf,
      data_start_in_leaf,
      cuda_data_indices_buffer_.RawData(),
      leaf_value);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}


template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_Huber(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  const double alpha, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      if (fabs(diff) <= alpha) {
        cuda_out_gradients[data_index] = static_cast<score_t>(diff);
      } else {
        const score_t sign = static_cast<score_t>((diff > 0.0f) - (diff < 0.0f));
        cuda_out_gradients[data_index] = static_cast<score_t>(sign * alpha);
      }
      cuda_out_hessians[data_index] = 1.0f;
    } else {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      if (fabs(diff) <= alpha) {
        cuda_out_gradients[data_index] = static_cast<score_t>(diff) * weight;
      } else {
        const score_t sign = static_cast<score_t>((diff > 0.0f) - (diff < 0.0f));
        cuda_out_gradients[data_index] = static_cast<score_t>(sign * alpha) * weight;
      }
      cuda_out_hessians[data_index] = weight;
    }
  }
}

void CUDARegressionHuberLoss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = CUDABlockCount(num_data_, GET_GRADIENTS_BLOCK_SIZE_REGRESSION);
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_Huber<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, alpha_, gradients, hessians);
  } else {
    GetGradientsKernel_Huber<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, alpha_, gradients, hessians);
  }
}


template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_Fair(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  const double c, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>(c * diff / (fabs(diff) + c));
      cuda_out_hessians[data_index] = static_cast<score_t>(c * c / ((fabs(diff) + c) * (fabs(diff) + c)));
    } else {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>(c * diff / (fabs(diff) + c) * weight);
      cuda_out_hessians[data_index] = static_cast<score_t>(c * c / ((fabs(diff) + c) * (fabs(diff) + c)) * weight);
    }
  }
}

void CUDARegressionFairLoss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = CUDABlockCount(num_data_, GET_GRADIENTS_BLOCK_SIZE_REGRESSION);
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_Fair<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, c_, gradients, hessians);
  } else {
    GetGradientsKernel_Fair<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, c_, gradients, hessians);
  }
}

void CUDARegressionPoissonLoss::LaunchCheckLabelKernel() const {
  ShuffleReduceSumGlobal<label_t, double>(cuda_labels_, static_cast<size_t>(num_data_), cuda_block_buffer_.RawData());
  double label_sum = 0.0f;
  CopyFromCUDADeviceToHost<double>(&label_sum, cuda_block_buffer_.RawData(), 1, __FILE__, __LINE__);

  ShuffleReduceMinGlobal<label_t, double>(cuda_labels_, static_cast<size_t>(num_data_), cuda_block_buffer_.RawData());
  double label_min = 0.0f;
  CopyFromCUDADeviceToHost<double>(&label_min, cuda_block_buffer_.RawData(), 1, __FILE__, __LINE__);

  if (label_min < 0.0f) {
    Log::Fatal("[%s]: at least one target label is negative", GetName());
  }
  if (label_sum == 0.0f) {
    Log::Fatal("[%s]: sum of labels is zero", GetName());
  }
}

namespace {

constexpr bool kPoissonLeanScoreIsFloat = std::is_same<score_t, float>::value;

__device__ const double kPoissonLeanExp2Table[64] = {  // 2^(j/64), j = 0..63, rounded to nearest double
  0x1.0000000000000p+0, 0x1.02c9a3e778061p+0, 0x1.059b0d3158574p+0, 0x1.0874518759bc8p+0,
  0x1.0b5586cf9890fp+0, 0x1.0e3ec32d3d1a2p+0, 0x1.11301d0125b51p+0, 0x1.1429aaea92de0p+0,
  0x1.172b83c7d517bp+0, 0x1.1a35beb6fcb75p+0, 0x1.1d4873168b9aap+0, 0x1.2063b88628cd6p+0,
  0x1.2387a6e756238p+0, 0x1.26b4565e27cddp+0, 0x1.29e9df51fdee1p+0, 0x1.2d285a6e4030bp+0,
  0x1.306fe0a31b715p+0, 0x1.33c08b26416ffp+0, 0x1.371a7373aa9cbp+0, 0x1.3a7db34e59ff7p+0,
  0x1.3dea64c123422p+0, 0x1.4160a21f72e2ap+0, 0x1.44e086061892dp+0, 0x1.486a2b5c13cd0p+0,
  0x1.4bfdad5362a27p+0, 0x1.4f9b2769d2ca7p+0, 0x1.5342b569d4f82p+0, 0x1.56f4736b527dap+0,
  0x1.5ab07dd485429p+0, 0x1.5e76f15ad2148p+0, 0x1.6247eb03a5585p+0, 0x1.6623882552225p+0,
  0x1.6a09e667f3bcdp+0, 0x1.6dfb23c651a2fp+0, 0x1.71f75e8ec5f74p+0, 0x1.75feb564267c9p+0,
  0x1.7a11473eb0187p+0, 0x1.7e2f336cf4e62p+0, 0x1.82589994cce13p+0, 0x1.868d99b4492edp+0,
  0x1.8ace5422aa0dbp+0, 0x1.8f1ae99157736p+0, 0x1.93737b0cdc5e5p+0, 0x1.97d829fde4e50p+0,
  0x1.9c49182a3f090p+0, 0x1.a0c667b5de565p+0, 0x1.a5503b23e255dp+0, 0x1.a9e6b5579fdbfp+0,
  0x1.ae89f995ad3adp+0, 0x1.b33a2b84f15fbp+0, 0x1.b7f76f2fb5e47p+0, 0x1.bcc1e904bc1d2p+0,
  0x1.c199bdd85529cp+0, 0x1.c67f12e57d14bp+0, 0x1.cb720dcef9069p+0, 0x1.d072d4a07897cp+0,
  0x1.d5818dcfba487p+0, 0x1.da9e603db3285p+0, 0x1.dfc97337b9b5fp+0, 0x1.e502ee78b3ff6p+0,
  0x1.ea4afa2a490dap+0, 0x1.efa1bee615a27p+0, 0x1.f50765b6e4540p+0, 0x1.fa7c1819e90d8p+0
};

__device__ __forceinline__ float PoissonLeanApproxFloat(const double v) {
  const unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(v));  // NOLINT(runtime/int)
  const int biased = static_cast<int>((bits >> 52) & 0x7ffu);
  if (biased <= 1023 - 127) {
    return 0.0f;
  }
  if (biased >= 1023 + 128) {
    return (bits >> 63) ? -INFINITY : INFINITY;
  }
  const unsigned int sign = static_cast<unsigned int>(bits >> 63) << 31;
  const unsigned int fbits = sign | (static_cast<unsigned int>(biased - 1023 + 127) << 23) |
                             static_cast<unsigned int>((bits >> 29) & 0x7fffffu);
  return __int_as_float(static_cast<int>(fbits));
}

__device__ __forceinline__ double PoissonLeanFloatToDouble(const float f) {
  const unsigned int fbits = static_cast<unsigned int>(__float_as_int(f));
  const unsigned long long sign = static_cast<unsigned long long>(fbits >> 31) << 63;  // NOLINT(runtime/int)
  if ((fbits & 0x7fffffffu) == 0) {
    return __longlong_as_double(static_cast<long long>(sign));  // NOLINT(runtime/int)
  }
  const unsigned long long biased = ((fbits >> 23) & 0xffu) - 127 + 1023;  // NOLINT(runtime/int)
  return __longlong_as_double(static_cast<long long>(sign | (biased << 52) |  // NOLINT(runtime/int)
                                                     (static_cast<unsigned long long>(fbits & 0x7fffffu) << 29)));  // NOLINT(runtime/int)
}

__device__ __forceinline__ bool PoissonLeanWideNormal(const double v) {
  const int biased = static_cast<int>((static_cast<unsigned long long>(__double_as_longlong(v)) >> 52) & 0x7ffu);  // NOLINT(runtime/int)
  return biased >= 64 && biased <= 2047 - 64;
}

__device__ __forceinline__ bool PoissonLeanSettledFloat(const double v, const float margin_ulps, float* out) {
  const unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(v));  // NOLINT(runtime/int)
  const int biased = static_cast<int>((bits >> 52) & 0x7ffu);
  const int low = static_cast<int>(bits & ((1ull << 29) - 1));
  const int dist = low >= (1 << 28) ? low - (1 << 28) : (1 << 28) - low;
  if (biased < 1023 - 125 || biased > 1023 + 126 || !(margin_ulps < static_cast<float>(1 << 27)) ||
      dist <= static_cast<int>(margin_ulps) + 1) {
    return false;
  }
  const unsigned long long magnitude = (bits & 0x7fffffffffffffffull) + (1ull << 28);  // NOLINT(runtime/int)
  const unsigned int fbits = static_cast<unsigned int>(bits >> 63) << 31 |
                             (static_cast<unsigned int>(magnitude >> 29) - (static_cast<unsigned int>(1023 - 127) << 23));
  *out = __int_as_float(static_cast<int>(fbits));
  return true;
}

// cuda_plan key gradient_fp64_lean for Poisson regression: Cody-Waite range reduction + FP32 tail / FP64 Horner exp(x)
// (~6 FP64 ops vs libdevice exp's ~25), relative error < 2^-46.5 (total vs libdevice exp < 2^-46 = 128 double ULPs of
// exp_score). Gradient cancellation in (exp_score - label) amplifies error by exp_score / |exp_score - label|; the
// settled-float check guarantees every double within a >2x safety margin rounds to the exact same float as the
// original double path, otherwise falling back to libdevice exp. Compiled out under SCORE_T_USE_DOUBLE.
template <bool USE_WEIGHT>
__device__ __forceinline__ bool PoissonLean(const double score, const double label, const double exp_max_delta_step,
                                            const double sample_weight, float* out_gradient, float* out_hessian) {
  const unsigned long long x_magnitude =  // NOLINT(runtime/int)
      static_cast<unsigned long long>(__double_as_longlong(score)) & 0x7fffffffffffffffull;  // NOLINT(runtime/int)
  if (!(x_magnitude <= 0x4054000000000000ull)) {  // |score| <= 80 (NaN fails)
    return false;
  }
  const float kf = rintf(PoissonLeanApproxFloat(score) * 92.3324826168936f);  // 64 / ln2
  const int k = static_cast<int>(kf);
  const double kd = PoissonLeanFloatToDouble(kf);
  double r = fma(-kd, 0x1.62e42fefa39efp-7, score);  // ln2/64, high and low parts
  r = fma(-kd, 0x1.abc9e3b39803fp-62, r);
  const float rf = PoissonLeanApproxFloat(r);
  float t = 1.38888888888888889e-03f;  // 1/6!
  t = fmaf(t, rf, 8.33333333333333333e-03f);  // 1/5!
  t = fmaf(t, rf, 4.16666666666666667e-02f);  // 1/4!
  t = fmaf(t, rf, 1.66666666666666667e-01f);  // 1/3!
  double p = fma(PoissonLeanFloatToDouble(t), r, 0.5);
  p = fma(p, r, 1.0);
  p = fma(p, r, 1.0);
  const double scaled = p * kPoissonLeanExp2Table[k & 63];
  const double e = __longlong_as_double(__double_as_longlong(scaled) + (static_cast<long long>(k >> 6) << 52));  // NOLINT(runtime/int)
  const double g_raw = e - label;
  double g = g_raw;
  double h = e * exp_max_delta_step;
  if (USE_WEIGHT) {
    if (!PoissonLeanWideNormal(g_raw) || !PoissonLeanWideNormal(h)) {
      return false;
    }
    g = g * sample_weight;
    h = h * sample_weight;
  }
  const float abs_g_raw_f = PoissonLeanApproxFloat(fabs(g_raw));
  if (!(abs_g_raw_f > 0.0f)) {
    return false;
  }
  const float g_ratio = PoissonLeanApproxFloat(e) / abs_g_raw_f;
  float g_out, h_out;
  if (!PoissonLeanSettledFloat(g, 256.0f * g_ratio + 16.0f, &g_out) ||
      !PoissonLeanSettledFloat(h, 256.0f + 16.0f, &h_out)) {
    return false;
  }
  *out_gradient = g_out;
  *out_hessian = h_out;
  return true;
}

__global__ void ComputeExpScalarKernel(const double x, double* out) {
  *out = exp(x);
}

}  // namespace

void CUDARegressionPoissonLoss::LaunchComputeExpMaxDeltaStepKernel() {
  if (cuda_block_buffer_.Size() > 0) {
    // Evaluate exp(max_delta_step_) once on device so the hoisted constant is bit-identical to device exp().
    ComputeExpScalarKernel<<<1, 1>>>(max_delta_step_, cuda_block_buffer_.RawData());
    CopyFromCUDADeviceToHost<double>(&exp_max_delta_step_, cuda_block_buffer_.RawData(), 1, __FILE__, __LINE__);
  } else {
    exp_max_delta_step_ = std::exp(max_delta_step_);
  }
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_Poisson(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  const double exp_max_delta_step, score_t* cuda_out_gradients, score_t* cuda_out_hessians, const bool lean) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    const double score = cuda_scores[data_index];
    const label_t label = cuda_labels[data_index];
    const score_t weight = USE_WEIGHT ? static_cast<score_t>(cuda_weights[data_index]) : static_cast<score_t>(1.0f);
    if (kPoissonLeanScoreIsFloat && lean) {
      float lean_gradient, lean_hessian;
      if (PoissonLean<USE_WEIGHT>(score, static_cast<double>(label), exp_max_delta_step,
                                  static_cast<double>(weight), &lean_gradient, &lean_hessian)) {
        cuda_out_gradients[data_index] = lean_gradient;
        cuda_out_hessians[data_index] = lean_hessian;
        return;
      }
    }
    const double exp_score = exp(score);
    if (!USE_WEIGHT) {
      cuda_out_gradients[data_index] = static_cast<score_t>(exp_score - label);
      cuda_out_hessians[data_index] = static_cast<score_t>(exp_score * exp_max_delta_step);
    } else {
      cuda_out_gradients[data_index] = static_cast<score_t>((exp_score - label) * weight);
      cuda_out_hessians[data_index] = static_cast<score_t>(exp_score * exp_max_delta_step * weight);
    }
  }
}

void CUDARegressionPoissonLoss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = CUDABlockCount(num_data_, GET_GRADIENTS_BLOCK_SIZE_REGRESSION);
  const bool lean = FalcataPlan::Get().gradient_fp64_lean;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_Poisson<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(
      score, cuda_labels_, nullptr, num_data_, exp_max_delta_step_, gradients, hessians, lean);
  } else {
    GetGradientsKernel_Poisson<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(
      score, cuda_labels_, cuda_weights_, num_data_, exp_max_delta_step_, gradients, hessians, lean);
  }
}

__global__ void ConvertOutputCUDAKernel_Regression_Poisson(const data_size_t num_data, const double* input, double* output) {
  const int data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (data_index < num_data) {
    output[data_index] = exp(input[data_index]);
  }
}

const double* CUDARegressionPoissonLoss::LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const {
  const int num_blocks = CUDABlockCount(num_data, GET_GRADIENTS_BLOCK_SIZE_REGRESSION);
  ConvertOutputCUDAKernel_Regression_Poisson<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(num_data, input, output);
  return output;
}


double CUDARegressionQuantileloss::LaunchCalcInitScoreKernel(const int /*class_id*/) const {
  if (cuda_weights_ == nullptr) {
    PercentileGlobal<label_t, data_size_t, label_t, double, false, false>(
      cuda_labels_, nullptr, cuda_data_indices_buffer_.RawData(), nullptr, nullptr, alpha_, num_data_, cuda_percentile_result_.RawData());
  } else {
    PercentileGlobal<label_t, data_size_t, label_t, double, true, true>(
      cuda_labels_, cuda_weights_, cuda_data_indices_buffer_.RawData(), cuda_weights_prefix_sum_.RawData(),
      cuda_weights_prefix_sum_buffer_.RawData(), alpha_, num_data_, cuda_percentile_result_.RawData());
  }
  label_t percentile_result = 0.0f;
  CopyFromCUDADeviceToHost<label_t>(&percentile_result, cuda_percentile_result_.RawData(), 1, __FILE__, __LINE__);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  return static_cast<label_t>(percentile_result);
}

template <bool USE_WEIGHT>
__global__ void RenewTreeOutputCUDAKernel_RegressionQuantile(
  const double* score,
  const label_t* label,
  const label_t* weight,
  double* residual_buffer,
  label_t* weight_by_leaf,
  double* weight_prefix_sum_buffer,
  const data_size_t* data_indices_in_leaf,
  const data_size_t* num_data_in_leaf,
  const data_size_t* data_start_in_leaf,
  data_size_t* data_indices_buffer,
  double* leaf_value,
  const double alpha) {
  const int leaf_index = static_cast<int>(blockIdx.x);
  const data_size_t data_start = data_start_in_leaf[leaf_index];
  const data_size_t num_data = num_data_in_leaf[leaf_index];
  data_size_t* data_indices_buffer_pointer = data_indices_buffer + data_start;
  const label_t* weight_by_leaf_pointer = weight_by_leaf + data_start;
  double* weight_prefix_sum_buffer_pointer = weight_prefix_sum_buffer + data_start;
  const double* residual_buffer_pointer = residual_buffer + data_start;
  for (data_size_t inner_data_index = data_start + static_cast<data_size_t>(threadIdx.x); inner_data_index < data_start + num_data; inner_data_index += static_cast<data_size_t>(blockDim.x)) {
    const data_size_t data_index = data_indices_in_leaf[inner_data_index];
    const label_t data_label = label[data_index];
    const double data_score = score[data_index];
    residual_buffer[inner_data_index] = static_cast<double>(data_label) - data_score;
    if (USE_WEIGHT) {
      weight_by_leaf[inner_data_index] = weight[data_index];
    }
  }
  __syncthreads();
  const double renew_leaf_value = PercentileDevice<double, data_size_t, label_t, double, USE_WEIGHT, USE_WEIGHT>(
    residual_buffer_pointer, weight_by_leaf_pointer, data_indices_buffer_pointer,
    weight_prefix_sum_buffer_pointer, alpha, num_data);
  if (threadIdx.x == 0) {
    leaf_value[leaf_index] = renew_leaf_value;
  }
}

void CUDARegressionQuantileloss::LaunchRenewTreeOutputCUDAKernel(
  const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
  const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const {
  if (cuda_weights_ == nullptr) {
    RenewTreeOutputCUDAKernel_RegressionQuantile<false><<<num_leaves, GET_GRADIENTS_BLOCK_SIZE_REGRESSION / 2>>>(
      score,
      cuda_labels_,
      cuda_weights_,
      cuda_residual_buffer_.RawData(),
      cuda_weight_by_leaf_buffer_.RawData(),
      cuda_weights_prefix_sum_.RawData(),
      data_indices_in_leaf,
      num_data_in_leaf,
      data_start_in_leaf,
      cuda_data_indices_buffer_.RawData(),
      leaf_value,
      alpha_);
  } else {
    RenewTreeOutputCUDAKernel_RegressionQuantile<true><<<num_leaves, GET_GRADIENTS_BLOCK_SIZE_REGRESSION / 4>>>(
      score,
      cuda_labels_,
      cuda_weights_,
      cuda_residual_buffer_.RawData(),
      cuda_weight_by_leaf_buffer_.RawData(),
      cuda_weights_prefix_sum_.RawData(),
      data_indices_in_leaf,
      num_data_in_leaf,
      data_start_in_leaf,
      cuda_data_indices_buffer_.RawData(),
      leaf_value,
      alpha_);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_RegressionQuantile(const double* cuda_scores, const label_t* cuda_labels,
  const label_t* cuda_weights, const data_size_t num_data, const double alpha,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      if (diff >= 0.0f) {
        cuda_out_gradients[data_index] = (1.0f - alpha);
      } else {
        cuda_out_gradients[data_index] = -alpha;
      }
      cuda_out_hessians[data_index] = 1.0f;
    } else {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      if (diff >= 0.0f) {
        cuda_out_gradients[data_index] = (1.0f - alpha) * weight;
      } else {
        cuda_out_gradients[data_index] = -alpha * weight;
      }
      cuda_out_hessians[data_index] = weight;
    }
  }
}

void CUDARegressionQuantileloss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = CUDABlockCount(num_data_, GET_GRADIENTS_BLOCK_SIZE_REGRESSION);
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_RegressionQuantile<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, alpha_, gradients, hessians);
  } else {
    GetGradientsKernel_RegressionQuantile<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, alpha_, gradients, hessians);
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
