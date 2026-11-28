/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_regression_objective.hpp"

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "../../treelearner/cuda/cuda_pdl_launch.hpp"
#include <Falcata/cuda/cuda_algorithms.hpp>

namespace Falcata {

namespace {

constexpr int kFusedScoreGradBlockSize = 256;

#if !defined(SCORE_T_USE_DOUBLE) && !defined(LABEL_T_USE_DOUBLE)
inline bool IsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xfu) == 0;
}

inline bool IsAligned8(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0x7u) == 0;
}
#endif  // !SCORE_T_USE_DOUBLE && !LABEL_T_USE_DOUBLE

enum class RegFusedLoss : int {
  kL2 = 0,
  kL1 = 1,
  kHuber = 2,
  kQuantile = 3,
};

template <bool USE_WEIGHT>
__device__ __forceinline__ void ComputeRegressionL2GradHess(
  const double score, const label_t label, const label_t raw_weight,
  score_t* out_grad, score_t* out_hess) {
  if (!USE_WEIGHT) {
    *out_grad = static_cast<score_t>(score - label);
    *out_hess = 1.0f;
  } else {
    const score_t weight = static_cast<score_t>(raw_weight);
    *out_grad = static_cast<score_t>(score - label) * weight;
    *out_hess = weight;
  }
}

template <bool USE_WEIGHT>
__device__ __forceinline__ void ComputeRegressionL1GradHess(
  const double score, const label_t label, const label_t raw_weight,
  score_t* out_grad, score_t* out_hess) {
  const double diff = score - static_cast<double>(label);
  if (!USE_WEIGHT) {
    *out_grad = static_cast<score_t>((diff > 0.0f) - (diff < 0.0f));
    *out_hess = 1.0f;
  } else {
    const score_t weight = static_cast<score_t>(raw_weight);
    *out_grad = static_cast<score_t>((diff > 0.0f) - (diff < 0.0f)) * weight;
    *out_hess = weight;
  }
}

template <bool USE_WEIGHT>
__device__ __forceinline__ void ComputeHuberGradHess(
  const double score, const label_t label, const label_t raw_weight, const double alpha,
  score_t* out_grad, score_t* out_hess) {
  const double diff = score - static_cast<double>(label);
  if (!USE_WEIGHT) {
    if (fabs(diff) <= alpha) {
      *out_grad = static_cast<score_t>(diff);
    } else {
      const score_t sign = static_cast<score_t>((diff > 0.0f) - (diff < 0.0f));
      *out_grad = static_cast<score_t>(sign * alpha);
    }
    *out_hess = 1.0f;
  } else {
    const score_t weight = static_cast<score_t>(raw_weight);
    if (fabs(diff) <= alpha) {
      *out_grad = static_cast<score_t>(diff) * weight;
    } else {
      const score_t sign = static_cast<score_t>((diff > 0.0f) - (diff < 0.0f));
      *out_grad = static_cast<score_t>(sign * alpha) * weight;
    }
    *out_hess = weight;
  }
}

template <bool USE_WEIGHT>
__device__ __forceinline__ void ComputeQuantileGradHess(
  const double score, const label_t label, const label_t raw_weight, const double alpha,
  score_t* out_grad, score_t* out_hess) {
  const double diff = score - static_cast<double>(label);
  if (!USE_WEIGHT) {
    if (diff >= 0.0f) {
      *out_grad = static_cast<score_t>(1.0f - alpha);
    } else {
      *out_grad = static_cast<score_t>(-alpha);
    }
    *out_hess = 1.0f;
  } else {
    const score_t weight = static_cast<score_t>(raw_weight);
    if (diff >= 0.0f) {
      *out_grad = static_cast<score_t>((1.0f - alpha) * weight);
    } else {
      *out_grad = static_cast<score_t>(-alpha * weight);
    }
    *out_hess = weight;
  }
}

template <RegFusedLoss LOSS, bool USE_WEIGHT>
__device__ __forceinline__ void ComputeRegFusedGradHess(
  const double score, const label_t label, const label_t raw_weight, const double param,
  score_t* out_grad, score_t* out_hess) {
  if constexpr (LOSS == RegFusedLoss::kL2) {
    ComputeRegressionL2GradHess<USE_WEIGHT>(score, label, raw_weight, out_grad, out_hess);
  } else if constexpr (LOSS == RegFusedLoss::kL1) {
    ComputeRegressionL1GradHess<USE_WEIGHT>(score, label, raw_weight, out_grad, out_hess);
  } else if constexpr (LOSS == RegFusedLoss::kHuber) {
    ComputeHuberGradHess<USE_WEIGHT>(score, label, raw_weight, param, out_grad, out_hess);
  } else {
    ComputeQuantileGradHess<USE_WEIGHT>(score, label, raw_weight, param, out_grad, out_hess);
  }
}

#if !defined(SCORE_T_USE_DOUBLE) && !defined(LABEL_T_USE_DOUBLE)
template <RegFusedLoss LOSS, bool USE_WEIGHT>
__global__ void AddPredictionAndGetGradientsKernel_Regression_Vec2(
  const int* __restrict__ cuda_data_index_to_leaf_index,
  const double* __restrict__ cuda_leaf_value,
  double* __restrict__ cuda_scores,
  const label_t* __restrict__ cuda_labels,
  const label_t* __restrict__ cuda_weights,
  const data_size_t num_data,
  const double param,
  score_t* __restrict__ cuda_out_gradients,
  score_t* __restrict__ cuda_out_hessians) {
  PdlWaitForProducer();
  PdlLaunchDependents();
  const data_size_t pair_idx = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  const data_size_t base = pair_idx * 2;
  if (base + 1 < num_data) {
    const int2 leaf = reinterpret_cast<const int2*>(cuda_data_index_to_leaf_index)[pair_idx];
    double2 s = reinterpret_cast<const double2*>(cuda_scores)[pair_idx];
    s.x += __ldg(&cuda_leaf_value[leaf.x]);
    s.y += __ldg(&cuda_leaf_value[leaf.y]);
    reinterpret_cast<double2*>(cuda_scores)[pair_idx] = s;
    const float2 lbl = reinterpret_cast<const float2*>(cuda_labels)[pair_idx];
    const float2 w = USE_WEIGHT ? reinterpret_cast<const float2*>(cuda_weights)[pair_idx] : make_float2(1.0f, 1.0f);
    float2 g, h;
    ComputeRegFusedGradHess<LOSS, USE_WEIGHT>(s.x, lbl.x, w.x, param, &g.x, &h.x);
    ComputeRegFusedGradHess<LOSS, USE_WEIGHT>(s.y, lbl.y, w.y, param, &g.y, &h.y);
    reinterpret_cast<float2*>(cuda_out_gradients)[pair_idx] = g;
    reinterpret_cast<float2*>(cuda_out_hessians)[pair_idx] = h;
  } else if (base < num_data) {
    const int leaf = cuda_data_index_to_leaf_index[base];
    const double s = cuda_scores[base] + __ldg(&cuda_leaf_value[leaf]);
    cuda_scores[base] = s;
    const label_t w = USE_WEIGHT ? cuda_weights[base] : static_cast<label_t>(1.0f);
    score_t g, h;
    ComputeRegFusedGradHess<LOSS, USE_WEIGHT>(s, cuda_labels[base], w, param, &g, &h);
    cuda_out_gradients[base] = g;
    cuda_out_hessians[base] = h;
  }
}
#endif  // !SCORE_T_USE_DOUBLE && !LABEL_T_USE_DOUBLE

template <RegFusedLoss LOSS, bool USE_WEIGHT>
__global__ void AddPredictionAndGetGradientsKernel_Regression_Scalar(
  const int* __restrict__ cuda_data_index_to_leaf_index,
  const double* __restrict__ cuda_leaf_value,
  double* __restrict__ cuda_scores,
  const label_t* __restrict__ cuda_labels,
  const label_t* __restrict__ cuda_weights,
  const data_size_t num_data,
  const double param,
  score_t* __restrict__ cuda_out_gradients,
  score_t* __restrict__ cuda_out_hessians) {
  PdlWaitForProducer();
  PdlLaunchDependents();
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    const int leaf = cuda_data_index_to_leaf_index[data_index];
    const double s = cuda_scores[data_index] + __ldg(&cuda_leaf_value[leaf]);
    cuda_scores[data_index] = s;
    const label_t w = USE_WEIGHT ? cuda_weights[data_index] : static_cast<label_t>(1.0f);
    score_t g, h;
    ComputeRegFusedGradHess<LOSS, USE_WEIGHT>(s, cuda_labels[data_index], w, param, &g, &h);
    cuda_out_gradients[data_index] = g;
    cuda_out_hessians[data_index] = h;
  }
}

template <RegFusedLoss LOSS>
bool LaunchFusedRegressionScoreGrad(
  const int* cuda_data_index_to_leaf_index,
  const double* cuda_leaf_value,
  double* cuda_scores,
  const label_t* cuda_labels,
  const label_t* cuda_weights,
  const data_size_t num_data,
  const double param,
  score_t* cuda_gradients,
  score_t* cuda_hessians) {
  if (num_data <= 0) {
    return false;
  }
  const bool pdl = PdlLevelChainEnabled();
#if !defined(SCORE_T_USE_DOUBLE) && !defined(LABEL_T_USE_DOUBLE)
  if (IsAligned8(cuda_data_index_to_leaf_index) && IsAligned16(cuda_scores) &&
      IsAligned8(cuda_labels) && IsAligned8(cuda_gradients) && IsAligned8(cuda_hessians) &&
      (cuda_weights == nullptr || IsAligned8(cuda_weights))) {
    const data_size_t num_pairs = (num_data + 1) / 2;
    const int num_blocks = CUDABlockCount(num_pairs, kFusedScoreGradBlockSize);
    LaunchLevelChainKernel(
      pdl,
      cuda_weights == nullptr ? AddPredictionAndGetGradientsKernel_Regression_Vec2<LOSS, false>
                              : AddPredictionAndGetGradientsKernel_Regression_Vec2<LOSS, true>,
      dim3(num_blocks), dim3(kFusedScoreGradBlockSize), 0, nullptr,
      cuda_data_index_to_leaf_index, cuda_leaf_value, cuda_scores, cuda_labels,
      cuda_weights, num_data, param, cuda_gradients, cuda_hessians);
    return true;
  }
#endif  // !SCORE_T_USE_DOUBLE && !LABEL_T_USE_DOUBLE
  const int num_blocks = CUDABlockCount(num_data, kFusedScoreGradBlockSize);
  LaunchLevelChainKernel(
    pdl,
    cuda_weights == nullptr ? AddPredictionAndGetGradientsKernel_Regression_Scalar<LOSS, false>
                            : AddPredictionAndGetGradientsKernel_Regression_Scalar<LOSS, true>,
    dim3(num_blocks), dim3(kFusedScoreGradBlockSize), 0, nullptr,
    cuda_data_index_to_leaf_index, cuda_leaf_value, cuda_scores, cuda_labels,
    cuda_weights, num_data, param, cuda_gradients, cuda_hessians);
  return true;
}

}  // namespace

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
    const label_t w = USE_WEIGHT ? cuda_weights[data_index] : static_cast<label_t>(1.0f);
    score_t g, h;
    ComputeRegressionL2GradHess<USE_WEIGHT>(cuda_scores[data_index], cuda_labels[data_index], w, &g, &h);
    cuda_out_gradients[data_index] = g;
    cuda_out_hessians[data_index] = h;
  }
}

#if !defined(SCORE_T_USE_DOUBLE) && !defined(LABEL_T_USE_DOUBLE)
template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_RegressionL2_Vec2(
  const double* __restrict__ cuda_scores,
  const label_t* __restrict__ cuda_labels,
  const label_t* __restrict__ cuda_weights,
  const data_size_t num_data,
  score_t* __restrict__ cuda_out_gradients,
  score_t* __restrict__ cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t pair_idx = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  const data_size_t base = pair_idx * 2;
  if (base + 1 < num_data) {
    const double2 s = reinterpret_cast<const double2*>(cuda_scores)[pair_idx];
    const float2 lbl = reinterpret_cast<const float2*>(cuda_labels)[pair_idx];
    const float2 w = USE_WEIGHT ? reinterpret_cast<const float2*>(cuda_weights)[pair_idx] : make_float2(1.0f, 1.0f);
    float2 g, h;
    ComputeRegressionL2GradHess<USE_WEIGHT>(s.x, lbl.x, w.x, &g.x, &h.x);
    ComputeRegressionL2GradHess<USE_WEIGHT>(s.y, lbl.y, w.y, &g.y, &h.y);
    reinterpret_cast<float2*>(cuda_out_gradients)[pair_idx] = g;
    reinterpret_cast<float2*>(cuda_out_hessians)[pair_idx] = h;
  } else if (base < num_data) {
    const label_t w = USE_WEIGHT ? cuda_weights[base] : static_cast<label_t>(1.0f);
    score_t g, h;
    ComputeRegressionL2GradHess<USE_WEIGHT>(cuda_scores[base], cuda_labels[base], w, &g, &h);
    cuda_out_gradients[base] = g;
    cuda_out_hessians[base] = h;
  }
}
#endif  // !SCORE_T_USE_DOUBLE && !LABEL_T_USE_DOUBLE

void CUDARegressionL2loss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  if (num_data_ <= 0) {
    return;
  }
#if !defined(SCORE_T_USE_DOUBLE) && !defined(LABEL_T_USE_DOUBLE)
  if (IsAligned16(score) && IsAligned8(cuda_labels_) && IsAligned8(gradients) && IsAligned8(hessians) &&
      (cuda_weights_ == nullptr || IsAligned8(cuda_weights_))) {
    const data_size_t num_pairs = (num_data_ + 1) / 2;
    const int num_blocks = CUDABlockCount(num_pairs, kFusedScoreGradBlockSize);
    if (cuda_weights_ == nullptr) {
      GetGradientsKernel_RegressionL2_Vec2<false><<<num_blocks, kFusedScoreGradBlockSize>>>(
        score, cuda_labels_, nullptr, num_data_, gradients, hessians);
    } else {
      GetGradientsKernel_RegressionL2_Vec2<true><<<num_blocks, kFusedScoreGradBlockSize>>>(
        score, cuda_labels_, cuda_weights_, num_data_, gradients, hessians);
    }
    return;
  }
#endif  // !SCORE_T_USE_DOUBLE && !LABEL_T_USE_DOUBLE
  const int num_blocks = CUDABlockCount(num_data_, GET_GRADIENTS_BLOCK_SIZE_REGRESSION);
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_RegressionL2<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, gradients, hessians);
  } else {
    GetGradientsKernel_RegressionL2<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, gradients, hessians);
  }
}

bool CUDARegressionL2loss::LaunchAddPredictionToScoreAndGetGradients(
  const int* cuda_data_index_to_leaf_index, const double* cuda_leaf_value,
  double* cuda_scores, score_t* cuda_gradients, score_t* cuda_hessians) const {
  return LaunchFusedRegressionScoreGrad<RegFusedLoss::kL2>(
    cuda_data_index_to_leaf_index, cuda_leaf_value, cuda_scores, cuda_labels_,
    cuda_weights_, num_data_, 0.0, cuda_gradients, cuda_hessians);
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

bool CUDARegressionL1loss::LaunchAddPredictionToScoreAndGetGradients(
  const int* cuda_data_index_to_leaf_index, const double* cuda_leaf_value,
  double* cuda_scores, score_t* cuda_gradients, score_t* cuda_hessians) const {
  return LaunchFusedRegressionScoreGrad<RegFusedLoss::kL1>(
    cuda_data_index_to_leaf_index, cuda_leaf_value, cuda_scores, cuda_labels_,
    cuda_weights_, num_data_, 0.0, cuda_gradients, cuda_hessians);
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

bool CUDARegressionHuberLoss::LaunchAddPredictionToScoreAndGetGradients(
  const int* cuda_data_index_to_leaf_index, const double* cuda_leaf_value,
  double* cuda_scores, score_t* cuda_gradients, score_t* cuda_hessians) const {
  return LaunchFusedRegressionScoreGrad<RegFusedLoss::kHuber>(
    cuda_data_index_to_leaf_index, cuda_leaf_value, cuda_scores, cuda_labels_,
    cuda_weights_, num_data_, alpha_, cuda_gradients, cuda_hessians);
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

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_Poisson(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  const double max_delta_step, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  const double exp_max_delta_step = std::exp(max_delta_step);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      const double exp_score = exp(cuda_scores[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>(exp_score - cuda_labels[data_index]);
      cuda_out_hessians[data_index] = static_cast<score_t>(exp_score * exp_max_delta_step);
    } else {
      const double exp_score = exp(cuda_scores[data_index]);
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>((exp_score - cuda_labels[data_index]) * weight);
      cuda_out_hessians[data_index] = static_cast<score_t>(exp_score * exp_max_delta_step * weight);
    }
  }
}

void CUDARegressionPoissonLoss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = CUDABlockCount(num_data_, GET_GRADIENTS_BLOCK_SIZE_REGRESSION);
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_Poisson<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(
      score, cuda_labels_, nullptr, num_data_, max_delta_step_, gradients, hessians);
  } else {
    GetGradientsKernel_Poisson<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(
      score, cuda_labels_, cuda_weights_, num_data_, max_delta_step_, gradients, hessians);
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

bool CUDARegressionQuantileloss::LaunchAddPredictionToScoreAndGetGradients(
  const int* cuda_data_index_to_leaf_index, const double* cuda_leaf_value,
  double* cuda_scores, score_t* cuda_gradients, score_t* cuda_hessians) const {
  return LaunchFusedRegressionScoreGrad<RegFusedLoss::kQuantile>(
    cuda_data_index_to_leaf_index, cuda_leaf_value, cuda_scores, cuda_labels_,
    cuda_weights_, num_data_, alpha_, cuda_gradients, cuda_hessians);
}

}  // namespace Falcata

#endif  // USE_CUDA
