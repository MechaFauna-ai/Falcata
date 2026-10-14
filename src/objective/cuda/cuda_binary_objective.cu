/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include "cuda_binary_objective.hpp"

#include <Falcata/cuda/cuda_rocm_interop.h>
#include <Falcata/falcata_plan.h>

#include <algorithm>

namespace Falcata {

template <bool USE_WEIGHT>
__global__ void BoostFromScoreKernel_1_BinaryLogloss(const label_t* cuda_labels, const data_size_t num_data, double* out_cuda_sum_labels,
                                                     double* out_cuda_sum_weights, const label_t* cuda_weights) {
  __shared__ double shared_buffer[WARPSIZE];
  const uint32_t mask = 0xffffffff;
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t num_warp = blockDim.x / warpSize;
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double label_value = 0.0;
  double weight_value = 0.0;
  if (index < num_data) {
    if (USE_WEIGHT) {
      const label_t cuda_label = cuda_labels[index];
      const double sample_weight = cuda_weights[index];
      const label_t label = cuda_label > 0 ? 1 : 0;
      label_value = label * sample_weight;
      weight_value = sample_weight;
    } else {
      const label_t cuda_label = cuda_labels[index];
      label_value = cuda_label > 0 ? 1 : 0;
    }
  }
  for (uint32_t offset = warpSize / 2; offset >= 1; offset >>= 1) {
    label_value += __shfl_down_sync(mask, label_value, offset);
  }
  if (warpLane == 0) {
    shared_buffer[warpID] = label_value;
  }
  __syncthreads();
  if (warpID == 0) {
    label_value = (warpLane < num_warp ? shared_buffer[warpLane] : 0);
    for (uint32_t offset = warpSize / 2; offset >= 1; offset >>= 1) {
      label_value += __shfl_down_sync(mask, label_value, offset);
    }
  }
  __syncthreads();
  if (USE_WEIGHT) {
    for (uint32_t offset = warpSize / 2; offset >= 1; offset >>= 1) {
      weight_value += __shfl_down_sync(mask, weight_value, offset);
    }
    if (warpLane == 0) {
      shared_buffer[warpID] = weight_value;
    }
    __syncthreads();
    if (warpID == 0) {
      weight_value = (warpLane < num_warp ? shared_buffer[warpLane] : 0);
      for (uint32_t offset = warpSize / 2; offset >= 1; offset >>= 1) {
        weight_value += __shfl_down_sync(mask, weight_value, offset);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd_system(out_cuda_sum_labels, label_value);
    if (USE_WEIGHT) {
      atomicAdd_system(out_cuda_sum_weights, weight_value);
    }
  }
}

template <bool USE_WEIGHT>
__global__ void BoostFromScoreKernel_2_BinaryLogloss(double* out_cuda_sum_labels, double* out_cuda_sum_weights,
                                                     const data_size_t num_data, const double sigmoid) {
  const double suml = *out_cuda_sum_labels;
  const double sumw = USE_WEIGHT ? *out_cuda_sum_weights : static_cast<double>(num_data);
  double pavg = suml / sumw;
  pavg = min(pavg, 1.0 - kEpsilon);
  pavg = max(pavg, kEpsilon);
  const double init_score = log(pavg / (1.0f - pavg)) / sigmoid;
  *out_cuda_sum_weights = pavg;
  *out_cuda_sum_labels = init_score;
}

double CUDABinaryLogloss::LaunchCalcInitScoreKernel(const int /*class_id*/) const {
  const int num_blocks = (num_data_ + CALC_INIT_SCORE_BLOCK_SIZE_BINARY - 1) / CALC_INIT_SCORE_BLOCK_SIZE_BINARY;
  SetCUDAMemory<double>(cuda_boost_from_score_.RawData(), 0, 1, __FILE__, __LINE__);
  if (cuda_weights_ == nullptr) {
    BoostFromScoreKernel_1_BinaryLogloss<false><<<num_blocks, CALC_INIT_SCORE_BLOCK_SIZE_BINARY>>>
      (cuda_label_, num_data_, cuda_boost_from_score_.RawData(), cuda_sum_weights_.RawData(), cuda_weights_);
  } else {
    BoostFromScoreKernel_1_BinaryLogloss<true><<<num_blocks, CALC_INIT_SCORE_BLOCK_SIZE_BINARY>>>
      (cuda_label_, num_data_, cuda_boost_from_score_.RawData(), cuda_sum_weights_.RawData(), cuda_weights_);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  if (cuda_weights_ == nullptr) {
    if (nccl_communicator_ == nullptr) {
      BoostFromScoreKernel_2_BinaryLogloss<false><<<1, 1>>>(cuda_boost_from_score_.RawData(), cuda_sum_weights_.RawData(), num_data_, sigmoid_);
    } else {
      NCCLAllReduce<double>(cuda_boost_from_score_.RawData(), cuda_boost_from_score_.RawData(), 1, ncclFloat64, ncclSum, nccl_communicator_);
      const data_size_t global_num_data = NCCLAllReduce<data_size_t>(num_data_, ncclInt32, ncclSum, nccl_communicator_);
      BoostFromScoreKernel_2_BinaryLogloss<false><<<1, 1>>>(cuda_boost_from_score_.RawData(), cuda_sum_weights_.RawData(), global_num_data, sigmoid_);
    }
  } else {
    if (nccl_communicator_ == nullptr) {
      BoostFromScoreKernel_2_BinaryLogloss<true><<<1, 1>>>(cuda_boost_from_score_.RawData(), cuda_sum_weights_.RawData(), num_data_, sigmoid_);
    } else {
      NCCLAllReduce<double>(cuda_boost_from_score_.RawData(), cuda_boost_from_score_.RawData(), 1, ncclFloat64, ncclSum, nccl_communicator_);
      NCCLAllReduce<double>(cuda_sum_weights_.RawData(), cuda_sum_weights_.RawData(), 1, ncclFloat64, ncclSum, nccl_communicator_);
      BoostFromScoreKernel_2_BinaryLogloss<true><<<1, 1>>>(cuda_boost_from_score_.RawData(), cuda_sum_weights_.RawData(), num_data_, sigmoid_);
    }
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  double boost_from_score = 0.0f;
  CopyFromCUDADeviceToHost<double>(&boost_from_score, cuda_boost_from_score_.RawData(), 1, __FILE__, __LINE__);
  double pavg = 0.0f;
  CopyFromCUDADeviceToHost<double>(&pavg, cuda_sum_weights_.RawData(), 1, __FILE__, __LINE__);
  // for some test cases in test_utilities.py which check the log output
  Log::Info("[%s:%s]: pavg=%f -> initscore=%f",  GetName(), "BoostFromScore", pavg, boost_from_score);
  return boost_from_score;
}

// cuda_plan key gradient_fp64_lean: the logloss response and hessian with ~16 FP64 operations per row instead of ~50
// (libdevice exp + a correctly rounded division), stored only where the floats provably equal the original's.
// Only the float casts of the double response and hessian are stored, and round-to-nearest is monotone: if every double
// within the evaluation's error bound of an approximation rounds to the same float, the original double does too.
//
// Normal-range double <-> float by bit manipulation (integer pipe): ApproxFloatBits truncates the mantissa (relative
// error < 2^-23; zero below float's normal range, infinite above it), FloatToDoubleBits is exact for normal floats.
__device__ __forceinline__ float LeanApproxFloat(const double v) {
  const unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(v));
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

__device__ __forceinline__ double LeanFloatToDouble(const float f) {
  const unsigned int fbits = static_cast<unsigned int>(__float_as_int(f));
  const unsigned long long sign = static_cast<unsigned long long>(fbits >> 31) << 63;
  const unsigned long long biased = ((fbits >> 23) & 0xffu) - 127 + 1023;
  return __longlong_as_double(static_cast<long long>(sign | (biased << 52) |
                                                     (static_cast<unsigned long long>(fbits & 0x7fffffu) << 29)));
}

// v's float rounding when every double within margin_ulps of v's double ulps rounds to it: v's distance to the
// nearest float rounding midpoint (|low 29 mantissa bits - 2^28| ulps) exceeds the margin, and v and its rounding lie
// in float's normal range (2^-125 <= |v| < 2^127). Then round-to-nearest is add half an ulp and truncate.
__device__ __forceinline__ bool LeanSettledFloat(const double v, const float margin_ulps, float* out) {
  const unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(v));
  const int biased = static_cast<int>((bits >> 52) & 0x7ffu);
  const int low = static_cast<int>(bits & ((1ull << 29) - 1));
  const int dist = low >= (1 << 28) ? low - (1 << 28) : (1 << 28) - low;
  if (biased < 1023 - 125 || biased > 1023 + 126 || !(margin_ulps < static_cast<float>(1 << 27)) ||
      dist <= static_cast<int>(margin_ulps) + 1) {
    return false;
  }
  const unsigned long long magnitude = (bits & 0x7fffffffffffffffull) + (1ull << 28);
  const unsigned int fbits = static_cast<unsigned int>(bits >> 63) << 31 |
                             (static_cast<unsigned int>(magnitude >> 29) - (static_cast<unsigned int>(1023 - 127) << 23));
  *out = __int_as_float(static_cast<int>(fbits));
  return true;
}

// The lean evaluation for |x| <= 80, x = label * sigmoid * score (the original's double): e^x = 2^k e^r with Cody-Waite
// r = x - k ln2 (|r| <= 0.3466; two FMAs with ln2's double split: error < 2^-54 absolute), e^r's head
// 1 + r + ... + r^5/120 + r^6 u in double Horner and its tail u = 1/6! + ... + r^5/11! in fp32 (|r^6 u| < 2.6e-6, the
// fp32 tail's error incl. the truncated r (< 2^-23 relative) < 2^-38.7, the series truncated after r^11: < 2^-46), so
// e^x to < 2^-38.5 relative; 1 + e^x in double; the quotient from an fp32 reciprocal (< 2^-22 relative) and one
// double Newton step (< 2^-43). The response is within 2^-38.3 of the true value, the original's double (libdevice
// exp, 1 ulp; the rounded sum and division) within 2^-50: margin eps_r = 2^-36, i.e. eps_r * 2^53 = 2^17 double ulps
// of the response's binade (|v| < 2^(e+1)). The hessian A (s - A), A = |response|, inherits A's error amplified by
// s / (s - A) (the cancellation in s - A); both arms round s - A and the product: eps_h = 1.02 eps_r s / (s - A) +
// 2^-49. Returns false (the caller evaluates the original expression) whenever a check fails.
__device__ __forceinline__ bool BinaryLoglossLean(const double score, const bool is_pos, const double sigmoid,
                                                  float* out_gradient, float* out_hessian) {
  const double ls = is_pos ? sigmoid : -sigmoid;  // label * sigmoid, label = +-1: exact
  const double x = ls * score;
  const unsigned long long x_magnitude =
    static_cast<unsigned long long>(__double_as_longlong(x)) & 0x7fffffffffffffffull;
  if (!(x_magnitude <= 0x4054000000000000ull)) {  // |x| <= 80 (NaN fails)
    return false;
  }
  const float kf = rintf(LeanApproxFloat(x) * 1.44269504088896341f);
  const int k = static_cast<int>(kf);  // |k| <= 116
  const double kd = static_cast<double>(k);
  double r = fma(-kd, 6.93147180559945286e-01, x);
  r = fma(-kd, 2.31904681384629956e-17, r);
  const float rf = LeanApproxFloat(r);
  float u = 2.50521083854417188e-08f;            // 1/11!
  u = fmaf(u, rf, 2.75573192239858907e-07f);     // 1/10!
  u = fmaf(u, rf, 2.75573192239858907e-06f);     // 1/9!
  u = fmaf(u, rf, 2.48015873015873016e-05f);     // 1/8!
  u = fmaf(u, rf, 1.98412698412698413e-04f);     // 1/7!
  u = fmaf(u, rf, 1.38888888888888889e-03f);     // 1/6!
  double p = fma(LeanFloatToDouble(u), r, 8.33333333333333333e-03);
  p = fma(p, r, 4.16666666666666667e-02);
  p = fma(p, r, 1.66666666666666667e-01);
  p = fma(p, r, 0.5);
  p = fma(p, r, 1.0);
  p = fma(p, r, 1.0);
  const double e = p * __longlong_as_double(static_cast<long long>(k + 1023) << 52);
  const double d = 1.0 + e;
  const double inv0 = LeanFloatToDouble(1.0f / LeanApproxFloat(d));
  const double num = -ls;
  const double q0 = num * inv0;
  const double q = fma(fma(-d, q0, num), inv0, q0);
  const double a = fabs(q);
  const double gap = sigmoid - a;
  const double h = a * gap;
  const float gap_f = LeanApproxFloat(gap);
  if (!(gap_f > 0.0f)) {
    return false;
  }
  const float ratio = LeanApproxFloat(sigmoid) / gap_f;
  float g_out, h_out;
  if (!LeanSettledFloat(q, 131072.0f, &g_out) || !LeanSettledFloat(h, 1.03f * 131072.0f * ratio + 16.0f, &h_out)) {
    return false;
  }
  *out_gradient = g_out;
  *out_hessian = h_out;
  return true;
}

template <bool USE_LABEL_WEIGHT, bool USE_WEIGHT>
__global__ void GetGradientsKernel_BinaryLogloss(const double* cuda_scores, const label_t* cuda_labels,
  const double* cuda_label_weights, const label_t* cuda_weights,
  const double sigmoid, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians, const bool lean) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    const label_t cuda_label = static_cast<int>(cuda_labels[data_index]);
    const int is_pos = cuda_label > 0 ? 1 : 0;
    if (!USE_WEIGHT && !USE_LABEL_WEIGHT && lean) {
      float lean_gradient, lean_hessian;
      if (BinaryLoglossLean(cuda_scores[data_index], is_pos != 0, sigmoid, &lean_gradient, &lean_hessian)) {
        cuda_out_gradients[data_index] = lean_gradient;
        cuda_out_hessians[data_index] = lean_hessian;
        return;
      }
    }
    const int label = is_pos ? 1 : -1;
    const double response = -label * sigmoid / (1.0f + exp(label * sigmoid * cuda_scores[data_index]));
    const double abs_response = fabs(response);
    if (!USE_WEIGHT) {
      if (USE_LABEL_WEIGHT) {
        // label_weights is indexed {0,1} like the CPU's label_weights_[is_pos];
        // indexing it with label {-1,+1} read one double BEFORE the allocation
        // for every negative sample (is_unbalance /
        // scale_pos_weight on device=cuda trained garbage, AUC 0.500 on fraud)
        const double label_weight = cuda_label_weights[is_pos];
        cuda_out_gradients[data_index] = static_cast<score_t>(response * label_weight);
        cuda_out_hessians[data_index] = static_cast<score_t>(abs_response * (sigmoid - abs_response) * label_weight);
      } else {
        cuda_out_gradients[data_index] = static_cast<score_t>(response);
        cuda_out_hessians[data_index] = static_cast<score_t>(abs_response * (sigmoid - abs_response));
      }
    } else {
      const double sample_weight = cuda_weights[data_index];
      if (USE_LABEL_WEIGHT) {
        const double label_weight = cuda_label_weights[is_pos];
        cuda_out_gradients[data_index] = static_cast<score_t>(response * label_weight * sample_weight);
        cuda_out_hessians[data_index] = static_cast<score_t>(abs_response * (sigmoid - abs_response) * label_weight * sample_weight);
      } else {
        cuda_out_gradients[data_index] = static_cast<score_t>(response * sample_weight);
        cuda_out_hessians[data_index] = static_cast<score_t>(abs_response * (sigmoid - abs_response) * sample_weight);
      }
    }
  }
}

#define GetGradientsKernel_BinaryLogloss_ARGS \
  scores, \
  cuda_label_, \
  cuda_label_weights_.RawData(), \
  cuda_weights_, \
  sigmoid_, \
  num_data_, \
  gradients, \
  hessians, \
  FalcataPlan::Get().gradient_fp64_lean

void CUDABinaryLogloss::LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_BINARY - 1) / GET_GRADIENTS_BLOCK_SIZE_BINARY;
  if (cuda_label_weights_.Size() == 0) {
    if (cuda_weights_ == nullptr) {
      GetGradientsKernel_BinaryLogloss<false, false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(GetGradientsKernel_BinaryLogloss_ARGS);
    } else {
      GetGradientsKernel_BinaryLogloss<false, true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(GetGradientsKernel_BinaryLogloss_ARGS);
    }
  } else {
    if (cuda_weights_ == nullptr) {
      GetGradientsKernel_BinaryLogloss<true, false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(GetGradientsKernel_BinaryLogloss_ARGS);
    } else {
      GetGradientsKernel_BinaryLogloss<true, true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(GetGradientsKernel_BinaryLogloss_ARGS);
    }
  }
}

#undef GetGradientsKernel_BinaryLogloss_ARGS

__global__ void ConvertOutputCUDAKernel_BinaryLogloss(const double sigmoid, const data_size_t num_data, const double* input, double* output) {
  const data_size_t data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (data_index < num_data) {
    output[data_index] = 1.0f / (1.0f + exp(-sigmoid * input[data_index]));
  }
}

const double* CUDABinaryLogloss::LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const {
  const int num_blocks = (num_data + GET_GRADIENTS_BLOCK_SIZE_BINARY - 1) / GET_GRADIENTS_BLOCK_SIZE_BINARY;
  ConvertOutputCUDAKernel_BinaryLogloss<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(sigmoid_, num_data, input, output);
  return output;
}

__global__ void ResetOVACUDALabelKernel(
  const int ova_class_id,
  const data_size_t num_data,
  label_t* cuda_label) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    const int int_label = static_cast<int>(cuda_label[data_index]);
    cuda_label[data_index] = (int_label == ova_class_id ? 1.0f : 0.0f);
  }
}

void CUDABinaryLogloss::LaunchResetOVACUDALabelKernel() const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_BINARY - 1) / GET_GRADIENTS_BLOCK_SIZE_BINARY;
  ResetOVACUDALabelKernel<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(ova_class_id_, num_data_, cuda_ova_label_.RawData());
}

}  // namespace Falcata

#endif  // USE_CUDA
