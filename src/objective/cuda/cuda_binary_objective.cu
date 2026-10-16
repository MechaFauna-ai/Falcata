/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifdef USE_CUDA

#include "cuda_binary_objective.hpp"

#include "../../treelearner/cuda/cuda_pdl_launch.hpp"

#include <Falcata/cuda/cuda_rocm_interop.h>
#include <Falcata/falcata_plan.h>

#include <algorithm>
#include <type_traits>

namespace Falcata {

// The label (and weight) sums of the init score: each block's partial sum goes to its own slot of block_sums
// (labels in [0, num_blocks), weights in [num_blocks, 2 * num_blocks)), and BoostFromScoreKernel_Sum_BinaryLogloss
// adds the slots in block order. (The blocks used to atomicAdd their partials into one double, in whatever order they
// finished: with weights whose sums are not exact in double, e.g. exp(U(-20, 20)), the init score, and with it the
// whole model, changed from run to run. Block order is one of the orders the atomics could take, so every exact sum
// -- labels, unweighted counts, weights of a few binades -- is unchanged.)
template <bool USE_WEIGHT>
__global__ void BoostFromScoreKernel_1_BinaryLogloss(const label_t* cuda_labels, const data_size_t num_data,
                                                     double* block_sums, const label_t* cuda_weights) {
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
    block_sums[blockIdx.x] = label_value;
    if (USE_WEIGHT) {
      block_sums[gridDim.x + blockIdx.x] = weight_value;
    }
  }
}

// one thread: the block partials added in block order, from 0.0 like the atomics they replace
template <bool USE_WEIGHT>
__global__ void BoostFromScoreKernel_Sum_BinaryLogloss(const double* block_sums, const int num_blocks,
                                                       double* out_cuda_sum_labels, double* out_cuda_sum_weights) {
  double suml = 0.0;
  double sumw = 0.0;
  for (int b = 0; b < num_blocks; ++b) {
    suml += block_sums[b];
    if (USE_WEIGHT) {
      sumw += block_sums[num_blocks + b];
    }
  }
  *out_cuda_sum_labels = suml;
  if (USE_WEIGHT) {
    *out_cuda_sum_weights = sumw;
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
  CHECK_GE(cuda_init_block_sums_.Size(), 2 * static_cast<size_t>(num_blocks));
  if (cuda_weights_ == nullptr) {
    BoostFromScoreKernel_1_BinaryLogloss<false><<<num_blocks, CALC_INIT_SCORE_BLOCK_SIZE_BINARY>>>
      (cuda_label_, num_data_, cuda_init_block_sums_.RawData(), cuda_weights_);
    BoostFromScoreKernel_Sum_BinaryLogloss<false><<<1, 1>>>(cuda_init_block_sums_.RawData(), num_blocks,
                                                            cuda_boost_from_score_.RawData(),
                                                            cuda_sum_weights_.RawData());
  } else {
    BoostFromScoreKernel_1_BinaryLogloss<true><<<num_blocks, CALC_INIT_SCORE_BLOCK_SIZE_BINARY>>>
      (cuda_label_, num_data_, cuda_init_block_sums_.RawData(), cuda_weights_);
    BoostFromScoreKernel_Sum_BinaryLogloss<true><<<1, 1>>>(cuda_init_block_sums_.RawData(), num_blocks,
                                                           cuda_boost_from_score_.RawData(),
                                                           cuda_sum_weights_.RawData());
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
// error < 2^-23; zero below float's normal range, infinite above it), LeanFloatToDouble is exact for normal floats and zeros.
__device__ __forceinline__ float LeanApproxFloat(const double v) {
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

__device__ __forceinline__ double LeanFloatToDouble(const float f) {
  const unsigned int fbits = static_cast<unsigned int>(__float_as_int(f));
  const unsigned long long sign = static_cast<unsigned long long>(fbits >> 31) << 63;  // NOLINT(runtime/int)
  if ((fbits & 0x7fffffffu) == 0) {
    return __longlong_as_double(static_cast<long long>(sign));  // NOLINT(runtime/int)
  }
  const unsigned long long biased = ((fbits >> 23) & 0xffu) - 127 + 1023;  // NOLINT(runtime/int)
  return __longlong_as_double(static_cast<long long>(sign | (biased << 52) |  // NOLINT(runtime/int)
                                                     (static_cast<unsigned long long>(fbits & 0x7fffffu) << 29)));  // NOLINT(runtime/int)
}

// v's float rounding when every double within margin_ulps of v's double ulps rounds to it: v's distance to the
// nearest float rounding midpoint (|low 29 mantissa bits - 2^28| ulps) exceeds the margin, and v and its rounding lie
// in float's normal range (2^-125 <= |v| < 2^127). Then round-to-nearest is add half an ulp and truncate.
__device__ __forceinline__ bool LeanSettledFloat(const double v, const float margin_ulps, float* out) {
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

__device__ const double kLeanExp2Table[64] = {  // 2^(j/64), j = 0..63, rounded to nearest double
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

// The lean evaluation for |x| <= 80, x = label * sigmoid * score (the original's double): e^x = 2^m 2^(j/64) e^r with
// k = 64 m + j = rint(x 64 / ln2) (from x's truncated float: |k| <= 7400) and r = x - k ln2/64 by two FMAs with ln2/64's
// double split (|r| <= 0.00543, error < 2^-60 absolute); e^r = 1 + r (1 + r (1/2 + r t)), t = 1/3! + r/4! + r^2/5! +
// r^3/6! in fp32 (|r^3 t| < 2.7e-8; t's error incl. the truncated r about 2^-23 relative: about 2^-48 of e^r, < 2^-46.2
// with a loose 2^-21 on t; the series
// truncated after r^6: < 2^-60), the head in double Horner; the table's 2^(j/64) (rounded, 2^-53) and the exact
// power of two: e^x to < 2^-46.5 relative. 1 + e^x in double; the quotient from an fp32 reciprocal of the truncated
// 1 + e^x (< 1.5 * 2^-23 relative: the correctly rounded __frcp_rn, whatever the build's -prec-div / fast-math
// flags) and one double Newton step (< 2^-44.8). The response is within 2^-44.2 of the true
// value, the original's double (libdevice exp, 1 ulp; the rounded sum and division) within 2^-50: margin eps_r =
// 2^-42, i.e. eps_r * 2^53 = 2^11 double ulps of the response's binade (|v| < 2^(e+1)). The hessian A (s - A), A =
// |response|, inherits A's error amplified by s / (s - A) (the cancellation in s - A); both arms round s - A and the
// product: eps_h = 1.02 eps_r s / (s - A) + 2^-49 (2^-48 with weights). Returns false (the caller evaluates the original expression)
// whenever a check fails.
// Weights (USE_LABEL_WEIGHT: label_weight, USE_WEIGHT: sample_weight) multiply the response and the hessian in the
// original's order; each product is rounded in both arms (2^-52 more per product, inside the margins' slack) and must
// stay in double's normal range (LeanWideNormal), where rounding is relative.
// The lean path stores the response and hessian rounded to float, i.e. the original's static_cast<score_t> only when
// score_t is float; with SCORE_T_USE_DOUBLE (meta.h) the original stores the unrounded double, so the key is compiled
// out there (every row runs the original expression) rather than training a different model.
constexpr bool kLeanScoreIsFloat = std::is_same<score_t, float>::value;

__device__ __forceinline__ bool LeanWideNormal(const double v) {
  const int biased = static_cast<int>((static_cast<unsigned long long>(__double_as_longlong(v)) >> 52) & 0x7ffu);  // NOLINT(runtime/int)
  return biased >= 64 && biased <= 2047 - 64;
}

template <bool USE_LABEL_WEIGHT, bool USE_WEIGHT>
__device__ __forceinline__ bool BinaryLoglossLean(const double score, const bool is_pos, const double sigmoid,
                                                  const double label_weight, const double sample_weight,
                                                  float* out_gradient, float* out_hessian) {
  const double ls = is_pos ? sigmoid : -sigmoid;  // label * sigmoid, label = +-1: exact
  const double x = ls * score;
  const unsigned long long x_magnitude =  // NOLINT(runtime/int)
    static_cast<unsigned long long>(__double_as_longlong(x)) & 0x7fffffffffffffffull;  // NOLINT(runtime/int)
  if (!(x_magnitude <= 0x4054000000000000ull)) {  // |x| <= 80 (NaN fails)
    return false;
  }
  const float kf = rintf(LeanApproxFloat(x) * 92.3324826168936f);  // 64 / ln2
  const int k = static_cast<int>(kf);
  const double kd = LeanFloatToDouble(kf);
  double r = fma(-kd, 0x1.62e42fefa39efp-7, x);  // ln2/64, high and low parts
  r = fma(-kd, 0x1.abc9e3b39803fp-62, r);
  const float rf = LeanApproxFloat(r);
  float t = 1.38888888888888889e-03f;  // 1/6!
  t = fmaf(t, rf, 8.33333333333333333e-03f);  // 1/5!
  t = fmaf(t, rf, 4.16666666666666667e-02f);  // 1/4!
  t = fmaf(t, rf, 1.66666666666666667e-01f);  // 1/3!
  double p = fma(LeanFloatToDouble(t), r, 0.5);
  p = fma(p, r, 1.0);
  p = fma(p, r, 1.0);
  const double scaled = p * kLeanExp2Table[k & 63];  // in [0.99, 2.01): 2^m keeps it normal for |m| <= 116
  const double e = __longlong_as_double(__double_as_longlong(scaled) + (static_cast<long long>(k >> 6) << 52));  // NOLINT(runtime/int)
  const double d = 1.0 + e;
  const double inv0 = LeanFloatToDouble(__frcp_rn(LeanApproxFloat(d)));  // IEEE round-to-nearest: the bound's premise
  const double num = -ls;
  const double q0 = num * inv0;
  const double q = fma(fma(-d, q0, num), inv0, q0);
  const double a = fabs(q);
  const double gap = sigmoid - a;
  double g = q;
  double h = a * gap;
  if (USE_LABEL_WEIGHT) {
    if (!LeanWideNormal(g) || !LeanWideNormal(h)) {
      return false;
    }
    g = g * label_weight;
    h = h * label_weight;
  }
  if (USE_WEIGHT) {
    if (!LeanWideNormal(g) || !LeanWideNormal(h)) {
      return false;
    }
    g = g * sample_weight;
    h = h * sample_weight;
  }
  const float gap_f = LeanApproxFloat(gap);
  if (!(gap_f > 0.0f)) {
    return false;
  }
  const float ratio = LeanApproxFloat(sigmoid) / gap_f;
  float g_out, h_out;
  if (!LeanSettledFloat(g, 2048.0f + 16.0f, &g_out) || !LeanSettledFloat(h, 1.03f * 2048.0f * ratio + 32.0f, &h_out)) {
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
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    const label_t cuda_label = static_cast<int>(cuda_labels[data_index]);
    const int is_pos = cuda_label > 0 ? 1 : 0;
    if (kLeanScoreIsFloat && lean) {
      float lean_gradient, lean_hessian;
      const double lean_label_weight = USE_LABEL_WEIGHT ? cuda_label_weights[is_pos] : 1.0;
      const double lean_sample_weight = USE_WEIGHT ? static_cast<double>(cuda_weights[data_index]) : 1.0;
      if (BinaryLoglossLean<USE_LABEL_WEIGHT, USE_WEIGHT>(cuda_scores[data_index], is_pos != 0, sigmoid,
                                                          lean_label_weight, lean_sample_weight, &lean_gradient,
                                                          &lean_hessian)) {
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
