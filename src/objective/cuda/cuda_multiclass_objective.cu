/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <Falcata/falcata_plan.h>

#include <algorithm>
#include <type_traits>

#include "cuda_multiclass_objective.hpp"

#include "../../treelearner/cuda/cuda_pdl_launch.hpp"

namespace Falcata {

// Softmax over a row's len values, which sit `stride` elements apart: the scratch is class-major (class k of row r at
// k * num_rows + r, the layout of the scores), so the threads of a warp, consecutive rows, touch consecutive doubles
// on every pass. (A row-major scratch put them num_class doubles apart, an uncoalesced access per pass.)
// Softmax in double on purpose: the CPU objective computes it in double, and a float softmax changes most
// gradients and breaks CPU parity.
__device__ void SoftmaxCUDA(double* softmax_buffer, const int len, const size_t stride) {
  double wmax = softmax_buffer[0];
  for (int i = 1; i < len; ++i) {
    wmax = max(softmax_buffer[i * stride], wmax);
  }
  double wsum = 0.0f;
  for (int i = 0; i < len; ++i) {
    softmax_buffer[i * stride] = exp(softmax_buffer[i * stride] - wmax);
    wsum += softmax_buffer[i * stride];
  }
  for (int i = 0; i < len; ++i) {
    softmax_buffer[i * stride] /= static_cast<double>(wsum);
  }
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_MulticlassSoftmax(
  const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights,
  const double factor, const int num_class, const data_size_t num_data,
  double* cuda_softmax_buffer, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    const size_t stride = static_cast<size_t>(num_data);
    double* softmax_result = cuda_softmax_buffer + data_index;
    for (int k = 0; k < num_class; ++k) {
      softmax_result[k * stride] = cuda_scores[stride * k + data_index];
    }
    SoftmaxCUDA(softmax_result, num_class, stride);
    if (!USE_WEIGHT) {
      for (int k = 0; k < num_class; ++k) {
        const double p = softmax_result[k * stride];
        size_t idx = static_cast<size_t>(num_data) * k + data_index;
        if (static_cast<int>(cuda_labels[data_index]) == k) {
          cuda_out_gradients[idx] = static_cast<score_t>(p - 1.0f);
        } else {
          cuda_out_gradients[idx] = static_cast<score_t>(p);
        }
        cuda_out_hessians[idx] = static_cast<score_t>(factor * p * (1.0f - p));
      }
    } else {
      for (int k = 0; k < num_class; ++k) {
        const double p = softmax_result[k * stride];
        const double weight = cuda_weights[data_index];
        size_t idx = static_cast<size_t>(num_data) * k + data_index;
        if (static_cast<int>(cuda_labels[data_index]) == k) {
          cuda_out_gradients[idx] = static_cast<score_t>((p - 1.0f) * weight);
        } else {
          cuda_out_gradients[idx] = static_cast<score_t>(p * weight);
        }
        cuda_out_hessians[idx] = static_cast<score_t>((factor * p * (1.0f - p)) * weight);
      }
    }
  }
}

// cuda_plan key softmax_registers: GetGradientsKernel_MulticlassSoftmax with the row's class scores in registers
// instead of the global scratch (num_class doubles per thread, 6 passes over it). MAX_CLASS is a capacity:
// classes k >= num_class are never read, and the loops (unrolled over the capacity) run SoftmaxCUDA's operations in
// its order: the max from class 0 up, the exponentials summed from 0.0 in class order, one division per class. The
// same doubles: bit-identical.
template <bool USE_WEIGHT, int MAX_CLASS>
__global__ void __launch_bounds__(GET_GRADIENTS_BLOCK_SIZE_MULTICLASS) GetGradientsKernel_MulticlassSoftmaxRegisters(
  const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights,
  const double factor, const int num_class, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index >= num_data) {
    return;
  }
  double p[MAX_CLASS];
#pragma unroll
  for (int k = 0; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      p[k] = cuda_scores[static_cast<size_t>(num_data) * k + data_index];
    }
  }
  double wmax = p[0];
#pragma unroll
  for (int k = 1; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      wmax = max(p[k], wmax);
    }
  }
  double wsum = 0.0f;
#pragma unroll
  for (int k = 0; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      p[k] = exp(p[k] - wmax);
      wsum += p[k];
    }
  }
  const int label = static_cast<int>(cuda_labels[data_index]);
  const double weight = USE_WEIGHT ? static_cast<double>(cuda_weights[data_index]) : 1.0;
#pragma unroll
  for (int k = 0; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      const double pk = p[k] / static_cast<double>(wsum);
      const size_t idx = static_cast<size_t>(num_data) * k + data_index;
      if (!USE_WEIGHT) {
        cuda_out_gradients[idx] = static_cast<score_t>(label == k ? pk - 1.0f : pk);
        cuda_out_hessians[idx] = static_cast<score_t>(factor * pk * (1.0f - pk));
      } else {
        cuda_out_gradients[idx] = static_cast<score_t>(label == k ? (pk - 1.0f) * weight : pk * weight);
        cuda_out_hessians[idx] = static_cast<score_t>((factor * pk * (1.0f - pk)) * weight);
      }
    }
  }
}

// cuda_plan key softmax_fp64_lean: cancellation-free reduced-FP64 multiclass softmax for num_class <= 32 when
// score_t is float.
//
// Why cancellation-free: the naive lean softmax loses precision in (1 - p[kmax]) and (p[kmax] - 1) as p[kmax] -> 1.
// Instead we find kmax = argmax_k s[k] (e[kmax] = 1.0 exact), compute e[k] = exp(s[k] - wmax) for k != kmax, sum
// S_other = sum_{k != kmax} e[k] >= 0, form wsum = 1.0 + S_other, and compute inv_wsum = 1 / wsum once per row via
// __frcp_rn + one double Newton step. Then:
//   - for k != kmax: p[k] = e[k] * inv_wsum <= 0.5, so 1 - p[k] >= 0.5 and |p[k] - 1| >= 0.5 (zero cancellation);
//   - for k == kmax: 1 - p[kmax] = S_other * inv_wsum = qmax (zero cancellation in our evaluation!).
//
// Error bounds vs the original double expression (N = num_class <= 32):
//   - SoftmaxLeanExpNeg(d) for d in [-700, 0] uses a 512-entry table (k = rint(d_rn * 512/ln2), |r| <= 0.000731,
//     single FMA reduction omitting lo: |k * lo| <= 2.34e-14 < 2^-45.27), fp32 tail = rf^2 * (1/2 + rf/6 + rf^2/24)
//     (error <= 2^-22 * 2.68e-7 = 6.4e-14), dr = r + tail, scaled = fma(dr, tbl, tbl): ONLY 3 FP64 ops per exp,
//     total relative error < 9.0e-14 < 2^-43.3 (measured max 2^-43.79).
//   - S_other (<= 31 non-negative adds): < 2^-43.2; wsum = 1 + S_other: < 2^-43.2; inv_wsum (one Newton step from
//     __frcp_rn(SoftmaxLeanApproxFloatRN(wsum)), error <= 2^-46): < 2^-43.0 relative to 1 / wsum*.
//   - Original kernel's wsum_orig has relative error < 2^-47.9 and absolute error <= (N - 1)*2^-53*wsum + 2^-51*S_other,
//     so p_orig[k] for k != kmax is within 2^-42.0 (< 2048 double ulps) of our p[k]; margin 4096 ulps (2x slack).
//   - For k == kmax:
//     (1) If S_other < 2^-55, then sum_{k != kmax} e_orig[k] < 2^-54 < 0.5 * ulp(1.0), so wsum_orig == 1.0
//         bit-for-bit, p_orig[kmax] == 1.0 exact, g_orig = (label == kmax ? +0.0 : weight), and h_orig = +0.0 exact.
//     (2) Otherwise, original double cancellation in 1 - 1/wsum_orig satisfies |q_orig - q*| <= N * 2^-53 + 2^-50.5*q*,
//         so margin = 4096 (or 6144 for h) + 1.25 * N / SoftmaxLeanApproxFloatRN(qmax).
//     (3) If that kmax margin fails (e.g. S_other in [2^-55, 1e-5]) and S_other < 0.25, we sum e[k] in exact class
//         order using Dekker's FastTwoSum to certify wsum_ord == wsum_orig bit-for-bit (fails only when an addition
//         lands within 2^-41 * S_other of a double midpoint of 1.0, < 0.5% at 1e-6 and < 1e-6 at 1e-10!), then run
//         ONE exact FP64 division 1.0 / wsum_ord for kmax alone without recomputing any exp().
//   - Underflow to +0.0f: for proven non-negative quantities with margin < 2^27 (relative error < 2^-26), any double in
//     [0, 2^-151) implies original double in [0, 2^-150 = 0.5 * FLT_TRUE_MIN), which provably rounds to +0.0f.
//   - Any row still failing a check runs MulticlassSoftmaxExactFallback (the exact original double expression).
constexpr bool kSoftmaxLeanScoreIsFloat = std::is_same<score_t, float>::value;
constexpr int kSoftmaxLeanBlockSize = 256;

// Integer round-to-nearest double -> float in float's normal range (relative error <= 2^-24).
__device__ __forceinline__ float SoftmaxLeanApproxFloatRN(const double v) {
  const unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(v));  // NOLINT(runtime/int)
  const int biased = static_cast<int>((bits >> 52) & 0x7ffu);
  if (biased <= 1023 - 127) {
    return 0.0f;
  }
  if (biased >= 1023 + 127) {
    return (bits >> 63) ? -INFINITY : INFINITY;
  }
  const unsigned long long magnitude = (bits & 0x7fffffffffffffffull) + (1ull << 28);  // NOLINT(runtime/int)
  const unsigned int fbits = (static_cast<unsigned int>(bits >> 63) << 31) |
                             (static_cast<unsigned int>(magnitude >> 29) - (static_cast<unsigned int>(1023 - 127) << 23));
  return __int_as_float(static_cast<int>(fbits));
}

__device__ __forceinline__ double SoftmaxLeanFloatToDouble(const float f) {
  const unsigned int fbits = static_cast<unsigned int>(__float_as_int(f));
  const unsigned long long sign = static_cast<unsigned long long>(fbits >> 31) << 63;  // NOLINT(runtime/int)
  if ((fbits & 0x7fffffffu) == 0) {
    return __longlong_as_double(static_cast<long long>(sign));  // NOLINT(runtime/int)
  }
  const unsigned long long biased = ((fbits >> 23) & 0xffu) - 127 + 1023;  // NOLINT(runtime/int)
  return __longlong_as_double(static_cast<long long>(sign | (biased << 52) |  // NOLINT(runtime/int)
                                                     (static_cast<unsigned long long>(fbits & 0x7fffffu) << 29)));  // NOLINT(runtime/int)
}

__device__ __forceinline__ bool SoftmaxLeanWideNormal(const double v) {
  const int biased = static_cast<int>((static_cast<unsigned long long>(__double_as_longlong(v)) >> 52) & 0x7ffu);  // NOLINT(runtime/int)
  return biased >= 64 && biased <= 2047 - 64;
}

__device__ __forceinline__ bool SoftmaxLeanSettledFloat(const double v, const float margin_ulps, float* out) {
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

__device__ __forceinline__ bool SoftmaxLeanSettledNonNegFloat(const double v, const float margin_ulps, float* out) {
  const unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(v));  // NOLINT(runtime/int)
  if ((bits >> 63) != 0 || !(margin_ulps < static_cast<float>(1 << 27))) {
    return false;
  }
  const int biased = static_cast<int>((bits >> 52) & 0x7ffu);
  if (biased < 1023 - 151) {
    *out = 0.0f;
    return true;
  }
  const int low = static_cast<int>(bits & ((1ull << 29) - 1));
  const int dist = low >= (1 << 28) ? low - (1 << 28) : (1 << 28) - low;
  if (biased < 1023 - 125 || biased > 1023 + 126 || dist <= static_cast<int>(margin_ulps) + 1) {
    return false;
  }
  const unsigned long long magnitude = bits + (1ull << 28);  // NOLINT(runtime/int)
  const unsigned int fbits =
    static_cast<unsigned int>(magnitude >> 29) - (static_cast<unsigned int>(1023 - 127) << 23);
  *out = __int_as_float(static_cast<int>(fbits));
  return true;
}

__device__ const double kSoftmaxLeanExp2Table512[512] = {  // 2^(j/512), j = 0..511, rounded to nearest double
  0x1.0000000000000p+0, 0x1.0058c86da1c0ap+0, 0x1.00b1afa5abcbfp+0, 0x1.010ab5b2cbd11p+0,
  0x1.0163da9fb3335p+0, 0x1.01bd1e77170b4p+0, 0x1.02168143b0281p+0, 0x1.027003103b10ep+0,
  0x1.02c9a3e778061p+0, 0x1.032363d42b027p+0, 0x1.037d42e11bbccp+0, 0x1.03d7411915a8ap+0,
  0x1.04315e86e7f85p+0, 0x1.048b9b35659d8p+0, 0x1.04e5f72f654b1p+0, 0x1.0540727fc1762p+0,
  0x1.059b0d3158574p+0, 0x1.05f5c74f0bec2p+0, 0x1.0650a0e3c1f89p+0, 0x1.06ab99fa6407cp+0,
  0x1.0706b29ddf6dep+0, 0x1.0761ead925493p+0, 0x1.07bd42b72a836p+0, 0x1.0818ba42e7d30p+0,
  0x1.0874518759bc8p+0, 0x1.08d0088f8093fp+0, 0x1.092bdf66607e0p+0, 0x1.0987d61701716p+0,
  0x1.09e3ecac6f383p+0, 0x1.0a402331b9715p+0, 0x1.0a9c79b1f3919p+0, 0x1.0af8f03834e52p+0,
  0x1.0b5586cf9890fp+0, 0x1.0bb23d833d93fp+0, 0x1.0c0f145e46c85p+0, 0x1.0c6c0b6bdae53p+0,
  0x1.0cc922b7247f7p+0, 0x1.0d265a4b520bap+0, 0x1.0d83b23395decp+0, 0x1.0de12a7b26300p+0,
  0x1.0e3ec32d3d1a2p+0, 0x1.0e9c7c55189c6p+0, 0x1.0efa55fdfa9c5p+0, 0x1.0f58503328e6dp+0,
  0x1.0fb66affed31bp+0, 0x1.1014a66f951cep+0, 0x1.1073028d7233ep+0, 0x1.10d17f64d9ef1p+0,
  0x1.11301d0125b51p+0, 0x1.118edb6db2dc1p+0, 0x1.11edbab5e2ab6p+0, 0x1.124cbae51a5c8p+0,
  0x1.12abdc06c31ccp+0, 0x1.130b1e264a0e9p+0, 0x1.136a814f204abp+0, 0x1.13ca058cbae1ep+0,
  0x1.1429aaea92de0p+0, 0x1.1489717425438p+0, 0x1.14e95934f312ep+0, 0x1.154962388149ep+0,
  0x1.15a98c8a58e51p+0, 0x1.1609d83606e12p+0, 0x1.166a45471c3c2p+0, 0x1.16cad3c92df73p+0,
  0x1.172b83c7d517bp+0, 0x1.178c554eaea89p+0, 0x1.17ed48695bbc0p+0, 0x1.184e5d23816c9p+0,
  0x1.18af9388c8deap+0, 0x1.1910eba4df41fp+0, 0x1.1972658375d2fp+0, 0x1.19d4013041dc2p+0,
  0x1.1a35beb6fcb75p+0, 0x1.1a979e2363cf8p+0, 0x1.1af99f8138a1cp+0, 0x1.1b5bc2dc40bf0p+0,
  0x1.1bbe084045cd4p+0, 0x1.1c206fb91588fp+0, 0x1.1c82f95281c6bp+0, 0x1.1ce5a51860746p+0,
  0x1.1d4873168b9aap+0, 0x1.1dab6358e15e8p+0, 0x1.1e0e75eb44027p+0, 0x1.1e71aad999e82p+0,
  0x1.1ed5022fcd91dp+0, 0x1.1f387bf9cda38p+0, 0x1.1f9c18438ce4dp+0, 0x1.1fffd7190241ep+0,
  0x1.2063b88628cd6p+0, 0x1.20c7bc96ffc18p+0, 0x1.212be3578a819p+0, 0x1.21902cd3d09b9p+0,
  0x1.21f49917ddc96p+0, 0x1.2259282fc1f27p+0, 0x1.22bdda27912d1p+0, 0x1.2322af0b63bffp+0,
  0x1.2387a6e756238p+0, 0x1.23ecc1c78903ap+0, 0x1.2451ffb82140ap+0, 0x1.24b760c547f15p+0,
  0x1.251ce4fb2a63fp+0, 0x1.25828c65fa1ffp+0, 0x1.25e85711ece75p+0, 0x1.264e450b3cb82p+0,
  0x1.26b4565e27cddp+0, 0x1.271a8b16f0a30p+0, 0x1.2780e341ddf29p+0, 0x1.27e75eeb3ab98p+0,
  0x1.284dfe1f56381p+0, 0x1.28b4c0ea83f36p+0, 0x1.291ba7591bb70p+0, 0x1.2982b17779965p+0,
  0x1.29e9df51fdee1p+0, 0x1.2a5130f50d65cp+0, 0x1.2ab8a66d10f13p+0, 0x1.2b203fc675d1fp+0,
  0x1.2b87fd0dad990p+0, 0x1.2befde4f2e280p+0, 0x1.2c57e39771b2fp+0, 0x1.2cc00cf2f6c18p+0,
  0x1.2d285a6e4030bp+0, 0x1.2d90cc15d5346p+0, 0x1.2df961f641589p+0, 0x1.2e621c1c14833p+0,
  0x1.2ecafa93e2f56p+0, 0x1.2f33fd6a454d2p+0, 0x1.2f9d24abd886bp+0, 0x1.300670653dfe4p+0,
  0x1.306fe0a31b715p+0, 0x1.30d975721b004p+0, 0x1.31432edeeb2fdp+0, 0x1.31ad0cf63eeacp+0,
  0x1.32170fc4cd831p+0, 0x1.3281375752b40p+0, 0x1.32eb83ba8ea32p+0, 0x1.3355f4fb45e20p+0,
  0x1.33c08b26416ffp+0, 0x1.342b46484ebb4p+0, 0x1.3496266e3fa2dp+0, 0x1.35012ba4ea77dp+0,
  0x1.356c55f929ff1p+0, 0x1.35d7a577dd72bp+0, 0x1.36431a2de883bp+0, 0x1.36aeb428335b4p+0,
  0x1.371a7373aa9cbp+0, 0x1.3786581d3f669p+0, 0x1.37f26231e754ap+0, 0x1.385e91be9c811p+0,
  0x1.38cae6d05d866p+0, 0x1.393761742d808p+0, 0x1.39a401b7140efp+0, 0x1.3a10c7a61d55bp+0,
  0x1.3a7db34e59ff7p+0, 0x1.3aeac4bcdf3eap+0, 0x1.3b57fbfec6cf4p+0, 0x1.3bc559212ef89p+0,
  0x1.3c32dc313a8e5p+0, 0x1.3ca0853c10f28p+0, 0x1.3d0e544ede173p+0, 0x1.3d7c4976d27fap+0,
  0x1.3dea64c123422p+0, 0x1.3e58a63b0a09bp+0, 0x1.3ec70df1c5175p+0, 0x1.3f359bf29743fp+0,
  0x1.3fa4504ac801cp+0, 0x1.40132b07a35dfp+0, 0x1.40822c367a024p+0, 0x1.40f153e4a136ap+0,
  0x1.4160a21f72e2ap+0, 0x1.41d016f44d8f5p+0, 0x1.423fb2709468ap+0, 0x1.42af74a1af3f1p+0,
  0x1.431f5d950a897p+0, 0x1.438f6d5817663p+0, 0x1.43ffa3f84b9d4p+0, 0x1.4470018321a1ap+0,
  0x1.44e086061892dp+0, 0x1.4551318eb43ecp+0, 0x1.45c2042a7d232p+0, 0x1.4632fde7006f4p+0,
  0x1.46a41ed1d0057p+0, 0x1.471566f8827d0p+0, 0x1.4786d668b3237p+0, 0x1.47f86d3001fe5p+0,
  0x1.486a2b5c13cd0p+0, 0x1.48dc10fa920a1p+0, 0x1.494e1e192aed2p+0, 0x1.49c052c5916c4p+0,
  0x1.4a32af0d7d3dep+0, 0x1.4aa532feaada6p+0, 0x1.4b17dea6db7d7p+0, 0x1.4b8ab213d5283p+0,
  0x1.4bfdad5362a27p+0, 0x1.4c70d073537cap+0, 0x1.4ce41b817c114p+0, 0x1.4d578e8bb586bp+0,
  0x1.4dcb299fddd0dp+0, 0x1.4e3eeccbd7b2ap+0, 0x1.4eb2d81d8abffp+0, 0x1.4f26eba2e35f0p+0,
  0x1.4f9b2769d2ca7p+0, 0x1.500f8b804f127p+0, 0x1.508417f4531eep+0, 0x1.50f8ccd3deb0dp+0,
  0x1.516daa2cf6642p+0, 0x1.51e2b00da3b14p+0, 0x1.5257de83f4eefp+0, 0x1.52cd359dfd53dp+0,
  0x1.5342b569d4f82p+0, 0x1.53b85df598d78p+0, 0x1.542e2f4f6ad27p+0, 0x1.54a4298571b06p+0,
  0x1.551a4ca5d920fp+0, 0x1.559098bed1bdfp+0, 0x1.56070dde910d2p+0, 0x1.567dac1351819p+0,
  0x1.56f4736b527dap+0, 0x1.576b63f4d854cp+0, 0x1.57e27dbe2c4cfp+0, 0x1.5859c0d59ca07p+0,
  0x1.58d12d497c7fdp+0, 0x1.5948c32824135p+0, 0x1.59c0827ff07ccp+0, 0x1.5a386b5f43d92p+0,
  0x1.5ab07dd485429p+0, 0x1.5b28b9ee20d1ep+0, 0x1.5ba11fba87a03p+0, 0x1.5c19af482fc8fp+0,
  0x1.5c9268a5946b7p+0, 0x1.5d0b4be135accp+0, 0x1.5d84590998b93p+0, 0x1.5dfd902d47c65p+0,
  0x1.5e76f15ad2148p+0, 0x1.5ef07ca0cbf0fp+0, 0x1.5f6a320dceb71p+0, 0x1.5fe411b078d26p+0,
  0x1.605e1b976dc09p+0, 0x1.60d84fd15612ap+0, 0x1.6152ae6cdf6f4p+0, 0x1.61cd3778bc944p+0,
  0x1.6247eb03a5585p+0, 0x1.62c2c91c56acdp+0, 0x1.633dd1d1929fdp+0, 0x1.63b90532205d8p+0,
  0x1.6434634ccc320p+0, 0x1.64afec30678b7p+0, 0x1.652b9febc8fb7p+0, 0x1.65a77e8dcc390p+0,
  0x1.6623882552225p+0, 0x1.669fbcc140be7p+0, 0x1.671c1c70833f6p+0, 0x1.6798a7420a036p+0,
  0x1.68155d44ca973p+0, 0x1.68923e87bfb7ap+0, 0x1.690f4b19e9538p+0, 0x1.698c830a4c8d4p+0,
  0x1.6a09e667f3bcdp+0, 0x1.6a877541ee718p+0, 0x1.6b052fa75173ep+0, 0x1.6b8315a736c75p+0,
  0x1.6c012750bdabfp+0, 0x1.6c7f64b30aa09p+0, 0x1.6cfdcddd47645p+0, 0x1.6d7c62dea2f8ap+0,
  0x1.6dfb23c651a2fp+0, 0x1.6e7a10a38cee8p+0, 0x1.6ef9298593ae5p+0, 0x1.6f786e7ba9fefp+0,
  0x1.6ff7df9519484p+0, 0x1.70777ce1303f6p+0, 0x1.70f7466f42e87p+0, 0x1.71773c4eaa988p+0,
  0x1.71f75e8ec5f74p+0, 0x1.7277ad3ef9011p+0, 0x1.72f8286ead08ap+0, 0x1.7378d02d50b8fp+0,
  0x1.73f9a48a58174p+0, 0x1.747aa5953c849p+0, 0x1.74fbd35d7cbfdp+0, 0x1.757d2df29ce7cp+0,
  0x1.75feb564267c9p+0, 0x1.768069c1a861dp+0, 0x1.77024b1ab6e09p+0, 0x1.7784597eeba8fp+0,
  0x1.780694fde5d3fp+0, 0x1.7888fda749e5dp+0, 0x1.790b938ac1cf6p+0, 0x1.798e56b7fcf03p+0,
  0x1.7a11473eb0187p+0, 0x1.7a94652e958aap+0, 0x1.7b17b0976cfdbp+0, 0x1.7b9b2988fb9ecp+0,
  0x1.7c1ed0130c132p+0, 0x1.7ca2a4456e7a3p+0, 0x1.7d26a62ff86f0p+0, 0x1.7daad5e2850acp+0,
  0x1.7e2f336cf4e62p+0, 0x1.7eb3bedf2e1b9p+0, 0x1.7f3878491c491p+0, 0x1.7fbd5fbab091fp+0,
  0x1.80427543e1a12p+0, 0x1.80c7b8f4abaa9p+0, 0x1.814d2add106d9p+0, 0x1.81d2cb0d1736ap+0,
  0x1.82589994cce13p+0, 0x1.82de968443d9ap+0, 0x1.8364c1eb941f7p+0, 0x1.83eb1bdadb46dp+0,
  0x1.8471a4623c7adp+0, 0x1.84f85b91e07f1p+0, 0x1.857f4179f5b21p+0, 0x1.8606562ab00ecp+0,
  0x1.868d99b4492edp+0, 0x1.87150c27004c2p+0, 0x1.879cad931a436p+0, 0x1.88247e08e1957p+0,
  0x1.88ac7d98a6699p+0, 0x1.8934ac52be8f7p+0, 0x1.89bd0a478580fp+0, 0x1.8a4597875c644p+0,
  0x1.8ace5422aa0dbp+0, 0x1.8b574029db01ep+0, 0x1.8be05bad61778p+0, 0x1.8c69a6bdb5598p+0,
  0x1.8cf3216b5448cp+0, 0x1.8d7ccbc6c19e6p+0, 0x1.8e06a5e0866d9p+0, 0x1.8e90afc931857p+0,
  0x1.8f1ae99157736p+0, 0x1.8fa553499284bp+0, 0x1.902fed0282c8ap+0, 0x1.90bab6ccce12cp+0,
  0x1.9145b0b91ffc6p+0, 0x1.91d0dad829e70p+0, 0x1.925c353aa2fe2p+0, 0x1.92e7bff148396p+0,
  0x1.93737b0cdc5e5p+0, 0x1.93ff669e2802bp+0, 0x1.948b82b5f98e5p+0, 0x1.9517cf65253d1p+0,
  0x1.95a44cbc8520fp+0, 0x1.9630faccf9243p+0, 0x1.96bdd9a7670b3p+0, 0x1.974ae95cba768p+0,
  0x1.97d829fde4e50p+0, 0x1.98659b9bddb5bp+0, 0x1.98f33e47a22a2p+0, 0x1.9981121235681p+0,
  0x1.9a0f170ca07bap+0, 0x1.9a9d4d47f2598p+0, 0x1.9b2bb4d53fe0dp+0, 0x1.9bba4dc5a3dd3p+0,
  0x1.9c49182a3f090p+0, 0x1.9cd81414380f2p+0, 0x1.9d674194bb8d5p+0, 0x1.9df6a0bcfc15ep+0,
  0x1.9e86319e32323p+0, 0x1.9f15f4499c647p+0, 0x1.9fa5e8d07f29ep+0, 0x1.a0360f4424fcbp+0,
  0x1.a0c667b5de565p+0, 0x1.a156f23701b15p+0, 0x1.a1e7aed8eb8bbp+0, 0x1.a2789dacfe68cp+0,
  0x1.a309bec4a2d33p+0, 0x1.a39b1231475f7p+0, 0x1.a42c980460ad8p+0, 0x1.a4be504f696b1p+0,
  0x1.a5503b23e255dp+0, 0x1.a5e25893523d4p+0, 0x1.a674a8af46052p+0, 0x1.a7072b8950a73p+0,
  0x1.a799e1330b358p+0, 0x1.a82cc9be14dcap+0, 0x1.a8bfe53c12e59p+0, 0x1.a95333beb0b7ep+0,
  0x1.a9e6b5579fdbfp+0, 0x1.aa7a6a1897fd2p+0, 0x1.ab0e521356ebap+0, 0x1.aba26d59a09eep+0,
  0x1.ac36bbfd3f37ap+0, 0x1.accb3e100301ep+0, 0x1.ad5ff3a3c2774p+0, 0x1.adf4dcca5a413p+0,
  0x1.ae89f995ad3adp+0, 0x1.af1f4a17a4735p+0, 0x1.afb4ce622f2ffp+0, 0x1.b04a868742ee4p+0,
  0x1.b0e07298db666p+0, 0x1.b17692a8fa8cdp+0, 0x1.b20ce6c9a8952p+0, 0x1.b2a36f0cf3f3ap+0,
  0x1.b33a2b84f15fbp+0, 0x1.b3d11c43bbd62p+0, 0x1.b468415b749b1p+0, 0x1.b4ff9ade433c6p+0,
  0x1.b59728de5593ap+0, 0x1.b62eeb6ddfc87p+0, 0x1.b6c6e29f1c52ap+0, 0x1.b75f0e844bfc6p+0,
  0x1.b7f76f2fb5e47p+0, 0x1.b89004b3a7804p+0, 0x1.b928cf22749e4p+0, 0x1.b9c1ce8e77680p+0,
  0x1.ba5b030a1064ap+0, 0x1.baf46ca7a67a7p+0, 0x1.bb8e0b79a6f1fp+0, 0x1.bc27df9285775p+0,
  0x1.bcc1e904bc1d2p+0, 0x1.bd5c27e2cb5e5p+0, 0x1.bdf69c3f3a207p+0, 0x1.be91462c95b60p+0,
  0x1.bf2c25bd71e09p+0, 0x1.bfc73b0468d30p+0, 0x1.c06286141b33dp+0, 0x1.c0fe06ff301f4p+0,
  0x1.c199bdd85529cp+0, 0x1.c235aab23e61ep+0, 0x1.c2d1cd9fa652cp+0, 0x1.c36e26b34e065p+0,
  0x1.c40ab5fffd07ap+0, 0x1.c4a77b9881650p+0, 0x1.c544778fafb22p+0, 0x1.c5e1a9f8630adp+0,
  0x1.c67f12e57d14bp+0, 0x1.c71cb269e601fp+0, 0x1.c7ba88988c933p+0, 0x1.c8589584661a1p+0,
  0x1.c8f6d9406e7b5p+0, 0x1.c99553dfa8313p+0, 0x1.ca3405751c4dbp+0, 0x1.cad2ee13da7cbp+0,
  0x1.cb720dcef9069p+0, 0x1.cc1164b994d23p+0, 0x1.ccb0f2e6d1675p+0, 0x1.cd50b869d8f0fp+0,
  0x1.cdf0b555dc3fap+0, 0x1.ce90e9be12cb9p+0, 0x1.cf3155b5bab74p+0, 0x1.cfd1f95018d17p+0,
  0x1.d072d4a07897cp+0, 0x1.d113e7ba2c38cp+0, 0x1.d1b532b08c968p+0, 0x1.d256b596f948cp+0,
  0x1.d2f87080d89f2p+0, 0x1.d39a638197a3cp+0, 0x1.d43c8eacaa1d6p+0, 0x1.d4def2158a91fp+0,
  0x1.d5818dcfba487p+0, 0x1.d62461eec14bep+0, 0x1.d6c76e862e6d3p+0, 0x1.d76ab3a99745bp+0,
  0x1.d80e316c98398p+0, 0x1.d8b1e7e2d479dp+0, 0x1.d955d71ff6075p+0, 0x1.d9f9ff37adb4ap+0,
  0x1.da9e603db3285p+0, 0x1.db42fa45c4dfdp+0, 0x1.dbe7cd63a8315p+0, 0x1.dc8cd9ab294e4p+0,
  0x1.dd321f301b460p+0, 0x1.ddd79e065807dp+0, 0x1.de7d5641c0658p+0, 0x1.df2347f63c159p+0,
  0x1.dfc97337b9b5fp+0, 0x1.e06fd81a2ece1p+0, 0x1.e11676b197d17p+0, 0x1.e1bd4f11f8220p+0,
  0x1.e264614f5a129p+0, 0x1.e30bad7dcee90p+0, 0x1.e3b333b16ee12p+0, 0x1.e45af3fe592e8p+0,
  0x1.e502ee78b3ff6p+0, 0x1.e5ab2334ac7eep+0, 0x1.e653924676d76p+0, 0x1.e6fc3bc24e350p+0,
  0x1.e7a51fbc74c83p+0, 0x1.e84e3e4933c7ep+0, 0x1.e8f7977cdb740p+0, 0x1.e9a12b6bc3181p+0,
  0x1.ea4afa2a490dap+0, 0x1.eaf503ccd2be5p+0, 0x1.eb9f4867cca6ep+0, 0x1.ec49c80faa594p+0,
  0x1.ecf482d8e67f1p+0, 0x1.ed9f78d802dc2p+0, 0x1.ee4aaa2188510p+0, 0x1.eef616ca06dd6p+0,
  0x1.efa1bee615a27p+0, 0x1.f04da28a52e59p+0, 0x1.f0f9c1cb6412ap+0, 0x1.f1a61cbdf5be7p+0,
  0x1.f252b376bba97p+0, 0x1.f2ff860a70c22p+0, 0x1.f3ac948dd7274p+0, 0x1.f459df15b82acp+0,
  0x1.f50765b6e4540p+0, 0x1.f5b5288633625p+0, 0x1.f6632798844f8p+0, 0x1.f7116302bd526p+0,
  0x1.f7bfdad9cbe14p+0, 0x1.f86e8f32a4b45p+0, 0x1.f91d802243c89p+0, 0x1.f9ccadbdac61dp+0,
  0x1.fa7c1819e90d8p+0, 0x1.fb2bbf4c0ba54p+0, 0x1.fbdba3692d514p+0, 0x1.fc8bc4866e8adp+0,
  0x1.fd3c22b8f71f1p+0, 0x1.fdecbe15f6314p+0, 0x1.fe9d96b2a23d9p+0, 0x1.ff4eaca4391b6p+0
};

__device__ __forceinline__ double SoftmaxLeanExpNeg(const double d) {
  const float kf = rintf(SoftmaxLeanApproxFloatRN(d) * 738.65985107421875f);  // 512 / ln2
  const int k = static_cast<int>(kf);
  const double kd = SoftmaxLeanFloatToDouble(kf);
  const double r = fma(-kd, 0x1.62e42fefa39efp-10, d);  // ln2 / 512
  const float rf = SoftmaxLeanApproxFloatRN(r);
  float u = fmaf(4.16666666666666667e-02f, rf, 1.66666666666666667e-01f);  // 1/3! + r/4!
  u = fmaf(u, rf, 0.5f);  // 1/2! + r/3! + r^2/4!
  const float tail_f = (rf * rf) * u;
  const double dr = r + SoftmaxLeanFloatToDouble(tail_f);
  const double tbl = kSoftmaxLeanExp2Table512[k & 511];
  const double scaled = fma(dr, tbl, tbl);
  return __longlong_as_double(__double_as_longlong(scaled) + (static_cast<long long>(k >> 9) << 52));  // NOLINT(runtime/int)
}

template <bool USE_WEIGHT, int MAX_CLASS>
__device__ __noinline__ void MulticlassSoftmaxExactFallback(
  const double* cuda_scores, const int label, const double weight,
  const double factor, const int num_class, const data_size_t num_data,
  const data_size_t data_index, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const size_t stride = static_cast<size_t>(num_data);
  double p[MAX_CLASS];
#pragma unroll
  for (int k = 0; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      p[k] = cuda_scores[stride * k + data_index];
    }
  }
  double wmax = p[0];
#pragma unroll
  for (int k = 1; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      wmax = max(p[k], wmax);
    }
  }
  double wsum = 0.0f;
#pragma unroll
  for (int k = 0; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      p[k] = exp(p[k] - wmax);
      wsum += p[k];
    }
  }
#pragma unroll
  for (int k = 0; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      const double pk = p[k] / static_cast<double>(wsum);
      const size_t idx = stride * k + data_index;
      if (!USE_WEIGHT) {
        cuda_out_gradients[idx] = static_cast<score_t>(label == k ? pk - 1.0f : pk);
        cuda_out_hessians[idx] = static_cast<score_t>(factor * pk * (1.0f - pk));
      } else {
        cuda_out_gradients[idx] = static_cast<score_t>(label == k ? (pk - 1.0f) * weight : pk * weight);
        cuda_out_hessians[idx] = static_cast<score_t>((factor * pk * (1.0f - pk)) * weight);
      }
    }
  }
}

template <bool USE_WEIGHT, int MAX_CLASS>
__global__ void __launch_bounds__(kSoftmaxLeanBlockSize) GetGradientsKernel_MulticlassSoftmaxRegistersLean(
  const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights,
  const double factor, const int num_class, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  PdlLaunchDependents();  // cuda_plan key pdl_level_chain: the next kernel (index fill, bagging count) may get resident
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index >= num_data) {
    return;
  }
  const size_t stride = static_cast<size_t>(num_data);
  const int label = static_cast<int>(cuda_labels[data_index]);
  const double weight = USE_WEIGHT ? static_cast<double>(cuda_weights[data_index]) : 1.0;

  bool ok = (num_class >= 2) && SoftmaxLeanWideNormal(factor);
  if (USE_WEIGHT) {
    ok = ok && SoftmaxLeanWideNormal(weight) && (weight > 0.0);
  }

  double p[MAX_CLASS];
#pragma unroll
  for (int k = 0; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      p[k] = cuda_scores[stride * k + data_index];
    }
  }
  double wmax = p[0];
  int kmax = 0;
#pragma unroll
  for (int k = 1; k < MAX_CLASS; ++k) {
    if (k < num_class) {
      if (p[k] > wmax) {
        wmax = p[k];
        kmax = k;
      }
    }
  }
  const unsigned long long wmax_exp =  // NOLINT(runtime/int)
    static_cast<unsigned long long>(__double_as_longlong(wmax)) & 0x7ff0000000000000ull;  // NOLINT(runtime/int)
  if (wmax_exp == 0x7ff0000000000000ull) {
    ok = false;
  }

  double s_other = 0.0;
  double e_label = 0.0;
#pragma unroll
  for (int k = 0; k < MAX_CLASS; ++k) {
    if (k < num_class && k != kmax) {
      const double d = p[k] - wmax;
      double ek = 0.0;
      if (!(d <= 0.0)) {
        ok = false;
      } else if (d >= (USE_WEIGHT ? -650.0 : -700.0)) {
        ek = SoftmaxLeanExpNeg(d);
        s_other += ek;
      } else if (USE_WEIGHT) {
        ok = false;
      }
      p[k] = ek;
      if (k == label) {
        e_label = ek;
      }
    }
  }

  if (ok) {
    const double wsum = 1.0 + s_other;
    const double inv0 = SoftmaxLeanFloatToDouble(__frcp_rn(SoftmaxLeanApproxFloatRN(wsum)));
    const double inv_wsum = fma(fma(-wsum, inv0, 1.0), inv0, inv0);
    const double f_inv = factor * inv_wsum;

    // Precompute k == kmax ONCE per row (outside the unrolled class loop)
    float g_kmax_out = 0.0f;
    float h_kmax_out = 0.0f;
    bool kmax_ok = true;
    if (s_other < 0x1.0p-55) {
      // Proven wsum_orig == 1.0 exact -> p_max_orig == 1.0 exact -> (1 - p_max_orig) == +0.0 exact
      g_kmax_out = (label == kmax) ? 0.0f : (USE_WEIGHT ? static_cast<float>(weight) : 1.0f);
      h_kmax_out = 0.0f;
    } else {
      const double qmax = s_other * inv_wsum;
      const float qmax_f = SoftmaxLeanApproxFloatRN(qmax);
      if (!(qmax_f > 0.0f)) {
        kmax_ok = false;
      } else {
        const float qmax_cancel_margin = 1.25f * static_cast<float>(num_class) / qmax_f;
        const double g_val = (label == kmax) ? -qmax : inv_wsum;
        const double h_val = f_inv * qmax;
        const double gw = USE_WEIGHT ? (g_val * weight) : g_val;
        const double hw = USE_WEIGHT ? (h_val * weight) : h_val;
        const float g_margin = (label == kmax) ? (4096.0f + qmax_cancel_margin) : 4096.0f;
        const float h_margin = 6144.0f + qmax_cancel_margin;
        if (!SoftmaxLeanSettledFloat(gw, g_margin, &g_kmax_out) ||
            !SoftmaxLeanSettledNonNegFloat(hw, h_margin, &h_kmax_out)) {
          kmax_ok = false;
        }
      }
    }

    // If kmax failed the cancellation margin (e.g. S_other in [2^-55, 1e-5]), try to certify exact class-order
    // wsum_ord == wsum_orig via Dekker's FastTwoSum so we only pay ONE FP64 division (zero exp() calls!).
    if (!kmax_ok && s_other < 0.25) {
      bool wsum_exact = true;
      double acc = 0.0;
      bool seen_kmax = false;
#pragma unroll
      for (int k = 0; k < MAX_CLASS; ++k) {
        if (k < num_class) {
          if (k == kmax) {
            const double s = __dadd_rn(1.0, acc);
            const double rem = __dsub_rn(acc, __dsub_rn(s, 1.0));
            const double err = acc * 0x1.0p-41 + 0x1.0p-90;
            if (!(fabs(rem) + err < 0x1.0p-53)) {
              wsum_exact = false;
            }
            acc = s;
            seen_kmax = true;
          } else if (!seen_kmax) {
            acc = __dadd_rn(acc, p[k]);
          } else {
            const double s = __dadd_rn(acc, p[k]);
            const double rem = __dsub_rn(p[k], __dsub_rn(s, acc));
            const double err = p[k] * 0x1.0p-42 + 0x1.0p-90;
            if (!(fabs(rem) + err < 0x1.0p-53)) {
              wsum_exact = false;
            }
            acc = s;
          }
        }
      }
      if (wsum_exact) {
        const double pk = 1.0 / acc;
        if (!USE_WEIGHT) {
          g_kmax_out = static_cast<float>(label == kmax ? pk - 1.0f : pk);
          h_kmax_out = static_cast<float>(factor * pk * (1.0f - pk));
        } else {
          g_kmax_out = static_cast<float>(label == kmax ? (pk - 1.0f) * weight : pk * weight);
          h_kmax_out = static_cast<float>((factor * pk * (1.0f - pk)) * weight);
        }
        kmax_ok = true;
      }
    }
    if (!kmax_ok) {
      ok = false;
    }

    // Precompute k == label (when label != kmax) ONCE per row (1 DFMA instead of num_class DADDs)
    float g_label_out = 0.0f;
    if (ok && label != kmax && label >= 0 && label < num_class) {
      const double g_val = fma(e_label, inv_wsum, -1.0);
      const double gw = USE_WEIGHT ? (g_val * weight) : g_val;
      if (!SoftmaxLeanSettledFloat(gw, 4096.0f, &g_label_out)) {
        ok = false;
      }
    }

    if (ok) {
#pragma unroll
      for (int k = 0; k < MAX_CLASS; ++k) {
        if (k < num_class) {
          float g_out = g_kmax_out;
          float h_out = h_kmax_out;
          if (k != kmax) {
            const double pk = p[k] * inv_wsum;
            const double fp = p[k] * f_inv;
            const double h_val = fma(-pk, fp, fp);
            const double hw = USE_WEIGHT ? (h_val * weight) : h_val;
            if (!SoftmaxLeanSettledNonNegFloat(hw, 4096.0f, &h_out)) {
              ok = false;
            }
            if (k == label) {
              g_out = g_label_out;
            } else {
              const double gw = USE_WEIGHT ? (pk * weight) : pk;
              if (!SoftmaxLeanSettledNonNegFloat(gw, 4096.0f, &g_out)) {
                ok = false;
              }
            }
          }
          const size_t idx = stride * k + data_index;
          cuda_out_gradients[idx] = static_cast<score_t>(g_out);
          cuda_out_hessians[idx] = static_cast<score_t>(h_out);
        }
      }
    }
  }

  if (!ok) {
    MulticlassSoftmaxExactFallback<USE_WEIGHT, MAX_CLASS>(
      cuda_scores, label, weight, factor, num_class, num_data, data_index,
      cuda_out_gradients, cuda_out_hessians);
  }
}

template <bool USE_WEIGHT, int MAX_CLASS>
static bool LaunchMulticlassSoftmaxRegisters(const int num_blocks, const double* scores, const label_t* labels,
                                             const label_t* weights, const double factor, const int num_class,
                                             const data_size_t num_data, score_t* gradients, score_t* hessians,
                                             const bool lean) {
  if (num_class > MAX_CLASS) {
    return false;
  }
  if constexpr (kSoftmaxLeanScoreIsFloat) {
    if (lean) {
      const int lean_blocks = (num_data + kSoftmaxLeanBlockSize - 1) / kSoftmaxLeanBlockSize;
      GetGradientsKernel_MulticlassSoftmaxRegistersLean<USE_WEIGHT, MAX_CLASS><<<lean_blocks, kSoftmaxLeanBlockSize>>>(
        scores, labels, weights, factor, num_class, num_data, gradients, hessians);
      return true;
    }
  }
  GetGradientsKernel_MulticlassSoftmaxRegisters<USE_WEIGHT, MAX_CLASS><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
    scores, labels, weights, factor, num_class, num_data, gradients, hessians);
  return true;
}

void CUDAMulticlassSoftmax::LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_MULTICLASS - 1) / GET_GRADIENTS_BLOCK_SIZE_MULTICLASS;
  const FalcataPlan& plan = FalcataPlan::Get();
  if (plan.softmax_registers) {
    // class capacities 8 / 16 / 32 (at 32 the 64-register cap of __launch_bounds__(1024) spills part of the array to
    // thread-interleaved local memory, still coalesced); wider class counts keep the scratch kernel
    const bool lean = plan.softmax_fp64_lean;
    const bool launched = cuda_weights_ == nullptr ?
      (LaunchMulticlassSoftmaxRegisters<false, 8>(num_blocks, scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                  num_data_, gradients, hessians, lean) ||
       LaunchMulticlassSoftmaxRegisters<false, 16>(num_blocks, scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                   num_data_, gradients, hessians, lean) ||
       LaunchMulticlassSoftmaxRegisters<false, 32>(num_blocks, scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                   num_data_, gradients, hessians, lean)) :
      (LaunchMulticlassSoftmaxRegisters<true, 8>(num_blocks, scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                 num_data_, gradients, hessians, lean) ||
       LaunchMulticlassSoftmaxRegisters<true, 16>(num_blocks, scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                  num_data_, gradients, hessians, lean) ||
       LaunchMulticlassSoftmaxRegisters<true, 32>(num_blocks, scores, cuda_labels_, cuda_weights_, factor_, num_class_,
                                                  num_data_, gradients, hessians, lean));
    if (launched) {
      return;
    }
  }
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_MulticlassSoftmax<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
      scores, cuda_labels_, cuda_weights_, factor_, num_class_, num_data_,
      cuda_softmax_buffer_.RawData(), gradients, hessians);
  } else {
    GetGradientsKernel_MulticlassSoftmax<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
      scores, cuda_labels_, cuda_weights_, factor_, num_class_, num_data_,
      cuda_softmax_buffer_.RawData(), gradients, hessians);
  }
}

__global__ void ConvertOutputCUDAKernel_MulticlassSoftmax(
  const int num_class, const data_size_t num_data, const double* input, double* cuda_softmax_buffer, double* output) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    // class-major scratch at the call's row stride: num_data <= the training rows the buffer holds per class
    const size_t stride = static_cast<size_t>(num_data);
    double* cuda_softmax_buffer_ptr = cuda_softmax_buffer + data_index;
    for (int class_index = 0; class_index < num_class; ++class_index) {
      cuda_softmax_buffer_ptr[class_index * stride] = input[stride * class_index + data_index];
    }
    SoftmaxCUDA(cuda_softmax_buffer_ptr, num_class, stride);
    for (int class_index = 0; class_index < num_class; ++class_index) {
      output[stride * class_index + data_index] = cuda_softmax_buffer_ptr[class_index * stride];
    }
  }
}

const double* CUDAMulticlassSoftmax::LaunchConvertOutputCUDAKernel(
  const data_size_t num_data, const double* input, double* output) const {
  // the grid covers the rows the caller passes; the softmax scratch is sized for the training rows (Init)
  if (num_data > num_data_) {
    Log::Fatal("CUDAMulticlassSoftmax::ConvertOutputCUDA: %d rows exceed the %d rows its scratch buffer holds",
               num_data, num_data_);
  }
  if (num_data <= 0) {
    return output;
  }
  const int num_blocks = (num_data + GET_GRADIENTS_BLOCK_SIZE_MULTICLASS - 1) / GET_GRADIENTS_BLOCK_SIZE_MULTICLASS;
  ConvertOutputCUDAKernel_MulticlassSoftmax<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_MULTICLASS>>>(
    num_class_, num_data, input, cuda_softmax_buffer_.RawData(), output);
  return output;
}

}  // namespace Falcata

#endif  // USE_CUDA
