/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 *
 * Single source of truth for the per-leaf split-gain / leaf-output numeric
 * math shared by every backend (CPU serial learner, OpenCL, CUDA).
 *
 * Defined ONCE as __host__ __device__ free functions so the CPU and device
 * paths cannot disagree -- per-backend copies of this math inevitably drift
 * (sign handling, epsilon placement), and any drift is a model-correctness
 * bug, not a style problem.
 */
#ifndef FALCATA_TREE_SPLIT_MATH_H_
#define FALCATA_TREE_SPLIT_MATH_H_

#include <Falcata/meta.h>

#include <cmath>

#if defined(__CUDACC__)
#define FLC_HOSTDEV __host__ __device__
#else
#define FLC_HOSTDEV
#endif

namespace Falcata {

namespace SplitGainMath {

// Soft-threshold used for L1 regularization: sign(s) * max(0, |s| - l1).
// T = double everywhere except the CUDA fp32 gain mode (FALCATA_FP32_GAIN).
template <typename T = double>
FLC_HOSTDEV inline T ThresholdL1(T s, T l1) {
  const T reg_s = fmax(static_cast<T>(0), fabs(s) - l1);
  return s >= static_cast<T>(0) ? reg_s : -reg_s;
}

// Newton leaf output -g/(h+l2) with optional L1 shrink, max_delta_step cap, and
// path smoothing -- applied in that order (matching the CPU formula). Monotone
// clamping is applied by the caller (it depends on per-leaf constraints).
template <bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING, typename T = double>
FLC_HOSTDEV inline T CalculateLeafOutput(T sum_gradients, T sum_hessians,
                                          T l1, T l2, T max_delta_step,
                                          T path_smooth, data_size_t num_data,
                                          T parent_output) {
  T ret = USE_L1 ? (-ThresholdL1(sum_gradients, l1) / (sum_hessians + l2))
                 : (-sum_gradients / (sum_hessians + l2));
  if (USE_MAX_OUTPUT) {
    if (max_delta_step > static_cast<T>(0) && fabs(ret) > max_delta_step) {
      ret = ret >= static_cast<T>(0) ? max_delta_step : -max_delta_step;
    }
  }
  if (USE_SMOOTHING) {
    ret = ret * (num_data / path_smooth) / (num_data / path_smooth + 1)
        + parent_output / (num_data / path_smooth + 1);
  }
  return ret;
}

// Gain contributed by a leaf given a already-computed output value.
template <bool USE_L1, typename T = double>
FLC_HOSTDEV inline T LeafGainGivenOutput(T sum_gradients, T sum_hessians,
                                          T l1, T l2, T output) {
  const T g = USE_L1 ? ThresholdL1(sum_gradients, l1) : sum_gradients;
  return -(2 * g * output + (sum_hessians + l2) * output * output);
}

// Gain of a leaf. The closed-form g^2/(h+l2) is only valid when the output is
// the unconstrained Newton step, so it is used only when neither the
// max_delta_step cap nor smoothing can move the output; otherwise the gain is
// measured at the output actually used. Mirrors CPU's
// FeatureHistogram::GetLeafGain.
//
// Note the two branches are algebraically equal but NOT bitwise equal, so the
// USE_MAX_OUTPUT switch must track CPU's exactly to keep CPU/CUDA bit-identical
// when max_delta_step is unset.
template <bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING, typename T = double>
FLC_HOSTDEV inline T LeafGain(T sum_gradients, T sum_hessians, T l1,
                               T l2, T max_delta_step, T path_smooth,
                               data_size_t num_data, T parent_output) {
  if (!USE_MAX_OUTPUT && !USE_SMOOTHING) {
    const T g = USE_L1 ? ThresholdL1(sum_gradients, l1) : sum_gradients;
    return (g * g) / (sum_hessians + l2);
  }
  const T output = CalculateLeafOutput<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING, T>(
      sum_gradients, sum_hessians, l1, l2, max_delta_step, path_smooth, num_data, parent_output);
  return LeafGainGivenOutput<USE_L1, T>(sum_gradients, sum_hessians, l1, l2, output);
}

}  // namespace SplitGainMath

}  // namespace Falcata

#endif  // FALCATA_TREE_SPLIT_MATH_H_
