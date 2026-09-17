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

// split_midpoint: the threshold at the middle of a run of leaf-empty bins.
// Every threshold across a run of bins that hold none of the leaf's rows
// partitions those rows identically and scores the same gain, so the scan
// keeps whichever it met first -- an edge of the run -- and unseen values
// inside the run all route to one side. Shared by the CPU and CUDA finders so
// both store the same threshold. is_empty_hist(idx) reads stored histogram
// entry idx (bin = idx + offset); the walls -- the unstored most-frequent bin,
// the NaN bin, a default bin the scan routes by direction -- never count as
// empty because their rows are real. Callers pass plain functor structs, not
// lambdas: nvcc rejects a host lambda inside a __host__ __device__ template.
// Every bin the returned threshold crosses is empty by the functor's rule, so
// the split's sums and partition stay consistent without recomputation.
// Quantized (integer) histograms never qualify: rows whose gradient and
// hessian both round to zero -- most confidently-classified rows on an
// imbalanced objective -- leave no trace in the bin, and crossing them
// re-routes real rows (fraud: 91% of thresholds moved, training diverged).
// (grad, hess) pairs: empty when both sums are within `tol` of zero. The
// tolerance comes from MidpointEmptyTolerance: exactly zero where the histogram
// is exact, a relative epsilon where it carries subtraction residue, and
// negative (never empty) where residue and real rows cannot be told apart.
// A row-count test (RoundInt(hess * cnt_factor) == 0) is NOT safe here: on an
// imbalanced binary objective a bin of a few confidently-classified rows
// rounds to zero rows, the threshold then crosses real rows, and the split's
// recorded sums no longer match its partition (a model collapsed that way).
template <typename T>
struct PairHistEmpty {
  const T* data;
  double tol;
  FLC_HOSTDEV bool operator()(int idx) const {
    return tol >= 0.0 && fabs(static_cast<double>(data[idx << 1])) <= tol &&
      fabs(static_cast<double>(data[(idx << 1) + 1])) <= tol;
  }
};

// Emptiness tolerance for PairHistEmpty. A directly built histogram has exact
// zeros in empty bins. One built by parent-minus-sibling subtraction carries
// residue of a few ulp of the parent's bin: ~1e-16 relative in double, where
// 1e-12 of the leaf's totals separates it from any row that matters, but
// ~1e-7 relative in fp32, comparable to a real low-hessian row, so fp32
// subtracted histograms never certify a bin empty.
FLC_HOSTDEV inline double MidpointEmptyTolerance(double sum_gradient, double sum_hessian,
                                                 bool fp32_hist, bool subtracted) {
  if (!fp32_hist) {
    return 1e-12 * (fabs(sum_gradient) + fabs(sum_hessian));
  }
  return subtracted ? -1.0 : 0.0;
}

template <typename IS_EMPTY_HIST>
FLC_HOSTDEV inline int GapMidpointThreshold(int threshold, bool reverse, int offset, int num_bin,
                                            bool na_as_missing, bool skip_default_bin, int default_bin,
                                            IS_EMPTY_HIST is_empty_hist) {
  // the threshold range the producing scan covers (its t bounds mapped
  // through t - 1 + offset for the reverse scan, t + offset otherwise)
  const int min_threshold = reverse ? 0 : ((na_as_missing && offset == 1) ? 0 : offset);
  const int max_threshold = reverse ? (num_bin - 2 - (na_as_missing ? 1 : 0)) : (num_bin - 2);
  struct EmptyBin {
    int offset, num_bin, default_bin;
    bool na_as_missing, skip_default_bin;
    IS_EMPTY_HIST hist;
    FLC_HOSTDEV bool operator()(int bin) const {
      const int idx = bin - offset;
      if (idx < 0 || idx >= num_bin - offset) {
        return false;
      }
      if (skip_default_bin && bin == default_bin) {
        return false;
      }
      if (na_as_missing && bin == num_bin - 1) {
        return false;
      }
      return hist(idx);
    }
  };
  const EmptyBin is_empty_bin{offset, num_bin, default_bin, na_as_missing, skip_default_bin, is_empty_hist};
  int lo = threshold;
  int hi = threshold;
  // bin `lo` empty: threshold lo - 1 sends the same rows left
  while (lo - 1 >= min_threshold && is_empty_bin(lo)) {
    --lo;
  }
  // bin `hi + 1` empty: threshold hi + 1 sends the same rows left
  while (hi + 1 <= max_threshold && is_empty_bin(hi + 1)) {
    ++hi;
  }
  return lo + (hi - lo + 1) / 2;
}

}  // namespace SplitGainMath

}  // namespace Falcata

#endif  // FALCATA_TREE_SPLIT_MATH_H_
