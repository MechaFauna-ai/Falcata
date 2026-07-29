/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef FALCATA_SRC_TREELEARNER_CUDA_CUDA_LEAF_SPLITS_HPP_
#define FALCATA_SRC_TREELEARNER_CUDA_CUDA_LEAF_SPLITS_HPP_

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/bin.h>
#include <Falcata/tree_split_math.h>
#include <Falcata/utils/log.h>
#include <Falcata/falcata_plan.h>
#include <Falcata/meta.h>

#include <cstdlib>
#include <string>

#define NUM_THREADS_PER_BLOCK_LEAF_SPLITS (1024)
#define NUM_DATA_THREAD_ADD_LEAF_SPLITS (6)

namespace Falcata {

/*! \brief kill switch for the wide-shape batched level support (many split-find
 *  tasks and/or compact-column-view histogram data): FALCATA_BATCH_WIDE=0
 *  restores the previous fallback to the per-pair kernels for those shapes */
inline bool FalcataBatchWideEnabled() {
  return FalcataPlan::Get().batch_wide;
}

struct CUDALeafSplitsStruct {
 public:
  int leaf_index;
  double sum_of_gradients;
  double sum_of_hessians;
  int64_t sum_of_gradients_hessians;
  data_size_t num_data_in_leaf;
  double gain;
  double leaf_value;
  const data_size_t* data_indices_in_leaf;
  hist_t* hist_in_leaf;
};

/*! \brief Per-sibling-pair metadata for the hybrid level-batched growth phase.
 *  One entry per pair of a level; uploaded to the device in a single H2D copy so
 *  the batched construct/fix/subtract/find/sync kernels can cover all pairs of a
 *  level with one launch each (indexed by a grid dimension). All fields are
 *  host-known at level start (single-GPU only, so global == local leaf counts). */
struct CUDAHybridPairDescriptor {
  const CUDALeafSplitsStruct* smaller_struct;
  const CUDALeafSplitsStruct* larger_struct;
  int smaller_leaf_index;
  int larger_leaf_index;
  data_size_t num_data_in_smaller_leaf;
  data_size_t num_data_in_larger_leaf;
  /*! \brief histogram construction needed (mirror of ConstructHistogramForLeaf's
   *  min_data/min_hessian early-return) */
  uint8_t construct_valid;
  /*! \brief smaller/larger leaf pass the best-split-search validity checks
   *  (min_data_in_leaf, min_sum_hessian_in_leaf, below max_depth) */
  uint8_t smaller_valid;
  uint8_t larger_valid;
  /*! \brief per-pair histogram bit widths for quantized training (0 when unused) */
  uint8_t parent_num_bits;
  uint8_t smaller_num_bits;
  uint8_t larger_num_bits;
};

class CUDALeafSplits: public NCCLInfo {
 public:
  explicit CUDALeafSplits(const data_size_t num_data);

  ~CUDALeafSplits();

  void Init(const bool use_quantized_grad);

  void InitValues(
    const double lambda_l1, const double lambda_l2, const double max_delta_step,
    const score_t* cuda_gradients, const score_t* cuda_hessians,
    const data_size_t* cuda_bagging_data_indices,
    const data_size_t* cuda_data_indices_in_leaf, const data_size_t num_used_indices,
    hist_t* cuda_hist_in_leaf, double* root_sum_gradients, double* root_sum_hessians,
    const bool defer_root_sum_readback = false);

  /*! \brief deferred counterpart of InitValues' root-sum readback (see there) */
  void CopyRootSumsToHost(double* root_sum_gradients, double* root_sum_hessians) const;

  void InitValues(
    const double lambda_l1, const double lambda_l2, const double max_delta_step,
    const int16_t* cuda_gradients_and_hessians,
    const data_size_t* cuda_bagging_data_indices,
    const data_size_t* cuda_data_indices_in_leaf, const data_size_t num_used_indices,
    hist_t* cuda_hist_in_leaf, double* root_sum_gradients, double* root_sum_hessians,
    const score_t* grad_scale, const score_t* hess_scale);

  void InitValues();

  const CUDALeafSplitsStruct* GetCUDAStruct() const { return cuda_struct_.RawDataReadOnly(); }

  CUDALeafSplitsStruct* GetCUDAStructRef() { return cuda_struct_.RawData(); }

  void Resize(const data_size_t num_data);

  // These delegate to the single shared SplitGainMath core (tree_split_math.h)
  // so the CUDA and CPU paths use identical formulas. Names/signatures are kept
  // for the existing device call sites; T = float only in the fp32 gain mode
  // (FALCATA_FP32_GAIN), always spelled out explicitly at those call sites.
  //
  // CPU selects the USE_MAX_OUTPUT specialisation at compile time. Doing the
  // same here would add another template dimension to every split-finder
  // kernel, so instead the specialisation is selected at runtime from the value
  // of max_delta_step. The two branches must not be collapsed into one: with the
  // cap enabled the leaf gain is measured at the (possibly capped) output, which
  // is algebraically but *not* bitwise equal to the closed-form g^2/(h+l2) that
  // CPU uses when max_delta_step is unset. Keeping both preserves CPU/CUDA
  // bit-parity for the (overwhelmingly common) max_delta_step == 0 case.
  __device__ static double ThresholdL1(double s, double l1) {
    return SplitGainMath::ThresholdL1(s, l1);
  }

  template <bool USE_L1, bool USE_SMOOTHING, typename T = double>
  __device__ static T CalculateSplittedLeafOutput(T sum_gradients,
                                          T sum_hessians, T l1, T l2,
                                          T path_smooth, T max_delta_step,
                                          data_size_t num_data,
                                          T parent_output) {
    return max_delta_step > static_cast<T>(0)
      ? SplitGainMath::CalculateLeafOutput<USE_L1, true, USE_SMOOTHING, T>(
            sum_gradients, sum_hessians, l1, l2, max_delta_step, path_smooth,
            num_data, parent_output)
      : SplitGainMath::CalculateLeafOutput<USE_L1, false, USE_SMOOTHING, T>(
            sum_gradients, sum_hessians, l1, l2, static_cast<T>(0), path_smooth,
            num_data, parent_output);
  }

  template <bool USE_L1, typename T = double>
  __device__ static T GetLeafGainGivenOutput(T sum_gradients,
                                      T sum_hessians, T l1,
                                      T l2, T output) {
    return SplitGainMath::LeafGainGivenOutput<USE_L1, T>(
        sum_gradients, sum_hessians, l1, l2, output);
  }

  template <bool USE_L1, bool USE_SMOOTHING, typename T = double>
  __device__ static T GetLeafGain(T sum_gradients, T sum_hessians,
                          T l1, T l2,
                          T path_smooth, T max_delta_step,
                          data_size_t num_data,
                          T parent_output) {
    return max_delta_step > static_cast<T>(0)
      ? SplitGainMath::LeafGain<USE_L1, true, USE_SMOOTHING, T>(
            sum_gradients, sum_hessians, l1, l2, max_delta_step, path_smooth,
            num_data, parent_output)
      : SplitGainMath::LeafGain<USE_L1, false, USE_SMOOTHING, T>(
            sum_gradients, sum_hessians, l1, l2, static_cast<T>(0), path_smooth,
            num_data, parent_output);
  }

  template <bool USE_L1, bool USE_SMOOTHING, typename T = double>
  __device__ static T GetSplitGains(T sum_left_gradients,
                            T sum_left_hessians,
                            T sum_right_gradients,
                            T sum_right_hessians,
                            T l1, T l2,
                            T path_smooth,
                            T max_delta_step,
                            data_size_t left_count,
                            data_size_t right_count,
                            T parent_output) {
    return GetLeafGain<USE_L1, USE_SMOOTHING, T>(sum_left_gradients,
                      sum_left_hessians,
                      l1, l2, path_smooth, max_delta_step, left_count, parent_output) +
          GetLeafGain<USE_L1, USE_SMOOTHING, T>(sum_right_gradients,
                      sum_right_hessians,
                      l1, l2, path_smooth, max_delta_step, right_count, parent_output);
  }

  // Monotone-constraint-aware leaf output: analytic output clamped into the
  // per-leaf [constraint_min, constraint_max] interval. Mirrors the CPU
  // FeatureHistogram::CalculateSplittedLeafOutput<USE_MC, ...> clamp.
  template <bool USE_L1, bool USE_SMOOTHING>
  __device__ static double CalculateSplittedLeafOutputMC(double sum_gradients,
                                          double sum_hessians, double l1, double l2,
                                          double path_smooth, double max_delta_step,
                                          data_size_t num_data,
                                          double parent_output,
                                          double constraint_min, double constraint_max) {
    // The max_delta_step cap is part of the analytic (unconstrained) output and
    // is applied inside CalculateSplittedLeafOutput, before the MC clamp below,
    // matching the CPU ordering (USE_MAX_OUTPUT cap then USE_MC clamp).
    double ret = CalculateSplittedLeafOutput<USE_L1, USE_SMOOTHING>(
        sum_gradients, sum_hessians, l1, l2, path_smooth, max_delta_step, num_data, parent_output);
    if (ret < constraint_min) {
      ret = constraint_min;
    } else if (ret > constraint_max) {
      ret = constraint_max;
    }
    return ret;
  }

  // Monotone-constraint-aware split gain. Mirrors the CPU
  // FeatureHistogram::GetSplitGains<USE_MC=true, ...> branch:
  //   * clamp both child outputs into the per-leaf [min,max]
  //   * return 0 if the monotonicity relation between children is violated
  //   * otherwise return GetLeafGainGivenOutput(left)+GetLeafGainGivenOutput(right)
  // For a feature with monotone_constraint==0 and an unconstrained leaf
  // ([-DBL_MAX, +DBL_MAX]) this is mathematically identical to the non-MC
  // analytic gain, and is exactly what the CPU computes when
  // monotone_constraints is non-empty (USE_MC is then on for every feature).
  template <bool USE_L1, bool USE_SMOOTHING>
  __device__ static double GetSplitGainsMC(double sum_left_gradients,
                            double sum_left_hessians,
                            double sum_right_gradients,
                            double sum_right_hessians,
                            double l1, double l2,
                            double path_smooth,
                            double max_delta_step,
                            data_size_t left_count,
                            data_size_t right_count,
                            double parent_output,
                            double constraint_min, double constraint_max,
                            int8_t monotone_constraint) {
    const double left_output = CalculateSplittedLeafOutputMC<USE_L1, USE_SMOOTHING>(
        sum_left_gradients, sum_left_hessians, l1, l2, path_smooth, max_delta_step, left_count,
        parent_output, constraint_min, constraint_max);
    const double right_output = CalculateSplittedLeafOutputMC<USE_L1, USE_SMOOTHING>(
        sum_right_gradients, sum_right_hessians, l1, l2, path_smooth, max_delta_step, right_count,
        parent_output, constraint_min, constraint_max);
    if (((monotone_constraint > 0) && (left_output > right_output)) ||
        ((monotone_constraint < 0) && (left_output < right_output))) {
      return 0;
    }
    return GetLeafGainGivenOutput<USE_L1>(sum_left_gradients, sum_left_hessians, l1, l2, left_output) +
           GetLeafGainGivenOutput<USE_L1>(sum_right_gradients, sum_right_hessians, l1, l2, right_output);
  }

 private:
  void LaunchInitValuesEmptyKernel();

  void LaunchInitValuesKernel(
    const double lambda_l1, const double lambda_l2, const double max_delta_step,
    const data_size_t* cuda_bagging_data_indices,
    const data_size_t* cuda_data_indices_in_leaf,
    const data_size_t num_used_indices,
    hist_t* cuda_hist_in_leaf);

  void LaunchInitValuesKernel(
    const double lambda_l1, const double lambda_l2, const double max_delta_step,
    const data_size_t* cuda_bagging_data_indices,
    const data_size_t* cuda_data_indices_in_leaf,
    const data_size_t num_used_indices,
    hist_t* cuda_hist_in_leaf,
    const score_t* grad_scale,
    const score_t* hess_scale);

  // Host memory
  data_size_t num_data_;
  int num_blocks_init_from_gradients_;

  // CUDA memory, held by this object
  CUDAVector<CUDALeafSplitsStruct> cuda_struct_;
  CUDAVector<double> cuda_sum_of_gradients_buffer_;
  CUDAVector<double> cuda_sum_of_hessians_buffer_;
  CUDAVector<int64_t> cuda_sum_of_gradients_hessians_buffer_;

  // CUDA memory, held by other object
  const score_t* cuda_gradients_;
  const score_t* cuda_hessians_;
};

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_SRC_TREELEARNER_CUDA_CUDA_LEAF_SPLITS_HPP_
