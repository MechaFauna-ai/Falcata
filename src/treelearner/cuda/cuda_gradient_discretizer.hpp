/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef FALCATA_SRC_TREELEARNER_CUDA_CUDA_GRADIENT_DISCRETIZER_HPP_
#define FALCATA_SRC_TREELEARNER_CUDA_CUDA_GRADIENT_DISCRETIZER_HPP_

#ifdef USE_CUDA

#include <Falcata/bin.h>
#include <Falcata/meta.h>
#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/utils/threading.h>

#include <algorithm>
#include <random>
#include <vector>

#include "cuda_leaf_splits.hpp"
#include "../gradient_discretizer.hpp"

namespace Falcata {

#define CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE (1024)

class CUDAGradientDiscretizer: public GradientDiscretizer, public NCCLInfo {
 public:
  CUDAGradientDiscretizer(int num_grad_quant_bins, int num_trees, int random_seed, bool is_constant_hessian, bool stochastic_roudning):
    GradientDiscretizer(num_grad_quant_bins, num_trees, random_seed, is_constant_hessian, stochastic_roudning) {
  }

  ~CUDAGradientDiscretizer() {}

  void DiscretizeGradients(
    const data_size_t num_data,
    const score_t* input_gradients,
    const score_t* input_hessians) override;

  const int8_t* discretized_gradients_and_hessians() const override { return discretized_gradients_and_hessians_.RawData(); }

  double grad_scale() const override {
    Log::Fatal("grad_scale() of CUDAGradientDiscretizer should not be called.");
    return 0.0;
  }

  double hess_scale() const override {
    Log::Fatal("hess_scale() of CUDAGradientDiscretizer should not be called.");
    return 0.0;
  }

  const score_t* grad_scale_ptr() const { return grad_max_block_buffer_.RawData(); }

  const score_t* hess_scale_ptr() const { return hess_max_block_buffer_.RawData(); }

  // Enable the outlier-robust gradient scale (fixed-point mode only). When on,
  // the grad scale is derived from a high percentile of |grad| via a cheap
  // log-magnitude histogram instead of the global max, and the discretize kernel
  // clamps quantized magnitudes to +/-(bins/2) so rare outliers saturate.
  void SetRobustScale(bool robust_scale) { robust_scale_ = robust_scale; }

  // Enable error-feedback accumulation (fixed-point mode only). Each row keeps
  // the residual of its last rounding and adds it before the next one, so the
  // per-row quantization error telescopes across trees instead of accumulating
  // as bias (the visible failure mode: single-row leaves at low bin budgets).
  // num_slots must be the number of trees per boosting iteration (num_class
  // for multiclass): the learner trains classes in a fixed order and iter_
  // advances once per tree, so iter_ % num_slots addresses this tree's class
  // and residuals never mix across classes.
  void SetErrorFeedback(bool error_feedback, int num_slots) {
    error_feedback_ = error_feedback;
    ef_num_slots_ = num_slots > 0 ? num_slots : 1;
  }

  // Under bagging, residuals must only move for rows whose quantized gradient
  // is actually consumed this tree. The kernel discretizes ALL rows, so
  // without this an out-of-bag row accumulates corrections for values nothing
  // ever read -- a stale, wrong carry by the time it re-enters the bag
  // (measured: EF helped the no-bagging fuzz corner and HURT the bagged one
  // until this gate existed). Pass the device array of in-bag row indices
  // before each tree's DiscretizeGradients; nullptr means no bagging.
  void SetBagForThisTree(const data_size_t* inbag_indices, data_size_t inbag_count) {
    ef_inbag_indices_ = inbag_indices;
    ef_inbag_count_ = inbag_count;
  }

  void Init(const data_size_t num_data, const int num_leaves,
    const int num_features, const Dataset* train_data) override {
    GradientDiscretizer::Init(num_data, num_leaves, num_features, train_data);
    // Each data point stores an int16 gradient and an int16 hessian (see
    // DiscretizeGradientsKernel, which writes through an int16_t* view). That is
    // 2 * sizeof(int16_t) = 4 bytes per data point. The buffer is int8_t, so it
    // must hold num_data * 4 elements; sizing it as num_data * 2 (the original
    // 8-bit layout) under-allocates by 2x and lets the discretize kernel overrun
    // into the adjacent gradient/hessian scale buffers, corrupting the dequant
    // scales and producing garbage leaf sums (no splits under use_quantized_grad).
    discretized_gradients_and_hessians_.Resize(num_data * 4);
    num_reduce_blocks_ = (num_data + CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE - 1) / CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE;
    grad_min_block_buffer_.Resize(num_reduce_blocks_);
    grad_max_block_buffer_.Resize(num_reduce_blocks_);
    hess_min_block_buffer_.Resize(num_reduce_blocks_);
    hess_max_block_buffer_.Resize(num_reduce_blocks_);
    // Magnitude histogram for the outlier-robust grad scale (fixed-point mode).
    // kNumMagBuckets log-spaced buckets over |grad| in (0, grad_abs_max]; a bucket
    // scan then finds the percentile threshold. Allocated unconditionally (a few
    // KB) so toggling the robust flag needs no re-Init.
    grad_mag_hist_buffer_.Resize(kNumMagBuckets);
    if (error_feedback_) {
      // one (hess, grad) float pair per row per class slot, zero-initialized:
      // a fresh model starts with no carried residual
      // int8 residuals in units of 1/128 of a quant bin: the carry's own
      // quantum (1/256 bin worst case) is 128x finer than the 1/2-bin rounding
      // error it corrects, and the buffer costs 2 bytes/row/class instead of 8
      // -- this matters because the numerai treecount gate already peaks
      // within ~300MiB of the card, where a float buffer measurably tipped
      // runs into OOM. The in-bag mask is allocated lazily on first bagged
      // tree for the same reason.
      const size_t ef_size = static_cast<size_t>(num_data) * 2 * ef_num_slots_;
      ef_residuals_.Resize(ef_size);
      SetCUDAMemory<int8_t>(ef_residuals_.RawData(), 0, ef_size, __FILE__, __LINE__);
    }
    // Stochastic rounding noise is Philox-generated in-kernel from
    // (random_seed_, tree, row) -- no tables (formerly 2 float arrays of
    // num_data each, 8 bytes/row of resident VRAM + per-tree reads).
    iter_ = 0;
  }

 protected:
  // Number of log-spaced magnitude buckets for the robust-scale gap scan.
  // At 128 buckets/octave (see kMagBucketsPerOctave) this spans 16 octaves of
  // |grad| dynamic range downward from the global max -- fine enough to place the
  // bulk anchor precisely, and small enough (2048*4B = 8KB) to hold a per-block
  // histogram in shared memory (cutting global-atomic contention on large data).
  static const int kNumMagBuckets = 2048;

  mutable CUDAVector<int8_t> discretized_gradients_and_hessians_;
  mutable CUDAVector<score_t> grad_min_block_buffer_;
  mutable CUDAVector<score_t> grad_max_block_buffer_;
  mutable CUDAVector<score_t> hess_min_block_buffer_;
  mutable CUDAVector<score_t> hess_max_block_buffer_;
  mutable CUDAVector<unsigned int> grad_mag_hist_buffer_;
  mutable CUDAVector<int8_t> ef_residuals_;
  mutable CUDAVector<uint8_t> ef_inbag_mask_;
  int num_reduce_blocks_;
  bool robust_scale_ = false;
  bool error_feedback_ = false;
  int ef_num_slots_ = 1;
  const data_size_t* ef_inbag_indices_ = nullptr;
  data_size_t ef_inbag_count_ = 0;
};

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_SRC_TREELEARNER_CUDA_CUDA_GRADIENT_DISCRETIZER_HPP_
