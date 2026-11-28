/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef FALCATA_INCLUDE_FALCATA_CUDA_CUDA_OBJECTIVE_FUNCTION_HPP_
#define FALCATA_INCLUDE_FALCATA_CUDA_CUDA_OBJECTIVE_FUNCTION_HPP_

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/falcata_plan.h>
#include <Falcata/objective_function.h>
#include <Falcata/meta.h>

#include <string>
#include <vector>

namespace Falcata {

template <typename HOST_OBJECTIVE>
class CUDAObjectiveInterface: public HOST_OBJECTIVE, public NCCLInfo {
 public:
  explicit CUDAObjectiveInterface(const Config& config): HOST_OBJECTIVE(config) {
    if (config.num_gpu <= 1) {
      const int gpu_device_id = config.gpu_device_id >= 0 ? config.gpu_device_id : 0;
      SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
    }
  }

  explicit CUDAObjectiveInterface(const std::vector<std::string>& strs): HOST_OBJECTIVE(strs) {}

  void Init(const Metadata& metadata, data_size_t num_data) {
    HOST_OBJECTIVE::Init(metadata, num_data);
    if (metadata.cuda_metadata() == nullptr) {
      Log::Fatal(
          "CUDA objective requires a Dataset constructed with device_type=\"cuda\". "
          "Pass device=\"cuda\" in the Dataset's params (e.g. flc.Dataset(X, y, params={\"device\": \"cuda\"})).");
    }
    cuda_labels_ = metadata.cuda_metadata()->cuda_label();
    cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
    ClearPrecomputedGradMinMax();
    const_hess_filled_ptr_ = nullptr;
    const_hess_filled_count_ = 0;
  }

  void SetNCCLInfo(
    ncclComm_t nccl_communicator,
    int nccl_gpu_rank,
    int local_gpu_rank,
    int gpu_device_id,
    data_size_t global_num_data) override {
    NCCLInfo::SetNCCLInfo(nccl_communicator, nccl_gpu_rank, local_gpu_rank, gpu_device_id, global_num_data);
  }

  virtual const double* ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
    return LaunchConvertOutputCUDAKernel(num_data, input, output);
  }

  double BoostFromScore(int class_id) const override {
    return LaunchCalcInitScoreKernel(class_id);
  }

  bool IsCUDAObjective() const override { return true; }

  void GetGradients(const double* scores, score_t* gradients, score_t* hessians) const override {
    ClearPrecomputedGradMinMax();
    LaunchGetGradientsKernel(scores, gradients, hessians);
    if (!GradientsNeedNoSync(gradients, hessians)) {
      SynchronizeCUDADevice(__FILE__, __LINE__);
    }
  }

  // all queries get gradients (a superset); bagging_by_query sampling is applied through the bag
  // indices the tree learner receives
  void GetGradientsWithSampledQueries(const double* scores, const data_size_t /*num_sampled_queries*/, const data_size_t* /*sampled_query_indices*/, score_t* gradients, score_t* hessians) const override {
    ClearPrecomputedGradMinMax();
    LaunchGetGradientsKernel(scores, gradients, hessians);
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }

  bool GetPrecomputedGradMinMax(int class_id, ObjectiveFunction::CUDAPrecomputedGradMinMax* out) const override {
    if (out == nullptr || minmax_num_classes_ <= 0 || class_id < 0 || class_id >= minmax_num_classes_ ||
        minmax_num_blocks_ <= 0) {
      return false;
    }
    const size_t stride = static_cast<size_t>(minmax_num_blocks_);
    const score_t* base = cuda_block_minmax_.RawDataReadOnly() + static_cast<size_t>(class_id) * 4 * stride;
    out->grad_min_blocks = base;
    out->grad_max_blocks = base + stride;
    out->hess_min_blocks = base + 2 * stride;
    out->hess_max_blocks = base + 3 * stride;
    out->for_gradients = minmax_base_gradients_ + static_cast<size_t>(class_id) * static_cast<size_t>(minmax_num_data_);
    out->for_hessians = minmax_base_hessians_ + static_cast<size_t>(class_id) * static_cast<size_t>(minmax_num_data_);
    out->num_data = minmax_num_data_;
    out->num_blocks = minmax_num_blocks_;
    return true;
  }

  void RenewTreeOutputCUDA(const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const override {
    global_timer.Start("CUDAObjectiveInterface::LaunchRenewTreeOutputCUDAKernel");
    LaunchRenewTreeOutputCUDAKernel(score, data_indices_in_leaf, num_data_in_leaf, data_start_in_leaf, num_leaves, leaf_value);
    SynchronizeCUDADevice(__FILE__, __LINE__);
    global_timer.Stop("CUDAObjectiveInterface::LaunchRenewTreeOutputCUDAKernel");
  }

 protected:
  /*! \brief cuda_plan key gradients_no_sync: true when both outputs are device memory (the host cannot read them
   *  without a stream-ordered copy) and the plan allows it */
  static bool GradientsNeedNoSync(const score_t* gradients, const score_t* hessians) {
    const FalcataPlan& plan = FalcataPlan::Get();
    if (!plan.gradients_no_sync || plan.compact_prefill) {
      return false;
    }
    for (const score_t* ptr : {gradients, hessians}) {
      cudaPointerAttributes attributes;
      if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) {
        cudaGetLastError();
        return false;
      }
      if (attributes.type != cudaMemoryTypeDevice) {
        return false;
      }
    }
    return true;
  }

  virtual void LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const = 0;

  virtual double LaunchCalcInitScoreKernel(const int class_id) const {
    return HOST_OBJECTIVE::BoostFromScore(class_id);
  }

  virtual const double* LaunchConvertOutputCUDAKernel(const data_size_t /*num_data*/, const double* input, double* /*output*/) const { return input; }

  virtual void LaunchRenewTreeOutputCUDAKernel(
    const double* /*score*/, const data_size_t* /*data_indices_in_leaf*/, const data_size_t* /*num_data_in_leaf*/,
    const data_size_t* /*data_start_in_leaf*/, const int /*num_leaves*/, double* /*leaf_value*/) const {}

  void ClearPrecomputedGradMinMax() const {
    minmax_base_gradients_ = nullptr;
    minmax_base_hessians_ = nullptr;
    minmax_num_data_ = 0;
    minmax_num_blocks_ = 0;
    minmax_num_classes_ = 0;
  }

  // cuda_plan key grad_minmax_fuse: allocate (lazily) the [num_classes][4][num_blocks] block min/max buffer and
  // record which (gradients, hessians) arrays it corresponds to.
  score_t* PreparePrecomputedGradMinMax(const score_t* gradients, const score_t* hessians,
                                        const data_size_t num_data, const int num_blocks,
                                        const int num_classes) const {
    const size_t total = static_cast<size_t>(num_classes) * 4 * static_cast<size_t>(num_blocks);
    if (cuda_block_minmax_.Size() < total) {
      cuda_block_minmax_.Resize(total);
    }
    minmax_base_gradients_ = gradients;
    minmax_base_hessians_ = hessians;
    minmax_num_data_ = num_data;
    minmax_num_blocks_ = num_blocks;
    minmax_num_classes_ = num_classes;
    return cuda_block_minmax_.RawData();
  }

  // cuda_plan key const_hess_no_write: after the first unweighted constant-hessian GetGradients call fills `hessians`
  // with 1.0f for all `count` elements, later calls targeting the same buffer skip rewriting `hessians[i] = 1.0f`.
  bool SkipConstantHessianWrite(const score_t* hessians, const size_t count) const {
    if (!FalcataPlan::Get().const_hess_no_write || cuda_weights_ != nullptr || hessians == nullptr || count == 0) {
      return false;
    }
    if (hessians == const_hess_filled_ptr_ && count == const_hess_filled_count_) {
      return true;
    }
    const_hess_filled_ptr_ = hessians;
    const_hess_filled_count_ = count;
    return false;
  }

  const label_t* cuda_labels_;
  const label_t* cuda_weights_;
  mutable CUDAVector<score_t> cuda_block_minmax_;
  mutable const score_t* minmax_base_gradients_ = nullptr;
  mutable const score_t* minmax_base_hessians_ = nullptr;
  mutable data_size_t minmax_num_data_ = 0;
  mutable int minmax_num_blocks_ = 0;
  mutable int minmax_num_classes_ = 0;
  mutable const score_t* const_hess_filled_ptr_ = nullptr;
  mutable size_t const_hess_filled_count_ = 0;
};

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_INCLUDE_FALCATA_CUDA_CUDA_OBJECTIVE_FUNCTION_HPP_
