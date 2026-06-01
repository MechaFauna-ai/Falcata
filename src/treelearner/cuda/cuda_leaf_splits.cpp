/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_leaf_splits.hpp"

namespace LightGBM {

CUDALeafSplits::CUDALeafSplits(const data_size_t num_data):
num_data_(num_data) {}

CUDALeafSplits::~CUDALeafSplits() {}

void CUDALeafSplits::Init(const bool use_quantized_grad) {
  num_blocks_init_from_gradients_ = (num_data_ + NUM_THREADS_PER_BLOCK_LEAF_SPLITS - 1) / NUM_THREADS_PER_BLOCK_LEAF_SPLITS;

  // allocate more memory for sum reduction in CUDA
  // only the first element records the final sum
  cuda_sum_of_gradients_buffer_.Resize(static_cast<size_t>(num_blocks_init_from_gradients_));
  cuda_sum_of_hessians_buffer_.Resize(static_cast<size_t>(num_blocks_init_from_gradients_));
  if (use_quantized_grad) {
    cuda_sum_of_gradients_hessians_buffer_.Resize(static_cast<size_t>(num_blocks_init_from_gradients_));
  }

  cuda_struct_.Resize(1);
}

void CUDALeafSplits::InitValues() {
  // legacy default stream: later consumers on blocking streams implicitly
  // order after this kernel, so no device sync is needed
  LaunchInitValuesEmptyKernel();
}

void CUDALeafSplits::InitValues(
  const double lambda_l1, const double lambda_l2, const double max_delta_step,
  const score_t* cuda_gradients, const score_t* cuda_hessians,
  const data_size_t* cuda_bagging_data_indices, const data_size_t* cuda_data_indices_in_leaf,
  const data_size_t num_used_indices, hist_t* cuda_hist_in_leaf,
  double* root_sum_gradients, double* root_sum_hessians,
  const bool defer_root_sum_readback) {
  cuda_gradients_ = cuda_gradients;
  cuda_hessians_ = cuda_hessians;
  // async memsets on the legacy default stream: the init kernels follow on the
  // same stream and the synchronous D2H copies below block until completion
  // (SetValue would pay one full device sync each, plus one at the end)
  CUDASUCCESS_OR_FATAL(cudaMemset(reinterpret_cast<void*>(cuda_sum_of_gradients_buffer_.RawData()), 0,
    cuda_sum_of_gradients_buffer_.Size() * sizeof(double)));
  CUDASUCCESS_OR_FATAL(cudaMemset(reinterpret_cast<void*>(cuda_sum_of_hessians_buffer_.RawData()), 0,
    cuda_sum_of_hessians_buffer_.Size() * sizeof(double)));
  LaunchInitValuesKernel(lambda_l1, lambda_l2, max_delta_step, cuda_bagging_data_indices, cuda_data_indices_in_leaf, num_used_indices, cuda_hist_in_leaf);
  if (!defer_root_sum_readback) {
    // synchronous D2H: blocks on everything enqueued so far, including the
    // (large) per-tree histogram-buffer zeroing on the legacy stream. Flows that
    // do not need the host root sums up front (single-sync hybrid growth derives
    // all root gating on-device) defer this to CopyRootSumsToHost, letting the
    // level-0 search enqueue overlap the zeroing.
    CopyFromCUDADeviceToHost<double>(root_sum_gradients, cuda_sum_of_gradients_buffer_.RawData(), 1, __FILE__, __LINE__);
    CopyFromCUDADeviceToHost<double>(root_sum_hessians, cuda_sum_of_hessians_buffer_.RawData(), 1, __FILE__, __LINE__);
  }
}

void CUDALeafSplits::CopyRootSumsToHost(double* root_sum_gradients, double* root_sum_hessians) const {
  CopyFromCUDADeviceToHost<double>(root_sum_gradients, cuda_sum_of_gradients_buffer_.RawDataReadOnly(), 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(root_sum_hessians, cuda_sum_of_hessians_buffer_.RawDataReadOnly(), 1, __FILE__, __LINE__);
}

void CUDALeafSplits::InitValues(
  const double lambda_l1, const double lambda_l2, const double max_delta_step,
  const int16_t* cuda_gradients_and_hessians,
  const data_size_t* cuda_bagging_data_indices,
  const data_size_t* cuda_data_indices_in_leaf, const data_size_t num_used_indices,
  hist_t* cuda_hist_in_leaf, double* root_sum_gradients, double* root_sum_hessians,
  const score_t* grad_scale, const score_t* hess_scale) {
  cuda_gradients_ = reinterpret_cast<const score_t*>(cuda_gradients_and_hessians);
  cuda_hessians_ = nullptr;
  LaunchInitValuesKernel(lambda_l1, lambda_l2, max_delta_step, cuda_bagging_data_indices, cuda_data_indices_in_leaf, num_used_indices, cuda_hist_in_leaf, grad_scale, hess_scale);
  // the synchronous D2H copies block until the kernels above complete; no
  // extra device sync is needed
  CopyFromCUDADeviceToHost<double>(root_sum_gradients, cuda_sum_of_gradients_buffer_.RawData(), 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(root_sum_hessians, cuda_sum_of_hessians_buffer_.RawData(), 1, __FILE__, __LINE__);
}

void CUDALeafSplits::Resize(const data_size_t num_data) {
  num_data_ = num_data;
  num_blocks_init_from_gradients_ = (num_data + NUM_THREADS_PER_BLOCK_LEAF_SPLITS - 1) / NUM_THREADS_PER_BLOCK_LEAF_SPLITS;
  cuda_sum_of_gradients_buffer_.Resize(static_cast<size_t>(num_blocks_init_from_gradients_));
  cuda_sum_of_hessians_buffer_.Resize(static_cast<size_t>(num_blocks_init_from_gradients_));
  cuda_sum_of_gradients_hessians_buffer_.Resize(static_cast<size_t>(num_blocks_init_from_gradients_));
}

}  // namespace LightGBM

#endif  // USE_CUDA
