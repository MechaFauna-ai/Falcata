/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_CUDA

#include "cuda_histogram_constructor_kernels.cuh"

namespace Falcata {

template void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner0<double, DP_SHARED_HIST_SIZE, uint8_t>(const CUDALeafSplitsStruct*, const data_size_t, const uint8_t);
template void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner0<float, SP_SHARED_HIST_SIZE, uint8_t>(const CUDALeafSplitsStruct*, const data_size_t, const uint8_t);
template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<double, DP_SHARED_HIST_SIZE, uint8_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<double, DP_SHARED_HIST_SIZE, uint8_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<float, SP_SHARED_HIST_SIZE, uint8_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<float, SP_SHARED_HIST_SIZE, uint8_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, DP_SHARED_HIST_SIZE, uint8_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, DP_SHARED_HIST_SIZE, uint8_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, SP_SHARED_HIST_SIZE, uint8_t, true>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);
template void CUDAHistogramConstructor::LaunchConstructHistogramBatchedKernelInner0<nq_fixed_t, SP_SHARED_HIST_SIZE, uint8_t, false>(const CUDAHybridPairDescriptor*, const int, const data_size_t, const data_size_t*);

void CUDAHistogramConstructor::WarmupBatchedByteHistograms(const bool device_levels) const {
  if (cuda_row_data_->bit_type() != 8 || cuda_row_data_->is_4bit_packed()) {
    return;
  }
  // Query the first-use kernels in their own translation units. Under lazy loading
  // this pays the two module loads during booster creation, before the first update.
  WarmupBatchedMetadataKernel();
  cudaFuncAttributes attributes{};
  const auto host_level_kernel = gpu_use_dp_ ?
    CUDAConstructDiscretizedHistogramDenseBatchedKernel<uint8_t, DP_SHARED_HIST_SIZE, PackRaw8, false> :
    CUDAConstructDiscretizedHistogramDenseBatchedKernel<uint8_t, SP_SHARED_HIST_SIZE, PackRaw8, false>;
  CUDASUCCESS_OR_FATAL(cudaFuncGetAttributes(&attributes, host_level_kernel));
  if (device_levels) {
    const auto device_level_kernel = gpu_use_dp_ ?
      CUDAConstructDiscretizedHistogramDenseBatchedKernel<uint8_t, DP_SHARED_HIST_SIZE, PackRaw8, true> :
      CUDAConstructDiscretizedHistogramDenseBatchedKernel<uint8_t, SP_SHARED_HIST_SIZE, PackRaw8, true>;
    CUDASUCCESS_OR_FATAL(cudaFuncGetAttributes(&attributes, device_level_kernel));
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
