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

namespace {

template <int QuantHistSlots>
void WarmupQuantizedByteKernels(const bool device_levels) {
  cudaFuncAttributes attributes{};
  const auto host_kernel = CUDAConstructDiscretizedHistogramDenseBatchedKernel<
    uint8_t, QuantHistSlots, PackRaw8, false>;
  if (cudaFuncGetAttributes(&attributes, host_kernel) != cudaSuccess) {
    cudaGetLastError();
    return;
  }
  const auto device_kernel = CUDAConstructDiscretizedHistogramDenseBatchedKernel<
    uint8_t, QuantHistSlots, PackRaw8, true>;
  if (device_levels && cudaFuncGetAttributes(&attributes, device_kernel) != cudaSuccess) {
    cudaGetLastError();
  }
}

}  // namespace

void CUDAHistogramConstructor::WarmupBatchedByteHistograms(const bool device_levels) const {
  if (cuda_row_data_ == nullptr || num_features_ <= 0 ||
      cuda_row_data_->bit_type() != 8 || cuda_row_data_->is_4bit_packed()) {
    return;
  }
  // Query the first-use kernels in their own translation units. Under lazy loading
  // this pays the two module loads during booster creation, before the first update.
  if (!WarmupBatchedMetadataKernel()) {
    return;
  }
  switch (cuda_row_data_->quant_hist_slots()) {
    case DP_SHARED_HIST_SIZE:
      WarmupQuantizedByteKernels<DP_SHARED_HIST_SIZE>(device_levels);
      break;
    case SP_SHARED_HIST_SIZE:
      WarmupQuantizedByteKernels<SP_SHARED_HIST_SIZE>(device_levels);
      break;
    case SP_SHARED_HIST_SIZE * 2:
      WarmupQuantizedByteKernels<SP_SHARED_HIST_SIZE * 2>(device_levels);
      break;
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
