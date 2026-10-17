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

}  // namespace Falcata

#endif  // USE_CUDA
