/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
// The level split finder's L1 / path-smoothing variants, in their own module: a run without L1 and path smoothing
// (the default) never loads them (see CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLevelKernel).
#ifdef USE_CUDA

#include "cuda_best_split_finder_level.cuh"

namespace Falcata {

template void CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLevelKernelInner<true, false>(
  dim3, const int*, const int*, const CUDAHybridPairDescriptor*, const score_t*, const score_t*);
template void CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLevelKernelInner<false, true>(
  dim3, const int*, const int*, const CUDAHybridPairDescriptor*, const score_t*, const score_t*);
template void CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLevelKernelInner<true, true>(
  dim3, const int*, const int*, const CUDAHybridPairDescriptor*, const score_t*, const score_t*);

}  // namespace Falcata

#endif  // USE_CUDA
