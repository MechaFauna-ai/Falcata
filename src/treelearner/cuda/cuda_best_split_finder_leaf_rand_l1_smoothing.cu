/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_CUDA

#include "cuda_best_split_finder_kernels.cuh"

namespace Falcata {

template void CUDABestSplitFinder::LaunchFindBestSplitsForLeafKernelInner2<true, true, true>(LaunchFindBestSplitsForLeafKernel_PARAMS);
template void CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLeafKernelInner2<true, true, true>(LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS);

}  // namespace Falcata

#endif  // USE_CUDA
