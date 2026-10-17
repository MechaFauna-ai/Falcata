/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_CUDA

#include "cuda_best_split_finder_kernels.cuh"

namespace Falcata {

template void CUDABestSplitFinder::LaunchFindBestSplitsForLeafKernelInner2<false, true, false>(LaunchFindBestSplitsForLeafKernel_PARAMS);
template void CUDABestSplitFinder::LaunchFindBestSplitsDiscretizedForLeafKernelInner2<false, true, false>(LaunchFindBestSplitsDiscretizedForLeafKernel_PARAMS);

}  // namespace Falcata

#endif  // USE_CUDA
