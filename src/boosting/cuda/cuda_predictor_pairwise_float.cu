/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#include "cuda_predictor_pairwise.cuh"

#ifdef USE_CUDA

namespace Falcata {

template void LaunchPredictPairwiseKernel<float, float>(const DeviceForestView<float>&, const DeviceRows<float>&, int,
                                                        int, int, const PairwisePlan&, double*, double*,
                                                        cudaStream_t);

}  // namespace Falcata

#endif  // USE_CUDA
