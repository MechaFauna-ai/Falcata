/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef FALCATA_SRC_TREELEARNER_CUDA_CUDA_NONQUANT_FIXED_CUH_
#define FALCATA_SRC_TREELEARNER_CUDA_CUDA_NONQUANT_FIXED_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace Falcata {

// Finite score_t inputs only. Store the exponent, not a float multiplier: S can exceed 127 for tiny inputs.
__device__ __forceinline__ int NonquantFixedExponent(const uint32_t absmax_bits, const int lg_n) {
  const float m = __uint_as_float(absmax_bits);
  if (!(m > 0.0f) || isinf(m)) {
    return 0;
  }
  int e = 0;
  frexpf(m, &e);
  return 62 - e - lg_n;
}

__device__ __forceinline__ int64_t NonquantFixedAddend(const float value, const int exponent) {
  return __float2ll_rn(scalbnf(value, exponent));
}

}  // namespace Falcata

#endif  // FALCATA_SRC_TREELEARNER_CUDA_CUDA_NONQUANT_FIXED_CUH_
