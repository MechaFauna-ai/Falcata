/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "../../src/treelearner/cuda/cuda_nonquant_fixed.cuh"

namespace {
struct Result {
  double restored;
  double previous;
  int exponent;
};

__global__ void RoundTripFixed(const float value, const int lg_n, Result* result) {
  const int exponent = Falcata::NonquantFixedExponent(__float_as_uint(fabsf(value)), lg_n);
  const int64_t addend = Falcata::NonquantFixedAddend(value, exponent);
  result->exponent = exponent;
  result->restored = ldexp(static_cast<double>(addend), -exponent);
  // The previous implementation's exact clamp/multiply, retained only as a discriminating counterexample.
  const int previous_exponent = max(-126, min(127, exponent));
  result->previous = ldexp(static_cast<double>(__float2ll_rn(value * ldexpf(1.0f, previous_exponent))),
                           -previous_exponent);
}
}  // namespace

TEST(NonquantFixedCUDA, PreservesFiniteFloatRangeWithoutScaleClamp) {
  Result* device = nullptr;
  ASSERT_EQ(cudaMalloc(&device, sizeof(Result)), cudaSuccess);
  const float values[] = {0.0f, std::ldexp(1.0f, -149), std::ldexp(1.0f, -140),
                          std::ldexp(1.25f, -126), -std::ldexp(1.25f, -126),
                          1.25f, std::numeric_limits<float>::max()};
  for (const int lg_n : {0, 12, 31}) {
    for (const float value : values) {
      RoundTripFixed<<<1, 1>>>(value, lg_n, device);
      ASSERT_EQ(cudaGetLastError(), cudaSuccess);
      Result result{};
      ASSERT_EQ(cudaMemcpy(&result, device, sizeof(Result), cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(result.restored, static_cast<double>(value)) << "value=" << value << " lg_n=" << lg_n;
      if (value == std::ldexp(1.0f, -140)) {
        EXPECT_GT(result.exponent, 127);
        EXPECT_EQ(result.previous, 0.0);
      }
      if (value == std::ldexp(1.25f, -126)) {
        EXPECT_GT(result.exponent, 127);
        EXPECT_NEAR(result.previous / static_cast<double>(value), 0.8, 1e-15);
      }
    }
  }
  ASSERT_EQ(cudaFree(device), cudaSuccess);
}
