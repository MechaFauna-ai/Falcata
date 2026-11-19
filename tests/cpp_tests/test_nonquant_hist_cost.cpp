/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

// cuda_plan key nonquant_hist_cost: the rule that picks the cheaper non-quantized histogram construct
// (FalcataPlan::NonquantOrderedPays). Pure arithmetic, so it runs in the CPU test binary too.

#include <gtest/gtest.h>

#include <Falcata/falcata_plan.h>

#include <cmath>

using Falcata::FalcataPlan;

TEST(NonquantHistCost, RhoIsHistogramWorkOverRowWork) {
  // 1023 leaves at depth 10 over 400000 rows of 2000 features with 255 bins each: 1023 * 255 / (400000 * 10)
  EXPECT_NEAR(FalcataPlan::NonquantHistRho(400000, 2000, 2000 * 255, 1023, 10), 1023.0 * 255 / 4000000.0, 1e-12);
  // levels: a balanced tree of num_leaves leaves (ceil(log2)), capped by max_depth when that is set
  EXPECT_NEAR(FalcataPlan::NonquantHistRho(1000000, 10, 2550, 63, -1), 63.0 * 255 / (1000000.0 * 6), 1e-12);
  EXPECT_NEAR(FalcataPlan::NonquantHistRho(1000000, 10, 2550, 1023, 4), 1023.0 * 255 / (1000000.0 * 4), 1e-12);
  EXPECT_NEAR(FalcataPlan::NonquantHistRho(1000000, 10, 2550, 2, -1), 2.0 * 255 / 1000000.0, 1e-12);
}

TEST(NonquantHistCost, EmptyShapesTakeTheAtomicKernels) {
  EXPECT_TRUE(std::isinf(FalcataPlan::NonquantHistRho(0, 10, 100, 31, -1)));
  EXPECT_TRUE(std::isinf(FalcataPlan::NonquantHistRho(1000, 0, 100, 31, -1)));
  EXPECT_TRUE(std::isinf(FalcataPlan::NonquantHistRho(1000, 10, 0, 31, -1)));
  EXPECT_TRUE(std::isinf(FalcataPlan::NonquantHistRho(1000, 10, 100, 1, -1)));
  EXPECT_FALSE(FalcataPlan::NonquantOrderedPays(0, 10, 100, 31, -1, false));
  EXPECT_FALSE(FalcataPlan::NonquantOrderedPays(0, 10, 100, 31, -1, true));
}

TEST(NonquantHistCost, ThresholdPerLayout) {
  // at the threshold the fixed-point construct is taken, just above it the atomic kernels; rows set rho exactly
  const double t64 = FalcataPlan::kNonquantOrderedMaxRhoFp64;
  const double t32 = FalcataPlan::kNonquantOrderedMaxRhoFp32;
  // rho = 64 * 100 / (rows * 6) for 64 leaves, 100 bins per feature
  const auto rows_for = [](double rho) { return static_cast<int64_t>(std::ceil(64.0 * 100 / (rho * 6))); };
  EXPECT_TRUE(FalcataPlan::NonquantOrderedPays(rows_for(t64), 10, 1000, 64, 6, false));
  EXPECT_FALSE(FalcataPlan::NonquantOrderedPays(rows_for(t64) / 2, 10, 1000, 64, 6, false));
  EXPECT_TRUE(FalcataPlan::NonquantOrderedPays(rows_for(t32), 10, 1000, 64, 6, true));
  EXPECT_FALSE(FalcataPlan::NonquantOrderedPays(rows_for(t32) / 2, 10, 1000, 64, 6, true));
}
