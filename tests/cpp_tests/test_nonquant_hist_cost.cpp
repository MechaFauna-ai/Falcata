/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

// cuda_plan key nonquant_hist_cost: the rule that picks the cheaper non-quantized histogram construct
// Pure arithmetic, so it runs in the CPU test binary too.

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
  const double rho = FalcataPlan::NonquantHistRho(0, 10, 100, 31, -1);
  EXPECT_GT(FalcataPlan::NonquantOrderedCost(rho, 100, false), FalcataPlan::NonquantOrderedMaxCost(false));
  EXPECT_GT(FalcataPlan::NonquantOrderedCost(rho, 100, true), FalcataPlan::NonquantOrderedMaxCost(true));
}

TEST(NonquantHistCost, Fp32WeightsTheFullLeafFootprint) {
  // Both shapes have the same bins per feature, leaves, rows and depth.
  const double rho = FalcataPlan::NonquantHistRho(460000, 8, 2048, 63, 6);
  EXPECT_DOUBLE_EQ(rho, FalcataPlan::NonquantHistRho(460000, 96, 24576, 63, 6));
  EXPECT_DOUBLE_EQ(FalcataPlan::NonquantOrderedCost(rho, 2048, true), rho);
  EXPECT_DOUBLE_EQ(FalcataPlan::NonquantOrderedCost(rho, 24576, true), 8 * rho);
  EXPECT_LT(FalcataPlan::NonquantOrderedCost(rho, 2048, true), FalcataPlan::NonquantOrderedMaxCost(true));
  EXPECT_GT(FalcataPlan::NonquantOrderedCost(rho, 24576, true), FalcataPlan::NonquantOrderedMaxCost(true));
  EXPECT_DOUBLE_EQ(FalcataPlan::NonquantOrderedCost(rho, 24576, false), rho);
}

TEST(NonquantHistCost, Fp64ThresholdSeparatesDeepShapes) {
  const double year = FalcataPlan::NonquantHistRho(463715, 90, 23027, 1023, 10);
  const double fraud = FalcataPlan::NonquantHistRho(227845, 29, 7423, 1023, 10);
  EXPECT_LT(FalcataPlan::NonquantOrderedCost(year, 23027, false), FalcataPlan::NonquantOrderedMaxCost(false));
  EXPECT_GT(FalcataPlan::NonquantOrderedCost(fraud, 7423, false), FalcataPlan::NonquantOrderedMaxCost(false));
}
