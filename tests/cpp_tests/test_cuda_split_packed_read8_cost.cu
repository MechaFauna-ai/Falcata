/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include "../../src/treelearner/cuda/cuda_single_gpu_tree_learner.hpp"

namespace {

using Falcata::CUDASingleGPUTreeLearner;
constexpr size_t kL2 = static_cast<size_t>(96) << 20;  // an RTX 5090's L2

// The shapes the rule was measured on (2026-11-18, RTX 5090): where the strided 8-bit split read lost to the gather,
// and where #613 measured it winning.
TEST(SplitPackedRead8Cost, DeepTreesOnNarrowRowsGatherOutOfCache) {
  // higgs, 10.5M rows x 28 columns (294 MB), feature_fraction 0.9, depth 10: strided 280 bytes per row, gather 63
  EXPECT_FALSE(CUDASingleGPUTreeLearner::SplitPackedRead8Pays(10'500'000, 28, 25, 10, 1023, kL2));
}

TEST(SplitPackedRead8Cost, RowsThatFitInL2ReadStrided) {
  // the same shape at 2M rows (56 MB): the strided levels after the first read the cache
  EXPECT_TRUE(CUDASingleGPUTreeLearner::SplitPackedRead8Pays(2'000'000, 28, 25, 10, 1023, kL2));
}

TEST(SplitPackedRead8Cost, WideSamplesReadStrided) {
  // #613's widest fit: 727k x 2,568 columns, 1,525 sampled, 6 leaves; Numerai: 5.46M rows, compact matrix 355 wide
  EXPECT_TRUE(CUDASingleGPUTreeLearner::SplitPackedRead8Pays(727'000, 2568, 1525, -1, 6, kL2));
  EXPECT_TRUE(CUDASingleGPUTreeLearner::SplitPackedRead8Pays(5'460'000, 355, 355, 5, 32, kL2));
}

TEST(SplitPackedRead8Cost, UnknownCacheSizeStillDecidesByBytes) {
  EXPECT_FALSE(CUDASingleGPUTreeLearner::SplitPackedRead8Pays(2'000'000, 28, 25, 10, 1023, 0));
  EXPECT_TRUE(CUDASingleGPUTreeLearner::SplitPackedRead8Pays(727'000, 2568, 1525, -1, 6, 0));
}

}  // namespace
