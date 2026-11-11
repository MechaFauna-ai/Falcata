/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "../../src/objective/cuda/cuda_regression_objective.hpp"

using Falcata::data_size_t;

// The multi-target L2 gradient kernel addresses element (row, k) of the
// column-major [num_target][num_data] buffers. On origin/master the launcher
// formed num_data * num_target in a 32-bit data_size_t, which wraps once a
// dataset has more than INT32_MAX label cells (e.g. 300M rows x 8 targets).
TEST(CUDAMultiTarget, ElementIndexAndRowBlocksPastInt32) {
  const data_size_t num_data = 300000000;
  const int num_target = 8;
  const int64_t cells = static_cast<int64_t>(num_data) * num_target;
  ASSERT_GT(cells, std::numeric_limits<int32_t>::max());
  // the last element of the last target is the last cell
  EXPECT_EQ(Falcata::MultiTargetElementIndex(num_target - 1, num_data, num_data - 1), static_cast<size_t>(cells - 1));
  EXPECT_EQ(Falcata::MultiTargetElementIndex(num_target - 1, num_data, 0),
            static_cast<size_t>(cells - num_data));
  EXPECT_EQ(Falcata::MultiTargetElementIndex(0, num_data, 12345), 12345u);
  // one thread per row: the grid covers num_data rows, however many targets there are
  constexpr data_size_t kMax = std::numeric_limits<data_size_t>::max();
  EXPECT_EQ(Falcata::MultiTargetRowBlocks(num_data, 1024), 292969);
  EXPECT_EQ(Falcata::MultiTargetRowBlocks(kMax, 1024), 2097152);
  EXPECT_EQ(Falcata::MultiTargetRowBlocks(1, 1024), 1);
}
