/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <stdexcept>

#include "../../src/treelearner/data_partition.hpp"

TEST(DataPartition, RejectsInvalidLeafPredictionsBeforeChangingState) {
  Falcata::DataPartition partition(2, 2);
  partition.ResetByLeafPred({0, 1}, 2);
  EXPECT_THROW(partition.ResetByLeafPred({0, 1, 0}, 2), std::runtime_error);
  EXPECT_THROW(partition.ResetByLeafPred({0}, 2), std::runtime_error);
  EXPECT_THROW(partition.ResetByLeafPred({0, -1}, 2), std::runtime_error);
  EXPECT_THROW(partition.ResetByLeafPred({0, 2}, 2), std::runtime_error);
  EXPECT_THROW(partition.ResetByLeafPred({0, 1}, 0), std::runtime_error);
  Falcata::data_size_t count;
  const auto* rows = partition.GetIndexOnLeaf(1, &count);
  ASSERT_EQ(count, 1);
  EXPECT_EQ(rows[0], 1);
  partition.ResetByLeafPred({1, 0}, 2);
  rows = partition.GetIndexOnLeaf(1, &count);
  ASSERT_EQ(count, 1);
  EXPECT_EQ(rows[0], 0);
}
