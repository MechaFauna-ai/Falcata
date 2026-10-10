/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "../../src/treelearner/data_partition.hpp"
#include "../../src/treelearner/leaf_splits.hpp"

using Falcata::data_size_t;
using Falcata::DataPartition;
using Falcata::LeafSplits;

// The leaf-partitioned discretized Init must sum the packed (grad, hess) words
// of the rows on the leaf, not of the first num_data_in_leaf rows.
TEST(LeafSplits, DiscretizedInitOnPartitionSumsTheLeafRows) {
  const data_size_t kNumData = 16;
  // Interleaved int8 pairs per row: [2 * row] = hessian, [2 * row + 1] = gradient.
  std::vector<int8_t> grad_hess(2 * kNumData);
  for (data_size_t row = 0; row < kNumData; ++row) {
    grad_hess[2 * row] = static_cast<int8_t>(row + 1);
    grad_hess[2 * row + 1] = static_cast<int8_t>(3 * row - 20);
  }

  // Bagging-style root: only the odd rows sit on leaf 0.
  std::vector<int> leaf_pred(kNumData);
  int64_t expected_packed_sum = 0;
  double expected_grad_sum = 0.0;
  double expected_hess_sum = 0.0;
  for (data_size_t row = 0; row < kNumData; ++row) {
    leaf_pred[row] = (row % 2 == 1) ? 0 : 1;
    if (leaf_pred[row] == 0) {
      const int8_t hess = grad_hess[2 * row];
      const int8_t grad = grad_hess[2 * row + 1];
      expected_packed_sum += (static_cast<int64_t>(grad) << 32) | static_cast<int64_t>(static_cast<uint8_t>(hess));
      expected_grad_sum += grad;
      expected_hess_sum += hess;
    }
  }
  DataPartition partition(kNumData, 2);
  partition.ResetByLeafPred(leaf_pred, 2);

  LeafSplits leaf_splits(kNumData, nullptr);
  leaf_splits.Init(0, &partition, grad_hess.data(), 1.0f, 1.0f);

  EXPECT_EQ(leaf_splits.num_data_in_leaf(), kNumData / 2);
  EXPECT_DOUBLE_EQ(leaf_splits.sum_gradients(), expected_grad_sum);
  EXPECT_DOUBLE_EQ(leaf_splits.sum_hessians(), expected_hess_sum);
  EXPECT_EQ(leaf_splits.int_sum_gradients_and_hessians(), expected_packed_sum);
}
