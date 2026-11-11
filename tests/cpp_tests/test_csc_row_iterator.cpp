/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/c_api.h>

#include <cstdint>
#include <vector>

#include "../../src/c_api_csc_row_iterator.hpp"

// Get() only moved forward: asking for a row before the current position
// returned 0.0 even for a stored value.
TEST(CSCRowIterator, GetAnswersRowsInAnyOrder) {
  // one column, rows 0..9, stored values at rows 2, 5 and 7
  const std::vector<int32_t> col_ptr = {0, 3};
  const std::vector<int32_t> indices = {2, 5, 7};
  const std::vector<double> data = {20.0, 50.0, 70.0};
  CSC_RowIterator it(col_ptr.data(), C_API_DTYPE_INT32, indices.data(), data.data(), C_API_DTYPE_FLOAT64, 2, 3, 0);
  const std::vector<double> dense = {0, 0, 20, 0, 0, 50, 0, 70, 0, 0};
  for (int row = 0; row < 10; ++row) {
    EXPECT_EQ(it.Get(row), dense[row]) << "forward " << row;
  }
  for (int row : {9, 5, 2, 7, 0, 6, 5, 3, 7, 2}) {
    EXPECT_EQ(it.Get(row), dense[row]) << "any order " << row;
  }
}
