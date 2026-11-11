/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "../../src/treelearner/quantized_index.hpp"

TEST(QuantizedIndex, LargeOffsetsDoNotOverflow) {
  const auto max_rows = std::numeric_limits<Falcata::data_size_t>::max();
  EXPECT_EQ(Falcata::QuantizedPairOffset(max_rows), uint64_t{4294967294});
  EXPECT_EQ(Falcata::QuantizedPairOffset(max_rows - 1) + 1, uint64_t{4294967293});
  EXPECT_EQ(Falcata::QuantizedRandomIndex(max_rows - 1, 3, max_rows), 2);
  EXPECT_EQ(Falcata::QuantizedRandomIndex(max_rows - 1, max_rows, max_rows), max_rows - 1);
}

TEST(QuantizedIndex, SmallOffsetsMatchOriginalArithmetic) {
  for (int n = 1; n < 64; ++n) {
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(Falcata::QuantizedPairOffset(i), static_cast<size_t>(2 * i));
      for (int start = 0; start <= n; ++start) {
        EXPECT_EQ(Falcata::QuantizedRandomIndex(i, start, n), (i + start) % n);
      }
    }
  }
}
