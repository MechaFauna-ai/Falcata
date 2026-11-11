/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <Falcata/meta.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

#include "../../src/io/init_score_layout.hpp"

using Falcata::data_size_t;
using Falcata::InitScoreOffset;
using Falcata::InitScoreSize;

// The loader (initial-model scores) and Metadata::InsertInitScores address class k's rows of the class-major
// init-score buffer through InitScoreOffset. Past INT32_MAX class-rows (e.g. 30M rows x 100 classes) the old int
// products wrapped and wrote outside the buffer.
TEST(InitScoreLayout, OffsetsDoNotWrap) {
  const data_size_t num_rows = 30000000;
  EXPECT_EQ(InitScoreOffset(0, num_rows, 5), 5u);
  EXPECT_EQ(InitScoreOffset(71, num_rows), 2130000000u);
  EXPECT_EQ(InitScoreOffset(72, num_rows), 2160000000u);
  EXPECT_EQ(InitScoreOffset(99, num_rows, num_rows - 1), 2999999999u);
}

// A buffer whose length SetInitScore's data_size_t cannot describe is refused instead of being passed a wrapped
// length; anything that fits is sized exactly.
TEST(InitScoreLayout, SizeMustFitDataSizeT) {
  const data_size_t max = std::numeric_limits<data_size_t>::max();
  EXPECT_EQ(InitScoreSize(1000, 3), 3000u);
  EXPECT_EQ(InitScoreSize(max, 1), static_cast<size_t>(max));
  EXPECT_THROW(InitScoreSize(30000000, 100), std::runtime_error);
  EXPECT_THROW(InitScoreSize(max / 2 + 1, 2), std::runtime_error);
}
