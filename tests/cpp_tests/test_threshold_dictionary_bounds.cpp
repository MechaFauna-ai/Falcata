/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "../../src/boosting/falb.h"

TEST(ThresholdDictionary, FeatureIndexBounds) {
  using Falcata::FALB::ThresholdFeatureIndex;
  EXPECT_THROW(ThresholdFeatureIndex(-1, 2), std::runtime_error);
  EXPECT_THROW(ThresholdFeatureIndex(2, 2), std::runtime_error);
  EXPECT_THROW(ThresholdFeatureIndex(0, 0), std::runtime_error);
  EXPECT_THROW(ThresholdFeatureIndex(std::numeric_limits<int>::max(), 2), std::runtime_error);
  EXPECT_EQ(ThresholdFeatureIndex(0, 2), 0);
  EXPECT_EQ(ThresholdFeatureIndex(1, 2), 1);
}
