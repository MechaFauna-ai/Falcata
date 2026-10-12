/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <Falcata/meta.h>

#include <limits>

#include "../../src/treelearner/split_info.hpp"

using Falcata::LightSplitInfo;
using Falcata::SplitInfo;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

template <typename T>
T MakeSplit(int feature, double gain) {
  T s;
  s.feature = feature;
  s.gain = gain;
  return s;
}

template <typename T>
void ExpectNaNGainRanksAsMinusInfinity() {
  const T finite = MakeSplit<T>(3, 1.0);
  const T nan_gain = MakeSplit<T>(1, kNaN);
  const T min_gain = MakeSplit<T>(1, Falcata::kMinScore);

  EXPECT_TRUE(finite > nan_gain);
  EXPECT_FALSE(nan_gain > finite);
  EXPECT_TRUE(nan_gain == min_gain);
  EXPECT_TRUE(min_gain == nan_gain);
  EXPECT_FALSE(nan_gain == finite);
  // Equal (-inf) gains fall through to the smaller-feature-index tie break.
  EXPECT_TRUE(nan_gain > MakeSplit<T>(2, kNaN));
  EXPECT_FALSE(MakeSplit<T>(2, kNaN) > nan_gain);
}

}  // namespace

TEST(SplitInfo, NaNGainRanksAsMinusInfinity) {
  ExpectNaNGainRanksAsMinusInfinity<SplitInfo>();
}

TEST(LightSplitInfo, NaNGainRanksAsMinusInfinity) {
  ExpectNaNGainRanksAsMinusInfinity<LightSplitInfo>();
}
