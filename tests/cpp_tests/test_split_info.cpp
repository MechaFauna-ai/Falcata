/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <Falcata/meta.h>

#include <limits>
#include <vector>

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

// CopyTo / CopyFrom serialise a split for the parallel learners. A numerical split has an empty cat_threshold, whose
// data() is null: the zero-length memcpy from (and, after resize(0), into) it was a null-pointer argument
// (UBSan nonnull-attribute in test_voting_parallel_reset_top_k.py).
TEST(SplitInfo, CopyRoundTripsNumericalAndCategoricalSplits) {
  for (const std::vector<uint32_t>& cats : {std::vector<uint32_t>{}, std::vector<uint32_t>{1, 4, 9}}) {
    SplitInfo split = MakeSplit<SplitInfo>(5, 2.5);
    split.threshold = 7;
    split.num_cat_threshold = static_cast<int>(cats.size());
    split.cat_threshold = cats;
    std::vector<char> buffer(SplitInfo::Size(static_cast<int>(cats.size())));
    split.CopyTo(buffer.data());
    SplitInfo copy;
    copy.cat_threshold = {3};  // a stale category list from an earlier split is replaced
    copy.CopyFrom(buffer.data());
    EXPECT_EQ(copy.feature, 5);
    EXPECT_EQ(copy.gain, 2.5);
    EXPECT_EQ(copy.threshold, 7u);
    EXPECT_EQ(copy.num_cat_threshold, static_cast<int>(cats.size()));
    EXPECT_EQ(copy.cat_threshold, cats);
  }
}
