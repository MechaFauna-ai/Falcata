/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
// Generated-header-first entry order, linked with policy-first test_common.cpp.
#include <gtest/gtest.h>
#include <Falcata/cuda_controller_learned.h>
#include <Falcata/cuda_controller_policy.h>

#include <cstdint>
#include <limits>

namespace {
Falcata::CUDAControllerFacts NumericFacts() {
  Falcata::CUDAControllerFacts facts;
  facts.metadata_known = true;
  facts.num_rows = 200000;
  facts.num_features = facts.sampled_features = 104;
  facts.num_bins = 104 * 4;
  facts.max_feature_bins = 4;
  facts.num_leaves = 1024;
  facts.max_depth = 10;
  facts.min_data_in_leaf = 20;
  return facts;
}
}  // namespace

TEST(CUDAControllerLearned, GeneratedFirstInclusionAndUnknownMetadata) {
  auto facts = NumericFacts();
  facts.metadata_known = false;
  EXPECT_EQ(Falcata::LearnedCUDAControllerCandidate(facts), Falcata::CUDAControllerRoute::kHost);
  EXPECT_EQ(Falcata::ResolveCUDAController({}, facts).resolved, Falcata::CUDAControllerRoute::kHost);
}

TEST(CUDAControllerLearned, Int53GuardsApplyBeforeDoubleConversion) {
  // An active baseline makes guard removal observable, even for inputs the
  // installed tree does not split on.
  ASSERT_NE(Falcata::LearnedCUDAControllerCandidate(NumericFacts()), Falcata::CUDAControllerRoute::kHost);
  const int64_t bound = int64_t{1} << 53;
  for (const int64_t value : {-bound - 1, bound, std::numeric_limits<int64_t>::min(),
                              std::numeric_limits<int64_t>::max()}) {
    auto facts = NumericFacts();
    facts.num_rows = value;
    EXPECT_EQ(Falcata::LearnedCUDAControllerCandidate(facts), Falcata::CUDAControllerRoute::kHost);
    facts = NumericFacts();
    facts.num_bins = value;
    EXPECT_EQ(Falcata::LearnedCUDAControllerCandidate(facts), Falcata::CUDAControllerRoute::kHost);
  }
}

TEST(CUDAControllerLearned, EveryFloatingInputIsGuardedEvenIfUnusedByTree) {
  ASSERT_NE(Falcata::LearnedCUDAControllerCandidate(NumericFacts()), Falcata::CUDAControllerRoute::kHost);
  for (const double value : {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity()}) {
    auto facts = NumericFacts();
    facts.feature_fraction = value;
    EXPECT_EQ(Falcata::LearnedCUDAControllerCandidate(facts), Falcata::CUDAControllerRoute::kHost);
    facts = NumericFacts();
    facts.bagging_fraction = value;
    EXPECT_EQ(Falcata::LearnedCUDAControllerCandidate(facts), Falcata::CUDAControllerRoute::kHost);
  }
}
