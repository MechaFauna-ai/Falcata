/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
// Generated-header-first entry order, linked with policy-first test_common.cpp.
#include <gtest/gtest.h>
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-constant-out-of-range-compare"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif
#include <Falcata/cuda_controller_learned.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
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

TEST(CUDAControllerLearned, ConstrainedModelDepthTwoBoundaryGolden) {
  // Model 11fe1d31: artifact-specific goldens, independent of the generic
  // wrapper/native-capability matrix in test_common.cpp. These assert the raw
  // candidate only; they do not establish native eligibility or engagement.
  //   num_leaves <= 1536 ? (num_features <= 1472 ? WHOLE : HOST)
  //                      : (num_bins <= 1824 ? PREFIX : HOST)
  using Route = Falcata::CUDAControllerRoute;
  auto facts = NumericFacts();
  auto set_features = [&facts](int features) { facts.num_features = facts.sampled_features = features; };
  // Left subtree: the leaf budget threshold is inclusive, then the feature-count split.
  for (const int leaves : {1535, 1536}) {
    facts.num_leaves = leaves;
    for (const int features : {1471, 1472}) {
      set_features(features);
      EXPECT_EQ(Falcata::LearnedCUDAControllerCandidate(facts), Route::kWhole) << leaves << " " << features;
    }
    set_features(1473);
    EXPECT_EQ(Falcata::LearnedCUDAControllerCandidate(facts), Route::kHost) << leaves;
  }
  // num_bins is not consulted on the left.
  facts = NumericFacts();
  facts.num_leaves = 1536;
  facts.num_bins = 1825;
  EXPECT_EQ(Falcata::LearnedCUDAControllerCandidate(facts), Route::kWhole);
  // Right subtree: the histogram-bin split, threshold inclusive.
  facts = NumericFacts();
  facts.num_leaves = 1537;
  for (const int64_t bins : {int64_t{1823}, int64_t{1824}}) {
    facts.num_bins = bins;
    EXPECT_EQ(Falcata::LearnedCUDAControllerCandidate(facts), Route::kPrefix) << bins;
  }
  facts.num_bins = 1825;
  EXPECT_EQ(Falcata::LearnedCUDAControllerCandidate(facts), Route::kHost);
  // num_features is not consulted on the right.
  facts.num_bins = 1824;
  set_features(1473);
  EXPECT_EQ(Falcata::LearnedCUDAControllerCandidate(facts), Route::kPrefix);
}

TEST(CUDAControllerLearned, ConstrainedModelLeavesResolveThroughNativeGates) {
  // One case per model leaf through the shipped wrapper on qualified facts:
  // a device route needs its own native eligibility, otherwise HOST.
  using Route = Falcata::CUDAControllerRoute;
  auto qualified = []() {
    auto facts = NumericFacts();
    facts.quantized = facts.scope_supported = facts.performance_modifiers_supported = true;
    facts.native_prefix_eligible = facts.native_whole_eligible = true;
    facts.device_major = 12;
    facts.device_minor = 0;
    facts.device_sm_count = 170;
    facts.device_memory_bus_bits = 512;
    return facts;
  };
  struct Case {
    int leaves;
    int features;
    int64_t bins;
    Route resolved;
    const char* reason;
    Route unavailable_resolved;
    const char* unavailable_reason;
  };
  const Case cases[] = {
      {1536, 1472, 1824 * 2, Route::kWhole, "learned_whole", Route::kHost, "learned_whole_native_unavailable"},
      {1536, 1473, 1824, Route::kHost, "learned_host", Route::kHost, "learned_host"},
      {1537, 1473, 1824, Route::kPrefix, "learned_prefix", Route::kHost, "learned_prefix_native_unavailable"},
      {1537, 104, 1825, Route::kHost, "learned_host", Route::kHost, "learned_host"},
  };
  for (const auto& c : cases) {
    auto facts = qualified();
    facts.num_leaves = c.leaves;
    facts.num_features = facts.sampled_features = c.features;
    facts.num_bins = c.bins;
    auto decision = Falcata::ResolveCUDAController({}, facts);
    EXPECT_EQ(decision.requested, Route::kAuto);
    EXPECT_EQ(decision.resolved, c.resolved) << c.leaves << " " << c.features << " " << c.bins;
    EXPECT_STREQ(decision.reason, c.reason);
    facts.native_prefix_eligible = facts.native_whole_eligible = false;
    decision = Falcata::ResolveCUDAController({}, facts);
    EXPECT_EQ(decision.resolved, c.unavailable_resolved) << c.leaves << " " << c.features << " " << c.bins;
    EXPECT_STREQ(decision.reason, c.unavailable_reason);
  }
}

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
