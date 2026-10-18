/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "../../src/treelearner/cuda/cuda_device_level_graph_admission.hpp"

using Falcata::CUDADeviceLevelGraphAdmission;

TEST(CUDADeviceLevelGraphAdmission, PromotesOnlyOnThirdObservationThenStartsCold) {
  CUDADeviceLevelGraphAdmission policy;
  const int tag = 0;
  const std::vector<const void*> key{&tag};
  EXPECT_FALSE(policy.Observe(key, 32));
  EXPECT_FALSE(policy.Observe(key, 32));
  EXPECT_TRUE(policy.Observe(key, 32));
  EXPECT_EQ(policy.size(), 0u);
  // Production cache hits bypass Observe. If that executable is evicted, its next miss starts from one again.
  EXPECT_FALSE(policy.Observe(key, 32));
  EXPECT_FALSE(policy.Observe(key, 32));
  EXPECT_TRUE(policy.Observe(key, 32));
}

TEST(CUDADeviceLevelGraphAdmission, RequiresTheCompleteOrderedCaptureKey) {
  CUDADeviceLevelGraphAdmission policy;
  const std::array<int, 3> tags{};
  const std::vector<const void*> a{&tags[0], &tags[1]};
  const std::vector<const void*> changed_tail{&tags[0], &tags[2]};
  const std::vector<const void*> longer{&tags[0], &tags[1], &tags[2]};
  const std::vector<const void*> reversed{&tags[1], &tags[0]};
  for (const auto& key : {a, changed_tail, longer, reversed}) EXPECT_FALSE(policy.Observe(key, 32));
  for (const auto& key : {a, changed_tail, longer, reversed}) EXPECT_FALSE(policy.Observe(key, 32));
  for (const auto& key : {a, changed_tail, longer, reversed}) EXPECT_TRUE(policy.Observe(key, 32));
}

TEST(CUDADeviceLevelGraphAdmission, RepeatedMissRefreshesLRUAndEvictionForgetsOldSightings) {
  CUDADeviceLevelGraphAdmission policy;
  const std::array<int, 3> tags{};
  const std::vector<const void*> a{&tags[0]}, b{&tags[1]}, c{&tags[2]};
  EXPECT_FALSE(policy.Observe(a, 2));
  EXPECT_FALSE(policy.Observe(b, 2));
  EXPECT_FALSE(policy.Observe(a, 2));  // a becomes most recent, with two sightings
  EXPECT_FALSE(policy.Observe(c, 2));  // b is evicted
  EXPECT_EQ(policy.size(), 2u);
  EXPECT_TRUE(policy.Observe(a, 2));
  EXPECT_FALSE(policy.Observe(b, 2));  // old observation did not survive eviction
  EXPECT_FALSE(policy.Observe(b, 2));
  EXPECT_TRUE(policy.Observe(b, 2));
}

TEST(CUDADeviceLevelGraphAdmission, PrefixAndWholeHistoriesAreBoundedUnderUniqueKeyChurn) {
  const std::array<int, 260> tags{};
  for (const size_t capacity : {size_t{32}, size_t{256}}) {
    CUDADeviceLevelGraphAdmission policy;
    for (size_t i = 0; i < tags.size(); ++i) {
      EXPECT_FALSE(policy.Observe({&tags[i]}, capacity));
      EXPECT_LE(policy.size(), capacity);
    }
    EXPECT_EQ(policy.size(), capacity);
    EXPECT_FALSE(policy.Observe({&tags[0]}, capacity));
    EXPECT_FALSE(policy.Observe({&tags[0]}, capacity));
    EXPECT_TRUE(policy.Observe({&tags[0]}, capacity));
  }
}

TEST(CUDADeviceLevelGraphAdmission, ClearAndCapacityReductionDiscardPriorHistory) {
  CUDADeviceLevelGraphAdmission policy;
  const std::array<int, 3> tags{};
  for (const int& tag : tags) EXPECT_FALSE(policy.Observe({&tag}, 256));
  EXPECT_FALSE(policy.Observe({&tags[2]}, 1));  // trim to newest entry, then refresh it
  EXPECT_EQ(policy.size(), 1u);
  EXPECT_FALSE(policy.Observe({&tags[0]}, 1));  // evicted history starts cold
  policy.Clear();
  EXPECT_EQ(policy.size(), 0u);
  EXPECT_FALSE(policy.Observe({&tags[0]}, 32));
  EXPECT_FALSE(policy.Observe({&tags[0]}, 32));
  EXPECT_TRUE(policy.Observe({&tags[0]}, 32));
  EXPECT_FALSE(policy.Observe({&tags[1]}, 0));
  EXPECT_EQ(policy.size(), 0u);
}
