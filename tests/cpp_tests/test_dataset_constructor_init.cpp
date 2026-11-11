/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/dataset.h>

#include <cstddef>
#include <new>

namespace {

// Builds a Dataset in storage filled with a non-zero pattern, so a scalar member that a constructor leaves
// uninitialised reads back as the pattern instead of whatever the stack or heap happened to hold.
template <typename... Args>
void ExpectEmptyCounts(Args... args) {
  alignas(Falcata::Dataset) unsigned char storage[sizeof(Falcata::Dataset)];
  // volatile: a plain memset before the constructor is a dead store to GCC (-flifetime-dse) and may be dropped
  volatile unsigned char* fill = storage;
  for (size_t i = 0; i < sizeof(storage); ++i) {
    fill[i] = 0xAB;
  }
  auto* dataset = new (storage) Falcata::Dataset(args...);
  EXPECT_EQ(dataset->num_features(), 0);
  EXPECT_EQ(dataset->num_total_features(), 0);
  EXPECT_EQ(dataset->num_feature_groups(), 0);
  dataset->~Dataset();
}

}  // namespace

// Dataset() and Dataset(num_data) left num_features_, num_total_features_, num_groups_ and other scalar members
// uninitialised. A Dataset built that way and handed to code that sizes vectors by num_features() (HistogramPool's
// SetFeatureInfo) read garbage: on MSVC, "vector too long" in HistogramPool.MoveOntoCachedDestinationSurvivesEviction.
TEST(Dataset, ConstructorsStartWithNoFeatures) {
  ExpectEmptyCounts();
  ExpectEmptyCounts(Falcata::data_size_t{1});
}
