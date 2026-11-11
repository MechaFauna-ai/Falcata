/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include "../../src/treelearner/feature_histogram.hpp"

TEST(HistogramPool, MoveOntoCachedDestinationSurvivesEviction) {
  Falcata::Dataset data(1);
  Falcata::Config config;
  Falcata::HistogramPool pool;
  pool.DynamicChangeSize(&data, 0, {}, &config, 2, 4);
  Falcata::FeatureHistogram* out;
  EXPECT_FALSE(pool.Get(0, &out));
  auto* source = out;
  EXPECT_FALSE(pool.Get(1, &out));
  pool.Move(0, 1);
  EXPECT_FALSE(pool.Get(2, &out));
  EXPECT_TRUE(pool.Get(1, &out));
  EXPECT_EQ(out, source);
  pool.Move(1, 1);
  EXPECT_TRUE(pool.Get(1, &out));
  EXPECT_EQ(out, source);
  pool.Move(3, 1);
  EXPECT_TRUE(pool.Get(1, &out));
  EXPECT_EQ(out, source);
}
