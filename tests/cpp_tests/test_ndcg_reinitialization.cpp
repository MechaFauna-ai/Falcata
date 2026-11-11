/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include "../../src/metric/rank_metric.hpp"

TEST(NDCGMetric, ReinitializationPreservesNamesAndResults) {
  Falcata::Metadata metadata;
  metadata.Init(4, -1, -1);
  const Falcata::label_t labels[] = {0, 1, 2, 1};
  const Falcata::data_size_t groups[] = {2, 2};
  metadata.SetLabel(labels, 4);
  metadata.SetQuery(groups, 2);
  Falcata::Config config;
  config.eval_at = {1, 2};
  Falcata::NDCGMetric metric(config);
  metric.Init(metadata, 4);
  const double scores[] = {0, 1, 2, 1};
  const auto names = metric.GetName();
  const auto values = metric.Eval(scores, nullptr);
  metric.Init(metadata, 4);
  EXPECT_EQ(metric.GetName(), names);
  EXPECT_EQ(metric.GetName().size(), metric.Eval(scores, nullptr).size());
  EXPECT_EQ(metric.Eval(scores, nullptr), values);
}
