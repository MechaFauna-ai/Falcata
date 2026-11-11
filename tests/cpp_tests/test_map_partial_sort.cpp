/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "../../src/metric/map_metric.hpp"

TEST(MapMetric, TopKMatchesStableFullSortWithTies) {
  Falcata::Config config;
  Falcata::MapMetric metric(config);
  std::mt19937 rng(7);
  for (int n : {1, 2, 10, 100, 10000}) {
    std::vector<Falcata::label_t> labels(n);
    std::vector<double> scores(n);
    int positives = 0;
    for (int i = 0; i < n; ++i) {
      labels[i] = rng() % 2;
      positives += labels[i];
      scores[i] = rng() % 8;
    }
    for (const std::vector<int>& ks : {std::vector<int>{1, 3, 5}, std::vector<int>{1, 3, 5, 10001}}) {
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&scores](int a, int b) { return scores[a] > scores[b]; });
    std::vector<double> expected;
    for (int k : ks) {
      double sum = 0;
      int hits = 0;
      for (int i = 0; i < std::min(k, n); ++i) {
        if (labels[order[i]] > 0.5) sum += static_cast<double>(++hits) / (i + 1);
      }
      expected.push_back(positives ? sum / std::min(positives, std::min(k, n)) : 1.0);
    }
    std::vector<double> actual(ks.size());
    metric.CalMapAtK(ks, positives, labels.data(), scores.data(), n, &actual);
    EXPECT_EQ(actual, expected);
    const std::vector<Falcata::label_t> zero_labels(n, 0);
    metric.CalMapAtK(ks, 0, zero_labels.data(), scores.data(), n, &actual);
    EXPECT_EQ(actual, std::vector<double>(ks.size(), 1.0));
    }
  }
}
