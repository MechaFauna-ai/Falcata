/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/prediction_early_stop.h>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

TEST(MulticlassEarlyStop, LinearScanMatchesSortedTopTwo) {
  std::mt19937 rng(7);
  for (double margin : {0.0, 0.2, 1.0}) {
    const auto stop = Falcata::CreatePredictionEarlyStopInstance("multiclass", {10, margin});
    for (int classes : {2, 3, 10, 100}) {
      for (int row = 0; row < 1000; ++row) {
        std::vector<double> predictions(classes);
        for (auto& value : predictions) value = static_cast<int>(rng() % 31) * 0.1 - 1.5;
        auto sorted = predictions;
        std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<double>());
        EXPECT_EQ(stop.callback_function(predictions.data(), classes), sorted[0] - sorted[1] > margin);
      }
    }
  }
}
