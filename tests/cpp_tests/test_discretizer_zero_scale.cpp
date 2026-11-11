/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include "../../src/treelearner/gradient_discretizer.hpp"

TEST(GradientDiscretizer, ZeroScalesProduceZeroQuantizedValues) {
  const Falcata::score_t zeros[] = {0, 0};
  const Falcata::score_t nonzero[] = {-1, 1};
  for (bool constant : {false, true}) {
    for (bool stochastic : {false, true}) {
      for (bool zero_gradient : {false, true}) {
        Falcata::GradientDiscretizer discretizer(4, 1, 3, constant, stochastic);
        discretizer.Init(2, 2, 0, nullptr);
        discretizer.DiscretizeGradients(2, zero_gradient ? zeros : nonzero, zeros);
        const auto* values = discretizer.discretized_gradients_and_hessians();
        for (int row = 0; row < 2; ++row) {
          EXPECT_EQ(values[2 * row + 1], zero_gradient ? 0 : (row == 0 ? -2 : 2));
          EXPECT_EQ(values[2 * row], constant ? 1 : 0);
        }
      }
    }
  }
}
