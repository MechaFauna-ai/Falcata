/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/objective/binary_objective.hpp"

TEST(BinaryObjective, LoadedObjectiveCanInitializeForTraining) {
  Falcata::Config config;
  config.sigmoid = 1.5;
  const Falcata::label_t labels[] = {0, 1, 1, 0};
  const Falcata::label_t weights[] = {1, 2, 3, 4};
  const double scores[] = {-2, -0.5, 0.5, 2};
  for (bool weighted : {false, true}) {
    Falcata::Metadata metadata;
    metadata.Init(4, -1, -1);
    metadata.SetLabel(labels, 4);
    if (weighted) metadata.SetWeights(weights, 4);
    Falcata::BinaryLogloss reference(config);
    Falcata::BinaryLogloss loaded(std::vector<std::string>{"binary", "sigmoid:1.5"});
    reference.Init(metadata, 4);
    loaded.Init(metadata, 4);
    EXPECT_EQ(reference.BoostFromScore(0), loaded.BoostFromScore(0));
    EXPECT_EQ(reference.NumPositiveData(), loaded.NumPositiveData());
    std::vector<Falcata::score_t> expected_g(4), expected_h(4), actual_g(4), actual_h(4);
    reference.GetGradients(scores, expected_g.data(), expected_h.data());
    loaded.GetGradients(scores, actual_g.data(), actual_h.data());
    EXPECT_EQ(expected_g, actual_g);
    EXPECT_EQ(expected_h, actual_h);
  }
}
