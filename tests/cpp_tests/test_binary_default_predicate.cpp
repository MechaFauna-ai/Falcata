/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <vector>

#include "../../src/objective/binary_objective.hpp"

namespace {
class ReplacedBinaryPredicate : public Falcata::BinaryLogloss {
 public:
  explicit ReplacedBinaryPredicate(const Falcata::Config& config) : BinaryLogloss(config) {
    is_pos_ = [](Falcata::label_t label) { return label == 2; };
  }
};
}  // namespace

TEST(BinaryObjective, DefaultAndCustomPredicatesAgree) {
  Falcata::Config config;
  config.sigmoid = 1.5;
  config.scale_pos_weight = 2.5;
  const Falcata::label_t labels[] = {-3, 0, 1, 2};
  const Falcata::label_t weights[] = {1, 2, 3, 4};
  const double scores[] = {-2, -0.5, 0.5, 2};
  for (bool weighted : {false, true}) {
    for (bool one_vs_all : {false, true}) {
      Falcata::Metadata metadata, remapped;
      metadata.Init(4, -1, -1);
      remapped.Init(4, -1, -1);
      metadata.SetLabel(labels, 4);
      std::vector<Falcata::label_t> mapped;
      for (auto label : labels) mapped.push_back(one_vs_all ? label == 2 : label > 0);
      remapped.SetLabel(mapped.data(), 4);
      if (weighted) {
        metadata.SetWeights(weights, 4);
        remapped.SetWeights(weights, 4);
      }
      Falcata::BinaryLogloss direct(config);
      Falcata::BinaryLogloss custom(config, [one_vs_all](Falcata::label_t label) {
        return one_vs_all ? label == 2 : label > 0;
      });
      direct.Init(remapped, 4);
      custom.Init(metadata, 4);
      std::vector<Falcata::score_t> expected_g(4), expected_h(4), actual_g(4), actual_h(4);
      direct.GetGradients(scores, expected_g.data(), expected_h.data());
      custom.GetGradients(scores, actual_g.data(), actual_h.data());
      EXPECT_EQ(expected_g, actual_g);
      EXPECT_EQ(expected_h, actual_h);
      if (one_vs_all) {
        ReplacedBinaryPredicate replaced(config);
        replaced.Init(metadata, 4);
        replaced.GetGradients(scores, actual_g.data(), actual_h.data());
        EXPECT_EQ(expected_g, actual_g);
        EXPECT_EQ(expected_h, actual_h);
      }
    }
  }
}
