/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/c_api.h>

#include "../../src/treelearner/linear_tree_learner.h"

namespace {
class PartialLinearLearner : public Falcata::LinearTreeLearner<Falcata::SerialTreeLearner> {
 public:
  PartialLinearLearner(const Falcata::Config* config, const Falcata::Dataset* data) : LinearTreeLearner(config) {
    train_data_ = data;
    num_data_ = data->num_data();
    leaf_map_ = {0, 1, 0, 1};
  }
};
}  // namespace

TEST(LinearPrediction, IgnoreUnusedCoefficientsPerLeaf) {
  const double values[] = {0, 1, 2, 3};
  DatasetHandle handle;
  ASSERT_EQ(FLC_DatasetCreateFromMat(values, C_API_DTYPE_FLOAT64, 4, 1, 1,
      "linear_tree=true min_data_in_bin=1 min_data_in_leaf=1 feature_pre_filter=false verbosity=-1",
      nullptr, &handle), 0);
  Falcata::Config config;
  PartialLinearLearner learner(&config, static_cast<const Falcata::Dataset*>(handle));
  Falcata::Tree tree(2, false, true);
  tree.Split(0, 0, 0, 0, 0.5, -1, 1, 2, 2, 1, 1, 1, Falcata::MissingType::None, true);
  tree.SetLeafCoeffs(0, {99});
  tree.SetLeafFeaturesInner(1, {0});
  tree.SetLeafCoeffs(1, {2, 42});
  tree.SetLeafConst(0, 1);
  tree.SetLeafConst(1, 1);
  double scores[4] = {};
  learner.AddPredictionToScoreInner<false>(&tree, scores);
  EXPECT_EQ(scores[0], 1);
  EXPECT_EQ(scores[1], 3);
  EXPECT_EQ(scores[2], 1);
  EXPECT_EQ(scores[3], 7);
  EXPECT_EQ(FLC_DatasetFree(handle), 0);
}
