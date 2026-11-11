// Copyright (c) 2026 The Falcata developers. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root.
#include <Falcata/c_api.h>
#include <cassert>
#include <limits>

#include "src/treelearner/linear_tree_learner.h"
int main() {
  double data[80];
  for (int i = 0; i < 40; ++i) {
    data[2 * i] = 0.0;
    data[2 * i + 1] = i;
  }
  data[1] = std::numeric_limits<double>::quiet_NaN();
  DatasetHandle handle = nullptr;
  assert(FLC_DatasetCreateFromMat(data, C_API_DTYPE_FLOAT64, 40, 2, 1,
      "linear_tree=true min_data_in_leaf=1 num_threads=1", nullptr, &handle) == 0);
  auto* dataset = static_cast<Falcata::Dataset*>(handle);
  assert(dataset->InnerFeatureIndex(0) == -1);
  Falcata::Config config;
  config.linear_tree = true;
  config.num_threads = 1;
  config.min_data_in_leaf = 1;
  {
    Falcata::LinearTreeLearner<Falcata::SerialTreeLearner> learner(&config);
    learner.Init(dataset, false);
    Falcata::Tree tree(2, false, true);
    tree.Split(0, 0, 0, 0, 0.5, 3.0, 7.0, 1, 1, 1.0, 1.0, 1.0f, Falcata::MissingType::None, false);
    double scores[40] = {};
    learner.AddPredictionToScore(&tree, scores);
    for (double score : scores) assert(score == 0.0);
  }
  assert(FLC_DatasetFree(handle) == 0);
}
