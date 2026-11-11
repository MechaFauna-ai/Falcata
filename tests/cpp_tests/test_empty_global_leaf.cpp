/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/network.h>

#include <cstring>

#include "../../src/treelearner/serial_tree_learner.h"
#include "../../src/objective/regression_objective.hpp"

namespace {
class EmptyLeafLearner : public Falcata::SerialTreeLearner {
 public:
  explicit EmptyLeafLearner(const Falcata::Config* config) : SerialTreeLearner(config) {
    num_data_ = 2;
    data_partition_.reset(new Falcata::DataPartition(2, 2));
    data_partition_->ResetByLeafPred({0, 0}, 2);
  }
};

// Simulate two workers with identical local partitions, including an empty leaf.
void GatherIdenticalWorkers(char* input, Falcata::comm_size_t size,
                           const Falcata::comm_size_t*, const Falcata::comm_size_t*,
                           int, char* output, Falcata::comm_size_t) {
  std::memcpy(output, input, size);
  std::memcpy(output + size, input, size);
}
}  // namespace

TEST(LeafRenewal, GloballyEmptyLeafStaysFinite) {
  Falcata::Config config;
  EmptyLeafLearner learner(&config);
  Falcata::RegressionL1loss objective(config);
  Falcata::Metadata metadata;
  metadata.Init(2, -1, -1);
  const Falcata::label_t labels[] = {0, 1};
  metadata.SetLabel(labels, 2);
  objective.Init(metadata, 2);
  Falcata::Tree tree(2, false, false);
  tree.Split(0, 0, 0, 1, 0.5, -1, 1, 2, 0, 1, 0, 1, Falcata::MissingType::None, true);
  Falcata::Network::Init(2, 0, nullptr, GatherIdenticalWorkers);
  learner.RenewTreeOutput(&tree, &objective, [](const Falcata::label_t*, int) { return 1.0; },
                          2, nullptr, 0, nullptr);
  Falcata::Network::Dispose();
  EXPECT_EQ(tree.LeafOutput(0), 1.0);
  EXPECT_EQ(tree.LeafOutput(1), 0.0);
}
