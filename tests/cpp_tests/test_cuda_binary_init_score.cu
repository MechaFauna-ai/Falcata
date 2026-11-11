/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/config.h>
#include <Falcata/dataset.h>

#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../src/objective/binary_objective.hpp"
#include "../../src/objective/cuda/cuda_binary_objective.hpp"

// CUDABinaryLogloss::BoostFromScore (LaunchCalcInitScoreKernel) used to atomicAdd the weighted label and weight sums
// into device accumulators that only the label one was zeroed before, so a second weighted call (another class of a
// one-vs-all model, or a second Init score request) started from the first call's pavg. The sums are now written, not
// accumulated: every call returns the same init score, and it matches the CPU objective's.
TEST(CUDABinaryLogloss, WeightedInitScoreIsTheSameOnEveryCall) {
  const int num_data = 5000;  // several init-score blocks
  Falcata::Config config;
  std::unordered_map<std::string, std::string> params = {{"objective", "binary"}};
  config.Set(params);
  Falcata::Metadata metadata;
  metadata.Init(num_data, -1, -1);
  std::vector<Falcata::label_t> labels(num_data);
  std::vector<Falcata::label_t> weights(num_data);
  for (int i = 0; i < num_data; ++i) {
    labels[i] = static_cast<Falcata::label_t>((i % 7) < 2 ? 1 : 0);
    weights[i] = 0.25f + static_cast<Falcata::label_t>(i % 5);
  }
  metadata.SetLabel(labels.data(), num_data);
  metadata.SetWeights(weights.data(), num_data);
  metadata.CreateCUDAMetadata(0);

  Falcata::BinaryLogloss cpu_objective(config);
  cpu_objective.Init(metadata, num_data);
  const double expected = cpu_objective.BoostFromScore(0);

  Falcata::CUDABinaryLogloss objective(config);
  objective.Init(metadata, num_data);
  const double first = objective.BoostFromScore(0);
  EXPECT_NEAR(first, expected, 1e-9);
  for (int call = 1; call < 4; ++call) {
    EXPECT_EQ(objective.BoostFromScore(call % 2), first) << "call " << call;
  }
}
