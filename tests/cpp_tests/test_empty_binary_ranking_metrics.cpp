/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <stdexcept>

#include "../../src/metric/binary_metric.hpp"

TEST(BinaryRankingMetrics, RejectEmptyInputBeforeEvaluation) {
  Falcata::Config config;
  Falcata::Metadata metadata;
  Falcata::AUCMetric auc(config);
  Falcata::AveragePrecisionMetric ap(config);
  EXPECT_THROW(auc.Init(metadata, 0), std::runtime_error);
  EXPECT_THROW(ap.Init(metadata, 0), std::runtime_error);
}

TEST(BinaryRankingMetrics, NonemptyResultsArePreserved) {
  Falcata::Config config;
  Falcata::Metadata metadata;
  metadata.Init(2, -1, -1);
  const Falcata::label_t labels[] = {0, 1};
  metadata.SetLabel(labels, 2);
  const double scores[] = {0, 1};
  Falcata::AUCMetric auc(config);
  Falcata::AveragePrecisionMetric ap(config);
  auc.Init(metadata, 2);
  ap.Init(metadata, 2);
  EXPECT_EQ(auc.Eval(scores, nullptr)[0], 1.0);
  EXPECT_EQ(ap.Eval(scores, nullptr)[0], 1.0);
}
