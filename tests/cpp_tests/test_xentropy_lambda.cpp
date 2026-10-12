/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <Falcata/config.h>
#include <Falcata/dataset.h>

#include <cmath>
#include <vector>

#include "../../src/objective/xentropy_objective.hpp"

using Falcata::Config;
using Falcata::CrossEntropyLambda;
using Falcata::data_size_t;
using Falcata::label_t;
using Falcata::Metadata;
using Falcata::score_t;

namespace {

struct GradHess {
  std::vector<score_t> grad;
  std::vector<score_t> hess;
};

GradHess WeightedGradients(const std::vector<double>& scores, const std::vector<label_t>& labels,
                           const std::vector<label_t>& weights) {
  const data_size_t n = static_cast<data_size_t>(scores.size());
  Metadata metadata;
  metadata.Init(n, -1, -1);
  metadata.SetLabel(labels.data(), n);
  metadata.SetWeights(weights.data(), n);
  Config config;
  CrossEntropyLambda objective(config);
  objective.Init(metadata, n);
  GradHess out{std::vector<score_t>(n), std::vector<score_t>(n)};
  objective.GetGradients(scores.data(), out.grad.data(), out.hess.data());
  return out;
}

// The weighted expression GetGradients has always used, kept here as the
// reference wherever it is finite.
void ReferenceGradHess(double s, double w, double y, score_t* grad, score_t* hess) {
  const double epf = std::exp(s);
  const double hhat = std::log1p(epf);
  const double z = 1.0f - std::exp(-w * hhat);
  const double enf = 1.0f / epf;
  *grad = static_cast<score_t>((1.0f - y / z) * w / (1.0f + enf));
  const double c = 1.0f / (1.0f - z);
  double d = 1.0f + epf;
  const double a = w * epf / (d * d);
  d = c - 1.0f;
  const double b = (c / (d * d)) * (1.0f + w * epf - c);
  *hess = static_cast<score_t>(a * (1.0f + y * b));
}

}  // namespace

// With unit weights cross_entropy_lambda is plain logistic loss: gradient
// sigmoid(s) - y and Hessian sigmoid(s) * (1 - sigmoid(s)), for every score.
TEST(CrossEntropyLambda, UnitWeightsMatchLogisticLossOverTheWholeScoreRange) {
  std::vector<double> scores;
  std::vector<label_t> labels;
  for (double s = -800.0; s <= 800.0; s += 0.25) {
    for (label_t y : {0.0f, 0.25f, 1.0f}) {
      scores.push_back(s);
      labels.push_back(y);
    }
  }
  const std::vector<label_t> weights(scores.size(), 1.0f);
  const GradHess gh = WeightedGradients(scores, labels, weights);
  for (size_t i = 0; i < scores.size(); ++i) {
    const double s = scores[i];
    const double p = s >= 0.0 ? 1.0 / (1.0 + std::exp(-s)) : std::exp(s) / (1.0 + std::exp(s));
    const double one_minus_p = s >= 0.0 ? std::exp(-s) / (1.0 + std::exp(-s)) : 1.0 / (1.0 + std::exp(s));
    const double want_grad = s >= 0.0 ? (1.0 - labels[i]) - one_minus_p : p - labels[i];
    const double want_hess = p * one_minus_p;
    ASSERT_TRUE(std::isfinite(gh.grad[i]) && std::isfinite(gh.hess[i])) << "s=" << s << " y=" << labels[i];
    EXPECT_NEAR(gh.grad[i], want_grad, 1e-6 * std::fabs(want_grad) + 1e-37) << "s=" << s << " y=" << labels[i];
    EXPECT_NEAR(gh.hess[i], want_hess, 1e-6 * want_hess + 1e-37) << "s=" << s << " y=" << labels[i];
  }
}

// For any positive weight the gradient and Hessian are finite and the Hessian
// is non-negative. Where u = w * log1p(exp(s)) lies in [1e-3, 20] and s <= 50,
// the original expression is accurate and the result is bit-identical to it.
TEST(CrossEntropyLambda, WeightedGradientsFiniteAndUnchangedWhereAlreadyAccurate) {
  std::vector<double> scores;
  std::vector<label_t> labels;
  std::vector<label_t> weights;
  for (double s = -800.0; s <= 800.0; s += 0.25) {
    for (label_t w : {1e-3f, 0.1f, 0.5f, 1.0f, 2.0f, 10.0f, 100.0f}) {
      for (label_t y : {0.0f, 0.3f, 1.0f}) {
        scores.push_back(s);
        weights.push_back(w);
        labels.push_back(y);
      }
    }
  }
  const GradHess gh = WeightedGradients(scores, labels, weights);
  int in_band = 0;
  for (size_t i = 0; i < scores.size(); ++i) {
    ASSERT_TRUE(std::isfinite(gh.grad[i]) && std::isfinite(gh.hess[i]))
        << "s=" << scores[i] << " w=" << weights[i] << " y=" << labels[i];
    EXPECT_GE(gh.hess[i], 0.0f) << "s=" << scores[i] << " w=" << weights[i] << " y=" << labels[i];
    const double u = weights[i] * std::log1p(std::exp(scores[i]));
    if (u >= 1e-3 && u <= 20.0 && scores[i] <= 50.0) {
      ++in_band;
      score_t ref_grad, ref_hess;
      ReferenceGradHess(scores[i], weights[i], labels[i], &ref_grad, &ref_hess);
      EXPECT_EQ(gh.grad[i], ref_grad) << "s=" << scores[i] << " w=" << weights[i] << " y=" << labels[i];
      EXPECT_EQ(gh.hess[i], ref_hess) << "s=" << scores[i] << " w=" << weights[i] << " y=" << labels[i];
    }
  }
  EXPECT_GT(in_band, 0);
}
