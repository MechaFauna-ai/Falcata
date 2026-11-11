/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
// CUDA lambdarank must compute the CPU objective's gradients: one round of lambdas and hessians from
// CUDALambdarankNDCG against LambdarankNDCG on the same scores, for every kernel family (queries up to 1024 items,
// 1025-2048 items, and the globally sorted kernels past 2048), deterministic or not, with the default settings,
// lambdarank_norm=false and a truncation level of 10.
//
// Tolerance. Each pair's lambda and hessian are the CPU's value exactly: the same discount table, the same sigmoid
// table lookup and the same double arithmetic. What may differ is how they are added up:
//  - the deterministic kernels add each item's float terms, and the double sum behind the norm factor, in the CPU's
//    pair order: the gradients are the CPU's bit for bit, tolerance zero.
//  - the non-deterministic kernels add an item's float terms with atomics in no fixed order. Any order of n float
//    additions is within (n - 1) * 2^-24 * sum|term| of the exact sum, so two orders differ by at most twice that;
//    the test computes sum|term| for each item from the CPU objective's own tables.
#include <gtest/gtest.h>
#include <Falcata/config.h>
#include <Falcata/dataset.h>
#include <Falcata/metric.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../src/objective/cuda/cuda_rank_objective.hpp"

namespace {

using Falcata::data_size_t;
using Falcata::label_t;
using Falcata::score_t;

struct Case {
  std::vector<data_size_t> query_sizes;
  std::vector<label_t> labels;
  std::vector<label_t> weights;  // empty: unweighted
  std::vector<double> scores;
};

Case MakeCase(const std::vector<data_size_t>& query_sizes, const uint32_t seed, const bool ties, const bool weighted) {
  Case c;
  c.query_sizes = query_sizes;
  const data_size_t n = std::accumulate(query_sizes.begin(), query_sizes.end(), data_size_t(0));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> label_dist(0, 4);
  std::normal_distribution<double> score_dist(0.0, 1.0);
  std::uniform_real_distribution<double> weight_dist(0.5, 1.5);
  for (data_size_t i = 0; i < n; ++i) {
    c.labels.push_back(static_cast<label_t>(label_dist(rng)));
    double s = score_dist(rng);
    if (ties) {
      // many exact ties, plus pairs that tie in float but not in double: the CPU stable-sorts the double scores
      s = std::round(s * 4.0) / 4.0 + ((i % 3 == 0) ? 1e-12 : 0.0);
    }
    c.scores.push_back(s);
    if (weighted) {
      c.weights.push_back(static_cast<label_t>(weight_dist(rng)));
    }
  }
  return c;
}

Falcata::Config MakeConfig(const std::unordered_map<std::string, std::string>& extra, const bool deterministic) {
  std::unordered_map<std::string, std::string> params = {{"objective", "lambdarank"},
                                                         {"deterministic", deterministic ? "true" : "false"}};
  for (const auto& kv : extra) params[kv.first] = kv.second;
  Falcata::Config config;
  config.Set(params);
  return config;
}

// per item, an upper bound on the sum over its pairs of |lambda term| and |hessian term| (before the norm factor)
// and on their count, from the CPU tables: every pair of different labels counts (truncation only drops pairs), with
// the largest discount difference the query has
void AbsoluteTermSums(const Case& c, const Falcata::Config& config, const Falcata::LambdarankNDCG& cpu,
                      std::vector<double>* abs_lambda, std::vector<double>* abs_hessian, std::vector<int>* num_terms) {
  const data_size_t n = static_cast<data_size_t>(c.labels.size());
  abs_lambda->assign(n, 0.0);
  abs_hessian->assign(n, 0.0);
  num_terms->assign(n, 0);
  std::vector<double> label_gain = config.label_gain;
  Falcata::DCGCalculator::DefaultLabelGain(&label_gain);
  data_size_t start = 0;
  for (const data_size_t cnt : c.query_sizes) {
    const double max_dcg = Falcata::DCGCalculator::CalMaxDCGAtK(config.lambdarank_truncation_level,
                                                                c.labels.data() + start, cnt);
    const double inverse_max_dcg = max_dcg > 0.0 ? 1.0 / max_dcg : max_dcg;
    for (data_size_t i = 0; i < cnt; ++i) {
      for (data_size_t j = 0; j < cnt; ++j) {
        const data_size_t a = start + i, b = start + j;
        if (c.labels[a] <= c.labels[b]) continue;  // a is the higher label of the pair
        // the discount difference is largest at the extreme ranks; bound it by the largest possible
        const double paired_discount = Falcata::DCGCalculator::GetDiscount(0) - Falcata::DCGCalculator::GetDiscount(
            std::min<data_size_t>(cnt - 1, 9999));
        double delta_ndcg = (label_gain[static_cast<int>(c.labels[a])] - label_gain[static_cast<int>(c.labels[b])]) *
                            paired_discount * inverse_max_dcg;
        if (config.lambdarank_norm) {
          // the score-distance regularization divides by 0.01 + |delta score|, which may enlarge the term
          delta_ndcg /= std::min(1.0, 0.01 + std::fabs(c.scores[a] - c.scores[b]));
        }
        const double p = cpu.GetSigmoid(c.scores[a] - c.scores[b]);
        const double term_lambda = std::fabs(p * config.sigmoid * delta_ndcg);
        const double term_hessian = std::fabs(p * (1.0 - p) * config.sigmoid * config.sigmoid * delta_ndcg);
        (*abs_lambda)[a] += term_lambda;
        (*abs_lambda)[b] += term_lambda;
        (*abs_hessian)[a] += term_hessian;
        (*abs_hessian)[b] += term_hessian;
        ++(*num_terms)[a];
        ++(*num_terms)[b];
      }
    }
    start += cnt;
  }
}

void ExpectCPUGradients(const Case& c, const std::unordered_map<std::string, std::string>& extra,
                        const bool deterministic) {
  const data_size_t n = static_cast<data_size_t>(c.labels.size());
  const Falcata::Config config = MakeConfig(extra, deterministic);
  Falcata::Metadata metadata;
  metadata.Init(n, -1, -1);
  metadata.SetLabel(c.labels.data(), n);
  if (!c.weights.empty()) metadata.SetWeights(c.weights.data(), n);
  metadata.SetQuery(c.query_sizes.data(), static_cast<data_size_t>(c.query_sizes.size()));
  metadata.CreateCUDAMetadata(0);

  Falcata::LambdarankNDCG cpu(config);
  cpu.Init(metadata, n);
  std::vector<score_t> cpu_g(n), cpu_h(n);
  cpu.GetGradients(c.scores.data(), cpu_g.data(), cpu_h.data());

  Falcata::CUDALambdarankNDCG cuda(config);
  cuda.Init(metadata, n);
  double* d_scores = nullptr;
  score_t* d_g = nullptr;
  score_t* d_h = nullptr;
  ASSERT_EQ(cudaMalloc(&d_scores, n * sizeof(double)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_g, n * sizeof(score_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_h, n * sizeof(score_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_scores, c.scores.data(), n * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
  cuda.GetGradients(d_scores, d_g, d_h);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<score_t> cuda_g(n), cuda_h(n);
  ASSERT_EQ(cudaMemcpy(cuda_g.data(), d_g, n * sizeof(score_t), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(cuda_h.data(), d_h, n * sizeof(score_t), cudaMemcpyDeviceToHost), cudaSuccess);
  cudaFree(d_scores);
  cudaFree(d_g);
  cudaFree(d_h);

  std::vector<double> abs_lambda, abs_hessian;
  std::vector<int> num_terms;
  if (!deterministic) AbsoluteTermSums(c, config, cpu, &abs_lambda, &abs_hessian, &num_terms);
  int mismatches = 0;
  for (data_size_t i = 0; i < n && mismatches < 10; ++i) {
    double tol_g, tol_h;
    if (deterministic) {
      tol_g = 0.0;
      tol_h = 0.0;
    } else {
      // two float summation orders of the same terms, scaled by the weight and by the norm factor
      // log2(1 + x) / x, which is below 1 / ln 2 < 1.45
      const double w = (c.weights.empty() ? 1.0 : c.weights[i]) * (config.lambdarank_norm ? 1.45 : 1.0);
      const double u = std::ldexp(1.0, -24);
      tol_g = 2.0 * std::max(num_terms[i] - 1, 0) * u * abs_lambda[i] * w + std::fabs(cpu_g[i]) * 2.0 * u;
      tol_h = 2.0 * std::max(num_terms[i] - 1, 0) * u * abs_hessian[i] * w + std::fabs(cpu_h[i]) * 2.0 * u;
    }
    const bool bad = std::fabs(static_cast<double>(cuda_g[i]) - cpu_g[i]) > tol_g ||
                     std::fabs(static_cast<double>(cuda_h[i]) - cpu_h[i]) > tol_h;
    if (bad) {
      ++mismatches;
      ADD_FAILURE() << "item " << i << ": gradient cuda " << cuda_g[i] << " cpu " << cpu_g[i] << " (tol " << tol_g
                    << "), hessian cuda " << cuda_h[i] << " cpu " << cpu_h[i] << " (tol " << tol_h << ")";
    }
  }
}

std::vector<std::unordered_map<std::string, std::string>> Settings() {
  return {{}, {{"lambdarank_norm", "false"}}, {{"lambdarank_truncation_level", "10"}}};
}

std::vector<data_size_t> Sizes(const int count, const data_size_t lo, const data_size_t hi, const uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<data_size_t> dist(lo, hi);
  std::vector<data_size_t> sizes;
  for (int q = 0; q < count; ++q) sizes.push_back(dist(rng));
  return sizes;
}

void RunFamily(const std::vector<data_size_t>& sizes) {
  for (const bool deterministic : {true, false}) {
    for (const auto& setting : Settings()) {
      SCOPED_TRACE(std::string("deterministic=") + (deterministic ? "true" : "false") +
                   (setting.empty() ? " default" : " " + setting.begin()->first + "=" + setting.begin()->second));
      {
        SCOPED_TRACE("distinct scores");
        ExpectCPUGradients(MakeCase(sizes, 1, false, false), setting, deterministic);
      }
      {
        SCOPED_TRACE("tied scores");
        ExpectCPUGradients(MakeCase(sizes, 2, true, false), setting, deterministic);
      }
      {
        SCOPED_TRACE("weighted");
        ExpectCPUGradients(MakeCase(sizes, 3, false, true), setting, deterministic);
      }
    }
  }
}

}  // namespace

TEST(CUDALambdarankCPUParity, QueriesUpTo1024Items) { RunFamily(Sizes(40, 150, 250, 11)); }

TEST(CUDALambdarankCPUParity, Queries1025To2048Items) { RunFamily(Sizes(4, 1300, 1700, 12)); }

TEST(CUDALambdarankCPUParity, QueriesPast2048ItemsSortedKernels) { RunFamily(Sizes(3, 2600, 3100, 13)); }

// All scores equal (the first boosting round): the CPU's stable sort keeps item order, and so must the device's.
TEST(CUDALambdarankCPUParity, FirstRoundAllScoresTied) {
  for (const auto& sizes : {Sizes(10, 150, 250, 21), Sizes(2, 1300, 1700, 22), Sizes(2, 2600, 3100, 23)}) {
    for (const bool deterministic : {true, false}) {
      Case c = MakeCase(sizes, 4, false, false);
      std::fill(c.scores.begin(), c.scores.end(), 0.0);
      ExpectCPUGradients(c, {}, deterministic);
    }
  }
}
