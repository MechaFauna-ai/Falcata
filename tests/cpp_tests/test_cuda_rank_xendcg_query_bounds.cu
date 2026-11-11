/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
// CUDARankXENDCG writes each query's gradients and hessians inside the query's own rows only:
//  - a 1-item query is zeroed over [0, 1), not [0, 1] (the kernels once looped i <= query_item_count, writing the
//    next query's first item, or one element past the buffer for the last query);
//  - a query of 65536+ items sizes the kernels from its true length (the per-thread maxima were once uint16_t, so
//    the length wrapped and the launch was sized for a short query).
// Every kernel family is covered: shared memory up to 1024 and up to 2048 items, and global memory past 2048. The
// output buffers carry sentinels past every row, and each 1-item query's own rows must come back zero.
#include <gtest/gtest.h>
#include <Falcata/config.h>
#include <Falcata/dataset.h>

#include <cmath>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../src/objective/cuda/cuda_rank_objective.hpp"

namespace {

using Falcata::data_size_t;
using Falcata::label_t;
using Falcata::score_t;

constexpr score_t kSentinel = 12345.0f;
constexpr data_size_t kPad = 64;

void ExpectWritesStayInQueries(const std::vector<data_size_t>& query_sizes) {
  const data_size_t n = std::accumulate(query_sizes.begin(), query_sizes.end(), data_size_t(0));
  std::vector<label_t> labels(n);
  std::vector<double> scores(n);
  for (data_size_t i = 0; i < n; ++i) {
    labels[i] = static_cast<label_t>(i % 3);
    scores[i] = 0.001 * (i % 97);
  }
  Falcata::Config config;
  std::unordered_map<std::string, std::string> params = {{"objective", "rank_xendcg"}};
  config.Set(params);
  Falcata::Metadata metadata;
  metadata.Init(n, -1, -1);
  metadata.SetLabel(labels.data(), n);
  metadata.SetQuery(query_sizes.data(), static_cast<data_size_t>(query_sizes.size()));
  metadata.CreateCUDAMetadata(0);
  Falcata::CUDARankXENDCG objective(config);
  objective.Init(metadata, n);

  double* d_scores = nullptr;
  score_t* d_g = nullptr;
  score_t* d_h = nullptr;
  ASSERT_EQ(cudaMalloc(&d_scores, n * sizeof(double)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_g, (n + kPad) * sizeof(score_t)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_h, (n + kPad) * sizeof(score_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_scores, scores.data(), n * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
  const std::vector<score_t> sentinel(n + kPad, kSentinel);
  ASSERT_EQ(cudaMemcpy(d_g, sentinel.data(), sentinel.size() * sizeof(score_t), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_h, sentinel.data(), sentinel.size() * sizeof(score_t), cudaMemcpyHostToDevice), cudaSuccess);
  objective.GetGradients(d_scores, d_g, d_h);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<score_t> g(n + kPad), h(n + kPad);
  ASSERT_EQ(cudaMemcpy(g.data(), d_g, g.size() * sizeof(score_t), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(h.data(), d_h, h.size() * sizeof(score_t), cudaMemcpyDeviceToHost), cudaSuccess);
  cudaFree(d_scores);
  cudaFree(d_g);
  cudaFree(d_h);

  for (data_size_t i = n; i < n + kPad; ++i) {
    ASSERT_EQ(g[i], kSentinel) << "gradient written past the last row, at " << i;
    ASSERT_EQ(h[i], kSentinel) << "hessian written past the last row, at " << i;
  }
  data_size_t start = 0;
  for (const data_size_t cnt : query_sizes) {
    for (data_size_t i = start; i < start + cnt; ++i) {
      ASSERT_TRUE(std::isfinite(g[i]) && g[i] != kSentinel) << "row " << i << " not written";
      ASSERT_TRUE(std::isfinite(h[i]) && h[i] != kSentinel) << "row " << i << " not written";
      if (cnt == 1) {
        ASSERT_EQ(g[i], 0.0f) << "1-item query at row " << i;
        ASSERT_EQ(h[i], 0.0f) << "1-item query at row " << i;
      }
    }
    start += cnt;
  }
}

}  // namespace

TEST(CUDARankXENDCGQueryBounds, SingletonQueriesSharedMemory1024) {
  ExpectWritesStayInQueries({1, 40, 1, 1, 300, 1, 7, 1});
}

TEST(CUDARankXENDCGQueryBounds, SingletonQueriesSharedMemory2048) {
  ExpectWritesStayInQueries({1, 1500, 1, 1, 30, 1});
}

TEST(CUDARankXENDCGQueryBounds, SingletonQueriesGlobalMemory) {
  ExpectWritesStayInQueries({1, 3000, 1, 1, 30, 1});
}

TEST(CUDARankXENDCGQueryBounds, QueryOf65536PlusItems) {
  ExpectWritesStayInQueries({1, 70000, 3, 1});
}
