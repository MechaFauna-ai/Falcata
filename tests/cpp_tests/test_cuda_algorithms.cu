/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <vector>

#include <Falcata/cuda/cuda_algorithms.hpp>

namespace {
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t size) {
    EXPECT_EQ(cudaMalloc(&data_, size * sizeof(T)), cudaSuccess);
  }
  ~DeviceBuffer() { EXPECT_EQ(cudaFree(data_), cudaSuccess); }
  T* data() { return data_; }

 private:
  T* data_ = nullptr;
};
}  // namespace

TEST(CUDAAlgorithms, SortedPrefixSumIgnoresScratchAndScansEveryBlock) {
  // The last two sizes give each reduction thread multiple block totals.
  for (const int n : {1, 4, 1023, 1024, 1025, 3079, 1024 * 1024 + 17, 2048 * 1024 + 1}) {
    const int blocks = (n + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
    std::vector<float> weights(n);
    std::vector<int> indices(n);
    std::vector<double> expected(n);
    for (int i = 0; i < n; ++i) {
      weights[i] = static_cast<float>(i % 13) * 0.25f;
      indices[i] = n - 1 - i;
    }
    double sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += weights[indices[i]];
      expected[i] = sum;
    }
    DeviceBuffer<float> device_weights(n);
    DeviceBuffer<int> device_indices(n);
    DeviceBuffer<double> device_prefix(n);
    DeviceBuffer<double> device_scratch(blocks + 1);
    ASSERT_EQ(cudaMemcpy(device_weights.data(), weights.data(), n * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_indices.data(), indices.data(), n * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    for (const double poison : {0.0, 10.0, std::numeric_limits<double>::quiet_NaN()}) {
      SCOPED_TRACE(::testing::Message() << "n=" << n << " scratch=" << poison);
      std::vector<double> scratch(blocks + 1, poison);
      ASSERT_EQ(cudaMemcpy(device_scratch.data(), scratch.data(), scratch.size() * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
      Falcata::GlobalInclusiveArgPrefixSum<float, double, int>(
        device_indices.data(), device_weights.data(), device_prefix.data(), device_scratch.data(), n);
      std::vector<double> actual(n);
      ASSERT_EQ(cudaMemcpy(actual.data(), device_prefix.data(), n * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
      // Dyadic weights keep every expected sum exact in double precision.
      const auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
      if (mismatch.first != expected.end()) {
        ADD_FAILURE() << "first wrong prefix at " << (mismatch.first - expected.begin())
                      << ": expected " << *mismatch.first << " actual " << *mismatch.second;
      }
    }
  }
}

TEST(CUDAAlgorithms, WeightedPercentileMatchesCPUSemanticsWithDirtyScratch) {
  // Unsorted labels exercise the sort permutation. Include zero weights, a
  // CDF gap below one (CPU selects the lower value), and zero total weight
  // (CPU selects the largest label). This changes no percentile conventions.
  const std::vector<float> labels{5, 2, 4, 3};
  const std::vector<std::vector<float>> weights{{1, 4, 2, 3}, {0, 0, 2, 3},
                                               {0.0625f, 0.25f, 0.125f, 0.1875f}, {0, 0, 0, 0}};
  DeviceBuffer<float> device_labels(4);
  DeviceBuffer<float> device_weights(4);
  DeviceBuffer<int> indices(4);
  DeviceBuffer<double> prefix(4);
  DeviceBuffer<double> scratch(2);
  DeviceBuffer<float> output(1);
  ASSERT_EQ(cudaMemcpy(device_labels.data(), labels.data(), 4 * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  for (const auto& w : weights) {
    ASSERT_EQ(cudaMemcpy(device_weights.data(), w.data(), 4 * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    for (const double alpha : {0.0, 0.3, 0.5, 0.7, 1.0}) {
      std::vector<int> sorted{0, 1, 2, 3};
      std::stable_sort(sorted.begin(), sorted.end(), [&](int a, int b) { return labels[a] < labels[b]; });
      std::vector<double> cdf;
      double total = 0;
      for (const int i : sorted) {
        total += w[i];
        cdf.push_back(total);
      }
      const double threshold = total * alpha;
      const size_t pos = std::min(static_cast<size_t>(std::upper_bound(cdf.begin(), cdf.end(), threshold) - cdf.begin()), labels.size() - 1);
      float expected = labels[sorted[pos]];
      if (pos != 0 && pos != labels.size() - 1) {
        const double gap = cdf[pos] - cdf[pos - 1];
        expected = labels[sorted[pos - 1]];
        if (gap >= 1.0) {
          expected = static_cast<float>((threshold - cdf[pos - 1]) / gap * (labels[sorted[pos]] - expected) + expected);
        }
      }
      for (const double poison : {10.0, std::numeric_limits<double>::quiet_NaN()}) {
        SCOPED_TRACE(::testing::Message() << "weights=" << ::testing::PrintToString(w) << " alpha=" << alpha << " scratch=" << poison);
        const double dirty[] = {poison, poison};
        ASSERT_EQ(cudaMemcpy(scratch.data(), dirty, sizeof(dirty), cudaMemcpyHostToDevice), cudaSuccess);
        Falcata::PercentileGlobal<float, int, float, double, true, true>(
          device_labels.data(), device_weights.data(), indices.data(), prefix.data(), scratch.data(), alpha, 4, output.data());
        float actual = 0;
        ASSERT_EQ(cudaMemcpy(&actual, output.data(), sizeof(actual), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_FLOAT_EQ(actual, expected);
      }
    }
  }
}
