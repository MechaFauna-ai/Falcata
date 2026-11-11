/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/config.h>
#include <Falcata/dataset.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../src/objective/cuda/cuda_multiclass_objective.hpp"

namespace {

// Softmax-converts \p num_rows class-major rows (row r, class k at k * num_rows + r) on the device. Returns false
// when the objective refuses the call.
bool Convert(const Falcata::CUDAMulticlassSoftmax& objective, const int num_rows, std::vector<double>* out) {
  std::vector<double> input(static_cast<size_t>(num_rows) * 3);
  for (int r = 0; r < num_rows; ++r) {
    for (int k = 0; k < 3; ++k) input[static_cast<size_t>(k) * num_rows + r] = 0.001 * r + k;
  }
  double* d_in = nullptr;
  double* d_out = nullptr;
  EXPECT_EQ(cudaMalloc(&d_in, input.size() * sizeof(double)), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&d_out, input.size() * sizeof(double)), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(d_in, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
  EXPECT_EQ(cudaMemset(d_out, 0, input.size() * sizeof(double)), cudaSuccess);
  bool converted = true;
  try {
    objective.ConvertOutputCUDA(num_rows, d_in, d_out);
  } catch (const std::exception&) {
    converted = false;
  }
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  out->assign(input.size(), 0.0);
  EXPECT_EQ(cudaMemcpy(out->data(), d_out, input.size() * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(cudaFree(d_in), cudaSuccess);
  EXPECT_EQ(cudaFree(d_out), cudaSuccess);
  return converted;
}

// Rows whose three outputs are a softmax (positive, summing to 1).
int CountConvertedRows(const std::vector<double>& out, const int num_rows) {
  int converted = 0;
  for (int r = 0; r < num_rows; ++r) {
    double sum = 0.0;
    bool positive = true;
    for (int k = 0; k < 3; ++k) {
      const double p = out[static_cast<size_t>(k) * num_rows + r];
      sum += p;
      positive = positive && p > 0.0;
    }
    converted += (positive && std::fabs(sum - 1.0) < 1e-9);
  }
  return converted;
}

}  // namespace

// CUDAMulticlassSoftmax::ConvertOutputCUDA sized its grid from the training rows instead of the rows passed in, so a
// call with more rows than the training set returned rows it never converted. Every row must be converted, or the
// call refused (its softmax scratch only holds the training rows).
TEST(CUDAMulticlassSoftmax, ConvertOutputCoversTheRowsPassedIn) {
  const int num_train = 1000;
  Falcata::Config config;
  std::unordered_map<std::string, std::string> params = {{"objective", "multiclass"}, {"num_class", "3"}};
  config.Set(params);
  Falcata::Metadata metadata;
  metadata.Init(num_train, -1, -1);
  std::vector<Falcata::label_t> labels(num_train);
  for (int i = 0; i < num_train; ++i) labels[i] = static_cast<Falcata::label_t>(i % 3);
  metadata.SetLabel(labels.data(), num_train);
  metadata.CreateCUDAMetadata(0);
  Falcata::CUDAMulticlassSoftmax objective(config);
  objective.Init(metadata, num_train);

  std::vector<double> out;
  ASSERT_TRUE(Convert(objective, 300, &out));
  EXPECT_EQ(CountConvertedRows(out, 300), 300);

  const int num_rows = 2 * num_train;
  if (Convert(objective, num_rows, &out)) {
    EXPECT_EQ(CountConvertedRows(out, num_rows), num_rows) << "rows returned without conversion";
  }
}
