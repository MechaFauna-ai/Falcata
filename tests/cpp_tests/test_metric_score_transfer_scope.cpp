/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_CUDA
#include <gtest/gtest.h>
#include <Falcata/metric.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/boosting/gbdt.h"

namespace {
void CheckCUDA(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct DeviceScores {
  double* data = nullptr;
  int rows;
  explicit DeviceScores(int count) : rows(count) { CheckCUDA(cudaMalloc(&data, rows * sizeof(double))); }
  ~DeviceScores() { cudaFree(data); }
  void Fill(double value) {
    const std::vector<double> host(rows, value);
    CheckCUDA(cudaMemcpy(data, host.data(), rows * sizeof(double), cudaMemcpyHostToDevice));
  }
};

class SumMetric : public Falcata::Metric {
 public:
  explicit SumMetric(int rows) : rows_(rows) {}
  void Init(const Falcata::Metadata&, Falcata::data_size_t) override {}
  const std::vector<std::string>& GetName() const override { return name_; }
  double factor_to_bigger_better() const override { return -1; }
  std::vector<double> Eval(const double* score, const Falcata::ObjectiveFunction*) const override {
    double sum = 0;
    for (int i = 0; i < rows_; ++i) sum += score[i];
    return {sum};
  }
 protected:
  int rows_;
  std::vector<std::string> name_ = {"sum"};
};

class DeviceSumMetric : public SumMetric {
 public:
  explicit DeviceSumMetric(int rows) : SumMetric(rows) {}
  bool IsCUDAMetric() const override { return true; }
  std::vector<double> Eval(const double* score, const Falcata::ObjectiveFunction*) const override {
    std::vector<double> host(rows_);
    CheckCUDA(cudaMemcpy(host.data(), score, rows_ * sizeof(double), cudaMemcpyDeviceToHost));
    return SumMetric::Eval(host.data(), nullptr);
  }
};

class ThrowMetric : public SumMetric {
 public:
  ThrowMetric() : SumMetric(0) {}
  std::vector<double> Eval(const double*, const Falcata::ObjectiveFunction*) const override {
    throw std::runtime_error("metric error");
  }
};

class TransferModel : public Falcata::GBDT {
 public:
  explicit TransferModel(bool device) { boosting_on_gpu_ = device; }
  double Single(const Falcata::Metric* metric, const double* score, int rows) {
    return EvalOneMetric(metric, score, rows)[0];
  }
  std::vector<double> Pair(const Falcata::Metric* metric, const double* score, int rows) {
    MetricScoreTransferScope scope(this);
    return {Single(metric, score, rows), Single(metric, score, rows)};
  }
  std::vector<double> Nested(const Falcata::Metric* metric, const double* first, const double* second, int rows) {
    MetricScoreTransferScope outer(this);
    const double a = Single(metric, first, rows);
    const auto inner = Pair(metric, second, rows);
    return {a, inner[0], Single(metric, first, rows)};
  }
  void ThrowAfterTransfer(const Falcata::Metric* metric, const double* score, int rows) {
    MetricScoreTransferScope scope(this);
    Single(metric, score, rows);
    ThrowMetric throwing;
    Single(&throwing, score, rows);
  }
};
}  // namespace

TEST(MetricScoreTransfer, RefreshEachScopeAndUnscopedCall) {
  DeviceScores device(4096);
  SumMetric metric(device.rows);
  TransferModel model(true);
  device.Fill(1);
  EXPECT_EQ(model.Pair(&metric, device.data, device.rows), (std::vector<double>{4096, 4096}));
  device.Fill(2);
  EXPECT_EQ(model.Pair(&metric, device.data, device.rows), (std::vector<double>{8192, 8192}));
  device.Fill(3);
  EXPECT_EQ(model.Single(&metric, device.data, device.rows), 12288);
  device.Fill(4);
  EXPECT_EQ(model.Single(&metric, device.data, device.rows), 16384);
  EXPECT_THROW(model.ThrowAfterTransfer(&metric, device.data, device.rows), std::runtime_error);
  device.Fill(5);
  EXPECT_EQ(model.Single(&metric, device.data, device.rows), 20480);
}

TEST(MetricScoreTransfer, NestedScopesAndDifferentSources) {
  DeviceScores first(4096), second(4096);
  first.Fill(1);
  second.Fill(3);
  SumMetric metric(first.rows);
  TransferModel model(true);
  EXPECT_EQ(model.Nested(&metric, first.data, second.data, first.rows), (std::vector<double>{4096, 12288, 4096}));
}

TEST(MetricScoreTransfer, HostToDeviceRefreshesEachScope) {
  TransferModel model(false);
  DeviceSumMetric metric(4096);
  std::vector<double> host(4096, 1);
  EXPECT_EQ(model.Pair(&metric, host.data(), host.size()), (std::vector<double>{4096, 4096}));
  std::fill(host.begin(), host.end(), 2);
  EXPECT_EQ(model.Pair(&metric, host.data(), host.size()), (std::vector<double>{8192, 8192}));
}
#endif  // USE_CUDA
