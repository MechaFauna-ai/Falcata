/* Copyright (c) 2026 Falcata contributors. MIT license; see LICENSE.
 * UNBUILT/UNRUN standalone integration projection. Link to the actual proposed library; execute only via gpuq.
 * Exercise the SAME learner's two public refit overloads, unlike Python Booster.refit which creates a new learner.
 */
#include <Falcata/c_api.h>
#include <Falcata/config.h>
#include <Falcata/dataset.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/treelearner/cuda/cuda_single_gpu_tree_learner.hpp"

namespace {
using Falcata::Config;
using Falcata::Dataset;
using Falcata::score_t;
using Falcata::CUDASingleGPUTreeLearner;
using Falcata::CUDAVector;
using Falcata::Tree;
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
std::vector<std::string> Run(const Dataset* data, const std::vector<float>& x,
                             const std::vector<score_t>& initial_gradients, bool defer) {
  std::fprintf(stderr, "[map-only-refit-probe] arm=%s start\n", defer ? "deferred" : "eager");
  std::fflush(stderr);
  Config config;
  config.Set({{"device_type", "cuda"}, {"objective", "regression"}, {"boosting", "gbdt"},
    {"metric", "None"}, {"quant_mode", "fixedpoint"}, {"quant_bins", "64"}, {"num_threads", "32"},
    {"num_leaves", "16"}, {"max_depth", "3"}, {"feature_fraction", "1"}, {"min_data_in_leaf", "2"},
    {"min_gain_to_split", "10"}, {"quant_train_renew_leaf", "false"}, {"verbosity", "-1"},
    {"boost_from_average", "false"}, {"feature_pre_filter", "false"},
    {"cuda_plan", std::string("auto,device_tree:on,final_map_only:") + (defer ? "on" : "off")}});
  Config smooth_config = config;  // outlives learner even after ResetConfig takes its address
  CUDASingleGPUTreeLearner learner(&config, true);
  learner.Init(data, true);
  CUDAVector<score_t> gradients, hessians;
  gradients.InitFromHostVector(initial_gradients);
  hessians.InitFromHostVector(std::vector<score_t>(initial_gradients.size(), 1.0f));
  std::vector<std::string> result;
  auto record = [&](const Tree& tree) {
    for (int leaf = 0; leaf < tree.num_leaves(); ++leaf) {
      Require(std::isfinite(tree.LeafOutput(leaf)), "nonfinite refit output");
    }
    result.push_back(tree.ToString());
  };
  auto leaf_pred = [&](const Tree& tree) {
    std::vector<int> pred(data->num_data());
    for (int row = 0; row < data->num_data(); ++row) {
      double features[3] = {x[3 * row], x[3 * row + 1], x[3 * row + 2]};
      pred[row] = tree.PredictLeafIndex(features);
    }
    return pred;
  };
  // First, a positive complete Train directly followed by the leaf_pred overload on that same learner.
  std::unique_ptr<Tree> first(learner.Train(gradients.RawData(), hessians.RawData(), true));
  Require(first->num_leaves() == 6, "mixed fixture must actually reach the static final level with terminal gaps");
  record(*first);
  std::vector<score_t> refit_gradients = initial_gradients;
  for (auto& g : refit_gradients) g *= 0.75f;
  gradients.InitFromHostVector(refit_gradients);
  std::unique_ptr<Tree> leaf_refit(learner.FitByExistingTree(first.get(), leaf_pred(*first),
                                                         gradients.RawData(), hessians.RawData()));
  record(*leaf_refit);
  Require(result[0] != result[1], "leaf_pred refit must do real work");
  // That public call also initializes the existing refit_num_data_ contract. Do not invoke the direct overload
  // with an uninitialized legacy row count; train another complete pending tree and then test direct refit.
  std::unique_ptr<Tree> second(learner.Train(gradients.RawData(), hessians.RawData(), false));
  Require(second->num_leaves() == 6, "second complete tree lost final-level coverage");
  record(*second);
  std::unique_ptr<Tree> direct_refit(learner.FitByExistingTree(second.get(), gradients.RawData(), hessians.RawData()));
  record(*direct_refit);
  // A config reset before smoothed refit must consume the pending assignment before launch geometry/scratch
  // can change. Training itself used path_smooth=0; no unsupported smoothing controller engagement is claimed.
  std::unique_ptr<Tree> third(learner.Train(gradients.RawData(), hessians.RawData(), false));
  Require(third->num_leaves() == 6, "pre-reset complete tree lost final-level coverage");
  record(*third);
  smooth_config.path_smooth = 2.0;
  smooth_config.num_leaves = 19;
  learner.ResetConfig(&smooth_config);
  std::unique_ptr<Tree> smoothed(learner.FitByExistingTree(third.get(), leaf_pred(*third),
                                                        gradients.RawData(), hessians.RawData()));
  record(*smoothed);
  std::fprintf(stderr, "[map-only-refit-probe] arm=%s end\n", defer ? "deferred" : "eager");
  std::fflush(stderr);
  return result;
}
}  // namespace

int main() {
  DatasetHandle handle = nullptr;
  try {
    constexpr int kRows = 32771;
    std::vector<float> x(3 * kRows), labels(kRows);
    std::vector<Falcata::score_t> gradients(kRows);
    for (int row = 0; row < kRows; ++row) {
      for (int feature = 0; feature < 3; ++feature) x[3 * row + feature] = (row >> feature) & 1;
      labels[row] = 1 + 7 * x[3 * row] + 3 * x[3 * row + 1] + 2 * x[3 * row] * x[3 * row + 2];
      gradients[row] = -labels[row];
    }
    Require(FLC_DatasetCreateFromMat(x.data(), C_API_DTYPE_FLOAT32, kRows, 3, 1,
      "device_type=cuda max_bin=7 min_data_in_leaf=2 feature_pre_filter=false num_threads=32", nullptr, &handle) == 0,
      "create CUDA dataset failed");
    Require(FLC_DatasetSetField(handle, "label", labels.data(), kRows, C_API_DTYPE_FLOAT32) == 0, "set labels failed");
    const auto* dataset = reinterpret_cast<Falcata::Dataset*>(handle);
    const auto eager = Run(dataset, x, gradients, false);
    const auto deferred = Run(dataset, x, gradients, true);
    Require(eager.size() == 6 && eager == deferred, "train/direct/leaf_pred/reset-smoothed refit model bytes differ");
    FLC_DatasetFree(handle);
    std::puts("PASS: same-learner direct/leaf_pred and reset-smoothed refit match eager partition model bytes");
    return 0;
  } catch (const std::exception& error) {
    if (handle != nullptr) FLC_DatasetFree(handle);
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
