/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/c_api.h>
#include <Falcata/config.h>
#include <Falcata/dataset.h>
#include <Falcata/tree.h>
#include <Falcata/tree_learner.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Falcata {
namespace {

// Init rejects an invalid host config before reading the Dataset, including
// for parallel and linear wrappers. These tests need no network or GPU device.
TEST(HostQuantization, FactoryChecksSelectedBackendForEveryFamily) {
  for (const std::string device : {"cpu", "gpu"}) {
    for (const std::string family : {"serial", "feature", "data", "voting"}) {
      for (bool linear : {false, true}) {
        SCOPED_TRACE(device + "/" + family + (linear ? "/linear" : ""));
        Config config;
        config.device_type = "cuda";
        config.use_quantized_grad = true;
        config.linear_tree = linear;
        EXPECT_THROW({
          std::unique_ptr<TreeLearner> learner(
              TreeLearner::CreateTreeLearner(family, device, &config, false));
        }, std::runtime_error);
      }
    }
  }
}

TEST(HostQuantization, OpenCLConflictResolutionKeepsFloatTrainingAdmitted) {
  Config config({{"device_type", "gpu"}, {"quant_mode", "stochastic"}, {"verbosity", "-1"}});
  EXPECT_EQ("gpu", config.device_type);
  EXPECT_FALSE(config.use_quantized_grad);
#ifdef USE_GPU
  std::unique_ptr<TreeLearner> learner(TreeLearner::CreateTreeLearner("serial", "gpu", &config, false));
  EXPECT_NE(nullptr, learner);
#endif
}

TEST(HostQuantization, RejectsHostInitAndConfigResetBeforeAccessingTrainingState) {
  for (const std::string family : {"serial", "feature", "data", "voting"}) {
    for (bool linear : {false, true}) {
      SCOPED_TRACE(family + (linear ? "/linear" : ""));
      Config config;
      config.device_type = "cpu";
      config.linear_tree = linear;
      std::unique_ptr<TreeLearner> learner(TreeLearner::CreateTreeLearner(family, "cpu", &config, false));
      ASSERT_NE(nullptr, learner);
      Config invalid = config;
      invalid.use_quantized_grad = true;
      EXPECT_THROW(learner->ResetConfig(&invalid), std::runtime_error);
      config.use_quantized_grad = true;
      EXPECT_THROW(learner->Init(nullptr, false), std::runtime_error);
    }
  }
}

class HostQuantizationLifecycle : public ::testing::Test {
 protected:
  void SetUp() override {
    constexpr int rows = 128;
    std::vector<double> values(rows);
    for (int row = 0; row < rows; ++row) {
      values[row] = row % 17;
    }
    ASSERT_EQ(0, FLC_DatasetCreateFromMat(values.data(), C_API_DTYPE_FLOAT64, rows, 1, 1,
        "device_type=cpu quant_mode=none linear_tree=true min_data_in_leaf=2 min_data_in_bin=1 num_threads=2",
        nullptr, &dataset_));
  }

  void TearDown() override {
    if (dataset_ != nullptr) {
      EXPECT_EQ(0, FLC_DatasetFree(dataset_));
    }
  }

  Config FloatConfig(const std::string& family, bool linear) {
    Config config({{"device_type", "cpu"}, {"quant_mode", "none"}, {"num_threads", "2"},
                   {"min_data_in_leaf", "2"}, {"num_leaves", "4"}, {"force_col_wise", "true"}});
    config.tree_learner = family;
    config.linear_tree = linear;
    return config;
  }

  DatasetHandle dataset_ = nullptr;
};

TEST_F(HostQuantizationLifecycle, RejectsInPlaceMutationBeforeTrainingOrDatasetReset) {
  for (const std::string family : {"serial", "feature", "data", "voting"}) {
    for (bool linear : {false, true}) {
      for (bool reset_data : {false, true}) {
        SCOPED_TRACE(family + (linear ? "/linear" : "") + (reset_data ? "/reset" : "/train"));
        Config config = FloatConfig(family, linear);
        std::unique_ptr<TreeLearner> learner(TreeLearner::CreateTreeLearner(family, "cpu", &config, false));
        learner->Init(static_cast<Dataset*>(dataset_), false);
        config.use_quantized_grad = true;
        if (reset_data) {
          EXPECT_THROW(learner->ResetTrainingData(static_cast<Dataset*>(dataset_), false), std::runtime_error);
        } else {
          std::vector<score_t> gradients(128, 1.0f), hessians(128, 1.0f);
          EXPECT_THROW({
            std::unique_ptr<Tree> tree(learner->Train(gradients.data(), hessians.data(), true));
          }, std::runtime_error);
        }
      }
    }
  }
}

TEST_F(HostQuantizationLifecycle, RejectedConfigResetLeavesFloatLearnerUsable) {
  Config config = FloatConfig("serial", false);
  std::unique_ptr<TreeLearner> learner(TreeLearner::CreateTreeLearner("serial", "cpu", &config, false));
  learner->Init(static_cast<Dataset*>(dataset_), false);
  Config invalid = config;
  invalid.use_quantized_grad = true;
  EXPECT_THROW(learner->ResetConfig(&invalid), std::runtime_error);
  std::vector<score_t> gradients(128), hessians(128, 1.0f);
  for (int row = 0; row < 128; ++row) {
    gradients[row] = static_cast<score_t>(row % 17);
  }
  std::unique_ptr<Tree> tree(learner->Train(gradients.data(), hessians.data(), true));
  ASSERT_NE(nullptr, tree);
  EXPECT_GT(tree->num_leaves(), 1);
}

}  // namespace
}  // namespace Falcata
