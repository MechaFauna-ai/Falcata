/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <Falcata/boosting.h>
#include <Falcata/c_api.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// trains a small model through the C API and returns it as model text
std::string TrainModelText(const char* dataset_params, const char* params) {
  const int nrow = 200;
  const int ncol = 3;
  std::vector<double> data(nrow * ncol);
  std::vector<float> label(nrow);
  for (int i = 0; i < nrow; ++i) {
    for (int j = 0; j < ncol; ++j) {
      data[i * ncol + j] = static_cast<double>((i * (j + 3) * 37) % 101) / 10.0;
    }
    label[i] = static_cast<float>(2.0 * data[i * ncol] - data[i * ncol + 1]);
  }
  DatasetHandle ds = nullptr;
  EXPECT_EQ(0, FLC_DatasetCreateFromMat(data.data(), C_API_DTYPE_FLOAT64, nrow, ncol, 1, dataset_params, nullptr, &ds));
  EXPECT_EQ(0, FLC_DatasetSetField(ds, "label", label.data(), nrow, C_API_DTYPE_FLOAT32));
  BoosterHandle booster = nullptr;
  EXPECT_EQ(0, FLC_BoosterCreate(ds, params, &booster));
  int finished = 0;
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(0, FLC_BoosterUpdateOneIter(booster, &finished));
  }
  int64_t len = 0;
  std::vector<char> buf(1 << 20);
  EXPECT_EQ(0, FLC_BoosterSaveModelToString(booster, 0, -1, 0, static_cast<int64_t>(buf.size()), &len, buf.data()));
  EXPECT_EQ(0, FLC_BoosterFree(booster));
  EXPECT_EQ(0, FLC_DatasetFree(ds));
  return std::string(buf.data());
}

std::unique_ptr<Falcata::Boosting> Load(const std::string& model) {
  std::unique_ptr<Falcata::Boosting> boosting(Falcata::Boosting::CreateBoosting("gbdt", nullptr, "cpu", 0));
  EXPECT_TRUE(boosting->LoadModelFromString(model.c_str(), model.size()));
  return boosting;
}

}  // namespace

// The if-else export returns each leaf's constant: a linear tree's per-leaf linear model was dropped, so the
// converted model silently predicted something else. It must refuse linear trees.
TEST(IfElseExport, RejectsLinearTrees) {
  auto boosting = Load(TrainModelText("linear_tree=true verbose=-1", "objective=regression linear_tree=true num_leaves=4 verbose=-1"));
  try {
    boosting->ModelToIfElse(-1);
    FAIL() << "linear trees were exported";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("does not support linear trees"), std::string::npos) << e.what();
  }
}

TEST(IfElseExport, PlainTreesStillExport) {
  auto boosting = Load(TrainModelText("verbose=-1", "objective=regression num_leaves=4 verbose=-1"));
  const std::string code = boosting->ModelToIfElse(-1);
  EXPECT_NE(code.find("PredictTree0"), std::string::npos);
}
