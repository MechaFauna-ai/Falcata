/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/c_api.h>
#include <Falcata/utils/openmp_wrapper.h>

#include <cmath>
#include <vector>

// Predictor sized its per-thread feature buffers with OMP_NUM_THREADS() when it
// was built (num_threads=1 for this FastConfig) but indexed them with
// omp_get_thread_num() at call time. A caller predicting single rows from its
// own OpenMP parallel region, a common way to use the fast single-row API,
// indexed past the buffers.
TEST(PredictorThreads, SingleRowFastFromCallerParallelRegion) {
  const int nrow = 400, ncol = 6;
  std::vector<double> X(nrow * ncol);
  std::vector<float> y(nrow);
  for (int i = 0; i < nrow; ++i) {
    for (int j = 0; j < ncol; ++j) {
      X[i * ncol + j] = std::sin(0.37 * i + 1.3 * j) + 0.1 * j;
    }
    y[i] = static_cast<float>(X[i * ncol] + 0.5 * X[i * ncol + 1] > 0.3);
  }
  DatasetHandle dataset = nullptr;
  ASSERT_EQ(0, FLC_DatasetCreateFromMat(X.data(), C_API_DTYPE_FLOAT64, nrow, ncol, 1, "verbosity=-1 min_data_in_bin=1",
                                        nullptr, &dataset));
  ASSERT_EQ(0, FLC_DatasetSetField(dataset, "label", y.data(), nrow, C_API_DTYPE_FLOAT32));
  BoosterHandle booster = nullptr;
  ASSERT_EQ(0, FLC_BoosterCreate(dataset, "objective=binary verbosity=-1 num_leaves=7 min_data_in_leaf=5", &booster));
  for (int it = 0; it < 5; ++it) {
    int finished = 0;
    ASSERT_EQ(0, FLC_BoosterUpdateOneIter(booster, &finished));
  }

  std::vector<double> expected(nrow);
  int64_t out_len = 0;
  ASSERT_EQ(0, FLC_BoosterPredictForMat(booster, X.data(), C_API_DTYPE_FLOAT64, nrow, ncol, 1, C_API_PREDICT_NORMAL,
                                        0, -1, "num_threads=1", &out_len, expected.data()));

  for (const int predict_type : {C_API_PREDICT_NORMAL, C_API_PREDICT_RAW_SCORE, C_API_PREDICT_LEAF_INDEX}) {
    FastConfigHandle fast_config = nullptr;
    ASSERT_EQ(0, FLC_BoosterPredictForMatSingleRowFastInit(booster, predict_type, 0, -1, C_API_DTYPE_FLOAT64, ncol,
                                                           "num_threads=1", &fast_config));
    std::vector<double> got(nrow * 8, 0.0);
    std::vector<int> rc(nrow, -1);
#pragma omp parallel for num_threads(4) schedule(static, 1)
    for (int i = 0; i < nrow; ++i) {
      int64_t len = 0;
      rc[i] = FLC_BoosterPredictForMatSingleRowFast(fast_config, X.data() + i * ncol, &len, got.data() + i * 8);
    }
    for (int i = 0; i < nrow; ++i) {
      ASSERT_EQ(rc[i], 0) << i;
      if (predict_type == C_API_PREDICT_NORMAL) {
        EXPECT_DOUBLE_EQ(got[i * 8], expected[i]) << i;
      }
    }
    FLC_FastConfigFree(fast_config);
  }
  FLC_BoosterFree(booster);
  FLC_DatasetFree(dataset);
}
