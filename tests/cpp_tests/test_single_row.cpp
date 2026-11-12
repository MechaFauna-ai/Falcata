/*!
 * Copyright (c) 2022-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <testutils.h>
#include <Falcata/c_api.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

#include <Falcata/utils/yamc/alternate_shared_mutex.hpp>
#include <Falcata/utils/yamc/yamc_shared_lock.hpp>

using Falcata::TestUtils;

void test_predict_type(int predict_type, int num_predicts) {
    // Load some test data
    int result;

    DatasetHandle train_dataset;
    result = TestUtils::LoadDatasetFromExamples("binary_classification/binary.train", "max_bin=15", &train_dataset);
    EXPECT_EQ(0, result) << "LoadDatasetFromExamples train result code: " << result;

    BoosterHandle booster_handle;
    result = FLC_BoosterCreate(train_dataset, "app=binary metric=auc num_leaves=31 verbose=0", &booster_handle);
    EXPECT_EQ(0, result) << "FLC_BoosterCreate result code: " << result;

    for (int i = 0; i < 51; i++) {
        int produced_empty_tree;
        result = FLC_BoosterUpdateOneIter(
            booster_handle,
            &produced_empty_tree);
        EXPECT_EQ(0, result) << "FLC_BoosterUpdateOneIter result code: " << result;
    }

    int n_features;
    result = FLC_BoosterGetNumFeature(
        booster_handle,
        &n_features);
    EXPECT_EQ(0, result) << "FLC_BoosterGetNumFeature result code: " << result;
    EXPECT_EQ(28, n_features) << "FLC_BoosterGetNumFeature number of features: " << n_features;

    // Run a single row prediction and compare with regular Mat prediction:
    int64_t output_size;
    result = FLC_BoosterCalcNumPredict(
        booster_handle,
        1,
        predict_type,          // predict_type
        0,                     // start_iteration
        -1,                    // num_iteration
        &output_size);
    EXPECT_EQ(0, result) << "FLC_BoosterCalcNumPredict result code: " << result;
    EXPECT_EQ(num_predicts, output_size) << "FLC_BoosterCalcNumPredict output size: " << output_size;

    std::ifstream test_file("examples/binary_classification/binary.test");
    std::vector<double> test;
    double x;
    int test_set_size = 0;
    while (test_file >> x) {
        if (test_set_size % (n_features + 1) == 0) {
            // Drop the result from the dataset, we only care about checking that prediction results are equal
            // in both cases
            test_file >> x;
            test_set_size++;
        }
        test.push_back(x);
        test_set_size++;
    }
    EXPECT_EQ(test_set_size % (n_features + 1), 0) << "Test size mismatch with dataset size (%)";
    test_set_size /= (n_features + 1);
    EXPECT_EQ(test_set_size, 500) << "Improperly parsed test file (test_set_size)";
    EXPECT_EQ(test.size(), test_set_size * n_features) << "Improperly parsed test file (test len)";

    std::vector<double> mat_output(output_size * test_set_size, -1);
    int64_t written;
    result = FLC_BoosterPredictForMat(
        booster_handle,
        &test[0],
        C_API_DTYPE_FLOAT64,
        test_set_size,         // nrow
        n_features,            // ncol
        1,                     // is_row_major
        predict_type,          // predict_type
        0,                     // start_iteration
        -1,                    // num_iteration
        "",
        &written,
        &mat_output[0]);
    EXPECT_EQ(0, result) << "FLC_BoosterPredictForMat result code: " << result;

    // Test FLC_BoosterPredictForMat in multi-threaded mode
    const int kNThreads = 10;
    const int numIterations = 5;
    std::vector<std::thread> predict_for_mat_threads(kNThreads);
    for (int i = 0; i < kNThreads; i++) {
        predict_for_mat_threads[i] = std::thread(
            [
                test_set_size, output_size, n_features,
                    test = &test[0], booster_handle, predict_type
            ]() {
                for (int j = 0; j < numIterations; j++) {
                    int result;
                    std::vector<double> mat_output(output_size * test_set_size, -1);
                    int64_t written;
                    result = FLC_BoosterPredictForMat(
                        booster_handle,
                        &test[0],
                        C_API_DTYPE_FLOAT64,
                        test_set_size,         // nrow
                        n_features,            // ncol
                        1,                     // is_row_major
                        predict_type,          // predict_type
                        0,                     // start_iteration
                        -1,                    // num_iteration
                        "",
                        &written,
                        &mat_output[0]);
                    EXPECT_EQ(0, result) << "FLC_BoosterPredictForMat result code: " << result;
                }
            });
    }
    for (std::thread& t : predict_for_mat_threads) {
        t.join();
    }

    // Now let's run with the single row fast prediction API:
    FastConfigHandle fast_configs[kNThreads];
    for (int i = 0; i < kNThreads; i++) {
        result = FLC_BoosterPredictForMatSingleRowFastInit(
            booster_handle,
            predict_type,          // predict_type
            0,                     // start_iteration
            -1,                    // num_iteration
            C_API_DTYPE_FLOAT64,
            n_features,
            "",
            &fast_configs[i]);
        EXPECT_EQ(0, result) << "FLC_BoosterPredictForMatSingleRowFastInit result code: " << result;
    }

    std::vector<double> single_row_output(output_size * test_set_size, -1);
    std::vector<std::thread> single_row_threads(kNThreads);
    int batch_size = (test_set_size + kNThreads - 1) / kNThreads;  // round up
    for (int i = 0; i < kNThreads; i++) {
        single_row_threads[i] = std::thread(
            [
                i, batch_size, test_set_size, output_size, n_features,
                    test = &test[0], fast_configs = &fast_configs[0], single_row_output = &single_row_output[0]
            ]() {
                int result;
                int64_t written;
                for (int j = i * batch_size; j < std::min((i + 1) * batch_size, test_set_size); j++) {
                    result = FLC_BoosterPredictForMatSingleRowFast(
                        fast_configs[i],
                        &test[j * n_features],
                        &written,
                        &single_row_output[j * output_size]);
                    EXPECT_EQ(0, result) << "FLC_BoosterPredictForMatSingleRowFast result code: " << result;
                    EXPECT_EQ(written, output_size) << "FLC_BoosterPredictForMatSingleRowFast unexpected written output size";
                }
            });
      }
    for (std::thread& t : single_row_threads) {
        t.join();
    }

    EXPECT_EQ(single_row_output, mat_output) << "FLC_BoosterPredictForMatSingleRowFast output mismatch with FLC_BoosterPredictForMat";

    // Free all:
    for (int i = 0; i < kNThreads; i++) {
        result = FLC_FastConfigFree(fast_configs[i]);
        EXPECT_EQ(0, result) << "FLC_FastConfigFree result code: " << result;
    }

    result = FLC_BoosterFree(booster_handle);
    EXPECT_EQ(0, result) << "FLC_BoosterFree result code: " << result;

    result = FLC_DatasetFree(train_dataset);
    EXPECT_EQ(0, result) << "FLC_DatasetFree result code: " << result;
}

TEST(SingleRow, Normal) {
    test_predict_type(C_API_PREDICT_NORMAL, 1);
}

TEST(SingleRow, Contrib) {
    test_predict_type(C_API_PREDICT_CONTRIB, 29);
}

// Regression test for issue #7220: FLC_BoosterPredictForMatSingleRow
// reused a cached single-row predictor without taking `start_iteration`
// into account, so consecutive calls with different start_iteration values
// but the same num_iteration returned stale (identical) predictions.
TEST(SingleRow, StartIterationChangesPrediction) {
    int result;

    DatasetHandle train_dataset;
    result = TestUtils::LoadDatasetFromExamples("binary_classification/binary.train", "max_bin=15", &train_dataset);
    EXPECT_EQ(0, result) << "LoadDatasetFromExamples train result code: " << result;

    BoosterHandle booster_handle;
    result = FLC_BoosterCreate(train_dataset, "app=binary metric=auc num_leaves=31 verbose=0", &booster_handle);
    EXPECT_EQ(0, result) << "FLC_BoosterCreate result code: " << result;

    const int kNumIterTrained = 30;
    for (int i = 0; i < kNumIterTrained; i++) {
        int produced_empty_tree;
        result = FLC_BoosterUpdateOneIter(booster_handle, &produced_empty_tree);
        EXPECT_EQ(0, result) << "FLC_BoosterUpdateOneIter result code: " << result;
    }

    int n_features;
    result = FLC_BoosterGetNumFeature(booster_handle, &n_features);
    EXPECT_EQ(0, result) << "FLC_BoosterGetNumFeature result code: " << result;

    // Load a single test row.
    std::ifstream test_file("examples/binary_classification/binary.test");
    std::vector<double> single_row;
    double x;
    int field = 0;
    while (test_file >> x) {
        if (field == 0) {
            // Drop the label column.
            field++;
            continue;
        }
        single_row.push_back(x);
        field++;
        if (static_cast<int>(single_row.size()) == n_features) {
            break;
        }
    }
    EXPECT_EQ(static_cast<int>(single_row.size()), n_features) << "Failed to parse a single row from the test file";

    int64_t output_size;
    result = FLC_BoosterCalcNumPredict(
        booster_handle,
        1,
        C_API_PREDICT_NORMAL,
        0,
        10,
        &output_size);
    EXPECT_EQ(0, result) << "FLC_BoosterCalcNumPredict result code: " << result;

    // Call FLC_BoosterPredictForMatSingleRow twice with the same num_iteration
    // but different start_iteration. The two boosting iteration windows are
    // disjoint, so the resulting scores must differ.
    std::vector<double> pred_early(output_size, -1);
    std::vector<double> pred_late(output_size, -1);
    int64_t written;

    result = FLC_BoosterPredictForMatSingleRow(
        booster_handle,
        &single_row[0],
        C_API_DTYPE_FLOAT64,
        n_features,
        1,                     // is_row_major
        C_API_PREDICT_NORMAL,
        0,                     // start_iteration
        10,                    // num_iteration
        "",
        &written,
        &pred_early[0]);
    EXPECT_EQ(0, result) << "FLC_BoosterPredictForMatSingleRow (start=0) result code: " << result;

    result = FLC_BoosterPredictForMatSingleRow(
        booster_handle,
        &single_row[0],
        C_API_DTYPE_FLOAT64,
        n_features,
        1,                     // is_row_major
        C_API_PREDICT_NORMAL,
        20,                    // start_iteration — window [20, 30) disjoint from [0, 10)
        10,                    // num_iteration
        "",
        &written,
        &pred_late[0]);
    EXPECT_EQ(0, result) << "FLC_BoosterPredictForMatSingleRow (start=20) result code: " << result;

    EXPECT_NE(pred_early[0], pred_late[0])
        << "FLC_BoosterPredictForMatSingleRow returned identical predictions for "
           "disjoint iteration windows [0,10) and [20,30); start_iteration is being "
           "ignored by the predictor cache key.";

    // Cross-check against FLC_BoosterPredictForMat (the non-single-row path),
    // which does not use the buggy cache and is known to respect start_iteration.
    std::vector<double> ref_early(output_size, -1);
    std::vector<double> ref_late(output_size, -1);
    result = FLC_BoosterPredictForMat(
        booster_handle,
        &single_row[0],
        C_API_DTYPE_FLOAT64,
        1,                     // nrow
        n_features,
        1,                     // is_row_major
        C_API_PREDICT_NORMAL,
        0,                     // start_iteration
        10,                    // num_iteration
        "",
        &written,
        &ref_early[0]);
    EXPECT_EQ(0, result) << "FLC_BoosterPredictForMat (start=0) result code: " << result;

    result = FLC_BoosterPredictForMat(
        booster_handle,
        &single_row[0],
        C_API_DTYPE_FLOAT64,
        1,
        n_features,
        1,
        C_API_PREDICT_NORMAL,
        20,
        10,
        "",
        &written,
        &ref_late[0]);
    EXPECT_EQ(0, result) << "FLC_BoosterPredictForMat (start=20) result code: " << result;

    EXPECT_DOUBLE_EQ(pred_early[0], ref_early[0])
        << "FLC_BoosterPredictForMatSingleRow (start=0) disagrees with FLC_BoosterPredictForMat";
    EXPECT_DOUBLE_EQ(pred_late[0], ref_late[0])
        << "FLC_BoosterPredictForMatSingleRow (start=20) disagrees with FLC_BoosterPredictForMat";

    result = FLC_BoosterFree(booster_handle);
    EXPECT_EQ(0, result) << "FLC_BoosterFree result code: " << result;

    result = FLC_DatasetFree(train_dataset);
    EXPECT_EQ(0, result) << "FLC_DatasetFree result code: " << result;
}

// FLC_BoosterPredictForCSRSingleRowFastInit must forward predict_type,
// start_iteration and num_iteration in that order; the CSR fast path must
// agree with FLC_BoosterPredictForMat for non-default values of all three.
void test_csr_fast_matches_mat(int predict_type, int start_iteration, int num_iteration) {
    int result;

    DatasetHandle train_dataset;
    result = TestUtils::LoadDatasetFromExamples("binary_classification/binary.train", "max_bin=15", &train_dataset);
    ASSERT_EQ(0, result) << "LoadDatasetFromExamples train result code: " << result;

    BoosterHandle booster_handle;
    result = FLC_BoosterCreate(train_dataset, "app=binary metric=auc num_leaves=31 verbose=0", &booster_handle);
    ASSERT_EQ(0, result) << "FLC_BoosterCreate result code: " << result;

    for (int i = 0; i < 30; i++) {
        int produced_empty_tree;
        result = FLC_BoosterUpdateOneIter(booster_handle, &produced_empty_tree);
        ASSERT_EQ(0, result) << "FLC_BoosterUpdateOneIter result code: " << result;
    }

    int n_features;
    result = FLC_BoosterGetNumFeature(booster_handle, &n_features);
    ASSERT_EQ(0, result) << "FLC_BoosterGetNumFeature result code: " << result;

    std::ifstream test_file("examples/binary_classification/binary.test");
    std::vector<double> rows;
    double x;
    int field = 0;
    const int kNumRows = 20;
    while (test_file >> x && static_cast<int>(rows.size()) < kNumRows * n_features) {
        if (field % (n_features + 1) != 0) {
            rows.push_back(x);
        }
        field++;
    }
    ASSERT_EQ(static_cast<int>(rows.size()), kNumRows * n_features) << "Failed to parse test rows";

    int64_t output_size;
    result = FLC_BoosterCalcNumPredict(booster_handle, 1, predict_type, start_iteration, num_iteration, &output_size);
    ASSERT_EQ(0, result) << "FLC_BoosterCalcNumPredict result code: " << result;

    std::vector<double> mat_output(output_size * kNumRows, -1);
    int64_t written;
    result = FLC_BoosterPredictForMat(
        booster_handle, &rows[0], C_API_DTYPE_FLOAT64, kNumRows, n_features, 1,
        predict_type, start_iteration, num_iteration, "", &written, &mat_output[0]);
    ASSERT_EQ(0, result) << "FLC_BoosterPredictForMat result code: " << result;

    FastConfigHandle fast_config;
    result = FLC_BoosterPredictForCSRSingleRowFastInit(
        booster_handle, predict_type, start_iteration, num_iteration,
        C_API_DTYPE_FLOAT64, n_features, "", &fast_config);
    ASSERT_EQ(0, result) << "FLC_BoosterPredictForCSRSingleRowFastInit result code: " << result;

    std::vector<int32_t> indptr = {0, n_features};
    std::vector<int32_t> indices(n_features);
    for (int j = 0; j < n_features; ++j) {
        indices[j] = j;
    }
    std::vector<double> csr_output(output_size * kNumRows, -1);
    for (int r = 0; r < kNumRows; ++r) {
        result = FLC_BoosterPredictForCSRSingleRowFast(
            fast_config, &indptr[0], C_API_DTYPE_INT32, &indices[0], &rows[r * n_features],
            2, n_features, &written, &csr_output[r * output_size]);
        ASSERT_EQ(0, result) << "FLC_BoosterPredictForCSRSingleRowFast result code: " << result;
        ASSERT_EQ(written, output_size) << "FLC_BoosterPredictForCSRSingleRowFast unexpected written output size";
    }

    EXPECT_EQ(csr_output, mat_output)
        << "FLC_BoosterPredictForCSRSingleRowFast disagrees with FLC_BoosterPredictForMat for predict_type="
        << predict_type << " start_iteration=" << start_iteration << " num_iteration=" << num_iteration;

    EXPECT_EQ(0, FLC_FastConfigFree(fast_config));
    EXPECT_EQ(0, FLC_BoosterFree(booster_handle));
    EXPECT_EQ(0, FLC_DatasetFree(train_dataset));
}

TEST(SingleRow, CSRFastRawScoreIterationWindow) {
    test_csr_fast_matches_mat(C_API_PREDICT_RAW_SCORE, 5, 10);
}

TEST(SingleRow, CSRFastLeafIndex) {
    test_csr_fast_matches_mat(C_API_PREDICT_LEAF_INDEX, 0, -1);
}

TEST(SingleRow, CSRFastContribIterationWindow) {
    test_csr_fast_matches_mat(C_API_PREDICT_CONTRIB, 3, 7);
}

// A Predictor (FastConfig, or one cached by the C API) used to read its iteration range from state
// shared by every prediction on the booster: a later prediction with another num_iteration changed the
// range of a FastConfig made earlier, and two concurrent predictions with different ranges sized their
// buffers for one range and filled them for the other. Each predictor now keeps its own range.
TEST(SingleRow, FastConfigKeepsItsRangeAfterOtherPredictions) {
    const int num_row = 200, num_col = 3;
    std::vector<double> X(num_row * num_col);
    std::vector<float> y(num_row);
    for (int i = 0; i < num_row; ++i) {
        for (int j = 0; j < num_col; ++j) {
            X[i * num_col + j] = std::sin(0.37 * i + 1.3 * j);
        }
        y[i] = static_cast<float>(X[i * num_col] * 2.0);
    }
    DatasetHandle dataset;
    ASSERT_EQ(0, FLC_DatasetCreateFromMat(X.data(), C_API_DTYPE_FLOAT64, num_row, num_col, 1,
                                          "verbose=-1 min_data_in_bin=1", nullptr, &dataset));
    ASSERT_EQ(0, FLC_DatasetSetField(dataset, "label", y.data(), num_row, C_API_DTYPE_FLOAT32));
    BoosterHandle booster;
    ASSERT_EQ(0, FLC_BoosterCreate(dataset, "objective=regression verbose=-1 num_threads=1", &booster));
    int is_finished;
    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(0, FLC_BoosterUpdateOneIter(booster, &is_finished));
    }

    for (int predict_type : {C_API_PREDICT_RAW_SCORE, C_API_PREDICT_LEAF_INDEX, C_API_PREDICT_CONTRIB}) {
        FastConfigHandle fast;
        ASSERT_EQ(0, FLC_BoosterPredictForMatSingleRowFastInit(booster, predict_type, 0, -1, C_API_DTYPE_FLOAT64,
                                                               num_col, "", &fast));
        int64_t len = 0;
        std::vector<double> before(64, 0.0), after(64, 0.0), other(64 * num_row, 0.0);
        ASSERT_EQ(0, FLC_BoosterPredictForMatSingleRowFast(fast, X.data(), &len, before.data()));
        int64_t full_len = len;
        // another prediction on the same booster, over the first iteration only
        ASSERT_EQ(0, FLC_BoosterPredictForMat(booster, X.data(), C_API_DTYPE_FLOAT64, 5, num_col, 1, predict_type,
                                              0, 1, "", &len, other.data()));
        ASSERT_EQ(0, FLC_BoosterPredictForMatSingleRowFast(fast, X.data(), &len, after.data()));
        EXPECT_EQ(full_len, len) << "predict_type " << predict_type;
        for (int64_t k = 0; k < full_len; ++k) {
            EXPECT_EQ(before[k], after[k]) << "predict_type " << predict_type << " output " << k;
        }
        FLC_FastConfigFree(fast);
    }
    FLC_BoosterFree(booster);
    FLC_DatasetFree(dataset);
}

// The vendored yamc mutexes only compiled for the members c_api.cpp happens
// to use: try_lock/try_lock_shared and the timed waits passed the policy
// state by value to functions taking a pointer, the WriterPrefer policy's
// before/after_wait_wlock took a pointer every caller passed by value, and
// shared_lock's defer/try/adopt/timed constructors stored &m of a const
// reference in a non-const pointer. Instantiate all of them.
namespace {
template <class Mutex>
void ExerciseSharedMutex() {
  Mutex m;
  m.lock();
  EXPECT_FALSE(m.try_lock_shared());
  m.unlock();
  ASSERT_TRUE(m.try_lock());
  m.unlock();
  ASSERT_TRUE(m.try_lock_shared());
  EXPECT_FALSE(m.try_lock());
  m.unlock_shared();
  {
    yamc::shared_lock<Mutex> lock(m, std::defer_lock);
    EXPECT_FALSE(lock.owns_lock());
    lock.lock();
    EXPECT_TRUE(lock.owns_lock());
  }
  {
    yamc::shared_lock<Mutex> lock(m, std::try_to_lock);
    EXPECT_TRUE(lock.owns_lock());
  }
  m.lock_shared();
  yamc::shared_lock<Mutex> adopted(m, std::adopt_lock);
  EXPECT_TRUE(adopted.owns_lock());
}

template <class Mutex>
void ExerciseSharedTimedMutex() {
  ExerciseSharedMutex<Mutex>();
  Mutex m;
  ASSERT_TRUE(m.try_lock_for(std::chrono::milliseconds(1)));
  m.unlock();
  ASSERT_TRUE(m.try_lock_shared_until(std::chrono::steady_clock::now() + std::chrono::milliseconds(1)));
  m.unlock_shared();
  yamc::shared_lock<Mutex> lock(m, std::chrono::milliseconds(1));
  EXPECT_TRUE(lock.owns_lock());
}
}  // namespace

TEST(YamcSharedMutex, AllMembersInstantiate) {
  ExerciseSharedMutex<yamc::alternate::shared_mutex>();
  ExerciseSharedMutex<yamc::alternate::basic_shared_mutex<yamc::rwlock::ReaderPrefer>>();
  ExerciseSharedMutex<yamc::alternate::basic_shared_mutex<yamc::rwlock::WriterPrefer>>();
  ExerciseSharedTimedMutex<yamc::alternate::basic_shared_timed_mutex<yamc::rwlock::ReaderPrefer>>();
  ExerciseSharedTimedMutex<yamc::alternate::basic_shared_timed_mutex<yamc::rwlock::WriterPrefer>>();
}
