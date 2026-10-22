/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root.
 */
#include <Falcata/dataset.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <vector>

#include <Falcata/cuda/cuda_column_data.hpp>
#include <Falcata/cuda/cuda_metadata.hpp>

namespace {
using Falcata::CUDAColumnData;
using Falcata::CUDAMetadata;
using Falcata::data_size_t;
using Falcata::label_t;

static_assert(!std::is_copy_constructible<CUDAMetadata>::value, "Device metadata must not be shallow-copied");
static_assert(!std::is_move_constructible<CUDAMetadata>::value, "Device metadata has a fixed owner");
static_assert(!std::is_copy_constructible<CUDAColumnData>::value, "Column data owns device resources");
static_assert(!std::is_move_constructible<CUDAColumnData>::value, "Column views have a fixed owner");

template <typename T>
void ExpectDeviceValues(const T* device, const std::vector<T>& expected) {
  ASSERT_NE(device, nullptr);
  std::vector<T> actual(expected.size());
  ASSERT_EQ(cudaMemcpy(actual.data(), device, actual.size() * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(actual, expected);
}

__global__ void ReadWeights(const label_t* weights, label_t* output) {
  output[threadIdx.x] = weights[threadIdx.x];
}

void InitColumns(CUDAColumnData* columns, bool flip) {
  const std::array<uint8_t, 4> bins = {{0, 1, 2, 3}};
  const std::vector<uint8_t> zeros = flip ? std::vector<uint8_t>{1, 0, 1} : std::vector<uint8_t>{0, 1, 0};
  const std::vector<uint8_t> nas = flip ? std::vector<uint8_t>{0, 1, 0} : std::vector<uint8_t>{1, 0, 1};
  columns->Init(3, {bins.data(), bins.data(), bins.data()}, {nullptr, nullptr, nullptr},
                {8, 8, 8}, {3, 3, 3}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
                zeros, nas, nas, zeros, {0, 1, 2});
}

void ExpectFlags(const CUDAColumnData& columns, bool flip) {
  const std::vector<uint8_t> zeros = flip ? std::vector<uint8_t>{1, 0, 1} : std::vector<uint8_t>{0, 1, 0};
  const std::vector<uint8_t> nas = flip ? std::vector<uint8_t>{0, 1, 0} : std::vector<uint8_t>{1, 0, 1};
  ExpectDeviceValues(columns.cuda_feature_missing_is_zero(), zeros);
  ExpectDeviceValues(columns.cuda_feature_missing_is_na(), nas);
  ExpectDeviceValues(columns.cuda_feature_mfb_is_zero(), nas);
  ExpectDeviceValues(columns.cuda_feature_mfb_is_na(), zeros);
  EXPECT_EQ(columns.cuda_feature_missing_is_zero(), columns.cuda_feature_missing_is_zero());
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(columns.feature_missing_is_zero(i), zeros[i]);
    EXPECT_EQ(columns.feature_missing_is_na(i), nas[i]);
    EXPECT_EQ(columns.feature_mfb_is_zero(i), nas[i]);
    EXPECT_EQ(columns.feature_mfb_is_na(i), zeros[i]);
  }
}
}  // namespace

TEST(CUDAMetadata, QuerySnapshotSurvivesInputsAndUpdatesAndClears) {
  CUDAMetadata metadata(0);
  metadata.Init({0, 1, 2, 3}, {1, 2, 3, 4}, {0, 2, 4}, {1.5f, 3.5f}, {10, 20, 30, 40});
  ExpectDeviceValues(metadata.cuda_label(), std::vector<label_t>{0, 1, 2, 3});
  ExpectDeviceValues(metadata.cuda_weights(), std::vector<label_t>{1, 2, 3, 4});
  ExpectDeviceValues(metadata.cuda_query_boundaries(), std::vector<data_size_t>{0, 2, 4});
  const label_t* first = metadata.cuda_query_weights();
  ExpectDeviceValues(first, std::vector<label_t>{1.5f, 3.5f});
  EXPECT_EQ(first, metadata.cuda_query_weights());

  const data_size_t boundaries[] = {0, 1, 4};
  {
    std::vector<label_t> temporary{7, 9};
    metadata.SetQuery(boundaries, temporary.data(), 2);
    temporary.assign(2, -1);
  }
  ExpectDeviceValues(metadata.cuda_query_weights(), std::vector<label_t>{7, 9});
  metadata.SetQuery(boundaries, nullptr, 2);
  EXPECT_EQ(nullptr, metadata.cuda_query_weights());
  ExpectDeviceValues(metadata.cuda_query_boundaries(), std::vector<data_size_t>{0, 1, 4});
  metadata.SetQuery(nullptr, nullptr, 0);
  EXPECT_EQ(nullptr, metadata.cuda_query_boundaries());
  EXPECT_EQ(nullptr, metadata.cuda_query_weights());

  metadata.Init({1, 0}, {}, {0, 2}, {5}, {});
  ExpectDeviceValues(metadata.cuda_query_weights(), std::vector<label_t>{5});
  metadata.Init({}, {}, {}, {}, {});
  EXPECT_EQ(nullptr, metadata.cuda_label());
  EXPECT_EQ(nullptr, metadata.cuda_weights());
  EXPECT_EQ(nullptr, metadata.cuda_query_weights());
}

TEST(CUDAMetadata, ConcurrentFirstGetPublishesOneCompleteUpload) {
  CUDAMetadata metadata(0);
  metadata.Init({0, 1, 2, 3}, {}, {0, 1, 2, 3, 4}, {2, 4, 6, 8}, {});
  std::array<const label_t*, 8> pointers{};
  std::vector<std::thread> threads;
  for (size_t i = 0; i < pointers.size(); ++i) {
    threads.emplace_back([&metadata, &pointers, i]() { pointers[i] = metadata.cuda_query_weights(); });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto* pointer : pointers) {
    EXPECT_EQ(pointer, pointers[0]);
  }

  // A caller can use the first result on a nonblocking stream without knowing
  // which thread uploaded it or adding an event from that thread's stream.
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  label_t* output = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&output), 4 * sizeof(label_t)), cudaSuccess);
  ReadWeights<<<1, 4, 0, stream>>>(pointers[0], output);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ExpectDeviceValues(output, std::vector<label_t>{2, 4, 6, 8});
  EXPECT_EQ(cudaFree(output), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(CUDAMetadata, RankingSubsetKeepsOwnedQueryWeightsAndHostInitScores) {
  Falcata::Metadata full;
  full.Init(6, -1, -1);
  const label_t labels[] = {0, 1, 2, 0, 1, 2};
  const label_t weights[] = {1, 3, 2, 4, 6, 8};
  const double scores[] = {10, 20, 30, 40, 50, 60};
  const data_size_t groups[] = {2, 1, 3};
  full.SetLabel(labels, 6);
  full.SetWeights(weights, 6);
  full.SetInitScore(scores, 6);
  full.SetQuery(groups, 3);
  full.CreateCUDAMetadata(0);
  ExpectDeviceValues(full.cuda_metadata()->cuda_query_weights(), std::vector<label_t>{2, 2, 6});

  const data_size_t indices[] = {0, 1, 3, 4, 5};
  Falcata::Metadata subset;
  subset.Init(full, indices, 5);
  subset.CreateCUDAMetadata(0);
  ExpectDeviceValues(subset.cuda_metadata()->cuda_query_boundaries(), std::vector<data_size_t>{0, 2, 5});
  ExpectDeviceValues(subset.cuda_metadata()->cuda_query_weights(), std::vector<label_t>{2, 6});
  const double replacement_scores[] = {-1, -2, -3, -4, -5};
  subset.SetInitScore(replacement_scores, 5);
  EXPECT_EQ(std::vector<double>(subset.init_score(), subset.init_score() + 5),
            (std::vector<double>{-1, -2, -3, -4, -5}));
  // The compatibility setter has no getter or consumer; host Metadata remains
  // the score updater's source of truth even after it has CUDA metadata.
  subset.cuda_metadata()->SetInitScore(nullptr, 0);
  EXPECT_EQ(subset.init_score()[4], -5);
}

TEST(CUDAMetadata, LazyAccessRestoresCallerDevice) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "Requires two CUDA devices";
  }
  {
    CUDAMetadata metadata(0);
    metadata.Init({}, {}, {}, {3, 7}, {});
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    const auto* pointer = metadata.cuda_query_weights();
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, pointer), cudaSuccess);
    EXPECT_EQ(attributes.device, 0);
    int current = -1;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, 1);
    EXPECT_EQ(pointer, metadata.cuda_query_weights());
    // Cache destruction also frees on device 0 and restores device 1.
  }
  int current = -1;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 1);
  EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
}

TEST(CUDAMetadata, MissingFlagsRemainCorrectInReusedSubsets) {
  CUDAColumnData source(4, 0);
  InitColumns(&source, false);
  const data_size_t indices[] = {0, 3};
  CUDAColumnData subset(2, 0);
  subset.CopySubrow(&source, indices, 2);
  ExpectFlags(source, false);
  ExpectFlags(subset, false);

  CUDAColumnData changed(4, 0);
  InitColumns(&changed, true);
  subset.CopySubrow(&changed, indices, 2);
  ExpectFlags(subset, true);
  ExpectFlags(source, false);
}
