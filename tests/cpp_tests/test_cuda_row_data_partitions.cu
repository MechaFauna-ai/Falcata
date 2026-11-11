/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/c_api.h>
#include <Falcata/dataset.h>
#include <Falcata/train_share_states.h>

#include <memory>
#include <vector>

#include <Falcata/cuda/cuda_row_data.hpp>

// CUDARowData::DivideCUDAFeatureGroups gives a column with more bins than a shared-memory partition holds a partition
// of its own. It used to push only that partition's end, so the columns already open before it were folded into the
// large column's partition, whose local histogram base is 0: their bins landed on top of the large column's first
// bins. Every column's partition base plus its local offset must be its global histogram offset, and the large column
// must be alone in its partition.
TEST(CUDARowData, LargeBinColumnClosesTheOpenPartition) {
  // columns: small, small, large (more distinct values than a partition's bins), small
  const int num_rows = 14000;  // column 2 gets more bins than a single-precision partition (6144)
  const int num_cols = 4;
  std::vector<double> x(static_cast<size_t>(num_rows) * num_cols);
  for (int r = 0; r < num_rows; ++r) {
    x[static_cast<size_t>(r) * num_cols + 0] = r % 10;
    x[static_cast<size_t>(r) * num_cols + 1] = (r * 7) % 12;
    x[static_cast<size_t>(r) * num_cols + 2] = r;
    x[static_cast<size_t>(r) * num_cols + 3] = (r * 3) % 9;
  }
  DatasetHandle handle = nullptr;
  ASSERT_EQ(FLC_DatasetCreateFromMat(x.data(), C_API_DTYPE_FLOAT64, num_rows, num_cols, 1,
                                     "verbosity=-1 max_bin=16000 min_data_in_bin=1 enable_bundle=false", nullptr,
                                     &handle),
            0);
  const auto* dataset = reinterpret_cast<const Falcata::Dataset*>(handle);
  ASSERT_EQ(dataset->num_features(), num_cols);

  const std::vector<int8_t> is_feature_used(num_cols, 1);
  std::unique_ptr<Falcata::TrainingShareStates> share_states(dataset->GetShareStates<false, 0>(
    nullptr, nullptr, is_feature_used, false, /*force_col_wise=*/false, /*force_row_wise=*/true, 0,
    /*is_cuda_tree_learner=*/true));
  const std::vector<uint32_t>& global_offsets = share_states->column_hist_offsets();
  ASSERT_EQ(global_offsets.size(), static_cast<size_t>(num_cols) + 1);

  Falcata::CUDARowData row_data(dataset, share_states.get(), 0, /*gpu_use_dp=*/false);
  row_data.Init(dataset, share_states.get());
  const uint32_t max_bins = static_cast<uint32_t>(row_data.shared_hist_size()) / 2;
  ASSERT_GT(global_offsets[3] - global_offsets[2], max_bins) << "column 2 must be a large-bin column";

  const std::vector<int>& columns = row_data.host_feature_partition_column_index_offsets();
  const std::vector<uint32_t>& bases = row_data.host_partition_hist_offsets();
  const std::vector<uint32_t>& local = row_data.host_column_hist_offsets();
  EXPECT_EQ(columns, (std::vector<int>{0, 2, 3, 4}));
  EXPECT_EQ(row_data.num_feature_partitions(), 3);
  EXPECT_EQ(row_data.NumLargeBinPartition(), 1);
  ASSERT_EQ(bases.size(), columns.size());
  for (size_t p = 0; p + 1 < columns.size(); ++p) {
    const bool large = columns[p + 1] - columns[p] == 1 &&
                       global_offsets[columns[p] + 1] - global_offsets[columns[p]] > max_bins;
    if (!large) {
      EXPECT_LE(bases[p + 1] - bases[p], max_bins) << "partition " << p;
    }
    for (int c = columns[p]; c < columns[p + 1]; ++c) {
      EXPECT_EQ(bases[p] + local[c], global_offsets[c]) << "partition " << p << " column " << c;
    }
  }
  EXPECT_EQ(FLC_DatasetFree(handle), 0);
}
