/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_row_data.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace LightGBM {

namespace {

bool FastRowDataEnabled() {
  const char* env = std::getenv("EXABOOST_FAST_ROWDATA");
  return env == nullptr || std::string(env) != std::string("0");
}

bool FastRowDataVerifyEnabled() {
  const char* env = std::getenv("EXABOOST_FAST_ROWDATA_VERIFY");
  return env != nullptr && std::string(env) == std::string("1");
}

bool RowData4BitEnabled() {
  const char* env = std::getenv("EXABOOST_ROWDATA_4BIT");
  return env == nullptr || std::string(env) != std::string("0");
}

bool RowData4BitVerifyEnabled() {
  const char* env = std::getenv("EXABOOST_ROWDATA_4BIT_VERIFY");
  return env != nullptr && std::string(env) == std::string("1");
}

// Shape-specialized construct prototype (JIT phase 1): for low-bin, many-feature
// data (e.g. numerai ~6 bins/feature) the 504-column-per-partition cap forces
// block_dim_y = NUM_THREADS_PER_BLOCK / max_col_per_part = 1, i.e. every column
// thread walks one row at a time with no row-level ILP to hide the scattered bin
// read. Lowering the cap regroups columns into more, narrower partitions so
// block_dim_y rises (504/cap) -- each thread walks that many rows, overlapping
// the dependent bin-read latencies. Integer-atomic accumulation is order-
// invariant, so the histogram sums are BIT-IDENTICAL regardless of the column
// grouping (verified: covtype/numerai/RMSE md5 locks unchanged). Measured ~5-6%
// numerai-example wall / ~8% construct-kernel win; other benchmarks are bin-cap
// bound (few columns per partition already), so the auto-trigger is a no-op there.
//
// EXABOOST_CONSTRUCT_COLCAP: unset -> auto (252 when the low-bin shape is
// column-capped, else upstream 504); "0" -> force upstream 504 (kill switch);
// "N" (0<N<504) -> force cap N. Returns -1 for "auto".
int ConstructColumnCapEnv() {
  const char* env = std::getenv("EXABOOST_CONSTRUCT_COLCAP");
  if (env == nullptr) {
    return -1;  // auto
  }
  const int v = std::atoi(env);
  if (v <= 0 || v >= 504) {
    return 504;  // "0" / out-of-range -> upstream behavior (kill switch)
  }
  return v;
}

// Read one bin value out of a raw host column (bit type 4/8/16/32).
inline uint8_t FetchColumnBin(const void* column_data, uint8_t column_bit_type, data_size_t row) {
  if (column_bit_type == 4) {
    const uint8_t* in = reinterpret_cast<const uint8_t*>(column_data);
    return (in[row >> 1] >> ((row & 1) << 2)) & 0xf;
  } else if (column_bit_type == 8) {
    return reinterpret_cast<const uint8_t*>(column_data)[row];
  } else if (column_bit_type == 16) {
    return static_cast<uint8_t>(reinterpret_cast<const uint16_t*>(column_data)[row]);
  } else {
    return static_cast<uint8_t>(reinterpret_cast<const uint32_t*>(column_data)[row]);
  }
}

// Scatter one column's rows [start, end) into its slot of a row-major partition:
// out_data points at (partition base + local column index), row_stride is the
// number of columns in the partition.
template <typename BIN_TYPE, typename COLUMN_TYPE>
void TransposeColumnTile(const void* column_data, data_size_t start, data_size_t end,
                         int row_stride, BIN_TYPE* out_data) {
  const COLUMN_TYPE* in_data = reinterpret_cast<const COLUMN_TYPE*>(column_data);
  for (data_size_t row = start; row < end; ++row) {
    out_data[static_cast<size_t>(row) * row_stride] = static_cast<BIN_TYPE>(in_data[row]);
  }
}

template <typename BIN_TYPE>
void TransposeColumnTile4Bit(const void* column_data, data_size_t start, data_size_t end,
                             int row_stride, BIN_TYPE* out_data) {
  const uint8_t* in_data = reinterpret_cast<const uint8_t*>(column_data);
  for (data_size_t row = start; row < end; ++row) {
    out_data[static_cast<size_t>(row) * row_stride] =
      static_cast<BIN_TYPE>((in_data[row >> 1] >> ((row & 1) << 2)) & 0xf);
  }
}

}  // anonymous namespace

CUDARowData::CUDARowData(const Dataset* train_data,
                         const TrainingShareStates* train_share_state,
                         const int gpu_device_id,
                         const bool gpu_use_dp):
gpu_device_id_(gpu_device_id),
gpu_use_dp_(gpu_use_dp) {
  num_threads_ = OMP_NUM_THREADS();
  num_data_ = train_data->num_data();
  const auto& feature_hist_offsets = train_share_state->feature_hist_offsets();
  if (gpu_use_dp_) {
    shared_hist_size_ = DP_SHARED_HIST_SIZE;
  } else {
    shared_hist_size_ = SP_SHARED_HIST_SIZE;
  }
  if (feature_hist_offsets.empty()) {
    num_total_bin_ = 0;
  } else {
    num_total_bin_ = static_cast<int>(feature_hist_offsets.back());
  }
  num_feature_group_ = train_data->num_feature_groups();
  num_feature_ = train_data->num_features();
  if (gpu_device_id >= 0) {
    SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
  } else {
    SetCUDADevice(0, __FILE__, __LINE__);
  }
}

CUDARowData::~CUDARowData() {}

void CUDARowData::Init(const Dataset* train_data, TrainingShareStates* train_share_state) {
  if (num_feature_ == 0) {
    return;
  }
  DivideCUDAFeatureGroups(train_data, train_share_state);
  bit_type_ = 0;
  size_t total_size = 0;
  const void* host_row_ptr = nullptr;
  row_ptr_bit_type_ = 0;
  const void* host_data = train_share_state->GetRowWiseData(&bit_type_, &total_size, &is_sparse_, &host_row_ptr, &row_ptr_bit_type_);
  if (host_data == nullptr) {
    // Dataset::GetShareStates skipped the host multi-val bin build (EXABOOST_FAST_ROWDATA):
    // the row-wise data is known to be dense; recover the bin bit width from the
    // per-column bin counts, exactly as MultiValBin::CreateMultiValDenseBin would.
    const std::vector<uint32_t>& column_hist_offsets = train_share_state->column_hist_offsets();
    uint32_t max_bin_per_column = 0;
    for (size_t i = 0; i + 1 < column_hist_offsets.size(); ++i) {
      max_bin_per_column = std::max(max_bin_per_column, column_hist_offsets[i + 1] - column_hist_offsets[i]);
    }
    is_sparse_ = false;
    bit_type_ = max_bin_per_column <= 256 ? 8 : (max_bin_per_column <= 65536 ? 16 : 32);
  }
  // 4-bit packed row data: eligible when the data is dense uint8 and EVERY
  // column fits in <= 16 bins (bin values index the per-column histogram span,
  // so they are < the column's hist-offset delta <= 16, i.e. nibble-sized).
  // Large-bin partitions are impossible under that cap, so the global-memory
  // histogram kernels (which do not unpack) are unreachable.
  is_4bit_packed_ = false;
  if (!is_sparse_ && bit_type_ == 8 && large_bin_partitions_.empty() &&
      num_feature_partitions_ > 0 && RowData4BitEnabled()) {
    const std::vector<uint32_t>& column_hist_offsets = train_share_state->column_hist_offsets();
    uint32_t max_bin_per_column = 0;
    for (size_t i = 0; i + 1 < column_hist_offsets.size(); ++i) {
      max_bin_per_column = std::max(max_bin_per_column, column_hist_offsets[i + 1] - column_hist_offsets[i]);
    }
    is_4bit_packed_ = max_bin_per_column <= 16;
  }
  if (!is_sparse_) {
    if (is_4bit_packed_) {
      InitDense4BitData(train_data, reinterpret_cast<const uint8_t*>(host_data));
    } else if (bit_type_ == 8) {
      InitDenseData<uint8_t>(train_data, reinterpret_cast<const uint8_t*>(host_data), &cuda_data_uint8_t_);
    } else if (bit_type_ == 16) {
      InitDenseData<uint16_t>(train_data, reinterpret_cast<const uint16_t*>(host_data), &cuda_data_uint16_t_);
    } else if (bit_type_ == 32) {
      InitDenseData<uint32_t>(train_data, reinterpret_cast<const uint32_t*>(host_data), &cuda_data_uint32_t_);
    } else {
      Log::Fatal("Unknown bit type = %d", bit_type_);
    }
  } else if (bit_type_ == 8) {
    if (row_ptr_bit_type_ == 16) {
      InitSparseData<uint8_t, uint16_t>(
        reinterpret_cast<const uint8_t*>(host_data),
        reinterpret_cast<const uint16_t*>(host_row_ptr),
        &cuda_data_uint8_t_,
        &cuda_row_ptr_uint16_t_,
        &cuda_partition_ptr_uint16_t_);
    } else if (row_ptr_bit_type_ == 32) {
      InitSparseData<uint8_t, uint32_t>(
        reinterpret_cast<const uint8_t*>(host_data),
        reinterpret_cast<const uint32_t*>(host_row_ptr),
        &cuda_data_uint8_t_,
        &cuda_row_ptr_uint32_t_,
        &cuda_partition_ptr_uint32_t_);
    } else if (row_ptr_bit_type_ == 64) {
      InitSparseData<uint8_t, uint64_t>(
        reinterpret_cast<const uint8_t*>(host_data),
        reinterpret_cast<const uint64_t*>(host_row_ptr),
        &cuda_data_uint8_t_,
        &cuda_row_ptr_uint64_t_,
        &cuda_partition_ptr_uint64_t_);
    } else {
      Log::Fatal("Unknown data ptr bit type %d", row_ptr_bit_type_);
    }
  } else if (bit_type_ == 16) {
    if (row_ptr_bit_type_ == 16) {
      InitSparseData<uint16_t, uint16_t>(
        reinterpret_cast<const uint16_t*>(host_data),
        reinterpret_cast<const uint16_t*>(host_row_ptr),
        &cuda_data_uint16_t_,
        &cuda_row_ptr_uint16_t_,
        &cuda_partition_ptr_uint16_t_);
    } else if (row_ptr_bit_type_ == 32) {
      InitSparseData<uint16_t, uint32_t>(
        reinterpret_cast<const uint16_t*>(host_data),
        reinterpret_cast<const uint32_t*>(host_row_ptr),
        &cuda_data_uint16_t_,
        &cuda_row_ptr_uint32_t_,
        &cuda_partition_ptr_uint32_t_);
    } else if (row_ptr_bit_type_ == 64) {
      InitSparseData<uint16_t, uint64_t>(
        reinterpret_cast<const uint16_t*>(host_data),
        reinterpret_cast<const uint64_t*>(host_row_ptr),
        &cuda_data_uint16_t_,
        &cuda_row_ptr_uint64_t_,
        &cuda_partition_ptr_uint64_t_);
    } else {
      Log::Fatal("Unknown data ptr bit type %d", row_ptr_bit_type_);
    }
  } else if (bit_type_ == 32) {
    if (row_ptr_bit_type_ == 16) {
      InitSparseData<uint32_t, uint16_t>(
        reinterpret_cast<const uint32_t*>(host_data),
        reinterpret_cast<const uint16_t*>(host_row_ptr),
        &cuda_data_uint32_t_,
        &cuda_row_ptr_uint16_t_,
        &cuda_partition_ptr_uint16_t_);
    } else if (row_ptr_bit_type_ == 32) {
      InitSparseData<uint32_t, uint32_t>(
        reinterpret_cast<const uint32_t*>(host_data),
        reinterpret_cast<const uint32_t*>(host_row_ptr),
        &cuda_data_uint32_t_,
        &cuda_row_ptr_uint32_t_,
        &cuda_partition_ptr_uint32_t_);
    } else if (row_ptr_bit_type_ == 64) {
      InitSparseData<uint32_t, uint64_t>(
        reinterpret_cast<const uint32_t*>(host_data),
        reinterpret_cast<const uint64_t*>(host_row_ptr),
        &cuda_data_uint32_t_,
        &cuda_row_ptr_uint64_t_,
        &cuda_partition_ptr_uint64_t_);
    } else {
      Log::Fatal("Unknown data ptr bit type %d", row_ptr_bit_type_);
    }
  } else {
    Log::Fatal("Unknown bit type = %d", bit_type_);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDARowData::DivideCUDAFeatureGroups(const Dataset* train_data, TrainingShareStates* share_state) {
  const uint32_t max_num_bin_per_partition = shared_hist_size_ / 2;
  // Shape-specialized column cap (see ConstructColumnCapEnv): 504 upstream, lower
  // for low-bin many-feature data to raise construct block_dim_y. Auto-trigger:
  // per-column bins are small (<= kLowBinPerCol) AND there are enough columns that
  // the 504 cap would bind (forcing block_dim_y=1). Bit-identical either way.
  const int column_cap = [&]() {
    const int env = ConstructColumnCapEnv();
    if (env >= 0) {
      return env;  // explicit override or kill switch
    }
    constexpr uint32_t kLowBinPerCol = 32;  // low-bin regime (numerai ~6)
    constexpr int kAutoCap = 252;           // block_dim_y=2; measured sweet spot
    const std::vector<uint32_t>& cho = share_state->column_hist_offsets();
    uint32_t max_bin_per_col = 0;
    for (size_t i = 0; i + 1 < cho.size(); ++i) {
      max_bin_per_col = std::max(max_bin_per_col, cho[i + 1] - cho[i]);
    }
    // Would the 504 cap bind? Only if a partition could hold >504 columns without
    // hitting the bin cap first, i.e. 504 * max_bin_per_col <= shared bin cap.
    const bool col_cap_binds = num_feature_ > 504 &&
      static_cast<uint32_t>(504) * max_bin_per_col <= max_num_bin_per_partition;
    return (max_bin_per_col <= kLowBinPerCol && col_cap_binds) ? kAutoCap : 504;
  }();
  const std::vector<uint32_t>& column_hist_offsets = share_state->column_hist_offsets();
  std::vector<int> feature_group_num_feature_offsets;
  int offsets = 0;
  int prev_group_index = -1;
  for (int feature_index = 0; feature_index < num_feature_; ++feature_index) {
    const int feature_group_index = train_data->Feature2Group(feature_index);
    if (prev_group_index == -1 || feature_group_index != prev_group_index) {
      feature_group_num_feature_offsets.emplace_back(offsets);
      prev_group_index = feature_group_index;
    }
    ++offsets;
  }
  CHECK_EQ(offsets, num_feature_);
  feature_group_num_feature_offsets.emplace_back(offsets);

  uint32_t start_hist_offset = 0;
  feature_partition_column_index_offsets_.clear();
  column_hist_offsets_.clear();
  partition_hist_offsets_.clear();
  feature_partition_column_index_offsets_.emplace_back(0);
  partition_hist_offsets_.emplace_back(0);
  const int num_feature_groups = train_data->num_feature_groups();
  int column_index = 0;
  num_feature_partitions_ = 0;
  large_bin_partitions_.clear();
  small_bin_partitions_.clear();
  for (int feature_group_index = 0; feature_group_index < num_feature_groups; ++feature_group_index) {
    if (!train_data->IsMultiGroup(feature_group_index)) {
      const uint32_t column_feature_hist_start = column_hist_offsets[column_index];
      const uint32_t column_feature_hist_end = column_hist_offsets[column_index + 1];
      const uint32_t num_bin_in_dense_group = column_feature_hist_end - column_feature_hist_start;

      // if one column has too many bins, use a separate partition for that column
      if (num_bin_in_dense_group > max_num_bin_per_partition) {
        feature_partition_column_index_offsets_.emplace_back(column_index + 1);
        start_hist_offset = column_feature_hist_end;
        partition_hist_offsets_.emplace_back(start_hist_offset);
        large_bin_partitions_.emplace_back(num_feature_partitions_);
        ++num_feature_partitions_;
        column_hist_offsets_.emplace_back(0);
        ++column_index;
        continue;
      }

      // try if adding this column exceed the maximum number per partition
      const uint32_t cur_hist_num_bin = column_feature_hist_end - start_hist_offset;
      const int cur_partition_columns = column_index - feature_partition_column_index_offsets_.back();
      if (cur_hist_num_bin > max_num_bin_per_partition ||
          cur_partition_columns >= column_cap) {  // 504 upstream; column_cap tunable for low-bin shape-specialized construct
        feature_partition_column_index_offsets_.emplace_back(column_index);
        start_hist_offset = column_feature_hist_start;
        partition_hist_offsets_.emplace_back(start_hist_offset);
        small_bin_partitions_.emplace_back(num_feature_partitions_);
        ++num_feature_partitions_;
      }
      column_hist_offsets_.emplace_back(column_hist_offsets[column_index] - start_hist_offset);
      if (feature_group_index == num_feature_groups - 1) {
        feature_partition_column_index_offsets_.emplace_back(column_index + 1);
        partition_hist_offsets_.emplace_back(column_hist_offsets.back());
        small_bin_partitions_.emplace_back(num_feature_partitions_);
        ++num_feature_partitions_;
      }
      ++column_index;
    } else {
      const int group_feature_index_start = feature_group_num_feature_offsets[feature_group_index];
      const int num_feature_in_group = feature_group_num_feature_offsets[feature_group_index + 1] - group_feature_index_start;
      for (int sub_feature_index = 0; sub_feature_index < num_feature_in_group; ++sub_feature_index) {
        const int feature_index = group_feature_index_start + sub_feature_index;
        const uint32_t column_feature_hist_start = column_hist_offsets[column_index];
        const uint32_t column_feature_hist_end = column_hist_offsets[column_index + 1];
        const uint32_t num_bin_in_dense_group = column_feature_hist_end - column_feature_hist_start;

        // if one column has too many bins, use a separate partition for that column
        if (num_bin_in_dense_group > max_num_bin_per_partition) {
          feature_partition_column_index_offsets_.emplace_back(column_index + 1);
          start_hist_offset = column_feature_hist_end;
          partition_hist_offsets_.emplace_back(start_hist_offset);
          large_bin_partitions_.emplace_back(num_feature_partitions_);
          ++num_feature_partitions_;
          column_hist_offsets_.emplace_back(0);
          ++column_index;
          continue;
        }

        // try if adding this column exceed the maximum number per partition
        const uint32_t cur_hist_num_bin = column_feature_hist_end - start_hist_offset;
        const int cur_partition_columns = column_index - feature_partition_column_index_offsets_.back();
        if (cur_hist_num_bin > max_num_bin_per_partition ||
            cur_partition_columns >= column_cap) {  // 504 upstream; column_cap tunable for low-bin shape-specialized construct
          feature_partition_column_index_offsets_.emplace_back(column_index);
          start_hist_offset = column_feature_hist_start;
          partition_hist_offsets_.emplace_back(start_hist_offset);
          small_bin_partitions_.emplace_back(num_feature_partitions_);
          ++num_feature_partitions_;
        }
        column_hist_offsets_.emplace_back(column_hist_offsets[column_index] - start_hist_offset);
        if (feature_group_index == num_feature_groups - 1 && sub_feature_index == num_feature_in_group - 1) {
          CHECK_EQ(feature_index, num_feature_ - 1);
          feature_partition_column_index_offsets_.emplace_back(column_index + 1);
          partition_hist_offsets_.emplace_back(column_hist_offsets.back());
          small_bin_partitions_.emplace_back(num_feature_partitions_);
          ++num_feature_partitions_;
        }
        ++column_index;
      }
    }
  }
  column_hist_offsets_.emplace_back(column_hist_offsets.back() - start_hist_offset);
  max_num_column_per_partition_ = 0;
  for (size_t i = 0; i < feature_partition_column_index_offsets_.size() - 1; ++i) {
    const int num_column = feature_partition_column_index_offsets_[i + 1] - feature_partition_column_index_offsets_[i];
    if (num_column > max_num_column_per_partition_) {
      max_num_column_per_partition_ = num_column;
    }
  }
  if (max_num_column_per_partition_ == 0) {
    max_num_column_per_partition_ = 1;
  }

  cuda_feature_partition_column_index_offsets_.InitFromHostVector(feature_partition_column_index_offsets_);
  cuda_column_hist_offsets_.InitFromHostVector(column_hist_offsets_);
  cuda_partition_hist_offsets_.InitFromHostVector(partition_hist_offsets_);

  // Diagnostic-only (perf characterization; behind EXABOOST_DUMP_PARTITIONS, no
  // behavior change): report the partitioning + binding constraint per benchmark.
  if (std::getenv("EXABOOST_DUMP_PARTITIONS") != nullptr) {
    uint32_t max_partition_bins = 0, min_partition_bins = 0xffffffffu;
    int col_capped = 0, bin_capped = 0;
    for (size_t i = 0; i + 1 < partition_hist_offsets_.size(); ++i) {
      const uint32_t pb = partition_hist_offsets_[i + 1] - partition_hist_offsets_[i];
      max_partition_bins = std::max(max_partition_bins, pb);
      min_partition_bins = std::min(min_partition_bins, pb);
      const int ncol = feature_partition_column_index_offsets_[i + 1] - feature_partition_column_index_offsets_[i];
      if (ncol >= column_cap) ++col_capped;
      if (pb + 8 > max_num_bin_per_partition) ++bin_capped;  // near the bin cap
    }
    fprintf(stderr, "[DUMP_PARTITIONS] num_feature=%d num_feature_group=%d num_partitions=%d "
      "max_col_per_part=%d column_cap=%d shared_hist_size=%d(bins=%u) max_part_bins=%u min_part_bins=%u "
      "col_capped_parts=%d bin_capped_parts=%d bit_type=%d is_4bit_packed=%d small_parts=%zu large_parts=%zu\n",
      num_feature_, num_feature_group_, num_feature_partitions_, max_num_column_per_partition_,
      column_cap, shared_hist_size_, max_num_bin_per_partition, max_partition_bins,
      min_partition_bins == 0xffffffffu ? 0 : min_partition_bins,
      col_capped, bin_capped, static_cast<int>(bit_type_), is_4bit_packed_ ? 1 : 0,
      small_bin_partitions_.size(), large_bin_partitions_.size());
    fflush(stderr);
  }
}

template <typename BIN_TYPE>
void CUDARowData::GetDenseDataPartitionedToBuffer(const BIN_TYPE* row_wise_data, BIN_TYPE* out_data) {
  const int num_total_columns = feature_partition_column_index_offsets_.back();
  Threading::For<data_size_t>(0, num_data_, 512,
    [this, num_total_columns, row_wise_data, out_data] (int /*thread_index*/, data_size_t start, data_size_t end) {
      for (size_t i = 0; i < feature_partition_column_index_offsets_.size() - 1; ++i) {
        const int num_prev_columns = static_cast<int>(feature_partition_column_index_offsets_[i]);
        const size_t offset = static_cast<size_t>(num_data_) * static_cast<size_t>(num_prev_columns);
        const int partition_column_start = feature_partition_column_index_offsets_[i];
        const int partition_column_end = feature_partition_column_index_offsets_[i + 1];
        const int num_columns_in_cur_partition = partition_column_end - partition_column_start;
        for (data_size_t data_index = start; data_index < end; ++data_index) {
          const size_t data_offset = offset + static_cast<size_t>(data_index) * num_columns_in_cur_partition;
          const size_t read_data_offset = static_cast<size_t>(data_index) * num_total_columns;
          for (int column_index = 0; column_index < num_columns_in_cur_partition; ++column_index) {
            const size_t true_column_index = read_data_offset + column_index + partition_column_start;
            const BIN_TYPE bin = row_wise_data[true_column_index];
            out_data[data_offset + column_index] = bin;
          }
        }
      }
    });
}

template <typename BIN_TYPE>
void CUDARowData::GetDenseDataPartitioned(const BIN_TYPE* row_wise_data, std::vector<BIN_TYPE>* partitioned_data) {
  const int num_total_columns = feature_partition_column_index_offsets_.back();
  partitioned_data->resize(static_cast<size_t>(num_total_columns) * static_cast<size_t>(num_data_), 0);
  BIN_TYPE* out_data = partitioned_data->data();
  Threading::For<data_size_t>(0, num_data_, 512,
    [this, num_total_columns, row_wise_data, out_data] (int /*thread_index*/, data_size_t start, data_size_t end) {
      for (size_t i = 0; i < feature_partition_column_index_offsets_.size() - 1; ++i) {
        const int num_prev_columns = static_cast<int>(feature_partition_column_index_offsets_[i]);
        const size_t offset = static_cast<size_t>(num_data_) * static_cast<size_t>(num_prev_columns);
        const int partition_column_start = feature_partition_column_index_offsets_[i];
        const int partition_column_end = feature_partition_column_index_offsets_[i + 1];
        const int num_columns_in_cur_partition = partition_column_end - partition_column_start;
        for (data_size_t data_index = start; data_index < end; ++data_index) {
          const size_t data_offset = offset + static_cast<size_t>(data_index) * num_columns_in_cur_partition;
          const size_t read_data_offset = static_cast<size_t>(data_index) * num_total_columns;
          for (int column_index = 0; column_index < num_columns_in_cur_partition; ++column_index) {
            const size_t true_column_index = read_data_offset + column_index + partition_column_start;
            const BIN_TYPE bin = row_wise_data[true_column_index];
            out_data[data_offset + column_index] = bin;
          }
        }
      }
    });
}

bool CUDARowData::CollectDenseColumnData(const Dataset* train_data,
                                         std::vector<const void*>* column_data,
                                         std::vector<uint8_t>* column_bit_types) const {
  const int num_feature_groups = train_data->num_feature_groups();
  for (int group_index = 0; group_index < num_feature_groups; ++group_index) {
    if (train_data->IsMultiGroup(group_index)) {
      // multi-val group columns need per-feature most_freq_bin remapping,
      // which only the multi-val bin path implements
      return false;
    }
    uint8_t bit_type = 0;
    bool is_sparse = false;
    BinIterator* bin_iterator = nullptr;
    const void* one_column_data = train_data->GetColWiseData(group_index, -1, &bit_type, &is_sparse, &bin_iterator);
    if (bin_iterator != nullptr) {
      delete bin_iterator;
    }
    if (is_sparse || one_column_data == nullptr) {
      return false;
    }
    column_data->push_back(one_column_data);
    column_bit_types->push_back(bit_type);
  }
  CHECK_EQ(static_cast<int>(column_data->size()), feature_partition_column_index_offsets_.back());
  return true;
}

template <typename BIN_TYPE>
void CUDARowData::BuildDensePartitionedFromColumns(
    const std::vector<const void*>& column_data,
    const std::vector<uint8_t>& column_bit_types,
    BIN_TYPE* out_data) const {
  // For a non-multi-val dense group the multi-val bin stores the raw group bin
  // value unchanged (FeatureGroupIterator: min_bin == 1, most_freq_bin == 0, so
  // Get() is the identity), so the columns can be transposed directly.
  // Cache-tiled: per 512-row tile, each source column is read sequentially and
  // scattered into the row-major partition tile, which stays L2-resident.
  constexpr data_size_t kRowTileSize = 512;
  Threading::For<data_size_t>(0, num_data_, kRowTileSize,
    [this, &column_data, &column_bit_types, out_data] (int /*thread_index*/, data_size_t start, data_size_t end) {
      for (data_size_t tile_start = start; tile_start < end; tile_start += kRowTileSize) {
        const data_size_t tile_end = std::min<data_size_t>(tile_start + kRowTileSize, end);
        for (size_t i = 0; i + 1 < feature_partition_column_index_offsets_.size(); ++i) {
          const int partition_column_start = feature_partition_column_index_offsets_[i];
          const int partition_column_end = feature_partition_column_index_offsets_[i + 1];
          const int num_columns_in_cur_partition = partition_column_end - partition_column_start;
          BIN_TYPE* partition_out_data = out_data + static_cast<size_t>(num_data_) * static_cast<size_t>(partition_column_start);
          for (int column_index = partition_column_start; column_index < partition_column_end; ++column_index) {
            BIN_TYPE* column_out_data = partition_out_data + (column_index - partition_column_start);
            const void* one_column_data = column_data[column_index];
            const uint8_t column_bit_type = column_bit_types[column_index];
            if (column_bit_type == 4) {
              TransposeColumnTile4Bit<BIN_TYPE>(one_column_data, tile_start, tile_end, num_columns_in_cur_partition, column_out_data);
            } else if (column_bit_type == 8) {
              TransposeColumnTile<BIN_TYPE, uint8_t>(one_column_data, tile_start, tile_end, num_columns_in_cur_partition, column_out_data);
            } else if (column_bit_type == 16) {
              TransposeColumnTile<BIN_TYPE, uint16_t>(one_column_data, tile_start, tile_end, num_columns_in_cur_partition, column_out_data);
            } else if (column_bit_type == 32) {
              TransposeColumnTile<BIN_TYPE, uint32_t>(one_column_data, tile_start, tile_end, num_columns_in_cur_partition, column_out_data);
            } else {
              Log::Fatal("Unknown column bit type %d", static_cast<int>(column_bit_type));
            }
          }
        }
      }
    });
}

template <typename BIN_TYPE>
void CUDARowData::InitDenseData(const Dataset* train_data, const BIN_TYPE* host_data, CUDAVector<BIN_TYPE>* cuda_data) {
  const bool fast_enabled = FastRowDataEnabled();
  std::vector<const void*> column_data;
  std::vector<uint8_t> column_bit_types;
  const bool use_fast_build = fast_enabled && CollectDenseColumnData(train_data, &column_data, &column_bit_types);
  const size_t total_size = static_cast<size_t>(feature_partition_column_index_offsets_.back()) * static_cast<size_t>(num_data_);
  if (use_fast_build) {
    Log::Debug("CUDARowData: fast dense row data build from column bins (disable with EXABOOST_FAST_ROWDATA=0)");
    // new[] leaves the staging buffer uninitialized on purpose; the transpose writes every element
    std::unique_ptr<BIN_TYPE[]> buffer(new BIN_TYPE[total_size]);
    BuildDensePartitionedFromColumns<BIN_TYPE>(column_data, column_bit_types, buffer.get());
    if (FastRowDataVerifyEnabled()) {
      if (host_data == nullptr) {
        Log::Fatal("EXABOOST_FAST_ROWDATA_VERIFY=1 requires the host multi-val bin, but its build was skipped.");
      }
      std::vector<BIN_TYPE> reference;
      GetDenseDataPartitioned<BIN_TYPE>(host_data, &reference);
      CHECK_EQ(reference.size(), total_size);
      if (std::memcmp(reference.data(), buffer.get(), total_size * sizeof(BIN_TYPE)) != 0) {
        Log::Fatal("EXABOOST_FAST_ROWDATA_VERIFY: fast dense row data differs from the multi-val bin path.");
      }
      Log::Info("EXABOOST_FAST_ROWDATA_VERIFY: fast dense row data matches the multi-val bin path (%d columns, %d rows).",
                feature_partition_column_index_offsets_.back(), num_data_);
    }
    cuda_data->InitFromHostMemory(buffer.get(), total_size);
    return;
  }
  if (host_data == nullptr) {
    Log::Fatal("The host multi-val bin build was skipped but the fast dense row data build is not applicable.");
  }
  if (fast_enabled) {
    // fallback repack from the host multi-val bin, but without the redundant zero-fill
    std::unique_ptr<BIN_TYPE[]> buffer(new BIN_TYPE[total_size]);
    GetDenseDataPartitionedToBuffer<BIN_TYPE>(host_data, buffer.get());
    cuda_data->InitFromHostMemory(buffer.get(), total_size);
  } else {
    std::vector<BIN_TYPE> partitioned_data;
    GetDenseDataPartitioned<BIN_TYPE>(host_data, &partitioned_data);
    cuda_data->InitFromHostVector(partitioned_data);
  }
}

void CUDARowData::BuildDensePacked4BitFromColumns(
    const std::vector<const void*>& column_data,
    const std::vector<uint8_t>& column_bit_types,
    uint8_t* out_data) const {
  // Same cache-tiled transpose as BuildDensePartitionedFromColumns, but the
  // output is the packed layout. Columns of a partition are processed in order
  // by the same thread, so the even column assigns its byte (initializing it)
  // and the odd column ORs its high nibble in -- no read-modify-write hazard.
  constexpr data_size_t kRowTileSize = 512;
  Threading::For<data_size_t>(0, num_data_, kRowTileSize,
    [this, &column_data, &column_bit_types, out_data] (int /*thread_index*/, data_size_t start, data_size_t end) {
      for (data_size_t tile_start = start; tile_start < end; tile_start += kRowTileSize) {
        const data_size_t tile_end = std::min<data_size_t>(tile_start + kRowTileSize, end);
        for (size_t i = 0; i + 1 < feature_partition_column_index_offsets_.size(); ++i) {
          const int partition_column_start = feature_partition_column_index_offsets_[i];
          const int partition_column_end = feature_partition_column_index_offsets_[i + 1];
          const int packed_width = packed_partition_byte_offsets_[i + 1] - packed_partition_byte_offsets_[i];
          uint8_t* partition_out_data = out_data +
            static_cast<size_t>(num_data_) * static_cast<size_t>(packed_partition_byte_offsets_[i]);
          for (int column_index = partition_column_start; column_index < partition_column_end; ++column_index) {
            const int local_column = column_index - partition_column_start;
            uint8_t* column_out_data = partition_out_data + (local_column >> 1);
            const void* one_column_data = column_data[column_index];
            const uint8_t column_bit_type = column_bit_types[column_index];
            if ((local_column & 1) == 0) {
              // even column initializes the byte (high nibble 0 covers odd-width padding)
              for (data_size_t row = tile_start; row < tile_end; ++row) {
                column_out_data[static_cast<size_t>(row) * packed_width] =
                  FetchColumnBin(one_column_data, column_bit_type, row);
              }
            } else {
              for (data_size_t row = tile_start; row < tile_end; ++row) {
                column_out_data[static_cast<size_t>(row) * packed_width] |=
                  static_cast<uint8_t>(FetchColumnBin(one_column_data, column_bit_type, row) << 4);
              }
            }
          }
        }
      }
    });
}

void CUDARowData::Pack4BitFromPartitioned(const uint8_t* unpacked, uint8_t* packed) const {
  Threading::For<data_size_t>(0, num_data_, 512,
    [this, unpacked, packed] (int /*thread_index*/, data_size_t start, data_size_t end) {
      for (size_t i = 0; i + 1 < feature_partition_column_index_offsets_.size(); ++i) {
        const int partition_column_start = feature_partition_column_index_offsets_[i];
        const int num_columns = feature_partition_column_index_offsets_[i + 1] - partition_column_start;
        const int packed_width = packed_partition_byte_offsets_[i + 1] - packed_partition_byte_offsets_[i];
        const uint8_t* in_partition = unpacked +
          static_cast<size_t>(num_data_) * static_cast<size_t>(partition_column_start);
        uint8_t* out_partition = packed +
          static_cast<size_t>(num_data_) * static_cast<size_t>(packed_partition_byte_offsets_[i]);
        for (data_size_t row = start; row < end; ++row) {
          const uint8_t* in_row = in_partition + static_cast<size_t>(row) * num_columns;
          uint8_t* out_row = out_partition + static_cast<size_t>(row) * packed_width;
          for (int j = 0; j < num_columns; j += 2) {
            const uint8_t lo = in_row[j];
            const uint8_t hi = j + 1 < num_columns ? in_row[j + 1] : 0;
            out_row[j >> 1] = static_cast<uint8_t>(lo | (hi << 4));
          }
        }
      }
    });
}

void CUDARowData::InitDense4BitData(const Dataset* train_data, const uint8_t* host_data) {
  // Packed layout invariant: each partition's packed row width is
  // ceil(num_columns_in_partition / 2) bytes, i.e. the column count is padded
  // to an even number PER PARTITION, so every (row, partition) segment starts
  // byte-aligned even when partitions start at odd global column offsets.
  // Within a partition, column j lives in byte (j >> 1), nibble (j & 1)
  // (low nibble = even column, matching DenseBin<uint8_t, IS_4BIT=true>).
  packed_partition_byte_offsets_.clear();
  packed_partition_byte_offsets_.emplace_back(0);
  for (size_t i = 0; i + 1 < feature_partition_column_index_offsets_.size(); ++i) {
    const int num_columns = feature_partition_column_index_offsets_[i + 1] - feature_partition_column_index_offsets_[i];
    packed_partition_byte_offsets_.emplace_back(packed_partition_byte_offsets_.back() + ((num_columns + 1) >> 1));
  }
  const size_t packed_total = static_cast<size_t>(packed_partition_byte_offsets_.back()) * static_cast<size_t>(num_data_);
  const size_t unpacked_total = static_cast<size_t>(feature_partition_column_index_offsets_.back()) * static_cast<size_t>(num_data_);
  std::vector<const void*> column_data;
  std::vector<uint8_t> column_bit_types;
  const bool use_fast_build = FastRowDataEnabled() && CollectDenseColumnData(train_data, &column_data, &column_bit_types);
  std::unique_ptr<uint8_t[]> packed(new uint8_t[packed_total]);
  if (use_fast_build) {
    BuildDensePacked4BitFromColumns(column_data, column_bit_types, packed.get());
  } else {
    if (host_data == nullptr) {
      Log::Fatal("The host multi-val bin build was skipped but the fast dense row data build is not applicable.");
    }
    std::unique_ptr<uint8_t[]> staging(new uint8_t[unpacked_total]);
    GetDenseDataPartitionedToBuffer<uint8_t>(host_data, staging.get());
    Pack4BitFromPartitioned(staging.get(), packed.get());
  }
  if (RowData4BitVerifyEnabled()) {
    // build the exact 8-bit representation the non-packed path would produce and
    // compare element-wise after unpacking (the layouts differ by design)
    std::unique_ptr<uint8_t[]> reference(new uint8_t[unpacked_total]);
    if (use_fast_build) {
      BuildDensePartitionedFromColumns<uint8_t>(column_data, column_bit_types, reference.get());
    } else {
      GetDenseDataPartitionedToBuffer<uint8_t>(host_data, reference.get());
    }
    std::vector<int> thread_mismatch(num_threads_, 0);
    Threading::For<data_size_t>(0, num_data_, 512,
      [this, &reference, &packed, &thread_mismatch] (int thread_index, data_size_t start, data_size_t end) {
        for (size_t i = 0; i + 1 < feature_partition_column_index_offsets_.size(); ++i) {
          const int partition_column_start = feature_partition_column_index_offsets_[i];
          const int num_columns = feature_partition_column_index_offsets_[i + 1] - partition_column_start;
          const int packed_width = packed_partition_byte_offsets_[i + 1] - packed_partition_byte_offsets_[i];
          const uint8_t* ref_partition = reference.get() +
            static_cast<size_t>(num_data_) * static_cast<size_t>(partition_column_start);
          const uint8_t* packed_partition = packed.get() +
            static_cast<size_t>(num_data_) * static_cast<size_t>(packed_partition_byte_offsets_[i]);
          for (data_size_t row = start; row < end; ++row) {
            const uint8_t* ref_row = ref_partition + static_cast<size_t>(row) * num_columns;
            const uint8_t* packed_row = packed_partition + static_cast<size_t>(row) * packed_width;
            for (int j = 0; j < num_columns; ++j) {
              const uint8_t unpacked_value = (packed_row[j >> 1] >> ((j & 1) << 2)) & 0xf;
              if (unpacked_value != ref_row[j]) {
                thread_mismatch[thread_index] = 1;
              }
            }
          }
        }
      });
    for (int t = 0; t < num_threads_; ++t) {
      if (thread_mismatch[t] != 0) {
        Log::Fatal("EXABOOST_ROWDATA_4BIT_VERIFY: packed row data differs from the 8-bit representation.");
      }
    }
    Log::Info("EXABOOST_ROWDATA_4BIT_VERIFY: 4-bit packed row data matches the 8-bit representation (%d columns, %d rows).",
              feature_partition_column_index_offsets_.back(), num_data_);
  }
  Log::Debug("CUDARowData: 4-bit packed dense row data engaged (%d partitions, %d -> %d bytes per row; disable with EXABOOST_ROWDATA_4BIT=0)",
             num_feature_partitions_, feature_partition_column_index_offsets_.back(), packed_partition_byte_offsets_.back());
  cuda_data_uint8_t_.InitFromHostMemory(packed.get(), packed_total);
  cuda_packed_partition_byte_offsets_.InitFromHostVector(packed_partition_byte_offsets_);
}

template <typename BIN_TYPE, typename DATA_PTR_TYPE>
void CUDARowData::GetSparseDataPartitioned(
  const BIN_TYPE* row_wise_data,
  const DATA_PTR_TYPE* row_ptr,
  std::vector<std::vector<BIN_TYPE>>* partitioned_data,
  std::vector<std::vector<DATA_PTR_TYPE>>* partitioned_row_ptr,
  std::vector<DATA_PTR_TYPE>* partition_ptr) {
  const int num_partitions = static_cast<int>(feature_partition_column_index_offsets_.size()) - 1;
  partitioned_data->resize(num_partitions);
  partitioned_row_ptr->resize(num_partitions);
  std::vector<int> thread_max_elements_per_row(num_threads_, 0);
  Threading::For<int>(0, num_partitions, 1,
    [partitioned_data, partitioned_row_ptr, row_ptr, row_wise_data, &thread_max_elements_per_row, this] (int thread_index, int start, int end) {
      for (int partition_index = start; partition_index < end; ++partition_index) {
        std::vector<BIN_TYPE>& data_for_this_partition = partitioned_data->at(partition_index);
        std::vector<DATA_PTR_TYPE>& row_ptr_for_this_partition = partitioned_row_ptr->at(partition_index);
        const int partition_hist_start = partition_hist_offsets_[partition_index];
        const int partition_hist_end = partition_hist_offsets_[partition_index + 1];
        DATA_PTR_TYPE offset = 0;
        row_ptr_for_this_partition.clear();
        data_for_this_partition.clear();
        row_ptr_for_this_partition.emplace_back(offset);
        for (data_size_t data_index = 0; data_index < num_data_; ++data_index) {
          const DATA_PTR_TYPE row_start = row_ptr[data_index];
          const DATA_PTR_TYPE row_end = row_ptr[data_index + 1];
          const BIN_TYPE* row_data_start = row_wise_data + row_start;
          const BIN_TYPE* row_data_end = row_wise_data + row_end;
          const size_t partition_start_in_row = std::lower_bound(row_data_start, row_data_end, partition_hist_start) - row_data_start;
          const size_t partition_end_in_row = std::lower_bound(row_data_start, row_data_end, partition_hist_end) - row_data_start;
          for (size_t pos = partition_start_in_row; pos < partition_end_in_row; ++pos) {
            const BIN_TYPE bin = row_data_start[pos];
            CHECK_GE(bin, static_cast<BIN_TYPE>(partition_hist_start));
            data_for_this_partition.emplace_back(bin - partition_hist_start);
          }
          CHECK_GE(partition_end_in_row, partition_start_in_row);
          const data_size_t num_elements_in_row = partition_end_in_row - partition_start_in_row;
          offset += static_cast<DATA_PTR_TYPE>(num_elements_in_row);
          row_ptr_for_this_partition.emplace_back(offset);
          if (num_elements_in_row > thread_max_elements_per_row[thread_index]) {
            thread_max_elements_per_row[thread_index] = num_elements_in_row;
          }
        }
      }
    });
  partition_ptr->clear();
  DATA_PTR_TYPE offset = 0;
  partition_ptr->emplace_back(offset);
  for (size_t i = 0; i < partitioned_row_ptr->size(); ++i) {
    offset += partitioned_row_ptr->at(i).back();
    partition_ptr->emplace_back(offset);
  }
  max_num_column_per_partition_ = 0;
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    if (thread_max_elements_per_row[thread_index] > max_num_column_per_partition_) {
      max_num_column_per_partition_ = thread_max_elements_per_row[thread_index];
    }
  }
  if (max_num_column_per_partition_ == 0) {
    max_num_column_per_partition_ = 1;
  }
}

template <typename BIN_TYPE, typename ROW_PTR_TYPE>
void CUDARowData::InitSparseData(const BIN_TYPE* host_data,
                                 const ROW_PTR_TYPE* host_row_ptr,
                                 CUDAVector<BIN_TYPE>* cuda_data,
                                 CUDAVector<ROW_PTR_TYPE>* cuda_row_ptr,
                                 CUDAVector<ROW_PTR_TYPE>* cuda_partition_ptr) {
  std::vector<std::vector<BIN_TYPE>> partitioned_data;
  std::vector<std::vector<ROW_PTR_TYPE>> partitioned_data_ptr;
  std::vector<ROW_PTR_TYPE> partition_ptr;
  GetSparseDataPartitioned<BIN_TYPE, ROW_PTR_TYPE>(host_data, host_row_ptr, &partitioned_data, &partitioned_data_ptr, &partition_ptr);
  cuda_partition_ptr->InitFromHostVector(partition_ptr);
  cuda_data->Resize(partition_ptr.back());
  cuda_row_ptr->Resize((num_data_ + 1) * partitioned_data_ptr.size());
  for (size_t i = 0; i < partitioned_data.size(); ++i) {
    const std::vector<ROW_PTR_TYPE>& data_ptr_for_this_partition = partitioned_data_ptr[i];
    const std::vector<BIN_TYPE>& data_for_this_partition = partitioned_data[i];
    CopyFromHostToCUDADevice<BIN_TYPE>(cuda_data->RawData() + partition_ptr[i], data_for_this_partition.data(), data_for_this_partition.size(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<ROW_PTR_TYPE>(cuda_row_ptr->RawData() + i * (num_data_ + 1), data_ptr_for_this_partition.data(), data_ptr_for_this_partition.size(), __FILE__, __LINE__);
  }
}

template <typename BIN_TYPE>
const BIN_TYPE* CUDARowData::GetBin() const {
  if (bit_type_ == 8) {
    return reinterpret_cast<const BIN_TYPE*>(cuda_data_uint8_t_.RawData());
  } else if (bit_type_ == 16) {
    return reinterpret_cast<const BIN_TYPE*>(cuda_data_uint16_t_.RawData());
  } else if (bit_type_ == 32) {
    return reinterpret_cast<const BIN_TYPE*>(cuda_data_uint32_t_.RawData());
  } else {
    Log::Fatal("Unknown bit_type %d for GetBin.", bit_type_);
  }
}

template const uint8_t* CUDARowData::GetBin<uint8_t>() const;

template const uint16_t* CUDARowData::GetBin<uint16_t>() const;

template const uint32_t* CUDARowData::GetBin<uint32_t>() const;

template <typename PTR_TYPE>
const PTR_TYPE* CUDARowData::GetRowPtr() const {
  if (row_ptr_bit_type_ == 16) {
    return reinterpret_cast<const PTR_TYPE*>(cuda_row_ptr_uint16_t_.RawData());
  } else if (row_ptr_bit_type_ == 32) {
    return reinterpret_cast<const PTR_TYPE*>(cuda_row_ptr_uint32_t_.RawData());
  } else if (row_ptr_bit_type_ == 64) {
    return reinterpret_cast<const PTR_TYPE*>(cuda_row_ptr_uint64_t_.RawData());
  } else {
    Log::Fatal("Unknown row_ptr_bit_type = %d for GetRowPtr.", row_ptr_bit_type_);
  }
}

template const uint16_t* CUDARowData::GetRowPtr<uint16_t>() const;

template const uint32_t* CUDARowData::GetRowPtr<uint32_t>() const;

template const uint64_t* CUDARowData::GetRowPtr<uint64_t>() const;

template <typename PTR_TYPE>
const PTR_TYPE* CUDARowData::GetPartitionPtr() const {
  if (row_ptr_bit_type_ == 16) {
    return reinterpret_cast<const PTR_TYPE*>(cuda_partition_ptr_uint16_t_.RawData());
  } else if (row_ptr_bit_type_ == 32) {
    return reinterpret_cast<const PTR_TYPE*>(cuda_partition_ptr_uint32_t_.RawData());
  } else if (row_ptr_bit_type_ == 64) {
    return reinterpret_cast<const PTR_TYPE*>(cuda_partition_ptr_uint64_t_.RawData());
  } else {
    Log::Fatal("Unknown row_ptr_bit_type = %d for GetPartitionPtr.", row_ptr_bit_type_);
  }
}

template const uint16_t* CUDARowData::GetPartitionPtr<uint16_t>() const;

template const uint32_t* CUDARowData::GetPartitionPtr<uint32_t>() const;

template const uint64_t* CUDARowData::GetPartitionPtr<uint64_t>() const;

}  // namespace LightGBM

#endif  // USE_CUDA
