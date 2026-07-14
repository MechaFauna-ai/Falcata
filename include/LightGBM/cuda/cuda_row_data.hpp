/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_ROW_DATA_HPP_
#define LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_ROW_DATA_HPP_

#ifdef USE_CUDA

#include <LightGBM/bin.h>
#include <LightGBM/config.h>
#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/dataset.h>
#include <LightGBM/train_share_states.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <cstdint>
#include <vector>

#define COPY_SUBROW_BLOCK_SIZE_ROW_DATA (1024)

#if CUDART_VERSION == 10000
#define DP_SHARED_HIST_SIZE (5176)
#else
#define DP_SHARED_HIST_SIZE (6144)
#endif
#define SP_SHARED_HIST_SIZE (DP_SHARED_HIST_SIZE * 2)

namespace LightGBM {

class CUDARowData {
 public:
  CUDARowData(const Dataset* train_data,
              const TrainingShareStates* train_share_state,
              const int gpu_device_id,
              const bool gpu_use_dp);

  ~CUDARowData();

  void Init(const Dataset* train_data,
            TrainingShareStates* train_share_state);

  void CopySubrow(const CUDARowData* full_set, const data_size_t* used_indices, const data_size_t num_used_indices);

  void CopySubcol(const CUDARowData* full_set, const std::vector<int8_t>& is_feature_used, const Dataset* train_data);

  void CopySubrowAndSubcol(const CUDARowData* full_set, const data_size_t* used_indices,
    const data_size_t num_used_indices, const std::vector<bool>& is_feature_used, const Dataset* train_data);

  template <typename BIN_TYPE>
  const BIN_TYPE* GetBin() const;

  template <typename PTR_TYPE>
  const PTR_TYPE* GetPartitionPtr() const;

  template <typename PTR_TYPE>
  const PTR_TYPE* GetRowPtr() const;

  int NumLargeBinPartition() const { return static_cast<int>(large_bin_partitions_.size()); }

  int num_feature_partitions() const { return num_feature_partitions_; }

  int max_num_column_per_partition() const { return max_num_column_per_partition_; }

  bool is_sparse() const { return is_sparse_; }

  uint8_t bit_type() const { return bit_type_; }

  /*! \brief whether cuda_data_uint8_t_ stores two bin values per byte (see
   *  InitDense4BitData for the layout invariant); bit_type() stays 8 */
  bool is_4bit_packed() const { return is_4bit_packed_; }

  /*! \brief prefix over partitions of the packed per-row byte widths
   *  (ceil(num_columns_in_partition / 2)); only valid when is_4bit_packed() */
  const int* cuda_packed_partition_byte_offsets() const { return cuda_packed_partition_byte_offsets_.RawData(); }

  const std::vector<int>& host_packed_partition_byte_offsets() const { return packed_partition_byte_offsets_; }

  uint8_t row_ptr_bit_type() const { return row_ptr_bit_type_; }

  const int* cuda_feature_partition_column_index_offsets() const { return cuda_feature_partition_column_index_offsets_.RawData(); }

  const uint32_t* cuda_column_hist_offsets() const { return cuda_column_hist_offsets_.RawData(); }

  const uint32_t* cuda_partition_hist_offsets() const { return cuda_partition_hist_offsets_.RawData(); }

  int shared_hist_size() const { return shared_hist_size_; }

  // Host-side accessors for the CompactView path in CUDAHistogramConstructor.
  const std::vector<int>& host_feature_partition_column_index_offsets() const { return feature_partition_column_index_offsets_; }
  const std::vector<uint32_t>& host_column_hist_offsets() const { return column_hist_offsets_; }
  const std::vector<uint32_t>& host_partition_hist_offsets() const { return partition_hist_offsets_; }
  data_size_t num_data() const { return num_data_; }
  bool is_data_host_mapped() const { return cuda_data_is_host_mapped_; }
  const uint8_t* host_partitioned_data_uint8_t() const { return host_partitioned_data_uint8_t_.data(); }

 private:
  void DivideCUDAFeatureGroups(const Dataset* train_data, TrainingShareStates* share_state);

  /*! \brief Initialize the dense row-wise data. Prefers the fast build directly from
   *  the Dataset's column bins (EXABOOST_FAST_ROWDATA, default on); host_data (the
   *  row-wise multi-val bin) may be nullptr when Dataset::GetShareStates skipped its build. */
  template <typename BIN_TYPE>
  void InitDenseData(const Dataset* train_data, const BIN_TYPE* host_data, CUDAVector<BIN_TYPE>* cuda_data);

  /*! \brief Initialize the dense row-wise data in 4-bit packed form (two bin
   *  values per byte; every column has <= 16 bins). See the implementation for
   *  the per-partition alignment invariant. */
  void InitDense4BitData(const Dataset* train_data, const uint8_t* host_data);

  /*! \brief Packed variant of BuildDensePartitionedFromColumns: writes nibbles. */
  void BuildDensePacked4BitFromColumns(const std::vector<const void*>& column_data,
    const std::vector<uint8_t>& column_bit_types, uint8_t* out_data) const;

  /*! \brief Pack an 8-bit partitioned row-major buffer into the 4-bit layout. */
  void Pack4BitFromPartitioned(const uint8_t* unpacked, uint8_t* packed) const;

  /*! \brief Collect the raw per-column bin data in multi-val column order.
   *  Returns false (fast build not applicable) on multi-val or sparse groups. */
  bool CollectDenseColumnData(const Dataset* train_data,
    std::vector<const void*>* column_data,
    std::vector<uint8_t>* column_bit_types) const;

  /*! \brief Build the partitioned row-major buffer directly from column bins
   *  with a cache-tiled parallel transpose, bypassing the host multi-val bin. */
  template <typename BIN_TYPE>
  void BuildDensePartitionedFromColumns(const std::vector<const void*>& column_data,
    const std::vector<uint8_t>& column_bit_types, BIN_TYPE* out_data) const;

  template <typename BIN_TYPE>
  void GetDenseDataPartitioned(const BIN_TYPE* row_wise_data, std::vector<BIN_TYPE>* partitioned_data);

  // In-place version: writes into a pre-sized buffer (e.g. a host-pinned vector).
  template <typename BIN_TYPE>
  void GetDenseDataPartitionedToBuffer(const BIN_TYPE* row_wise_data, BIN_TYPE* out_data);

  template <typename BIN_TYPE, typename ROW_PTR_TYPE>
  void GetSparseDataPartitioned(const BIN_TYPE* row_wise_data,
    const ROW_PTR_TYPE* row_ptr,
    std::vector<std::vector<BIN_TYPE>>* partitioned_data,
    std::vector<std::vector<ROW_PTR_TYPE>>* partitioned_row_ptr,
    std::vector<ROW_PTR_TYPE>* partition_ptr);

  template <typename BIN_TYPE, typename ROW_PTR_TYPE>
  void InitSparseData(const BIN_TYPE* host_data,
                      const ROW_PTR_TYPE* host_row_ptr,
                      CUDAVector<BIN_TYPE>* cuda_data,
                      CUDAVector<ROW_PTR_TYPE>* cuda_row_ptr,
                      CUDAVector<ROW_PTR_TYPE>* cuda_partition_ptr);

  /*! \brief number of threads to use */
  int num_threads_;
  /*! \brief number of training data */
  data_size_t num_data_;
  /*! \brief number of bins of all features */
  int num_total_bin_;
  /*! \brief number of feature groups in dataset */
  int num_feature_group_;
  /*! \brief number of features in dataset */
  int num_feature_;
  /*! \brief number of bits used to store each bin value */
  uint8_t bit_type_;
  /*! \brief number of bits used to store each row pointer value */
  uint8_t row_ptr_bit_type_;
  /*! \brief is sparse row wise data */
  bool is_sparse_;
  /*! \brief cuda_data_uint8_t_ stores two bin values per byte (all columns <= 16 bins) */
  bool is_4bit_packed_ = false;
  /*! \brief prefix over partitions of packed per-row byte widths (size P+1) */
  std::vector<int> packed_partition_byte_offsets_;
  /*! \brief start column index of each feature partition */
  std::vector<int> feature_partition_column_index_offsets_;
  /*! \brief histogram offset of each column */
  std::vector<uint32_t> column_hist_offsets_;
  /*! \brief hisotgram offset of each partition */
  std::vector<uint32_t> partition_hist_offsets_;
  /*! \brief maximum number of columns among all feature partitions */
  int max_num_column_per_partition_;
  /*! \brief number of partitions */
  int num_feature_partitions_;
  /*! \brief used when bagging with subset, number of used indices */
  data_size_t num_used_indices_;
  /*! \brief used when bagging with subset, number of total elements */
  uint64_t num_total_elements_;
  /*! \brief used when bagging with column subset, the size of maximum number of feature partitions */
  int cur_num_feature_partition_buffer_size_;
  /*! \brief CUDA device ID */
  int gpu_device_id_;
  /*! \brief index of partitions with large bins that its histogram cannot fit into shared memory, each large bin partition contains a single column */
  std::vector<int> large_bin_partitions_;
  /*! \brief index of partitions with small bins */
  std::vector<int> small_bin_partitions_;
  /*! \brief shared memory size used by histogram */
  int shared_hist_size_;
  /*! \brief whether to use double precision in histograms per block */
  bool gpu_use_dp_;
  /*! \brief When true, cuda_data_uint8_t_ is a host-pinned, device-mapped pointer
   *  (allocated via cudaHostRegister, freed via cudaHostUnregister). This is the
   *  "host-mapped" zero-copy fallback used when the bin matrix is too large to
   *  fit on the GPU alongside CUDAColumnData. */
  bool cuda_data_is_host_mapped_ = false;
  /*! \brief Backing storage for the host-pinned bin matrix when host-mapping is used. */
  std::vector<uint8_t> host_partitioned_data_uint8_t_;

  // CUDA memory

  /*! \brief row-wise data stored in CUDA, 8 bits */
  CUDAVector<uint8_t> cuda_data_uint8_t_;
  /*! \brief row-wise data stored in CUDA, 16 bits */
  CUDAVector<uint16_t> cuda_data_uint16_t_;
  /*! \brief row-wise data stored in CUDA, 32 bits */
  CUDAVector<uint32_t> cuda_data_uint32_t_;
  /*! \brief row pointer stored in CUDA, 16 bits */
  CUDAVector<uint16_t> cuda_row_ptr_uint16_t_;
  /*! \brief row pointer stored in CUDA, 32 bits */
  CUDAVector<uint32_t> cuda_row_ptr_uint32_t_;
  /*! \brief row pointer stored in CUDA, 64 bits */
  CUDAVector<uint64_t> cuda_row_ptr_uint64_t_;
  /*! \brief partition bin offsets, 16 bits */
  CUDAVector<uint16_t> cuda_partition_ptr_uint16_t_;
  /*! \brief partition bin offsets, 32 bits */
  CUDAVector<uint32_t> cuda_partition_ptr_uint32_t_;
  /*! \brief partition bin offsets, 64 bits */
  CUDAVector<uint64_t> cuda_partition_ptr_uint64_t_;
  /*! \brief start column index of each feature partition */
  CUDAVector<int> cuda_feature_partition_column_index_offsets_;
  /*! \brief prefix over partitions of packed per-row byte widths (4-bit mode) */
  CUDAVector<int> cuda_packed_partition_byte_offsets_;
  /*! \brief histogram offset of each column */
  CUDAVector<uint32_t> cuda_column_hist_offsets_;
  /*! \brief hisotgram offset of each partition */
  CUDAVector<uint32_t> cuda_partition_hist_offsets_;
  /*! \brief block buffer when calculating prefix sum */
  CUDAVector<uint16_t> cuda_block_buffer_uint16_t_;
  /*! \brief block buffer when calculating prefix sum */
  CUDAVector<uint32_t> cuda_block_buffer_uint32_t_;
  /*! \brief block buffer when calculating prefix sum */
  CUDAVector<uint64_t> cuda_block_buffer_uint64_t_;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_INCLUDE_LIGHTGBM_CUDA_CUDA_ROW_DATA_HPP_
