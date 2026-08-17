/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef FALCATA_INCLUDE_FALCATA_CUDA_CUDA_COLUMN_DATA_HPP_
#define FALCATA_INCLUDE_FALCATA_CUDA_CUDA_COLUMN_DATA_HPP_

#ifdef USE_CUDA

#include <Falcata/config.h>
#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/bin.h>
#include <Falcata/utils/openmp_wrapper.h>

#include <memory>
#include <cstdint>
#include <vector>

namespace Falcata {

class CUDAColumnData {
 public:
  CUDAColumnData(const data_size_t num_data, const int gpu_device_id);

  ~CUDAColumnData();

  void Init(const int num_columns,
            const std::vector<const void*>& column_data,
            const std::vector<BinIterator*>& column_bin_iterator,
            const std::vector<uint8_t>& column_bit_type,
            const std::vector<uint32_t>& feature_max_bin,
            const std::vector<uint32_t>& feature_min_bin,
            const std::vector<uint32_t>& feature_offset,
            const std::vector<uint32_t>& feature_most_freq_bin,
            const std::vector<uint32_t>& feature_default_bin,
            const std::vector<uint8_t>& feature_missing_is_zero,
            const std::vector<uint8_t>& feature_missing_is_na,
            const std::vector<uint8_t>& feature_mfb_is_zero,
            const std::vector<uint8_t>& feature_mfb_is_na,
            const std::vector<int>& feature_to_column);

  const uint8_t* GetColumnData(const int column_index) const {
    // In the skip-allocation path, data_by_column_[c] is null; the per-column
    // device pointers live in the per-tree compact view (set by
    // SetCompactColumnView). Read those instead to avoid a host null-deref.
    // Sparse-encoded columns keep their own buffer even there (see
    // column_is_sparse_).
    if (init_skipped_per_column_alloc_ && !column_is_sparse(column_index)) {
      return compact_column_host_view_[column_index];
    }
    return data_by_column_[column_index]->RawData();
  }

  /*! \brief Whether this column's bins came from a sparse bin iterator rather
   *  than a dense buffer. Such a column stores 0 for every row at the feature's
   *  most-frequent bin, while the row-wise matrix stores that bin's real index
   *  -- so its bytes are NOT interchangeable with the row-data-derived compact
   *  view the split-apply kernels otherwise read. */
  bool column_is_sparse(const int column_index) const {
    return column_index < static_cast<int>(column_is_sparse_.size()) &&
           column_is_sparse_[column_index] != 0;
  }

  void CopySubrow(const CUDAColumnData* full_set, const data_size_t* used_indices, const data_size_t num_used_indices);

  uint8_t* const* cuda_data_by_column() const { return cuda_data_by_column_.RawData(); }

  uint32_t feature_min_bin(const int feature_index) const { return feature_min_bin_[feature_index]; }

  uint32_t feature_max_bin(const int feature_index) const { return feature_max_bin_[feature_index]; }

  uint32_t feature_offset(const int feature_index) const { return feature_offset_[feature_index]; }

  uint32_t feature_most_freq_bin(const int feature_index) const { return feature_most_freq_bin_[feature_index]; }

  uint32_t feature_default_bin(const int feature_index) const { return feature_default_bin_[feature_index]; }

  uint8_t feature_missing_is_zero(const int feature_index) const { return feature_missing_is_zero_[feature_index]; }

  uint8_t feature_missing_is_na(const int feature_index) const { return feature_missing_is_na_[feature_index]; }

  uint8_t feature_mfb_is_zero(const int feature_index) const { return feature_mfb_is_zero_[feature_index]; }

  uint8_t feature_mfb_is_na(const int feature_index) const { return feature_mfb_is_na_[feature_index]; }

  const uint32_t* cuda_feature_min_bin() const { return cuda_feature_min_bin_.RawData(); }

  const uint32_t* cuda_feature_max_bin() const { return cuda_feature_max_bin_.RawData(); }

  const uint32_t* cuda_feature_offset() const { return cuda_feature_offset_.RawData(); }

  const uint32_t* cuda_feature_most_freq_bin() const { return cuda_feature_most_freq_bin_.RawData(); }

  const uint32_t* cuda_feature_default_bin() const { return cuda_feature_default_bin_.RawData(); }

  const uint8_t* cuda_feature_missing_is_zero() const { return cuda_feature_missing_is_zero_.RawData(); }

  const uint8_t* cuda_feature_missing_is_na() const { return cuda_feature_missing_is_na_.RawData(); }

  const uint8_t* cuda_feature_mfb_is_zero() const { return cuda_feature_mfb_is_zero_.RawData(); }

  const uint8_t* cuda_feature_mfb_is_na() const { return cuda_feature_mfb_is_na_.RawData(); }

  const int* cuda_feature_to_column() const { return cuda_feature_to_column_.RawData(); }

  const uint8_t* cuda_column_bit_type() const { return cuda_column_bit_type_.RawData(); }

  int feature_to_column(const int feature_index) const { return feature_to_column_[feature_index]; }

  uint8_t column_bit_type(const int column_index) const { return column_bit_type_[column_index]; }

  // ===== Per-tree compact column view =====
  // Set a compact column buffer for sampled features. Updates data_by_column_[c]
  // pointers for sampled features to point into compact_buf at strided offsets;
  // re-uploads cuda_data_by_column_. Non-sampled features get nullptr.
  // Compact layout: compact_buf[slot * num_data + r]; slot = compact index assigned
  // by the caller via column_to_compact_slot[c] (-1 if not in sample).
  void SetCompactColumnView(const std::vector<int>& column_to_compact_slot,
                            void* compact_buf,
                            size_t bytes_per_col);

  // Restore the GPU pointer table to the original per-column allocations.
  // Any consumer that reads columns the CURRENT tree did not sample -- tree
  // traversal over an older tree, above all -- must run against this view: a
  // per-tree compact view leaves non-sampled columns null and repoints sampled
  // ones at a scratch buffer the next tree overwrites.
  void RestoreOriginalColumnView();

  /*! \brief Whether the pointer table currently holds the original per-column
   *  allocations rather than a per-tree compact view. */
  bool original_column_view_active() const { return original_column_view_active_; }

  /*! \brief Whether original per-column buffers exist at all. They are skipped
   *  when they would not fit (see init_skipped_per_column_alloc_), and then only
   *  the per-tree compact views ever exist. */
  bool has_original_column_view() const { return !init_skipped_per_column_alloc_; }

  /*! \brief Whether the published pointer table can serve reads of this column.
   *  False for a column the current tree did not sample, and for every column
   *  while a nibble-packed view is published (the plain readers cannot decode
   *  it). */
  bool ColumnAvailableInCurrentView(const int column_index) const {
    if (original_column_view_active_) {
      return true;
    }
    if (packed_column_view_active_) {
      return false;
    }
    return column_index >= 0 && column_index < static_cast<int>(compact_column_host_view_.size()) &&
           compact_column_host_view_[column_index] != nullptr;
  }

  /*! \brief Bumped on every change of the published pointer table, so callers
   *  that cache an installed view can tell when someone else replaced it. */
  uint64_t column_view_generation() const { return column_view_generation_; }

  // ===== Per-tree packed compact column view (4-bit) =====
  // Register the histogram constructor's 4-bit packed compact matrix as this
  // tree's column source: column c's bin for row r is
  //   (packed_column_data(c)[r * packed_column_stride(c)] >> packed_column_shift(c)) & 0xf.
  // Only the batched apply path (CUDAHybridApplyDescriptor, bit_type 4) reads
  // it; callers needing a plain per-column buffer must first re-register one
  // via SetCompactColumnView (see the tree learner's lazy classic fallback).
  // Also nulls compact_column_host_view_ so a stale GetColumnData pointer can
  // never be consumed silently.
  void SetCompactPackedColumnView(const std::vector<int>& column_to_compact_slot,
                                  const uint8_t* packed_buf,
                                  const std::vector<size_t>& slot_base_byte,
                                  const std::vector<int>& slot_row_stride,
                                  const std::vector<uint8_t>& slot_shift);

  bool packed_column_view_active() const { return packed_column_view_active_; }

  const uint8_t* packed_column_data(const int column_index) const {
    return packed_column_ptr_[column_index];
  }

  int packed_column_stride(const int column_index) const {
    return packed_column_stride_[column_index];
  }

  uint8_t packed_column_shift(const int column_index) const {
    return packed_column_shift_[column_index];
  }

  // Skip per-column allocation in Init? Used when caller will provide compact view.
  bool init_skipped_per_column_alloc_ = false;

 private:
  template <bool IS_SPARSE, bool IS_4BIT, typename BIN_TYPE>
  void InitOneColumnData(const void* in_column_data, BinIterator* bin_iterator, CUDAVector<uint8_t>* out_column_data_pointer);

  void LaunchCopySubrowKernel(uint8_t* const* in_cuda_data_by_column);

  void InitColumnMetaInfo();

  void ResizeWhenCopySubrow(const data_size_t num_used_indices);

  std::vector<uint8_t*> GetDataByColumnPointers(const std::vector<std::unique_ptr<CUDAVector<uint8_t>>>& data_by_column) const {
    std::vector<uint8_t*> data_by_column_pointers(data_by_column.size(), nullptr);
    for (size_t i = 0; i < data_by_column.size(); ++i) {
      // unique_ptr can be null when init_skipped_per_column_alloc_ is set —
      // those slots are filled later by SetCompactColumnView().
      if (data_by_column[i]) {
        data_by_column_pointers[i] = reinterpret_cast<uint8_t*>(data_by_column[i]->RawData());
      }
    }
    return data_by_column_pointers;
  }

  int gpu_device_id_;
  int num_threads_;
  data_size_t num_data_;
  int num_columns_;
  std::vector<uint8_t> column_bit_type_;
  /*! \brief per column: bins came from a sparse bin iterator (see column_is_sparse) */
  std::vector<uint8_t> column_is_sparse_;
  std::vector<uint32_t> feature_min_bin_;
  std::vector<uint32_t> feature_max_bin_;
  std::vector<uint32_t> feature_offset_;
  std::vector<uint32_t> feature_most_freq_bin_;
  std::vector<uint32_t> feature_default_bin_;
  std::vector<uint8_t> feature_missing_is_zero_;
  std::vector<uint8_t> feature_missing_is_na_;
  std::vector<uint8_t> feature_mfb_is_zero_;
  std::vector<uint8_t> feature_mfb_is_na_;
  CUDAVector<uint8_t*> cuda_data_by_column_;
  std::vector<int> feature_to_column_;
  std::vector<std::unique_ptr<CUDAVector<uint8_t>>> data_by_column_;
  // Host mirror of the per-column device pointers when init_skipped_per_column_alloc_
  // is set: each entry points into the per-tree compact buffer (slot * num_data),
  // or nullptr for non-sampled columns. Populated by SetCompactColumnView so that
  // GetColumnData has a valid host-readable device pointer for the split column.
  std::vector<uint8_t*> compact_column_host_view_;
  // Per-tree packed compact view (see SetCompactPackedColumnView): host-side
  // per-column device base pointer / per-row byte stride / nibble shift. Only
  // consumed host-side when building batched apply descriptors.
  bool packed_column_view_active_ = false;
  // True while cuda_data_by_column_ holds the original per-column allocations.
  bool original_column_view_active_ = false;
  uint64_t column_view_generation_ = 0;
  std::vector<const uint8_t*> packed_column_ptr_;
  std::vector<int> packed_column_stride_;
  std::vector<uint8_t> packed_column_shift_;

  CUDAVector<uint8_t> cuda_column_bit_type_;
  CUDAVector<uint32_t> cuda_feature_min_bin_;
  CUDAVector<uint32_t> cuda_feature_max_bin_;
  CUDAVector<uint32_t> cuda_feature_offset_;
  CUDAVector<uint32_t> cuda_feature_most_freq_bin_;
  CUDAVector<uint32_t> cuda_feature_default_bin_;
  CUDAVector<uint8_t> cuda_feature_missing_is_zero_;
  CUDAVector<uint8_t> cuda_feature_missing_is_na_;
  CUDAVector<uint8_t> cuda_feature_mfb_is_zero_;
  CUDAVector<uint8_t> cuda_feature_mfb_is_na_;
  CUDAVector<int> cuda_feature_to_column_;

  // used when bagging with subset
  CUDAVector<data_size_t> cuda_used_indices_;
  data_size_t num_used_indices_;
  data_size_t cur_subset_buffer_size_;
};

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_INCLUDE_FALCATA_CUDA_CUDA_COLUMN_DATA_HPP_
