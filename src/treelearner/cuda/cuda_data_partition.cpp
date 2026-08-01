/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "cuda_data_partition.hpp"

namespace Falcata {

CUDADataPartition::CUDADataPartition(
  const Dataset* train_data,
  const int num_total_bin,
  const int num_leaves,
  const int num_threads,
  const bool use_quantized_grad,
  hist_t* cuda_hist):

  num_data_(train_data->num_data()),
  num_features_(train_data->num_features()),
  num_total_bin_(num_total_bin),
  num_leaves_(num_leaves),
  num_threads_(num_threads),
  use_quantized_grad_(use_quantized_grad),
  cuda_hist_(cuda_hist) {
  CalcBlockDim(num_data_);
  max_num_split_indices_blocks_ = grid_dim_;
  cur_num_leaves_ = 1;
  cuda_column_data_ = train_data->cuda_column_data();

  is_categorical_feature_.resize(train_data->num_features(), false);
  is_single_feature_in_column_.resize(train_data->num_features(), false);
  for (int feature_index = 0; feature_index < train_data->num_features(); ++feature_index) {
    if (train_data->FeatureBinMapper(feature_index)->bin_type() == BinType::CategoricalBin) {
      is_categorical_feature_[feature_index] = true;
    }
    const int feature_group_index = train_data->Feature2Group(feature_index);
    if (!train_data->IsMultiGroup(feature_group_index)) {
      if ((feature_index == 0 || train_data->Feature2Group(feature_index - 1) != feature_group_index) &&
        (feature_index == train_data->num_features() - 1 || train_data->Feature2Group(feature_index + 1) != feature_group_index)) {
        is_single_feature_in_column_[feature_index] = true;
      }
    } else {
      is_single_feature_in_column_[feature_index] = true;
    }
  }
}

CUDADataPartition::~CUDADataPartition() {
  CUDASUCCESS_OR_FATAL(cudaStreamDestroy(cuda_streams_[0]));
  CUDASUCCESS_OR_FATAL(cudaStreamDestroy(cuda_streams_[1]));
  CUDASUCCESS_OR_FATAL(cudaStreamDestroy(cuda_streams_[2]));
  CUDASUCCESS_OR_FATAL(cudaStreamDestroy(cuda_streams_[3]));
  cuda_streams_.clear();
  cuda_streams_.shrink_to_fit();
  if (indices_copy_done_event_ != nullptr) {
    CUDASUCCESS_OR_FATAL(cudaEventDestroy(indices_copy_done_event_));
  }
  if (pinned_split_info_ != nullptr) {
    CUDASUCCESS_OR_FATAL(cudaFreeHost(pinned_split_info_));
  }
}

void CUDADataPartition::Init() {
  // allocate CUDA memory
  cuda_data_indices_.Resize(static_cast<size_t>(num_data_));
  cuda_leaf_data_start_.Resize(static_cast<size_t>(num_leaves_));
  cuda_leaf_data_end_.Resize(static_cast<size_t>(num_leaves_));
  cuda_leaf_num_data_.Resize(static_cast<size_t>(num_leaves_));
  // leave some space for alignment
  cuda_block_to_left_offset_.Resize(static_cast<size_t>(num_data_));
  cuda_data_index_to_leaf_index_.Resize(static_cast<size_t>(num_data_));
  cuda_block_data_to_left_offset_.Resize(static_cast<size_t>(max_num_split_indices_blocks_) + 1);
  cuda_block_data_to_right_offset_.Resize(static_cast<size_t>(max_num_split_indices_blocks_) + 1);
  SetCUDAMemory<data_size_t>(cuda_block_data_to_left_offset_.RawData(), 0, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
  SetCUDAMemory<data_size_t>(cuda_block_data_to_right_offset_.RawData(), 0, static_cast<size_t>(max_num_split_indices_blocks_) + 1, __FILE__, __LINE__);
  cuda_out_data_indices_in_leaf_.Resize(static_cast<size_t>(num_data_));
  cuda_hist_pool_.Resize(static_cast<size_t>(num_leaves_));
  CopyFromHostToCUDADevice<hist_t*>(cuda_hist_pool_.RawData(), &cuda_hist_, 1, __FILE__, __LINE__);

  // 18 ints per split; slot 0 serves the classic immediate mode, slots [0, num_leaves)
  // serve the hybrid level-batched mode's deferred readback
  cuda_split_info_buffer_.Resize(18 * static_cast<size_t>(num_leaves_) + 18);
  // one smaller-child count per split of a level (a level has < num_leaves splits)
  cuda_level_smaller_counts_.Resize(static_cast<size_t>(num_leaves_) + 1);

  cuda_leaf_output_.Resize(static_cast<size_t>(num_leaves_));

  cuda_streams_.resize(4);
  gpuAssert(cudaStreamCreate(&cuda_streams_[0]), __FILE__, __LINE__);
  gpuAssert(cudaStreamCreate(&cuda_streams_[1]), __FILE__, __LINE__);
  gpuAssert(cudaStreamCreate(&cuda_streams_[2]), __FILE__, __LINE__);
  gpuAssert(cudaStreamCreate(&cuda_streams_[3]), __FILE__, __LINE__);
  CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&indices_copy_done_event_, cudaEventDisableTiming));

  cuda_num_data_.InitFromHostVector(std::vector<data_size_t>{num_data_});
  use_bagging_ = false;
  used_indices_ = nullptr;
}

void CUDADataPartition::BeforeTrain() {
  if (!use_bagging_) {
    LaunchFillDataIndicesBeforeTrain();
  }
  // async memsets on the default stream (SetCUDAMemory would pay one full
  // device sync each); the synchronous copies below order after them
  CUDASUCCESS_OR_FATAL(cudaMemset(reinterpret_cast<void*>(cuda_leaf_num_data_.RawData()), 0, sizeof(data_size_t) * static_cast<size_t>(num_leaves_)));
  CUDASUCCESS_OR_FATAL(cudaMemset(reinterpret_cast<void*>(cuda_leaf_data_start_.RawData()), 0, sizeof(data_size_t) * static_cast<size_t>(num_leaves_)));
  CUDASUCCESS_OR_FATAL(cudaMemset(reinterpret_cast<void*>(cuda_leaf_data_end_.RawData()), 0, sizeof(data_size_t) * static_cast<size_t>(num_leaves_)));
  if (!use_bagging_) {
    CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_leaf_num_data_.RawData(), cuda_num_data_.RawData(), 1, __FILE__, __LINE__);
    CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_leaf_data_end_.RawData(), cuda_num_data_.RawData(), 1, __FILE__, __LINE__);
  } else {
    CopyFromHostToCUDADevice<data_size_t>(cuda_leaf_num_data_.RawData(), &num_used_indices_, 1, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<data_size_t>(cuda_leaf_data_end_.RawData(), &num_used_indices_, 1, __FILE__, __LINE__);
  }
  CopyFromHostToCUDADevice<hist_t*>(cuda_hist_pool_.RawData(), &cuda_hist_, 1, __FILE__, __LINE__);
}

void CUDADataPartition::Split(
  // input best split info
  const CUDASplitInfo* best_split_info,
  const int left_leaf_index,
  const int right_leaf_index,
  const int leaf_best_split_feature,
  const uint32_t leaf_best_split_threshold,
  const uint32_t* categorical_bitset,
  const int categorical_bitset_len,
  const uint8_t leaf_best_split_default_left,
  const data_size_t num_data_in_leaf,
  const data_size_t leaf_data_start,
  // for leaf information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  // gather information for CPU, used for launching kernels
  data_size_t* left_leaf_num_data,
  data_size_t* right_leaf_num_data,
  data_size_t* left_leaf_start,
  data_size_t* right_leaf_start,
  double* left_leaf_sum_of_hessians,
  double* right_leaf_sum_of_hessians,
  double* left_leaf_sum_of_gradients,
  double* right_leaf_sum_of_gradients,
  data_size_t* global_left_leaf_num_data,
  data_size_t* global_right_leaf_num_data,
  const bool point_structs_at_main,
  const int deferred_slot) {
  CalcBlockDim(num_data_in_leaf);
  // CalcBlockDim is non-monotonic in num_data_in_leaf: the per-block data count is
  // rounded up to a power of two, so a *smaller* leaf can require *more* blocks than
  // the full dataset does (e.g. grid(200)=50 but grid(140)=70 with SPLIT=1024). The
  // per-block offset buffers, however, were sized once for the full-data grid
  // (max_num_split_indices_blocks_ = grid_dim_ at construction / ResetTrainingData).
  // A leaf whose grid_dim_ exceeds that capacity -- e.g. the bagged root
  // (~bagging_fraction * num_data), or any non-bagged leaf in the ~101-160 range on a
  // dataset whose full-data grid is smaller -- makes GenDataToLeftBitVectorKernel's
  // PrepareOffset write block_to_{left,right}_offset_buffer[blockIdx.x + 1] past the
  // end of cuda_block_data_to_{left,right}_offset_. compute-sanitizer flags this as an
  // out-of-bounds __global__ write; depending on the device allocation layout it
  // silently corrupts adjacent memory or faults outright. Grow the buffers on demand
  // here, mirroring the resize logic in ResetTrainingData.
  if (grid_dim_ > max_num_split_indices_blocks_) {
    max_num_split_indices_blocks_ = grid_dim_;
    // the batched level apply (SplitLevelBatched) may already have grown the
    // buffers past this size; never shrink them back
    if (cuda_block_data_to_left_offset_.Size() < static_cast<size_t>(max_num_split_indices_blocks_) + 1) {
      cuda_block_data_to_left_offset_.Resize(static_cast<size_t>(max_num_split_indices_blocks_) + 1);
      cuda_block_data_to_right_offset_.Resize(static_cast<size_t>(max_num_split_indices_blocks_) + 1);
      SetCUDAMemory<data_size_t>(cuda_block_data_to_left_offset_.RawData(), 0, cuda_block_data_to_left_offset_.Size(), __FILE__, __LINE__);
      SetCUDAMemory<data_size_t>(cuda_block_data_to_right_offset_.RawData(), 0, cuda_block_data_to_right_offset_.Size(), __FILE__, __LINE__);
    }
  }
  // the previous split's CopyDataIndicesKernel (stream 2) may still be reading the
  // shared scratch this split's kernels overwrite; order after it
  if (indices_copy_done_event_ != nullptr) {
    CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[0], indices_copy_done_event_, 0));
    CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[1], indices_copy_done_event_, 0));
    CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(cuda_streams_[3], indices_copy_done_event_, 0));
  }
  global_timer.Start("GenDataToLeftBitVector");
  GenDataToLeftBitVector(num_data_in_leaf,
                         leaf_best_split_feature,
                         leaf_best_split_threshold,
                         categorical_bitset,
                         categorical_bitset_len,
                         leaf_best_split_default_left,
                         leaf_data_start,
                         left_leaf_index,
                         right_leaf_index);
  global_timer.Stop("GenDataToLeftBitVector");
  global_timer.Start("SplitInner");

  SplitInner(num_data_in_leaf,
             best_split_info,
             left_leaf_index,
             right_leaf_index,
             smaller_leaf_splits,
             larger_leaf_splits,
             left_leaf_num_data,
             right_leaf_num_data,
             left_leaf_start,
             right_leaf_start,
             left_leaf_sum_of_hessians,
             right_leaf_sum_of_hessians,
             left_leaf_sum_of_gradients,
             right_leaf_sum_of_gradients,
             global_left_leaf_num_data,
             global_right_leaf_num_data,
             point_structs_at_main,
             deferred_slot,
             leaf_data_start);
  global_timer.Stop("SplitInner");
}

void CUDADataPartition::GenDataToLeftBitVector(
    const data_size_t num_data_in_leaf,
    const int split_feature_index,
    const uint32_t split_threshold,
    const uint32_t* categorical_bitset,
    const int categorical_bitset_len,
    const uint8_t split_default_left,
    const data_size_t leaf_data_start,
    const int left_leaf_index,
    const int right_leaf_index) {
  if (is_categorical_feature_[split_feature_index]) {
    LaunchGenDataToLeftBitVectorCategoricalKernel(
      num_data_in_leaf,
      split_feature_index,
      categorical_bitset,
      categorical_bitset_len,
      split_default_left,
      leaf_data_start,
      left_leaf_index,
      right_leaf_index);
  } else {
    LaunchGenDataToLeftBitVectorKernel(
      num_data_in_leaf,
      split_feature_index,
      split_threshold,
      split_default_left,
      leaf_data_start,
      left_leaf_index,
      right_leaf_index);
  }
}

void CUDADataPartition::EnsurePinnedSplitInfoCapacity(const size_t num_ints) {
  if (pinned_split_info_size_ >= num_ints) {
    return;
  }
  if (pinned_split_info_ != nullptr) {
    CUDASUCCESS_OR_FATAL(cudaFreeHost(pinned_split_info_));
  }
  const size_t capacity = std::max(num_ints, static_cast<size_t>(num_leaves_ / 2 + 2) * 18);
  CUDASUCCESS_OR_FATAL(cudaHostAlloc(reinterpret_cast<void**>(&pinned_split_info_),
    capacity * sizeof(int), cudaHostAllocDefault));
  pinned_split_info_size_ = capacity;
}

void CUDADataPartition::FinishSplitBatch(const int num_splits, std::vector<int>* out) {
  out->resize(static_cast<size_t>(num_splits) * 18);
  // synchronous D2H on the legacy default stream: implicitly waits for all
  // preceding work on the (blocking) streams, so no explicit device sync
  // needed. Staged through a PINNED buffer (once-per-level critical path; a
  // pageable sync D2H pays an extra driver staging round trip).
  const size_t num_ints = static_cast<size_t>(num_splits) * 18;
  EnsurePinnedSplitInfoCapacity(num_ints);
  CopyFromCUDADeviceToHost<int>(pinned_split_info_, cuda_split_info_buffer_.RawData(),
    num_ints, __FILE__, __LINE__);
  std::memcpy(out->data(), pinned_split_info_, num_ints * sizeof(int));
}

#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
void CUDADataPartition::PrefetchSplitBatchAsync(const int max_splits, cudaStream_t stream) {
  const size_t num_ints = static_cast<size_t>(max_splits) * 18;
  EnsurePinnedSplitInfoCapacity(num_ints);
  CopyFromCUDADeviceToHostAsync<int>(pinned_split_info_, cuda_split_info_buffer_.RawData(),
    num_ints, stream, __FILE__, __LINE__);
}

void CUDADataPartition::ReadPrefetchedSplitBatch(const int num_splits, std::vector<int>* out) const {
  const size_t num_ints = static_cast<size_t>(num_splits) * 18;
  out->resize(num_ints);
  std::memcpy(out->data(), pinned_split_info_, num_ints * sizeof(int));
}
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED

void CUDADataPartition::SetLeafDataLayout(const std::vector<data_size_t>& leaf_num_data,
                                          const std::vector<data_size_t>& leaf_data_start,
                                          int num_leaves) {
  if (num_leaves <= 0) {
    return;
  }
  std::vector<data_size_t> leaf_data_end(num_leaves);
  for (int i = 0; i < num_leaves; ++i) {
    leaf_data_end[i] = leaf_data_start[i] + leaf_num_data[i];
  }
  CopyFromHostToCUDADevice<data_size_t>(cuda_leaf_num_data_.RawData(), leaf_num_data.data(),
    static_cast<size_t>(num_leaves), __FILE__, __LINE__);
  CopyFromHostToCUDADevice<data_size_t>(cuda_leaf_data_start_.RawData(), leaf_data_start.data(),
    static_cast<size_t>(num_leaves), __FILE__, __LINE__);
  CopyFromHostToCUDADevice<data_size_t>(cuda_leaf_data_end_.RawData(), leaf_data_end.data(),
    static_cast<size_t>(num_leaves), __FILE__, __LINE__);
}

#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
void CUDADataPartition::BuildHybridGraphFeatureMeta(
    std::vector<CUDAHybridGraphFeatureMeta>* meta) const {
  // static half of SplitLevelBatched's per-split host metadata math; the graph
  // controller replays the arithmetic on-device from these fields
  meta->resize(static_cast<size_t>(num_features_));
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    CUDAHybridGraphFeatureMeta& m = (*meta)[feature_index];
    const bool is_single_feature_in_column = is_single_feature_in_column_[feature_index];
    m.default_bin = cuda_column_data_->feature_default_bin(feature_index);
    m.most_freq_bin = cuda_column_data_->feature_most_freq_bin(feature_index);
    m.min_bin = is_single_feature_in_column ? 1 : cuda_column_data_->feature_min_bin(feature_index);
    m.max_bin = cuda_column_data_->feature_max_bin(feature_index);
    m.real_feature_index = 0;      // filled by the learner (train data view)
    m.missing_type = 0;            // filled by the learner
    m.real_threshold_offset = 0;   // filled by the learner
    m.missing_is_zero = cuda_column_data_->feature_missing_is_zero(feature_index);
    m.missing_is_na = cuda_column_data_->feature_missing_is_na(feature_index);
    m.mfb_is_zero = cuda_column_data_->feature_mfb_is_zero(feature_index);
    m.mfb_is_na = cuda_column_data_->feature_mfb_is_na(feature_index);
    m.use_min_bin = is_single_feature_in_column ? 0 : 1;
  }
}

void CUDADataPartition::BuildHybridGraphFeatureSource(
    std::vector<CUDAHybridGraphFeatureSource>* source) const {
  // per-tree half: the column data source (the packed compact view is a
  // per-tree, double-buffered gather); mirrors SplitLevelBatched exactly
  source->resize(static_cast<size_t>(num_features_));
  const bool packed_view = cuda_column_data_->packed_column_view_active();
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    CUDAHybridGraphFeatureSource& s = (*source)[feature_index];
    const int column_index = cuda_column_data_->feature_to_column(feature_index);
    if (packed_view) {
      s.column_data = cuda_column_data_->packed_column_data(column_index);
      s.packed_row_stride = cuda_column_data_->packed_column_stride(column_index);
      s.packed_shift = cuda_column_data_->packed_column_shift(column_index);
      s.bit_type = 4;
    } else {
      s.column_data = cuda_column_data_->GetColumnData(column_index);
      s.packed_row_stride = 0;
      s.packed_shift = 0;
      s.bit_type = cuda_column_data_->column_bit_type(column_index);
    }
  }
}
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED

void CUDADataPartition::SplitLevelBatched(const std::vector<CUDAHybridApplySplitInput>& splits) {
  const int num_splits = static_cast<int>(splits.size());
  if (num_splits <= 0) {
    return;
  }
  CHECK_LE(num_splits, 65535);  // grid y-dimension limit
  global_timer.Start("CUDADataPartition::SplitLevelBatched");
  host_apply_descs_.resize(splits.size());
  data_size_t total_block_offset_slots = 0;
  int max_num_blocks = 0;
  uint32_t cat_arena_cursor = 0;
  std::vector<int> cat_desc_indices;
  std::vector<uint32_t> cat_mfb_bins;
  std::vector<uint32_t> cat_word_offsets;
  for (int k = 0; k < num_splits; ++k) {
    const CUDAHybridApplySplitInput& in = splits[k];
    CUDAHybridApplyDescriptor& desc = host_apply_descs_[k];
    const int split_feature_index = in.split_feature;
    const bool is_cat = is_categorical_feature_[split_feature_index];
    // host-side split metadata math, mirroring LaunchGenDataToLeftBitVectorKernel
    const bool missing_is_zero = static_cast<bool>(cuda_column_data_->feature_missing_is_zero(split_feature_index));
    const bool missing_is_na = static_cast<bool>(cuda_column_data_->feature_missing_is_na(split_feature_index));
    const bool mfb_is_zero = static_cast<bool>(cuda_column_data_->feature_mfb_is_zero(split_feature_index));
    const bool mfb_is_na = static_cast<bool>(cuda_column_data_->feature_mfb_is_na(split_feature_index));
    const bool is_single_feature_in_column = is_single_feature_in_column_[split_feature_index];
    const uint32_t default_bin = cuda_column_data_->feature_default_bin(split_feature_index);
    const uint32_t most_freq_bin = cuda_column_data_->feature_most_freq_bin(split_feature_index);
    const uint32_t min_bin = is_single_feature_in_column ? 1 : cuda_column_data_->feature_min_bin(split_feature_index);
    const uint32_t max_bin = cuda_column_data_->feature_max_bin(split_feature_index);
    uint32_t th = in.split_threshold + min_bin;
    uint32_t t_zero_bin = min_bin + default_bin;
    if (most_freq_bin == 0) {
      --th;
      --t_zero_bin;
    }
    uint8_t split_default_to_left = 0;
    uint8_t split_missing_default_to_left = 0;
    int default_leaf_index = in.right_leaf_index;
    int missing_default_leaf_index = in.right_leaf_index;
    if (most_freq_bin <= in.split_threshold) {
      split_default_to_left = 1;
      default_leaf_index = in.left_leaf_index;
    }
    if (missing_is_zero || missing_is_na) {
      if (in.split_default_left) {
        split_missing_default_to_left = 1;
        missing_default_leaf_index = in.left_leaf_index;
      }
    }
    const int column_index = cuda_column_data_->feature_to_column(split_feature_index);
    if (cuda_column_data_->packed_column_view_active()) {
      // 4-bit packed compact source (FALCATA_SPLIT_PACKED_READ): read the
      // split column's nibbles straight from the histogram constructor's packed
      // compact matrix instead of a per-tree column-major gather
      desc.column_data = cuda_column_data_->packed_column_data(column_index);
      desc.packed_row_stride = cuda_column_data_->packed_column_stride(column_index);
      desc.packed_shift = cuda_column_data_->packed_column_shift(column_index);
      CHECK(desc.column_data != nullptr);
    } else {
      desc.column_data = cuda_column_data_->GetColumnData(column_index);
      desc.packed_row_stride = 0;
      desc.packed_shift = 0;
    }
    desc.best_split_info = in.best_split_info;
    desc.smaller_leaf_splits = in.smaller_leaf_splits;
    desc.larger_leaf_splits = in.larger_leaf_splits;
    desc.leaf_data_start = in.leaf_data_start;
    desc.num_data_in_leaf = in.num_data_in_leaf;
    desc.block_offset_start = total_block_offset_slots;
    desc.num_blocks = (in.num_data_in_leaf + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) /
      SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
    desc.left_leaf_index = in.left_leaf_index;
    desc.right_leaf_index = in.right_leaf_index;
    desc.th = th;
    desc.t_zero_bin = t_zero_bin;
    desc.max_bin = max_bin;
    desc.min_bin = min_bin;
    desc.default_leaf_index = default_leaf_index;
    desc.missing_default_leaf_index = missing_default_leaf_index;
    desc.split_default_to_left = split_default_to_left;
    desc.split_missing_default_to_left = split_missing_default_to_left;
    desc.bit_type = cuda_column_data_->packed_column_view_active() ?
      4 : cuda_column_data_->column_bit_type(column_index);
    desc.min_is_max = (min_bin < max_bin) ? 0 : 1;
    desc.missing_is_zero = missing_is_zero ? 1 : 0;
    desc.missing_is_na = missing_is_na ? 1 : 0;
    desc.mfb_is_zero = mfb_is_zero ? 1 : 0;
    desc.mfb_is_na = mfb_is_na ? 1 : 0;
    desc.max_bin_to_left = (max_bin <= th) ? 1 : 0;
    desc.use_min_bin = is_single_feature_in_column ? 0 : 1;
    if (is_cat) {
      // categorical split: region of the per-level inner-bitset arena, sized
      // by the bin-range cap (trailing zero words are inert for the bitset
      // test, so no device length is ever read back). The default-direction
      // fields are patched ON DEVICE by the arena-build kernel (MFB
      // membership needs the bitset; the classic flow pays a D2H here).
      CHECK(!cuda_column_data_->packed_column_view_active());
      const int8_t mfb_offset = static_cast<int8_t>(most_freq_bin == 0);
      const uint32_t max_pos = max_bin - min_bin + static_cast<uint32_t>(mfb_offset);
      desc.cat_bitset_len = static_cast<int>(max_pos / 32 + 1);
      desc.cat_bitset = nullptr;  // patched below once the arena is sized
      desc.cat_mfb_offset = mfb_offset;
      cat_word_offsets.push_back(cat_arena_cursor);
      cat_arena_cursor += static_cast<uint32_t>(desc.cat_bitset_len);
      cat_desc_indices.push_back(k);
      cat_mfb_bins.push_back(most_freq_bin);
    } else {
      desc.cat_bitset_len = 0;
      desc.cat_bitset = nullptr;
      desc.cat_mfb_offset = 0;
    }
    total_block_offset_slots += desc.num_blocks + 1;
    if (desc.num_blocks > max_num_blocks) {
      max_num_blocks = desc.num_blocks;
    }
  }
  // The split kernels write each split leaf's partitioned indices into the leaf's
  // own window of cuda_out_data_indices_in_leaf_ (mirroring the main array
  // layout); instead of copying every window back we SWAP the two buffers after
  // the launches, making the out buffer the new main index array. Regions owned
  // by leaves that are NOT split this level (terminal leaves) must be carried
  // over explicitly: append one copy descriptor per gap in the split regions'
  // coverage of [0, root_num_data()). Terminal regions are typically a small
  // fraction of the data, so this replaces a full-size copy with a sparse one.
  int num_gaps = 0;
  int max_gap_blocks = 0;
  {
    std::vector<std::pair<data_size_t, data_size_t>> regions;  // (start, num)
    regions.reserve(splits.size());
    for (const CUDAHybridApplySplitInput& in : splits) {
      regions.emplace_back(in.leaf_data_start, in.num_data_in_leaf);
    }
    std::sort(regions.begin(), regions.end());
    data_size_t cursor = 0;
    const data_size_t num_data_total = root_num_data();
    auto append_gap = [this, &num_gaps, &max_gap_blocks](const data_size_t start, const data_size_t num) {
      CUDAHybridApplyDescriptor desc;
      std::memset(&desc, 0, sizeof(desc));
      desc.leaf_data_start = start;
      desc.num_data_in_leaf = num;
      desc.num_blocks = (num + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) /
        SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
      if (desc.num_blocks > max_gap_blocks) {
        max_gap_blocks = desc.num_blocks;
      }
      host_apply_descs_.push_back(desc);
      ++num_gaps;
    };
    for (const std::pair<data_size_t, data_size_t>& region : regions) {
      if (region.first > cursor) {
        append_gap(cursor, region.first - cursor);
      }
      cursor = region.first + region.second;
    }
    if (cursor < num_data_total) {
      append_gap(cursor, num_data_total - cursor);
    }
  }
  // per-split regions of the level-sized block offset buffers
  if (cuda_block_data_to_left_offset_.Size() < static_cast<size_t>(total_block_offset_slots)) {
    cuda_block_data_to_left_offset_.Resize(static_cast<size_t>(total_block_offset_slots));
    cuda_block_data_to_right_offset_.Resize(static_cast<size_t>(total_block_offset_slots));
    SetCUDAMemory<data_size_t>(cuda_block_data_to_left_offset_.RawData(), 0, cuda_block_data_to_left_offset_.Size(), __FILE__, __LINE__);
    SetCUDAMemory<data_size_t>(cuda_block_data_to_right_offset_.RawData(), 0, cuda_block_data_to_right_offset_.Size(), __FILE__, __LINE__);
  }
  const int num_descs = num_splits + num_gaps;
  if (cuda_apply_descs_.Size() < static_cast<size_t>(num_descs)) {
    // preallocate for the deepest possible level (splits + as many gaps) so
    // this resizes at most once
    cuda_apply_descs_.Resize(static_cast<size_t>(std::max(num_descs, num_leaves_ + 4)));
  }
  // single async H2D per level on cuda_streams_[0]: ordered after any batched
  // apply kernel of a previous level still reading the descriptor buffer (same
  // stream) and before the launches below. The host staging buffer is only
  // rewritten after the next FinishSplitBatch full sync, and a pageable async
  // H2D returns only once the data is staged, so reuse is safe.
  if (cat_arena_cursor > 0) {
    if (cuda_cat_bitset_arena_.Size() < static_cast<size_t>(cat_arena_cursor)) {
      cuda_cat_bitset_arena_.Resize(static_cast<size_t>(cat_arena_cursor) * 2);
    }
    for (size_t i = 0; i < cat_desc_indices.size(); ++i) {
      host_apply_descs_[cat_desc_indices[i]].cat_bitset =
        cuda_cat_bitset_arena_.RawData() + cat_word_offsets[i];
    }
  }
  CopyFromHostToCUDADeviceAsync<CUDAHybridApplyDescriptor>(
    cuda_apply_descs_.RawData(), host_apply_descs_.data(),
    static_cast<size_t>(num_descs), cuda_streams_[0], __FILE__, __LINE__);
  if (!cat_desc_indices.empty()) {
    // build the level's bitsets + patch descriptor default directions; on
    // stream 0, so ordered after the descriptor upload and before the
    // bit-vector kernels below
    LaunchBuildCatBitsetArenaKernel(static_cast<int>(cat_desc_indices.size()),
                                    cat_desc_indices, cat_mfb_bins);
  }
  LaunchSplitLevelBatchedKernels(num_splits, max_num_blocks, num_gaps, max_gap_blocks);
  // the out buffer now holds every leaf's indices at the main layout positions:
  // promote it to the main index array (the old main becomes the next scratch)
  cuda_data_indices_.Swap(&cuda_out_data_indices_in_leaf_);
  cur_num_leaves_ += num_splits;
  global_timer.Stop("CUDADataPartition::SplitLevelBatched");
}

void CUDADataPartition::SplitInner(
  const data_size_t num_data_in_leaf,
  const CUDASplitInfo* best_split_info,
  const int left_leaf_index,
  const int right_leaf_index,
  // for leaf splits information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  data_size_t* left_leaf_num_data,
  data_size_t* right_leaf_num_data,
  data_size_t* left_leaf_start,
  data_size_t* right_leaf_start,
  double* left_leaf_sum_of_hessians,
  double* right_leaf_sum_of_hessians,
  double* left_leaf_sum_of_gradients,
  double* right_leaf_sum_of_gradients,
  data_size_t* global_left_leaf_num_data,
  data_size_t* global_right_leaf_num_data,
  const bool point_structs_at_main,
  const int deferred_slot,
  const data_size_t leaf_data_start_for_copy) {
  LaunchSplitInnerKernel(
    num_data_in_leaf,
    best_split_info,
    left_leaf_index,
    right_leaf_index,
    smaller_leaf_splits,
    larger_leaf_splits,
    left_leaf_num_data,
    right_leaf_num_data,
    left_leaf_start,
    right_leaf_start,
    left_leaf_sum_of_hessians,
    right_leaf_sum_of_hessians,
    left_leaf_sum_of_gradients,
    right_leaf_sum_of_gradients,
    global_left_leaf_num_data,
    global_right_leaf_num_data,
    point_structs_at_main,
    deferred_slot,
    leaf_data_start_for_copy);
  ++cur_num_leaves_;
}

void CUDADataPartition::UpdateTrainScore(const Tree* tree, double* scores) {
  const CUDATree* cuda_tree = nullptr;
  std::unique_ptr<CUDATree> cuda_tree_ptr;
  if (tree->is_cuda_tree()) {
    cuda_tree = reinterpret_cast<const CUDATree*>(tree);
  } else {
    cuda_tree_ptr.reset(new CUDATree(tree));
    cuda_tree = cuda_tree_ptr.get();
  }
  if (use_bagging_) {
    // we need restore the order of indices in cuda_data_indices_
    CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_data_indices_.RawData(), used_indices_, static_cast<size_t>(num_used_indices_), __FILE__, __LINE__);
  }
  LaunchAddPredictionToScoreKernel(cuda_tree->cuda_leaf_value(), scores);
}

void CUDADataPartition::CalcBlockDim(const data_size_t num_data_in_leaf) {
  const int min_num_blocks = num_data_in_leaf <= 100 ? 1 : 80;
  const int num_blocks = std::max(min_num_blocks, (num_data_in_leaf + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION);
  int split_indices_block_size_data_partition = (num_data_in_leaf + num_blocks - 1) / num_blocks - 1;
  CHECK_GT(split_indices_block_size_data_partition, 0);
  int split_indices_block_size_data_partition_aligned = 1;
  while (split_indices_block_size_data_partition > 0) {
    split_indices_block_size_data_partition_aligned <<= 1;
    split_indices_block_size_data_partition >>= 1;
  }
  const int num_blocks_final = (num_data_in_leaf + split_indices_block_size_data_partition_aligned - 1) / split_indices_block_size_data_partition_aligned;
  grid_dim_ = num_blocks_final;
  block_dim_ = split_indices_block_size_data_partition_aligned;
}

void CUDADataPartition::SetUsedDataIndices(const data_size_t* used_indices, const data_size_t num_used_indices) {
  use_bagging_ = true;
  num_used_indices_ = num_used_indices;
  used_indices_ = used_indices;
  CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_data_indices_.RawData(), used_indices, static_cast<size_t>(num_used_indices), __FILE__, __LINE__);
  LaunchFillDataIndexToLeafIndex();
}

void CUDADataPartition::ResetTrainingData(const Dataset* train_data, const int num_total_bin, hist_t* cuda_hist) {
  const data_size_t old_num_data = num_data_;
  num_data_ = train_data->num_data();
  num_features_ = train_data->num_features();
  num_total_bin_ = num_total_bin;
  cuda_column_data_ = train_data->cuda_column_data();
  cuda_hist_ = cuda_hist;
  CopyFromHostToCUDADevice<hist_t*>(cuda_hist_pool_.RawData(), &cuda_hist_, 1, __FILE__, __LINE__);
  CopyFromHostToCUDADevice<int>(cuda_num_data_.RawData(), &num_data_, 1, __FILE__, __LINE__);
  if (num_data_ > old_num_data) {
    CalcBlockDim(num_data_);
    const int old_max_num_split_indices_blocks = max_num_split_indices_blocks_;
    max_num_split_indices_blocks_ = grid_dim_;
    if (max_num_split_indices_blocks_ > old_max_num_split_indices_blocks &&
        cuda_block_data_to_left_offset_.Size() < static_cast<size_t>(max_num_split_indices_blocks_) + 1) {
      cuda_block_data_to_left_offset_.Resize(static_cast<size_t>(max_num_split_indices_blocks_) + 1);
      cuda_block_data_to_right_offset_.Resize(static_cast<size_t>(max_num_split_indices_blocks_) + 1);
      SetCUDAMemory<data_size_t>(cuda_block_data_to_left_offset_.RawData(), 0, cuda_block_data_to_left_offset_.Size(), __FILE__, __LINE__);
      SetCUDAMemory<data_size_t>(cuda_block_data_to_right_offset_.RawData(), 0, cuda_block_data_to_right_offset_.Size(), __FILE__, __LINE__);
    }
    cuda_data_indices_.Resize(static_cast<size_t>(num_data_));
    cuda_block_to_left_offset_.Resize(static_cast<size_t>(num_data_));
    cuda_data_index_to_leaf_index_.Resize(static_cast<size_t>(num_data_));
    cuda_out_data_indices_in_leaf_.Resize(static_cast<size_t>(num_data_));
  }
  used_indices_ = nullptr;
  use_bagging_ = false;
  num_used_indices_ = 0;
  cur_num_leaves_ = 1;
}

void CUDADataPartition::ResetConfig(const Config* config, hist_t* cuda_hist) {
  num_threads_ = OMP_NUM_THREADS();
  num_leaves_ = config->num_leaves;
  cuda_hist_ = cuda_hist;
  cuda_leaf_data_start_.Resize(static_cast<size_t>(num_leaves_));
  cuda_leaf_data_end_.Resize(static_cast<size_t>(num_leaves_));
  cuda_leaf_num_data_.Resize(static_cast<size_t>(num_leaves_));
  cuda_hist_pool_.Resize(static_cast<size_t>(num_leaves_));
  cuda_leaf_output_.Resize(static_cast<size_t>(num_leaves_));
  cuda_split_info_buffer_.Resize(18 * static_cast<size_t>(num_leaves_) + 18);
  cuda_level_smaller_counts_.Resize(static_cast<size_t>(num_leaves_) + 1);
}

void CUDADataPartition::SetBaggingSubset(const Dataset* subset) {
  num_used_indices_ = subset->num_data();
  used_indices_ = nullptr;
  use_bagging_ = true;
  cuda_column_data_ = subset->cuda_column_data();
}

void CUDADataPartition::ResetByLeafPred(const std::vector<int>& leaf_pred, int num_leaves) {
  if (leaf_pred.size() != static_cast<size_t>(num_data_)) {
    cuda_data_index_to_leaf_index_.Clear();
    cuda_data_index_to_leaf_index_.InitFromHostVector(leaf_pred);
    num_data_ = static_cast<data_size_t>(leaf_pred.size());
  } else {
    CopyFromHostToCUDADevice<int>(cuda_data_index_to_leaf_index_.RawData(), leaf_pred.data(), leaf_pred.size(), __FILE__, __LINE__);
  }
  num_leaves_ = num_leaves;
  cur_num_leaves_ = num_leaves;
}

void CUDADataPartition::ReduceLeafGradStat(
  const score_t* gradients, const score_t* hessians,
  CUDATree* tree, double* leaf_grad_stat_buffer, double* leaf_hess_state_buffer) const {
  LaunchReduceLeafGradStat(gradients, hessians, tree, leaf_grad_stat_buffer, leaf_hess_state_buffer);
}

}  // namespace Falcata

#endif  // USE_CUDA
