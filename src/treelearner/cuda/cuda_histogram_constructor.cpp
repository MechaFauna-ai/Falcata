/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace Falcata {

namespace {

// FALCATA_CONSTRUCT_COMPACT_QUANT: enable the compact column view for quantized
// training (default ON). The compact view materializes ONLY the tree's sampled
// columns into a row-major-in-partition bin matrix and feeds the SAME discretized
// construct kernel with the compact metadata, so unused columns' zero/gather/merge
// work is skipped (numerai ff=0.1 -> ~90% of columns skipped). Integer-atomic
// accumulation is order-invariant and the per-used-column absolute hist offsets are
// preserved, so the histograms are BIT-IDENTICAL to the full-data path. Kill switch:
// FALCATA_CONSTRUCT_COMPACT_QUANT=0 -> quant falls back to the full-data kernel.
bool CompactQuantEnabled() {
  return FalcataPlan::Get().compact_quant;
}

constexpr size_t kMiB = size_t{1} << 20;

size_t ToMiB(const size_t bytes) { return (bytes + kMiB - 1) / kMiB; }

// colmajor_direct's diagnostic: Info under FALCATA_DEBUG=diag, Debug otherwise (as colmajor_fill's engagement line).
template <typename... Args>
void LogDirectView(const char* fmt, Args... args) {
  if (FalcataDebug().diag) {
    Log::Info(fmt, args...);
  } else {
    Log::Debug(fmt, args...);
  }
}

}  // namespace

CUDAHistogramConstructor::CUDAHistogramConstructor(
  const Dataset* train_data,
  const int num_leaves,
  const int num_threads,
  const std::vector<uint32_t>& feature_hist_offsets,
  const int min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const int gpu_device_id,
  const bool gpu_use_dp,
  const bool use_quantized_grad,
  const int num_grad_quant_bins,
  const double feature_fraction):
  num_data_(train_data->num_data()),
  num_features_(train_data->num_features()),
  num_leaves_(num_leaves),
  num_threads_(num_threads),
  min_data_in_leaf_(min_data_in_leaf),
  feature_fraction_(feature_fraction),
  min_sum_hessian_in_leaf_(min_sum_hessian_in_leaf),
  gpu_device_id_(gpu_device_id),
  gpu_use_dp_(gpu_use_dp),
  use_quantized_grad_(use_quantized_grad),
  num_grad_quant_bins_(num_grad_quant_bins) {
  num_compact_columns_ = 0;
  max_num_compact_cols_per_partition_ = 0;
  use_compact_view_ = false;
  InitFeatureMetaInfo(train_data, feature_hist_offsets);
  cuda_row_data_.reset(nullptr);
}

CUDAHistogramConstructor::~CUDAHistogramConstructor() {
  gpuAssert(cudaStreamDestroy(cuda_stream_), __FILE__, __LINE__);
  if (prefetch_stream_ != nullptr) {
    gpuAssert(cudaStreamDestroy(prefetch_stream_), __FILE__, __LINE__);
  }
  if (fill_done_event_alt_ != nullptr) {
    gpuAssert(cudaEventDestroy(fill_done_event_alt_), __FILE__, __LINE__);
  }
  if (prefill_pinned_ != nullptr) {
    gpuAssert(cudaFreeHost(prefill_pinned_), __FILE__, __LINE__);
  }
  if (construct_done_event_ != nullptr) {
    gpuAssert(cudaEventDestroy(construct_done_event_), __FILE__, __LINE__);
  }
  if (subtract_done_event_ != nullptr) {
    gpuAssert(cudaEventDestroy(subtract_done_event_), __FILE__, __LINE__);
  }
}

void CUDAHistogramConstructor::InitFeatureMetaInfo(const Dataset* train_data, const std::vector<uint32_t>& feature_hist_offsets) {
  need_fix_histogram_features_.clear();
  need_fix_histogram_features_num_bin_aligend_.clear();
  feature_num_bins_.clear();
  feature_most_freq_bins_.clear();
  has_categorical_feature_ = false;
  for (int feature_index = 0; feature_index < train_data->num_features(); ++feature_index) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(feature_index);
    if (bin_mapper->bin_type() == BinType::CategoricalBin) {
      has_categorical_feature_ = true;
    }
    const uint32_t most_freq_bin = bin_mapper->GetMostFreqBin();
    if (most_freq_bin != 0) {
      need_fix_histogram_features_.emplace_back(feature_index);
      uint32_t num_bin_ref = static_cast<uint32_t>(bin_mapper->num_bin()) - 1;
      uint32_t num_bin_aligned = 1;
      while (num_bin_ref > 0) {
        num_bin_aligned <<= 1;
        num_bin_ref >>= 1;
      }
      need_fix_histogram_features_num_bin_aligend_.emplace_back(num_bin_aligned);
    }
    feature_num_bins_.emplace_back(static_cast<uint32_t>(bin_mapper->num_bin()));
    feature_most_freq_bins_.emplace_back(most_freq_bin);
  }
  feature_hist_offsets_.clear();
  for (size_t i = 0; i < feature_hist_offsets.size(); ++i) {
    feature_hist_offsets_.emplace_back(feature_hist_offsets[i]);
  }
  if (feature_hist_offsets.empty()) {
    num_total_bin_ = 0;
  } else {
    num_total_bin_ = static_cast<int>(feature_hist_offsets.back());
  }
  // register-accumulation construct body (batched compact path only): usable
  // when EVERY feature fits the register bin cap (see kRegHistMaxBins == 8);
  // FALCATA_BATCH_REGHIST=0 disables it
  const bool reg_hist_enabled = FalcataPlan::Get().batch_reghist;
  uint32_t max_num_bin = 0;
  for (const uint32_t num_bin : feature_num_bins_) {
    if (num_bin > max_num_bin) {
      max_num_bin = num_bin;
    }
  }
  construct_reg_bins_ = reg_hist_enabled && max_num_bin <= 8 && !feature_num_bins_.empty();
}

void LaunchInterleaveGradHessKernel(
  const score_t* gradients,
  const score_t* hessians,
  float2* gradients_hessians,
  data_size_t num_data);

void CUDAHistogramConstructor::BeforeTrain(const score_t* gradients, const score_t* hessians) {
  cuda_gradients_ = gradients;
  cuda_hessians_ = hessians;
  // every tree starts on plane 0, whose hessians are the ones every consumer reads
  grad_only_plane_ = false;
  if (l2_carveout_bytes_ > 0 && gradients != nullptr) {
    // Pin the per-row gradient buffer (quant: packed int32 grad/hess pairs read
    // once per level per row through data_indices gather) as L2-persisting on
    // every construct stream. One window per stream is the API limit; the
    // gradient buffer is the highest-reuse scatter target.
    cudaStreamAttrValue attr{};
    const size_t bytes = std::min(l2_max_window_bytes_,
      static_cast<size_t>(num_data_) * (use_quantized_grad_ ? sizeof(int32_t) : 2 * sizeof(score_t)));
    attr.accessPolicyWindow.base_ptr = const_cast<score_t*>(gradients);
    attr.accessPolicyWindow.num_bytes = bytes;
    attr.accessPolicyWindow.hitRatio =
      bytes > l2_carveout_bytes_ ? static_cast<float>(l2_carveout_bytes_) / bytes : 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    for (int p = 0; p < kNumHistPipelines; ++p) {
      cudaStreamSetAttribute(pipeline_streams_[p], cudaStreamAttributeAccessPolicyWindow, &attr);
    }
  }
  // interleave (gradient, hessian) into float2 pairs for the dense construct
  // kernels (one scattered 32B sector per row instead of two). Launched on the
  // legacy default stream, so it orders before the construct kernels on the
  // (blocking) histogram streams; a trivial streaming kernel (~0.05ms at 5M rows).
  gh_interleave_valid_ = false;
  if (!use_quantized_grad_ && GHInterleaveEnabled() && num_hist_planes_ == 1 &&
      gradients != nullptr && hessians != nullptr) {
    if (cuda_gradients_hessians_.Size() < static_cast<size_t>(num_data_)) {
      cuda_gradients_hessians_.Resize(static_cast<size_t>(num_data_));
    }
    LaunchInterleaveGradHessKernel(gradients, hessians,
                                   cuda_gradients_hessians_.RawData(), num_data_);
    gh_interleave_valid_ = true;
  }
  // async memset on the legacy default stream: the construct kernels on the
  // (blocking) histogram streams implicitly order after it, so no device sync
  // is needed (SetValue would pay a full device sync on every tree).
  // Only the leaf slots the previous tree actually used can be dirty (leaf k's
  // histogram lives at slot k, and every tree assigns slots [0, num_leaves)
  // consecutively), so zero just those: with num_leaves=1023 buffers trees of
  // ~100 leaves this cuts >100MB (~65us of GPU memset) per tree.
  const size_t num_slots_to_zero = num_dirty_leaves_ < 0 ?
    static_cast<size_t>(num_leaves_) :
    std::min(static_cast<size_t>(num_dirty_leaves_), static_cast<size_t>(num_leaves_));
  CUDASUCCESS_OR_FATAL(cudaMemset(reinterpret_cast<void*>(cuda_hist_.RawData()), 0,
    num_slots_to_zero * static_cast<size_t>(2 * num_total_bin_) *
    static_cast<size_t>(num_hist_planes_) * sizeof(hist_t)));
}

void CUDAHistogramConstructor::SelectGradientPlane(const score_t* gradients, const score_t* hessians,
                                                   const bool grad_only) {
  // vector-leaf plane switch: pointer swap only. The interleaved float2 copy is
  // disabled in plane mode (BeforeTrain) and the per-tree histogram zeroing
  // already covered every plane, so neither is redone here.
  cuda_gradients_ = gradients;
  cuda_hessians_ = hessians;
  grad_only_plane_ = grad_only;
}

void CUDAHistogramConstructor::ZeroHistForLeaf(int /*leaf_index*/) {
  // No-op: BeforeTrain zeroes the entire cuda_hist_ buffer.
}

void CUDAHistogramConstructor::ZeroHistSlots(const std::vector<int>& slots) {
  const size_t slot_size = static_cast<size_t>(2 * num_total_bin_) *
    static_cast<size_t>(num_hist_planes_);
  for (const int slot : slots) {
    CUDASUCCESS_OR_FATAL(cudaMemsetAsync(
      reinterpret_cast<void*>(cuda_hist_.RawData() + static_cast<size_t>(slot) * slot_size), 0,
      slot_size * sizeof(hist_t)));
  }
}

void CUDAHistogramConstructor::SetFeatureUsedBytree(const std::vector<int8_t>& is_feature_used_bytree) {
  if (cuda_is_feature_used_bytree_.Size() != is_feature_used_bytree.size()) {
    cuda_is_feature_used_bytree_.Resize(is_feature_used_bytree.size());
  }
  UploadTreeStartMeta<int8_t>(cuda_is_feature_used_bytree_.RawData(), is_feature_used_bytree.data(),
                              is_feature_used_bytree.size());
  // per-tree bin-level used mask for the batched fix/subtract/construct-merge
  // kernels: with feature_fraction sampling, ~ (1 - fraction) of every leaf
  // histogram belongs to features no kernel of this tree will ever read, so the
  // elementwise batched kernels skip them. nullptr (no sampling) keeps every
  // kernel byte-identical to the unmasked behavior.
  any_feature_unused_bytree_ = false;
  const int mask_features = std::min(num_features_, static_cast<int>(is_feature_used_bytree.size()));
  for (int f = 0; f < mask_features; ++f) {
    if (!is_feature_used_bytree[f]) {
      any_feature_unused_bytree_ = true;
      break;
    }
  }
  if (any_feature_unused_bytree_) {
    host_bin_used_bytree_.assign(static_cast<size_t>(num_total_bin_), 0);
    for (int f = 0; f < mask_features; ++f) {
      if (!is_feature_used_bytree[f]) {
        continue;
      }
      const uint32_t bin_start = feature_hist_offsets_[f];
      const uint32_t bin_end = f + 1 < static_cast<int>(feature_hist_offsets_.size()) ?
        feature_hist_offsets_[f + 1] : static_cast<uint32_t>(num_total_bin_);
      for (uint32_t bin = bin_start; bin < bin_end && bin < static_cast<uint32_t>(num_total_bin_); ++bin) {
        host_bin_used_bytree_[bin] = 1;
      }
    }
    if (cuda_bin_used_bytree_.Size() < static_cast<size_t>(num_total_bin_)) {
      cuda_bin_used_bytree_.Resize(static_cast<size_t>(num_total_bin_));
    }
    UploadTreeStartMeta<uint8_t>(cuda_bin_used_bytree_.RawData(), host_bin_used_bytree_.data(),
                                 host_bin_used_bytree_.size());
  }
}

void LaunchTransposeColMajorToRowMajor(
    cudaStream_t stream,
    const uint8_t* staging,
    uint8_t* compact_data,
    const int* partition_for_compact,
    const int* compact_partition_column_offsets,
    data_size_t num_data,
    int total_compact_cols);

// Implemented in cuda_histogram_constructor.cu — does the kernel launch.
void LaunchFillCompactDataKernel(
  cudaStream_t stream,
  const uint8_t* src_data,
  uint8_t* compact_data,
  const size_t* slot_src_byte,
  const int* slot_src_stride,
  const size_t* slot_dst_byte,
  const int* slot_dst_stride,
  int total_compact_cols,
  data_size_t num_data,
  uint8_t* colmajor_out);

// Implemented in cuda_histogram_constructor.cu — 4-bit packed source AND
// destination variant (one thread per destination byte = compact column pair).
void LaunchTransposeToColMajorNibbleKernel(
  const uint8_t* src_data,
  uint8_t* colmajor_data,
  const size_t* col_src_nib_base,
  const int* col_src_stride_nib,
  int num_columns,
  data_size_t num_data,
  size_t num_data_pad);

// Implemented in cuda_histogram_constructor.cu: zero the nibble past the last row of each column (odd row counts).
void LaunchClearColMajorPadNibbles(uint8_t* colmajor, int num_columns, size_t column_bytes);

void LaunchFillCompactCodecKernel(
  cudaStream_t stream,
  PackCodecId codec,
  const uint8_t* src_data,
  uint8_t* compact_data,
  const size_t* col_src_nib_base,
  const int* col_src_stride_nib,
  const size_t* ws_dst_byte,
  const int* ws_dst_stride,
  const int* ws_first_col,
  const uint8_t* ws_num_digits,
  int total_word_slots,
  data_size_t num_data);

void LaunchFillCompactData4BitKernel(
  cudaStream_t stream,
  const uint8_t* src_data,
  uint8_t* compact_data,
  const size_t* bs_src_nib0,
  const size_t* bs_src_nib1,
  const int* bs_src_stride_nib,
  const size_t* bs_dst_byte,
  const int* bs_dst_stride,
  int total_byte_slots,
  data_size_t num_data,
  bool tiled);

bool LaunchFillCompactData4BitTiledRootHistKernel(
  cudaStream_t stream,
  const uint8_t* src_data,
  uint8_t* compact_data,
  const size_t* bs_src_nib0,
  const size_t* bs_src_nib1,
  const size_t* bs_dst_byte,
  const int* bs_dst_stride,
  int total_byte_slots,
  data_size_t num_data,
  const int* slot_first_bin,
  const uint32_t* local_bin_hist_pos,
  int num_local_bins,
  int slot_major_span,
  bool strided_tiles,
  bool run_copy,
  bool word_prefetch,
  int max_rows_per_block,
  const int32_t* grad_and_hess,
  bool hist_16bit,
  hist_t* root_hist_scratch,
  size_t root_hist_scratch_bytes,
  const uint32_t* dense_data,
  size_t dense_column_nibbles,
  size_t dense_pitch_words,
  int dense_codec);
void LaunchColMajorAnyAtLeast6(const uint8_t* store, size_t num_bytes, int* found);
void LaunchEncodeColMajorDense(const uint8_t* colmajor, size_t column_bytes, int num_columns, data_size_t num_data,
                               uint32_t* dense, size_t pitch_words, int codec, int* column_wide);
size_t ColMajorDensePitchWords(data_size_t num_data, int codec);

// The tiled 4-bit fill needs a column-major source (stride-1 nibble runs with even bases) and every partition laid
// out as a run of W consecutive byte slots whose destination bytes are contiguous with stride W.
static bool CompactFill4BitTiledEligible(const std::vector<size_t>& nib0, const std::vector<size_t>& nib1,
                                         const std::vector<int>& stride, const std::vector<size_t>& dst_byte,
                                         const std::vector<int>& dst_stride) {
#if defined(__HIP_PLATFORM_AMD__)
  // ROCm builds have no tiled kernel (see LaunchFillCompactData4BitKernel)
  (void)nib0; (void)nib1; (void)stride; (void)dst_byte; (void)dst_stride;
  return false;
#else
  const int n = static_cast<int>(nib0.size());
  if (!FalcataPlan::Get().tiled_fill || n == 0 || n > kFill4BitTiledMaxSlots) return false;
  const size_t kNone = ~static_cast<size_t>(0);
  for (int i = 0; i < n; ++i) {
    if (stride[i] != 1 || (nib0[i] & 1) != 0 || (nib1[i] != kNone && (nib1[i] & 1) != 0)) return false;
  }
  for (int s = 0; s < n;) {
    const int width = dst_stride[s];
    if (width < 1 || s + width > n) return false;
    for (int m = 0; m < width; ++m) {
      if (dst_byte[s + m] != dst_byte[s] + static_cast<size_t>(m) || dst_stride[s + m] != width) return false;
    }
    s += width;
  }
  return true;
#endif
}

bool CUDAHistogramConstructor::ScanCompactLayout(
    const std::vector<int8_t>& is_feature_used_bytree, CompactLayout* layout) const {
  // Gate: only support the standard dense path (uint8 bins, no large-bin partitions, no sparse).
  // This is what our Numerai workload uses; other paths fall back to the full kernel.
  if (cuda_row_data_->is_sparse() || cuda_row_data_->bit_type() != 8 ||
      cuda_row_data_->NumLargeBinPartition() > 0 ||
      (use_quantized_grad_ && !CompactQuantEnabled())) {
    return false;
  }
  if (is_feature_used_bytree.empty()) return false;

  // Read partition info from CUDARowData (host-side).
  const std::vector<int>& src_part_col_offsets = cuda_row_data_->host_feature_partition_column_index_offsets();
  const std::vector<uint32_t>& src_col_hist_offsets = cuda_row_data_->host_column_hist_offsets();
  const int num_partitions = static_cast<int>(src_part_col_offsets.size()) - 1;
  if (num_partitions <= 0) return false;
  layout->num_partitions = num_partitions;

  // Per-partition: compute used columns and their local idx within source partition.
  layout->part_col_offsets.clear();
  layout->src_part_stride.clear();
  layout->src_local_col.clear();
  layout->partition_for_slot.clear();
  layout->col_hist_offsets.clear();
  layout->part_col_offsets.push_back(0);
  int total_compact = 0;
  for (int p = 0; p < num_partitions; ++p) {
    const int p_start = src_part_col_offsets[p];
    const int p_end = src_part_col_offsets[p + 1];
    const int p_num = p_end - p_start;
    layout->src_part_stride.push_back(p_num);
    for (int c = p_start; c < p_end; ++c) {
      // is_feature_used_bytree is indexed by inner_feature_index. For dense single-feature groups
      // (Numerai), inner_feature_index == column_index; the bounds-check guards us if not.
      if (c < static_cast<int>(is_feature_used_bytree.size()) && is_feature_used_bytree[c]) {
        layout->src_local_col.push_back(c - p_start);
        layout->partition_for_slot.push_back(p);
        layout->col_hist_offsets.push_back(src_col_hist_offsets[c]);
        ++total_compact;
      }
    }
    layout->part_col_offsets.push_back(total_compact);
  }
  layout->total_compact = total_compact;

  if (total_compact == 0 || total_compact == src_part_col_offsets.back()) {
    // No win: either no features or all features used. Fall back to full path.
    return false;
  }

  // 4-bit mode: when the row matrix is packed, the compact matrix is packed the
  // same way (per-partition packed row width = ceil(used_columns / 2) bytes,
  // column j of a partition in byte (j >> 1), nibble (j & 1)).
  layout->is_4bit = cuda_row_data_->is_4bit_packed() && !cuda_row_data_->is_data_host_mapped();

  // Experimental packing codecs (pack_radix5/pack_radix6/pack_bit3 plan keys):
  // a denser-than-nibble layout is eligible when the source is the 4-bit row
  // matrix and EVERY used feature's bin count fits the codec (uniform codec
  // per tree; mixed-codec partition grouping is future work). Bit-identical by
  // construction -- packing is lossless.
  layout->codec = PackCodecId::kNibble4;
  // quant-only in this cut: the non-quantized batched kernels and the JIT read
  // nibble layout; codec eligibility therefore requires the discretized path
  if (layout->is_4bit && use_quantized_grad_) {
    // The stored value bound of a COLUMN is its hist-offset delta (the same
    // quantity the 4-bit row-data gate uses): with EFB, one column can carry
    // a multi-feature bundle whose values exceed any single feature's bin
    // count, so per-FEATURE bin counts are NOT a valid bound here (that
    // mistake corrupted radix words on bundled columns).
    const std::vector<uint32_t>& part_hist_offsets = cuda_row_data_->host_partition_hist_offsets();
    int max_bins_used = 0;
    for (int slot = 0; slot < total_compact; ++slot) {
      const int p = layout->partition_for_slot[slot];
      const int c = src_part_col_offsets[p] + layout->src_local_col[slot];
      const int part_end_col = src_part_col_offsets[p + 1];
      const uint32_t part_span = part_hist_offsets[p + 1] - part_hist_offsets[p];
      const uint32_t col_end = (c + 1 < part_end_col) ? src_col_hist_offsets[c + 1] : part_span;
      const int col_span = static_cast<int>(col_end - src_col_hist_offsets[c]);
      max_bins_used = std::max(max_bins_used, col_span);
    }
    const FalcataPlan& plan = FalcataPlan::Get();
    if (plan.pack_radix5 && max_bins_used <= PackRadix5x32::kMaxBins) {
      layout->codec = PackCodecId::kRadix5x32;
    } else if (plan.pack_radix6 && max_bins_used <= PackRadix6x32::kMaxBins) {
      layout->codec = PackCodecId::kRadix6x32;
    } else if (plan.pack_radix7 && max_bins_used <= PackRadix7x32::kMaxBins) {
      layout->codec = PackCodecId::kRadix7x32;
    } else if (plan.pack_bit3 && max_bins_used <= PackBit3x32::kMaxBins) {
      layout->codec = PackCodecId::kBit3x32;
    }
    if ((plan.pack_radix5 || plan.pack_radix6 || plan.pack_radix7 || plan.pack_bit3) && FalcataDebug().diag) {
      Log::Warning("[pack-codec] tree codec=%d (max column span among %d sampled columns = %d)",
                   static_cast<int>(layout->codec), total_compact, max_bins_used);
    }
  }

  layout->packed_part_offsets.clear();
  layout->packed_part_offsets.push_back(0);
  for (int p = 0; p < num_partitions; ++p) {
    const int used_in_p = layout->part_col_offsets[p + 1] - layout->part_col_offsets[p];
    layout->packed_part_offsets.push_back(
      layout->packed_part_offsets.back() + PackRowBytes(layout->codec, used_in_p));
  }
  // cuda_plan key compact_row_interleave: store each row's nibble bytes of all partitions contiguously. Below the
  // first levels a leaf's rows are sparse in the matrix, and a gathered row then costs the sectors of one
  // row-wide run instead of one partial run per partition.
  layout->row_interleave = layout->is_4bit && layout->codec == PackCodecId::kNibble4 && num_partitions > 1 &&
                           FalcataPlan::Get().compact_row_interleave;
  const data_size_t num_data = cuda_row_data_->num_data();
  layout->data_bytes = layout->is_4bit ?
    static_cast<size_t>(layout->packed_part_offsets.back()) * static_cast<size_t>(num_data) :
    static_cast<size_t>(total_compact) * static_cast<size_t>(num_data);
  return true;
}

bool CUDAHistogramConstructor::BuildCompactView(const std::vector<int8_t>& is_feature_used_bytree) {
  use_compact_view_ = false;
  compact_col_major_filled_ = false;
  fused_root_ready_ = false;
  full_view_used_ = false;
  if (view_mask_regime()) {
    // mask regime: no per-tree copy; every kernel reads the resident full view under this tree's column masks
    prefill_valid_ = false;
    EnsureFullView("the mask regime");
    return false;
  }
  CompactLayout layout;
  const bool eligible = ScanCompactLayout(is_feature_used_bytree, &layout);
  if (prefill_valid_) {
    // A prefill launched during the previous tree may still be in flight on
    // prefetch_stream_ and shares the fill-metadata device arrays with the
    // synchronous path: order everything after it before touching them.
    CUDASUCCESS_OR_FATAL(cudaEventSynchronize(fill_done_event_alt_));
  }
  if (!eligible) {
    prefill_valid_ = false;
    // every construct of this tree reads the whole matrix in the row-major layout: under colmajor_direct, the full
    // view (kept while trees keep needing it)
    if (colmajor_direct_) {
      EnsureFullView("a tree without a compact view");
    }
    return false;
  }

  const std::vector<int>& src_part_col_offsets = cuda_row_data_->host_feature_partition_column_index_offsets();
  const std::vector<int>& compact_part_col_offsets = layout.part_col_offsets;
  const std::vector<int>& src_local_col_for_compact_h = layout.src_local_col;
  const std::vector<int>& partition_for_compact_h = layout.partition_for_slot;
  const std::vector<uint32_t>& compact_col_hist_offsets_h = layout.col_hist_offsets;
  const std::vector<int>& compact_packed_part_offsets = layout.packed_part_offsets;
  const int num_partitions = layout.num_partitions;
  const int total_compact = layout.total_compact;
  const data_size_t num_data = cuda_row_data_->num_data();
  compact_is_4bit_ = layout.is_4bit;
  compact_codec_ = layout.codec;

  // Host copies of the compact layout for the tree learner's column-view build
  // (source column per slot + row-major-in-partition placement of each slot).
  // 8-bit: slot byte includes the column-in-partition offset (col entry 0);
  // 4-bit: slot byte is the partition's packed base and the logical
  // column-in-partition travels separately (the gather kernel derives
  // byte (col >> 1) / nibble (col & 1) from it).
  compact_src_cols_host_.resize(total_compact);
  compact_slot_byte_host_.resize(total_compact);
  compact_slot_stride_host_.resize(total_compact);
  compact_slot_col_host_.resize(total_compact);
  for (int s = 0; s < total_compact; ++s) {
    const int p = partition_for_compact_h[s];
    compact_src_cols_host_[s] = src_part_col_offsets[p] + src_local_col_for_compact_h[s];
    const int compact_part_start = compact_part_col_offsets[p];
    const int compact_col_in_p = s - compact_part_start;
    if (compact_is_4bit_ && layout.row_interleave) {
      compact_slot_byte_host_[s] = static_cast<size_t>(compact_packed_part_offsets[p]);
      compact_slot_stride_host_[s] = compact_packed_part_offsets[num_partitions];
      compact_slot_col_host_[s] = compact_col_in_p;
    } else if (compact_is_4bit_) {
      compact_slot_byte_host_[s] = static_cast<size_t>(compact_packed_part_offsets[p]) * static_cast<size_t>(num_data);
      compact_slot_stride_host_[s] = compact_packed_part_offsets[p + 1] - compact_packed_part_offsets[p];
      compact_slot_col_host_[s] = compact_col_in_p;
    } else {
      const int compact_stride = compact_part_col_offsets[p + 1] - compact_part_start;
      compact_slot_byte_host_[s] = static_cast<size_t>(compact_part_start) * static_cast<size_t>(num_data) +
        static_cast<size_t>(compact_col_in_p);
      compact_slot_stride_host_[s] = compact_stride;
      compact_slot_col_host_[s] = 0;
    }
  }

  // Allocate / resize compact buffers.
  const size_t compact_data_bytes = layout.data_bytes;
  if (compact_data_uint8_t_.Size() < compact_data_bytes) {
    // Discard: both fill paths below (LaunchCompactFill, and the host-mapped
    // staging transpose) rewrite every byte for the CURRENT layout, so Resize's
    // preserve-copy would carry over a previous layout that is about to be
    // overwritten — while holding both blocks. This is the allocation that
    // OOM-ed the rev-3 sweep's judge stage.
    compact_data_uint8_t_.ResizeDiscard(compact_data_bytes);
  }
  if (compact_is_4bit_) {
    if (compact_packed_partition_byte_offsets_.Size() < compact_packed_part_offsets.size()) {
      compact_packed_partition_byte_offsets_.Resize(compact_packed_part_offsets.size());
    }
    // row-interleaved rows: the device copy carries the row width as offsets[0] = -width (PackedPartitionRows)
    std::vector<int> device_packed_offsets = compact_packed_part_offsets;
    if (layout.row_interleave) {
      device_packed_offsets[0] = -compact_packed_part_offsets[num_partitions];
    }
    UploadTreeStartMeta<int>(compact_packed_partition_byte_offsets_.RawData(), device_packed_offsets.data(),
                             device_packed_offsets.size());
  }

  // Upload metadata.
  // We need cuda copies of:
  //   compact_part_col_offsets (P+1 ints)
  //   src_part_col_offsets     (P+1 ints) -- already on device, but we have host
  //   src_part_stride_h        (P ints)
  //   src_local_col_for_compact_h (total_compact ints)
  //   partition_for_compact_h     (total_compact ints)
  //   compact_col_hist_offsets_h  (total_compact uint32)

  if (compact_feature_partition_column_index_offsets_.Size() < compact_part_col_offsets.size()) {
    compact_feature_partition_column_index_offsets_.Resize(compact_part_col_offsets.size());
  }
  UploadTreeStartMeta(compact_feature_partition_column_index_offsets_.RawData(), compact_part_col_offsets.data(),
                      compact_part_col_offsets.size());

  // Nibble view: the same upload also carries the pair-joint construct layout
  // (cuda_plan key pair_hist) behind the T hist offsets: [T, 2T) each slot's
  // value span (its hist-offset delta, the bound of its stored nibbles) and
  // [2T, 3T) the shared-table offset of the joint (lo, hi) histogram of the
  // byte holding the slot, span(lo) * span(hi) cells per byte, laid out from 0
  // in every partition. compact_pair_joint_max_ is the largest partition table
  // (0: layout not built, the per-column kernel runs).
  std::vector<uint32_t> compact_col_meta_h = compact_col_hist_offsets_h;
  compact_pair_joint_max_ = 0;
  compact_pair_joint_total_ = 0;
  compact_row_interleave_ = layout.row_interleave;
  compact_row_bytes_ = layout.packed_part_offsets.empty() ? 0 : layout.packed_part_offsets.back();
  if (compact_is_4bit_ && compact_codec_ == PackCodecId::kNibble4) {
    const std::vector<uint32_t>& src_col_hist_offsets = cuda_row_data_->host_column_hist_offsets();
    const std::vector<uint32_t>& part_hist_offsets = cuda_row_data_->host_partition_hist_offsets();
    // [3T, 4T): the same joint offsets with all partitions' tables back to back (whole-row pair_hist blocks)
    compact_col_meta_h.resize(4 * static_cast<size_t>(total_compact));
    uint32_t joint_max = 0;
    uint32_t joint_base = 0;
    bool spans_ok = true;
    for (int p = 0; p < num_partitions && spans_ok; ++p) {
      const int part_end_col = src_part_col_offsets[p + 1];
      const uint32_t part_span = part_hist_offsets[p + 1] - part_hist_offsets[p];
      uint32_t joint = 0;
      for (int s = compact_part_col_offsets[p]; s < compact_part_col_offsets[p + 1]; ++s) {
        const int c = src_part_col_offsets[p] + src_local_col_for_compact_h[s];
        const uint32_t col_end = (c + 1 < part_end_col) ? src_col_hist_offsets[c + 1] : part_span;
        const uint32_t span = col_end - src_col_hist_offsets[c];
        if (span == 0 || span > static_cast<uint32_t>(PackNibble4::kMaxBins)) {
          spans_ok = false;
          break;
        }
        compact_col_meta_h[total_compact + s] = span;
      }
      if (!spans_ok) break;
      for (int s = compact_part_col_offsets[p]; s < compact_part_col_offsets[p + 1]; s += 2) {
        const bool has_hi = s + 1 < compact_part_col_offsets[p + 1];
        const uint32_t lo_span = compact_col_meta_h[total_compact + s];
        const uint32_t hi_span = has_hi ? compact_col_meta_h[total_compact + s + 1] : 1u;
        compact_col_meta_h[2 * static_cast<size_t>(total_compact) + s] = joint;
        if (has_hi) {
          compact_col_meta_h[2 * static_cast<size_t>(total_compact) + s + 1] = joint;
        }
        // cuda_plan key pair_pad: an odd per-byte stride puts the same cell of the 32 tables a warp's threads
        // own in 32 different shared banks (an even stride such as 6 x 6 = 36 maps every eighth thread to the
        // same bank, so rows concentrated on a few common cells serialize the atomics); pad cells stay zero
        const uint32_t stride = lo_span * hi_span;
        joint += stride + ((FalcataPlan::Get().pair_pad && (stride & 1u) == 0) ? 1u : 0u);
      }
      for (int s = compact_part_col_offsets[p]; s < compact_part_col_offsets[p + 1]; ++s) {
        compact_col_meta_h[3 * static_cast<size_t>(total_compact) + s] =
            joint_base + compact_col_meta_h[2 * static_cast<size_t>(total_compact) + s];
      }
      joint_base += joint;
      joint_max = std::max(joint_max, joint);
    }
    if (spans_ok) {
      compact_pair_joint_max_ = static_cast<int>(joint_max);
      compact_pair_joint_total_ = static_cast<int>(joint_base);
    } else {
      compact_col_meta_h.resize(total_compact);
    }
  }
  if (compact_column_hist_offsets_.Size() < compact_col_meta_h.size()) {
    compact_column_hist_offsets_.Resize(compact_col_meta_h.size());
  }
  UploadTreeStartMeta(compact_column_hist_offsets_.RawData(), compact_col_meta_h.data(), compact_col_meta_h.size());

  // Same partition_hist_offsets as the source (used for global hist write-back position per partition).
  const std::vector<uint32_t>& src_part_hist_offsets = cuda_row_data_->host_partition_hist_offsets();
  if (compact_partition_hist_offsets_.Size() < src_part_hist_offsets.size()) {
    compact_partition_hist_offsets_.Resize(src_part_hist_offsets.size());
  }
  UploadTreeStartMeta(compact_partition_hist_offsets_.RawData(), src_part_hist_offsets.data(),
                      src_part_hist_offsets.size());

  // When the source is host-pinned (zero-copy), the GPU fill kernel does
  // ~1-byte strided PCIe loads which are extremely inefficient (~1 GB/s effective).
  // For host-mapped sources, do a host-side OpenMP gather into pinned compact_data_host_,
  // then a single bulk cudaMemcpy. Empirically this is ~10× faster than the GPU fill kernel.
  //
  // For GPU-resident sources (small datasets that fit in VRAM), the original
  // GPU fill kernel is fastest.
  // Per-tree fill: when source is host (column-major), do one cudaMemcpy per sampled column.
  // Each is a contiguous num_data byte transfer at ~20 GB/s → ~85 ms for f=0.1 / 6.7M rows.
  if (cuda_row_data_->is_data_host_mapped()) {
    const uint8_t* src_col_major = cuda_row_data_->host_partitioned_data_uint8_t();
    // Compact GPU layout: row-major-in-partition (matches histogram kernel expectation).
    // For each compact col c in partition p, we need to write num_data bytes to
    // compact_data_uint8_t_ at strided positions (compact_stride_p apart per row).
    // To get a contiguous bulk transfer, we instead make compact GPU layout COLUMN-MAJOR
    // within partition: [r] = compact_data[part_offset + col_in_p * num_data + r].
    // The histogram kernel access pattern (`data_ptr[r * num_columns_in_partition + threadIdx.x]`)
    // becomes incorrect for column-major; we emit a separate column-major-aware
    // launcher branch.

    // Lazily allocate scratch buffers only used by this host-mapped path.
    CUDAVector<int> d_partition_for_compact(partition_for_compact_h.size());
    CopyFromHostToCUDADevice<int>(d_partition_for_compact.RawData(),
                                  partition_for_compact_h.data(),
                                  partition_for_compact_h.size(), __FILE__, __LINE__);
    // Multi-stream direct cudaMemcpyAsync per column.
    if (compact_staging_col_major_.Size() < compact_data_bytes) {
      // Discard: the per-column memcpys below write every sampled column, and
      // compact_data_bytes is sized for exactly those columns.
      compact_staging_col_major_.ResizeDiscard(compact_data_bytes);
    }
    static const int N_STREAMS = 4;
    static cudaStream_t copy_streams[N_STREAMS] = {nullptr};
    static cudaEvent_t copy_done[N_STREAMS] = {nullptr};
    static bool streams_init = false;
    if (!streams_init) {
      for (int i = 0; i < N_STREAMS; ++i) {
        cudaStreamCreate(&copy_streams[i]);
        cudaEventCreate(&copy_done[i]);
      }
      streams_init = true;
    }
    for (int compact_col = 0; compact_col < total_compact; ++compact_col) {
      const int p = partition_for_compact_h[compact_col];
      const int src_local = src_local_col_for_compact_h[compact_col];
      const int src_p_start = src_part_col_offsets[p];
      const size_t src_byte_offset = (static_cast<size_t>(src_p_start) + static_cast<size_t>(src_local)) * static_cast<size_t>(num_data);
      const size_t staging_byte_offset = static_cast<size_t>(compact_col) * static_cast<size_t>(num_data);
      cudaMemcpyAsync(compact_staging_col_major_.RawData() + staging_byte_offset,
                      src_col_major + src_byte_offset,
                      num_data,
                      cudaMemcpyHostToDevice,
                      copy_streams[compact_col % N_STREAMS]);
    }
    for (int i = 0; i < N_STREAMS; ++i) {
      cudaEventRecord(copy_done[i], copy_streams[i]);
      cudaStreamWaitEvent(cuda_stream_, copy_done[i], 0);
    }
    // GPU transpose staging (col-major) → compact_data (row-major-in-partition).
    LaunchTransposeColMajorToRowMajor(
        cuda_stream_,
        compact_staging_col_major_.RawData(),
        compact_data_uint8_t_.RawData(),
        d_partition_for_compact.RawData(),
        compact_feature_partition_column_index_offsets_.RawData(),
        num_data,
        total_compact);
    CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_stream_));
    // compact_data is now row-major-in-partition
    compact_col_major_filled_ = true;  // the col-major staging holds the same slots
  } else if (prefill_valid_ && prefill_bitmap_ == is_feature_used_bytree) {
    // The prefill already produced exactly these bytes into the alt buffer
    // during the previous tree's training (event-synced above): swap it in
    // and skip the fill entirely.
    prefill_valid_ = false;
    compact_data_uint8_t_.Swap(&compact_data_uint8_t_alt_);
  } else {
    prefill_valid_ = false;
    LaunchCompactFill(layout, compact_data_uint8_t_.RawData(), cuda_stream_, /*async_meta=*/false);
    // cuda_plan key async_tree_start: the host reads nothing the fill writes, so it goes on preparing the root
    // level meanwhile. Every consumer is a later GPU operation: on this stream (construct, root apply), on the
    // legacy default stream (uploads, readbacks, the column gather), which waits for this blocking stream, or on
    // another blocking stream behind such a legacy operation (split kernels follow the best-split readback)
    if (!AsyncTreeStart()) {
      CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_stream_));
    }
  }


  // Compute max compact cols per partition (sets block_dim_x for compact launches).
  int max_compact_per_p = 0;
  for (int p = 0; p < num_partitions; ++p) {
    const int cnt = compact_part_col_offsets[p + 1] - compact_part_col_offsets[p];
    if (cnt > max_compact_per_p) max_compact_per_p = cnt;
  }
  max_num_compact_cols_per_partition_ = max_compact_per_p;

  num_compact_columns_ = total_compact;
  use_compact_view_ = true;
  return true;
}

void CUDAHistogramConstructor::LaunchCompactFill(
    const CompactLayout& layout, uint8_t* dst, cudaStream_t stream, bool async_meta) {
  const std::vector<int>& src_part_col_offsets = cuda_row_data_->host_feature_partition_column_index_offsets();
  const std::vector<int>& compact_part_col_offsets = layout.part_col_offsets;
  const std::vector<int>& src_local_col_for_compact_h = layout.src_local_col;
  const std::vector<int>& partition_for_compact_h = layout.partition_for_slot;
  const std::vector<int>& compact_packed_part_offsets = layout.packed_part_offsets;
  const int num_partitions = layout.num_partitions;
  const int total_compact = layout.total_compact;
  const data_size_t num_data = cuda_row_data_->num_data();

  // Async metadata upload path (prefill): stage through pinned memory and
  // cudaMemcpyAsync on `stream`. A plain cudaMemcpy here would enqueue on the
  // synchronizing default stream and stall behind the current tree's kernels.
  auto upload = [&](void* device_dst, const void* host_src, size_t bytes, size_t* pin_off) {
    if (!async_meta) {
      if (AsyncTreeStart()) {
        // cuda_plan key async_tree_start: the same default stream as cudaMemcpy, without the host wait (the
        // pageable source is staged before the call returns, so host_src may go out of scope)
        CopyFromHostToCUDADeviceAsync<uint8_t>(static_cast<uint8_t*>(device_dst),
                                               static_cast<const uint8_t*>(host_src), bytes, 0,
                                               __FILE__, __LINE__);
      } else {
        CopyFromHostToCUDADevice<uint8_t>(static_cast<uint8_t*>(device_dst),
                                          static_cast<const uint8_t*>(host_src), bytes,
                                          __FILE__, __LINE__);
      }
      return;
    }
    if (*pin_off + bytes > prefill_pinned_bytes_) {
      // grow pinned staging (rare; sizes are KB-scale)
      const size_t need = ((*pin_off + bytes) * 2 + 4095) & ~static_cast<size_t>(4095);
      void* fresh = nullptr;
      CUDASUCCESS_OR_FATAL(cudaHostAlloc(&fresh, need, cudaHostAllocDefault));
      if (prefill_pinned_ != nullptr) {
        CUDASUCCESS_OR_FATAL(cudaFreeHost(prefill_pinned_));
      }
      prefill_pinned_ = fresh;
      prefill_pinned_bytes_ = need;
    }
    uint8_t* pin = static_cast<uint8_t*>(prefill_pinned_) + *pin_off;
    std::memcpy(pin, host_src, bytes);
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(device_dst, pin, bytes, cudaMemcpyHostToDevice, stream));
    *pin_off += bytes;
  };
  size_t pin_off = 0;

  if (layout.codec != PackCodecId::kNibble4) {
    // Codec fill (pack_bit3/pack_radix5/pack_radix6): transcode the sampled
    // columns from the 4-bit source into codec-packed uint32 words. One
    // thread per (destination word slot, row group); per-column source
    // nibble metadata plus per-word-slot placement metadata.
    const std::vector<int>& src_packed_offsets = cuda_row_data_->host_packed_partition_byte_offsets();
    const int D = PackValuesPerWord(layout.codec);
    std::vector<size_t> col_nib_h(total_compact);
    std::vector<int> col_stride_h(total_compact);
    for (int s = 0; s < total_compact; ++s) {
      const int p = partition_for_compact_h[s];
      if (colmajor_direct_) {
        // the source is the column-major store: column c is the nibble run at c * pad, per-row stride 1
        col_nib_h[s] = (static_cast<size_t>(src_part_col_offsets[p]) +
                        static_cast<size_t>(src_local_col_for_compact_h[s])) * colmajor_pad_;
        col_stride_h[s] = 1;
        continue;
      }
      col_nib_h[s] = static_cast<size_t>(src_packed_offsets[p]) * static_cast<size_t>(num_data) * 2 +
        static_cast<size_t>(src_local_col_for_compact_h[s]);
      col_stride_h[s] = (src_packed_offsets[p + 1] - src_packed_offsets[p]) * 2;
    }
    int total_word_slots = 0;
    for (int p = 0; p < num_partitions; ++p) {
      const int used_in_p = compact_part_col_offsets[p + 1] - compact_part_col_offsets[p];
      total_word_slots += (used_in_p + D - 1) / D;
    }
    std::vector<size_t> ws_dst_byte_h(total_word_slots);
    std::vector<int> ws_dst_stride_h(total_word_slots);
    std::vector<int> ws_first_col_h(total_word_slots);
    std::vector<uint8_t> ws_ndig_h(total_word_slots);
    int ws = 0;
    for (int p = 0; p < num_partitions; ++p) {
      const int part_start = compact_part_col_offsets[p];
      const int used_in_p = compact_part_col_offsets[p + 1] - part_start;
      const int width_p = compact_packed_part_offsets[p + 1] - compact_packed_part_offsets[p];
      const size_t dst_part_byte = static_cast<size_t>(compact_packed_part_offsets[p]) * static_cast<size_t>(num_data);
      const int num_words = (used_in_p + D - 1) / D;
      for (int w = 0; w < num_words; ++w) {
        ws_dst_byte_h[ws] = dst_part_byte + static_cast<size_t>(4 * w);
        ws_dst_stride_h[ws] = width_p;
        ws_first_col_h[ws] = part_start + w * D;
        ws_ndig_h[ws] = static_cast<uint8_t>(std::min(D, used_in_p - w * D));
        ++ws;
      }
    }
    CHECK_EQ(ws, total_word_slots);
    if (cuda_col_src_nib_base_.Size() < static_cast<size_t>(total_compact)) {
      cuda_col_src_nib_base_.Resize(total_compact);
      cuda_col_src_stride_nib_.Resize(total_compact);
    }
    if (cuda_ws_dst_byte_.Size() < static_cast<size_t>(total_word_slots)) {
      cuda_ws_dst_byte_.Resize(total_word_slots);
      cuda_ws_dst_stride_.Resize(total_word_slots);
      cuda_ws_first_col_.Resize(total_word_slots);
      cuda_ws_ndig_.Resize(total_word_slots);
    }
    upload(cuda_col_src_nib_base_.RawData(), col_nib_h.data(), sizeof(size_t) * total_compact, &pin_off);
    upload(cuda_col_src_stride_nib_.RawData(), col_stride_h.data(), sizeof(int) * total_compact, &pin_off);
    upload(cuda_ws_dst_byte_.RawData(), ws_dst_byte_h.data(), sizeof(size_t) * total_word_slots, &pin_off);
    upload(cuda_ws_dst_stride_.RawData(), ws_dst_stride_h.data(), sizeof(int) * total_word_slots, &pin_off);
    upload(cuda_ws_first_col_.RawData(), ws_first_col_h.data(), sizeof(int) * total_word_slots, &pin_off);
    upload(cuda_ws_ndig_.RawData(), ws_ndig_h.data(), sizeof(uint8_t) * total_word_slots, &pin_off);
    LaunchFillCompactCodecKernel(
      stream, layout.codec,
      colmajor_direct_ ? colmajor_bin_.RawDataReadOnly() : RowMajorBin<uint8_t>(),
      dst,
      cuda_col_src_nib_base_.RawData(),
      cuda_col_src_stride_nib_.RawData(),
      cuda_ws_dst_byte_.RawData(),
      cuda_ws_dst_stride_.RawData(),
      cuda_ws_first_col_.RawData(),
      cuda_ws_ndig_.RawData(),
      total_word_slots,
      num_data);
    return;
  }

  if (layout.is_4bit) {
    // 4-bit fill: one thread per DESTINATION byte (a pair of adjacent compact
    // slots), so no two threads share an output byte. Source positions are
    // nibble indices: column c of a packed partition with byte base B and row
    // width W sits at nibble 2*B + c + row * (2*W).
    const std::vector<int>& src_packed_offsets = cuda_row_data_->host_packed_partition_byte_offsets();
    int total_byte_slots = 0;
    for (int p = 0; p < num_partitions; ++p) {
      const int used_in_p = compact_part_col_offsets[p + 1] - compact_part_col_offsets[p];
      total_byte_slots += (used_in_p + 1) >> 1;
    }
    std::vector<size_t> bs_src_nib0_h(total_byte_slots);
    std::vector<size_t> bs_src_nib1_h(total_byte_slots);
    std::vector<int> bs_src_stride_nib_h(total_byte_slots);
    std::vector<size_t> bs_dst_byte_h(total_byte_slots);
    std::vector<int> bs_dst_stride_h(total_byte_slots);
    int byte_slot = 0;
    for (int p = 0; p < num_partitions; ++p) {
      const int compact_part_start = compact_part_col_offsets[p];
      const int used_in_p = compact_part_col_offsets[p + 1] - compact_part_start;
      // row-interleaved: partition p's bytes sit at offset compact_packed_part_offsets[p] of a full-width row
      const int dst_packed_width = layout.row_interleave ? compact_packed_part_offsets[num_partitions] :
        compact_packed_part_offsets[p + 1] - compact_packed_part_offsets[p];
      const size_t dst_part_byte = layout.row_interleave ? static_cast<size_t>(compact_packed_part_offsets[p]) :
        static_cast<size_t>(compact_packed_part_offsets[p]) * static_cast<size_t>(num_data);
      const size_t src_part_nib = static_cast<size_t>(src_packed_offsets[p]) * static_cast<size_t>(num_data) * 2;
      const int src_stride_nib = (src_packed_offsets[p + 1] - src_packed_offsets[p]) * 2;
      for (int m = 0; m < ((used_in_p + 1) >> 1); ++m) {
        const int s0 = compact_part_start + 2 * m;
        if (colmajor_pad_ > 0) {
          // column-major source: column c is a contiguous nibble run at
          // c * colmajor_pad_, so the per-row stride is 1 -- the SAME fill
          // kernel then reads coalesced instead of gathering across rows
          const size_t c0 = static_cast<size_t>(src_part_col_offsets[p]) +
            static_cast<size_t>(src_local_col_for_compact_h[s0]);
          bs_src_nib0_h[byte_slot] = c0 * colmajor_pad_;
          bs_src_nib1_h[byte_slot] = (2 * m + 1) < used_in_p ?
            (static_cast<size_t>(src_part_col_offsets[p]) +
             static_cast<size_t>(src_local_col_for_compact_h[s0 + 1])) * colmajor_pad_ :
            ~static_cast<size_t>(0);
          bs_src_stride_nib_h[byte_slot] = 1;
        } else {
        bs_src_nib0_h[byte_slot] = src_part_nib + static_cast<size_t>(src_local_col_for_compact_h[s0]);
        bs_src_nib1_h[byte_slot] = (2 * m + 1) < used_in_p ?
          src_part_nib + static_cast<size_t>(src_local_col_for_compact_h[s0 + 1]) :
          ~static_cast<size_t>(0);
        bs_src_stride_nib_h[byte_slot] = src_stride_nib;
        }
        bs_dst_byte_h[byte_slot] = dst_part_byte + static_cast<size_t>(m);
        bs_dst_stride_h[byte_slot] = dst_packed_width;
        ++byte_slot;
      }
    }
    CHECK_EQ(byte_slot, total_byte_slots);
    if (cuda_bs_src_nib0_.Size() < static_cast<size_t>(total_byte_slots)) {
      cuda_bs_src_nib0_.Resize(total_byte_slots);
      cuda_bs_src_nib1_.Resize(total_byte_slots);
      cuda_bs_src_stride_nib_.Resize(total_byte_slots);
      cuda_bs_dst_byte_.Resize(total_byte_slots);
      cuda_bs_dst_stride_.Resize(total_byte_slots);
    }
    upload(cuda_bs_src_nib0_.RawData(), bs_src_nib0_h.data(), sizeof(size_t) * total_byte_slots, &pin_off);
    upload(cuda_bs_src_nib1_.RawData(), bs_src_nib1_h.data(), sizeof(size_t) * total_byte_slots, &pin_off);
    upload(cuda_bs_src_stride_nib_.RawData(), bs_src_stride_nib_h.data(), sizeof(int) * total_byte_slots, &pin_off);
    upload(cuda_bs_dst_byte_.RawData(), bs_dst_byte_h.data(), sizeof(size_t) * total_byte_slots, &pin_off);
    upload(cuda_bs_dst_stride_.RawData(), bs_dst_stride_h.data(), sizeof(int) * total_byte_slots, &pin_off);
    const bool tiled = CompactFill4BitTiledEligible(bs_src_nib0_h, bs_src_nib1_h, bs_src_stride_nib_h, bs_dst_byte_h,
                                                    bs_dst_stride_h);
    if (FalcataDebug().diag) {
      Log::Info("compact fill: %s 4-bit kernel, %d byte slots in %d partitions", tiled ? "tiled" : "per-cell",
                total_byte_slots, num_partitions);
    }
    // fused root histogram (cuda_plan key fused_root_hist): requested by the learner for this tree (root = every
    // row), synchronous fill of the live view, quantized gradients in place
    bool fused_root = false;
    if (tiled && fused_root_requested_ && !async_meta && use_quantized_grad_ && cuda_gradients_ != nullptr &&
        dst == compact_data_uint8_t_.RawData() && FalcataPlan::Get().fused_root_hist) {
      // local histogram: each compact column's bins in slot order; a column spans its hist-offset delta (as the
      // codec scan above), which a nibble always fits
      const std::vector<uint32_t>& src_col_hist_offsets = cuda_row_data_->host_column_hist_offsets();
      const std::vector<uint32_t>& part_hist_offsets = cuda_row_data_->host_partition_hist_offsets();
      std::vector<int> col_first_bin(total_compact);
      std::vector<int> meta(2 * static_cast<size_t>(total_byte_slots));
      std::vector<int> bin_pos;
      int num_local_bins = 0;
      int max_span = 0;
      bool spans_ok = true;
      for (int s = 0; s < total_compact && spans_ok; ++s) {
        const int p = partition_for_compact_h[s];
        const int c = src_part_col_offsets[p] + src_local_col_for_compact_h[s];
        const uint32_t part_span = part_hist_offsets[p + 1] - part_hist_offsets[p];
        const uint32_t col_end = (c + 1 < src_part_col_offsets[p + 1]) ? src_col_hist_offsets[c + 1] : part_span;
        const int span = static_cast<int>(col_end) - static_cast<int>(src_col_hist_offsets[c]);
        if (span <= 0 || span > 16) {
          spans_ok = false;
          break;
        }
        max_span = std::max(max_span, span);
        col_first_bin[s] = static_cast<int>(bin_pos.size());
        for (int b = 0; b < span; ++b) {
          bin_pos.push_back(static_cast<int>(part_hist_offsets[p] + src_col_hist_offsets[c] + b));
        }
      }
      if (spans_ok) {
        int slot = 0;
        for (int p = 0; p < num_partitions; ++p) {
          const int compact_part_start = compact_part_col_offsets[p];
          const int used_in_p = compact_part_col_offsets[p + 1] - compact_part_start;
          for (int m = 0; m < ((used_in_p + 1) >> 1); ++m) {
            meta[2 * slot] = col_first_bin[compact_part_start + 2 * m];
            meta[2 * slot + 1] = (2 * m + 1) < used_in_p ? col_first_bin[compact_part_start + 2 * m + 1] : -1;
            ++slot;
          }
        }
        CHECK_EQ(slot, total_byte_slots);
        num_local_bins = static_cast<int>(bin_pos.size());
        // the kernel packs two first-bin offsets of a slot into 16 bits each
        spans_ok = num_local_bins < 0xffff;
      }
      if (spans_ok) {
        meta.insert(meta.end(), bin_pos.begin(), bin_pos.end());
        if (fused_root_meta_.Size() < meta.size()) {
          fused_root_meta_.Resize(meta.size());
        }
        upload(fused_root_meta_.RawData(), meta.data(), sizeof(int) * meta.size(), &pin_off);
        // aligned 32-bit source words: the column-major store (column c at nibble c * colmajor_pad_) with every
        // column a multiple of 4 bytes long, so no word straddles a column end or the store's end
        bool word_prefetch = FalcataPlan::Get().root_hist_prefetch && colmajor_pad_ > 0 && colmajor_pad_ % 8 == 0 &&
                             reinterpret_cast<uintptr_t>(colmajor_bin_.RawDataReadOnly()) % 4 == 0;
        for (int i = 0; word_prefetch && i < total_byte_slots; ++i) {
          word_prefetch = bs_src_nib0_h[i] % colmajor_pad_ == 0 &&
                          (bs_src_nib1_h[i] == ~static_cast<size_t>(0) || bs_src_nib1_h[i] % colmajor_pad_ == 0);
        }
        // cuda_plan key colmajor_dense3: the fill reads the 3-bit copy of the store when every slot's columns
        // start on a column of the store and hold only values below 8
        const uint32_t* dense3 = nullptr;
        if (FalcataPlan::Get().colmajor_dense3 && colmajor_dense3_.Size() > 0 && colmajor_pad_ > 0) {
          bool dense_ok = true;
          auto narrow_column = [&](size_t nib) {
            if (nib % colmajor_pad_ != 0) return false;
            const size_t c = nib / colmajor_pad_;
            return c < colmajor_dense3_wide_.size() && colmajor_dense3_wide_[c] == 0;
          };
          for (int i = 0; dense_ok && i < total_byte_slots; ++i) {
            dense_ok = narrow_column(bs_src_nib0_h[i]) &&
                       (bs_src_nib1_h[i] == ~static_cast<size_t>(0) || narrow_column(bs_src_nib1_h[i]));
          }
          if (dense_ok) {
            dense3 = colmajor_dense3_.RawDataReadOnly();
          }
        }
        if (fused_root_scratch_.Size() < static_cast<size_t>(num_total_bin_)) {
          fused_root_scratch_.Resize(static_cast<size_t>(num_total_bin_));
        }
        fused_root = LaunchFillCompactData4BitTiledRootHistKernel(
          stream,
          // the same source as the plain tiled fill below: the column-major store (colmajor_direct's compact
          // regime, or colmajor_fill's copy), else the row-major matrix through the one accessor
          colmajor_pad_ > 0 ? colmajor_bin_.RawDataReadOnly() : RowMajorBin<uint8_t>(),
          dst,
          cuda_bs_src_nib0_.RawData(),
          cuda_bs_src_nib1_.RawData(),
          cuda_bs_dst_byte_.RawData(),
          cuda_bs_dst_stride_.RawData(),
          total_byte_slots,
          num_data,
          fused_root_meta_.RawData(),
          reinterpret_cast<const uint32_t*>(fused_root_meta_.RawData() + 2 * static_cast<size_t>(total_byte_slots)),
          num_local_bins,
          // slot-major block histogram rows: every column's nibbles lie below the widest span
          FalcataPlan::Get().root_hist_slot_major ? max_span : 0,
          // each block's tiles a grid apart, so the resident blocks stage neighbouring tiles of every column
          FalcataPlan::Get().root_hist_tile_stride,
          // one partition: stage the tile as a byte copy of its output run, written with aligned vector copies
          FalcataPlan::Get().root_hist_run_copy,
          word_prefetch,
          // the packed per-block row bound of the gradients as discretized (the effective quant bins, which the
          // learner may raise above the configured count), exactly the construct's 65534 / bins rule
          65534 / std::max(1, num_grad_quant_bins_),
          reinterpret_cast<const int32_t*>(cuda_gradients_),
          fused_root_request_16bit_,
          fused_root_scratch_.RawData(),
          static_cast<size_t>(num_total_bin_) * sizeof(hist_t),
          dense3, colmajor_pad_, colmajor_dense3_pitch_, colmajor_dense3_codec_);
        fused_root_ready_ = fused_root;
        fused_root_bits16_ = fused_root_request_16bit_;
      }
    }
    if (!fused_root) {
    LaunchFillCompactData4BitKernel(
      stream,
      colmajor_pad_ > 0 ? colmajor_bin_.RawDataReadOnly() : RowMajorBin<uint8_t>(),
      dst,
      cuda_bs_src_nib0_.RawData(),
      cuda_bs_src_nib1_.RawData(),
      cuda_bs_src_stride_nib_.RawData(),
      cuda_bs_dst_byte_.RawData(),
      cuda_bs_dst_stride_.RawData(),
      total_byte_slots,
      num_data,
      tiled);
    }
  } else {
    // Build per-slot src/dst metadata host-side. Each compact slot has a fully
    // computed source byte offset and destination byte offset, so the kernel
    // does no per-thread partition lookups (massive win on this gather pattern).
    const std::vector<int>& src_part_stride_h = layout.src_part_stride;
    std::vector<size_t> slot_src_byte_h(total_compact);
    std::vector<int> slot_src_stride_h(total_compact);
    std::vector<size_t> slot_dst_byte_h(total_compact);
    std::vector<int> slot_dst_stride_h(total_compact);
    for (int s = 0; s < total_compact; ++s) {
      const int p = partition_for_compact_h[s];
      const int p_start = src_part_col_offsets[p];
      const size_t src_p_byte = static_cast<size_t>(p_start) * static_cast<size_t>(num_data);
      const int src_local = src_local_col_for_compact_h[s];
      slot_src_byte_h[s] = src_p_byte + static_cast<size_t>(src_local);
      slot_src_stride_h[s] = src_part_stride_h[p];
      const int compact_part_start = compact_part_col_offsets[p];
      const int compact_stride = compact_part_col_offsets[p + 1] - compact_part_start;
      const int compact_col_in_p = s - compact_part_start;
      const size_t compact_p_byte = static_cast<size_t>(compact_part_start) * static_cast<size_t>(num_data);
      slot_dst_byte_h[s] = compact_p_byte + static_cast<size_t>(compact_col_in_p);
      slot_dst_stride_h[s] = compact_stride;
    }
    if (cuda_slot_src_byte_.Size() < static_cast<size_t>(total_compact)) {
      cuda_slot_src_byte_.Resize(total_compact);
      cuda_slot_src_stride_.Resize(total_compact);
      cuda_slot_dst_byte_.Resize(total_compact);
      cuda_slot_dst_stride_.Resize(total_compact);
    }
    upload(cuda_slot_src_byte_.RawData(), slot_src_byte_h.data(), sizeof(size_t) * total_compact, &pin_off);
    upload(cuda_slot_src_stride_.RawData(), slot_src_stride_h.data(), sizeof(int) * total_compact, &pin_off);
    upload(cuda_slot_dst_byte_.RawData(), slot_dst_byte_h.data(), sizeof(size_t) * total_compact, &pin_off);
    upload(cuda_slot_dst_stride_.RawData(), slot_dst_stride_h.data(), sizeof(int) * total_compact, &pin_off);
    // NOTE: a fused column-major second output here was measured SLOWER than
    // the tree learner's separate tile-transposed gather from the compact
    // matrix (the fill's slot-major warps write the column-major layout one
    // 32-byte sector per byte); keep the fill single-output.
    LaunchFillCompactDataKernel(
      stream,
      RowMajorBin<uint8_t>(),
      dst,
      cuda_slot_src_byte_.RawData(),
      cuda_slot_src_stride_.RawData(),
      cuda_slot_dst_byte_.RawData(),
      cuda_slot_dst_stride_.RawData(),
      total_compact,
      num_data,
      nullptr);
  }
}

void CUDAHistogramConstructor::PrefillNextCompactView(
    const std::vector<int8_t>& is_feature_used_bytree) {
  // Any previous prefill was consumed or invalidated by BuildCompactView.
  prefill_valid_ = false;
  if (!FalcataPlan::Get().compact_prefill) return;
  if (cuda_row_data_ == nullptr || cuda_row_data_->is_data_host_mapped()) return;
  CompactLayout layout;
  if (!ScanCompactLayout(is_feature_used_bytree, &layout)) return;
  if (compact_data_uint8_t_alt_.Size() < layout.data_bytes) {
    // Discard: LaunchCompactFill on the next line rewrites all of it.
    compact_data_uint8_t_alt_.ResizeDiscard(layout.data_bytes);
  }
  LaunchCompactFill(layout, compact_data_uint8_t_alt_.RawData(), prefetch_stream_,
                    /*async_meta=*/true);
  CUDASUCCESS_OR_FATAL(cudaEventRecord(fill_done_event_alt_, prefetch_stream_));
  prefill_bitmap_ = is_feature_used_bytree;
  prefill_valid_ = true;
}

void CUDAHistogramConstructor::InvalidateCompactPrefill() {
  if (prefill_valid_) {
    CUDASUCCESS_OR_FATAL(cudaEventSynchronize(fill_done_event_alt_));
    prefill_valid_ = false;
  }
}

void CUDAHistogramConstructor::Init(const Dataset* train_data, TrainingShareStates* share_state) {
  cuda_hist_.Resize(static_cast<size_t>(num_total_bin_ * 2 * num_leaves_) *
                    static_cast<size_t>(num_hist_planes_));
  cuda_hist_.SetValue(0);
  // Deterministic float-mode construct scratch (see ResetTrainingData; both
  // entry points size it because either can be the only one a flow calls).
  det_tile_alloc_ = std::min(kDetTileCap, (train_data->num_data() + kDetRowsPerThread - 1) / kDetRowsPerThread);
  cuda_det_tile_partials_.Resize(static_cast<size_t>(kNumHistPipelines) * det_tile_alloc_ * 2 * num_total_bin_);

  cuda_feature_num_bins_.InitFromHostVector(feature_num_bins_);
  cuda_feature_hist_offsets_.InitFromHostVector(feature_hist_offsets_);
  cuda_feature_most_freq_bins_.InitFromHostVector(feature_most_freq_bins_);

  cuda_row_data_.reset(new CUDARowData(train_data, share_state, gpu_device_id_, gpu_use_dp_));
  // colmajor_direct: no row-major matrix when the Dataset's columns can serve (the tree learner then builds a view)
  cuda_row_data_->RequestDense4BitColumnsOnly(FalcataPlan::Get().colmajor_fill && FalcataPlan::Get().colmajor_direct);
  cuda_row_data_->Init(train_data, share_state);

  // Deterministic dense-construct scratch (same as ResetTrainingData; both
  // entry points size it because either can be the only one a flow calls).
  // Slot rows span the FULL histogram (positions are global per partition,
  // so partitions own disjoint ranges of each row).
  const std::vector<uint32_t>& dense_part_offsets = cuda_row_data_->host_partition_hist_offsets();
  const uint32_t dense_stride = dense_part_offsets.empty() ? 0u : (dense_part_offsets.back() << 1);
  det_dense_slot_stride_ = dense_stride;
  if (!use_quantized_grad_ && dense_stride > 0) {
    const size_t per_row = static_cast<size_t>(dense_stride) * sizeof(hist_t);
    det_dense_dy_ = std::max(1, std::min<int>(kDetDenseDyCap,
        static_cast<int>(kDetDenseSlotBudget / (static_cast<size_t>(kDetTileCap) * per_row))));
    // The dy formula floors at 1; ultra-wide strides then need the TILE
    // count to yield instead, or the slab exceeds the budget kDetTileCap-fold.
    det_dense_tile_cap_ = std::max(1, std::min<int>(kDetTileCap,
        static_cast<int>(kDetDenseSlotBudget / (static_cast<size_t>(det_dense_dy_) * per_row))));
    cuda_det_dense_slots_.Resize(static_cast<size_t>(kNumHistPipelines) * det_dense_tile_cap_ * det_dense_dy_ * dense_stride);
  } else {
    det_dense_dy_ = 0;
    det_dense_tile_cap_ = 0;
  }

  // fp32-pair global histograms: dense shared-memory non-quantized layout only
  // (sparse / large-bin construct kernels and the categorical find path stay
  // hist_t and are excluded)
  hist_fp32_ = FalcataFP32HistRequested() && !use_quantized_grad_ && !gpu_use_dp_ &&
    !cuda_row_data_->is_sparse() && cuda_row_data_->NumLargeBinPartition() == 0 &&
    !has_categorical_feature_;
  if (FalcataFP32HistRequested() && !use_quantized_grad_) {
    Log::Debug("CUDAHistogramConstructor: fp32 histogram mode %s", hist_fp32_ ? "engaged" : "unsupported for this dataset, using fp64");
  }

  if (FalcataPlan::Get().l2_policy) {
    // Reserve a persisting-L2 carve-out once; the per-tree window is set in
    // BeforeTrain when the gradient buffer for the tree is known. Failure is
    // non-fatal (arch without persisting L2): the key becomes a no-op.
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, gpu_device_id_ < 0 ? 0 : gpu_device_id_) == cudaSuccess &&
        prop.persistingL2CacheMaxSize > 0) {
      // Size the carve-out to the device, not to any one GPU model: at most
      // 2/3 of total L2 (the streaming bin-matrix reads need the rest -- the
      // 2/3 ratio is the 64MB/96MB split validated on the 5090), at most the
      // arch's persisting max, and no larger than the buffer we actually pin
      // (a fixed carve would strand L2 on small datasets).
      const size_t needed = static_cast<size_t>(num_data_) *
        (use_quantized_grad_ ? sizeof(int32_t) : 2 * sizeof(score_t));
      const size_t want = std::min({needed,
        static_cast<size_t>(prop.persistingL2CacheMaxSize),
        static_cast<size_t>(prop.l2CacheSize) * 2 / 3});
      if (want > 0 &&
          cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want) == cudaSuccess) {
        l2_carveout_bytes_ = want;
        l2_max_window_bytes_ = static_cast<size_t>(prop.accessPolicyMaxWindowSize);
      }
    }
  }
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));
  // Non-blocking side stream for the next-tree compact-view prefill: must not
  // synchronize with the legacy default stream while training kernels run.
  CUDASUCCESS_OR_FATAL(cudaStreamCreateWithFlags(&prefetch_stream_, cudaStreamNonBlocking));
  CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&fill_done_event_alt_, cudaEventDisableTiming));
  // Lightweight (timing-disabled) events used to order the best split finder's
  // per-leaf kernels after histogram construction/subtraction without a device sync.
  // One (stream, event-pair) pipeline per concurrently-processed sibling pair.
  pipeline_streams_[0] = cuda_stream_;
  // FALCATA_SINGLE_STREAM (diagnostic): alias every pipeline to the main
  // stream, removing all cross-stream event ordering from the flow.
  const bool single_stream = std::getenv("FALCATA_SINGLE_STREAM") != nullptr;
  for (int p = 1; p < kNumHistPipelines; ++p) {
    if (single_stream) {
      pipeline_streams_[p] = cuda_stream_;
    } else {
      CUDASUCCESS_OR_FATAL(cudaStreamCreate(&pipeline_streams_[p]));
    }
  }
  for (int p = 0; p < kNumHistPipelines; ++p) {
    CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&construct_done_events_[p], cudaEventDisableTiming));
    CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(&subtract_done_events_[p], cudaEventDisableTiming));
  }

  cuda_need_fix_histogram_features_.InitFromHostVector(need_fix_histogram_features_);
  cuda_need_fix_histogram_features_num_bin_aligned_.InitFromHostVector(need_fix_histogram_features_num_bin_aligend_);
  cuda_hybrid_construct_dim_y_.Resize(1);
  InitFixMFBMask();

  if (cuda_row_data_->NumLargeBinPartition() > 0) {
    int grid_dim_x = 0, grid_dim_y = 0, block_dim_x = 0, block_dim_y = 0;
    CalcConstructHistogramKernelDim(&grid_dim_x, &grid_dim_y, &block_dim_x, &block_dim_y, num_data_);
    const size_t buffer_size = static_cast<size_t>(grid_dim_y) * static_cast<size_t>(num_total_bin_);
    if (!use_quantized_grad_) {
      if (gpu_use_dp_) {
        // need to double the size of histogram buffer in global memory when using double precision in histogram construction
        cuda_hist_buffer_.Resize(buffer_size * 4);
      } else {
        cuda_hist_buffer_.Resize(buffer_size * 2);
      }
    } else {
      // use only half the size of histogram buffer in global memory when quantized training since each gradient and hessian takes only 2 bytes
      cuda_hist_buffer_.Resize(buffer_size);
    }
  }
  // The column-major fill source (cuda_plan key colmajor_fill) is not made here. Under colmajor_direct the tree
  // learner chooses the regime and its view is built from the Dataset's columns (ChooseViewRegime) at the end of its
  // Init; otherwise the tree learner calls InitColMajorFill before its second tree.
  colmajor_bin_.Clear();
  ClearColMajorDense3();
  colmajor_pad_ = 0;
  full_view_.Clear();
  colmajor_direct_ = cuda_row_data_->dense_4bit_columns_only();

  // one int32 region of num_total_bin_ entries per histogram pipeline (the pairs
  // of a level run concurrently on different pipeline streams and must not share
  // the 64->32-bit compaction scratch space); sized in hist_t (8 byte) units
  hist_buffer_for_num_bit_change_.Resize(
    std::max<size_t>(static_cast<size_t>(num_total_bin_) * 2,
                     (static_cast<size_t>(num_total_bin_) * kNumHistPipelines + 1) / 2));
}

// colmajor_direct. The column-major store colmajor_fill gathers from holds, byte for byte, the Dataset's 4-bit
// columns laid end to end (column c from nibble c * pad, even row in the low nibble), so it is uploaded from them
// and there is no row-major matrix to transpose. The device keeps ONE copy of the bin matrix, in the layout the
// training regime reads:
//  - compact regime (low feature_fraction): the store; every tree fills its compact view of the sampled columns
//    from it. A reader that needs every column in the row-major layout (a tree that samples every column, a split
//    view the compact matrix cannot serve) gets the full view, filled from the store and kept while trees use it.
//  - mask regime (high feature_fraction): the full view, the compact view of every column (the row-major matrix's
//    bytes and layout), filled once from the Dataset's columns a staging chunk at a time, with no store; no tree
//    copies anything, the kernels apply the per-tree sample through their column masks.
// INVARIANT (quantized training): both regimes feed every kernel the same bin values for the same sampled columns,
// and the quantized histograms are integer sums (order-invariant), so a tree trains to the same bytes in either
// regime. Switching regimes between trees -- the tree learner's tuner probe does, as does a feature_fraction change
// -- is therefore legal at any tree and never changes the model (tests/python_package_test/test_colmajor_direct.py
// checks it). Float histograms are not order-invariant: non-quantized training takes the mask regime only where every
// tree samples every column (the same kernels either way), see ViewMaskWanted.
// The tree learner chooses (ChooseViewRegime: FalcataPlan::view_mode / view_mask_ff, the memory check, the tuner's
// probe or its wisdom) at the end of its Init, on ResetTrainingData and when ResetConfig changes feature_fraction;
// the row data only says whether the direct build applies.
bool CUDAHistogramConstructor::ViewMaskWanted() const {
  const FalcataPlan& plan = FalcataPlan::Get();
  if (plan.view_mode == FalcataPlan::kViewModeCompact) {
    return false;
  }
  if (plan.view_mode == FalcataPlan::kViewModeMask) {
    return true;
  }
  if (!use_quantized_grad_) {
    // float histograms: the full-matrix kernels group rows by the partition's column count, so under a strict column
    // sample they sum in another order than the compact view's (the last bits of the model move); only a training in
    // which every tree samples every column runs the same kernels in both regimes
    return SampledColumns() >= static_cast<size_t>(cuda_row_data_->host_feature_partition_column_index_offsets().back());
  }
  if (!CompactQuantEnabled()) {
    return true;  // compact_quant:off: no tree builds a compact view, so the store would serve nothing
  }
  return ViewFraction() >= plan.view_mask_ff;
}

size_t CUDAHistogramConstructor::SampledColumns() const {
  // ColSampler samples round(F x feature_fraction) of the F features (at least one); a column carries at least one
  const size_t num_columns = static_cast<size_t>(cuda_row_data_->host_feature_partition_column_index_offsets().back());
  const int sampled_features =
    std::max(std::min(1, num_features_), static_cast<int>(Common::RoundInt(num_features_ * ViewFraction())));
  return std::min(num_columns, static_cast<size_t>(sampled_features));
}

double CUDAHistogramConstructor::ViewFraction() const {
  return (feature_fraction_ > 0.0 && feature_fraction_ < 1.0) ? feature_fraction_ : 1.0;
}

CUDAHistogramConstructor::CompactRegimeBytes CUDAHistogramConstructor::CompactRegimeNeed(
    const bool one_byte_split_view) const {
  CompactRegimeBytes need;
  const std::vector<int>& part_cols = cuda_row_data_->host_feature_partition_column_index_offsets();
  const size_t num_data = static_cast<size_t>(num_data_);
  const size_t num_columns = static_cast<size_t>(part_cols.back());
  const size_t num_partitions = part_cols.size() - 1;
  need.store = num_columns * cuda_row_data_->colmajor_column_bytes();
  // The compact view (ScanCompactLayout's arithmetic): a partition's s_p sampled columns take ceil(s_p / 2) bytes
  // per row, at most (S + P) / 2 rounded up over the P partitions for S sampled columns.
  const size_t sampled = SampledColumns();
  need.view = std::min(cuda_row_data_->dense_4bit_bytes(), (sampled + num_partitions + 1) / 2 * num_data);
  if (FalcataPlan::Get().compact_prefill) {
    need.view *= 2;  // the next tree's view, filled on the side
  }
  // the tree learner's one-byte-per-value split view of the sampled columns, when it builds one (its own predicate,
  // CUDASingleGPUTreeLearner::BuildsOneByteSplitView)
  if (one_byte_split_view) {
    need.column_view = sampled * num_data;
  }
  return need;
}

size_t CUDAHistogramConstructor::HeldViewBytes() const {
  return colmajor_bin_.Size() + full_view_.Size() + compact_data_uint8_t_.Size() + compact_data_uint8_t_alt_.Size() +
         colmajor_dense3_.Size() * sizeof(uint32_t);
}

void CUDAHistogramConstructor::ClearColMajorDense3() {
  colmajor_dense3_.Clear();
  colmajor_dense3_pitch_ = 0;
  colmajor_dense3_codec_ = 0;
  colmajor_dense3_wide_.clear();
}

// cuda_plan key colmajor_dense3: the store's 3-bit copy (3 words per 32 rows instead of 4), read by the fused fill
// only. The fill reads the store once per tree, so the denser copy cuts its source bytes by a quarter for one ALU
// decode per 32 rows; the per-level split reads and the compact view (the construct's hot path) keep the nibbles.
// Built when it fits next to the compact regime's views and the tree learner's reserve; else the fill reads the
// 4-bit store as before. Same nibbles staged: bit-identical.
void CUDAHistogramConstructor::MaybeBuildColMajorDense3() {
  ClearColMajorDense3();
  if (!FalcataPlan::Get().colmajor_dense3 || !use_quantized_grad_ || !FalcataPlan::Get().fused_root_hist ||
      colmajor_pad_ == 0 || view_mask_) {
    return;
  }
  const int num_columns = cuda_row_data_->host_feature_partition_column_index_offsets().back();
  // the codec: base 6 (12 rows per word) where every value of the store is below 6, else 3 bits per row (columns
  // holding 8 or more are not served)
  CUDAVector<int> flags(static_cast<size_t>(num_columns) + 1);
  SetCUDAMemory<int>(flags.RawData(), 0, flags.Size(), __FILE__, __LINE__);
  int any_wide6 = 1;
  if (FalcataPlan::Get().colmajor_dense6) {
    LaunchColMajorAnyAtLeast6(colmajor_bin_.RawDataReadOnly(), colmajor_bin_.Size(), flags.RawData() + num_columns);
    CopyFromCUDADeviceToHost<int>(&any_wide6, flags.RawDataReadOnly() + num_columns, 1, __FILE__, __LINE__);
  }
  const int codec = any_wide6 == 0 ? 6 : 3;
  const int radix = codec == 6 ? 6 : 8;
  const size_t pitch = ColMajorDensePitchWords(num_data_, codec);
  const size_t bytes = static_cast<size_t>(num_columns) * pitch * sizeof(uint32_t);
  const CompactRegimeBytes need = CompactRegimeNeed(view_one_byte_split_view_);
  size_t free_b = 0, total_b = 0;
  CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_b, &total_b));
  const int64_t cap_mib = FalcataDebug().vramfree_mib;
  if (cap_mib >= 0) {
    free_b = std::min(free_b, static_cast<size_t>(cap_mib) << 20);
  }
  // the compact view and the split view are not allocated yet on entering the regime
  if (bytes + need.view + need.column_view + view_reserve_bytes_ > free_b) {
    LogDirectView("colmajor_dense3: the dense copy (%zu MiB) does not fit (%zu MiB free)", ToMiB(bytes),
                  free_b / kMiB);
    return;
  }
  if (!colmajor_dense3_.TryResizeDiscard(bytes / sizeof(uint32_t))) {
    colmajor_dense3_.Clear();
    return;
  }
  LaunchEncodeColMajorDense(colmajor_bin_.RawDataReadOnly(), colmajor_pad_ / 2, num_columns, num_data_,
                            colmajor_dense3_.RawData(), pitch, codec, flags.RawData());
  colmajor_dense3_wide_.resize(static_cast<size_t>(num_columns));
  CopyFromCUDADeviceToHost<int>(colmajor_dense3_wide_.data(), flags.RawDataReadOnly(), static_cast<size_t>(num_columns),
                                __FILE__, __LINE__);
  colmajor_dense3_pitch_ = pitch;
  colmajor_dense3_codec_ = codec;
  const size_t narrow = static_cast<size_t>(std::count(colmajor_dense3_wide_.begin(), colmajor_dense3_wide_.end(), 0));
  if (narrow == 0) {
    ClearColMajorDense3();
  }
  LogDirectView("colmajor_dense3: codec %d copy of the store for the fill, %zu of %d columns below %d (%zu MiB)", codec,
                narrow, num_columns, radix, ToMiB(bytes));
}

void CUDAHistogramConstructor::ChooseViewRegime(const size_t reserve_bytes, const bool one_byte_split_view,
                                                const bool trees_read_full_matrix, const int prefer,
                                                const char* why) {
  if (!colmajor_direct_) {
    return;
  }
  const bool initial = colmajor_bin_.Size() == 0 && full_view_.Size() == 0;
  // Does the compact regime fit? Deterministic: its sizes against what cudaMemGetInfo reports free now, plus what
  // the views hold now (a regime change frees them first), plus the reserve the tree learner computed for
  // everything it can still allocate. FALCATA_DEBUG=vramfree=N caps the free figure, for tests.
  const CompactRegimeBytes need = CompactRegimeNeed(one_byte_split_view);
  size_t free_b = 0, total_b = 0;
  CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_b, &total_b));
  const int64_t cap_mib = FalcataDebug().vramfree_mib;
  if (cap_mib >= 0) {
    free_b = std::min(free_b, static_cast<size_t>(cap_mib) << 20);
  }
  const size_t available = free_b + HeldViewBytes();
  compact_fits_ = need.total() + reserve_bytes <= available;
  view_reserve_bytes_ = reserve_bytes;  // for the 3-bit copy (MaybeBuildColMajorDense3)
  view_one_byte_split_view_ = one_byte_split_view;
  LogDirectView("colmajor_direct: the compact regime needs store %zu + compact view %zu + split view %zu + reserve %zu "
                "MiB, %zu MiB available", ToMiB(need.store), ToMiB(need.view), ToMiB(need.column_view),
                ToMiB(reserve_bytes), available / kMiB);
  bool mask = prefer >= 0 ? (prefer == 1) : ViewMaskWanted();
  const char* reason = why;
  const bool forced_compact = FalcataPlan::Get().view_mode == FalcataPlan::kViewModeCompact;
  if (!use_quantized_grad_) {
    // float histograms: the regime is the static rule's (ViewMaskWanted), never the memory check's, since the mask
    // regime would change the model under a strict column sample. The compact regime holds the store and the views
    // the row-major matrix held them next to before, so it needs no more memory than colmajor_direct:off.
    compact_fits_ = false;  // and no probe (the tuner is quantized-only anyway)
  } else if (trees_read_full_matrix && !forced_compact) {
    // every tree's per-leaf construct reads the whole matrix row-major: the compact regime would hold the store and
    // the full view and fill a compact view nothing reads
    compact_fits_ = false;
    if (!mask) {
      mask = true;
      reason = "the per-leaf quantized construct reads every column";
    }
  }
  if (!mask && !compact_fits_ && !forced_compact && use_quantized_grad_) {
    LogDirectView("colmajor_direct: the compact regime does not fit (store %zu + compact view %zu + split view %zu + "
                  "reserve %zu MiB > %zu MiB available): mask regime", ToMiB(need.store), ToMiB(need.view),
                  ToMiB(need.column_view), ToMiB(reserve_bytes), available / kMiB);
    mask = true;
    reason = "the compact regime does not fit";
  }
  if (initial || mask != view_mask_) {
    if (mask) {
      EnterMaskRegime(reason);
    } else {
      EnterCompactRegime(reason);
    }
  } else if (!view_mask_ && colmajor_dense3_.Size() > 0 &&
             need.total() + reserve_bytes + colmajor_dense3_.Size() * sizeof(uint32_t) > available) {
    // the regime stays compact, but its dense copy (counted as held above) no longer fits next to the regime's need
    // and the reserve (a grown one, say): release it, the fill reads the 4-bit store
    LogDirectView("colmajor_dense3: dense copy released (%zu MiB), it no longer fits next to the reserve",
                  ToMiB(colmajor_dense3_.Size() * sizeof(uint32_t)));
    ClearColMajorDense3();
  }
}

void CUDAHistogramConstructor::SwitchViewRegime(const bool mask, const char* why) {
  if (colmajor_direct_ && mask != view_mask_) {
    if (mask) {
      EnterMaskRegime(why);
    } else {
      EnterCompactRegime(why);
    }
  }
}

namespace {
const char* ViewModeName() {
  switch (FalcataPlan::Get().view_mode) {
    case FalcataPlan::kViewModeCompact: return "compact";
    case FalcataPlan::kViewModeMask: return "mask";
    default: return "auto";
  }
}
}  // namespace

// Both transitions free the old layout before building the new one, so a switch peaks at one matrix (plus the
// mask fill's 256 MiB staging chunk).
void CUDAHistogramConstructor::EnterCompactRegime(const char* why) {
  InvalidateCompactPrefill();
  view_mask_ = false;
  const size_t view_bytes = full_view_.Size();
  full_view_.Clear();
  ClearColMajorDense3();
  const int num_columns = cuda_row_data_->host_feature_partition_column_index_offsets().back();
  const size_t column_bytes = cuda_row_data_->colmajor_column_bytes();
  const size_t bytes = static_cast<size_t>(num_columns) * column_bytes;
  colmajor_bin_.ResizeDiscard(bytes);
  cuda_row_data_->UploadDense4BitColumnsColMajor(colmajor_bin_.RawData(), 0, num_columns);
  if ((num_data_ & 1) != 0) {
    // the nibble past the last row: the Dataset's byte may hold anything, the transpose this replaces wrote 0
    LaunchClearColMajorPadNibbles(colmajor_bin_.RawData(), num_columns, column_bytes);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  colmajor_pad_ = 2 * column_bytes;
  if (FalcataVerifyEnabled()) {
    cuda_row_data_->VerifyDense4BitColMajor(colmajor_bin_.RawDataReadOnly());
  }
  LogDirectView("colmajor_direct: compact regime (%s; feature_fraction %g, view_mode %s, view_mask_ff %g): "
                "column-major store built from the Dataset's columns, %d columns (%zu MiB); no row-major matrix on "
                "the device%s", why, feature_fraction_, ViewModeName(), FalcataPlan::Get().view_mask_ff, num_columns,
                ToMiB(bytes), view_bytes > 0 ? "; full view released" : "");
  MaybeBuildColMajorDense3();
}

void CUDAHistogramConstructor::EnterMaskRegime(const char* why) {
  InvalidateCompactPrefill();
  view_mask_ = true;
  const size_t store_bytes = colmajor_bin_.Size();
  colmajor_bin_.Clear();
  ClearColMajorDense3();
  colmajor_pad_ = 0;
  // no tree builds a compact view in this regime: its buffers go too
  compact_data_uint8_t_.Clear();
  compact_data_uint8_t_alt_.Clear();
  full_view_.Clear();
  FillFullViewFromColumns();
  if (FalcataVerifyEnabled()) {
    cuda_row_data_->VerifyDense4BitRowMajor(full_view_.RawDataReadOnly(), "full view filled from the Dataset's columns");
  }
  LogDirectView("colmajor_direct: mask regime (%s; feature_fraction %g, view_mode %s, view_mask_ff %g): full view "
                "filled from the Dataset's columns (%zu MiB); no column-major store on the device%s", why,
                feature_fraction_, ViewModeName(), FalcataPlan::Get().view_mask_ff, ToMiB(full_view_.Size()),
                store_bytes > 0 ? "; column-major store released" : "");
}

CUDAHistogramConstructor::CompactLayout CUDAHistogramConstructor::FullLayout() const {
  const std::vector<int>& part_cols = cuda_row_data_->host_feature_partition_column_index_offsets();
  const std::vector<uint32_t>& col_hist_offsets = cuda_row_data_->host_column_hist_offsets();
  CompactLayout layout;
  layout.num_partitions = static_cast<int>(part_cols.size()) - 1;
  layout.part_col_offsets = part_cols;
  for (int p = 0; p < layout.num_partitions; ++p) {
    layout.src_part_stride.push_back(part_cols[p + 1] - part_cols[p]);
    for (int c = part_cols[p]; c < part_cols[p + 1]; ++c) {
      layout.src_local_col.push_back(c - part_cols[p]);
      layout.partition_for_slot.push_back(p);
      layout.col_hist_offsets.push_back(col_hist_offsets[c]);
    }
  }
  layout.total_compact = part_cols.back();
  layout.is_4bit = true;
  layout.codec = PackCodecId::kNibble4;
  layout.packed_part_offsets = cuda_row_data_->host_packed_partition_byte_offsets();
  layout.data_bytes = cuda_row_data_->dense_4bit_bytes();
  return layout;
}

void CUDAHistogramConstructor::FillFullViewFromStore() {
  const CompactLayout layout = FullLayout();
  full_view_.ResizeDiscard(layout.data_bytes);
  // the compact fill of every column, reading the store (colmajor_pad_ > 0)
  LaunchCompactFill(layout, full_view_.RawData(), cuda_stream_, /*async_meta=*/false);
  CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_stream_));
}

void CUDAHistogramConstructor::FillFullViewFromColumns() {
  // The same 4-bit fill kernel as the compact fill, reading a staging slice of the store instead of the whole
  // store: per partition, chunks of an even number of columns are uploaded column-major into the staging buffer
  // and their byte slots (column pairs 2m, 2m + 1) written into the full view. The full view and one staging
  // chunk are the only device allocations, so the mask regime never holds two copies of the matrix.
  const std::vector<int>& part_cols = cuda_row_data_->host_feature_partition_column_index_offsets();
  const std::vector<int>& packed = cuda_row_data_->host_packed_partition_byte_offsets();
  const size_t column_bytes = cuda_row_data_->colmajor_column_bytes();
  const size_t pad = 2 * column_bytes;
  const size_t num_data = static_cast<size_t>(num_data_);
  full_view_.ResizeDiscard(cuda_row_data_->dense_4bit_bytes());
  constexpr size_t kStagingBytes = size_t{256} << 20;
  const int chunk_columns = std::max(2, static_cast<int>(kStagingBytes / column_bytes) & ~1);
  int widest = 0;
  for (size_t p = 0; p + 1 < part_cols.size(); ++p) {
    widest = std::max(widest, part_cols[p + 1] - part_cols[p]);
  }
  const int staged_columns = std::min(chunk_columns, widest);
  CUDAVector<uint8_t> staging(static_cast<size_t>(staged_columns) * column_bytes);
  const size_t kNone = ~static_cast<size_t>(0);
  std::vector<size_t> nib0, nib1, dst_byte;
  std::vector<int> stride_nib, dst_stride;
  for (size_t p = 0; p + 1 < part_cols.size(); ++p) {
    const int width = part_cols[p + 1] - part_cols[p];
    const int packed_width = packed[p + 1] - packed[p];
    for (int j0 = 0; j0 < width; j0 += chunk_columns) {
      const int j1 = std::min(width, j0 + chunk_columns);
      cuda_row_data_->UploadDense4BitColumnsColMajor(staging.RawData(), part_cols[p] + j0, part_cols[p] + j1);
      nib0.clear(); nib1.clear(); dst_byte.clear(); stride_nib.clear(); dst_stride.clear();
      // j0 is even and every chunk but a partition's last has an even width, so no byte slot straddles two chunks
      for (int m = j0 / 2; m < (j1 + 1) / 2; ++m) {
        nib0.push_back(static_cast<size_t>(2 * m - j0) * pad);
        nib1.push_back(2 * m + 1 < width ? static_cast<size_t>(2 * m + 1 - j0) * pad : kNone);
        stride_nib.push_back(1);
        dst_byte.push_back(static_cast<size_t>(packed[p]) * num_data + static_cast<size_t>(m));
        dst_stride.push_back(packed_width);
      }
      const int slots = static_cast<int>(nib0.size());
      if (cuda_bs_src_nib0_.Size() < static_cast<size_t>(slots)) {
        cuda_bs_src_nib0_.Resize(slots);
        cuda_bs_src_nib1_.Resize(slots);
        cuda_bs_src_stride_nib_.Resize(slots);
        cuda_bs_dst_byte_.Resize(slots);
        cuda_bs_dst_stride_.Resize(slots);
      }
      CopyFromHostToCUDADevice<size_t>(cuda_bs_src_nib0_.RawData(), nib0.data(), slots, __FILE__, __LINE__);
      CopyFromHostToCUDADevice<size_t>(cuda_bs_src_nib1_.RawData(), nib1.data(), slots, __FILE__, __LINE__);
      CopyFromHostToCUDADevice<int>(cuda_bs_src_stride_nib_.RawData(), stride_nib.data(), slots, __FILE__, __LINE__);
      CopyFromHostToCUDADevice<size_t>(cuda_bs_dst_byte_.RawData(), dst_byte.data(), slots, __FILE__, __LINE__);
      CopyFromHostToCUDADevice<int>(cuda_bs_dst_stride_.RawData(), dst_stride.data(), slots, __FILE__, __LINE__);
      LaunchFillCompactData4BitKernel(cuda_stream_, staging.RawDataReadOnly(), full_view_.RawData(),
                                      cuda_bs_src_nib0_.RawData(), cuda_bs_src_nib1_.RawData(),
                                      cuda_bs_src_stride_nib_.RawData(), cuda_bs_dst_byte_.RawData(),
                                      cuda_bs_dst_stride_.RawData(), slots, num_data_, /*tiled=*/false);
      // the next chunk overwrites the staging buffer and the slot metadata
      CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_stream_));
    }
  }
}

const uint8_t* CUDAHistogramConstructor::EnsureFullView(const char* reason) {
  if (!colmajor_direct_) {
    return RowMajorBin<uint8_t>();
  }
  full_view_used_ = true;
  if (full_view_.Size() == 0) {
    if (colmajor_pad_ > 0) {
      FillFullViewFromStore();
    } else {
      FillFullViewFromColumns();
    }
    if (FalcataVerifyEnabled()) {
      cuda_row_data_->VerifyDense4BitRowMajor(full_view_.RawDataReadOnly(), "full view filled on demand");
    }
    LogDirectView("colmajor_direct: full view filled from the %s for %s (%zu MiB)",
                  colmajor_pad_ > 0 ? "column-major store" : "Dataset's columns", reason, ToMiB(full_view_.Size()));
  }
  return full_view_.RawDataReadOnly();
}

void CUDAHistogramConstructor::ReleaseUnusedFullView() {
  if (!colmajor_direct_ || view_mask_ || full_view_.Size() == 0 || full_view_used_) {
    return;
  }
  const size_t bytes = full_view_.Size();
  full_view_.Clear();
  LogDirectView("colmajor_direct: full view released (%zu MiB), this tree read only its compact view", ToMiB(bytes));
}

bool CUDAHistogramConstructor::ColMajorFillApplicable() const {
  return FalcataPlan::Get().colmajor_fill && cuda_row_data_ != nullptr && cuda_row_data_->is_4bit_packed() &&
         !cuda_row_data_->is_data_host_mapped() && colmajor_pad_ == 0 && !colmajor_direct_;
}

// The column-major copy duplicates the 4-bit row matrix (one nibble per value) so the per-tree compact fill reads
// contiguous columns. It is optional: the fill reads the row-major matrix without it and writes the same bytes, so
// it must never be the allocation that makes training run out of memory. Three checks, each against free device
// memory as cudaMemGetInfo reports it (FALCATA_DEBUG=vramfree=N caps that figure, for tests):
//  1. before allocating, free >= copy + reserve, so an allocation that cannot fit is never asked for;
//  2. the allocation itself, through a non-fatal cudaMalloc (another process may have taken memory since);
//  3. after allocating, free >= reserve; otherwise the copy is released again.
// reserve_bytes is what training can still allocate after this point (the tree learner's
// ColMajorFillReserveBytes). A decline logs a warning with the figures; FALCATA_DEBUG=diag logs an engagement.
bool CUDAHistogramConstructor::InitColMajorFill(const size_t reserve_bytes) {
  if (!ColMajorFillApplicable()) {
    return false;
  }
  const std::vector<int>& part_cols = cuda_row_data_->host_feature_partition_column_index_offsets();
  const std::vector<int>& packed_offsets = cuda_row_data_->host_packed_partition_byte_offsets();
  const int num_columns = part_cols.back();
  const size_t pad = (static_cast<size_t>(num_data_) + 1) & ~static_cast<size_t>(1);
  const size_t bytes = static_cast<size_t>(num_columns) * (pad / 2);
  if (bytes == 0) {
    return false;
  }
  const int64_t cap_mib = FalcataDebug().vramfree_mib;
  const size_t cap = cap_mib >= 0 ? static_cast<size_t>(cap_mib) << 20 : std::numeric_limits<size_t>::max();
  auto free_bytes = [](size_t limit) {
    size_t free_b = 0, total_b = 0;
    CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_b, &total_b));
    return std::min(free_b, limit);
  };
  constexpr size_t kMiB = 1ULL << 20;
  const size_t free_before = free_bytes(cap);
  auto decline = [&](const char* reason) {
    Log::Warning("colmajor_fill: declined, %s (copy %zu MiB, reserve %zu MiB, free %zu MiB before the copy); "
                 "the per-tree fill reads the row-major matrix instead (same model, slower fill)",
                 reason, (bytes + kMiB - 1) / kMiB, (reserve_bytes + kMiB - 1) / kMiB, free_before / kMiB);
    return false;
  };
  if (free_before < bytes || free_before - bytes < reserve_bytes) {
    return decline("not enough free device memory");
  }
  if (!colmajor_bin_.TryResizeDiscard(bytes)) {
    return decline("the allocation failed");
  }
  const size_t free_after = free_bytes(cap > bytes ? cap - bytes : 0);
  if (free_after < reserve_bytes) {
    colmajor_bin_.Clear();
    return decline("free memory after the copy is below the reserve");
  }
  std::vector<size_t> base_h(num_columns);
  std::vector<int> stride_h(num_columns);
  for (int p = 0; p + 1 < static_cast<int>(part_cols.size()); ++p) {
    const size_t part_nib = static_cast<size_t>(packed_offsets[p]) * static_cast<size_t>(num_data_) * 2;
    const int stride_nib = (packed_offsets[p + 1] - packed_offsets[p]) * 2;
    for (int c = part_cols[p]; c < part_cols[p + 1]; ++c) {
      base_h[c] = part_nib + static_cast<size_t>(c - part_cols[p]);
      stride_h[c] = stride_nib;
    }
  }
  CUDAVector<size_t> d_base(num_columns);
  CUDAVector<int> d_stride(num_columns);
  CopyFromHostToCUDADevice<size_t>(d_base.RawData(), base_h.data(), num_columns, __FILE__, __LINE__);
  CopyFromHostToCUDADevice<int>(d_stride.RawData(), stride_h.data(), num_columns, __FILE__, __LINE__);
  // synchronizes: every later fill may read the copy
  LaunchTransposeToColMajorNibbleKernel(
    RowMajorBin<uint8_t>(), colmajor_bin_.RawData(),
    d_base.RawData(), d_stride.RawData(), num_columns, num_data_, pad);
  colmajor_pad_ = pad;
  const char* fmt = "colmajor_fill: engaged, %d columns (copy %zu MiB, reserve %zu MiB, free %zu MiB before the copy)";
  if (FalcataDebug().diag) {
    Log::Info(fmt, num_columns, (bytes + kMiB - 1) / kMiB, (reserve_bytes + kMiB - 1) / kMiB, free_before / kMiB);
  } else {
    Log::Debug(fmt, num_columns, (bytes + kMiB - 1) / kMiB, (reserve_bytes + kMiB - 1) / kMiB, free_before / kMiB);
  }
  return true;
}

// Upper bound on how much the compact view buffers can grow after the trees built so far. Every tree samples the
// same number of columns (num_compact_columns_), but how they fall into the feature partitions, and with the
// experimental pack codecs the codec, changes from tree to tree. The nibble layout is the widest 4-bit layout:
// ceil(c / 2) bytes per row for a partition's c sampled columns, at most (num_compact_columns_ + partitions) / 2
// over all partitions. ResizeDiscard frees a buffer before regrowing it, so the growth is that bound minus the
// current size, for the view and (compact_prefill) its double buffer.
size_t CUDAHistogramConstructor::CompactViewGrowthBytes() const {
  if (cuda_row_data_ == nullptr || !cuda_row_data_->is_4bit_packed() || num_compact_columns_ <= 0) {
    return 0;
  }
  const size_t num_partitions = cuda_row_data_->host_feature_partition_column_index_offsets().size() - 1;
  const size_t bound = (static_cast<size_t>(num_compact_columns_) + num_partitions + 1) / 2 *
                       static_cast<size_t>(num_data_);
  auto growth = [bound](size_t current) { return bound > current ? bound - current : 0; };
  return growth(compact_data_uint8_t_.Size()) +
         (FalcataPlan::Get().compact_prefill ? growth(compact_data_uint8_t_alt_.Size()) : 0);
}

size_t CUDAHistogramConstructor::BitChangeScratchGrowthBytes(const int max_pairs) const {
  if (!use_quantized_grad_) {
    return 0;
  }
  // ConstructHistogramsForLevel's Resize keeps the old block until the new one is allocated
  const size_t needed = static_cast<size_t>(std::max(max_pairs, 0)) * static_cast<size_t>(num_total_bin_);
  return hist_buffer_for_num_bit_change_.Size() >= needed ? 0 : needed * sizeof(hist_t);
}

void CUDAHistogramConstructor::ConstructHistogramForLeaf(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* /*cuda_larger_leaf_splits*/,
  const data_size_t global_num_data_in_smaller_leaf,
  const data_size_t global_num_data_in_larger_leaf,
  const data_size_t num_data_in_smaller_leaf,
  const data_size_t /*num_data_in_larger_leaf*/,
  const double sum_hessians_in_smaller_leaf,
  const double sum_hessians_in_larger_leaf,
  const uint8_t num_bits_in_histogram_bins) {
if ((global_num_data_in_smaller_leaf <= min_data_in_leaf_ || sum_hessians_in_smaller_leaf <= min_sum_hessian_in_leaf_) &&
    (global_num_data_in_larger_leaf <= min_data_in_leaf_ || sum_hessians_in_larger_leaf <= min_sum_hessian_in_leaf_)) {
    return;
  }
  LaunchConstructHistogramKernel(cuda_smaller_leaf_splits, num_data_in_smaller_leaf, num_bits_in_histogram_bins);
  // Record completion on cuda_stream_ instead of a device-wide sync. The best split
  // finder waits on this event before reading the smaller-leaf histogram, so the host
  // is not stalled here (see CUDABestSplitFinder::FindBestSplitsForLeaf).
  CUDASUCCESS_OR_FATAL(cudaEventRecord(construct_done_events_[active_pipeline_], current_stream()));
}

void CUDAHistogramConstructor::ConstructHistogramsForLevel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const data_size_t max_num_data_in_smaller_leaf,
  const bool any_pair_needs_bit_change_copy,
  const data_size_t* level_smaller_num_data,
  const bool defer_subtract,
  const bool use_fused_root) {
  // a fused root histogram serves only the root level, the first level construct of the tree
  fused_root_ready_ = false;
  if (num_pairs <= 0) {
    return;
  }
  if (use_quantized_grad_ && any_pair_needs_bit_change_copy) {
    // one int32 region of num_total_bin_ entries per pair; sized in hist_t (8 byte)
    // units so this allocates twice the strict need, but is only reached when a
    // level actually contains a >16-bit parent with a <=16-bit larger child
    const size_t needed = static_cast<size_t>(num_pairs) * static_cast<size_t>(num_total_bin_);
    if (hist_buffer_for_num_bit_change_.Size() < needed) {
      hist_buffer_for_num_bit_change_.Resize(needed);
    }
  }
  global_timer.Start("CUDAHistogramConstructor::ConstructHistogramsForLevel");
  // The batched construct kernel routes each pair whose ACTUAL smaller leaf is
  // tiny (on-device check against SmallLeafRowThreshold(); non-quantized only)
  // to a direct global-atomic body that skips the shared-histogram zero+merge
  // dominating small leaves. The quantized path never takes it: its integer
  // shared-then-merge accumulation (and the covtype quant md5 lock) stays
  // byte-identical.
  if (use_fused_root) {
    // the root level, whose rows the fill already accumulated (ConsumeFusedRootHist)
    LaunchApplyFusedRootHistogram(pair_descs);
  } else {
    LaunchConstructHistogramBatchedKernel(pair_descs, num_pairs, max_num_data_in_smaller_leaf,
                                          level_smaller_num_data);
  }
  // (no construct_done event here: the batched find kernel only waits on the
  // subtract event below, and the per-pair path re-records its own events)
  if (defer_subtract) {
    // Multi-GPU: the caller all-reduces the smaller-leaf histograms and then
    // runs SubtractHistogramsForLevel. Record the construct completion so the
    // reduce stream can wait on it (the subtract event is NOT recorded yet).
    CUDASUCCESS_OR_FATAL(cudaEventRecord(construct_done_events_[0], cuda_stream_));
    global_timer.Stop("CUDAHistogramConstructor::ConstructHistogramsForLevel");
    return;
  }
  SubtractHistogramsForLevel(pair_descs, num_pairs, any_pair_needs_bit_change_copy);
  global_timer.Stop("CUDAHistogramConstructor::ConstructHistogramsForLevel");
}

void CUDAHistogramConstructor::SubtractHistogramsForLevel(
  const CUDAHybridPairDescriptor* pair_descs,
  const int num_pairs,
  const bool any_pair_needs_bit_change_copy) {
  if (num_pairs <= 0) {
    return;
  }
  if (!use_quantized_grad_ && SmallLeafConstructEnabled()) {
    // fused fix + subtract: one launch, bit-identical to the sequential pair
    LaunchFixSubtractHistogramSmallLeafBatchedKernel(pair_descs, num_pairs);
  } else {
    LaunchSubtractHistogramBatchedKernel(pair_descs, num_pairs, any_pair_needs_bit_change_copy);
  }
  // the best split finder's batched find kernel waits on this event before reading
  // any of this level's histograms (construct/fix/subtract are stream-ordered here)
  CUDASUCCESS_OR_FATAL(cudaEventRecord(subtract_done_events_[0], cuda_stream_));
}

void CUDAHistogramConstructor::InitFixMFBMask() {
  // mask over the 2 * num_total_bin_ histogram entries: 1 at the most-frequent-
  // bin gradient/hessian slots of the need-fix features (owned by the fused
  // small-leaf kernel's fix blocks), 0 everywhere else (subtract blocks)
  std::vector<uint8_t> host_mask(static_cast<size_t>(2 * num_total_bin_), 0);
  for (const int feature_index : need_fix_histogram_features_) {
    const size_t pos = (static_cast<size_t>(feature_hist_offsets_[feature_index]) +
                        static_cast<size_t>(feature_most_freq_bins_[feature_index])) << 1;
    host_mask[pos] = 1;
    host_mask[pos + 1] = 1;
  }
  cuda_fix_mfb_mask_.InitFromHostVector(host_mask);
}

void CUDAHistogramConstructor::SubtractHistogramForLeaf(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits,
  const bool use_quantized_grad,
  const uint8_t parent_num_bits_in_histogram_bins,
  const uint8_t smaller_num_bits_in_histogram_bins,
  const uint8_t larger_num_bits_in_histogram_bins) {
  global_timer.Start("CUDAHistogramConstructor::ConstructHistogramForLeaf::LaunchSubtractHistogramKernel");
  LaunchSubtractHistogramKernel(cuda_smaller_leaf_splits, cuda_larger_leaf_splits, use_quantized_grad,
                                parent_num_bits_in_histogram_bins, smaller_num_bits_in_histogram_bins, larger_num_bits_in_histogram_bins);
  // Record completion on cuda_stream_; the best split finder waits on this
  // event before reading the larger-leaf (subtracted) histogram -- the only
  // ordering between the subtract and the FindBestSplits launches.
  CUDASUCCESS_OR_FATAL(cudaEventRecord(subtract_done_events_[active_pipeline_], current_stream()));
  global_timer.Stop("CUDAHistogramConstructor::ConstructHistogramForLeaf::LaunchSubtractHistogramKernel");
}

// Quantized construct kernels accumulate a block's rows into PACKED int32
// shared-memory cells (hessian in the low 16 bits, gradient in the high 16;
// see ConstructDiscretizedHistogramDenseInner). A single bin of one block must
// therefore never receive more than 65534 / num_grad_quant_bins rows: the
// per-row quantized hessian is at most num_grad_quant_bins, so beyond that the
// hessian field overflows and carries into the gradient field (and the signed
// gradient field, at most num_grad_quant_bins / 2 per row, would overflow at
// the same row count). The historical sizing (NUM_DATA_PER_THREAD = 400 rows
// per thread) admits up to 400 * block_dim_y rows per block-bin, which is safe
// for the default 4 (and 8) gradient quantization bins but silently corrupts
// histograms at num_grad_quant_bins >= 16 on skewed features (observed as a
// catastrophic, scale-dependent AUC collapse on higgs at >= 2.2M rows).
// Returns the maximum safe rows-per-thread for the quantized path.
int CUDAHistogramConstructor::QuantConstructMaxRowsPerThread(const int block_dim_y) const {
  // single source shared with the graph controller and the device row-grouping
  // replica (cuda_histogram_constructor.hpp)
  return HybridQuantConstructMaxRowsPerThread(num_grad_quant_bins_, block_dim_y);
}

void CUDAHistogramConstructor::CalcConstructHistogramKernelDim(
  int* grid_dim_x,
  int* grid_dim_y,
  int* block_dim_x,
  int* block_dim_y,
  const data_size_t num_data_in_smaller_leaf) {
  // wide partitions (>504 cols): each thread covers two columns
  {
    const int cols = cuda_row_data_->max_num_column_per_partition();
    *block_dim_x = cols > NUM_THREADS_PER_BLOCK ? (cols + 1) / 2 : cols;
  }
  *block_dim_y = NUM_THREADS_PER_BLOCK / *block_dim_x;
  // packed-shared-hist row budget is per block (see HybridQuantConstructBlockDimY)
  *block_dim_y = HybridQuantConstructBlockDimY(
    *block_dim_y, use_quantized_grad_ ? num_grad_quant_bins_ : 0);
  *grid_dim_x = cuda_row_data_->num_feature_partitions();
  int rows_per_thread = NUM_DATA_PER_THREAD;
  if (use_quantized_grad_) {
    rows_per_thread = std::min(rows_per_thread, QuantConstructMaxRowsPerThread(*block_dim_y));
  }
  *grid_dim_y = std::max(min_grid_dim_y_,
    ((num_data_in_smaller_leaf + rows_per_thread - 1) / rows_per_thread + (*block_dim_y) - 1) / (*block_dim_y));
}

void CUDAHistogramConstructor::CalcConstructHistogramBatchedKernelDim(
  int* grid_dim_x,
  int* grid_dim_y,
  int* block_dim_x,
  int* block_dim_y,
  const data_size_t max_num_data_in_smaller_leaf,
  const int num_pairs) {
  // wide partitions (>504 cols): each thread covers two columns
  {
    const int cols = cuda_row_data_->max_num_column_per_partition();
    *block_dim_x = cols > NUM_THREADS_PER_BLOCK ? (cols + 1) / 2 : cols;
  }
  *block_dim_y = NUM_THREADS_PER_BLOCK / *block_dim_x;
  // packed-shared-hist row budget is per block (see HybridQuantConstructBlockDimY)
  *block_dim_y = HybridQuantConstructBlockDimY(
    *block_dim_y, use_quantized_grad_ ? num_grad_quant_bins_ : 0);
  *grid_dim_x = cuda_row_data_->num_feature_partitions();
  // The per-leaf sizing forces min_grid_dim_y_ y-blocks to saturate the device
  // for a SINGLE leaf; every active block, however, pays a fixed shared-hist
  // zero + global-merge cost, which dominates small leaves. In the batched
  // kernel the pair grid dimension already provides parallelism, so share the
  // saturation floor across pairs and otherwise cap the y-grid at
  // min_rows_per_thread rows per thread (identical to the per-leaf sizing for
  // single-pair levels and for leaves large enough that the cap is inactive).
  // The formula lives in HybridBatchedConstructGridDimY(Quant) so the device
  // replicas (speculative single-sync flow, quantized graph loop) and the graph
  // controller compute the identical value; the quantized variant includes the
  // packed int32 shared-histogram overflow guard (QuantConstructMaxRowsPerThread).
  *grid_dim_y = HybridBatchedConstructGridDimYQuant(
    max_num_data_in_smaller_leaf, num_pairs, *block_dim_y, min_grid_dim_y_,
    BatchConstructMinRowsPerThread(), BatchConstructSaturationFloor(),
    use_quantized_grad_ ? num_grad_quant_bins_ : 0);
}

void CUDAHistogramConstructor::ResetTrainingData(const Dataset* train_data, TrainingShareStates* share_states) {
  num_data_ = train_data->num_data();
  num_features_ = train_data->num_features();
  InitFeatureMetaInfo(train_data, share_states->feature_hist_offsets());

  cuda_hist_.Resize(static_cast<size_t>(num_total_bin_ * 2 * num_leaves_) *
                    static_cast<size_t>(num_hist_planes_));
  cuda_hist_.SetValue(0);
  // Deterministic float-mode construct scratch: launches clamp their tile
  // grid to det_tile_alloc_, so the buffer can never be written past its end.
  det_tile_alloc_ = std::min(kDetTileCap, (num_data_ + kDetRowsPerThread - 1) / kDetRowsPerThread);
  // One region per histogram pipeline: pipelined pair-constructs run on
  // different pipeline streams, so a shared region would let one construct's
  // tiles overwrite another's before its merge reads them.
  cuda_det_tile_partials_.Resize(static_cast<size_t>(kNumHistPipelines) * det_tile_alloc_ * 2 * num_total_bin_);
  num_dirty_leaves_ = -1;
  cuda_feature_num_bins_.InitFromHostVector(feature_num_bins_);
  cuda_feature_hist_offsets_.InitFromHostVector(feature_hist_offsets_);
  cuda_feature_most_freq_bins_.InitFromHostVector(feature_most_freq_bins_);

  // the column-major copy (and colmajor_direct's views) belong to the old row data; the tree learner decides on a
  // new copy (colmajor_fill) or chooses a regime for the new views (colmajor_direct). An in-flight prefill may still
  // read the old store.
  InvalidateCompactPrefill();
  colmajor_bin_.Clear();
  ClearColMajorDense3();
  colmajor_pad_ = 0;
  full_view_.Clear();
  colmajor_direct_ = false;
  cuda_row_data_.reset(new CUDARowData(train_data, share_states, gpu_device_id_, gpu_use_dp_));
  cuda_row_data_->RequestDense4BitColumnsOnly(FalcataPlan::Get().colmajor_fill && FalcataPlan::Get().colmajor_direct);
  cuda_row_data_->Init(train_data, share_states);

  // Deterministic dense-construct scratch: sized from the widest partition's
  // item count once the row data exists. Non-quantized only; quantized
  // training keeps its order-invariant integer atomics.
  // Slot rows span the FULL histogram (positions are global per partition,
  // so partitions own disjoint ranges of each row).
  const std::vector<uint32_t>& dense_part_offsets = cuda_row_data_->host_partition_hist_offsets();
  const uint32_t dense_stride = dense_part_offsets.empty() ? 0u : (dense_part_offsets.back() << 1);
  det_dense_slot_stride_ = dense_stride;
  if (!use_quantized_grad_ && dense_stride > 0) {
    const size_t per_row = static_cast<size_t>(dense_stride) * sizeof(hist_t);
    det_dense_dy_ = std::max(1, std::min<int>(kDetDenseDyCap,
        static_cast<int>(kDetDenseSlotBudget / (static_cast<size_t>(kDetTileCap) * per_row))));
    // The dy formula floors at 1; ultra-wide strides then need the TILE
    // count to yield instead, or the slab exceeds the budget kDetTileCap-fold.
    det_dense_tile_cap_ = std::max(1, std::min<int>(kDetTileCap,
        static_cast<int>(kDetDenseSlotBudget / (static_cast<size_t>(det_dense_dy_) * per_row))));
    // One region per histogram pipeline: pipelined pair-constructs run on
    // different pipeline streams, so a shared region would let one
    // construct's slot rows overwrite another's before its merge reads them
    // (same failure mode the sparse tile partials needed per-pipeline
    // regions for). Worst case 4x the slot budget.
    cuda_det_dense_slots_.Resize(static_cast<size_t>(kNumHistPipelines) * det_dense_tile_cap_ * det_dense_dy_ * dense_stride);
  } else {
    det_dense_dy_ = 0;
    det_dense_tile_cap_ = 0;
  }

  hist_fp32_ = FalcataFP32HistRequested() && !use_quantized_grad_ && !gpu_use_dp_ &&
    !cuda_row_data_->is_sparse() && cuda_row_data_->NumLargeBinPartition() == 0 &&
    !has_categorical_feature_;

  cuda_need_fix_histogram_features_.InitFromHostVector(need_fix_histogram_features_);
  cuda_need_fix_histogram_features_num_bin_aligned_.InitFromHostVector(need_fix_histogram_features_num_bin_aligend_);
  InitFixMFBMask();
  // the tree learner chooses the regime for the new data (ChooseViewRegime)
  colmajor_direct_ = cuda_row_data_->dense_4bit_columns_only();
}

void CUDAHistogramConstructor::ResetConfig(const Config* config) {
  num_threads_ = OMP_NUM_THREADS();
  num_leaves_ = config->num_leaves;
  min_data_in_leaf_ = config->min_data_in_leaf;
  feature_fraction_ = config->feature_fraction;
  min_sum_hessian_in_leaf_ = config->min_sum_hessian_in_leaf;
  cuda_hist_.Resize(static_cast<size_t>(num_total_bin_ * 2 * num_leaves_) *
                    static_cast<size_t>(num_hist_planes_));
  cuda_hist_.SetValue(0);
  num_dirty_leaves_ = -1;
  // (colmajor_direct: the tree learner re-chooses the regime when feature_fraction changed)
}

}  // namespace Falcata

#endif  // USE_CUDA
