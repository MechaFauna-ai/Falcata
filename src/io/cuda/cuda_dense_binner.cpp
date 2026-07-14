/*!
 * Copyright (c) 2026 The ExaBoost developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 *
 * GPU-side dense matrix binning for device_type=cuda dataset construction:
 * replaces the host per-value binning + packing push loops. Row chunks are
 * streamed through a pinned staging ring (the raw matrix is never fully
 * device-resident), binned on device (per-column LUT for small-int/half
 * inputs, exact ValueToBin double binary search for float/double), and the
 * packed per-group column bins are copied back into the normal host DenseBin
 * storage so every downstream consumer works unchanged.
 */
#ifdef USE_CUDA

#include <LightGBM/dataset.h>

#include <LightGBM/bin.h>
#include <LightGBM/cuda/cuda_utils.hu>
#include <LightGBM/feature_group.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(__SSE2__) || (defined(_MSC_VER) && defined(_M_X64))
#include <emmintrin.h>
#define EXABOOST_DENSE_BINNER_SSE2 1
#endif

#include "cuda_dense_binner.hpp"

namespace LightGBM {

namespace {

// memcpy with non-temporal stores: the staging buffers are written once and
// consumed by the DMA engine (or much later), so bypassing the cache saves
// the read-for-ownership traffic that dominates these multi-GB copies
void StreamMemcpy(uint8_t* dst, const uint8_t* src, size_t len) {
#ifdef EXABOOST_DENSE_BINNER_SSE2
  while (len > 0 && (reinterpret_cast<uintptr_t>(dst) & 15) != 0) {
    *dst++ = *src++;
    --len;
  }
  const size_t vec = len / 16;
  __m128i* vdst = reinterpret_cast<__m128i*>(dst);
  if ((reinterpret_cast<uintptr_t>(src) & 15) == 0) {
    const __m128i* vsrc = reinterpret_cast<const __m128i*>(src);
    for (size_t i = 0; i < vec; ++i) {
      _mm_stream_si128(vdst + i, _mm_load_si128(vsrc + i));
    }
  } else {
    const __m128i* vsrc = reinterpret_cast<const __m128i*>(src);
    for (size_t i = 0; i < vec; ++i) {
      _mm_stream_si128(vdst + i, _mm_loadu_si128(vsrc + i));
    }
  }
  _mm_sfence();
  dst += vec * 16;
  src += vec * 16;
  len -= vec * 16;
  if (len > 0) {
    std::memcpy(dst, src, len);
  }
#else
  std::memcpy(dst, src, len);
#endif
}

int DTypeCode(Dataset::DenseBinnerDType dtype) {
  // kernel dtype codes; fp16 is binned through its uint16 bit patterns
  if (dtype == Dataset::DenseBinnerDType::kFloat16Bits) {
    return 5;
  }
  return static_cast<int>(dtype);
}

int64_t DTypeSize(Dataset::DenseBinnerDType dtype) {
  switch (dtype) {
    case Dataset::DenseBinnerDType::kFloat64:
      return 8;
    case Dataset::DenseBinnerDType::kFloat32:
      return 4;
    case Dataset::DenseBinnerDType::kInt16:
    case Dataset::DenseBinnerDType::kUInt16:
    case Dataset::DenseBinnerDType::kFloat16Bits:
      return 2;
    default:
      return 1;
  }
}

bool DTypeUsesLut(Dataset::DenseBinnerDType dtype) {
  return dtype != Dataset::DenseBinnerDType::kFloat32 &&
         dtype != Dataset::DenseBinnerDType::kFloat64;
}

int64_t DTypeLutSize(Dataset::DenseBinnerDType dtype) {
  return DTypeSize(dtype) == 2 ? 65536 : 256;
}

double LutIndexToValue(Dataset::DenseBinnerDType dtype, int64_t v) {
  switch (dtype) {
    case Dataset::DenseBinnerDType::kInt8:
      return static_cast<double>(static_cast<int8_t>(v - 128));
    case Dataset::DenseBinnerDType::kInt16:
      return static_cast<double>(static_cast<int16_t>(v - 32768));
    case Dataset::DenseBinnerDType::kFloat16Bits:
      return static_cast<double>(
          Common::HalfBitsToFloat(static_cast<uint16_t>(v)));
    default:  // kUInt8 / kUInt16 index the table directly
      return static_cast<double>(v);
  }
}

// bytes each value of a group's dense bin occupies; 0 means 4-bit packed
int GroupElemWidth(uint8_t bit_type) {
  switch (bit_type) {
    case 4:
      return 0;
    case 8:
      return 1;
    case 16:
      return 2;
    case 32:
      return 4;
    default:
      return -1;
  }
}

int64_t GroupBytesPerPair(int width) { return width == 0 ? 1 : 2 * width; }

int64_t GroupTotalBytes(int width, data_size_t num_data) {
  return width == 0 ? (static_cast<int64_t>(num_data) + 1) / 2
                    : static_cast<int64_t>(width) * num_data;
}

struct ChunkDesc {
  data_size_t start_row = 0;
  data_size_t rows = 0;
  data_size_t pairs = 0;
  bool valid = false;
};

}  // anonymous namespace

bool Dataset::GPUBinDenseRows(const void* data, DenseBinnerDType dtype,
                              data_size_t nrow, int ncol, bool is_row_major,
                              bool data_is_device) {
  if (device_type_ != std::string("cuda")) {
    return false;
  }
  const char* enable_env = std::getenv("EXABOOST_GPU_CONSTRUCT");
  if (enable_env != nullptr && std::string(enable_env) == std::string("0")) {
    return false;
  }
  const char* verify_env = std::getenv("EXABOOST_GPU_CONSTRUCT_VERIFY");
  const bool verify =
      verify_env != nullptr && std::string(verify_env) == std::string("1");
  const auto fallback = [verify](const char* reason) {
    if (verify) {
      Log::Warning("GPU construct: ineligible (%s), using the host path",
                   reason);
    } else {
      Log::Debug("GPU construct: ineligible (%s), using the host path",
                 reason);
    }
    return false;
  };
  if (num_groups_ <= 0 || nrow <= 0 || nrow != num_data_ || ncol <= 0) {
    return fallback("empty dataset or partial-matrix push");
  }
  if (has_raw_) {
    return fallback("linear-tree raw data requested");
  }
  if (num_groups_ > 65535) {
    return fallback("too many feature groups for the kernel grid");
  }
  const bool use_lut = DTypeUsesLut(dtype);
  // per-group storage layout
  std::vector<int> group_width(num_groups_, -1);
  for (int gid = 0; gid < num_groups_; ++gid) {
    if (feature_groups_[gid]->is_multi_val_) {
      return fallback("multi-val feature group");
    }
    uint8_t bit_type = 0;
    bool is_sparse = false;
    BinIterator* bin_iterator = nullptr;
    GetColWiseData(gid, -1, &bit_type, &is_sparse, &bin_iterator);
    if (bin_iterator != nullptr) {
      delete bin_iterator;
    }
    if (is_sparse) {
      return fallback("sparse bin storage");
    }
    group_width[gid] = GroupElemWidth(bit_type);
    if (group_width[gid] < 0) {
      return fallback("unsupported bin bit type");
    }
  }
  // columns of each group in ascending column order (the host push order)
  std::vector<std::vector<std::pair<int, int>>> group_col_sub(num_groups_);
  for (int fidx = 0; fidx < num_features_; ++fidx) {
    const int col = real_feature_idx_[fidx];
    if (col >= ncol) {
      continue;  // never pushed by the host path either
    }
    if (!use_lut &&
        FeatureBinMapper(fidx)->bin_type() != BinType::NumericalBin) {
      return fallback("categorical feature on the float/double path");
    }
    group_col_sub[feature2group_[fidx]].emplace_back(
        col, feature2subfeature_[fidx]);
  }
  for (auto& cols : group_col_sub) {
    std::sort(cols.begin(), cols.end());
  }
  const auto time_start = std::chrono::steady_clock::now();
  // flatten group->column lists and assign per-chunk output offsets;
  // wider groups first so every group segment stays element-aligned
  // (offsets are multiples of 8 pair-bytes ahead of any narrower group)
  std::vector<int> group_col_ptr(num_groups_ + 1, 0);
  std::vector<int> group_cols;
  for (int gid = 0; gid < num_groups_; ++gid) {
    for (const auto& cs : group_col_sub[gid]) {
      group_cols.push_back(cs.first);
    }
    group_col_ptr[gid + 1] = static_cast<int>(group_cols.size());
  }
  if (group_cols.empty()) {
    group_cols.push_back(0);  // keep the device upload well-defined
  }
  std::vector<int> group_order(num_groups_);
  for (int gid = 0; gid < num_groups_; ++gid) {
    group_order[gid] = gid;
  }
  std::stable_sort(group_order.begin(), group_order.end(),
                   [&group_width](int a, int b) {
                     return group_width[a] > group_width[b];
                   });
  std::vector<int64_t> group_pair_off(num_groups_, 0);
  int64_t bytes_per_pair_total = 0;
  for (int gid : group_order) {
    group_pair_off[gid] = bytes_per_pair_total;
    bytes_per_pair_total += GroupBytesPerPair(group_width[gid]);
  }
  std::vector<uint8_t> group_width_u8(num_groups_);
  for (int gid = 0; gid < num_groups_; ++gid) {
    group_width_u8[gid] = static_cast<uint8_t>(group_width[gid]);
  }
  // encoding tables
  std::vector<uint32_t> lut;
  std::vector<int64_t> col_lut_off;
  std::vector<double> bounds;
  std::vector<CUDADenseBinnerColMeta> col_meta;
  const int64_t lut_size = DTypeLutSize(dtype);
  std::vector<std::pair<int, int>> used_cols;  // (column, feature index)
  for (int fidx = 0; fidx < num_features_; ++fidx) {
    if (real_feature_idx_[fidx] < ncol) {
      used_cols.emplace_back(real_feature_idx_[fidx], fidx);
    }
  }
  if (use_lut) {
    col_lut_off.resize(ncol, 0);
    lut.resize(static_cast<size_t>(used_cols.size()) * lut_size,
               Bin::kSkipBin);
    for (size_t i = 0; i < used_cols.size(); ++i) {
      col_lut_off[used_cols[i].first] = static_cast<int64_t>(i) * lut_size;
    }
    const int num_used_cols = static_cast<int>(used_cols.size());
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (int i = 0; i < num_used_cols; ++i) {
      const int fidx = used_cols[i].second;
      const BinMapper* mapper = FeatureBinMapper(fidx);
      FeatureGroup* fg = feature_groups_[feature2group_[fidx]].get();
      const int sub = feature2subfeature_[fidx];
      uint32_t* col_lut = lut.data() + static_cast<int64_t>(i) * lut_size;
      for (int64_t v = 0; v < lut_size; ++v) {
        col_lut[v] = fg->EncodeBinForPush(
            sub, mapper->ValueToBin(LutIndexToValue(dtype, v)));
      }
    }
  } else {
    col_meta.resize(ncol);
    for (const auto& cf : used_cols) {
      const int fidx = cf.second;
      const BinMapper* mapper = FeatureBinMapper(fidx);
      CUDADenseBinnerColMeta meta;
      meta.bounds_offset = static_cast<int64_t>(bounds.size());
      meta.num_bin = mapper->num_bin();
      meta.most_freq_bin = mapper->GetMostFreqBin();
      meta.sub_offset = feature_groups_[feature2group_[fidx]]
                            ->bin_offsets_[feature2subfeature_[fidx]];
      meta.missing_is_nan = mapper->missing_type() == MissingType::NaN;
      const std::vector<double>& ub = mapper->bin_upper_bound();
      bounds.insert(bounds.end(), ub.begin(), ub.end());
      col_meta[cf.first] = meta;
    }
    if (bounds.empty()) {
      bounds.push_back(0.0);  // keep the device upload well-defined
    }
  }
  // chunk sizing and device memory budget
  const int64_t elem_size = DTypeSize(dtype);
  const int64_t row_bytes = static_cast<int64_t>(ncol) * elem_size;
  const int64_t table_bytes =
      static_cast<int64_t>(lut.size() * sizeof(uint32_t)) +
      static_cast<int64_t>(col_lut_off.size() * sizeof(int64_t)) +
      static_cast<int64_t>(bounds.size() * sizeof(double)) +
      static_cast<int64_t>(col_meta.size() * sizeof(CUDADenseBinnerColMeta)) +
      static_cast<int64_t>((group_cols.size() + group_col_ptr.size() +
                            group_pair_off.size()) * sizeof(int64_t)) +
      static_cast<int64_t>(group_width_u8.size());
  if (gpu_device_id_ >= 0) {
    SetCUDADevice(gpu_device_id_, __FILE__, __LINE__);
  }
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_bytes, &total_bytes));
  const int64_t budget =
      static_cast<int64_t>(free_bytes) - table_bytes - (64LL << 20);
  data_size_t chunk_rows = static_cast<data_size_t>(std::min<int64_t>(
      std::max<int64_t>((512LL << 20) / std::max<int64_t>(row_bytes, 1), 2),
      static_cast<int64_t>(nrow) + 1));
  chunk_rows = (chunk_rows + 1) / 2 * 2;  // keep row pairs within one chunk
  const auto chunk_footprint = [&](data_size_t rows) {
    const int64_t pairs = (static_cast<int64_t>(rows) + 1) / 2;
    const int64_t in_bytes = data_is_device ? 0 : rows * row_bytes;
    return 2 * (in_bytes + pairs * bytes_per_pair_total);
  };
  while (chunk_rows > 2 && chunk_footprint(chunk_rows) > budget) {
    chunk_rows = (chunk_rows / 2 + 1) / 2 * 2;
  }
  if (chunk_footprint(chunk_rows) > budget) {
    return fallback("insufficient device memory");
  }
  // upload tables
  int* d_group_col_ptr = nullptr;
  int* d_group_cols = nullptr;
  uint8_t* d_group_width = nullptr;
  int64_t* d_group_pair_off = nullptr;
  uint32_t* d_lut = nullptr;
  int64_t* d_col_lut_off = nullptr;
  double* d_bounds = nullptr;
  CUDADenseBinnerColMeta* d_col_meta = nullptr;
  InitCUDAMemoryFromHostMemory<int>(&d_group_col_ptr, group_col_ptr.data(),
                                    group_col_ptr.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemory<int>(&d_group_cols, group_cols.data(),
                                    group_cols.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemory<uint8_t>(&d_group_width, group_width_u8.data(),
                                        group_width_u8.size(), __FILE__,
                                        __LINE__);
  InitCUDAMemoryFromHostMemory<int64_t>(&d_group_pair_off,
                                        group_pair_off.data(),
                                        group_pair_off.size(), __FILE__,
                                        __LINE__);
  if (use_lut) {
    InitCUDAMemoryFromHostMemory<uint32_t>(&d_lut, lut.data(), lut.size(),
                                           __FILE__, __LINE__);
    InitCUDAMemoryFromHostMemory<int64_t>(&d_col_lut_off, col_lut_off.data(),
                                          col_lut_off.size(), __FILE__,
                                          __LINE__);
  } else {
    InitCUDAMemoryFromHostMemory<double>(&d_bounds, bounds.data(),
                                         bounds.size(), __FILE__, __LINE__);
    InitCUDAMemoryFromHostMemory<CUDADenseBinnerColMeta>(
        &d_col_meta, col_meta.data(), col_meta.size(), __FILE__, __LINE__);
  }
  CUDADenseBinnerTables tables;
  tables.group_col_ptr = d_group_col_ptr;
  tables.group_cols = d_group_cols;
  tables.group_width = d_group_width;
  tables.group_pair_off = d_group_pair_off;
  tables.lut = d_lut;
  tables.col_lut_off = d_col_lut_off;
  tables.bounds = d_bounds;
  tables.col_meta = d_col_meta;
  // staging ring: H2D + kernel on one stream, bulk D2H on a second stream so
  // uploads and downloads overlap; the host scatters chunk k-2 while the GPU
  // works on chunk k-1
  const int64_t chunk_in_bytes = static_cast<int64_t>(chunk_rows) * row_bytes;
  const int64_t chunk_pairs_max = (static_cast<int64_t>(chunk_rows) + 1) / 2;
  const int64_t chunk_out_bytes = chunk_pairs_max * bytes_per_pair_total;
  uint8_t* h_in[2] = {nullptr, nullptr};
  uint8_t* h_out[2] = {nullptr, nullptr};
  uint8_t* d_in[2] = {nullptr, nullptr};
  uint8_t* d_out[2] = {nullptr, nullptr};
  cudaStream_t stream;
  cudaStream_t d2h_stream;
  cudaEvent_t kernel_done[2];
  cudaEvent_t d2h_done[2];
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&stream));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&d2h_stream));
  for (int b = 0; b < 2; ++b) {
    if (!data_is_device) {
      CUDASUCCESS_OR_FATAL(cudaHostAlloc(
          reinterpret_cast<void**>(&h_in[b]),
          static_cast<size_t>(chunk_in_bytes), cudaHostAllocDefault));
      AllocateCUDAMemory<uint8_t>(&d_in[b],
                                  static_cast<size_t>(chunk_in_bytes),
                                  __FILE__, __LINE__);
    }
    CUDASUCCESS_OR_FATAL(cudaHostAlloc(
        reinterpret_cast<void**>(&h_out[b]),
        static_cast<size_t>(chunk_out_bytes), cudaHostAllocDefault));
    AllocateCUDAMemory<uint8_t>(&d_out[b],
                                static_cast<size_t>(chunk_out_bytes),
                                __FILE__, __LINE__);
    CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(
        &kernel_done[b], cudaEventDisableTiming));
    CUDASUCCESS_OR_FATAL(cudaEventCreateWithFlags(
        &d2h_done[b], cudaEventDisableTiming | cudaEventBlockingSync));
  }
  // scatter destinations: the live bin storage, or capture buffers in
  // verify mode (the host path then also runs and FinishLoad compares)
  std::vector<uint8_t*> group_dst(num_groups_, nullptr);
  if (verify) {
    gpu_bin_verify_data_.assign(num_groups_, std::vector<uint8_t>());
  }
  for (int gid = 0; gid < num_groups_; ++gid) {
    const int64_t total = GroupTotalBytes(group_width[gid], nrow);
    if (verify) {
      gpu_bin_verify_data_[gid].resize(total, 0);
      group_dst[gid] = gpu_bin_verify_data_[gid].data();
    } else {
      group_dst[gid] =
          static_cast<uint8_t*>(feature_groups_[gid]->bin_data_->get_data());
    }
  }
  const int num_threads = OMP_NUM_THREADS();
  const uint8_t* in_base = static_cast<const uint8_t*>(data);
  const int dtype_code = DTypeCode(dtype);
  ChunkDesc pending[2];
  const auto scatter = [&](const ChunkDesc& desc, const uint8_t* src_buf) {
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int gid = 0; gid < num_groups_; ++gid) {
      const int width = group_width[gid];
      const uint8_t* src = src_buf + group_pair_off[gid] * desc.pairs;
      uint8_t* dst;
      int64_t len;
      if (width == 0) {
        dst = group_dst[gid] + desc.start_row / 2;  // start_row is even
        len = desc.pairs;
      } else {
        dst = group_dst[gid] + static_cast<int64_t>(width) * desc.start_row;
        len = static_cast<int64_t>(width) * desc.rows;
      }
      StreamMemcpy(dst, src, static_cast<size_t>(len));
    }
  };
  const data_size_t num_chunks = static_cast<data_size_t>(
      (static_cast<int64_t>(nrow) + chunk_rows - 1) / chunk_rows);
  for (data_size_t k = 0; k < num_chunks; ++k) {
    const int b = static_cast<int>(k & 1);
    if (k >= 2) {
      // chunk k-2 fully downloaded implies h_in[b]'s H2D finished too, so
      // both staging buffers of this ring slot are reusable after the sync
      CUDASUCCESS_OR_FATAL(cudaEventSynchronize(d2h_done[b]));
      scatter(pending[b], h_out[b]);
      pending[b].valid = false;
    }
    ChunkDesc desc;
    desc.start_row = k * chunk_rows;
    desc.rows = std::min(chunk_rows, nrow - desc.start_row);
    desc.pairs = (desc.rows + 1) / 2;
    const void* kernel_in;
    int64_t in_stride;
    if (data_is_device) {
      const int64_t elem_off = is_row_major
          ? static_cast<int64_t>(desc.start_row) * ncol
          : static_cast<int64_t>(desc.start_row);
      kernel_in = in_base + elem_off * elem_size;
      in_stride = is_row_major ? ncol : nrow;
    } else {
      if (is_row_major) {
        // one contiguous block; split the copy across threads for bandwidth
        const uint8_t* src = in_base + desc.start_row * row_bytes;
        const int64_t bytes = static_cast<int64_t>(desc.rows) * row_bytes;
        const int64_t slice = (bytes + num_threads - 1) / num_threads;
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < num_threads; ++t) {
          const int64_t lo = t * slice;
          const int64_t hi = std::min(bytes, lo + slice);
          if (lo < hi) {
            StreamMemcpy(h_in[b] + lo, src + lo, static_cast<size_t>(hi - lo));
          }
        }
      } else {
        const int64_t seg = static_cast<int64_t>(desc.rows) * elem_size;
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int c = 0; c < ncol; ++c) {
          StreamMemcpy(h_in[b] + c * seg,
                       in_base + (static_cast<int64_t>(c) * nrow +
                                  desc.start_row) * elem_size,
                       static_cast<size_t>(seg));
        }
      }
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(
          d_in[b], h_in[b],
          static_cast<size_t>(static_cast<int64_t>(desc.rows) * row_bytes),
          cudaMemcpyHostToDevice, stream));
      kernel_in = d_in[b];
      in_stride = is_row_major ? ncol : desc.rows;
    }
    LaunchCUDADenseBinChunkKernel(kernel_in, dtype_code, is_row_major,
                                  in_stride, desc.rows, desc.pairs,
                                  num_groups_, tables, d_out[b], stream);
    CUDASUCCESS_OR_FATAL(cudaEventRecord(kernel_done[b], stream));
    CUDASUCCESS_OR_FATAL(cudaStreamWaitEvent(d2h_stream, kernel_done[b], 0));
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(
        h_out[b], d_out[b],
        static_cast<size_t>(bytes_per_pair_total * desc.pairs),
        cudaMemcpyDeviceToHost, d2h_stream));
    CUDASUCCESS_OR_FATAL(cudaEventRecord(d2h_done[b], d2h_stream));
    pending[b] = desc;
    pending[b].valid = true;
  }
  for (data_size_t k = std::max<data_size_t>(num_chunks - 2, 0);
       k < num_chunks; ++k) {
    const int b = static_cast<int>(k & 1);
    if (pending[b].valid) {
      CUDASUCCESS_OR_FATAL(cudaEventSynchronize(d2h_done[b]));
      scatter(pending[b], h_out[b]);
      pending[b].valid = false;
    }
  }
  CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(stream));
  CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(d2h_stream));
  for (int b = 0; b < 2; ++b) {
    if (h_in[b] != nullptr) {
      CUDASUCCESS_OR_FATAL(cudaFreeHost(h_in[b]));
    }
    CUDASUCCESS_OR_FATAL(cudaFreeHost(h_out[b]));
    if (d_in[b] != nullptr) {
      DeallocateCUDAMemory<uint8_t>(&d_in[b], __FILE__, __LINE__);
    }
    DeallocateCUDAMemory<uint8_t>(&d_out[b], __FILE__, __LINE__);
    CUDASUCCESS_OR_FATAL(cudaEventDestroy(kernel_done[b]));
    CUDASUCCESS_OR_FATAL(cudaEventDestroy(d2h_done[b]));
  }
  CUDASUCCESS_OR_FATAL(cudaStreamDestroy(stream));
  CUDASUCCESS_OR_FATAL(cudaStreamDestroy(d2h_stream));
  DeallocateCUDAMemory<int>(&d_group_col_ptr, __FILE__, __LINE__);
  DeallocateCUDAMemory<int>(&d_group_cols, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint8_t>(&d_group_width, __FILE__, __LINE__);
  DeallocateCUDAMemory<int64_t>(&d_group_pair_off, __FILE__, __LINE__);
  if (d_lut != nullptr) {
    DeallocateCUDAMemory<uint32_t>(&d_lut, __FILE__, __LINE__);
  }
  if (d_col_lut_off != nullptr) {
    DeallocateCUDAMemory<int64_t>(&d_col_lut_off, __FILE__, __LINE__);
  }
  if (d_bounds != nullptr) {
    DeallocateCUDAMemory<double>(&d_bounds, __FILE__, __LINE__);
  }
  if (d_col_meta != nullptr) {
    DeallocateCUDAMemory<CUDADenseBinnerColMeta>(&d_col_meta, __FILE__,
                                                 __LINE__);
  }
  const auto time_end = std::chrono::steady_clock::now();
  Log::Debug("GPU construct: binned %d x %d dense matrix on device in %.3fs "
             "(disable with EXABOOST_GPU_CONSTRUCT=0)",
             nrow, ncol,
             std::chrono::duration<double>(time_end - time_start).count());
  if (verify) {
    // let the host path run too; FinishLoad compares against the capture
    return false;
  }
  for (int gid = 0; gid < num_groups_; ++gid) {
    feature_groups_[gid]->bin_data_->SetLoadedFromRawData();
  }
  return true;
}

}  // namespace LightGBM

#endif  // USE_CUDA
