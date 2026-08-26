/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef FALCATA_IO_CUDA_CUDA_DENSE_BINNER_HPP_
#define FALCATA_IO_CUDA_CUDA_DENSE_BINNER_HPP_

#ifdef USE_CUDA

#include <Falcata/meta.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace Falcata {

class Bin;
class Dataset;

// per-column metadata for the on-device float/double binning
// (replicates BinMapper::ValueToBin + FeatureGroup::EncodeBinForPush)
struct CUDADenseBinnerColMeta {
  int64_t bounds_offset;   // into the flattened bin_upper_bound_ array
  int num_bin;
  uint32_t most_freq_bin;
  uint32_t sub_offset;     // FeatureGroup bin offset of the subfeature (0 pre-decrement)
  uint8_t missing_is_nan;  // missing_type == MissingType::NaN
};

// device tables shared by every chunk of one matrix
struct CUDADenseBinnerTables {
  const int* group_col_ptr;      // [num_groups + 1]
  const int* group_cols;         // ascending column ids per group
  const uint8_t* group_width;    // bytes per element; 0 means 4-bit packed
  const int64_t* group_pair_off;  // bytes-per-row-pair prefix offset per group
  const uint32_t* lut;           // LUT mode: encoded bins (kSkipBin sentinel)
  const int64_t* col_lut_off;    // LUT mode: per-column offset into lut
  const double* bounds;          // float mode: flattened bin upper bounds
  const CUDADenseBinnerColMeta* col_meta;  // float mode
};

// bins one chunk of a dense matrix into per-group packed column bins.
// dtype_code: 0=float32 1=float64 2=int8 3=int16 4=uint8 5=uint16(+fp16 bits)
// in_stride: elements per row (row-major) or per column (col-major)
// parity: 1 when the chunk starts at an odd dataset row (shifts row pairing
// so nibble-packed groups stay aligned to the dataset-global byte layout)
void LaunchCUDADenseBinChunkKernel(const void* in, int dtype_code,
                                   bool is_row_major, int64_t in_stride,
                                   data_size_t rows, data_size_t pairs,
                                   int parity, int num_groups,
                                   const CUDADenseBinnerTables& tables,
                                   uint8_t* out, cudaStream_t stream);

/*!
 * \brief One device dense-binning session over a Dataset whose bin mappers
 *        are already constructed. Created by Dataset::GPUDenseBinnerBegin
 *        (nullptr when the dataset is ineligible and the host push loops
 *        must run instead); the encode tables and staging ring live for the
 *        whole session so BinChunk can be called once per input chunk with
 *        the chunk's dataset row offset. Every BinChunk call is synchronous:
 *        when it returns the input buffer is no longer referenced and may be
 *        freed or reused. Chunks must be pushed in ascending row order and
 *        may start at any (odd or even) dataset row.
 *
 *        Output placement is chunk-invariant: any split of the matrix into
 *        row chunks scatters byte-identical values into the dense bin
 *        storage, which is what the one-shot Dataset::GPUBinDenseRows path
 *        (a single BinChunk over all rows) relies on.
 */
class CUDADenseBinnerCtx {
 public:
  ~CUDADenseBinnerCtx();

  /*!
   * \brief Bin dataset rows [start_row, start_row + nrow) from one dense
   *        matrix (host or device resident). Internally sub-chunks to the
   *        session's device budget and overlaps H2D, kernel, D2H and the
   *        host scatter; fully drained before returning.
   */
  void BinChunk(const void* data, data_size_t nrow, bool is_row_major,
                bool data_is_device, data_size_t start_row);

  /*! \brief FALCATA_GPU_CONSTRUCT_VERIFY=1: results go to capture buffers
   *         and the host push path must run as well (FinishLoad compares). */
  bool verify() const { return verify_; }

  /*!
   * \brief End the session: log, and outside verify mode mark the bin
   *        storage as device-loaded. Call after the last BinChunk, before
   *        Dataset::FinishLoad.
   */
  void Finalize();

 private:
  friend class Dataset;  // constructed only by Dataset::GPUDenseBinnerBegin
  CUDADenseBinnerCtx() = default;

  struct ChunkDesc {
    data_size_t start_row = 0;  // dataset-global first row
    data_size_t rows = 0;
    data_size_t pairs = 0;
    int parity = 0;
    bool valid = false;
  };

  void EnsureInputStaging();
  void Scatter(const ChunkDesc& desc, const uint8_t* src_buf);

  // dataset-shape state captured at Begin
  int num_groups_ = 0;
  data_size_t num_data_ = 0;
  int ncol_ = 0;
  int gpu_device_id_ = -1;
  int num_threads_ = 1;
  bool verify_ = false;
  int dtype_code_ = 0;
  int64_t elem_size_ = 0;
  int64_t row_bytes_ = 0;
  std::vector<int> group_width_;
  std::vector<int64_t> group_pair_off_;
  int64_t bytes_per_pair_total_ = 0;
  std::vector<uint8_t*> group_dst_;   // live bin storage or verify capture
  std::vector<Bin*> group_bins_;      // for SetLoadedFromRawData in Finalize
  // device tables
  int* d_group_col_ptr_ = nullptr;
  int* d_group_cols_ = nullptr;
  uint8_t* d_group_width_ = nullptr;
  int64_t* d_group_pair_off_ = nullptr;
  uint32_t* d_lut_ = nullptr;
  int64_t* d_col_lut_off_ = nullptr;
  double* d_bounds_ = nullptr;
  CUDADenseBinnerColMeta* d_col_meta_ = nullptr;
  CUDADenseBinnerTables tables_;
  // staging ring
  data_size_t chunk_rows_ = 0;
  int64_t chunk_in_bytes_ = 0;
  int64_t chunk_pairs_max_ = 0;
  int64_t chunk_out_bytes_ = 0;
  uint8_t* h_in_[2] = {nullptr, nullptr};
  uint8_t* h_out_[2] = {nullptr, nullptr};
  uint8_t* d_in_[2] = {nullptr, nullptr};
  uint8_t* d_out_[2] = {nullptr, nullptr};
  cudaStream_t stream_ = nullptr;
  cudaStream_t d2h_stream_ = nullptr;
  cudaEvent_t kernel_done_[2] = {nullptr, nullptr};
  cudaEvent_t d2h_done_[2] = {nullptr, nullptr};
  // session stats
  data_size_t rows_binned_ = 0;
  double seconds_ = 0.0;
};

// gathers the given rows of a device-resident dense matrix into a row-major
// host block (bitwise copies); used to sample device inputs for bin finding
void CUDAGatherSampleRowsToHost(const void* device_data, int64_t elem_size,
                                bool is_row_major, data_size_t nrow, int ncol,
                                const int32_t* sample_row_ids, int num_samples,
                                void* host_out);

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_IO_CUDA_CUDA_DENSE_BINNER_HPP_
