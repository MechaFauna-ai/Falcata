/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2022-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef FALCATA_INCLUDE_FALCATA_CUDA_CUDA_SPLIT_INFO_HPP_
#define FALCATA_INCLUDE_FALCATA_CUDA_CUDA_SPLIT_INFO_HPP_

#ifdef USE_CUDA

#include <Falcata/meta.h>

namespace Falcata {

class CUDASplitInfo {
 public:
  bool is_valid;
  int leaf_index;
  double gain;
  int inner_feature_index;
  uint32_t threshold;
  bool default_left;

  double left_sum_gradients;
  double left_sum_hessians;
  int64_t left_sum_of_gradients_hessians;
  data_size_t left_count;
  double left_gain;
  double left_value;

  double right_sum_gradients;
  double right_sum_hessians;
  int64_t right_sum_of_gradients_hessians;
  data_size_t right_count;
  double right_gain;
  double right_value;

  int num_cat_threshold = 0;
  uint32_t* cat_threshold = nullptr;
  int* cat_threshold_real = nullptr;

  __host__ __device__ CUDASplitInfo() {
    num_cat_threshold = 0;
    cat_threshold = nullptr;
    cat_threshold_real = nullptr;
  }

  __host__ __device__ ~CUDASplitInfo() {
    // cudaFree is host-only; device instances live in device memory owned elsewhere
    // and are never destroyed through this destructor.
#ifndef __CUDA_ARCH__
    if (num_cat_threshold > 0) {
      if (cat_threshold != nullptr) {
        CUDASUCCESS_OR_FATAL(cudaFree(cat_threshold));
      }
      if (cat_threshold_real != nullptr) {
        CUDASUCCESS_OR_FATAL(cudaFree(cat_threshold_real));
      }
    }
#endif
  }

  // Copy construction must NOT shallow-copy the threshold pointers: the
  // implicit memberwise copy would duplicate an owning pointer under two
  // destructors (std::vector<CUDASplitInfo> reallocation would double-free).
  // Delegate to operator=, whose pointer discipline is explicit.
  __host__ __device__ CUDASplitInfo(const CUDASplitInfo& other) : CUDASplitInfo() {
    *this = other;
  }

  __host__ __device__ CUDASplitInfo& operator=(const CUDASplitInfo& other) {
    is_valid = other.is_valid;
    leaf_index = other.leaf_index;
    gain = other.gain;
    inner_feature_index = other.inner_feature_index;
    threshold = other.threshold;
    default_left = other.default_left;

    left_sum_gradients = other.left_sum_gradients;
    left_sum_hessians = other.left_sum_hessians;
    left_sum_of_gradients_hessians = other.left_sum_of_gradients_hessians;
    left_count = other.left_count;
    left_gain = other.left_gain;
    left_value = other.left_value;

    right_sum_gradients = other.right_sum_gradients;
    right_sum_hessians = other.right_sum_hessians;
    right_sum_of_gradients_hessians = other.right_sum_of_gradients_hessians;
    right_count = other.right_count;
    right_gain = other.right_gain;
    right_value = other.right_value;

    // Threshold storage discipline: device instances use slab slots
    // pre-assigned by AllocateCatVectorsKernel; host instances only ever see
    // scrubbed copies (ReadPrefetchedLeafBestSplits). Assignment therefore
    // NEVER allocates -- the old `new[]` here paired device-heap allocation
    // with the destructor's cudaFree, an allocator mismatch that could never
    // be freed correctly on either side.
    // The count is METADATA and survives every assignment except one: a
    // destination WITH a slab receiving a scrubbed source would pair a
    // positive count with stale slab words, so only that case records zero.
    // (Scrubbed host copies legitimately carry count > 0 with null pointers --
    // the selective flow's SelectiveApplied records depend on the count
    // surviving vector push_back / struct assignment.)
    num_cat_threshold = other.num_cat_threshold;
    if (num_cat_threshold > 0 && cat_threshold != nullptr) {
      if (other.cat_threshold != nullptr) {
        for (int i = 0; i < num_cat_threshold; ++i) {
          cat_threshold[i] = other.cat_threshold[i];
        }
        if (cat_threshold_real != nullptr && other.cat_threshold_real != nullptr) {
          for (int i = 0; i < num_cat_threshold; ++i) {
            cat_threshold_real[i] = other.cat_threshold_real[i];
          }
        }
      } else {
        num_cat_threshold = 0;
      }
    }
    return *this;
  }
};

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_INCLUDE_FALCATA_CUDA_CUDA_SPLIT_INFO_HPP_
