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

/*! \brief Field order of the vector-leaf (multi-target) payload slab attached
 *  to a CUDASplitInfo slot: per-target child gradient sums and leaf outputs,
 *  laid out [field][target] (field stride == num_vec_targets). The shared
 *  hessian and the counts live in the scalar fields. */
enum VecPayloadField {
  kVecLeftSumGradients = 0,
  kVecRightSumGradients = 1,
  kVecLeftValue = 2,
  kVecRightValue = 3,
  /*! \brief quantized training only: the child's INTEGER gradient sum for this
   *  target, the high half of the packed (grad32, hess32) accumulator the next
   *  level's discretized search needs as an exact parent total. Held as a
   *  double because every int32 is exactly representable, which keeps the whole
   *  payload one slab under one copy discipline. The shared hessian half comes
   *  from the child's primary leaf-splits struct. */
  kVecLeftGradInt = 4,
  kVecRightGradInt = 5,
  kNumVecPayloadFields = 6,
};

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

  /*! \brief vector-leaf payload: kNumVecPayloadFields * num_vec_targets
   *  doubles in a finder-owned slab slot (see VecPayloadField). Same pointer
   *  discipline as the categorical thresholds -- device slots are pre-assigned
   *  by the finder, host copies are scrubbed to nullptr, assignment never
   *  allocates -- except the slab is never freed through this struct. */
  int num_vec_targets = 0;
  double* vec_payload = nullptr;

  __host__ __device__ CUDASplitInfo() {
    num_cat_threshold = 0;
    cat_threshold = nullptr;
    cat_threshold_real = nullptr;
    num_vec_targets = 0;
    vec_payload = nullptr;
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
    // scrubbed copies (ReadPrefetchedLeafBestSplits). Assignment must NEVER
    // allocate: any allocation here would pair with the destructor's cudaFree
    // across the host/device boundary, an allocator mismatch.
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
    // vector-leaf payload: same discipline as the categorical thresholds
    // above -- deep-copy into a pre-assigned destination slab, never allocate,
    // and zero the count only when a slab-holding destination receives a
    // scrubbed source (positive count over stale slab words).
    num_vec_targets = other.num_vec_targets;
    if (num_vec_targets > 0 && vec_payload != nullptr) {
      if (other.vec_payload != nullptr) {
        for (int i = 0; i < kNumVecPayloadFields * num_vec_targets; ++i) {
          vec_payload[i] = other.vec_payload[i];
        }
      } else {
        num_vec_targets = 0;
      }
    }
    return *this;
  }
};

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_INCLUDE_FALCATA_CUDA_CUDA_SPLIT_INFO_HPP_
