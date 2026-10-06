/*!
 * Copyright (c) 2020-2021 IBM Corporation, Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 * Modifications Copyright(C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */
#ifndef FALCATA_INCLUDE_FALCATA_CUDA_VECTOR_CUDAHOST_H_
#define FALCATA_INCLUDE_FALCATA_CUDA_VECTOR_CUDAHOST_H_

#include <Falcata/utils/common.h>

#ifdef USE_CUDA
#ifndef USE_ROCM
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // USE_ROCM
#include <Falcata/cuda/cuda_utils.hu>
#endif  // USE_CUDA
#include <stdio.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif

#include <cstddef>

enum FLC_Device {
  lgbm_device_cpu,
  lgbm_device_gpu,
  lgbm_device_cuda
};

enum Use_Learner {
  use_cpu_learner,
  use_gpu_learner,
  use_cuda_learner
};

namespace Falcata {

class FLC_config_ {
 public:
  static int current_device;  // Default: lgbm_device_cpu
  static int current_learner;  // Default: use_cpu_learner
  // false while host bin storage is created without page-locking (CHAllocator
  // then allocates pageable memory); read by allocate, set by the dataset build
  static bool pin_host_allocs;
};


/*!
 * \brief The kernel's transparent huge page size (Linux: hpage_pmd_size, read
 * once; 2 MB on x86-64), or 0 where there is none. Alignment and madvise
 * granularity of the large pageable host bin buffers.
 */
inline size_t HostHugePageBytes() {
  static const size_t bytes = []() -> size_t {
    size_t value = 0;
#if defined(__linux__)
    FILE* f = fopen("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size", "r");
    if (f != nullptr) {
      size_t v = 0;
      if (fscanf(f, "%zu", &v) == 1 && v > 0 && (v & (v - 1)) == 0) {
        value = v;
      }
      fclose(f);
    }
#endif
    return value;
  }();
  return bytes;
}

template <class T>
struct CHAllocator {
  typedef T value_type;
  CHAllocator() {}
  template <class U> CHAllocator(const CHAllocator<U>& other);
  T* allocate(std::size_t n) {
    T* ptr;
    if (n == 0) return NULL;
    n = SIZE_ALIGNED(n);
    #ifdef USE_CUDA
      if (FLC_config_::current_device == lgbm_device_cuda && FLC_config_::pin_host_allocs) {
        cudaError_t ret = cudaHostAlloc(reinterpret_cast<void**>(&ptr), n*sizeof(T), cudaHostAllocPortable);
        if (ret != cudaSuccess) {
          Log::Warning("Defaulting to malloc in CHAllocator!!!");
          ptr = reinterpret_cast<T*>(_mm_malloc(n*sizeof(T), 16));
        }
      } else if (FLC_config_::current_device == lgbm_device_cuda && HostHugePageBytes() > 0 &&
                 n*sizeof(T) >= HostHugePageBytes()) {
        // pageable bins of a large matrix: huge-page aligned, and the kernel may back
        // them with huge pages (far fewer first-touch faults and TLB misses)
        ptr = reinterpret_cast<T*>(_mm_malloc(n*sizeof(T), HostHugePageBytes()));
        #if defined(__linux__)
          if (ptr != NULL) {
            madvise(ptr, n*sizeof(T), MADV_HUGEPAGE);  // a hint: its result does not matter
          }
        #endif
      } else {
        ptr = reinterpret_cast<T*>(_mm_malloc(n*sizeof(T), 16));
      }
    #else
      ptr = reinterpret_cast<T*>(_mm_malloc(n*sizeof(T), 16));
    #endif
    return ptr;
  }

  void deallocate(T* p, std::size_t n) {
    (void)n;  // UNUSED
    if (p == NULL) return;
    #ifdef USE_CUDA
      if (FLC_config_::current_device == lgbm_device_cuda) {
        // page-locked or pageable: allocate makes either (pin_host_allocs, or the
        // malloc fallback when cudaHostAlloc fails), so free by what the pointer is
        cudaPointerAttributes attributes;
        const cudaError_t ret = cudaPointerGetAttributes(&attributes, p);
        if (ret != cudaSuccess) {
          #ifdef USE_ROCM
            // HIP reports host memory it did not allocate or register as an error
            (void)cudaGetLastError();
            _mm_free(p);
            return;
          #else
            // CUDA (>= 11) reports pageable memory as unregistered, not as an error
            CUDASUCCESS_OR_FATAL(ret);
          #endif
        }
        if (attributes.type == cudaMemoryTypeHost) {
          CUDASUCCESS_OR_FATAL(cudaFreeHost(p));
        } else {
          _mm_free(p);
        }
      } else {
        _mm_free(p);
      }
    #else
      _mm_free(p);
    #endif
  }
};
template <class T, class U>
bool operator==(const CHAllocator<T>&, const CHAllocator<U>&);
template <class T, class U>
bool operator!=(const CHAllocator<T>&, const CHAllocator<U>&);

}  // namespace Falcata

#endif  // FALCATA_INCLUDE_FALCATA_CUDA_VECTOR_CUDAHOST_H_
