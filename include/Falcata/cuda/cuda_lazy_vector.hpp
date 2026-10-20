/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root.
 */
#ifndef FALCATA_INCLUDE_FALCATA_CUDA_CUDA_LAZY_VECTOR_HPP_
#define FALCATA_INCLUDE_FALCATA_CUDA_CUDA_LAZY_VECTOR_HPP_

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_utils.hu>

#include <mutex>
#include <vector>

namespace Falcata {

// An owned host snapshot for compatibility metadata with no training consumer.
// Get() uploads once, synchronously; its pointer remains valid until Set() or
// destruction. Callers must finish using a returned pointer before either.
// The mutex serializes getters and snapshot replacement, not external kernels.
template <typename T>
class CUDALazyVector {
 public:
  CUDALazyVector() = default;
  CUDALazyVector(const CUDALazyVector&) = delete;
  CUDALazyVector& operator=(const CUDALazyVector&) = delete;
  CUDALazyVector(CUDALazyVector&&) = delete;
  CUDALazyVector& operator=(CUDALazyVector&&) = delete;

  ~CUDALazyVector() { ClearDevice(); }

  void Set(const T* values, size_t size, int device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Do not form a range from a null pointer for an empty snapshot.
    if (size == 0) {
      host_.clear();
    } else {
      host_.assign(values, values + size);
    }
    ClearDevice();
    device_id_ = device_id >= 0 ? device_id : 0;
  }

  const T* Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!host_.empty() && device_.Size() == 0) {
      DeviceScope scope(device_id_);
      device_.InitFromHostVector(host_);
      // Pageable H2D cudaMemcpy may return after staging. Complete the upload
      // before publishing a pointer that can be used on any caller stream.
      SynchronizeCUDADevice(__FILE__, __LINE__);
    }
    return device_.RawData();
  }

 private:
  class DeviceScope {
   public:
    explicit DeviceScope(int device_id) : device_id_(device_id) {
      CUDASUCCESS_OR_FATAL(cudaGetDevice(&previous_));
      if (previous_ != device_id_) {
        CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id_));
      }
    }
    ~DeviceScope() {
      if (previous_ != device_id_) {
        CUDASUCCESS_OR_FATAL(cudaSetDevice(previous_));
      }
    }

   private:
    int previous_;
    int device_id_;
  };

  void ClearDevice() {
    if (device_.Size() != 0) {
      DeviceScope scope(device_id_);
      device_.Clear();
    }
  }

  std::vector<T> host_;
  mutable CUDAVector<T> device_;
  mutable std::mutex mutex_;
  int device_id_ = 0;
};

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_INCLUDE_FALCATA_CUDA_CUDA_LAZY_VECTOR_HPP_
