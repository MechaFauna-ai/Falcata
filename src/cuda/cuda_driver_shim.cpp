/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#if defined(USE_CUDA) && !defined(USE_ROCM)

#include <Falcata/cuda/cuda_driver_shim.hpp>

#include <dlfcn.h>

#include <mutex>

namespace Falcata {
namespace CudaDriverShim {

namespace {

bool Load() {
  static bool ok = false;
  static std::once_flag loaded;
  std::call_once(loaded, []() {
    // the soname the driver installs (the CUDA runtime loads the same name),
    // then the unversioned name for dev setups
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) {
      h = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    }
    ok = h != nullptr && dlsym(h, "cuInit") != nullptr;
  });
  return ok;
}

}  // namespace

const char* MissingDriverMessage() {
  return Load() ? nullptr
                : "device_type=cuda needs the NVIDIA driver, and its library libcuda.so.1 "
                  "was not found on this machine. Install the NVIDIA driver to train on the "
                  "GPU; CPU training works without it (device_type=cpu).";
}

}  // namespace CudaDriverShim
}  // namespace Falcata

#endif  // defined(USE_CUDA) && !defined(USE_ROCM)
