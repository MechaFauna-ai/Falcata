/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#if defined(USE_CUDA) && !defined(USE_ROCM)

#define FALCATA_CUDA_DRIVER_SHIM_IMPL
#include <Falcata/cuda/cuda_driver_shim.hpp>

#include <Falcata/utils/log.h>

#include <dlfcn.h>

#include <mutex>

namespace Falcata {
namespace CudaDriverShim {

namespace {

struct Fns {
  decltype(&::cuInit) init = nullptr;
  decltype(&::cuModuleLoadData) module_load_data = nullptr;
  decltype(&::cuModuleGetFunction) module_get_function = nullptr;
  decltype(&::cuModuleUnload) module_unload = nullptr;
  decltype(&::cuLaunchKernel) launch_kernel = nullptr;
  bool ok = false;
};

const Fns& Load() {
  static Fns fns;
  static std::once_flag loaded;
  std::call_once(loaded, []() {
    // the soname the driver installs (the CUDA runtime loads the same name),
    // then the unversioned name for dev setups
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) {
      h = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (h == nullptr) {
      return;
    }
    fns.init = reinterpret_cast<decltype(fns.init)>(dlsym(h, "cuInit"));
    fns.module_load_data = reinterpret_cast<decltype(fns.module_load_data)>(dlsym(h, "cuModuleLoadData"));
    fns.module_get_function =
        reinterpret_cast<decltype(fns.module_get_function)>(dlsym(h, "cuModuleGetFunction"));
    fns.module_unload = reinterpret_cast<decltype(fns.module_unload)>(dlsym(h, "cuModuleUnload"));
    fns.launch_kernel = reinterpret_cast<decltype(fns.launch_kernel)>(dlsym(h, "cuLaunchKernel"));
    fns.ok = fns.init && fns.module_load_data && fns.module_get_function &&
             fns.module_unload && fns.launch_kernel;
  });
  return fns;
}

const char* MissingMessage() {
  return "device_type=cuda needs the NVIDIA driver, and its library libcuda.so.1 "
         "was not found on this machine. Install the NVIDIA driver to train on the "
         "GPU; CPU training works without it (device_type=cpu).";
}

void FatalIfMissing(const Fns& fns) {
  if (!fns.ok) {
    Log::Fatal(MissingMessage());
  }
}

}  // namespace

bool Available() { return Load().ok; }

const char* MissingDriverMessage() {
  return Load().ok ? nullptr : MissingMessage();
}

CUresult Init(unsigned int flags) {
  const Fns& fns = Load();
  FatalIfMissing(fns);
  return fns.init(flags);
}

CUresult ModuleLoadData(CUmodule* module, const void* image) {
  const Fns& fns = Load();
  FatalIfMissing(fns);
  return fns.module_load_data(module, image);
}

CUresult ModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
  const Fns& fns = Load();
  FatalIfMissing(fns);
  return fns.module_get_function(hfunc, hmod, name);
}

CUresult ModuleUnload(CUmodule hmod) {
  const Fns& fns = Load();
  FatalIfMissing(fns);
  return fns.module_unload(hmod);
}

CUresult LaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                      unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                      unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                      void** kernelParams, void** extra) {
  const Fns& fns = Load();
  FatalIfMissing(fns);
  return fns.launch_kernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                           sharedMemBytes, hStream, kernelParams, extra);
}

}  // namespace CudaDriverShim
}  // namespace Falcata

#endif  // defined(USE_CUDA) && !defined(USE_ROCM)
