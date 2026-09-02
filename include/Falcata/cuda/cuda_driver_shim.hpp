/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_CUDA_CUDA_DRIVER_SHIM_HPP_
#define FALCATA_CUDA_CUDA_DRIVER_SHIM_HPP_

#if defined(USE_CUDA) && !defined(USE_ROCM)

// The CUDA driver library (libcuda.so.1) is loaded at runtime, never linked:
// types come from the toolkit's cuda.h, the five driver entry points below
// (the NVRTC construct-kernel JIT's whole driver-API surface) resolve via
// dlopen on first use. libcuda ships with the NVIDIA driver, not the toolkit,
// so it is the one CUDA library a wheel cannot vendor; a wheel that linked it
// could not even be imported on a machine without the driver. The CUDA runtime
// (libcudart, linked statically) already loads the driver the same lazy way.
#include <cuda.h>

namespace Falcata {
namespace CudaDriverShim {

// true iff libcuda.so.1 was found and every entry point resolved;
// safe to call repeatedly (the dlopen happens once)
bool Available();

// The actionable explanation for every CUDA failure on a driverless host:
// nullptr when the driver is loadable (so the failure is something else),
// otherwise a message naming the missing library and the CPU fallback.
const char* MissingDriverMessage();

CUresult Init(unsigned int flags);
CUresult ModuleLoadData(CUmodule* module, const void* image);
CUresult ModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
CUresult ModuleUnload(CUmodule hmod);
CUresult LaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                      unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                      unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
                      void** kernelParams, void** extra);

}  // namespace CudaDriverShim
}  // namespace Falcata

// Route every call site through the shim without touching them. The real
// symbol names still resolve inside cuda_driver_shim.cpp, which is compiled
// with FALCATA_CUDA_DRIVER_SHIM_IMPL defined and therefore skips these macros.
#ifndef FALCATA_CUDA_DRIVER_SHIM_IMPL
#define cuInit Falcata::CudaDriverShim::Init
#define cuModuleLoadData Falcata::CudaDriverShim::ModuleLoadData
#define cuModuleGetFunction Falcata::CudaDriverShim::ModuleGetFunction
#define cuModuleUnload Falcata::CudaDriverShim::ModuleUnload
#define cuLaunchKernel Falcata::CudaDriverShim::LaunchKernel
#endif  // FALCATA_CUDA_DRIVER_SHIM_IMPL

#endif  // defined(USE_CUDA) && !defined(USE_ROCM)

#endif  // FALCATA_CUDA_CUDA_DRIVER_SHIM_HPP_
