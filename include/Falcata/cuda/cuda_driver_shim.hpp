/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_CUDA_CUDA_DRIVER_SHIM_HPP_
#define FALCATA_CUDA_CUDA_DRIVER_SHIM_HPP_

#if defined(USE_CUDA) && !defined(USE_ROCM)

// The CUDA driver library (libcuda.so.1) is never linked: libcuda ships with
// the NVIDIA driver, not the toolkit, so it is the one CUDA library a wheel
// cannot vendor, and a wheel that linked it could not even be imported on a
// machine without the driver. The CUDA runtime (libcudart, linked statically)
// loads the driver lazily; this shim only checks, by dlopen on first use,
// that the driver is there, so a driverless host gets an actionable message
// instead of a failure deep inside the runtime.

namespace Falcata {
namespace CudaDriverShim {

// The actionable explanation for every CUDA failure on a driverless host:
// nullptr when the driver is loadable (libcuda.so.1 found and exporting cuInit;
// the dlopen happens once), otherwise a message naming the missing library and
// the CPU fallback.
const char* MissingDriverMessage();

}  // namespace CudaDriverShim
}  // namespace Falcata

#endif  // defined(USE_CUDA) && !defined(USE_ROCM)

#endif  // FALCATA_CUDA_CUDA_DRIVER_SHIM_HPP_
