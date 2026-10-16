/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_TREELEARNER_CUDA_CUDA_PDL_LAUNCH_HPP_
#define FALCATA_TREELEARNER_CUDA_CUDA_PDL_LAUNCH_HPP_

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/falcata_plan.h>

#include <atomic>
#include <utility>

// programmatic dependent launch: CUDA 11.8 runtime (cudaLaunchKernelEx, the launch attribute) on sm_90 and later
#if !defined(__HIP_PLATFORM_AMD__) && defined(CUDART_VERSION) && CUDART_VERSION >= 11080
#define FALCATA_PDL_SUPPORTED 1
#else
#define FALCATA_PDL_SUPPORTED 0
#endif

namespace Falcata {

// cuda_plan key pdl_level_chain: a level-chain consumer kernel (it reads what the kernel launched just before it on
// the same stream wrote) is launched with programmatic stream serialization, so the device sets its grid up while
// the producer drains. The consumer's first instruction waits until every prerequisite grid has completed and its
// memory is visible (griddepcontrol.wait), so it reads and writes exactly what an ordinary launch would. On an
// ordinary launch (key off, a device below compute capability 9.0, or a non-CUDA build) the wait returns at once.
__device__ __forceinline__ void PdlWaitForProducer() {
#if FALCATA_PDL_SUPPORTED && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// the producer side: once every block of the running grid has passed this point (or exited), its programmatic
// dependent (if one was launched) may be scheduled; that dependent still waits in PdlWaitForProducer for this grid's
// completion and memory, so where the producer calls this changes when the dependent's blocks become resident, never
// what they read. A no-op without a programmatic dependent, below compute capability 9.0 and outside CUDA.
__device__ __forceinline__ void PdlLaunchDependents() {
#if FALCATA_PDL_SUPPORTED && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

#if FALCATA_PDL_SUPPORTED
// an empty kernel of this translation unit, built for the same targets as its level-chain consumers (every CUDA
// source of the library is compiled with the same architectures, so the driver loads the same image variant for all)
static __global__ void PdlProbeKernel() {}
#endif

// whether the plan asks for programmatic dependent launches and the current device can run them: compute capability
// >= 9.0, and the image the driver loaded for this device was compiled for a virtual architecture of at least
// compute 9.0 (cudaFuncGetAttributes ptxVersion of PdlProbeKernel), so the consumers' PdlWaitForProducer holds
// griddepcontrol.wait: a build for older targets (e.g. compute_80 PTX JIT-compiled for a newer device) compiles the
// wait out, and its launches must stay ordinary. Read once per device (per translation unit); no lookup per launch.
static inline bool PdlLevelChainEnabled() {
#if !FALCATA_PDL_SUPPORTED
  return false;
#else
  if (!FalcataPlan::Get().pdl_level_chain) {
    return false;
  }
  constexpr int kMaxDevices = 64;
  static std::atomic<int> capable[kMaxDevices] = {};  // 0 unknown, 1 yes, 2 no
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kMaxDevices) {
    return false;
  }
  int state = capable[device].load(std::memory_order_relaxed);
  if (state == 0) {
    int major = 0;
    cudaFuncAttributes attr;
    const bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
                    major >= 9 &&
                    cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(PdlProbeKernel)) == cudaSuccess &&
                    attr.ptxVersion >= 90;
    state = ok ? 1 : 2;
    capable[device].store(state, std::memory_order_relaxed);
  }
  return state == 1;
#endif
}

// launch `kernel` on `stream`, as a programmatic dependent of the stream's previous kernel when `pdl` is set (from
// PdlLevelChainEnabled; the kernel must call PdlWaitForProducer before touching memory); launch errors surface either
// way
template <typename... KernelArgs, typename... Args>
inline void LaunchLevelChainKernel(const bool pdl, void (*kernel)(KernelArgs...), const dim3 grid, const dim3 block,
                                   const size_t shared_bytes, cudaStream_t stream, Args&&... args) {
#if FALCATA_PDL_SUPPORTED
  if (pdl) {
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = shared_bytes;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attribute;
    config.numAttrs = 1;
    CUDASUCCESS_OR_FATAL(cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...));
    return;
  }
#endif
  kernel<<<grid, block, shared_bytes, stream>>>(std::forward<Args>(args)...);
}

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_TREELEARNER_CUDA_CUDA_PDL_LAUNCH_HPP_
