/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_CUDA_NCCL_SHIM_HPP_
#define FALCATA_CUDA_NCCL_SHIM_HPP_

#if defined(USE_NCCL) && !defined(USE_ROCM)

// NCCL is loaded at runtime, never linked: types come from the vendored
// header (external_libs/nccl), the six entry points below resolve via dlopen
// on first use. A wheel therefore works on machines without NCCL -- multi-GPU
// simply requires libnccl.so.2 to be discoverable (the nvidia-nccl-cu12 pip
// package, reachable through the wheel's rpath, or a system install) and
// fails with that exact instruction when it is not.
#include <nccl.h>

namespace Falcata {
namespace NcclShim {

// true iff libnccl.so.2 was found and every entry point resolved;
// safe to call repeatedly (the dlopen happens once)
bool Available();

ncclResult_t AllReduce(const void* sendbuff, void* recvbuff, size_t count,
                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                       cudaStream_t stream);
ncclResult_t CommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t GetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t GroupStart();
ncclResult_t GroupEnd();
const char* GetErrorString(ncclResult_t result);

}  // namespace NcclShim
}  // namespace Falcata

// Route every call site through the shim without touching them. The real
// symbol names still resolve inside nccl_shim.cpp, which is compiled with
// FALCATA_NCCL_SHIM_IMPL defined and therefore skips these macros.
#ifndef FALCATA_NCCL_SHIM_IMPL
#define ncclAllReduce Falcata::NcclShim::AllReduce
#define ncclCommInitRank Falcata::NcclShim::CommInitRank
#define ncclGetUniqueId Falcata::NcclShim::GetUniqueId
#define ncclGroupStart Falcata::NcclShim::GroupStart
#define ncclGroupEnd Falcata::NcclShim::GroupEnd
#define ncclGetErrorString Falcata::NcclShim::GetErrorString
#endif  // FALCATA_NCCL_SHIM_IMPL

#endif  // defined(USE_NCCL) && !defined(USE_ROCM)

#endif  // FALCATA_CUDA_NCCL_SHIM_HPP_
