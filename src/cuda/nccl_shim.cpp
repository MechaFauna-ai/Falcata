/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#if defined(USE_NCCL) && !defined(USE_ROCM)

#define FALCATA_NCCL_SHIM_IMPL
#include <Falcata/cuda/nccl_shim.hpp>

#include <Falcata/utils/log.h>

#include <dlfcn.h>

#include <mutex>

namespace Falcata {
namespace NcclShim {

namespace {

struct Fns {
  decltype(&::ncclAllReduce) all_reduce = nullptr;
  decltype(&::ncclCommInitRank) comm_init_rank = nullptr;
  decltype(&::ncclGetUniqueId) get_unique_id = nullptr;
  decltype(&::ncclGroupStart) group_start = nullptr;
  decltype(&::ncclGroupEnd) group_end = nullptr;
  decltype(&::ncclGetErrorString) get_error_string = nullptr;
  bool ok = false;
};

const Fns& Load() {
  static Fns fns;
  static std::once_flag loaded;
  std::call_once(loaded, []() {
    // the soname first (searches the caller's RUNPATH, which the wheel points
    // at the nvidia-nccl-cu12 package, then the system paths), then the
    // unversioned name for dev setups
    void* h = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) {
      h = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (h == nullptr) {
      return;
    }
    fns.all_reduce = reinterpret_cast<decltype(fns.all_reduce)>(dlsym(h, "ncclAllReduce"));
    fns.comm_init_rank = reinterpret_cast<decltype(fns.comm_init_rank)>(dlsym(h, "ncclCommInitRank"));
    fns.get_unique_id = reinterpret_cast<decltype(fns.get_unique_id)>(dlsym(h, "ncclGetUniqueId"));
    fns.group_start = reinterpret_cast<decltype(fns.group_start)>(dlsym(h, "ncclGroupStart"));
    fns.group_end = reinterpret_cast<decltype(fns.group_end)>(dlsym(h, "ncclGroupEnd"));
    fns.get_error_string = reinterpret_cast<decltype(fns.get_error_string)>(dlsym(h, "ncclGetErrorString"));
    fns.ok = fns.all_reduce && fns.comm_init_rank && fns.get_unique_id &&
             fns.group_start && fns.group_end && fns.get_error_string;
  });
  return fns;
}

void FatalIfMissing(const Fns& fns) {
  if (!fns.ok) {
    Log::Fatal(
        "multi-GPU training needs the NCCL library, and libnccl.so.2 was not "
        "found. Install it with: pip install nvidia-nccl-cu12 (or pip install "
        "'falcata[multigpu]').");
  }
}

}  // namespace

bool Available() { return Load().ok; }

ncclResult_t AllReduce(const void* sendbuff, void* recvbuff, size_t count,
                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                       cudaStream_t stream) {
  const Fns& fns = Load();
  FatalIfMissing(fns);
  return fns.all_reduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t CommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
  const Fns& fns = Load();
  FatalIfMissing(fns);
  return fns.comm_init_rank(comm, nranks, commId, rank);
}

ncclResult_t GetUniqueId(ncclUniqueId* uniqueId) {
  const Fns& fns = Load();
  FatalIfMissing(fns);
  return fns.get_unique_id(uniqueId);
}

ncclResult_t GroupStart() {
  const Fns& fns = Load();
  FatalIfMissing(fns);
  return fns.group_start();
}

ncclResult_t GroupEnd() {
  const Fns& fns = Load();
  FatalIfMissing(fns);
  return fns.group_end();
}

const char* GetErrorString(ncclResult_t result) {
  const Fns& fns = Load();
  if (!fns.ok) {
    return "NCCL is not loaded (libnccl.so.2 not found)";
  }
  return fns.get_error_string(result);
}

}  // namespace NcclShim
}  // namespace Falcata

#endif  // defined(USE_NCCL) && !defined(USE_ROCM)
