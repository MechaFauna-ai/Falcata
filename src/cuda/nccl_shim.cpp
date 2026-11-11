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
#include <string>

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
  // why loading failed (empty when ok): the dynamic loader's message, or the symbols libnccl lacked
  std::string error;
};

const Fns& Load() {
  static Fns fns;
  static std::once_flag loaded;
  std::call_once(loaded, []() {
    // the soname first (searches the caller's RUNPATH, which the wheel points
    // at the nvidia-nccl-cu12 package, then the system paths), then the
    // unversioned name for dev setups
    // (the handle stays open for the life of the process: the function pointers point into it)
    void* h = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) {
      // keep the soname's error: it says whether the library is absent or failed to load (missing dependency,
      // wrong architecture, ...)
      const char* soname_error = dlerror();
      fns.error = soname_error != nullptr ? soname_error : "libnccl.so.2 could not be loaded";
      h = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (h == nullptr) {
      return;
    }
    fns.error.clear();
    const auto symbol = [h](const char* name) -> void* {
      void* p = dlsym(h, name);
      if (p == nullptr) {
        fns.error += fns.error.empty() ? "libnccl does not export " : ", ";
        fns.error += name;
      }
      return p;
    };
    fns.all_reduce = reinterpret_cast<decltype(fns.all_reduce)>(symbol("ncclAllReduce"));
    fns.comm_init_rank = reinterpret_cast<decltype(fns.comm_init_rank)>(symbol("ncclCommInitRank"));
    fns.get_unique_id = reinterpret_cast<decltype(fns.get_unique_id)>(symbol("ncclGetUniqueId"));
    fns.group_start = reinterpret_cast<decltype(fns.group_start)>(symbol("ncclGroupStart"));
    fns.group_end = reinterpret_cast<decltype(fns.group_end)>(symbol("ncclGroupEnd"));
    fns.get_error_string = reinterpret_cast<decltype(fns.get_error_string)>(symbol("ncclGetErrorString"));
    fns.ok = fns.all_reduce && fns.comm_init_rank && fns.get_unique_id &&
             fns.group_start && fns.group_end && fns.get_error_string;
  });
  return fns;
}

void FatalIfMissing(const Fns& fns) {
  if (!fns.ok) {
    Log::Fatal(
        "multi-GPU training needs the NCCL library, and libnccl.so.2 could not "
        "be used (%s). Install it with: pip install nvidia-nccl-cu12 (or pip "
        "install 'falcata[multigpu]').",
        fns.error.c_str());
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
    // fns is loaded once, so the message is fixed for the process
    static const std::string message = "NCCL is not loaded (" + fns.error + ")";
    return message.c_str();
  }
  return fns.get_error_string(result);
}

}  // namespace NcclShim
}  // namespace Falcata

#endif  // defined(USE_NCCL) && !defined(USE_ROCM)
