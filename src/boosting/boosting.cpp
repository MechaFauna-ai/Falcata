/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <Falcata/boosting.h>
#include <Falcata/utils/file_io.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "dart.hpp"
#include "gbdt.h"
#include "rf.hpp"

#ifdef USE_NCCL
#include "cuda/nccl_gbdt.hpp"
#endif  // USE_NCCL

namespace Falcata {

namespace {

// Read a whole model file through the virtual reader (local or remote).
std::vector<char> ReadWholeFile(const char* filename) {
  auto reader = VirtualFileReader::Make(filename);
  if (!reader->Init()) {
    Log::Fatal("Could not open model file %s", filename);
  }
  std::vector<char> content;
  std::vector<char> chunk(1 << 20);
  size_t got = 0;
  while ((got = reader->Read(chunk.data(), chunk.size())) > 0) {
    content.insert(content.end(), chunk.data(), chunk.data() + got);
  }
  return content;
}

// FALB files start with the 4-byte magic "FALB" (falb.h); text models with "tree".
bool IsFalbFile(const char* filename) {
  auto reader = VirtualFileReader::Make(filename);
  char magic[4] = {0, 0, 0, 0};
  return reader->Init() && reader->Read(magic, sizeof(magic)) == sizeof(magic) &&
         std::memcmp(magic, "FALB", sizeof(magic)) == 0;
}

}  // namespace

std::string GetBoostingTypeFromModelFile(const char* filename) {
  TextReader<size_t> model_reader(filename, true);
  std::string type = model_reader.first_line();
  return type;
}

bool Boosting::LoadFileToBoosting(Boosting* boosting, const char* filename) {
  auto start_time = std::chrono::steady_clock::now();
  if (boosting != nullptr) {
    TextReader<size_t> model_reader(filename, true);
    size_t buffer_len = 0;
    auto buffer = model_reader.ReadContent(&buffer_len);
    if (!boosting->LoadModelFromString(buffer.data(), buffer_len)) {
      return false;
    }
  }
  std::chrono::duration<double, std::milli> delta = (std::chrono::steady_clock::now() - start_time);
  Log::Debug("Time for loading model: %f seconds", 1e-3 * delta.count());
  return true;
}

Boosting* Boosting::CreateBoosting(const std::string& type, const char* filename,
  const std::string&
  #ifdef USE_CUDA
  device_type
  #endif  // USE_CUDA
  , const int
  #ifdef USE_CUDA
  num_gpu
  #endif  // USE_CUDA
  ) {
  if (filename == nullptr || filename[0] == '\0') {
    if (type == std::string("gbdt")) {
      #ifdef USE_CUDA
      if (device_type == std::string("cuda") && num_gpu > 1) {
      #ifdef USE_NCCL
        if (NcclShim::Available()) {
          return new NCCLGBDT<GBDT>();
        }
        Log::Warning(
            "num_gpu > 1 needs the NCCL library, and libnccl.so.2 was not "
            "found -- install it with: pip install nvidia-nccl-cu12. "
            "Falling back to a single GPU.");
        return new GBDT();
      #else
        Log::Warning("num_gpu > 1 requires NCCL, which was not compiled in (USE_NCCL=OFF). Falling back to a single GPU.");
        return new GBDT();
      #endif  // USE_NCCL
      } else {
      #endif  // USE_CUDA
        return new GBDT();
      #ifdef USE_CUDA
      }
      #endif  // USE_CUDA
    } else if (type == std::string("dart")) {
      return new DART();
    } else if (type == std::string("goss")) {
      return new GBDT();
    } else if (type == std::string("rf")) {
      return new RF();
    } else {
      return nullptr;
    }
  } else {
    std::unique_ptr<Boosting> ret;
    // A FALB model file (save_model(..., format="falb")) loads like a text
    // one, so every path that takes a model FILE -- init_model, the CLI's
    // input_model, FLC_BoosterCreateFromModelfile -- accepts both formats.
    if (IsFalbFile(filename)) {
      #ifdef USE_CUDA
      ret.reset(CreateBoosting(type, nullptr, device_type, num_gpu));
      #else
      ret.reset(CreateBoosting(type, nullptr, "", 1));
      #endif  // USE_CUDA
      if (ret == nullptr) {
        Log::Fatal("Unknown boosting type %s", type.c_str());
      }
      const std::vector<char> content = ReadWholeFile(filename);
      if (!ret->LoadModelFromBinary(content.data(), content.size())) {
        Log::Fatal("Could not load the FALB model in %s", filename);
      }
      return ret.release();
    }
    if (GetBoostingTypeFromModelFile(filename) == std::string("tree")) {
      #ifdef USE_CUDA
      ret.reset(CreateBoosting(type, nullptr, device_type, num_gpu));
      #else
      ret.reset(CreateBoosting(type, nullptr, "", 1));
      #endif  // USE_CUDA
      if (ret == nullptr) {
        Log::Fatal("Unknown boosting type %s", type.c_str());
      }
      LoadFileToBoosting(ret.get(), filename);
    } else {
      Log::Fatal("Unknown model format or submodel type in model file %s", filename);
    }
    return ret.release();
  }
}

}  // namespace Falcata
