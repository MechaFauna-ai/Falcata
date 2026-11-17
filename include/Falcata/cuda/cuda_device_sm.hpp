/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_INCLUDE_FALCATA_CUDA_CUDA_DEVICE_SM_HPP_
#define FALCATA_INCLUDE_FALCATA_CUDA_CUDA_DEVICE_SM_HPP_

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_rocm_interop.h>
#include <Falcata/falcata_plan.h>

#include <cuda_runtime.h>

#include <atomic>

namespace Falcata {

/*! \brief The CUDA tree learner's launch-shape decisions that read the device's multiprocessor count. */
enum FalcataSMSite : unsigned {
  kSMSiteSaturationFloor = 1u << 0,    // batched quantized construct's saturation floor (tree learner Init)
  kSMSiteFusedRootFill = 1u << 1,      // fused root fill: tiles per block from the resident block count
  kSMSiteLevelTargetBlocks = 1u << 2,  // device-level construct target blocks
  kSMSiteCountTiles = 1u << 3,         // count-histogram tiling
  kSMSiteFindSplitCap = 1u << 4,       // split finder: 4 warps vs 1 warp per wide task
  kSMSitePairJointShape = 1u << 5,     // pair-joint code5 word construct: device slots
};

/*!
 * \brief The multiprocessor count of `device` as `site` sees it: the real one, or FALCATA_DEBUG=smcount=N when
 * that token is set and its smsites mask includes `site` (emulates another card of the same architecture).
 */
inline int FalcataDeviceSMCount(const int device, const unsigned site) {
  const FalcataDebugOptions& dbg = FalcataDebug();
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sms <= 0) {
    sms = 1;
  }
  if (dbg.smcount > 0 && (dbg.smsites & site) != 0) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
      Log::Info("FALCATA_DEBUG smcount: launch shapes see %d multiprocessors (device %d has %d)", dbg.smcount, device,
                sms);
    }
    return dbg.smcount;
  }
  return sms;
}

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_INCLUDE_FALCATA_CUDA_CUDA_DEVICE_SM_HPP_
