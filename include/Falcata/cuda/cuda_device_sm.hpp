/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_INCLUDE_FALCATA_CUDA_CUDA_DEVICE_SM_HPP_
#define FALCATA_INCLUDE_FALCATA_CUDA_CUDA_DEVICE_SM_HPP_

#ifdef USE_CUDA

#include <Falcata/cuda/cuda_rocm_interop.h>
#include <Falcata/cuda/cuda_utils.hu>
#include <Falcata/falcata_plan.h>

#include <cuda_runtime.h>

#include <atomic>

namespace Falcata {

/*! \brief The CUDA tree learner's launch-shape decisions that read the device's multiprocessor count. */
enum FalcataSMSite : unsigned {
  kSMSiteSaturationFloor = 1u << 0,    // batched quantized construct's saturation floor (tree learner Init)
  kSMSiteFusedRootFill = 1u << 1,      // fused root fill: tiles per block from the resident block count
  kSMSiteLevelTargetBlocks = 1u << 2,  // device-level construct target blocks
  kSMSiteCountTiles = 1u << 3,         // count-histogram tiling (forest_count_hist)
  kSMSiteFindSplitCap = 1u << 4,       // split finder: 4 warps vs 1 warp per wide task
  kSMSitePairJointShape = 1u << 5,     // pair-joint code5 word construct: device slots
  kSMSiteLiveGridBudget = 1u << 6,     // device_tree_live_grid: blocks per SM times the SM count
};

/*! \brief The name a site's FALCATA_DEBUG=smcount log line carries. */
inline const char* FalcataSMSiteName(const unsigned site) {
  switch (site) {
    case kSMSiteSaturationFloor: return "saturation_floor";
    case kSMSiteFusedRootFill: return "fused_root_fill";
    case kSMSiteLevelTargetBlocks: return "level_target_blocks";
    case kSMSiteCountTiles: return "count_tiles";
    case kSMSiteFindSplitCap: return "find_split_cap";
    case kSMSitePairJointShape: return "pair_joint_shape";
    case kSMSiteLiveGridBudget: return "live_grid_budget";
    default: return "unknown";
  }
}

/*! \brief FALCATA_DEBUG=smcount=N applied to `site` (whose real count is `real`): N when the token is set and its
 * smsites mask includes the site, else `real`. The first override at each site is logged (Info), which is how a
 * test proves that a training run reached the site. */
inline int FalcataApplySMOverride(const int device, const unsigned site, const int real) {
  const FalcataDebugOptions& dbg = FalcataDebug();
  if (dbg.smcount <= 0 || (dbg.smsites & site) == 0) {
    return real;
  }
  static std::atomic<unsigned> logged{0u};
  if ((logged.fetch_or(site) & site) == 0u) {
    Log::Info("FALCATA_DEBUG smcount: site %s sees %d multiprocessors (device %d has %d)", FalcataSMSiteName(site),
              dbg.smcount, device, real);
  }
  return dbg.smcount;
}

/*!
 * \brief The multiprocessor count of `device` as `site` sees it (FalcataApplySMOverride). A failed query is fatal,
 * as cudaDeviceGetAttribute under CUDASUCCESS_OR_FATAL was at every site that uses this.
 */
inline int FalcataDeviceSMCount(const int device, const unsigned site) {
  int sms = 0;
  CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  return FalcataApplySMOverride(device, site, sms);
}

/*!
 * \brief The same for the sites that handle a failed query themselves (unchanged from before this helper): false
 * when the query fails or reports no multiprocessor, and *out is untouched.
 */
inline bool FalcataTryDeviceSMCount(const int device, const unsigned site, int* out) {
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sms <= 0) {
    return false;
  }
  *out = FalcataApplySMOverride(device, site, sms);
  return true;
}

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_INCLUDE_FALCATA_CUDA_CUDA_DEVICE_SM_HPP_
