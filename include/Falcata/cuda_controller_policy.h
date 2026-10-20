/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_CUDA_CONTROLLER_POLICY_H_
#define FALCATA_CUDA_CONTROLLER_POLICY_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Falcata {

enum class CUDAControllerRoute { kAuto, kHost, kPrefix, kWhole, kLegacy };

inline const char* CUDAControllerRouteName(CUDAControllerRoute route) {
  switch (route) {
    case CUDAControllerRoute::kAuto: return "auto";
    case CUDAControllerRoute::kHost: return "host";
    case CUDAControllerRoute::kPrefix: return "prefix";
    case CUDAControllerRoute::kWhole: return "whole";
    case CUDAControllerRoute::kLegacy: return "legacy";
  }
  return "host";
}

struct CUDAControllerRequest {
  bool device_levels = true;
  bool device_tree = false;
  bool device_levels_explicit = false;
  bool device_tree_explicit = false;
};

// Host metadata collected at Init / reset, never by probing training or scanning
// rows. Hardware properties are cached at Init. Eligibility remains the
// learner's responsibility, including per-tree layout and kernel restrictions.
struct CUDAControllerFacts {
  int64_t num_rows = 0;
  int num_features = 0;
  int sampled_features = 0;
  int64_t num_bins = 0;
  int max_feature_bins = 0;
  int missing_feature_count = 0;
  bool metadata_known = false;
  int num_leaves = 0;
  int max_depth = 0;
  int min_data_in_leaf = 0;
  double feature_fraction = 1.0;
  double bagging_fraction = 1.0;
  int bagging_freq = 0;
  bool quantized = false;
  bool graph_cache_enabled = true;
  bool performance_modifiers_supported = false;
  bool scope_supported = false;
  bool native_prefix_eligible = false;
  bool native_whole_eligible = false;
  int device_major = 0;
  int device_minor = 0;
  int device_sm_count = 0;
  int device_memory_bus_bits = 0;
};

struct CUDAControllerDecision {
  CUDAControllerRoute requested = CUDAControllerRoute::kAuto;
  CUDAControllerRoute resolved = CUDAControllerRoute::kHost;
  const char* reason = "unresolved";
};

// The verbatim generic export includes this policy header itself. Declare its
// entry point after the shared types so either include order is valid.
inline CUDAControllerRoute LearnedCUDAControllerCandidate(const CUDAControllerFacts& facts);

}  // namespace Falcata

// Supplied only by the pinned generic exporter; no handwritten substitute.
// Its defensive int32 guards are redundant for these native field types.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-constant-out-of-range-compare"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif
#include <Falcata/cuda_controller_learned.h>  // NOLINT(build/include_order)
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace Falcata {

inline CUDAControllerDecision ResolveCUDAController(const CUDAControllerRequest& request,
                                                   const CUDAControllerFacts& facts) {
  if (request.device_levels_explicit || request.device_tree_explicit) {
    // Preserve the original independent boolean pair. In particular, explicit
    // tree:on owns its growth even with levels:off; tree:off alone retains the
    // historical prefix default. Token order across different keys is immaterial.
    const CUDAControllerRoute route = request.device_tree ? CUDAControllerRoute::kWhole :
        request.device_levels ? CUDAControllerRoute::kPrefix : CUDAControllerRoute::kHost;
    return {route, route, "explicit_controller_overrides"};
  }
  using Route = CUDAControllerRoute;
  // Resource identity of the measured desktop GPU, without device-name matching.
  if (facts.device_major != 12 || facts.device_minor != 0 || facts.device_sm_count != 170 ||
      facts.device_memory_bus_bits != 512) {
    return {Route::kAuto, Route::kHost, "unqualified_hardware"};
  }
  if (!facts.metadata_known || facts.num_rows <= 0 || facts.num_features <= 0 ||
      facts.sampled_features <= 0 || facts.sampled_features > facts.num_features ||
      facts.max_feature_bins <= 0 || facts.num_bins <= 0) {
    return {Route::kAuto, Route::kHost, "unavailable_metadata"};
  }
  if (!facts.quantized) {
    return {Route::kAuto, Route::kLegacy, "legacy_scope"};
  }
  if (!facts.scope_supported) return {Route::kAuto, Route::kLegacy, "legacy_scope"};
  if (!facts.graph_cache_enabled) return {Route::kAuto, Route::kHost, "v2_graph_disabled"};
  if (!facts.performance_modifiers_supported) return {Route::kAuto, Route::kHost, "v2_unqualified_modifiers"};
  // Exactly one raw prediction. Initial native feasibility can decline that
  // winner, but supplies no ranking or permission to substitute another route.
  const Route candidate = LearnedCUDAControllerCandidate(facts);
  switch (candidate) {
    case Route::kHost:
      return {Route::kAuto, Route::kHost, "learned_host"};
    case Route::kPrefix:
      return facts.native_prefix_eligible ?
          CUDAControllerDecision{Route::kAuto, Route::kPrefix, "learned_prefix"} :
          CUDAControllerDecision{Route::kAuto, Route::kHost, "learned_prefix_native_unavailable"};
    case Route::kWhole:
      return facts.native_whole_eligible ?
          CUDAControllerDecision{Route::kAuto, Route::kWhole, "learned_whole"} :
          CUDAControllerDecision{Route::kAuto, Route::kHost, "learned_whole_native_unavailable"};
    default:
      return {Route::kAuto, Route::kHost, "learned_invalid_candidate"};
  }
}

}  // namespace Falcata

#endif  // FALCATA_CUDA_CONTROLLER_POLICY_H_
