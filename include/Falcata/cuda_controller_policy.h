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
  // Derivation-only v1 hypotheses, awaiting independent performance validation.
  // Resource identity of the measured desktop GPU, without device-name matching.
  if (facts.device_major != 12 || facts.device_minor != 0 || facts.device_sm_count != 170 ||
      facts.device_memory_bus_bits != 512) {
    return {Route::kAuto, Route::kHost, "unqualified_hardware"};
  }
  if (!facts.metadata_known || facts.num_rows <= 0 || facts.num_features <= 0 ||
      facts.sampled_features <= 0 || facts.sampled_features > facts.num_features || facts.max_feature_bins <= 0) {
    return {Route::kAuto, Route::kHost, "unavailable_metadata"};
  }
  if (!facts.quantized) {
    return {Route::kAuto, Route::kLegacy, "legacy_scope"};
  }
  if (!facts.scope_supported) return {Route::kAuto, Route::kLegacy, "legacy_scope"};
  if (!facts.graph_cache_enabled) return {Route::kAuto, Route::kHost, "v1_graph_disabled"};
  if (!facts.performance_modifiers_supported) return {Route::kAuto, Route::kHost, "v1_unqualified_modifiers"};
  if (facts.num_rows < 100000) return {Route::kAuto, Route::kHost, "v1_min_rows"};
  if (facts.max_feature_bins > 256) return {Route::kAuto, Route::kHost, "v1_bin_envelope"};
  if (facts.missing_feature_count > 0) return {Route::kAuto, Route::kHost, "v1_missing_mode"};

  const double active_bag = facts.bagging_freq > 0 ? facts.bagging_fraction : 1.0;
  const double leaf_capacity = std::floor(static_cast<double>(facts.num_rows) * active_bag /
                                          std::max(1, facts.min_data_in_leaf));
  // Include the native final single-leaf budget bind. The n=2 derivation
  // replay motivates this boundary; it does not qualify performance.
  if (facts.native_whole_eligible && facts.max_depth >= 1 && facts.max_depth <= 10) {
    const int64_t depth_leaves = int64_t{1} << facts.max_depth;
    if (static_cast<int64_t>(facts.num_leaves) + 1 >= depth_leaves &&
        facts.sampled_features <= 128 && leaf_capacity >= depth_leaves) {
      return {Route::kAuto, Route::kWhole, "v1_whole"};
    }
  }
  if (facts.native_prefix_eligible) {
    if (facts.sampled_features <= 64 && facts.feature_fraction == 1.0 && active_bag == 1.0) {
      return {Route::kAuto, Route::kPrefix, "v1_prefix_dense"};
    }
    if (facts.sampled_features <= 128 && facts.max_feature_bins <= 4 &&
        facts.feature_fraction >= 0.5 && active_bag >= 0.5) {
      return {Route::kAuto, Route::kPrefix, "v1_prefix_smallbin"};
    }
  }
  return {Route::kAuto, Route::kHost, "v1_no_region"};
}

}  // namespace Falcata

#endif  // FALCATA_CUDA_CONTROLLER_POLICY_H_
