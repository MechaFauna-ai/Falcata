/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_INCLUDE_FALCATA_CUDA_CUDA_DEVICE_TREE_REASON_HPP_
#define FALCATA_INCLUDE_FALCATA_CUDA_CUDA_DEVICE_TREE_REASON_HPP_

#include <cstdint>

namespace Falcata {

/*!
 * \brief Why a CUDA tree did not grow as the complete device tree, and which controller grew it instead.
 *
 * Under FALCATA_DEBUG=diag a single-GPU CUDA learner reports every tree the complete device tree did not grow as
 *
 *   [hybrid-diag] device_tree: fallback (<reason>) route=<route>[ prefix=<reason>]
 *
 * once for its first such tree and again whenever the line would change (the reason, the route, or why a device
 * prefix did not run), so a run logs its route changes, not one line per tree. The comparison is with the learner's
 * last fallback line: complete trees in between do not make the same line print again. Every learner prints its own
 * lines (one per rank under num_gpu > 1, one per forest slot). Trees the complete device tree grew
 * keep their `device_tree: complete ...` line. <reason> is the first refusal in the order of this list that holds for
 * the tree (several can hold at once; the order puts the long-lived fallbacks first, so a later port changes the
 * fewest pins). `prefix=` names why a device_levels prefix did not run in a flow that would otherwise have taken one
 * (only printed for the host_two_sync and selective_two_sync routes).
 *
 * The tokens are a stable vocabulary: tests pin them (tests/python_package_test/test_cuda_route_pins.py parses this
 * list), so a token is renamed or removed only together with the pins that name it. Each maps to a row of the
 * refusal inventory in the full device controller program (the third column); "-" marks diagnostics outside it.
 */
#define FALCATA_DEVICE_TREE_REASONS(X)                                                             \
  X(kNone, "none", "-")                                   /* the complete device tree grew the tree */ \
  X(kDebugHook, "debug_hook", "25")                       /* FALCATA_DEBUG=debug, dump or bynode */    \
  X(kPlanOff, "plan_off", "26")         /* a cuda_plan key switched a device flow off (device_tree, ...) */ \
  X(kMultiGpu, "multi_gpu", "10")                         /* num_gpu > 1, data-parallel */             \
  X(kFeatureParallel, "feature_parallel", "11")           /* tree_learner=feature */                   \
  X(kVectorLeaves, "vector_leaves", "12")                 /* tree_mode=vector_leaf */                  \
  X(kForestSlots, "forest_slots", "9")                    /* rf groups of two or more trees */         \
  X(kNoSharedMemory, "no_shared_memory", "16")            /* nonquant fixed-point table > smem */      \
  X(kNonquantGt256Bins, "nonquant_gt_256_bins", "15")     /* global-memory split finder */             \
  X(kSparseLayout, "sparse_layout", "14")                 /* sparse row data */                        \
  X(kExtraTrees, "extra_trees", "13")                                                                  \
  X(kCatRandomSearch, "cat_random_search", "13")                                                       \
  X(kInteractionConstraints, "interaction_constraints", "8")                                           \
  X(kMonotone, "monotone", "17")                                                                       \
  X(kCegb, "cegb", "18")                                                                               \
  X(kForcedSplits, "forced_splits", "19")                                                              \
  X(kLinearBudget, "linear_budget", "20")                 /* linear trees, budget-limited */           \
  X(kNumLeavesLe2, "num_leaves_le_2", "21")                                                            \
  X(kNonquantSelective, "nonquant_selective", "5")        /* quant_mode=none, budget-limited */        \
  X(kNonquantOnesync, "nonquant_onesync", "4")            /* quant_mode=none outside the L1.5 bound */ \
  X(kNonquantL15, "nonquant_l15", "4")                    /* quant_mode=none inside the L1.5 bound */  \
  X(kLambdaL1, "lambda_l1", "13")                                                                      \
  X(kPathSmooth, "path_smooth", "13")                                                                  \
  X(kLargeBinLayout, "large_bin_layout", "14")            /* partitions on global-memory histograms */ \
  X(kCompact16Bit, "compact_16bit", "14")                 /* compact column view of non-8-bit rows */  \
  X(kCategorical, "categorical", "6")                                                                  \
  X(kBynodeHostMasks, "bynode_host_masks", "7")           /* in the bound, device masks unusable */    \
  X(kBynodeOutsideBound, "bynode_outside_bound", "7")     /* per-node masks outside the depth bound */ \
  X(kSelectiveUnbounded, "selective_unbounded", "3")      /* max_depth <= 0 */                         \
  X(kSelectiveBudget, "selective_budget", "3")            /* 2^max_depth > num_leaves + 1 */           \
  X(kDepthGt12, "depth_gt_12", "1")                       /* depth-limited, max_depth > 12 */          \
  X(kMemoryGate, "memory_gate", "2")                      /* scratch over half of free memory */       \
  X(kSetupFailed, "setup_failed", "-")                    /* SetupDeviceLevels refused */              \
  X(kPrefixGraphChurn, "prefix_graph_churn", "24")        /* prefix graphs off after cache misses */   \
  X(kEmptyTaskList, "empty_task_list", "23")              /* no searchable feature this tree */        \
  X(kRootTailParity, "root_tail_parity", "22")            /* the device root found no split */         \
  X(kUnclassified, "unclassified", "-")                   /* none of the above: a bug */

enum class DeviceTreeReason : uint8_t {
#define FALCATA_DEVICE_TREE_REASON_ENUM(name, token, row) name,
  FALCATA_DEVICE_TREE_REASONS(FALCATA_DEVICE_TREE_REASON_ENUM)
#undef FALCATA_DEVICE_TREE_REASON_ENUM
};

inline const char* DeviceTreeReasonToken(const DeviceTreeReason reason) {
  static const char* const kTokens[] = {
#define FALCATA_DEVICE_TREE_REASON_TOKEN(name, token, row) token,
    FALCATA_DEVICE_TREE_REASONS(FALCATA_DEVICE_TREE_REASON_TOKEN)
#undef FALCATA_DEVICE_TREE_REASON_TOKEN
  };
  return kTokens[static_cast<int>(reason)];
}

/*! \brief The controller that grew a tree (the `route=` of the reason line). */
#define FALCATA_GROWTH_ROUTES(X)                                                                        \
  X(kDeviceTree, "device_tree")                 /* the complete device tree */                          \
  X(kPrefix, "prefix")                          /* device_levels prefix, then the host two-sync loop */ \
  X(kHostTwoSync, "host_two_sync")              /* the host two-sync level loop */                      \
  X(kSelectivePrefix, "selective_prefix")       /* device prefix, then the host selective loop */       \
  X(kSelectiveTwoSync, "selective_two_sync")    /* the host selective two-sync loop */                  \
  X(kSelectiveOneSync, "selective_one_sync")    /* the host selective one-sync loop (non-quantized) */  \
  X(kL15Graph, "l15_graph")                     /* the non-quantized captured level graph */            \
  X(kOneSync, "one_sync")                       /* the non-quantized one-sync level loop */             \
  X(kClassic, "classic")                        /* the per-split loop */                                \
  X(kForestSlots, "forest_slots")               /* a forest slot beside other slot threads */

enum class GrowthRoute : uint8_t {
#define FALCATA_GROWTH_ROUTE_ENUM(name, token) name,
  FALCATA_GROWTH_ROUTES(FALCATA_GROWTH_ROUTE_ENUM)
#undef FALCATA_GROWTH_ROUTE_ENUM
};

inline const char* GrowthRouteToken(const GrowthRoute route) {
  static const char* const kTokens[] = {
#define FALCATA_GROWTH_ROUTE_TOKEN(name, token) token,
    FALCATA_GROWTH_ROUTES(FALCATA_GROWTH_ROUTE_TOKEN)
#undef FALCATA_GROWTH_ROUTE_TOKEN
  };
  return kTokens[static_cast<int>(route)];
}

}  // namespace Falcata

#endif  // FALCATA_INCLUDE_FALCATA_CUDA_CUDA_DEVICE_TREE_REASON_HPP_
