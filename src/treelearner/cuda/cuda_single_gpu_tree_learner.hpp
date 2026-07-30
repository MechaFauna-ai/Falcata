/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef FALCATA_SRC_TREELEARNER_CUDA_CUDA_SINGLE_GPU_TREE_LEARNER_HPP_
#define FALCATA_SRC_TREELEARNER_CUDA_CUDA_SINGLE_GPU_TREE_LEARNER_HPP_

#include <memory>
#include <vector>

#ifdef USE_CUDA

#include "cuda_leaf_splits.hpp"
#include "cuda_histogram_constructor.hpp"
#include "cuda_data_partition.hpp"
#include "cuda_best_split_finder.hpp"

#include "cuda_gradient_discretizer.hpp"
#include "../serial_tree_learner.h"

namespace Falcata {

#define CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE (1024)

class CUDASingleGPUTreeLearner: public SerialTreeLearner, public NCCLInfo {
 public:
  explicit CUDASingleGPUTreeLearner(const Config* config, const bool boosting_on_cuda);

  ~CUDASingleGPUTreeLearner();

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void ResetTrainingData(const Dataset* train_data,
                         bool is_constant_hessian) override;

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override;

  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override;

  void AddPredictionToScore(const Tree* tree, double* out_score) const override;

  void RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                       data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt, const double* train_score) const override;

  void ResetConfig(const Config* config) override;

  Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const override;

  Tree* FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                          const score_t* gradients, const score_t* hessians) const override;

  void ResetBoostingOnGPU(const bool boosting_on_gpu) override;

  void SetNCCLInfo(
    ncclComm_t nccl_communicator,
    int nccl_gpu_rank,
    int local_gpu_rank,
    int gpu_device_id,
    data_size_t global_num_data) override;

 protected:
  bool IsCUDATreeLearner() const override { return true; }

  void BeforeTrain() override;

  // ---- linear tree support ----
  // Upload raw (un-binned) numerical feature columns to the GPU (feature-major).
  // Only done when config_->linear_tree is set; mirrors the host Dataset::raw_data_.
  void InitLinearTreeCUDA(const Dataset* train_data);
  // Fit a linear model (const + sum coeff*rawfeat) at each leaf of the just-built
  // tree, matching the CPU LinearTreeLearner. Heavy XtHX/Xtg accumulation runs on
  // the GPU; the small per-leaf solve runs on the host (Eigen) for CPU parity.
  void CalculateLinearCUDA(CUDATree* tree, const score_t* gradients, const score_t* hessians, bool is_first_tree);
  // Build the device per-leaf linear model arrays from the tree's CURRENT (post-shrinkage)
  // coefficients and upload them, so the score kernel matches the tree exactly (as the CPU
  // LinearTreeLearner does by reading tree->LeafConst/LeafCoeffs at score time).
  void BuildAndUploadLinearScoreModel(const Tree* tree) const;
  // Launch the per-leaf XtHX (upper-triangle) + Xtg accumulation kernel.
  void LaunchCalcLinearGramKernel(
    const int* cuda_data_index_to_leaf_index, const score_t* gradients, const score_t* hessians,
    const int* leaf_num_feat, const int* leaf_feat_offset, const int* leaf_feat_col,
    const int* leaf_xthx_offset, const int* leaf_xtg_offset, int num_leaves,
    double* cuda_xthx, double* cuda_xtg, int* cuda_leaf_nonzero);
  // Launch the linear score-update kernel (score += const + sum coeff*rawfeat, NaN -> leaf_output).
  void LaunchLinearAddScoreKernel(
    const int* cuda_data_index_to_leaf_index, const double* cuda_leaf_const,
    const double* cuda_leaf_coeff, const int* cuda_leaf_coeff_offset, const int* cuda_leaf_coeff_col,
    const int* leaf_num_coeff, const double* cuda_leaf_output, double* score) const;

  void ReduceLeafStat(CUDATree* old_tree, const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf) const;

  void LaunchReduceLeafStatKernel(const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf,
    const int* leaf_parent, const int* left_child, const int* right_child,
    const int num_leaves, const data_size_t num_data, double* cuda_leaf_value, const double shrinkage_rate) const;

  void ConstructBitsetForCategoricalSplit(const CUDASplitInfo* best_split_info);

  void LaunchConstructBitsetForCategoricalSplitKernel(const CUDASplitInfo* best_split_info);

  void AllocateBitset();

  void SelectFeatureByNode(const Tree* tree);

  // Apply a split (device CUDASplitInfo + host-known feature/threshold) to the tree
  // and the data partition. Shared by the main loop and the forced-split pre-pass.
  // Returns the new (right) leaf index. When smaller_slot/larger_slot are given the
  // data partition writes the children's leaf-splits structs into them instead of the
  // fixed pair structs (used by the hybrid level-batched growth phase).
  int ApplySplit(CUDATree* tree, const CUDASplitInfo* best_split_info, const int leaf_index,
                 CUDALeafSplitsStruct* smaller_slot = nullptr,
                 CUDALeafSplitsStruct* larger_slot = nullptr,
                 int deferred_slot = -1);

  // ---- hybrid growth: depth-wise (level-batched) prefix + leaf-wise tail ----
  // While the leaf budget cannot bind, every positive-gain frontier leaf will be
  // split by leaf-wise growth eventually, so whole levels are grown with one
  // device synchronization instead of one full pipeline sync per split. Split
  // decisions are node-local, hence the resulting tree equals the leaf-wise tree
  // whenever the budget does not bind; the tail preserves exact leaf-wise
  // semantics once it might.
  bool HybridGrowthUsable() const;
  // Whether the hybrid prefix will run the single-sync (speculative) level
  // pipeline (see TrainLevelWisePrefixOneSync); requires both batched phases
  // and non-quantized training.
  bool UseOneSyncPrefix() const;
  // Lazily perform the root-sum readback (and the root leaf-output init) that
  // BeforeTrain deferred for the single-sync flow. No-op when not deferred.
  void EnsureRootSumsReadBack(CUDATree* tree);
  // Enqueue histogram construction + best-split search for one sibling pair
  // (device work only; no host synchronization).
  void EnqueuePairBestSplitSearch(const CUDATree* tree,
    const CUDALeafSplitsStruct* smaller_struct, const CUDALeafSplitsStruct* larger_struct,
    int smaller_leaf_index, int larger_leaf_index, bool synchronize = true,
    int pipeline = 0, int parent_leaf_index = -1);
  // One sibling pair of a level awaiting its best-split search. parent is the
  // pair's parent leaf index (== the split's left leaf; -1 for the root pair):
  // the quantized path stores the parent's histogram bit width at that index.
  // Without selective growth right > left always holds, so parent ==
  // min(smaller, larger); with selective growth right-child indices are
  // recycled and the explicit field is required.
  struct HybridPendingPair {
    int smaller;
    int larger;
    int parent;
    const CUDALeafSplitsStruct* smaller_struct;
    const CUDALeafSplitsStruct* larger_struct;
  };
  // Batched replacement for the per-pair EnqueuePairBestSplitSearch loop of one
  // level: builds per-pair descriptors (validity flags, leaf sizes, quantized
  // histogram bit widths), uploads them in a single H2D copy, then runs ONE
  // batched construct/fix/subtract launch (histogram constructor) and ONE batched
  // find + sync launch (best split finder) covering every pair. Device work only;
  // the caller synchronizes via SyncAllLeafBestSplitsToHost.
  void EnqueueLevelBestSplitSearch(const CUDATree* tree,
    const std::vector<HybridPendingPair>& pairs);
  // One applied split of a level: host-assigned child leaf indices plus the pair
  // struct slots the batched apply kernels fill (smaller/larger designation is
  // decided on-device; the host learns it from the deferred readback).
  struct HybridAppliedSplit {
    int left;
    int right;
    CUDALeafSplitsStruct* smaller_slot;
    CUDALeafSplitsStruct* larger_slot;
  };
  // Speculative variant of EnqueueLevelBestSplitSearch used by the single-sync
  // flow: enqueued right after the level's batched apply, BEFORE the child
  // statistics are read back. Descriptors carry only host-known data (struct slot
  // pointers, the max_depth gate); the batched kernels derive leaf indices,
  // sizes, hessian sums and the min_data/min_hessian gates from the child structs
  // the apply kernels write (ordered via the partition's apply-done event). The
  // construct grid is sized by the parents' sizes (smaller child <= parent / 2)
  // and the kernel restores the exact row grouping from the device-side actual
  // sizes, so results are bit-identical to the classic two-sync flow.
  void EnqueueLevelBestSplitSearchSpeculative(const CUDATree* tree,
    const std::vector<HybridAppliedSplit>& applied);
  // Collect all leaves with a valid cached best-split candidate, mirroring the
  // device cache into the host split arrays (the leaf-wise tail may apply any of
  // them later).
  void CollectSplittableLeaves(const CUDATree* tree, std::vector<int>* splittable);
  // Leaf-budget arbitration of one level: returns false when the budget binds
  // and the level must defer to the leaf-wise tail; truncates splittable to a
  // final partial level (*final_partial_level = true) when every child would sit
  // at max_depth. Also applies the FALCATA_HYBRID_MAXSPLITS debug cap.
  bool ArbitrateLevelBudget(const CUDATree* tree, std::vector<int>* splittable,
    bool* final_partial_level);
  // Batched apply of one level's splits: tree-structure update (SplitBatch) +
  // partition (SplitLevelBatched), zero host synchronization; fills *applied.
  void ApplyLevelBatched(CUDATree* tree, const std::vector<int>& splittable,
    std::vector<HybridAppliedSplit>* applied);
  // Shared per-level readback bookkeeping: consume the deferred split info of one
  // level's applied splits (child counts/starts/sums), update the host leaf
  // arrays, quantized histogram bit widths and smaller/larger designation, count
  // the splits, and (when next_pairs is non-null) emit the next level's pairs.
  void FinishLevelBookkeeping(const std::vector<HybridAppliedSplit>& applied,
    const std::vector<int>& batch_info,
    std::vector<HybridPendingPair>* next_pairs,
    int* num_splits);
  // Grow full levels until the next level could exceed num_leaves. Returns the
  // number of splits applied; afterwards every current leaf has a valid cached
  // best-split candidate and smaller_/larger_leaf_index_ name the last applied pair.
  int TrainLevelWisePrefix(CUDATree* tree);
  // Single-sync variant of TrainLevelWisePrefix: enqueues each level's APPLY and
  // the children's SEARCH back to back, then reads best splits + deferred split
  // info back with ONE device synchronization per level (the classic flow pays
  // two). Non-quantized batched-kernels + batched-apply configurations only.
  int TrainLevelWisePrefixOneSync(CUDATree* tree);

#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  // ---- graphs L1: device-driven level loop (see cuda_hybrid_graph.hpp) ----
  // One cached instantiated graph per per-tree pointer key (the compact column
  // view is double-buffered, so two keys alternate in the steady state).
  struct HybridGraphInstance {
    std::vector<const void*> key;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUDAHybridGraphLoopState* state_dev = nullptr;  // raw cudaMalloc
    ~HybridGraphInstance();
  };
  // Static config gate of the graph loop (depth-limited one-sync prefix only,
  // controller capacity limits, no debug envs, FALCATA_GRAPH_LEVEL_LOOP).
  bool HybridGraphPrefixUsable() const;
  // Lazy one-time setup: static feature tables, journal/staging buffers, and
  // worst-case capacity preallocation of every captured device buffer.
  bool SetupHybridGraphStatics();
  // Build (capture + instantiate) one graph instance for the current per-tree
  // pointer key. Returns false (and rolls back) on any failure.
  bool BuildHybridGraphInstance(CUDATree* tree, HybridGraphInstance* instance);
  // The per-tree pointer/flag key a graph instance is valid for.
  void ComputeHybridGraphKey(CUDATree* tree, std::vector<const void*>* key) const;
  // Cached-or-new instance for this tree; nullptr -> host loop fallback.
  HybridGraphInstance* GetOrBuildHybridGraph(CUDATree* tree);
  // The graph-driven prefix: root search (host, as usual), ONE graph launch,
  // ONE readback + host bookkeeping replay. Returns the number of splits, or
  // -1 when no instance is available (caller falls back to the host loop).
  int TrainLevelWisePrefixGraph(CUDATree* tree);
  void ReleaseHybridGraphs();
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED

  // ---- selective (grow-then-prune) hybrid growth: exact leaf-wise equivalence
  // in budget-limited configs (num_leaves << 2^max_depth) ----
  //
  // Leaf-wise growth is greedy selection: repeatedly apply the max-gain
  // candidate among AVAILABLE candidates (available = parent split already
  // applied) until num_leaves - 1 splits or no positive gain. Key monotonicity
  // property: newly discovered (deeper) candidates can only DISPLACE currently
  // selected candidates from the greedy selection, never promote unselected
  // ones (a new candidate inserts into the greedy sequence right after its
  // parent's selection and consumes budget earlier, shifting the tail out).
  // Hence, per level: run the greedy selection over all live applied splits +
  // the newly discovered frontier candidates, apply ONLY the selected frontier
  // candidates (unselected ones are permanently dormant), and eagerly collapse
  // applied splits displaced from the selection (their whole subtree is
  // displaced by availability closure). At termination the live applied splits
  // equal the classic leaf-wise selection, in the classic greedy order given by
  // the final simulation; the tree is then rebuilt host-side in that order
  // (classic node/leaf numbering => bit-identical model files in deterministic
  // configs). Eager collapse recycles hybrid leaf indices (and their histogram
  // slots), so live leaves never exceed num_leaves and no device buffer grows.

  // One applied split record of the selective growth phase.
  struct SelectiveApplied {
    int parent;             // id of the parent applied record (-1 for the root split)
    int left_leaf;          // hybrid leaf that was split (left child keeps it)
    int right_leaf;         // hybrid right child leaf (recycled index possible)
    int kid_left;           // applied child record at left_leaf (-1 none)
    int kid_right;          // applied child record at right_leaf (-1 none)
    bool dead;              // collapsed (pruned) -- never selectable again
    bool selected;          // in the current greedy selection (scratch)
    double gain;
    data_size_t data_start;  // the split node's window in the main index array
    data_size_t num_data;
    // captured host copy of the device CUDASplitInfo at apply time (categorical
    // pointers scrubbed by SyncAllLeafBestSplitsToHost)
    CUDASplitInfo info;
  };
  // One frontier candidate (discovered this level, not yet applied).
  struct SelectiveFrontier {
    int parent;             // applied record that created this leaf (-1 root)
    int leaf;               // hybrid leaf index
    double gain;
  };

  // Resolve the fp32 histogram mode (FALCATA_FP32_HIST) against the consumers
  // its layout does not cover and hand the result to the best split finder.
  void SyncHistFP32();
  // Whether the budget-limited selective mode governs this tree's growth.
  bool UseSelectiveGrowth() const;
  // Exact host replica of FindBestFromAllSplitsKernel's equal-gain tie-break:
  // the winner among equal-gain candidates is decided by the kernel's
  // strided per-thread scan + shuffle-down reductions (keep-current-on-tie),
  // NOT by the smallest leaf index. Returns the winning classic leaf index.
  int SelectiveTieBreak(const std::vector<int>& tied_classic_leaves) const;
  // Enqueue the root level's batched search of the single-sync flow (descriptor
  // with device-derived gating, exact host-known construct grid). Shared by the
  // exact-fit one-sync prefix and the selective one-sync loop.
  void EnqueueRootLevelSearchOneSync();
  // Reset the per-tree selective growth state.
  void SelectiveReset();
  // Collect the new frontier candidates from the given child leaves (children of
  // last level's applied splits; the root leaf on the first level).
  void SelectiveDiscoverFrontier(const std::vector<int>& child_leaves);
  // One level's host-side selection step: greedy simulation over live applied
  // splits + the current frontier (budget num_leaves - 1, gains descending,
  // ties to the smaller classic leaf index, matching FindBestFromAllSplits),
  // then eager collapse of displaced applied splits (device window rewrite +
  // histogram slot recycling) and creation of the applied records for the
  // selected frontier candidates (right-child indices from the free list).
  // Returns the number of frontier candidates to apply this level.
  int RunSelectiveLevel();
  // Batched partition-only apply of the selected frontier splits (the CUDA tree
  // object is not touched during selective growth; the final tree is rebuilt
  // host-side after pruning). Fills *applied for the readback bookkeeping.
  void ApplyLevelBatchedSelective(std::vector<HybridAppliedSplit>* applied);
  // The two growth loops (same level pipelines as the exact-fit prefix).
  void TrainSelectiveOneSync(CUDATree* tree);
  void TrainSelectiveTwoSync(CUDATree* tree);
  // Rebuild the final (pruned) tree host-side in classic greedy order, remap the
  // partition's data-index-to-leaf-index to final numbering and upload the final
  // per-leaf data layout.
  void SelectiveFinalize(CUDATree* tree);
  // Full selective training of one tree (grow + prune + finalize).
  void TrainSelective(CUDATree* tree);
  // Leaf depth during hybrid growth: the CUDA tree's host mirror on the exact-fit
  // paths, the selective host array when selective growth is active.
  int GrowthLeafDepth(const CUDATree* tree, const int leaf) const {
    return selective_active_ ? sel_leaf_depth_[leaf] : tree->leaf_depth(leaf);
  }

  // Apply forcedsplits_filename before the main split loop, mirroring
  // SerialTreeLearner::ForceSplits (numerical features, single-GPU, non-quantized).
  // Returns the number of forced splits applied.
  int ForceSplitsCUDA(CUDATree* tree, int* num_splits_done);

  #ifdef DEBUG
  void CheckSplitValid(
    const int left_leaf, const int right_leaf);
  #endif  // DEBUG

  void RenewDiscretizedTreeLeaves(CUDATree* cuda_tree);

  void LaunchCalcLeafValuesGivenGradStat(CUDATree* cuda_tree, const data_size_t* num_data_in_leaf);

  void NCCLReduceHistogram();

  // Per-tree compact column data: transposes the on-GPU row-major bin matrix into
  // a column-major buffer for partition kernels, avoiding the 17 GB per-column
  // allocation in CUDAColumnData.
  void BuildCompactColumnView();
  // Lazy fallback of the packed split read (FALCATA_SPLIT_PACKED_READ): the
  // batched apply path reads the packed compact matrix directly, so the
  // column-major gather is skipped in BuildCompactColumnView; any classic
  // per-split Split() (leaf-wise tail, forced splits, batched-apply fallback
  // env) still needs the plain per-column buffer, so ApplySplit calls this
  // first to run the deferred gather once for the tree.
  void EnsureClassicColumnView();
  CUDAVector<uint8_t> compact_column_buffer_;
  std::vector<int> compact_column_to_orig_;        // [slot] -> original column index
  std::vector<int> orig_column_to_compact_slot_;   // [col] -> slot, or -1
  CUDAVector<int> cuda_compact_slot_for_col_;      // GPU copy of orig_column_to_compact_slot_
  CUDAVector<int> cuda_src_part_col_offsets_;
  CUDAVector<int> cuda_src_part_stride_;
  // Per-slot precomputed metadata for the new (slot, row)-launched transpose:
  CUDAVector<size_t> cuda_slot_p_byte_;
  CUDAVector<int> cuda_slot_p_stride_;
  CUDAVector<int> cuda_slot_col_in_p_;
  uint64_t compact_col_signature_ = 0;
  // packed split read: this tree's column view is the packed compact matrix
  // (no column-major gather ran yet; see EnsureClassicColumnView)
  bool compact_packed_view_active_ = false;
  // gather inputs stashed for the lazy fallback (device pointer + 4-bit flag)
  const uint8_t* compact_gather_src_ = nullptr;
  bool compact_gather_src_is_4bit_ = false;

  // Mirror of BasicLeafConstraints::Update for the CUDA path. Given a just-applied
  // numerical split on `inner_feature_index`, the parent leaf (`left_leaf`) and the
  // newly created leaf (`right_leaf`), and the (clamped) child outputs, update the
  // host-side per-leaf [min,max] constraint arrays.
  void UpdateLeafConstraints(
    const int left_leaf, const int right_leaf, const int inner_feature_index,
    const double left_value, const double right_value, const bool is_numerical_split);

  // number of threads on CPU
  int num_threads_;

  // FALCATA_FIXEDPOINT_QUANT: non-stochastic near-lossless quant mode.
  // Resolved once in Init so the histogram constructor and the gradient
  // discretizer share the same effective quant bin count.
  bool fixedpoint_quant_ = false;
  // Outlier-robust gradient scale within fixed-point mode (default ON when
  // fixedpoint_quant_ is on; disabled by cuda_plan=auto,robust_scale:off).
  bool fixedpoint_robust_scale_ = false;
  int effective_quant_bins_ = 0;

  // CUDA components for tree training

  // leaf splits information for smaller and larger leaves
  std::unique_ptr<CUDALeafSplits> cuda_smaller_leaf_splits_;
  std::unique_ptr<CUDALeafSplits> cuda_larger_leaf_splits_;
  // hybrid growth phase-1 state: per-split (smaller, larger) struct slots for the
  // pairs of one level (2 slots per applied split, reused across levels), and the
  // host copy of the device per-leaf best-split cache. The slots are one device
  // slab of bare CUDALeafSplitsStructs (they are only ever written/read by the
  // split kernels through their struct pointers); allocating num_leaves + 2
  // CUDALeafSplits objects instead costs ~3 cudaMallocs each (~3000 calls,
  // ~3.7 ms) inside the first Train() call.
  bool use_hybrid_growth_ = false;
  CUDAVector<CUDALeafSplitsStruct> hybrid_pair_slots_;
  std::vector<CUDASplitInfo> host_leaf_best_splits_;
  // hybrid growth: batched per-level kernels (FALCATA_HYBRID_BATCH_KERNELS,
  // default on; "0" falls back to the per-pair kernel launches)
  bool use_hybrid_batch_kernels_ = false;
  std::vector<CUDAHybridPairDescriptor> host_hybrid_pair_descs_;
  CUDAVector<CUDAHybridPairDescriptor> cuda_hybrid_pair_descs_;
  // hybrid growth: batched per-level APPLY phase (FALCATA_HYBRID_BATCH_APPLY,
  // default on; "0" falls back to the per-split deferred ApplySplit loop)
  bool use_hybrid_batch_apply_ = false;
  std::vector<CUDATreeBatchSplit> host_tree_batch_splits_;
  std::vector<CUDAHybridApplySplitInput> host_apply_split_inputs_;
  // hybrid growth: single-sync (speculative) level pipeline
  // (FALCATA_HYBRID_ONE_SYNC, default on; "0" keeps the classic two-sync flow)
  bool use_hybrid_one_sync_ = false;
  // selective (grow-then-prune) hybrid growth for budget-limited configs
  // (FALCATA_HYBRID_SELECTIVE, default on; "0" falls back to the classic loop)
  bool use_hybrid_selective_ = false;
  bool selective_active_ = false;
#ifdef FALCATA_HYBRID_GRAPH_SUPPORTED
  // ---- graphs L1 state ----
  bool hybrid_graph_statics_ready_ = false;
  bool hybrid_graph_disabled_ = false;   // sticky off after build failures
  int hybrid_graph_build_failures_ = 0;
  std::vector<std::unique_ptr<HybridGraphInstance>> hybrid_graph_cache_;
  CUDAVector<CUDAHybridGraphFeatureMeta> cuda_hybrid_graph_feature_meta_;
  CUDAVector<CUDAHybridGraphFeatureSource> cuda_hybrid_graph_feature_source_;
  CUDAVector<double> cuda_hybrid_graph_real_thresholds_;
  CUDAVector<int> cuda_hybrid_graph_journal_;
  std::vector<CUDAHybridGraphFeatureSource> host_hybrid_graph_feature_source_;
  // pinned staging: per-tree state prefix + journal readback
  CUDAHybridGraphLoopState* pinned_hybrid_graph_state_ = nullptr;
  int* pinned_hybrid_graph_journal_ = nullptr;
  size_t hybrid_graph_journal_size_ = 0;
  // static config mirror written into every instance's device state
  CUDAHybridGraphLoopState hybrid_graph_state_template_;
  std::vector<int> hybrid_graph_splittable_scratch_;
  std::vector<int> hybrid_graph_batch_info_;
  std::vector<HybridAppliedSplit> hybrid_graph_applied_scratch_;
#endif  // FALCATA_HYBRID_GRAPH_SUPPORTED
  std::vector<SelectiveApplied> sel_applied_;
  std::vector<SelectiveFrontier> sel_frontier_;
  std::vector<int> sel_leaf_parent_;    // hybrid leaf -> applied record id (-1 root)
  std::vector<int> sel_leaf_depth_;
  std::vector<int> sel_leaf_classic_;   // classic leaf index (last simulation)
  std::vector<int> sel_leaf_frontier_;  // hybrid leaf -> frontier idx (per level)
  std::vector<char> sel_leaf_live_;
  std::vector<int> sel_free_leaves_;    // recycled hybrid leaf indices (LIFO)
  std::vector<int> sel_order_;          // last simulation: greedy order (applied ids, ~frontier idx)
  std::vector<int> sel_order_classic_leaf_;
  std::vector<int> sel_level_apply_;    // applied record ids to apply this level
  std::vector<CUDACollapseWindow> sel_collapse_windows_;
  std::vector<int> sel_freed_hist_slots_;
  int sel_num_allocated_ = 0;           // peak hybrid leaf count (dirty hist slots)
  int sel_num_splits_ = 0;              // applied splits surviving readback bookkeeping
  int sel_last_peak_ = 0;               // sel_num_allocated_ of the finished tree
  // churn statistics (FALCATA_HYBRID_DEBUG): applied / displaced split totals
  int64_t sel_stat_applied_ = 0;
  int64_t sel_stat_displaced_ = 0;
  int64_t sel_stat_levels_ = 0;
  int64_t sel_stat_trees_ = 0;
  // reusable device slab for CUDATree's per-tree arrays (see CUDATree ctor)
  CUDAVector<uint8_t> cuda_tree_pool_buffer_;
  // whether BeforeTrain deferred the root-sum readback (single-sync flow); the
  // readback (plus the root leaf-output init) then happens lazily in
  // EnsureRootSumsReadBack on the paths that need host root sums
  bool root_sums_deferred_ = false;
  // data partition that partitions data indices into different leaves
  std::unique_ptr<CUDADataPartition> cuda_data_partition_;
  // for histogram construction
  std::unique_ptr<CUDAHistogramConstructor> cuda_histogram_constructor_;
  // for best split information finding, given the histograms
  std::unique_ptr<CUDABestSplitFinder> cuda_best_split_finder_;
  // gradient discretizer for quantized training
  std::unique_ptr<CUDAGradientDiscretizer> cuda_gradient_discretizer_;

  std::vector<int> leaf_best_split_feature_;
  std::vector<uint32_t> leaf_best_split_threshold_;
  std::vector<uint8_t> leaf_best_split_default_left_;
  std::vector<data_size_t> leaf_num_data_;
  std::vector<data_size_t> leaf_data_start_;
  std::vector<double> leaf_sum_gradients_;
  std::vector<double> leaf_sum_hessians_;
  // host-side per-leaf monotone constraints (basic method). [-DBL_MAX, +DBL_MAX]
  // means unconstrained. Empty/unused when monotone_constraints is not set.
  std::vector<double> leaf_constraint_min_;
  std::vector<double> leaf_constraint_max_;
  // per-inner-feature monotone constraint, mapped from the real-indexed
  // config_->monotone_constraints; empty when monotone_constraints is not set.
  std::vector<int8_t> monotone_constraints_;
  bool use_monotone_constraints_;
  int smaller_leaf_index_;
  int larger_leaf_index_;
  int best_leaf_index_;
  int num_cat_threshold_;
  bool has_categorical_feature_;
  // whether need to select features by node
  bool select_features_by_node_;

  std::vector<int> categorical_bin_to_value_;
  std::vector<int> categorical_bin_offsets_;

  mutable CUDAVector<double> cuda_leaf_gradient_stat_buffer_;
  mutable CUDAVector<double> cuda_leaf_hessian_stat_buffer_;
  mutable data_size_t leaf_stat_buffer_size_;
  mutable data_size_t refit_num_data_;
  uint32_t* cuda_bitset_;
  size_t cuda_bitset_len_;
  uint32_t* cuda_bitset_inner_;
  size_t cuda_bitset_inner_len_;
  size_t* cuda_block_bitset_len_buffer_;
  int* cuda_categorical_bin_to_value_;
  int* cuda_categorical_bin_offsets_;

  /*! \brief gradients on CUDA */
  CUDAVector<score_t> cuda_gradients_;
  /*! \brief hessians on CUDA */
  CUDAVector<score_t> cuda_hessians_;
  /*! \brief whether boosting is done on CUDA */
  bool boosting_on_cuda_;

  // members used in multi-GPU training
  /*! \brief cuda stream for nccl operations */
  cudaStream_t nccl_stream_;
  /*! \brief index map from leaf index to histogram index */
  std::vector<int> leaf_to_hist_index_map_;
  /*! \brief number of total histogram bins */
  int num_total_bin_;
  /*! \brief global number of data in the leaves across */
  std::vector<data_size_t> global_num_data_in_leaf_;

  #ifdef DEBUG
  /*! \brief gradients on CPU */
  std::vector<score_t> host_gradients_;
  /*! \brief hessians on CPU */
  std::vector<score_t> host_hessians_;
  #endif  // DEBUG

  // ---- linear tree support ----
  // Raw un-binned numerical feature values on device, feature-major:
  // cuda_raw_data_[col * num_data_ + row]. col indexes numerical features in
  // dataset feature order (== host Dataset::numeric_feature_map_ order).
  CUDAVector<float> cuda_raw_data_;
  int num_numeric_features_ = 0;
  // inner feature index -> numeric column in cuda_raw_data_ (-1 if categorical)
  std::vector<int> inner_feature_to_numeric_col_;
  // device scratch for the per-leaf gram accumulation
  CUDAVector<double> cuda_linear_xthx_;
  CUDAVector<double> cuda_linear_xtg_;
  CUDAVector<int> cuda_linear_nonzero_;
  CUDAVector<int> cuda_linear_leaf_num_feat_;
  CUDAVector<int> cuda_linear_leaf_feat_offset_;
  CUDAVector<int> cuda_linear_leaf_feat_col_;
  CUDAVector<int> cuda_linear_leaf_xthx_offset_;
  CUDAVector<int> cuda_linear_leaf_xtg_offset_;
  // device copies of the fitted per-leaf linear model, used by the score kernel.
  // Rebuilt from the (post-shrinkage) tree at score time, so mutable/const-built.
  mutable CUDAVector<double> cuda_linear_leaf_const_;
  mutable CUDAVector<double> cuda_linear_leaf_coeff_;
  mutable CUDAVector<int> cuda_linear_leaf_coeff_offset_;
  mutable CUDAVector<int> cuda_linear_leaf_coeff_col_;
  mutable CUDAVector<int> cuda_linear_leaf_num_coeff_;
  mutable CUDAVector<double> cuda_linear_leaf_output_;
  mutable bool last_tree_is_linear_ = false;
};

}  // namespace Falcata

#else  // USE_CUDA

// When GPU support is not compiled in, quit with an error message

namespace Falcata {

class CUDASingleGPUTreeLearner: public SerialTreeLearner {
 public:
    #pragma warning(disable : 4702)
    explicit CUDASingleGPUTreeLearner(const Config* tree_config, const bool /*boosting_on_cuda*/) : SerialTreeLearner(tree_config) {
      Log::Fatal("CUDA Tree Learner was not enabled in this build.\n"
                 "Please recompile with CMake option -DUSE_CUDA=1 (NVIDIA GPUs) or -DUSE_ROCM=1 (AMD GPUs)");
    }
};

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_SRC_TREELEARNER_CUDA_CUDA_SINGLE_GPU_TREE_LEARNER_HPP_
