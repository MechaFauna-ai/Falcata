/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_HPP_
#define FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_HPP_

#ifdef USE_CUDA

#include <Falcata/boosting.h>
#include <Falcata/cuda/cuda_utils.hu>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace Falcata {

/*!
 * \brief The native CUDA predictor: evaluates a model's trees on dense rows on the device, with the same result as the
 *        CPU predictor (Predictor / GBDT::PredictRawInRange).
 *
 * Exactness. The raw score of each output is the sum, in tree order and in double precision, of the leaf values the
 * row reaches, exactly as GBDT::PredictRawInRange adds them; the device adds with __dadd_rn (no contraction), so the
 * raw scores are bit-identical to the CPU predictor's. Routing is exact too: a float64 input is compared with the
 * double thresholds; a float32 input x is compared with the largest float not above the threshold t, which decides
 * (double)x <= t exactly for every float x (floats are a subset of doubles). Values with |x| <= kZeroThreshold are
 * zero, NaN and zero go where Tree::NumericalDecision sends them, categorical splits follow
 * Tree::CategoricalDecision. Linear leaves add const + coeff_i * x_i with separate rounding of each product and sum,
 * as the CPU build (-ffp-contract=off) does. The normal (transformed) prediction is the raw score passed through
 * Boosting::ConvertRawToPrediction on the host, the same code PredictInRange runs, so it is bit-identical as well.
 */

/*! \brief flag bits of PackedNode::feature_flags; the feature index uses the bits below kNodeFeatureMask */
constexpr uint32_t kNodeCategorical = 1u << 31;
constexpr uint32_t kNodeDefaultLeft = 1u << 30;
constexpr int kNodeMissingShift = 28;           // two bits: MissingType::None / Zero / NaN
constexpr uint32_t kNodeFeatureMask = (1u << 28) - 1;

/*!
 * \brief One internal node. Children are global indices into the forest: >= 0 an internal node, < 0 the leaf ~child
 *        (a global leaf index). A categorical node keeps its index into the bitset ranges in the bits of threshold.
 */
template <typename T>
struct PackedNode {
  T threshold;
  uint32_t feature_flags;
  int32_t left;
  int32_t right;
};

/*! \brief device pointers of one packed forest (every tree of the model) */
template <typename T>
struct DeviceForestView {
  const PackedNode<T>* nodes = nullptr;
  const int32_t* tree_root = nullptr;         // [num_trees]: the root node, or ~leaf for a single-leaf tree
  const int32_t* tree_leaf_begin = nullptr;   // [num_trees]: global index of the tree's leaf 0
  const double* leaf_values = nullptr;        // [total_leaves * leaf_dim]
  const uint32_t* cat_words = nullptr;        // categorical bitsets
  const int64_t* cat_range = nullptr;         // [2 * num_cat_nodes]: word begin, word count
  const uint8_t* tree_is_linear = nullptr;    // [num_trees]
  const int64_t* linear_begin = nullptr;      // [total_leaves + 1] into linear_coeff / linear_feature
  const double* linear_const = nullptr;       // [total_leaves]
  const double* linear_coeff = nullptr;
  const int32_t* linear_feature = nullptr;
  const int32_t* node_count = nullptr;        // [total_nodes]: data count of each internal node (TreeSHAP)
  const int32_t* leaf_count = nullptr;        // [total_leaves]: data count of each leaf (TreeSHAP)
  const double* tree_expected = nullptr;      // [num_trees]: Tree::ExpectedValue (TreeSHAP)
  int leaf_dim = 1;
  int max_depth = 0;                          // edges on the longest root-to-leaf path
  bool has_categorical = false;
  bool has_linear = false;
};

/*! \brief one dense input matrix on the device: element (r, c) at data[r * row_stride + c * col_stride] */
template <typename T>
struct DeviceRows {
  const T* data = nullptr;
  int64_t num_row = 0;
  int32_t num_col = 0;
  int64_t row_stride = 0;
  int64_t col_stride = 0;
};

/*! \brief what a CUDA prediction measured: device bytes it held and how it split a host input */
struct CUDAPredictStats {
  int64_t forest_bytes = 0;     // the packed model, cached across calls
  int64_t scratch_bytes = 0;    // input chunk and output buffers of this call
  int64_t num_chunks = 0;       // host inputs: number of row chunks copied to the device
  int64_t chunk_rows = 0;
};

/*!
 * \brief How raw scores are computed; both kernels add each output's leaf values in tree order, so both are exact.
 *  - kTileWs (the plan's choice): a block stages `tile_rows` rows in shared memory; warp 0 keeps their running sums
 *    (lane r adds row r's leaf values in tree order) while the other warps walk the next tile of `tile_trees` trees
 *    for every (row, tree) pair, `ilp` walks in flight per thread; the tiles' leaf values are double-buffered.
 *  - kRow: a thread per row walks the trees, `ilp` of them at a time; for models with so many outputs per row that
 *    a block cannot hold its rows' sums.
 * Measured alternatives that lost (benchmarks/predict_gpu.py rounds, docs/performance.md): a thread per row for
 * every shape, tiles summed by one thread per row after a barrier, an 8-byte depth-first node layout, tree nodes
 * staged in shared memory, and two passes (all leaf values, then the sums) for small batches.
 */
struct PredictPlan {
  enum Kernel { kRow, kTileWs } kernel = kTileWs;
  int ilp = 2;
  int tile_rows = 8;
  int tile_trees = 64;
  bool stage_rows = true;   // rows in shared memory (else read from global memory)
  int block_threads = 256;  // warp 0 sums, the others walk trees
};

/*! \brief launchers (cuda_predictor.cu); out is row-major [num_row][num_output] doubles */
template <typename TIn, typename TThr>
void LaunchPredictRawKernel(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                            int num_iteration, int num_tree_per_iteration, int num_output, bool vector_leaf,
                            const PredictPlan& plan, double* out, cudaStream_t stream);

/*! \brief out is row-major [num_row][num_tree] int32 leaf indices of the trees [tree_begin, tree_begin + num_tree) */
template <typename TIn, typename TThr>
void LaunchPredictLeafKernel(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                             int num_tree, int32_t* out, cudaStream_t stream);

/*!
 * \brief Feature contributions (TreeSHAP, Tree::PredictContrib) of rows [0, rows.num_row): out is row-major
 *        [num_row][num_tree_per_iteration][num_feature + 1]; scratch holds the path buffers of the threads in flight
 *        (ContribScratchElements per row and output)
 */
template <typename TIn, typename TThr>
void LaunchPredictContribKernel(const DeviceForestView<TThr>& forest, const DeviceRows<TIn>& rows, int tree_begin,
                                int num_iteration, int num_tree_per_iteration, int num_feature, double* out,
                                void* scratch, cudaStream_t stream);
size_t ContribScratchBytes(int max_depth);

/*! \brief shared memory a kTileWs launch needs, and the most the device grants a block */
size_t TileWsSharedBytes(const PredictPlan& plan, int num_col, size_t input_elem, int num_tree_per_iteration);
size_t MaxSharedBytesPerBlock();

class CUDAPackedForest;

/*! \brief A request to CUDAPredictor::Predict */
struct CUDAPredictRequest {
  const void* data = nullptr;
  int data_type = 0;            // C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
  int64_t num_row = 0;
  int32_t num_col = 0;
  int64_t row_stride = 0;       // in elements; host inputs must be C or F contiguous
  int64_t col_stride = 0;
  bool on_device = false;       // data and out are device pointers (else host pointers)
  int predict_type = 0;         // C_API_PREDICT_NORMAL, _RAW_SCORE or _LEAF_INDEX
  int device_id = -1;           // host inputs: the device to use (-1: the current device)
  cudaStream_t stream = nullptr;  // device inputs: the stream the input was written on
  void* out = nullptr;          // doubles [num_row][num_output], or int32 [num_row][num_tree] for leaf indices
};

/*!
 * \brief The predictor a Booster keeps: packs the model once per (device, threshold precision) and evaluates any
 *        iteration range from that one copy. Invalidate() must be called whenever the model changes.
 */
class CUDAPredictor {
 public:
  CUDAPredictor();
  ~CUDAPredictor();
  void Predict(const Boosting* boosting, const Boosting::PredictRange& range, const CUDAPredictRequest& request,
               CUDAPredictStats* stats);
  void Invalidate();

 private:
  std::shared_ptr<const CUDAPackedForest> GetForest(const Boosting* boosting, int device_id, bool double_thresholds);
  std::mutex mutex_;
  std::map<std::pair<int, bool>, std::shared_ptr<const CUDAPackedForest>> forests_;
};

}  // namespace Falcata

#endif  // USE_CUDA

#endif  // FALCATA_SRC_BOOSTING_CUDA_CUDA_PREDICTOR_HPP_
