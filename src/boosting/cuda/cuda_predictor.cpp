/*!
 * Copyright (c) 2026 Falcata contributors. Licensed under the MIT License.
 * See LICENSE file in the project root for license information.
 */
#include "cuda_predictor.hpp"

#ifdef USE_CUDA

#include <Falcata/c_api.h>
#include <Falcata/falcata_plan.h>
#include <Falcata/tree.h>
#include <Falcata/utils/log.h>
#include <Falcata/utils/openmp_wrapper.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <type_traits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Falcata {

/*! \brief reads the tree arrays the device layout is packed from (a friend of Tree) */
struct TreeDevicePacker {
  static int NumLeaves(const Tree& t) { return t.num_leaves_; }
  static int NumCat(const Tree& t) { return t.num_cat_; }
  static int LeafDim(const Tree& t) { return t.leaf_value_dim_; }
  static bool IsLinear(const Tree& t) { return t.is_linear_; }
  static const std::vector<int>& SplitFeature(const Tree& t) { return t.split_feature_; }
  static const std::vector<double>& Threshold(const Tree& t) { return t.threshold_; }
  static const std::vector<int8_t>& DecisionType(const Tree& t) { return t.decision_type_; }
  static const std::vector<int>& LeftChild(const Tree& t) { return t.left_child_; }
  static const std::vector<int>& RightChild(const Tree& t) { return t.right_child_; }
  static const std::vector<int>& CatBoundaries(const Tree& t) { return t.cat_boundaries_; }
  static const std::vector<uint32_t>& CatThreshold(const Tree& t) { return t.cat_threshold_; }
  static double LeafValue(const Tree& t, int leaf, int k) {
    return t.leaf_value_dim_ > 1 ? t.leaf_values_vec_[static_cast<size_t>(leaf) * t.leaf_value_dim_ + k]
                                 : t.leaf_value_[leaf];
  }
  static double LeafConst(const Tree& t, int leaf) { return t.leaf_const_[leaf]; }
  static const std::vector<double>& LeafCoeff(const Tree& t, int leaf) { return t.leaf_coeff_[leaf]; }
  static const std::vector<int>& LeafFeatures(const Tree& t, int leaf) { return t.leaf_features_[leaf]; }
  static int NodeCount(const Tree& t, int node) { return t.internal_count_[node]; }
  static int LeafCount(const Tree& t, int leaf) { return t.leaf_count_[leaf]; }
  static double ExpectedValue(const Tree& t) { return t.ExpectedValue(); }
};

namespace {

// the largest float that is not above t: for every float x, (double)x <= t exactly when x <= FloatThresholdBelow(t)
float FloatThresholdBelow(double t) {
  float f = static_cast<float>(t);
  if (static_cast<double>(f) > t) {
    f = std::nextafter(f, -std::numeric_limits<float>::infinity());
  }
  return f;
}

template <typename T>
T ThresholdFor(double t);
template <>
float ThresholdFor<float>(double t) { return FloatThresholdBelow(t); }
template <>
double ThresholdFor<double>(double t) { return t; }

template <typename T>
void SetThresholdBits(T* threshold, uint32_t bits);
template <>
void SetThresholdBits<float>(float* threshold, uint32_t bits) { std::memcpy(threshold, &bits, sizeof(bits)); }
template <>
void SetThresholdBits<double>(double* threshold, uint32_t bits) {
  // the low word on the device is __double2loint: the low 32 bits of the IEEE pattern
  const uint64_t pattern = bits;
  std::memcpy(threshold, &pattern, sizeof(pattern));
}

template <typename T>
CompactSlot<T> LeafSlot(double value);
template <>
CompactSlot<float> LeafSlot<float>(double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  CompactSlot<float> s;
  s.lo = static_cast<uint32_t>(bits);
  s.meta = static_cast<uint32_t>(bits >> 32);
  return s;
}
template <>
CompactSlot<double> LeafSlot<double>(double value) {
  CompactSlot<double> s;
  s.value = value;
  s.meta = 0;
  s.pad = 0;
  return s;
}

template <typename T>
CompactSlot<T> InternalSlot(T threshold, uint32_t meta);
template <>
CompactSlot<float> InternalSlot<float>(float threshold, uint32_t meta) {
  CompactSlot<float> s;
  std::memcpy(&s.lo, &threshold, sizeof(threshold));
  s.meta = meta;
  return s;
}
template <>
CompactSlot<double> InternalSlot<double>(double threshold, uint32_t meta) {
  CompactSlot<double> s;
  s.value = threshold;
  s.meta = meta;
  s.pad = 0;
  return s;
}

// a categorical node keeps its range index where the threshold would be (the low word for doubles, as ThresholdBits)
template <typename T>
void SetRangeBits(CompactSlot<T>* slot, uint32_t range);
template <>
void SetRangeBits<float>(CompactSlot<float>* slot, uint32_t range) { slot->lo = range; }
template <>
void SetRangeBits<double>(CompactSlot<double>* slot, uint32_t range) {
  const uint64_t pattern = range;
  std::memcpy(&slot->value, &pattern, sizeof(pattern));
}

template <typename T>
int64_t VectorBytes(const CUDAVector<T>& v) {
  return static_cast<int64_t>(v.Size() * sizeof(T));
}

// restores the calling thread's current device: a prediction must not move the caller's (e.g. CuPy's) device
class DeviceGuard {
 public:
  explicit DeviceGuard(int device_id) {
    CUDASUCCESS_OR_FATAL(cudaGetDevice(&previous_));
    if (device_id >= 0 && device_id != previous_) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(device_id));
    }
  }
  ~DeviceGuard() { cudaSetDevice(previous_); }

 private:
  int previous_ = 0;
};

// a non-blocking stream for the chunks of one host input, destroyed however the call ends
class OwnedStream {
 public:
  OwnedStream() { CUDASUCCESS_OR_FATAL(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  ~OwnedStream() {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
  OwnedStream(const OwnedStream&) = delete;
  OwnedStream& operator=(const OwnedStream&) = delete;
  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};


}  // namespace

/*! \brief every tree of a model, packed for one device and one threshold precision; immutable once built */
class CUDAPackedForest {
 public:
  CUDAPackedForest(const Boosting* boosting, bool double_thresholds) : double_thresholds_(double_thresholds) {
    Boosting::PredictRange all;
    all.start_iteration = 0;
    all.num_iteration = boosting->GetCurrentIteration();
    const Boosting::TreeRangeView view = boosting->TreesInRange(all);
    num_trees_ = static_cast<int>(view.trees.size());
    leaf_dim_ = view.vector_leaf ? view.num_output : 1;
    if (double_thresholds) {
      Pack<double>(view, &nodes_double_);
      PackCompact<double>(view, &slots_double_);
    } else {
      Pack<float>(view, &nodes_float_);
      PackCompact<float>(view, &slots_float_);
    }
  }

  template <typename T>
  DeviceForestView<T> View() const;

  int64_t Bytes() const {
    return VectorBytes(nodes_float_) + VectorBytes(nodes_double_) + VectorBytes(slots_float_) +
           VectorBytes(slots_double_) + VectorBytes(slot_start_) + VectorBytes(tree_root_) +
           VectorBytes(tree_leaf_begin_) + VectorBytes(leaf_values_) + VectorBytes(node_count_) +
           VectorBytes(leaf_count_) + VectorBytes(tree_expected_) + VectorBytes(cat_words_) +
           VectorBytes(cat_range_) + VectorBytes(tree_is_linear_) + VectorBytes(linear_begin_) +
           VectorBytes(linear_const_) + VectorBytes(linear_coeff_) + VectorBytes(linear_feature_);
  }

  int num_trees() const { return num_trees_; }
  int max_depth() const { return max_depth_; }

 private:
  template <typename T>
  void Pack(const Boosting::TreeRangeView& view, CUDAVector<PackedNode<T>>* device_nodes) {
    using P = TreeDevicePacker;
    std::vector<PackedNode<T>> nodes;
    std::vector<int32_t> tree_root, tree_leaf_begin;
    int64_t num_leaves_total = 0;
    std::vector<double> leaf_values;
    std::vector<int32_t> node_count, leaf_count;
    std::vector<double> tree_expected;
    std::vector<uint32_t> cat_words;
    std::vector<int64_t> cat_range;
    std::vector<uint8_t> tree_is_linear;
    std::vector<int64_t> linear_begin(1, 0);
    std::vector<double> linear_const, linear_coeff;
    std::vector<int32_t> linear_feature;
    for (const Tree* tree : view.trees) {
      const Tree& t = *tree;
      const int num_leaves = P::NumLeaves(t);
      const bool linear = P::IsLinear(t);
      if (P::LeafDim(t) != leaf_dim_) {
        Log::Fatal("CUDA predictor: a tree stores %d values per leaf, the model %d", P::LeafDim(t), leaf_dim_);
      }
      // children become global indices: internal nodes offset by the nodes before this tree, leaves by the leaves
      const int64_t node_base = static_cast<int64_t>(nodes.size());
      const int64_t leaf_base = num_leaves_total;
      if (node_base + num_leaves > std::numeric_limits<int32_t>::max() ||
          leaf_base + num_leaves > std::numeric_limits<int32_t>::max()) {
        Log::Fatal("CUDA predictor: the model has more than 2^31 nodes or leaves");
      }
      auto global = [node_base, leaf_base](int child) {
        return static_cast<int32_t>(child >= 0 ? node_base + child : ~(leaf_base + ~child));
      };
      tree_root.push_back(num_leaves > 1 ? static_cast<int32_t>(node_base) : static_cast<int32_t>(~leaf_base));
      tree_leaf_begin.push_back(static_cast<int32_t>(leaf_base));
      if (num_leaves > 1) {
        const auto& feature = P::SplitFeature(t);
        const auto& threshold = P::Threshold(t);
        const auto& decision = P::DecisionType(t);
        const auto& left = P::LeftChild(t);
        const auto& right = P::RightChild(t);
        for (int i = 0; i < num_leaves - 1; ++i) {
          PackedNode<T> n;
          if (feature[i] < 0 || static_cast<uint32_t>(feature[i]) > kNodeFeatureMask) {
            Log::Fatal("CUDA predictor: split feature %d is outside [0, %u]", feature[i], kNodeFeatureMask);
          }
          n.feature_flags = static_cast<uint32_t>(feature[i]);
          n.left = global(left[i]);
          n.right = global(right[i]);
          // Tree::GetLeaf takes the categorical branch only when the tree has categorical splits at all
          const bool categorical = P::NumCat(t) > 0 && (decision[i] & kCategoricalMask);
          if (categorical) {
            const auto& boundaries = P::CatBoundaries(t);
            const auto& bits = P::CatThreshold(t);
            const int cat_idx = static_cast<int>(threshold[i]);
            const uint32_t range = static_cast<uint32_t>(cat_range.size() / 2);
            cat_range.push_back(static_cast<int64_t>(cat_words.size()));
            cat_range.push_back(boundaries[cat_idx + 1] - boundaries[cat_idx]);
            cat_words.insert(cat_words.end(), bits.begin() + boundaries[cat_idx],
                             bits.begin() + boundaries[cat_idx + 1]);
            n.threshold = T(0);
            SetThresholdBits<T>(&n.threshold, range);
            n.feature_flags |= kNodeCategorical;
          } else {
            n.threshold = ThresholdFor<T>(threshold[i]);
            const uint32_t missing = static_cast<uint32_t>((decision[i] >> 2) & 3);
            n.feature_flags |= missing << kNodeMissingShift;
            if (decision[i] & kDefaultLeftMask) {
              n.feature_flags |= kNodeDefaultLeft;
            }
          }
          nodes.push_back(n);
          node_count.push_back(P::NodeCount(t, i));
        }
        // depth of the deepest leaf: a walk is never longer, which sizes nothing but is reported to the planner
        std::vector<std::pair<int, int>> stack(1, std::make_pair(0, 1));
        while (!stack.empty()) {
          const auto cur = stack.back();
          stack.pop_back();
          max_depth_ = std::max(max_depth_, cur.second);
          if (left[cur.first] >= 0) stack.emplace_back(left[cur.first], cur.second + 1);
          if (right[cur.first] >= 0) stack.emplace_back(right[cur.first], cur.second + 1);
        }
      }
      num_leaves_total += num_leaves;
      for (int leaf = 0; leaf < num_leaves; ++leaf) {
        for (int k = 0; k < leaf_dim_; ++k) {
          leaf_values.push_back(P::LeafValue(t, leaf, k));
        }
        leaf_count.push_back(P::LeafCount(t, leaf));
        if (linear) {
          const auto& coeff = P::LeafCoeff(t, leaf);
          const auto& feats = P::LeafFeatures(t, leaf);
          linear_const.push_back(P::LeafConst(t, leaf));
          linear_coeff.insert(linear_coeff.end(), coeff.begin(), coeff.end());
          linear_feature.insert(linear_feature.end(), feats.begin(), feats.end());
        } else {
          linear_const.push_back(0.0);
        }
        linear_begin.push_back(static_cast<int64_t>(linear_coeff.size()));
      }
      tree_is_linear.push_back(linear ? 1 : 0);
      // TreeSHAP's base value; vector-leaf trees have none (their contributions are refused before any kernel)
      tree_expected.push_back(leaf_dim_ == 1 ? P::ExpectedValue(t) : 0.0);
      has_linear_ = has_linear_ || linear;
    }
    has_categorical_ = !cat_range.empty();
    cat_range_count_ = static_cast<int64_t>(cat_range.size() / 2);
    if (cat_range.size() / 2 > std::numeric_limits<uint32_t>::max()) {
      Log::Fatal("CUDA predictor: more than 2^32 categorical splits");
    }
    for (int32_t f : linear_feature) {
      if (f < 0) {
        Log::Fatal("CUDA predictor: a linear leaf uses feature %d", f);
      }
    }
    // a CUDAVector of size 0 holds no pointer; kernels never read an empty array
    if (!nodes.empty()) device_nodes->InitFromHostVector(nodes);
    if (!tree_root.empty()) {
      tree_root_.InitFromHostVector(tree_root);
      tree_leaf_begin_.InitFromHostVector(tree_leaf_begin);
    }
    if (!leaf_values.empty()) leaf_values_.InitFromHostVector(leaf_values);
    if (!node_count.empty()) node_count_.InitFromHostVector(node_count);
    if (!leaf_count.empty()) leaf_count_.InitFromHostVector(leaf_count);
    if (!tree_expected.empty()) tree_expected_.InitFromHostVector(tree_expected);
    if (has_categorical_) {
      if (!cat_words.empty()) cat_words_.InitFromHostVector(cat_words);
      cat_range_.InitFromHostVector(cat_range);
    }
    if (has_linear_) {
      tree_is_linear_.InitFromHostVector(tree_is_linear);
      linear_begin_.InitFromHostVector(linear_begin);
      linear_const_.InitFromHostVector(linear_const);
      if (!linear_coeff.empty()) {
        linear_coeff_.InitFromHostVector(linear_coeff);
        linear_feature_.InitFromHostVector(linear_feature);
      }
    }
  }

  // the compact layout (CompactSlot): depth-first slots per tree, leaf values in place. Categorical nodes get the
  // range indices Pack gave them (the same tree and node order). A model it cannot hold keeps PackedNode only.
  template <typename T>
  void PackCompact(const Boosting::TreeRangeView& view, CUDAVector<CompactSlot<T>>* device_slots) {
    using P = TreeDevicePacker;
    if (has_linear_ || leaf_dim_ != 1) {
      return;
    }
    int max_feature = 0;
    for (const Tree* tree : view.trees) {
      const auto& feature = P::SplitFeature(*tree);
      for (int i = 0; i + 1 < P::NumLeaves(*tree); ++i) {
        max_feature = std::max(max_feature, feature[i]);
      }
    }
    int feature_bits = 1;
    while ((int64_t(1) << feature_bits) <= max_feature) {
      ++feature_bits;
    }
    if (feature_bits > 20) {
      return;
    }
    const int64_t max_offset = (int64_t(1) << (26 - feature_bits)) - 1;
    std::vector<CompactSlot<T>> slots;
    std::vector<int32_t> start;
    start.reserve(view.trees.size());
    uint32_t next_range = 0;
    struct Pending {
      int child;
      int64_t parent;   // the slot whose right offset this child sets, or -1
    };
    std::vector<Pending> stack;
    for (const Tree* tree : view.trees) {
      const Tree& t = *tree;
      const int num_leaves = P::NumLeaves(t);
      if (static_cast<int64_t>(slots.size()) + 2 * num_leaves > std::numeric_limits<int32_t>::max()) {
        return;
      }
      if (num_leaves <= 1) {
        start.push_back(~static_cast<int32_t>(slots.size()));
        slots.push_back(LeafSlot<T>(P::LeafValue(t, 0, 0)));
        continue;
      }
      const auto& feature = P::SplitFeature(t);
      const auto& threshold = P::Threshold(t);
      const auto& decision = P::DecisionType(t);
      const auto& left = P::LeftChild(t);
      const auto& right = P::RightChild(t);
      std::vector<uint32_t> range(num_leaves - 1, 0);
      for (int i = 0; i < num_leaves - 1; ++i) {
        if (P::NumCat(t) > 0 && (decision[i] & kCategoricalMask)) {
          range[i] = next_range++;
        }
      }
      start.push_back(static_cast<int32_t>(slots.size()));
      stack.assign(1, Pending{0, -1});
      while (!stack.empty()) {
        const Pending cur = stack.back();
        stack.pop_back();
        const int64_t slot = static_cast<int64_t>(slots.size());
        if (cur.parent >= 0) {
          const int64_t offset = slot - cur.parent;
          if (offset > max_offset) {
            return;
          }
          slots[cur.parent].meta |= static_cast<uint32_t>(offset) << feature_bits;
        }
        if (cur.child < 0) {
          slots.push_back(LeafSlot<T>(P::LeafValue(t, ~cur.child, 0)));
          continue;
        }
        const int i = cur.child;
        uint32_t meta = static_cast<uint32_t>(feature[i]);
        if (left[i] < 0) meta |= kSlotLeftLeaf;
        if (right[i] < 0) meta |= kSlotRightLeaf;
        CompactSlot<T> node;
        if (P::NumCat(t) > 0 && (decision[i] & kCategoricalMask)) {
          meta |= kSlotCategorical;
          node = InternalSlot<T>(T(0), meta);
          SetRangeBits<T>(&node, range[i]);
        } else {
          meta |= static_cast<uint32_t>((decision[i] >> 2) & 3) << kSlotMissingShift;
          if (decision[i] & kDefaultLeftMask) meta |= kSlotDefaultLeft;
          node = InternalSlot<T>(ThresholdFor<T>(threshold[i]), meta);
        }
        slots.push_back(node);
        stack.push_back(Pending{right[i], slot});   // walked second: its offset is known once the left subtree is out
        stack.push_back(Pending{left[i], -1});      // the next slot
      }
    }
    if (next_range != static_cast<uint32_t>(cat_range_count_)) {
      Log::Fatal("CUDA predictor: compact layout found %u categorical splits, the node layout %lld", next_range,
                 static_cast<long long>(cat_range_count_));  // NOLINT(runtime/int)
    }
    device_slots->InitFromHostVector(slots);
    slot_start_.InitFromHostVector(start);
    feature_bits_ = feature_bits;
  }

  template <typename T>
  DeviceForestView<T> FillView(const CUDAVector<PackedNode<T>>& nodes, const CUDAVector<CompactSlot<T>>& slots) const {
    DeviceForestView<T> v;
    v.nodes = nodes.RawData();
    v.slots = slots.Size() > 0 ? slots.RawData() : nullptr;
    v.slot_start = slot_start_.RawData();
    v.feature_bits = feature_bits_;
    v.tree_root = tree_root_.RawData();
    v.tree_leaf_begin = tree_leaf_begin_.RawData();
    v.max_depth = max_depth_;
    v.leaf_values = leaf_values_.RawData();
    v.node_count = node_count_.RawData();
    v.leaf_count = leaf_count_.RawData();
    v.tree_expected = tree_expected_.RawData();
    v.cat_words = cat_words_.RawData();
    v.cat_range = cat_range_.RawData();
    v.tree_is_linear = tree_is_linear_.RawData();
    v.linear_begin = linear_begin_.RawData();
    v.linear_const = linear_const_.RawData();
    v.linear_coeff = linear_coeff_.RawData();
    v.linear_feature = linear_feature_.RawData();
    v.leaf_dim = leaf_dim_;
    v.has_categorical = has_categorical_;
    v.has_linear = has_linear_;
    return v;
  }

  bool double_thresholds_;
  int num_trees_ = 0;
  int leaf_dim_ = 1;
  int max_depth_ = 0;
  bool has_categorical_ = false;
  bool has_linear_ = false;
  CUDAVector<PackedNode<float>> nodes_float_;
  CUDAVector<PackedNode<double>> nodes_double_;
  CUDAVector<CompactSlot<float>> slots_float_;
  CUDAVector<CompactSlot<double>> slots_double_;
  CUDAVector<int32_t> slot_start_;
  int feature_bits_ = 0;
  int64_t cat_range_count_ = 0;
  CUDAVector<int32_t> tree_root_;
  CUDAVector<int32_t> tree_leaf_begin_;
  CUDAVector<double> leaf_values_;
  CUDAVector<int32_t> node_count_;
  CUDAVector<int32_t> leaf_count_;
  CUDAVector<double> tree_expected_;
  CUDAVector<uint32_t> cat_words_;
  CUDAVector<int64_t> cat_range_;
  CUDAVector<uint8_t> tree_is_linear_;
  CUDAVector<int64_t> linear_begin_;
  CUDAVector<double> linear_const_;
  CUDAVector<double> linear_coeff_;
  CUDAVector<int32_t> linear_feature_;
};

template <>
DeviceForestView<float> CUDAPackedForest::View<float>() const { return FillView(nodes_float_, slots_float_); }
template <>
DeviceForestView<double> CUDAPackedForest::View<double>() const { return FillView(nodes_double_, slots_double_); }

class CUDAPredictor::Scratch {
 public:
  Scratch(CUDAPredictor* owner, int device) : owner_(owner), device_(device) {}
  /*!
   * \brief at least `bytes` of device memory for the work of one stream lane, valid until this object ends (the first
   *        call takes the lock). A lane's launches are ordered on its stream, so they may reuse its buffer; a buffer
   *        that has to grow is replaced only after the lane's earlier work is done.
   */
  void* Get(size_t bytes, int lane, cudaStream_t stream) {
    if (!lock_.owns_lock()) {
      lock_ = std::unique_lock<std::mutex>(owner_->scratch_mutex_);
    }
    std::unique_ptr<CUDAVector<char>>& buf = owner_->scratch_[std::make_pair(device_, lane)];
    if (buf == nullptr) {
      buf.reset(new CUDAVector<char>());
    }
    if (buf->Size() < bytes) {
      CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(stream));
      buf->ResizeDiscard(bytes);
    }
    return buf->RawData();
  }

 private:
  CUDAPredictor* owner_;
  int device_;
  std::unique_lock<std::mutex> lock_;
};

CUDAPredictor::CUDAPredictor() {}

CUDAPredictor::~CUDAPredictor() {}

void CUDAPredictor::Invalidate() {
  std::lock_guard<std::mutex> lock(mutex_);
  forests_.clear();
}


std::shared_ptr<const CUDAPackedForest> CUDAPredictor::GetForest(const Boosting* boosting, int device_id,
                                                                 bool double_thresholds) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& slot = forests_[std::make_pair(device_id, double_thresholds)];
  if (slot == nullptr) {
    slot = std::make_shared<const CUDAPackedForest>(boosting, double_thresholds);
  }
  return slot;
}

namespace {

/*! \brief what the planners read of a device, queried once per device */
struct DeviceLimits {
  int sms = 1;
  size_t max_shared = size_t(48) << 10;   // dynamic shared memory a block may opt in to
};

const DeviceLimits& LimitsOf(int device) {
  static std::mutex mutex;
  static std::map<int, DeviceLimits> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(device);
  if (it == cache.end()) {
    DeviceLimits limits;
    int shared = 0;
    CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&limits.sms, cudaDevAttrMultiProcessorCount, device));
    CUDASUCCESS_OR_FATAL(cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    limits.max_shared = static_cast<size_t>(shared);
    it = cache.emplace(device, limits).first;
  }
  return it->second;
}

int CeilPow2(int64_t n) {
  int p = 1;
  while (p < n && p < (1 << 30)) {
    p <<= 1;
  }
  return p;
}

int FloorLog2(int64_t n) {
  int l = 0;
  while ((int64_t(2) << l) <= n) {
    ++l;
  }
  return l;
}

/*!
 * \brief The default mode's shape for one launch. The grid aims at `target` blocks (four per multiprocessor): row
 *        tiles first; when the rows alone do not fill it, each row tile's groups are cut into aligned spans. A wave
 *        holds as many groups as give about 256 tasks. Rows are staged in shared memory (up to 48 KiB) when the walks
 *        of a block read more features than staging copies, with fewer rows per block for wide rows. FALCATA_DEBUG
 *        predict_* tokens override (benchmarks; the pairwise sum is one fixed order, so no shape changes a result).
 */
PairwisePlan ChoosePairwisePlan(int device, int64_t num_row, int num_col, int num_tree_per_iteration,
                                int num_iteration, size_t elem, int max_depth) {
  const FalcataDebugOptions& debug = FalcataDebug();
  const DeviceLimits& limits = LimitsOf(device);
  const int K = num_tree_per_iteration;
  const int num_groups = (num_iteration + kPairwiseGroupIterations - 1) / kPairwiseGroupIterations;
  const int target = debug.predict_target_blocks > 0 ? debug.predict_target_blocks : 4 * limits.sms;
  auto build = [&](int tile_rows, bool stage) {
    PairwisePlan plan;
    plan.out_chunk = std::min(K, 32);
    plan.tile_rows = std::max(1, static_cast<int>(std::min<int64_t>(tile_rows, std::max<int64_t>(num_row, 1))));
    while (plan.tile_rows > 1 && plan.tile_rows * plan.out_chunk > 256) {
      plan.tile_rows /= 2;
    }
    const int pairs = plan.tile_rows * plan.out_chunk;
    const int group_cap = CeilPow2(std::max(num_groups, 1));
    plan.wave_groups = 1;
    while (plan.wave_groups * 2 * pairs <= 256 && plan.wave_groups * 2 <= group_cap) {
      plan.wave_groups *= 2;
    }
    if (debug.predict_wave_groups > 0) {
      plan.wave_groups = 1 << FloorLog2(std::min(debug.predict_wave_groups, 1024));
      while (plan.wave_groups > 1 && plan.wave_groups * pairs > 1024) {
        plan.wave_groups /= 2;
      }
    }
    plan.seg = std::min(plan.wave_groups, 32);
    plan.block_threads = std::min(1024, std::max(32, (pairs * plan.wave_groups + 31) / 32 * 32));
    const int64_t row_tiles = (std::max<int64_t>(num_row, 1) + plan.tile_rows - 1) / plan.tile_rows;
    const int64_t blocks = row_tiles * ((K + plan.out_chunk - 1) / plan.out_chunk);
    int64_t spans = blocks >= target ? 1 : (target + blocks - 1) / blocks;
    if (debug.predict_spans > 0) {
      spans = debug.predict_spans;
    }
    const int64_t per_span = (std::max(num_groups, 1) + spans - 1) / spans;
    plan.span_groups = std::max(plan.wave_groups, CeilPow2(per_span));
    plan.num_spans = std::max(1, (num_groups + plan.span_groups - 1) / plan.span_groups);
    plan.levels = FloorLog2(plan.span_groups / plan.seg) + 1;
    plan.ilp = debug.predict_ilp > 0 ? debug.predict_ilp : 8;
    plan.stage_rows = stage;
    plan.compact = debug.predict_compact != 0;
    return plan;
  };
  const int tile_rows = debug.predict_tile_rows > 0 ? std::min(debug.predict_tile_rows, 256) : 8;
  PairwisePlan plan = build(tile_rows, true);
  const size_t stage_budget = std::min<size_t>(limits.max_shared, size_t(48) << 10);
  while (plan.tile_rows > 1 && PairwiseSharedBytes(plan, num_col, elem) > stage_budget) {
    plan = build(plan.tile_rows / 2, true);
  }
  // staging copies every column of a row once; it pays when the block's walks read more features than that
  const int64_t span_trees = std::min<int64_t>(static_cast<int64_t>(plan.span_groups) * kPairwiseGroupIterations,
                                               num_iteration) * plan.out_chunk;
  const bool worth = span_trees * std::max(1, std::min(max_depth, 12)) >= num_col;
  bool stage = worth && PairwiseSharedBytes(plan, num_col, elem) <= stage_budget;
  if (debug.predict_stage >= 0) {
    stage = debug.predict_stage == 1;
  }
  if (!stage) {
    plan = build(tile_rows, false);
  }
  if (PairwiseSharedBytes(plan, num_col, elem) > limits.max_shared) {
    Log::Fatal("CUDA predictor: the pairwise plan needs %zu bytes of shared memory, the device grants %zu",
               PairwiseSharedBytes(plan, num_col, elem), limits.max_shared);
  }
  return plan;
}

// the deterministic mode's kernel for a launch: row-parallel when the rows alone fill the device, else tree-parallel
// tiles; FALCATA_DEBUG predict_* tokens override (benchmarks; every kernel gives the same result)
PredictPlan ChoosePlan(int64_t num_row, int num_col, int num_tree_per_iteration, size_t elem, int num_tree) {
  const int K = num_tree_per_iteration;
  PredictPlan plan;
  int device = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device));
  const int sms = LimitsOf(device).sms;
  const FalcataDebugOptions& debug = FalcataDebug();
  const std::string& forced = debug.predict_kernel;
  if (forced == "row") {
    plan.kernel = PredictPlan::kRow;
  } else if (forced.empty() || forced == "tile") {
    plan.kernel = PredictPlan::kTileWs;
  } else {
    Log::Fatal("FALCATA_DEBUG predict_kernel=%s: expected row or tile", forced.c_str());
  }
  // tiles of 8 rows; 4 when 8 would leave fewer than two blocks per multiprocessor (small batches)
  plan.tile_rows = (num_row + 7) / 8 < 2 * sms ? 4 : 8;
  // wide rows: staging copies every column of a row once, and pays when the walks read more features than that
  // (about eight per tree); then as many rows (8, 4, 2 or 1) as fit in 48 KiB, two blocks per multiprocessor
  // (5000 columns: 2.4x the 8-row tile read from global memory; a 559-tree model on 6287 columns gains nothing)
  if (TileWsSharedBytes(plan, num_col, elem, K) > (size_t(48) << 10) &&
      static_cast<int64_t>(num_tree) * 8 >= num_col) {
    while (plan.tile_rows > 1 && TileWsSharedBytes(plan, num_col, elem, K) > (size_t(48) << 10)) {
      plan.tile_rows /= 2;
    }
  }
  if (debug.predict_ilp > 0) {
    plan.ilp = debug.predict_ilp;
  }
  if (debug.predict_tile_rows > 0) {
    plan.tile_rows = std::min(debug.predict_tile_rows, 256);
  }
  if (debug.predict_tile_trees > 0) {
    plan.tile_trees = std::min(debug.predict_tile_trees, 4096);
  }
  const size_t budget = std::min<size_t>(LimitsOf(device).max_shared, size_t(96) << 10);
  plan.stage_rows = true;
  plan.stage_rows = TileWsSharedBytes(plan, num_col, elem, K) <= budget;
  while (TileWsSharedBytes(plan, num_col, elem, K) > budget && plan.tile_trees > 16) {
    plan.tile_trees /= 2;
  }
  if (plan.kernel == PredictPlan::kTileWs && TileWsSharedBytes(plan, num_col, elem, K) > budget) {
    // so many outputs that one block cannot hold its rows' running sums: the row kernel needs no shared memory
    if (!forced.empty()) {
      Log::Fatal("FALCATA_DEBUG predict_kernel=tile: %d outputs per row do not fit in a block's shared memory", K);
    }
    plan.kernel = PredictPlan::kRow;
  }
  return plan;
}

/*! \brief what a launch needs beyond the rows: the contribution kernel's path buffers and the model's width */
struct RunContext {
  int num_feature = 0;
  int device = 0;
  int max_depth = 0;
  bool deterministic = false;
  void* contrib_scratch = nullptr;
  int64_t contrib_rows = 0;   // rows the scratch holds path buffers for
  CUDAPredictor::Scratch* scratch = nullptr;   // the default mode's span sums
  int lane = 0;                                 // the stream lane of a chunked host input
};

template <typename T>
void RunRows(const CUDAPackedForest& forest, const Boosting::TreeRangeView& view, int tree_begin,
             int num_iteration, int predict_type, const DeviceRows<T>& rows, void* out, cudaStream_t stream,
             const RunContext& ctx) {
  const DeviceForestView<T> fv = forest.View<T>();
  if (predict_type == C_API_PREDICT_LEAF_INDEX) {
    LaunchPredictLeafKernel<T, T>(fv, rows, tree_begin, num_iteration * view.num_tree_per_iteration,
                                  reinterpret_cast<int32_t*>(out), stream);
  } else if (predict_type == C_API_PREDICT_CONTRIB) {
    const int64_t width = static_cast<int64_t>(view.num_tree_per_iteration) * (ctx.num_feature + 1);
    for (int64_t r0 = 0; r0 < rows.num_row; r0 += ctx.contrib_rows) {
      DeviceRows<T> part = rows;
      part.data = rows.data + r0 * rows.row_stride;
      part.num_row = std::min<int64_t>(ctx.contrib_rows, rows.num_row - r0);
      LaunchPredictContribKernel<T, T>(fv, part, tree_begin, num_iteration, view.num_tree_per_iteration,
                                       ctx.num_feature, reinterpret_cast<double*>(out) + r0 * width,
                                       ctx.contrib_scratch, stream);
    }
  } else if (ctx.deterministic || view.vector_leaf) {
    // vector leaves always add in tree order (one thread per row)
    const PredictPlan plan = ChoosePlan(rows.num_row, rows.num_col, view.num_tree_per_iteration, sizeof(T),
                                        num_iteration * view.num_tree_per_iteration);
    LaunchPredictRawKernel<T, T>(fv, rows, tree_begin, num_iteration, view.num_tree_per_iteration, view.num_output,
                                 view.vector_leaf, plan, reinterpret_cast<double*>(out), stream);
  } else {
    const PairwisePlan plan = ChoosePairwisePlan(ctx.device, rows.num_row, rows.num_col,
                                                 view.num_tree_per_iteration, num_iteration, sizeof(T),
                                                 ctx.max_depth);
    double* partials = nullptr;
    if (plan.num_spans > 1) {
      partials = reinterpret_cast<double*>(ctx.scratch->Get(static_cast<size_t>(plan.num_spans) * rows.num_row *
                                                            view.num_tree_per_iteration * sizeof(double),
                                                            ctx.lane, stream));
    }
    LaunchPredictPairwiseKernel<T, T>(fv, rows, tree_begin, num_iteration, view.num_tree_per_iteration, plan,
                                      reinterpret_cast<double*>(out), partials, stream);
  }
}

void ConvertOnHost(const Boosting* boosting, const Boosting::PredictRange& range, double* out, int64_t num_row,
                   int num_output) {
  #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (int64_t i = 0; i < num_row; ++i) {
    boosting->ConvertRawToPrediction(out + i * num_output, range);
  }
}

}  // namespace

void CUDAPredictor::Predict(const Boosting* boosting, const Boosting::PredictRange& range,
                            const CUDAPredictRequest& req, CUDAPredictStats* stats) {
  if (req.data_type != C_API_DTYPE_FLOAT32 && req.data_type != C_API_DTYPE_FLOAT64) {
    Log::Fatal("CUDA predictor: the input must be float32 or float64");
  }
  const bool contrib = req.predict_type == C_API_PREDICT_CONTRIB;
  if (contrib && boosting->IsLinear()) {
    Log::Fatal("Predicting SHAP feature contributions is not implemented for linear trees.");
  }
  const bool is_double = req.data_type == C_API_DTYPE_FLOAT64;
  const size_t elem = is_double ? sizeof(double) : sizeof(float);
  int device_id = req.device_id;
  if (req.on_device) {
    cudaPointerAttributes attr;
    CUDASUCCESS_OR_FATAL(cudaPointerGetAttributes(&attr, req.data));
    if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
      Log::Fatal("CUDA predictor: the input is not in device memory");
    }
    device_id = attr.device;
  }
  DeviceGuard guard(device_id);
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&device_id));
  const Boosting::TreeRangeView view = boosting->RangeShape(range);
  const int tree_begin = range.start_iteration * view.num_tree_per_iteration;
  const int num_output = req.predict_type == C_API_PREDICT_LEAF_INDEX
                             ? range.num_iteration * view.num_tree_per_iteration
                             : contrib ? view.num_tree_per_iteration * (view.num_feature + 1) : view.num_output;
  const size_t out_elem = req.predict_type == C_API_PREDICT_LEAF_INDEX ? sizeof(int32_t) : sizeof(double);
  const int expected = boosting->NumPredictOneRow(range.start_iteration, range.num_iteration,
                                                  req.predict_type == C_API_PREDICT_LEAF_INDEX, contrib);
  if (expected != num_output) {
    Log::Fatal("CUDA predictor: %d values per row, the model's prediction has %d", num_output, expected);
  }
  const bool convert = req.predict_type == C_API_PREDICT_NORMAL && boosting->HasOutputConversion();

  std::shared_ptr<const CUDAPackedForest> forest = GetForest(boosting, device_id, is_double);
  stats->forest_bytes = forest->Bytes();
  const int range_trees = range.num_iteration * view.num_tree_per_iteration;
  if (forest->num_trees() < tree_begin + range_trees) {
    Log::Fatal("CUDA predictor: the packed model has %d trees, the prediction needs %d", forest->num_trees(),
               tree_begin + range_trees);
  }
  if (req.num_row == 0) {
    return;
  }
  RunContext ctx;
  ctx.num_feature = view.num_feature;
  ctx.device = device_id;
  ctx.max_depth = forest->max_depth();
  ctx.deterministic = req.deterministic;
  // released (with the scratch's lock) when the call returns, after its streams are synchronized
  Scratch scratch(this, device_id);
  ctx.scratch = &scratch;
  CUDAVector<char> contrib_scratch;
  if (contrib) {
    // path buffers for the (row, output) threads of one launch, at most a quarter of the free memory (and 1 GiB)
    size_t free_bytes = 0, total_bytes = 0;
    CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t per_row = ContribScratchBytes(forest->max_depth()) * view.num_tree_per_iteration;
    const size_t budget = std::min<size_t>(std::max<size_t>(free_bytes / 4, size_t(64) << 20), size_t(1) << 30);
    ctx.contrib_rows = std::max<int64_t>(1, std::min<int64_t>(req.num_row, static_cast<int64_t>(budget / per_row)));
    contrib_scratch.Resize(static_cast<size_t>(ctx.contrib_rows) * per_row);
    ctx.contrib_scratch = contrib_scratch.RawData();
    stats->scratch_bytes += VectorBytes(contrib_scratch);
  }

  if (req.on_device) {
    cudaStream_t stream = req.stream;
    if (is_double) {
      DeviceRows<double> rows{reinterpret_cast<const double*>(req.data), req.num_row, req.num_col, req.row_stride,
                              req.col_stride};
      RunRows(*forest, view, tree_begin, range.num_iteration, req.predict_type, rows, req.out, stream, ctx);
    } else {
      DeviceRows<float> rows{reinterpret_cast<const float*>(req.data), req.num_row, req.num_col, req.row_stride,
                             req.col_stride};
      RunRows(*forest, view, tree_begin, range.num_iteration, req.predict_type, rows, req.out, stream, ctx);
    }
    if (convert) {
      // the transform is the host's ConvertRawToPrediction, so the normal prediction equals the CPU predictor's
      const size_t n = static_cast<size_t>(req.num_row) * num_output;
      std::vector<double> host(n);
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(host.data(), req.out, n * sizeof(double), cudaMemcpyDeviceToHost, stream));
      CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(stream));
      ConvertOnHost(boosting, range, host.data(), req.num_row, num_output);
      CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(req.out, host.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
    }
    CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(stream));
    stats->num_chunks = 1;
    stats->chunk_rows = req.num_row;
    return;
  }

  // host input: rows go to the device in chunks; each chunk's input and output buffers are the call's only
  // device memory beside the cached model
  const bool row_major = req.col_stride == 1 && req.row_stride == req.num_col;
  const bool col_major = req.row_stride == 1 && req.col_stride == req.num_row;
  if (!row_major && !col_major) {
    Log::Fatal("CUDA predictor: a host input must be C- or F-contiguous");
  }
  size_t free_bytes = 0, total_bytes = 0;
  CUDASUCCESS_OR_FATAL(cudaMemGetInfo(&free_bytes, &total_bytes));
  const size_t row_bytes = std::max<size_t>(1, static_cast<size_t>(req.num_col)) * elem + num_output * out_elem;
  // two chunks in flight: the next chunk is copied while the kernel runs on the current one
  const size_t budget = std::min<size_t>(std::max<size_t>(free_bytes / 8, size_t(32) << 20), size_t(512) << 20);
  int64_t chunk_rows = std::max<int64_t>(1, std::min<int64_t>(req.num_row, budget / row_bytes));
  if (FalcataDebug().predict_chunk_rows > 0) {
    chunk_rows = std::min<int64_t>(req.num_row, FalcataDebug().predict_chunk_rows);
  }
  const int64_t num_chunks = (req.num_row + chunk_rows - 1) / chunk_rows;
  const int lanes = num_chunks > 1 ? 2 : 1;
  std::vector<CUDAVector<char>> in_buf(lanes), out_buf(lanes);
  std::vector<OwnedStream> streams(lanes);
  for (int l = 0; l < lanes; ++l) {
    in_buf[l].Resize(static_cast<size_t>(chunk_rows) * req.num_col * elem + 1);
    out_buf[l].Resize(static_cast<size_t>(chunk_rows) * num_output * out_elem + 1);
    stats->scratch_bytes += VectorBytes(in_buf[l]) + VectorBytes(out_buf[l]);
  }
  // the second lane's contribution kernels run beside the first's: their own path buffers
  RunContext lane_ctx[2] = {ctx, ctx};
  lane_ctx[1].lane = 1;
  CUDAVector<char> contrib_scratch_2;
  if (contrib && lanes == 2) {
    contrib_scratch_2.Resize(contrib_scratch.Size());
    lane_ctx[1].contrib_scratch = contrib_scratch_2.RawData();
    stats->scratch_bytes += VectorBytes(contrib_scratch_2);
  }
  stats->num_chunks = num_chunks;
  stats->chunk_rows = chunk_rows;
  const char* src = reinterpret_cast<const char*>(req.data);
  char* dst = reinterpret_cast<char*>(req.out);
  auto copy_out = [&](int64_t c) {
    const int l = static_cast<int>(c % lanes);
    const int64_t r0 = c * chunk_rows;
    const int64_t nr = std::min<int64_t>(chunk_rows, req.num_row - r0);
    CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(dst + r0 * num_output * out_elem, out_buf[l].RawData(),
                                         static_cast<size_t>(nr) * num_output * out_elem, cudaMemcpyDeviceToHost,
                                         streams[l].get()));
  };
  for (int64_t c = 0; c < num_chunks; ++c) {
    const int l = static_cast<int>(c % lanes);
    cudaStream_t stream = streams[l].get();
    const int64_t r0 = c * chunk_rows;
    const int64_t nr = std::min<int64_t>(chunk_rows, req.num_row - r0);
    if (req.num_col > 0) {
      // the chunk's rows as one contiguous device block, in the input's layout
      // pageable copies (a pinned staging copy measured slower: the host memcpy into it costs more than it saves)
      if (row_major) {
        CUDASUCCESS_OR_FATAL(cudaMemcpyAsync(in_buf[l].RawData(), src + r0 * req.num_col * elem,
                                             static_cast<size_t>(nr) * req.num_col * elem, cudaMemcpyHostToDevice,
                                             stream));
      } else {
        CUDASUCCESS_OR_FATAL(cudaMemcpy2DAsync(in_buf[l].RawData(), static_cast<size_t>(nr) * elem, src + r0 * elem,
                                               static_cast<size_t>(req.num_row) * elem,
                                               static_cast<size_t>(nr) * elem, req.num_col, cudaMemcpyHostToDevice,
                                               stream));
      }
    }
    const int64_t rs = row_major ? req.num_col : 1;
    const int64_t cs = row_major ? 1 : nr;
    if (is_double) {
      DeviceRows<double> rows{reinterpret_cast<const double*>(in_buf[l].RawData()), nr, req.num_col, rs, cs};
      RunRows(*forest, view, tree_begin, range.num_iteration, req.predict_type, rows, out_buf[l].RawData(), stream,
              lane_ctx[l]);
    } else {
      DeviceRows<float> rows{reinterpret_cast<const float*>(in_buf[l].RawData()), nr, req.num_col, rs, cs};
      RunRows(*forest, view, tree_begin, range.num_iteration, req.predict_type, rows, out_buf[l].RawData(), stream,
              lane_ctx[l]);
    }
    // the previous chunk's result is read back only now, so its kernel overlapped this chunk's copy
    if (c > 0) {
      copy_out(c - 1);
    }
  }
  copy_out(num_chunks - 1);
  for (auto& st : streams) {
    CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(st.get()));
  }
  if (convert) {
    ConvertOnHost(boosting, range, reinterpret_cast<double*>(req.out), req.num_row, num_output);
  }
}

}  // namespace Falcata

#endif  // USE_CUDA
