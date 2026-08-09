/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 *
 * FALB binary model serialization (M1 core: structure, dictionary thresholds,
 * f64 leaves, categoricals; stats/diagnostics behind flags).
 *
 * The model metadata block -- objective, num_class, feature names/infos,
 * average_output, params -- is stored VERBATIM as its existing text form. It
 * is well under 1% of a real model's bytes (the 657 MB numerai reference is
 * ~40M ASCII numbers in the TREES), and reusing the text serializer for it
 * means the "exact predict-semantics set" the roadmap enumerates cannot drift
 * from what the text path produces. All the size and load-time value is in the
 * tree arrays, which are binary structure-of-arrays here.
 */
#include <Falcata/utils/common.h>
#include <Falcata/utils/log.h>

#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "falb.h"
#include "gbdt.h"

namespace Falcata {

namespace FALB {

/*! \brief direct access to Tree's arrays for both serialization directions. */
struct TreeIO {
  static int NumLeaves(const Tree& t) { return t.num_leaves_; }
  static int NumCat(const Tree& t) { return t.num_cat_; }
  static bool IsLinear(const Tree& t) { return t.is_linear_; }
  static double Shrinkage(const Tree& t) { return t.shrinkage_; }
  static int MaxDepth(const Tree& t) { return t.max_depth_; }

  static const std::vector<int>& SplitFeature(const Tree& t) { return t.split_feature_; }
  static const std::vector<double>& Threshold(const Tree& t) { return t.threshold_; }
  static const std::vector<int8_t>& DecisionType(const Tree& t) { return t.decision_type_; }
  static const std::vector<int>& LeftChild(const Tree& t) { return t.left_child_; }
  static const std::vector<int>& RightChild(const Tree& t) { return t.right_child_; }
  static const std::vector<double>& LeafValue(const Tree& t) { return t.leaf_value_; }
  static const std::vector<int>& CatBoundaries(const Tree& t) { return t.cat_boundaries_; }
  static const std::vector<uint32_t>& CatThreshold(const Tree& t) { return t.cat_threshold_; }
  static const std::vector<float>& SplitGain(const Tree& t) { return t.split_gain_; }
  static const std::vector<double>& InternalValue(const Tree& t) { return t.internal_value_; }
  static const std::vector<double>& InternalWeight(const Tree& t) { return t.internal_weight_; }
  static const std::vector<int>& InternalCount(const Tree& t) { return t.internal_count_; }
  static const std::vector<double>& LeafWeight(const Tree& t) { return t.leaf_weight_; }
  static const std::vector<int>& LeafCount(const Tree& t) { return t.leaf_count_; }

  /*! \brief build an empty tree shell sized for `num_leaves`; the loader then
   *  fills the arrays. Mirrors what Tree(const char*) does before parsing. */
  static std::unique_ptr<Tree> MakeShell(int num_leaves, int num_cat, bool is_linear) {
    auto tree = std::unique_ptr<Tree>(new Tree(std::max(num_leaves, 2), false, is_linear));
    tree->num_leaves_ = num_leaves;
    tree->num_cat_ = num_cat;
    return tree;
  }

  static void SetStructure(Tree* t, std::vector<int>&& split_feature,
                           std::vector<double>&& threshold,
                           std::vector<int8_t>&& decision_type,
                           std::vector<int>&& left_child,
                           std::vector<int>&& right_child,
                           std::vector<double>&& leaf_value, double shrinkage,
                           int max_depth) {
    t->split_feature_ = std::move(split_feature);
    t->threshold_ = std::move(threshold);
    t->decision_type_ = std::move(decision_type);
    t->left_child_ = std::move(left_child);
    t->right_child_ = std::move(right_child);
    t->leaf_value_ = std::move(leaf_value);
    t->shrinkage_ = shrinkage;
    t->max_depth_ = max_depth;
  }

  static void SetCategorical(Tree* t, std::vector<int>&& cat_boundaries,
                             std::vector<uint32_t>&& cat_threshold) {
    t->cat_boundaries_ = std::move(cat_boundaries);
    t->cat_threshold_ = std::move(cat_threshold);
    // the *_inner_ arrays are the training-time (inner feature index) mirrors;
    // a deserialized predict-only tree uses the real-feature ones, exactly as
    // the text loader leaves them.
    t->cat_boundaries_inner_ = t->cat_boundaries_;
    t->cat_threshold_inner_ = t->cat_threshold_;
  }

  static void SetStats(Tree* t, std::vector<double>&& leaf_weight,
                       std::vector<int>&& leaf_count,
                       std::vector<double>&& internal_weight,
                       std::vector<int>&& internal_count) {
    t->leaf_weight_ = std::move(leaf_weight);
    t->leaf_count_ = std::move(leaf_count);
    t->internal_weight_ = std::move(internal_weight);
    t->internal_count_ = std::move(internal_count);
  }

  static void SetDiagnostics(Tree* t, std::vector<float>&& split_gain,
                             std::vector<double>&& internal_value) {
    t->split_gain_ = std::move(split_gain);
    t->internal_value_ = std::move(internal_value);
  }

  /*! \brief text-loaded trees leave these sized-but-zero; match that so a
   *  FALB-loaded tree behaves identically when a section was omitted.
   *
   *  leaf_depth_ and leaf_parent_ are deliberately left EMPTY, exactly as the
   *  text constructor leaves them. They are lazily derived: RecomputeMaxDepth()
   *  only calls RecomputeLeafDepths() when leaf_depth_ is empty, so pre-filling
   *  it with zeros makes max_depth_ come out 0. TreeSHAP then sizes its path
   *  buffer as max_depth_ + 1 == 1 and writes a real, deeper path past the end
   *  -- silent heap corruption on any pred_contrib call. max_depth_ is set to
   *  -1 for the same reason: the value is recomputed, never trusted. */
  static void ZeroFillMissing(Tree* t) {
    const int n_node = t->num_leaves_ - 1;
    if (t->split_gain_.empty()) t->split_gain_.resize(n_node, 0.0f);
    if (t->internal_value_.empty()) t->internal_value_.resize(n_node, 0.0);
    if (t->internal_weight_.empty()) t->internal_weight_.resize(n_node, 0.0);
    if (t->internal_count_.empty()) t->internal_count_.resize(n_node, 0);
    if (t->leaf_weight_.empty()) t->leaf_weight_.resize(t->num_leaves_, 0.0);
    if (t->leaf_count_.empty()) t->leaf_count_.resize(t->num_leaves_, 0);
    // The Tree(max_leaves, ...) constructor SIZES leaf_depth_/leaf_parent_
    // (tree.cpp: leaf_depth_.resize(max_leaves_)), but the text loader leaves
    // them empty and everything downstream derives them lazily. Clearing them
    // is what makes RecomputeMaxDepth() actually recompute: it only calls
    // RecomputeLeafDepths() when leaf_depth_ is EMPTY, so a pre-sized array of
    // zeros yields max_depth_ == 0, and TreeSHAP then allocates a path buffer
    // of max_depth_+1 == 1 entries and writes a real, deeper path past its end
    // -- nondeterministic heap corruption on pred_contrib.
    t->leaf_depth_.clear();
    t->leaf_parent_.clear();
    t->max_depth_ = -1;
  }
};

namespace {

/*! \brief per-feature dictionary of the distinct thresholds occurring in the
 *  trees. See docs/design/binary-model-format-plan.md §3.1: bin bounds are NOT
 *  in the model (feature_infos is only [min:max]), so the format carries its
 *  own codebook. Entries are the trees' own doubles copied verbatim, which
 *  makes the encoding exact by construction -- no grid to fall off, hence no
 *  whole-model raw-f64 fallback for refit or text-edited models. */
class ThresholdDictionary {
 public:
  void Build(const std::vector<std::unique_ptr<Tree>>& models, int start,
             int end, int num_features) {
    per_feature_.assign(num_features + 1, {});
    for (int i = start; i < end; ++i) {
      const Tree& t = *models[i];
      const auto& feats = TreeIO::SplitFeature(t);
      const auto& thr = TreeIO::Threshold(t);
      const auto& dtype = TreeIO::DecisionType(t);
      const int n_node = TreeIO::NumLeaves(t) - 1;
      for (int n = 0; n < n_node; ++n) {
        // categorical nodes store a cat_threshold INDEX in threshold_, not a
        // value -- it must never enter the numeric dictionary
        if ((dtype[n] & kCategoricalMask) > 0) continue;
        const int f = feats[n];
        if (f < 0 || f > num_features) continue;
        per_feature_[f].insert(thr[n]);
      }
    }
    offsets_.assign(per_feature_.size() + 1, 0);
    for (size_t f = 0; f < per_feature_.size(); ++f) {
      for (double v : per_feature_[f]) values_.push_back(v);
      offsets_[f + 1] = static_cast<uint32_t>(values_.size());
      max_per_feature_ = std::max<size_t>(max_per_feature_, per_feature_[f].size());
    }
    // index lookup: value -> position within its feature's block
    index_.resize(per_feature_.size());
    for (size_t f = 0; f < per_feature_.size(); ++f) {
      uint32_t k = 0;
      for (double v : per_feature_[f]) index_[f][v] = k++;
    }
  }

  uint32_t IndexOf(int feature, double value) const {
    const auto& m = index_[feature];
    auto it = m.find(value);
    if (it == m.end()) {
      Log::Fatal("FALB: threshold %.17g of feature %d missing from dictionary",
                 value, feature);
    }
    return it->second;
  }

  const std::vector<uint32_t>& offsets() const { return offsets_; }
  const std::vector<double>& values() const { return values_; }
  size_t max_per_feature() const { return max_per_feature_; }

 private:
  // std::map keeps a deterministic (sorted) order, so the same model always
  // produces byte-identical output
  std::vector<std::set<double>> per_feature_;
  std::vector<std::map<double, uint32_t>> index_;
  std::vector<uint32_t> offsets_;
  std::vector<double> values_;
  size_t max_per_feature_ = 0;
};

/*! \brief write `count` values narrowed to `dtype`. */
template <typename SrcT>
void WriteNarrowed(ByteWriter* w, const SrcT* src, size_t count, uint32_t dtype) {
  switch (dtype) {
    case kDTypeU8:
      for (size_t i = 0; i < count; ++i) w->Write(static_cast<uint8_t>(src[i]));
      break;
    case kDTypeU16:
      for (size_t i = 0; i < count; ++i) w->Write(static_cast<uint16_t>(src[i]));
      break;
    case kDTypeU32:
      for (size_t i = 0; i < count; ++i) w->Write(static_cast<uint32_t>(src[i]));
      break;
    case kDTypeI8:
      for (size_t i = 0; i < count; ++i) w->Write(static_cast<int8_t>(src[i]));
      break;
    case kDTypeI16:
      for (size_t i = 0; i < count; ++i) w->Write(static_cast<int16_t>(src[i]));
      break;
    case kDTypeI32:
      for (size_t i = 0; i < count; ++i) w->Write(static_cast<int32_t>(src[i]));
      break;
    default:
      Log::Fatal("FALB: cannot write dtype %s", DTypeName(dtype));
  }
}

/*! \brief read `count` values of `dtype` widening into DstT. */
template <typename DstT>
void ReadWidened(const ByteReader& r, uint64_t offset, size_t count,
                 uint32_t dtype, const char* where, std::vector<DstT>* out) {
  out->resize(count);
  switch (dtype) {
    case kDTypeU8: {
      const uint8_t* p = r.View<uint8_t>(offset, count, where);
      for (size_t i = 0; i < count; ++i) (*out)[i] = static_cast<DstT>(p[i]);
      break;
    }
    case kDTypeU16: {
      const uint16_t* p = r.View<uint16_t>(offset, count, where);
      for (size_t i = 0; i < count; ++i) (*out)[i] = static_cast<DstT>(p[i]);
      break;
    }
    case kDTypeU32: {
      const uint32_t* p = r.View<uint32_t>(offset, count, where);
      for (size_t i = 0; i < count; ++i) (*out)[i] = static_cast<DstT>(p[i]);
      break;
    }
    case kDTypeI8: {
      const int8_t* p = r.View<int8_t>(offset, count, where);
      for (size_t i = 0; i < count; ++i) (*out)[i] = static_cast<DstT>(p[i]);
      break;
    }
    case kDTypeI16: {
      const int16_t* p = r.View<int16_t>(offset, count, where);
      for (size_t i = 0; i < count; ++i) (*out)[i] = static_cast<DstT>(p[i]);
      break;
    }
    case kDTypeI32: {
      const int32_t* p = r.View<int32_t>(offset, count, where);
      for (size_t i = 0; i < count; ++i) (*out)[i] = static_cast<DstT>(p[i]);
      break;
    }
    default:
      Log::Fatal("FALB: unknown dtype %u for %s -- written by a newer Falcata?",
                 dtype, where);
  }
}

struct SectionBuilder {
  SectionEntry entry;
  std::vector<char> payload;
};

}  // namespace
}  // namespace FALB

using namespace FALB;  // NOLINT(build/namespaces)

std::string GBDT::SaveModelToBinary(int start_iteration, int num_iteration,
                                    int feature_importance_type,
                                    bool with_stats,
                                    bool with_diagnostics,
                                    bool f32_leaves,
                                    int compress_level) const {
  // ---- tree range (mirrors SaveModelToString's slicing) ----
  int num_used_model = static_cast<int>(models_.size());
  const int total_iteration = num_used_model / num_tree_per_iteration_;
  start_iteration = std::max(start_iteration, 0);
  start_iteration = std::min(start_iteration, total_iteration);
  if (num_iteration > 0) {
    const int end_iteration = start_iteration + num_iteration;
    num_used_model = std::min(end_iteration * num_tree_per_iteration_, num_used_model);
  }
  const int start_model = start_iteration * num_tree_per_iteration_;
  const int num_trees = std::max(0, num_used_model - start_model);

  for (int i = start_model; i < num_used_model; ++i) {
    if (TreeIO::IsLinear(*models_[i])) {
      Log::Fatal(
          "FALB v1 cannot store linear trees (flag bit reserved). "
          "Save this model with save_model(..., format='txt') instead.");
    }
  }

  // ---- metadata block: the text header, verbatim, with NO tree bodies ----
  // SaveModelToString writes header, then trees, then importances/params. Ask
  // it for a slice that starts past the last iteration: num_iteration <= 0
  // means "no end bound", and start_model then equals num_used_model, so the
  // tree loop emits nothing and we get exactly the metadata the text path
  // would produce. (Passing num_iteration = 0 instead would embed the WHOLE
  // model -- <= 0 is "to the end", not "none".)
  std::string meta = SaveModelToString(total_iteration, 0, feature_importance_type);

  // ---- per-array sizing ----
  size_t total_cat_thr = 0;
  int max_feature = 0;
  int max_children_mag = 0;
  for (int i = start_model; i < num_used_model; ++i) {
    const Tree& t = *models_[i];
    total_cat_thr += TreeIO::CatThreshold(t).size();
    for (int f : TreeIO::SplitFeature(t)) max_feature = std::max(max_feature, f);
    for (int c : TreeIO::LeftChild(t)) max_children_mag = std::max(max_children_mag, std::abs(c));
    for (int c : TreeIO::RightChild(t)) max_children_mag = std::max(max_children_mag, std::abs(c));
  }

  ThresholdDictionary dict;
  dict.Build(models_, start_model, num_used_model, max_feature_idx_ + 1);

  const uint32_t feat_dtype = NarrowestUnsigned(static_cast<uint64_t>(max_feature));
  const uint32_t thr_idx_dtype =
      NarrowestUnsigned(dict.max_per_feature() == 0 ? 0 : dict.max_per_feature() - 1);
  // children are signed (leaves are encoded as ~leaf), so pick by magnitude:
  // i8 covers the common shallow tree (<=127 leaves) at half the width
  const uint32_t child_dtype = (max_children_mag <= INT8_MAX)
      ? kDTypeI8
      : ((max_children_mag <= INT16_MAX) ? kDTypeI16 : kDTypeI32);

  const bool has_cat = total_cat_thr > 0;

  // ---- build sections ----
  std::vector<SectionBuilder> sections;
  auto add_section = [&sections, compress_level](uint32_t id, uint32_t dtype,
                                                 ByteWriter&& w,
                                                 size_t shuffle_item = 0) {
    SectionBuilder sb;
    sb.entry.id = id;
    sb.entry.dtype = dtype;
    sb.entry.codec = kCodecRaw;
    sb.entry.shuffle = 0;
    sb.payload = std::move(w).release();
    sb.entry.raw_len = sb.payload.size();
    sb.entry.stored_len = sb.payload.size();
    if (compress_level > 0 && !sb.payload.empty()) {
      // byte-plane shuffle first when the array has multi-byte items: it is
      // what lets the entropy coder see the near-constant high-order bytes
      std::vector<char> staged;
      const char* src = sb.payload.data();
      if (shuffle_item > 1 && sb.payload.size() % shuffle_item == 0) {
        staged = ShuffleBytes(sb.payload.data(), sb.payload.size() / shuffle_item,
                              shuffle_item);
        src = staged.data();
        sb.entry.shuffle = static_cast<uint32_t>(shuffle_item);
      }
      uLongf bound = compressBound(static_cast<uLong>(sb.payload.size()));
      std::vector<char> out(bound);
      const int rc = compress2(reinterpret_cast<Bytef*>(out.data()), &bound,
                               reinterpret_cast<const Bytef*>(src),
                               static_cast<uLong>(sb.payload.size()),
                               compress_level);
      if (rc != Z_OK) Log::Fatal("FALB: zlib compression failed (%d)", rc);
      // keep whichever is actually smaller; a section that does not compress
      // stays raw and mmap-able
      if (static_cast<size_t>(bound) < sb.payload.size()) {
        out.resize(bound);
        sb.payload = std::move(out);
        sb.entry.codec = kCodecZlib;
        sb.entry.stored_len = sb.payload.size();
      } else {
        sb.entry.shuffle = 0;
      }
    }
    sections.push_back(std::move(sb));
  };

  {
    ByteWriter w;
    w.WriteBytes(meta.data(), meta.size());
    add_section(kSecMeta, kDTypeBytes, std::move(w), 1);
  }
  {
    ByteWriter w;
    uint64_t node_off = 0, leaf_off = 0, cb_off = 0, ct_off = 0;
    for (int i = start_model; i < num_used_model; ++i) {
      const Tree& t = *models_[i];
      TreeIndexRecord rec;
      rec.num_leaves = static_cast<uint32_t>(TreeIO::NumLeaves(t));
      rec.num_cat = static_cast<uint32_t>(TreeIO::NumCat(t));
      rec.max_depth = TreeIO::MaxDepth(t);
      rec.leaf_dim = 1;  // reserved for vector-leaf multi-target
      rec.node_offset = node_off;
      rec.leaf_offset = leaf_off;
      rec.cat_boundary_offset = cb_off;
      rec.cat_threshold_offset = ct_off;
      rec.shrinkage = TreeIO::Shrinkage(t);
      w.Write(rec);
      node_off += static_cast<uint64_t>(std::max(0, TreeIO::NumLeaves(t) - 1));
      leaf_off += static_cast<uint64_t>(TreeIO::NumLeaves(t));
      cb_off += TreeIO::CatBoundaries(t).size();
      ct_off += TreeIO::CatThreshold(t).size();
    }
    add_section(kSecTreeIndex, kDTypeBytes, std::move(w), sizeof(TreeIndexRecord));
  }
  {
    ByteWriter feat_w, thr_w, dt_w, lc_w, rc_w, lv_w;
    for (int i = start_model; i < num_used_model; ++i) {
      const Tree& t = *models_[i];
      const int n_node = std::max(0, TreeIO::NumLeaves(t) - 1);
      const auto& feats = TreeIO::SplitFeature(t);
      const auto& thr = TreeIO::Threshold(t);
      const auto& dtypes = TreeIO::DecisionType(t);
      WriteNarrowed(&feat_w, feats.data(), n_node, feat_dtype);
      for (int n = 0; n < n_node; ++n) {
        // categorical nodes carry a cat_threshold index in threshold_; store it
        // verbatim rather than through the numeric dictionary
        const uint32_t idx = ((dtypes[n] & kCategoricalMask) > 0)
            ? static_cast<uint32_t>(thr[n])
            : dict.IndexOf(feats[n], thr[n]);
        switch (thr_idx_dtype) {
          case kDTypeU8: thr_w.Write(static_cast<uint8_t>(idx)); break;
          case kDTypeU16: thr_w.Write(static_cast<uint16_t>(idx)); break;
          default: thr_w.Write(static_cast<uint32_t>(idx)); break;
        }
      }
      dt_w.WriteArray(dtypes.data(), n_node);
      WriteNarrowed(&lc_w, TreeIO::LeftChild(t).data(), n_node, child_dtype);
      WriteNarrowed(&rc_w, TreeIO::RightChild(t).data(), n_node, child_dtype);
      if (f32_leaves) {
        // opt-in and documented lossy (~6e-8 relative); f64 stays the default
        // so "predictions are bit-identical" remains the standard guarantee
        const auto& lv = TreeIO::LeafValue(t);
        for (int l = 0; l < TreeIO::NumLeaves(t); ++l) {
          lv_w.Write(static_cast<float>(lv[l]));
        }
      } else {
        lv_w.WriteArray(TreeIO::LeafValue(t).data(), TreeIO::NumLeaves(t));
      }
    }
    auto width = [](uint32_t d) -> size_t {
      switch (d) {
        case kDTypeU8: case kDTypeI8: return 1;
        case kDTypeU16: case kDTypeI16: return 2;
        case kDTypeU32: case kDTypeI32: case kDTypeF32: return 4;
        default: return 8;
      }
    };
    add_section(kSecSplitFeature, feat_dtype, std::move(feat_w), width(feat_dtype));
    add_section(kSecThresholdIdx, thr_idx_dtype, std::move(thr_w), width(thr_idx_dtype));
    add_section(kSecDecisionType, kDTypeI8, std::move(dt_w), 1);
    add_section(kSecLeftChild, child_dtype, std::move(lc_w), width(child_dtype));
    add_section(kSecRightChild, child_dtype, std::move(rc_w), width(child_dtype));
    add_section(kSecLeafValue, f32_leaves ? kDTypeF32 : kDTypeF64, std::move(lv_w),
                f32_leaves ? 4 : 8);
  }
  {
    ByteWriter off_w, val_w;
    off_w.WriteArray(dict.offsets().data(), dict.offsets().size());
    val_w.WriteArray(dict.values().data(), dict.values().size());
    add_section(kSecThresholdDictOffsets, kDTypeU32, std::move(off_w), 4);
    add_section(kSecThresholdDictValues, kDTypeF64, std::move(val_w), 8);
  }
  if (has_cat) {
    ByteWriter cb_w, ct_w;
    for (int i = start_model; i < num_used_model; ++i) {
      const Tree& t = *models_[i];
      WriteNarrowed(&cb_w, TreeIO::CatBoundaries(t).data(),
                    TreeIO::CatBoundaries(t).size(), kDTypeU32);
      ct_w.WriteArray(TreeIO::CatThreshold(t).data(), TreeIO::CatThreshold(t).size());
    }
    add_section(kSecCatBoundaries, kDTypeU32, std::move(cb_w), 4);
    add_section(kSecCatThreshold, kDTypeU32, std::move(ct_w), 4);
  }
  if (with_stats) {
    ByteWriter w;
    for (int i = start_model; i < num_used_model; ++i) {
      const Tree& t = *models_[i];
      w.WriteArray(TreeIO::LeafWeight(t).data(), TreeIO::LeafWeight(t).size());
    }
    for (int i = start_model; i < num_used_model; ++i) {
      const Tree& t = *models_[i];
      WriteNarrowed(&w, TreeIO::LeafCount(t).data(), TreeIO::LeafCount(t).size(), kDTypeI32);
    }
    for (int i = start_model; i < num_used_model; ++i) {
      const Tree& t = *models_[i];
      w.WriteArray(TreeIO::InternalWeight(t).data(), TreeIO::InternalWeight(t).size());
    }
    for (int i = start_model; i < num_used_model; ++i) {
      const Tree& t = *models_[i];
      WriteNarrowed(&w, TreeIO::InternalCount(t).data(), TreeIO::InternalCount(t).size(), kDTypeI32);
    }
    add_section(kSecStats, kDTypeBytes, std::move(w), 8);
  }
  if (with_diagnostics) {
    ByteWriter w;
    for (int i = start_model; i < num_used_model; ++i) {
      const Tree& t = *models_[i];
      w.WriteArray(TreeIO::SplitGain(t).data(), TreeIO::SplitGain(t).size());
    }
    for (int i = start_model; i < num_used_model; ++i) {
      const Tree& t = *models_[i];
      w.WriteArray(TreeIO::InternalValue(t).data(), TreeIO::InternalValue(t).size());
    }
    add_section(kSecDiagnostics, kDTypeBytes, std::move(w), 8);
  }

  // ---- container ----
  uint64_t flags = kFlagNone;
  if (with_stats) flags |= kFlagHasStats;
  if (with_diagnostics) flags |= kFlagHasDiagnostics;
  if (has_cat) flags |= kFlagHasCategorical;

  ByteWriter out;
  out.WriteBytes(kMagic, 4);
  out.Write(static_cast<uint32_t>(kVersion));
  out.Write(flags);
  out.Write(static_cast<uint64_t>(num_trees));
  out.Write(static_cast<uint32_t>(sections.size()));
  out.Write(static_cast<uint32_t>(0));  // pad to 8-byte alignment
  const size_t table_offset = out.size();
  for (size_t i = 0; i < sections.size(); ++i) out.Write(sections[i].entry);

  for (size_t i = 0; i < sections.size(); ++i) {
    out.Align(8);  // raw sections stay 8-byte aligned so they are mmap-friendly
    sections[i].entry.offset = out.size();
    out.WriteBytes(sections[i].payload.data(), sections[i].payload.size());
    out.PatchAt(table_offset + i * sizeof(SectionEntry), sections[i].entry);
  }

  const auto& buf = out.buffer();
  return std::string(buf.data(), buf.size());
}

bool GBDT::LoadModelFromBinary(const char* buffer, size_t len) {
  ByteReader r(buffer, len);
  if (len < 4 || std::memcmp(buffer, kMagic, 4) != 0) {
    Log::Fatal("FALB: not a Falcata binary model (bad magic)");
  }
  const uint32_t version = r.Read<uint32_t>(4, "version");
  if (version != kVersion) {
    Log::Fatal("FALB: model version %u, this build reads version %u", version, kVersion);
  }
  const uint64_t flags = r.Read<uint64_t>(8, "flags");
  if ((flags & ~static_cast<uint64_t>(kFlagKnownMask)) != 0) {
    Log::Fatal(
        "FALB: model requires capabilities this build does not have "
        "(unknown flag bits 0x%llx) -- written by a newer Falcata",
        static_cast<unsigned long long>(flags & ~static_cast<uint64_t>(kFlagKnownMask)));  // NOLINT(runtime/int): %llu format
  }
  const uint64_t num_trees = r.Read<uint64_t>(16, "num_trees");
  const uint32_t num_sections = r.Read<uint32_t>(24, "num_sections");
  const uint64_t table_offset = 32;
  r.Require(table_offset, num_sections, sizeof(SectionEntry), "section table");

  std::map<uint32_t, SectionEntry> sec;
  for (uint32_t i = 0; i < num_sections; ++i) {
    SectionEntry e = r.Read<SectionEntry>(table_offset + i * sizeof(SectionEntry),
                                          "section table entry");
    // validate the section's extent before anything reads through it
    r.Require(e.offset, e.stored_len, 1, "section payload");
    sec[e.id] = e;
  }

  auto need = [&sec](uint32_t id) -> const SectionEntry& {
    auto it = sec.find(id);
    if (it == sec.end()) Log::Fatal("FALB: corrupt model -- missing section %u", id);
    return it->second;
  };

  // Materialize sections: a raw one is viewed in place, a compressed one is
  // decoded into an owned buffer. Either way the caller gets a bounds-checked
  // reader whose offsets are SECTION-relative, so no arithmetic on file
  // offsets can escape a section.
  std::map<uint32_t, std::vector<char>> owned;
  auto section_reader = [&](uint32_t id) -> ByteReader {
    const SectionEntry& e = need(id);
    if (e.codec == kCodecRaw) {
      return ByteReader(r.View<char>(e.offset, e.stored_len, "section"),
                        static_cast<size_t>(e.stored_len));
    }
    if (e.codec != kCodecZlib) {
      Log::Fatal("FALB: section %u uses codec %u, unsupported in this build",
                 e.id, e.codec);
    }
    auto it = owned.find(id);
    if (it == owned.end()) {
      // raw_len comes from the file, so cap it before allocating: a hostile
      // header must not be able to ask for an arbitrary allocation
      if (e.raw_len > (1ULL << 40)) {
        Log::Fatal("FALB: corrupt model -- section %u declares an implausible "
                   "decompressed size of %llu bytes", e.id,
                   static_cast<unsigned long long>(e.raw_len));  // NOLINT(runtime/int): %llu format
      }
      std::vector<char> out(static_cast<size_t>(e.raw_len));
      uLongf out_len = static_cast<uLongf>(e.raw_len);
      const int rc = uncompress(reinterpret_cast<Bytef*>(out.data()), &out_len,
                                reinterpret_cast<const Bytef*>(
                                    r.View<char>(e.offset, e.stored_len, "section")),
                                static_cast<uLong>(e.stored_len));
      if (rc != Z_OK || out_len != e.raw_len) {
        Log::Fatal("FALB: corrupt model -- section %u failed to decompress (%d)",
                   e.id, rc);
      }
      if (e.shuffle > 1) {
        if (out.size() % e.shuffle != 0) {
          Log::Fatal("FALB: corrupt model -- section %u length %llu is not a "
                     "multiple of its shuffle width %u", e.id,
                     static_cast<unsigned long long>(out.size()), e.shuffle);  // NOLINT(runtime/int): %llu format
        }
        std::vector<char> flat(out.size());
        UnshuffleBytes(out.data(), out.size() / e.shuffle, e.shuffle, flat.data());
        out.swap(flat);
      }
      it = owned.emplace(id, std::move(out)).first;
    }
    return ByteReader(it->second.data(), it->second.size());
  };

  // ---- metadata: reuse the text loader so predict semantics cannot drift ----
  ByteReader meta_r = section_reader(kSecMeta);
  std::string meta_text(meta_r.View<char>(0, meta_r.size(), "metadata"),
                        meta_r.size());
  if (!LoadModelFromString(meta_text.c_str(), meta_text.size())) {
    Log::Fatal("FALB: model metadata block failed to parse");
  }
  models_.clear();

  // The metadata is a ZERO-tree text model, and the text loader drops the
  // parameter blob on those: it writes "tree_sizes=" with an empty value,
  // which splits into a single token, so the "tree_sizes" key never registers
  // and the loader takes its no-tree_sizes branch. The params are present in
  // the text either way, so extract them here rather than depend on that edge
  // case -- the roadmap requires the full param blob to survive a FALB round
  // trip, so that continued training / refit from a loaded booster either
  // works or refuses explicitly instead of silently losing its config.
  if (loaded_parameter_.empty()) {
    const std::string begin_tag = "\nparameters:\n";
    const std::string end_tag = "\nend of parameters";
    const size_t b = meta_text.find(begin_tag);
    if (b != std::string::npos) {
      const size_t body = b + begin_tag.size();
      const size_t e = meta_text.find(end_tag, body);
      if (e != std::string::npos) {
        loaded_parameter_ = meta_text.substr(body, e - body);
        // the text writer emits a trailing blank line before the end tag
        while (!loaded_parameter_.empty() && loaded_parameter_.back() == '\n') {
          loaded_parameter_.pop_back();
        }
        loaded_parameter_ += "\n";
      }
    }
  }

  // ---- threshold dictionary ----
  ByteReader doff_r = section_reader(kSecThresholdDictOffsets);
  ByteReader dval_r = section_reader(kSecThresholdDictValues);
  const size_t dict_n_off = doff_r.size() / sizeof(uint32_t);
  const size_t dict_n_val = dval_r.size() / sizeof(double);
  const uint32_t* dict_off = doff_r.View<uint32_t>(0, dict_n_off, "dict offsets");
  const double* dict_val = dval_r.View<double>(0, dict_n_val, "dict values");

  ByteReader idx_r = section_reader(kSecTreeIndex);
  const TreeIndexRecord* recs =
      idx_r.View<TreeIndexRecord>(0, num_trees, "tree index");

  ByteReader sf_r = section_reader(kSecSplitFeature);
  ByteReader ti_r = section_reader(kSecThresholdIdx);
  ByteReader dt_r = section_reader(kSecDecisionType);
  ByteReader lc_r = section_reader(kSecLeftChild);
  ByteReader rc_r = section_reader(kSecRightChild);
  ByteReader lv_r = section_reader(kSecLeafValue);
  const SectionEntry& sf = need(kSecSplitFeature);
  const SectionEntry& ti = need(kSecThresholdIdx);
  const SectionEntry& lc = need(kSecLeftChild);
  const SectionEntry& rc = need(kSecRightChild);
  const SectionEntry& lv = need(kSecLeafValue);

  const bool has_cat = (flags & kFlagHasCategorical) != 0;
  const bool has_stats = (flags & kFlagHasStats) != 0;
  const bool has_diag = (flags & kFlagHasDiagnostics) != 0;

  // total counts, for the stats/diagnostics sub-array offsets
  uint64_t total_nodes = 0, total_leaves = 0;
  for (uint64_t i = 0; i < num_trees; ++i) {
    if (recs[i].num_leaves == 0) Log::Fatal("FALB: corrupt model -- tree %llu has 0 leaves",
                                            static_cast<unsigned long long>(i));  // NOLINT(runtime/int): %llu format
    total_nodes += recs[i].num_leaves - 1;
    total_leaves += recs[i].num_leaves;
  }

  auto elem_size = [](uint32_t d) -> size_t {
    switch (d) {
      case kDTypeU8: case kDTypeI8: return 1;
      case kDTypeU16: case kDTypeI16: return 2;
      case kDTypeU32: case kDTypeI32: case kDTypeF32: return 4;
      default: return 8;
    }
  };

  models_.reserve(static_cast<size_t>(num_trees));
  for (uint64_t i = 0; i < num_trees; ++i) {
    const TreeIndexRecord& rec = recs[i];
    const size_t n_node = rec.num_leaves - 1;
    const size_t n_leaf = rec.num_leaves;
    if (rec.leaf_dim != 1) {
      Log::Fatal("FALB: tree %llu has leaf_dim=%u; this build reads only 1 "
                 "(vector-leaf multi-target models need a newer Falcata)",
                 static_cast<unsigned long long>(i), rec.leaf_dim);  // NOLINT(runtime/int): %llu format
    }

    std::vector<int> split_feature, left_child, right_child;
    std::vector<double> threshold(n_node);
    std::vector<int8_t> decision_type(n_node);
    std::vector<double> leaf_value(n_leaf);

    ReadWidened(sf_r, rec.node_offset * elem_size(sf.dtype), n_node,
                sf.dtype, "split_feature", &split_feature);
    ReadWidened(lc_r, rec.node_offset * elem_size(lc.dtype), n_node,
                lc.dtype, "left_child", &left_child);
    ReadWidened(rc_r, rec.node_offset * elem_size(rc.dtype), n_node,
                rc.dtype, "right_child", &right_child);
    {
      const int8_t* p = dt_r.View<int8_t>(rec.node_offset, n_node, "decision_type");
      std::copy(p, p + n_node, decision_type.begin());
    }
    if (lv.dtype == kDTypeF32) {
      const float* p = lv_r.View<float>(rec.leaf_offset * sizeof(float), n_leaf,
                                        "leaf_value");
      for (size_t l = 0; l < n_leaf; ++l) leaf_value[l] = static_cast<double>(p[l]);
    } else if (lv.dtype == kDTypeF64) {
      const double* p = lv_r.View<double>(rec.leaf_offset * sizeof(double), n_leaf,
                                          "leaf_value");
      std::copy(p, p + n_leaf, leaf_value.begin());
    } else {
      Log::Fatal("FALB: leaf_value has dtype %s, which this build cannot read",
                 DTypeName(lv.dtype));
    }
    {
      std::vector<uint32_t> tidx;
      ReadWidened(ti_r, rec.node_offset * elem_size(ti.dtype), n_node,
                  ti.dtype, "threshold index", &tidx);
      for (size_t n = 0; n < n_node; ++n) {
        if ((decision_type[n] & kCategoricalMask) > 0) {
          threshold[n] = static_cast<double>(tidx[n]);  // cat_threshold index
          continue;
        }
        const int f = split_feature[n];
        if (f < 0 || static_cast<size_t>(f) + 1 >= dict_n_off) {
          Log::Fatal("FALB: corrupt model -- feature %d out of dictionary range", f);
        }
        const uint64_t pos = static_cast<uint64_t>(dict_off[f]) + tidx[n];
        if (pos >= dict_off[f + 1] || pos >= dict_n_val) {
          Log::Fatal("FALB: corrupt model -- threshold index %u out of range for feature %d",
                     tidx[n], f);
        }
        threshold[n] = dict_val[pos];
      }
    }

    auto tree = TreeIO::MakeShell(static_cast<int>(rec.num_leaves),
                                 static_cast<int>(rec.num_cat), false);
    TreeIO::SetStructure(tree.get(), std::move(split_feature), std::move(threshold),
                         std::move(decision_type), std::move(left_child),
                         std::move(right_child), std::move(leaf_value),
                         rec.shrinkage, -1);

    if (has_cat && rec.num_cat > 0) {
      const SectionEntry& cb = need(kSecCatBoundaries);
      ByteReader cb_r = section_reader(kSecCatBoundaries);
      ByteReader ct_r = section_reader(kSecCatThreshold);
      const size_t n_bound = rec.num_cat + 1;
      std::vector<int> cat_boundaries;
      ReadWidened(cb_r, rec.cat_boundary_offset * sizeof(uint32_t), n_bound,
                  cb.dtype, "cat_boundaries", &cat_boundaries);
      const size_t n_thr = static_cast<size_t>(cat_boundaries.back());
      std::vector<uint32_t> cat_threshold(n_thr);
      const uint32_t* p = ct_r.View<uint32_t>(
          rec.cat_threshold_offset * sizeof(uint32_t), n_thr, "cat_threshold");
      std::copy(p, p + n_thr, cat_threshold.begin());
      TreeIO::SetCategorical(tree.get(), std::move(cat_boundaries), std::move(cat_threshold));
    }

    if (has_stats) {
      ByteReader st_r = section_reader(kSecStats);
      const uint64_t lw_base = 0;
      const uint64_t lcnt_base = lw_base + total_leaves * sizeof(double);
      const uint64_t iw_base = lcnt_base + total_leaves * sizeof(int32_t);
      const uint64_t icnt_base = iw_base + total_nodes * sizeof(double);
      std::vector<double> leaf_weight(n_leaf), internal_weight(n_node);
      std::vector<int> leaf_count, internal_count;
      {
        const double* p = st_r.View<double>(lw_base + rec.leaf_offset * sizeof(double),
                                            n_leaf, "leaf_weight");
        std::copy(p, p + n_leaf, leaf_weight.begin());
        const double* q = st_r.View<double>(iw_base + rec.node_offset * sizeof(double),
                                            n_node, "internal_weight");
        std::copy(q, q + n_node, internal_weight.begin());
      }
      ReadWidened(st_r, lcnt_base + rec.leaf_offset * sizeof(int32_t), n_leaf,
                  kDTypeI32, "leaf_count", &leaf_count);
      ReadWidened(st_r, icnt_base + rec.node_offset * sizeof(int32_t), n_node,
                  kDTypeI32, "internal_count", &internal_count);
      TreeIO::SetStats(tree.get(), std::move(leaf_weight), std::move(leaf_count),
                       std::move(internal_weight), std::move(internal_count));
    }

    if (has_diag) {
      ByteReader dg_r = section_reader(kSecDiagnostics);
      const uint64_t iv_base = total_nodes * sizeof(float);
      std::vector<float> split_gain(n_node);
      std::vector<double> internal_value(n_node);
      const float* p = dg_r.View<float>(rec.node_offset * sizeof(float),
                                        n_node, "split_gain");
      std::copy(p, p + n_node, split_gain.begin());
      const double* q = dg_r.View<double>(iv_base + rec.node_offset * sizeof(double),
                                          n_node, "internal_value");
      std::copy(q, q + n_node, internal_value.begin());
      TreeIO::SetDiagnostics(tree.get(), std::move(split_gain), std::move(internal_value));
    }

    TreeIO::ZeroFillMissing(tree.get());
    models_.push_back(std::move(tree));
  }

  num_iteration_for_pred_ = static_cast<int>(models_.size()) / num_tree_per_iteration_;
  num_init_iteration_ = num_iteration_for_pred_;
  iter_ = 0;
  // TreeSHAP divides by each node's data_count, so a model saved without the
  // stats section would produce NaN/Inf contributions rather than fail. Record
  // it and refuse loudly at the point of use.
  falb_without_stats_ = !has_stats;
  return true;
}

}  // namespace Falcata
