/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 *
 * FALB -- the Falcata binary model format. Container, dtype tags and the
 * bounds-checked cursors every read goes through.
 *
 * Design notes live in docs/design/binary-model-format-plan.md. Two properties
 * this header exists to enforce:
 *
 *   1. READS ARE BOUNDS-CHECKED. A model file is untrusted input -- it arrives
 *      by download or email like any other artifact -- so ByteReader validates
 *      every offset and length against the real buffer size BEFORE reading.
 *      There is no unchecked accessor; malformed input yields a clean
 *      Log::Fatal, never UB.
 *   2. ARRAYS CARRY DTYPE TAGS, not boolean flags. Widening an array later
 *      (u16 -> u32 features, f64 -> f16 leaves) is additive: an old reader
 *      meets an unknown dtype and fails loudly instead of misreading bytes.
 */
#ifndef FALCATA_BOOSTING_FALB_H_
#define FALCATA_BOOSTING_FALB_H_

#include <Falcata/utils/log.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace Falcata {
namespace FALB {

constexpr char kMagic[4] = {'F', 'A', 'L', 'B'};
constexpr uint32_t kVersion = 1;

/*! \brief required-capability bits: a reader MUST fail on any bit it does not
 *  know, so a future writer can never be silently misparsed. */
enum Flags : uint64_t {
  kFlagNone = 0,
  kFlagHasStats = 1ULL << 0,        // structural stats section present
  kFlagHasDiagnostics = 1ULL << 1,  // training diagnostics section present
  kFlagHasCategorical = 1ULL << 2,
  kFlagHasLinearTrees = 1ULL << 3,  // RESERVED: v1 writes never set it, v1
                                    // readers reject it (roadmap item)
  // every bit this build understands; anything else is "required unknown"
  kFlagKnownMask = kFlagHasStats | kFlagHasDiagnostics | kFlagHasCategorical,
};

enum SectionId : uint32_t {
  kSecMeta = 1,          // model metadata block (text form, verbatim)
  kSecTreeIndex = 2,     // fixed-width per-tree record -> O(1) tree access
  kSecSplitFeature = 3,
  kSecThresholdIdx = 4,  // index into the per-feature threshold dictionary
  kSecThresholdDictOffsets = 5,
  kSecThresholdDictValues = 6,
  kSecDecisionType = 7,
  kSecLeftChild = 8,
  kSecRightChild = 9,
  kSecLeafValue = 10,
  kSecCatBoundaries = 11,
  kSecCatThreshold = 12,
  kSecStats = 13,
  kSecDiagnostics = 14,
};

enum DType : uint32_t {
  kDTypeBytes = 0,
  kDTypeU8 = 1,
  kDTypeU16 = 2,
  kDTypeU32 = 3,
  kDTypeI8 = 4,
  kDTypeI16 = 5,
  kDTypeI32 = 6,
  kDTypeF32 = 7,
  kDTypeF64 = 8,
  kDTypeU64 = 9,
};

enum Codec : uint32_t {
  kCodecRaw = 0,   // mmap-able, O(1) tree access
  kCodecZstd = 1,  // RESERVED: never written, readers reject it
  kCodecZlib = 2,  // default: space is the point for an archived model
};

struct SectionEntry {
  uint32_t id;
  uint32_t dtype;
  uint32_t codec;
  /*! \brief byte-plane shuffle item size (0/1 = none). Splitting an array of
   *  N-byte items into N planes of like-significance bytes makes the
   *  high-order bytes near-constant, which is what the entropy coder can
   *  actually exploit: measured on a real 45k-tree model it takes the whole
   *  core file from 49.5 MB to 45.6 MB. */
  uint32_t shuffle;
  uint64_t offset;      // from file start
  uint64_t stored_len;  // bytes in file
  uint64_t raw_len;     // bytes after decode (== stored_len for kCodecRaw)
};

/*! \brief per-tree fixed-width record; keeping it fixed width is what makes
 *  start_iteration/num_iteration slicing and mmap access O(1). */
struct TreeIndexRecord {
  uint32_t num_leaves;
  uint32_t num_cat;
  int32_t max_depth;
  uint32_t leaf_dim;    // RESERVED for vector-leaf multi-target; v1 writes 1
  uint64_t node_offset;  // into the per-node arrays
  uint64_t leaf_offset;  // into the per-leaf arrays
  uint64_t cat_boundary_offset;
  uint64_t cat_threshold_offset;
  double shrinkage;
};

/*! \brief append-only little-endian writer. */
class ByteWriter {
 public:
  void Align(size_t alignment) {
    while (buf_.size() % alignment != 0) buf_.push_back(0);
  }
  size_t size() const { return buf_.size(); }
  const std::vector<char>& buffer() const { return buf_; }
  std::vector<char>&& release() { return std::move(buf_); }

  template <typename T>
  void Write(const T& v) {
    const char* p = reinterpret_cast<const char*>(&v);
    buf_.insert(buf_.end(), p, p + sizeof(T));
  }

  template <typename T>
  void WriteArray(const T* data, size_t count) {
    if (count == 0) return;
    const char* p = reinterpret_cast<const char*>(data);
    buf_.insert(buf_.end(), p, p + sizeof(T) * count);
  }

  void WriteBytes(const void* data, size_t len) {
    if (len == 0) return;
    const char* p = static_cast<const char*>(data);
    buf_.insert(buf_.end(), p, p + len);
  }

  /*! \brief patch a previously written fixed-width field (section table). */
  template <typename T>
  void PatchAt(size_t offset, const T& v) {
    CHECK_LE(offset + sizeof(T), buf_.size());
    std::memcpy(buf_.data() + offset, &v, sizeof(T));
  }

 private:
  std::vector<char> buf_;
};

/*! \brief bounds-checked reader over an untrusted buffer.
 *
 *  Every accessor validates against the real buffer length first. `where` is
 *  carried purely so a malformed file names the field it died on. */
class ByteReader {
 public:
  ByteReader(const char* data, size_t len) : data_(data), len_(len) {}

  bool InRange(uint64_t offset, uint64_t count, size_t elem_size) const {
    if (elem_size != 0 && count > (UINT64_MAX / elem_size)) return false;
    const uint64_t bytes = count * elem_size;
    if (offset > len_) return false;
    return bytes <= static_cast<uint64_t>(len_) - offset;
  }

  void Require(uint64_t offset, uint64_t count, size_t elem_size,
               const char* where) const {
    if (!InRange(offset, count, elem_size)) {
      Log::Fatal(
          "FALB: corrupt model -- %s claims %llu elements of %llu bytes at "
          "offset %llu, past the end of a %llu-byte buffer",
          where, static_cast<unsigned long long>(count),
          static_cast<unsigned long long>(elem_size),
          static_cast<unsigned long long>(offset),
          static_cast<unsigned long long>(len_));
    }
  }

  template <typename T>
  T Read(uint64_t offset, const char* where) const {
    Require(offset, 1, sizeof(T), where);
    T v;
    std::memcpy(&v, data_ + offset, sizeof(T));
    return v;
  }

  /*! \brief typed view into the buffer; the data is NOT copied, so the caller
   *  must not outlive the buffer (loads copy into the Tree immediately). */
  template <typename T>
  const T* View(uint64_t offset, uint64_t count, const char* where) const {
    Require(offset, count, sizeof(T), where);
    return reinterpret_cast<const T*>(data_ + offset);
  }

  const char* data() const { return data_; }
  size_t size() const { return len_; }

 private:
  const char* data_;
  size_t len_;
};

inline const char* DTypeName(uint32_t dtype) {
  switch (dtype) {
    case kDTypeBytes: return "bytes";
    case kDTypeU8: return "u8";
    case kDTypeU16: return "u16";
    case kDTypeU32: return "u32";
    case kDTypeI8: return "i8";
    case kDTypeI16: return "i16";
    case kDTypeI32: return "i32";
    case kDTypeF32: return "f32";
    case kDTypeF64: return "f64";
    case kDTypeU64: return "u64";
    default: return "<unknown>";
  }
}

/*! \brief widest-fitting unsigned dtype for a value range (dictionary indices,
 *  child pointers, feature ids). */
inline uint32_t NarrowestUnsigned(uint64_t max_value) {
  if (max_value <= UINT8_MAX) return kDTypeU8;
  if (max_value <= UINT16_MAX) return kDTypeU16;
  if (max_value <= UINT32_MAX) return kDTypeU32;
  return kDTypeU64;
}

/*! \brief split `count` items of `item_size` bytes into byte planes. */
inline std::vector<char> ShuffleBytes(const char* src, size_t count, size_t item_size) {
  std::vector<char> out(count * item_size);
  for (size_t b = 0; b < item_size; ++b) {
    char* dst = out.data() + b * count;
    for (size_t i = 0; i < count; ++i) dst[i] = src[i * item_size + b];
  }
  return out;
}

/*! \brief inverse of ShuffleBytes. */
inline void UnshuffleBytes(const char* src, size_t count, size_t item_size, char* dst) {
  for (size_t b = 0; b < item_size; ++b) {
    const char* plane = src + b * count;
    for (size_t i = 0; i < count; ++i) dst[i * item_size + b] = plane[i];
  }
}

struct TreeIO;  // friend of Tree; defined in gbdt_model_binary.cpp

}  // namespace FALB
}  // namespace Falcata

#endif  // FALCATA_BOOSTING_FALB_H_
