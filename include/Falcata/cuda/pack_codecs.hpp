/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 *
 * \brief Bin-packing codecs for the per-tree compact column view.
 *
 * A codec is a compile-time policy describing how a partition's bin values are
 * laid out inside its packed row bytes. Kernels template on the codec, so
 * extraction compiles to straight-line code with zero dispatch cost; the host
 * picks one codec per feature partition from the partition's maximum bin count
 * and records it in the compact layout.
 *
 * All packing is LOSSLESS (bins round-trip exactly), so every codec is
 * bit-identical BY CONSTRUCTION at the model level -- the gates enforce it.
 *
 * Layout contract shared by all codecs:
 *   - a partition packs its columns into fixed-size words; column j of the
 *     partition lives in word (j / kValuesPerWord), digit (j % kValuesPerWord);
 *   - a row's packed segment is row_words(cols) words = row_bytes(cols) bytes,
 *     and rows are laid out consecutively (row-major within the partition),
 *     matching the historical 4-bit path's byte-aligned rows;
 *   - the per-partition packed byte-width prefix (packed_partition_byte_offsets)
 *     is codec-agnostic: it just accumulates row_bytes per partition.
 *
 * Codecs:
 *   PackNibble4  -- the historical 4-bit path: 2 values / byte, <=16 bins.
 *   PackBit3x32  -- 10 x 3-bit digits per uint32 (2 bits wasted): 2.5 values
 *                   per byte, <=8 bins. Extraction = shift + mask.
 *   PackRadix5x32 -- 13 base-5 digits per uint32 (5^13 < 2^32): 3.25 values
 *                   per byte, <=5 bins. Extraction = umulhi magic division.
 *   PackRadix6x32 -- 12 base-6 digits per uint32 (6^12 < 2^32): 3.0 values
 *                   per byte, <=6 bins. Extraction = umulhi magic division.
 *
 * Magic-division scheme (radix codecs): for each digit position i we store a
 * 64-bit magic M_i and shift s_i with  floor(w / R^i) == (w * M_i) >> s_i  for
 * all w < 2^32 (classic Granlund-Montgomery constants, precomputed exactly for
 * the fixed divisors R^i). digit_i = q_i - R * q_{i+1}. On device the multiply
 * is one __umul64hi-free sequence: (uint64)w * M_i >> s_i with M_i < 2^32
 * where possible; we simply use 64-bit multiply of 32-bit operands (single
 * mul.wide.u32), which every arch does natively.
 */
#ifndef FALCATA_CUDA_PACK_CODECS_HPP_
#define FALCATA_CUDA_PACK_CODECS_HPP_

#ifdef USE_CUDA

#include <cstdint>

namespace Falcata {

/*! \brief runtime tag carried by the compact layout; template dispatch
 *  switches on it once per launch */
enum class PackCodecId : uint8_t {
  kNibble4 = 0,
  kBit3x32 = 1,
  kRadix5x32 = 2,
  kRadix6x32 = 3,
  kRadix7x32 = 4,
};

#ifdef __CUDACC__
#define FALCATA_PACK_FN __host__ __device__ __forceinline__
#else
#define FALCATA_PACK_FN inline
#endif

/*! \brief divisor table entry for radix codecs: floor(w / R^i) =
 *  (uint64_t(w) * mul) >> shift, exact for all w < 2^32 */
struct PackMagic {
  uint64_t mul;
  uint32_t shift;
};

/*! \brief compile-time Granlund-Montgomery magic for a fixed divisor d with
 *  shift = 32 + L (L = ceil(log2 d)) and mul = ceil(2^shift / d). This choice
 *  is EXACT for every 32-bit dividend unconditionally: the ceiling error
 *  err = mul*d - 2^shift is < d <= 2^L, and exactness requires err * 2^32 <=
 *  2^shift = 2^(32+L), i.e. err <= 2^L -- always satisfied. The price is that
 *  mul is up to 33 bits, so evaluation uses a split multiply (see DivPow). */
FALCATA_PACK_FN constexpr PackMagic MakeMagic(uint64_t d) {
  uint32_t L = 0;
  uint64_t p = 1;
  while (p < d) { p <<= 1; ++L; }
  const uint32_t shift = 32 + L;
  // 2^shift <= 2^63 for our divisors (d < 2^31 => L <= 31)
  const uint64_t two_pow = 1ULL << shift;
  const uint64_t mul = (two_pow + d - 1) / d;
  return PackMagic{mul, shift};
}

// ---------------------------------------------------------------------------
// PackRaw8: unpacked byte-per-value (the classic dense layout)
// ---------------------------------------------------------------------------
struct PackRaw8 {
  static constexpr PackCodecId kId = PackCodecId::kNibble4;  // never carried in layouts
  static constexpr int kMaxBins = 256;
  static constexpr int kValuesPerWord = 1;
  static constexpr int kWordBytes = 1;
  static constexpr bool kUsesPackedOffsets = false;
  static constexpr int RowBytes(int cols) { return cols; }
  template <typename T>
  FALCATA_PACK_FN static uint32_t Extract(const T* row_ptr, unsigned int col) {
    return static_cast<uint32_t>(row_ptr[col]);
  }
  struct Cursor { unsigned col; };
  FALCATA_PACK_FN static Cursor Prepare(unsigned col) { return Cursor{col}; }
  template <typename T>
  FALCATA_PACK_FN static uint32_t ExtractAt(const T* row_ptr, const Cursor& c) {
    return static_cast<uint32_t>(row_ptr[c.col]);
  }
};

// ---------------------------------------------------------------------------
// PackNibble4: the historical 4-bit layout (byte-addressed, no word structure)
// ---------------------------------------------------------------------------
struct PackNibble4 {
  static constexpr bool kUsesPackedOffsets = true;
  static constexpr PackCodecId kId = PackCodecId::kNibble4;
  static constexpr int kMaxBins = 16;
  static constexpr int kValuesPerWord = 2;   // per byte
  static constexpr int kWordBytes = 1;

  static constexpr int RowBytes(int cols) { return (cols + 1) >> 1; }

  template <typename T>
  FALCATA_PACK_FN static uint32_t Extract(const T* row_ptr, unsigned int col) {
    // packed layouts are byte-addressed; only ever instantiated live with T=uint8_t
    const uint8_t* p = reinterpret_cast<const uint8_t*>(row_ptr);
    const uint32_t packed = static_cast<uint32_t>(p[col >> 1]);
    return (packed >> ((col & 1u) << 2)) & 0xfu;
  }
  struct Cursor { unsigned byte; uint32_t shift; };
  FALCATA_PACK_FN static Cursor Prepare(unsigned col) {
    return Cursor{col >> 1, (col & 1u) << 2};
  }
  template <typename T>
  FALCATA_PACK_FN static uint32_t ExtractAt(const T* row_ptr, const Cursor& c) {
    return (static_cast<uint32_t>(reinterpret_cast<const uint8_t*>(row_ptr)[c.byte]) >> c.shift) & 0xfu;
  }

  /*! \brief host-side row packer (dst row segment must be zeroed) */
  static void PackRow(const uint8_t* vals, int cols, uint8_t* dst) {
    for (int j = 0; j < cols; ++j) {
      dst[j >> 1] = static_cast<uint8_t>(
        dst[j >> 1] | ((vals[j] & 0xf) << ((j & 1) << 2)));
    }
  }
};

// ---------------------------------------------------------------------------
// PackBit3x32: 10 x 3-bit digits per uint32 (bits 0..29), <=8 bins
// ---------------------------------------------------------------------------
struct PackBit3x32 {
  static constexpr bool kUsesPackedOffsets = true;
  static constexpr PackCodecId kId = PackCodecId::kBit3x32;
  static constexpr int kMaxBins = 8;
  static constexpr int kValuesPerWord = 10;
  static constexpr int kWordBytes = 4;

  static constexpr int RowBytes(int cols) {
    return ((cols + kValuesPerWord - 1) / kValuesPerWord) * kWordBytes;
  }

  template <typename T>
  FALCATA_PACK_FN static uint32_t Extract(const T* row_ptr, unsigned int col) {
    const uint32_t w = reinterpret_cast<const uint32_t*>(row_ptr)[col / kValuesPerWord];
    return (w >> (3u * (col % kValuesPerWord))) & 0x7u;
  }
  struct Cursor { unsigned word; uint32_t shift; };
  FALCATA_PACK_FN static Cursor Prepare(unsigned col) {
    return Cursor{col / kValuesPerWord, 3u * (col % kValuesPerWord)};
  }
  template <typename T>
  FALCATA_PACK_FN static uint32_t ExtractAt(const T* row_ptr, const Cursor& c) {
    return (reinterpret_cast<const uint32_t*>(row_ptr)[c.word] >> c.shift) & 0x7u;
  }

  static void PackRow(const uint8_t* vals, int cols, uint8_t* dst) {
    uint32_t* words = reinterpret_cast<uint32_t*>(dst);
    for (int j = 0; j < cols; ++j) {
      words[j / kValuesPerWord] |=
        (static_cast<uint32_t>(vals[j]) & 0x7u) << (3 * (j % kValuesPerWord));
    }
  }
};

// ---------------------------------------------------------------------------
// Radix codecs: D digits of radix R per uint32 word
// ---------------------------------------------------------------------------
template <uint32_t R, int D, PackCodecId ID>
struct PackRadix32 {
  static constexpr bool kUsesPackedOffsets = true;
  static constexpr PackCodecId kId = ID;
  static constexpr int kMaxBins = static_cast<int>(R);
  static constexpr int kValuesPerWord = D;
  static constexpr int kWordBytes = 4;

  static constexpr int RowBytes(int cols) {
    return ((cols + D - 1) / D) * kWordBytes;
  }

  /*! \brief R^i for i in [0, D] */
  FALCATA_PACK_FN static constexpr uint64_t Pow(int i) {
    uint64_t p = 1;
    for (int k = 0; k < i; ++k) p *= R;
    return p;
  }

  /*! \brief digit i of w: (w / R^i) % R via exact magic division. The two
   *  quotients trick (q_i - R * q_{i+1}) needs two magics; digit D-1 skips the
   *  second division (quotient past the top digit is 0 for packed words). */
  template <typename T>
  FALCATA_PACK_FN static uint32_t Extract(const T* row_ptr, unsigned int col) {
    const uint32_t w = reinterpret_cast<const uint32_t*>(row_ptr)[col / D];
    const int i = static_cast<int>(col % D);
    const uint32_t q_i = DivPow(w, i);
    const uint32_t q_next = (i + 1 < D) ? DivPow(w, i + 1) : 0u;
    return q_i - R * q_next;
  }

  /*! \brief per-column extraction constants, resolved ONCE per thread and
   *  held in registers through the row loop. Extract(row_ptr, col) indexes the
   *  constexpr magic table with a runtime index, which nvcc materializes in
   *  thread-LOCAL memory -- rereading it per row was a ~19x slowdown. */
  struct Cursor {
    unsigned word;      // word index within the row segment
    uint64_t mul_i;     // magic for R^i
    uint32_t sh_i;
    uint64_t mul_n;     // magic for R^(i+1); mul_n == 0 marks the top digit
    uint32_t sh_n;
  };
  FALCATA_PACK_FN static Cursor Prepare(unsigned col) {
    constexpr MagicTable kT = MakeTable();
    const int i = static_cast<int>(col % D);
    Cursor c;
    c.word = col / D;
    c.mul_i = kT.v[i].mul;
    c.sh_i = kT.v[i].shift;
    if (i + 1 < D) {
      c.mul_n = kT.v[i + 1].mul;
      c.sh_n = kT.v[i + 1].shift;
    } else {
      c.mul_n = 0;
      c.sh_n = 0;
    }
    return c;
  }
  template <typename T>
  FALCATA_PACK_FN static uint32_t ExtractAt(const T* row_ptr, const Cursor& c) {
    const uint32_t w = reinterpret_cast<const uint32_t*>(row_ptr)[c.word];
    const uint32_t q_i = DivMagic(w, c.mul_i, c.sh_i);
    const uint32_t q_n = (c.mul_n != 0) ? DivMagic(w, c.mul_n, c.sh_n) : 0u;
    return q_i - R * q_n;
  }
  FALCATA_PACK_FN static uint32_t DivMagic(uint32_t w, uint64_t mul, uint32_t shift) {
    const uint64_t hi_part = static_cast<uint64_t>(w) * (mul >> 32);
    const uint64_t lo_part = (static_cast<uint64_t>(w) * static_cast<uint32_t>(mul)) >> 32;
    return static_cast<uint32_t>((hi_part + lo_part) >> (shift - 32));
  }

  FALCATA_PACK_FN static uint32_t DivPow(uint32_t w, int i) {
    // mul is up to 33 bits: evaluate (w * mul) >> shift exactly via the split
    //   (w*mul) >> 32 == w*(mul>>32) + ((w*(mul & 0xffffffff)) >> 32)
    // (right shifts compose; the discarded low 32 bits cannot carry upward),
    // then shift the remaining (shift - 32). mul>>32 is 0..1, so the first
    // term folds to a predicated add at compile time when i is uniform.
    // function-local constexpr: nvcc materializes it for device code (a
    // static class member constexpr table would be undefined there)
    constexpr MagicTable kT = MakeTable();
    const PackMagic m = kT.v[i];
    const uint64_t hi_part = static_cast<uint64_t>(w) * (m.mul >> 32);
    const uint64_t lo_part =
      (static_cast<uint64_t>(w) * static_cast<uint32_t>(m.mul)) >> 32;
    return static_cast<uint32_t>((hi_part + lo_part) >> (m.shift - 32));
  }

  struct MagicTable {
    // digits 0..D-1 only: the quotient past the top digit is 0 by
    // construction (Extract short-circuits it), and R^D may exceed 2^32
    PackMagic v[D];
  };
  FALCATA_PACK_FN static constexpr MagicTable MakeTable() {
    MagicTable t{};
    for (int i = 0; i < D; ++i) t.v[i] = MakeMagic(Pow(i));
    return t;
  }
  // kMagicsTable.v[0] divides by 1 (mul = 2^32, shift 32) -- trivially exact
  static constexpr MagicTable kMagicsTable = MakeTable();

  static void PackRow(const uint8_t* vals, int cols, uint8_t* dst) {
    uint32_t* words = reinterpret_cast<uint32_t*>(dst);
    for (int j = 0; j < cols; ++j) {
      words[j / D] += static_cast<uint32_t>(vals[j]) * static_cast<uint32_t>(Pow(j % D));
    }
  }
};

using PackRadix5x32 = PackRadix32<5, 13, PackCodecId::kRadix5x32>;
using PackRadix6x32 = PackRadix32<6, 12, PackCodecId::kRadix6x32>;
using PackRadix7x32 = PackRadix32<7, 11, PackCodecId::kRadix7x32>;

static_assert(PackRadix5x32::Pow(13) == 1220703125ULL, "5^13");
static_assert(PackRadix6x32::Pow(12) == 2176782336ULL, "6^12");
static_assert(PackRadix5x32::Pow(13) < (1ULL << 32), "radix5 digits fit uint32");
static_assert(PackRadix6x32::Pow(12) < (1ULL << 32), "radix6 digits fit uint32");
// compile-time roundtrip proof for every digit position of both radices:
// packing digit v at position i then extracting must return v
template <class C>
constexpr bool CodecRoundtripProof() {
  for (int i = 0; i < C::kValuesPerWord; ++i) {
    for (uint32_t v = 0; v < static_cast<uint32_t>(C::kMaxBins); ++v) {
      // word holding v at digit i and (kMaxBins-1) at every other digit
      // (worst-case neighbors for carry/borrow errors)
      uint64_t w = 0;
      for (int k = 0; k < C::kValuesPerWord; ++k) {
        const uint64_t digit = (k == i) ? v : (C::kMaxBins - 1);
        w += digit * C::Pow(k);
      }
      const uint32_t word = static_cast<uint32_t>(w);
      const uint32_t q_i = ((static_cast<uint64_t>(word) * (C::kMagicsTable.v[i].mul >> 32)) +
        ((static_cast<uint64_t>(word) * static_cast<uint32_t>(C::kMagicsTable.v[i].mul)) >> 32))
        >> (C::kMagicsTable.v[i].shift - 32) ;
      const uint32_t q_n = (i + 1 < C::kValuesPerWord) ?
        static_cast<uint32_t>(((static_cast<uint64_t>(word) * (C::kMagicsTable.v[i + 1].mul >> 32)) +
          ((static_cast<uint64_t>(word) * static_cast<uint32_t>(C::kMagicsTable.v[i + 1].mul)) >> 32))
          >> (C::kMagicsTable.v[i + 1].shift - 32)) : 0u;
      if (q_i - static_cast<uint32_t>(C::kMaxBins) * q_n != v) return false;
    }
  }
  return true;
}
static_assert(PackRadix7x32::Pow(11) == 1977326743ULL, "7^11");
static_assert(CodecRoundtripProof<PackRadix5x32>(), "radix5 roundtrip");
static_assert(CodecRoundtripProof<PackRadix6x32>(), "radix6 roundtrip");
static_assert(CodecRoundtripProof<PackRadix7x32>(), "radix7 roundtrip");

/*! \brief bytes per packed row for a codec id (host-side layout math) */
inline int PackRowBytes(PackCodecId id, int cols) {
  switch (id) {
    case PackCodecId::kBit3x32: return PackBit3x32::RowBytes(cols);
    case PackCodecId::kRadix5x32: return PackRadix5x32::RowBytes(cols);
    case PackCodecId::kRadix6x32: return PackRadix6x32::RowBytes(cols);
    case PackCodecId::kRadix7x32: return PackRadix7x32::RowBytes(cols);
    case PackCodecId::kNibble4:
    default: return PackNibble4::RowBytes(cols);
  }
}

/*! \brief packed values per word for a codec id (1 word = kWordBytes) */
inline int PackValuesPerWord(PackCodecId id) {
  switch (id) {
    case PackCodecId::kBit3x32: return PackBit3x32::kValuesPerWord;
    case PackCodecId::kRadix5x32: return PackRadix5x32::kValuesPerWord;
    case PackCodecId::kRadix6x32: return PackRadix6x32::kValuesPerWord;
    case PackCodecId::kRadix7x32: return PackRadix7x32::kValuesPerWord;
    case PackCodecId::kNibble4:
    default: return PackNibble4::kValuesPerWord;
  }
}

/*! \brief max representable bins for a codec id */
inline int PackMaxBins(PackCodecId id) {
  switch (id) {
    case PackCodecId::kBit3x32: return PackBit3x32::kMaxBins;
    case PackCodecId::kRadix5x32: return PackRadix5x32::kMaxBins;
    case PackCodecId::kRadix6x32: return PackRadix6x32::kMaxBins;
    case PackCodecId::kRadix7x32: return PackRadix7x32::kMaxBins;
    case PackCodecId::kNibble4:
    default: return PackNibble4::kMaxBins;
  }
}

#undef FALCATA_PACK_FN

}  // namespace Falcata

#endif  // USE_CUDA
#endif  // FALCATA_CUDA_PACK_CODECS_HPP_
