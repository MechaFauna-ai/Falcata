/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_INCLUDE_FALCATA_UTILS_BOUNDED_READER_H_
#define FALCATA_INCLUDE_FALCATA_UTILS_BOUNDED_READER_H_

#include <Falcata/utils/binary_writer.h>
#include <Falcata/utils/log.h>

#include <cstddef>
#include <cstring>

namespace Falcata {

/*!
 * \brief Cursor over a byte range read from an untrusted serialized object
 *        (a binary dataset file, a serialized reference). Every read checks it
 *        stays inside the range and raises a clean error otherwise, so a
 *        truncated or corrupt input cannot make the parser read past it.
 */
class BoundedReader {
 public:
  BoundedReader(const char* data, size_t size, const char* what)
      : pos_(data), end_(data + size), what_(what) {}

  /*! \brief Consume `bytes` and return where they start. */
  const char* Take(size_t bytes) {
    if (bytes > Remaining()) {
      Log::Fatal("%s is truncated or corrupt: needs %zu more bytes, %zu left",
                 what_, bytes, Remaining());
    }
    const char* start = pos_;
    pos_ += bytes;
    return start;
  }

  /*! \brief Consume `count` items of `item_size` bytes, padded to 8 bytes when `aligned`. */
  const char* TakeArray(size_t count, size_t item_size, bool aligned) {
    if (item_size != 0 && count > Remaining() / item_size) {
      Log::Fatal("%s is truncated or corrupt: an array of %zu items does not fit", what_, count);
    }
    const size_t bytes = count * item_size;
    const char* start = Take(bytes);
    if (aligned) {
      Take(BinaryWriter::AlignedSize(bytes) - bytes);
    }
    return start;
  }

  template <typename T>
  T Read() {
    T value;
    std::memcpy(&value, Take(sizeof(T)), sizeof(T));
    return value;
  }

  /*! \brief A value written with AlignedWrite: its size padded to 8 bytes. */
  template <typename T>
  T ReadAligned() {
    T value = Read<T>();
    Take(BinaryWriter::AlignedSize(sizeof(T)) - sizeof(T));
    return value;
  }

  /*! \brief A bool written with AlignedWrite; any byte but 0 or 1 is corrupt. */
  bool ReadAlignedBool() {
    static_assert(sizeof(bool) == 1, "bool is serialized as one byte");
    const unsigned char byte = ReadAligned<unsigned char>();
    if (byte > 1) {
      Log::Fatal("%s is corrupt: invalid boolean byte %u", what_, static_cast<unsigned>(byte));
    }
    return byte == 1;
  }

  size_t Remaining() const { return static_cast<size_t>(end_ - pos_); }
  const char* pos() const { return pos_; }
  const char* what() const { return what_; }

 private:
  const char* pos_;
  const char* end_;
  const char* what_;
};

}  // namespace Falcata

#endif  // FALCATA_INCLUDE_FALCATA_UTILS_BOUNDED_READER_H_
