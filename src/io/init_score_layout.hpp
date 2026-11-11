/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_IO_INIT_SCORE_LAYOUT_HPP_
#define FALCATA_SRC_IO_INIT_SCORE_LAYOUT_HPP_

#include <Falcata/meta.h>
#include <Falcata/utils/log.h>

#include <cstddef>
#include <limits>

namespace Falcata {

// Init scores are stored class-major: num_class blocks of num_rows doubles. num_class * num_rows passes INT32_MAX on
// large multiclass data, so offsets and sizes are computed in size_t (the int products wrapped).

// Offset of row `row` of class `class_id` in a class-major buffer of `num_rows` rows per class.
inline size_t InitScoreOffset(const int class_id, const data_size_t num_rows, const data_size_t row = 0) {
  return static_cast<size_t>(class_id) * static_cast<size_t>(num_rows) + static_cast<size_t>(row);
}

// Entries of a num_rows x num_class buffer. Metadata::SetInitScore takes a data_size_t length, so a larger buffer is
// refused instead of being described by a wrapped length.
inline size_t InitScoreSize(const data_size_t num_rows, const int num_class) {
  const size_t size = InitScoreOffset(num_class, num_rows);
  if (size > static_cast<size_t>(std::numeric_limits<data_size_t>::max())) {
    Log::Fatal("Initial scores for %d rows x %d classes exceed the supported %d entries", num_rows, num_class,
               std::numeric_limits<data_size_t>::max());
  }
  return size;
}

}  // namespace Falcata

#endif  // FALCATA_SRC_IO_INIT_SCORE_LAYOUT_HPP_
