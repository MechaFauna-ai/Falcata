/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_TREELEARNER_QUANTIZED_INDEX_HPP_
#define FALCATA_SRC_TREELEARNER_QUANTIZED_INDEX_HPP_

#include <Falcata/meta.h>

#include <cstddef>
#include <cstdint>

namespace Falcata {

inline constexpr size_t QuantizedPairOffset(data_size_t row) {
  return static_cast<size_t>(row) * 2;
}

inline data_size_t QuantizedRandomIndex(data_size_t row, data_size_t start, data_size_t num_data) {
  return static_cast<data_size_t>((static_cast<int64_t>(row) + start) % num_data);
}

}  // namespace Falcata
#endif  // FALCATA_SRC_TREELEARNER_QUANTIZED_INDEX_HPP_
