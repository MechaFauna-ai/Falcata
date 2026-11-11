/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_C_API_CSC_ROW_ITERATOR_HPP_
#define FALCATA_SRC_C_API_CSC_ROW_ITERATOR_HPP_

#include <cstdint>
#include <functional>
#include <utility>

// Row iterator of one column of a CSC matrix (the C API's CSC prediction and
// dataset construction). Get is defined here so the per-(row, column) calls of
// the prediction loop inline.
class CSC_RowIterator {
 public:
  CSC_RowIterator(const void* col_ptr, int col_ptr_type, const int32_t* indices,
                  const void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int col_idx);
  ~CSC_RowIterator() {}
  // return value at idx; ascending order is fastest, an earlier idx rescans the column from its start
  double Get(int idx) {
    if (idx <= prev_idx_) {
      // at or before a stored value already passed: start again from the column's first stored value
      nonzero_idx_ = 0;
      cur_idx_ = -1;
      prev_idx_ = -1;
      cur_val_ = 0.0;
      is_end_ = false;
    }
    while (idx > cur_idx_ && !is_end_) {
      auto ret = iter_fun_(nonzero_idx_);
      if (ret.first < 0) {
        is_end_ = true;
        break;
      }
      prev_idx_ = cur_idx_;
      cur_idx_ = ret.first;
      cur_val_ = ret.second;
      ++nonzero_idx_;
    }
    return idx == cur_idx_ ? cur_val_ : 0.0;
  }
  // return next non-zero pair, if index < 0, means no more data
  std::pair<int, double> NextNonZero();

 private:
  int nonzero_idx_ = 0;
  int cur_idx_ = -1;
  double cur_val_ = 0.0f;
  bool is_end_ = false;
  int prev_idx_ = -1;
  std::function<std::pair<int, double>(int idx)> iter_fun_;
};

#endif  // FALCATA_SRC_C_API_CSC_ROW_ITERATOR_HPP_
