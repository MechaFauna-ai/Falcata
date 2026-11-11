/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>

#include <Falcata/bin.h>

#include <memory>
#include <string>
#include <vector>

// GetColWiseData(..., std::vector<BinIterator*>*, num_threads) fills the vector
// with the iterators the column needs: none for dense data (read directly), one
// per thread for sparse data. The dense overloads clear the vector first; the
// sparse ones appended, so a caller reusing the vector got stale (and, once
// freed, dangling) iterators in front of the new ones.
TEST(ColWiseData, IteratorVectorHoldsOnlyThisCallsIterators) {
  const int num_threads = 3;
  for (const int num_bin : {16, 200, 1000, 70000}) {  // 4-bit / 8 / 16 / 32-bit storage
    for (const bool sparse : {false, true}) {
      SCOPED_TRACE(std::to_string(num_bin) + (sparse ? " sparse" : " dense"));
      std::unique_ptr<Falcata::Bin> bin(sparse ? Falcata::Bin::CreateSparseBin(10, num_bin)
                                               : Falcata::Bin::CreateDenseBin(10, num_bin));
      bin->FinishLoad();
      std::vector<Falcata::BinIterator*> iterators;
      uint8_t bit_type = 0;
      bool is_sparse = false;
      for (int round = 0; round < 2; ++round) {
        bin->GetColWiseData(&bit_type, &is_sparse, &iterators, num_threads);
        EXPECT_EQ(iterators.size(), sparse ? static_cast<size_t>(num_threads) : 0u);
        EXPECT_EQ(is_sparse, sparse);
        for (auto* it : iterators) {
          delete it;
        }
      }
    }
  }
}
