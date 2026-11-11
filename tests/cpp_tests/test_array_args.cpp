/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <gtest/gtest.h>
#include <Falcata/meta.h>
#include <Falcata/utils/array_args.h>

#include <algorithm>
#include <random>
#include <vector>

using Falcata::data_size_t;
using Falcata::score_t;
using Falcata::ArrayArgs;


TEST(Partition, JustWorks) {
  std::vector<score_t> gradients({0.5f, 5.0f, 1.0f, 2.0f, 2.0f});
  data_size_t middle_begin, middle_end;

  ArrayArgs<score_t>::Partition(&gradients, 0, static_cast<int>(gradients.size()), &middle_begin, &middle_end);

  EXPECT_EQ(gradients[middle_begin + 1], gradients[middle_end - 1]);
  EXPECT_GT(gradients[0], gradients[middle_begin + 1]);
  EXPECT_GT(gradients[middle_begin + 1], gradients.back());
}

TEST(Partition, PartitionOneElement) {
  std::vector<score_t> gradients({0.5f});
  data_size_t middle_begin, middle_end;
  ArrayArgs<score_t>::Partition(&gradients, 0, static_cast<int>(gradients.size()), &middle_begin, &middle_end);
  EXPECT_EQ(gradients[middle_begin + 1], gradients[middle_end - 1]);
}

TEST(Partition, Empty) {
  std::vector<score_t> gradients;
  data_size_t middle_begin, middle_end;
  ArrayArgs<score_t>::Partition(&gradients, 0, static_cast<int>(gradients.size()), &middle_begin, &middle_end);

  EXPECT_EQ(middle_begin, -1);
  EXPECT_EQ(middle_end, 0);
}

TEST(Partition, AllEqual) {
  std::vector<score_t> gradients({0.5f, 0.5f, 0.5f});
  data_size_t middle_begin, middle_end;
  ArrayArgs<score_t>::Partition(&gradients, 0, static_cast<int>(gradients.size()), &middle_begin, &middle_end);

  EXPECT_EQ(gradients[middle_begin + 1], gradients[middle_end - 1]);
  EXPECT_EQ(middle_begin, -1);
  EXPECT_EQ(middle_end, 3);
}

// ArgMaxAtK leaves the k-th largest value at k, larger-or-equal values before
// it and smaller-or-equal after (PercentileFun and GOSS rely on both sides),
// on any input order. The quickselect it used took the last element as pivot,
// so sorted input (a label column of a sorted file) was quadratic: 1.6 s for
// 100,000 values.
TEST(ArgMaxAtK, SelectsAndPartitionsOnAnyOrder) {
  const int n = 200000;
  std::mt19937 gen(5);
  std::vector<score_t> base(n);
  for (int i = 0; i < n; ++i) {
    base[i] = static_cast<score_t>(gen() % 50000) / 8.0f;
  }
  std::vector<score_t> descending = base;
  std::sort(descending.begin(), descending.end(), [](score_t a, score_t b) { return a > b; });
  std::vector<std::vector<score_t>> inputs = {base, descending, std::vector<score_t>(descending.rbegin(), descending.rend()),
                                              std::vector<score_t>(n, 1.5f)};
  for (size_t t = 0; t < inputs.size(); ++t) {
    for (int k : {0, 1, n / 3, n / 2, n - 2, n - 1}) {
      std::vector<score_t> a = inputs[t];
      const std::vector<score_t> sorted_desc = t == 3 ? a : descending;
      ASSERT_EQ(ArrayArgs<score_t>::ArgMaxAtK(&a, 0, n, k), k);
      EXPECT_EQ(a[k], sorted_desc[k]) << t << " " << k;
      for (int i = 0; i < k; ++i) {
        ASSERT_GE(a[i], a[k]) << t << " " << k << " " << i;
      }
      for (int i = k + 1; i < n; ++i) {
        ASSERT_LE(a[i], a[k]) << t << " " << k << " " << i;
      }
    }
  }
}
