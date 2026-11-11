/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include <gtest/gtest.h>
#include <Falcata/utils/random.h>

#include <Falcata/cuda/cuda_random.hpp>

// rank_xendcg draws every item's uniform on the device as
// FloatFromState(AdvanceState(query_state, t + 1)); that must be the (t + 1)-th
// NextFloat of the CPU generator RankXENDCG uses for the query, and the query's
// next state AdvanceState(state, count) the CPU generator's after count draws.
TEST(CUDARandom, AdvanceStateMatchesCPURandomSequence) {
  for (const int seed : {0, 1, 7, 123456789, -5, 2147483647}) {
    Falcata::Random cpu(seed);
    const unsigned int state = static_cast<unsigned int>(seed);
    for (unsigned int t = 0; t < 70000; ++t) {
      const float expected = cpu.NextFloat();
      ASSERT_EQ(Falcata::CUDARandom::FloatFromState(Falcata::CUDARandom::AdvanceState(state, t + 1)), expected)
          << "seed " << seed << " draw " << t;
    }
    // the state after 70000 draws continues the CPU sequence
    const unsigned int next = Falcata::CUDARandom::AdvanceState(state, 70000);
    for (unsigned int t = 0; t < 1000; ++t) {
      ASSERT_EQ(Falcata::CUDARandom::FloatFromState(Falcata::CUDARandom::AdvanceState(next, t + 1)), cpu.NextFloat());
    }
  }
  EXPECT_EQ(Falcata::CUDARandom::AdvanceState(12345u, 0), 12345u);
}
