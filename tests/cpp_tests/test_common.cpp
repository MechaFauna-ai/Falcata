/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/cuda_controller_policy.h>
#include <Falcata/falcata_plan.h>
#include <Falcata/utils/common.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>


// This is a basic test for floating number parsing.
// Most of the test cases come from:
// https://github.com/dmlc/xgboost/blob/master/tests/cpp/common/test_charconv.cc
// https://github.com/Alexhuszagh/rust-lexical/blob/master/data/test-parse-unittests/strtod_tests.toml
class AtofPreciseTest : public testing::Test {
 public:
  struct AtofTestCase {
    const char* data;
    double expected;
  };

  static double TestAtofPrecise(
      const char* data, double expected, bool test_eq = true) {
    double got = 0;
    const char* end = Falcata::Common::AtofPrecise(data, &got);
    EXPECT_TRUE(end != data) << "fail to parse: " << data;
    EXPECT_EQ(*end, '\0') << "not parsing to end: " << data;
    if (test_eq) {
      EXPECT_EQ(expected, got) << "parse string: " << data;
    }
    return got;
  }

  static double Int64Bits2Double(uint64_t v) {
    union {
      uint64_t i;
      double d;
    } conv;
    conv.i = v;
    return conv.d;
  }
};

TEST_F(AtofPreciseTest, Basic) {
  AtofTestCase test_cases[] = {
      { "0", 0.0 },
      { "0E0", 0.0 },
      { "-0E0", 0.0 },
      { "-0", -0.0 },
      { "1", 1.0 },
      { "1E0", 1.0 },
      { "-1", -1.0 },
      { "-1E0", -1.0 },
      { "123456.0", 123456.0 },
      { "432E1", 432E1 },
      { "1.2345678", 1.2345678 },
      { "2.4414062E-4", 2.4414062E-4 },
      { "3.0540412E5", 3.0540412E5 },
      { "3.355445E7", 3.355445E7 },
      { "1.1754944E-38", 1.1754944E-38 },
  };

  for (auto const& test : test_cases) {
    TestAtofPrecise(test.data, test.expected);
  }
}

TEST_F(AtofPreciseTest, CornerCases) {
  AtofTestCase test_cases[] = {
      { "1e-400", 0.0 },
      { "2.4703282292062326e-324", 0.0 },
      { "4.9406564584124654e-324", Int64Bits2Double(0x0000000000000001LU) },
      { "8.44291197326099e-309", Int64Bits2Double(0x0006123400000001LU) },
      // FLT_MAX
      { "3.40282346638528859811704183484516925440e38",
        static_cast<double>(std::numeric_limits<float>::max()) },
      // FLT_MIN
      { "1.1754943508222875079687365372222456778186655567720875215087517062784172594547271728515625e-38",
        static_cast<double>(std::numeric_limits<float>::min()) },
      // DBL_MAX (1 + (1 - 2^-52)) * 2^1023 = (2^53 - 1) * 2^971
      { "17976931348623157081452742373170435679807056752584499659891747680315"
        "72607800285387605895586327668781715404589535143824642343213268894641"
        "82768467546703537516986049910576551282076245490090389328944075868508"
        "45513394230458323690322294816580855933212334827479782620414472316873"
        "8177180919299881250404026184124858368", std::numeric_limits<double>::max() },
      { "1.7976931348623158e+308", std::numeric_limits<double>::max() },
      // 2^971 * (2^53 - 1 + 1/2) : the smallest number resolving to inf
      {"179769313486231580793728971405303415079934132710037826936173778980444"
       "968292764750946649017977587207096330286416692887910946555547851940402"
       "630657488671505820681908902000708383676273854845817711531764475730270"
       "069855571366959622842914819860834936475292719074168444365510704342711"
       "559699508093042880177904174497792", std::numeric_limits<double>::infinity() },
      // Near DBL_MIN
      { "2.2250738585072009e-308", Int64Bits2Double(0x000fffffffffffffLU) },
      // DBL_MIN 2^-1022
      { "2.2250738585072012e-308", std::numeric_limits<double>::min() },
      { "2.2250738585072014e-308", std::numeric_limits<double>::min() },
  };

  for (auto const& test : test_cases) {
    TestAtofPrecise(test.data, test.expected);
  }
}

TEST_F(AtofPreciseTest, ErrorInput) {
  double got = 0;
  ASSERT_THROW(Falcata::Common::AtofPrecise("x1", &got),  std::runtime_error);
}

TEST_F(AtofPreciseTest, NaN) {
  AtofTestCase test_cases[] = {
      { "nan", std::numeric_limits<double>::quiet_NaN() },
      { "NaN", std::numeric_limits<double>::quiet_NaN() },
      { "NAN", std::numeric_limits<double>::quiet_NaN() },
      // The behavior for parsing -nan depends on implementation.
      // Thus we skip binary check for negative nan.
      { "-nan", -std::numeric_limits<double>::quiet_NaN() },
      { "-NaN", -std::numeric_limits<double>::quiet_NaN() },
      { "-NAN", -std::numeric_limits<double>::quiet_NaN() },
  };

  for (auto const& test : test_cases) {
    double got = TestAtofPrecise(test.data, test.expected, false);

    EXPECT_TRUE(std::isnan(got)) << "not parsed as NaN: " << test.data;
    if (!std::signbit(test.expected)) {
      // See comment in test_cases. The payload is implementation-defined too.
      uint64_t bits;
      memcpy(&bits, &got, sizeof(got));
      EXPECT_FALSE(std::signbit(got)) << "parsed NaN is negative: " << test.data;
      EXPECT_NE(bits & (uint64_t{1} << 51), 0u) << "parsed NaN is not quiet: " << test.data;
    }
  }
}

TEST_F(AtofPreciseTest, Inf) {
  AtofTestCase test_cases[] = {
      { "inf", std::numeric_limits<double>::infinity() },
      { "Inf", std::numeric_limits<double>::infinity() },
      { "INF", std::numeric_limits<double>::infinity() },
      { "-inf", -std::numeric_limits<double>::infinity() },
      { "-Inf", -std::numeric_limits<double>::infinity() },
      { "-INF", -std::numeric_limits<double>::infinity() },
  };

  for (auto const& test : test_cases) {
    double got = TestAtofPrecise(test.data, test.expected, false);

    EXPECT_EQ(Falcata::Common::Sign(test.expected), Falcata::Common::Sign(got)) << "sign differs parsing: " << test.data;
    EXPECT_TRUE(std::isinf(got)) << "not parsed as infinite: " << test.data;
    EXPECT_EQ(memcmp(&got, &test.expected, sizeof(test.expected)), 0)
              << "parsed infinite is not the same for every bit: " << test.data;
  }
}

TEST(SortForPair, SortsOnlyTheSuffixFromStart) {
  std::vector<int> keys = {9, 8, 3, 1, 2};
  std::vector<int> values = {90, 80, 30, 10, 20};
  Falcata::Common::SortForPair<int, int>(&keys, &values, 2);
  EXPECT_EQ(keys, (std::vector<int>{9, 8, 1, 2, 3}));
  EXPECT_EQ(values, (std::vector<int>{90, 80, 10, 20, 30}));

  Falcata::Common::SortForPair<int, int>(&keys, &values, 1, true);
  EXPECT_EQ(keys, (std::vector<int>{9, 8, 3, 2, 1}));
  EXPECT_EQ(values, (std::vector<int>{90, 80, 30, 20, 10}));
}

namespace {
Falcata::CUDAControllerRequest ControllerRequestFromSpec(const char* spec) {
  Falcata::Config config;
  config.cuda_plan = spec;
  const auto plan = Falcata::FalcataPlan::FromConfig(config);
  return {plan.device_levels, plan.device_tree, plan.device_levels_explicit, plan.device_tree_explicit};
}
}  // namespace

TEST(CUDAControllerPolicy, ExplicitControllerPairRetainsLegacyMeaning) {
  using Route = Falcata::CUDAControllerRoute;
  struct Case {
    const char* spec;
    bool levels_explicit;
    bool tree_explicit;
    Route route;
  };
  const Case cases[] = {
      {"auto,device_levels:off", true, false, Route::kHost},
      {"auto,device_levels:on", true, false, Route::kPrefix},
      {"auto,device_tree:off", false, true, Route::kPrefix},
      {"auto,device_tree:on", false, true, Route::kWhole},
      {"auto,device_levels:off,device_tree:on", true, true, Route::kWhole},
      {"auto,device_tree:on,device_levels:off", true, true, Route::kWhole},
      {"auto,device_tree:off,device_levels:on", true, true, Route::kPrefix},
      {"auto,device_levels:on,device_tree:off", true, true, Route::kPrefix},
      {"auto,device_levels:off,device_tree:off", true, true, Route::kHost},
      {"auto,device_tree:on,device_tree:off", false, true, Route::kPrefix},
      {"auto,device_tree:off,device_tree:on", false, true, Route::kWhole},
      {"auto,device_levels:on,device_levels:off", true, false, Route::kHost},
      {"auto,device_levels:off,device_levels:on", true, false, Route::kPrefix},
      {"auto,row_batch:off,device_tree:on,device_levels:off,graph_loop:off", true, true, Route::kWhole},
  };
  // Force is intent, not an eligibility verdict. Unknown facts must not erase
  // explicit requests; the learner's existing gates decide actual engagement.
  for (const auto& test : cases) {
    SCOPED_TRACE(test.spec);
    const auto request = ControllerRequestFromSpec(test.spec);
    EXPECT_EQ(request.device_levels_explicit, test.levels_explicit);
    EXPECT_EQ(request.device_tree_explicit, test.tree_explicit);
    const auto decision = Falcata::ResolveCUDAController(request, {});
    EXPECT_EQ(decision.requested, test.route);
    EXPECT_EQ(decision.resolved, test.route);
  }
}

TEST(CUDAControllerPolicy, BooleanAliasesRememberExplicitOff) {
  using Route = Falcata::CUDAControllerRoute;
  for (const char* on : {"on", "1", "true", "ON", "True"}) {
    for (const char* off : {"off", "0", "false", "OFF", "False"}) {
      const std::string spec = std::string(" auto, device_tree:") + on + ",device_levels:" + off + " ";
      SCOPED_TRACE(spec);
      auto request = ControllerRequestFromSpec(spec.c_str());
      EXPECT_TRUE(request.device_levels_explicit);
      EXPECT_TRUE(request.device_tree_explicit);
      EXPECT_EQ(Falcata::ResolveCUDAController(request, {}).resolved, Route::kWhole);
      const std::string disabled = spec + ",device_tree:" + off;
      request = ControllerRequestFromSpec(disabled.c_str());
      EXPECT_EQ(Falcata::ResolveCUDAController(request, {}).resolved, Route::kHost);
    }
  }
}

TEST(CUDAControllerPolicy, UnrelatedOverridesDoNotForceControllerDefaults) {
  for (const char* spec : {"auto", "auto,row_batch:off", "auto,device_level_graph:off", "auto,graph_loop:off"}) {
    SCOPED_TRACE(spec);
    const auto request = ControllerRequestFromSpec(spec);
    EXPECT_FALSE(request.device_levels_explicit);
    EXPECT_FALSE(request.device_tree_explicit);
    const auto decision = Falcata::ResolveCUDAController(request, {});
    EXPECT_EQ(decision.requested, Falcata::CUDAControllerRoute::kAuto);
    EXPECT_EQ(decision.resolved, Falcata::CUDAControllerRoute::kHost);  // unknown hardware is unqualified
  }
}

namespace {
Falcata::CUDAControllerFacts ControllerFacts() {
  Falcata::CUDAControllerFacts facts;
  facts.num_rows = 200000;
  facts.num_features = 104;
  facts.sampled_features = 104;
  facts.max_feature_bins = 256;
  facts.num_bins = 104 * 256;
  facts.metadata_known = true;
  facts.num_leaves = 1024;
  facts.max_depth = 10;
  facts.min_data_in_leaf = 20;
  facts.quantized = true;
  facts.performance_modifiers_supported = true;
  facts.scope_supported = true;
  facts.native_prefix_eligible = true;
  facts.native_whole_eligible = true;
  facts.device_major = 12;
  facts.device_minor = 0;
  facts.device_sm_count = 170;
  facts.device_memory_bus_bits = 512;
  return facts;
}

// The installed generic export is the tree oracle. Check only the independent
// wrapper contract here, never transcribe learned conditions or V2 winners.
void ExpectLearnedController(const Falcata::CUDAControllerFacts& facts) {
  using Route = Falcata::CUDAControllerRoute;
  const auto raw = Falcata::LearnedCUDAControllerCandidate(facts);
  ASSERT_TRUE(raw == Route::kHost || raw == Route::kPrefix || raw == Route::kWhole);
  const auto decision = Falcata::ResolveCUDAController({}, facts);
  EXPECT_EQ(decision.requested, Route::kAuto);
  if (raw == Route::kHost) {
    EXPECT_EQ(decision.resolved, Route::kHost);
    EXPECT_STREQ(decision.reason, "learned_host");
  } else if (raw == Route::kPrefix) {
    EXPECT_EQ(decision.resolved, facts.native_prefix_eligible ? Route::kPrefix : Route::kHost);
    EXPECT_STREQ(decision.reason, facts.native_prefix_eligible ? "learned_prefix" : "learned_prefix_native_unavailable");
  } else {
    EXPECT_EQ(decision.resolved, facts.native_whole_eligible ? Route::kWhole : Route::kHost);
    EXPECT_STREQ(decision.reason, facts.native_whole_eligible ? "learned_whole" : "learned_whole_native_unavailable");
  }
}
}  // namespace

TEST(CUDAControllerPolicy, LearnedCoverageRetainsOldWidthRowsAndCapacityScenarios) {
  auto facts = ControllerFacts();
  ExpectLearnedController(facts);
  facts.num_rows = 99999;
  ExpectLearnedController(facts);
  facts.num_rows = 100000;
  ExpectLearnedController(facts);
  facts.num_features = facts.sampled_features = 256;
  ExpectLearnedController(facts);
  facts.num_features = facts.sampled_features = 257;
  ExpectLearnedController(facts);
  facts = ControllerFacts();
  facts.min_data_in_leaf = 100;
  facts.num_rows = 102399;
  ExpectLearnedController(facts);
  facts.num_rows = 102400;
  ExpectLearnedController(facts);
  facts.num_rows = 204799;
  facts.bagging_freq = 1;
  facts.bagging_fraction = 0.5;
  ExpectLearnedController(facts);
  facts.num_rows = 204800;
  ExpectLearnedController(facts);
  facts.bagging_freq = 0;  // The raw inactive fraction stays an independent model input.
  facts.bagging_fraction = 0.01;
  ExpectLearnedController(facts);
  facts.min_data_in_leaf = 0;
  ExpectLearnedController(facts);
}

TEST(CUDAControllerPolicy, FinalSingleLeafBudgetBindingRequiresNativeEligibility) {
  using Route = Falcata::CUDAControllerRoute;
  auto facts = ControllerFacts();
  facts.num_features = facts.sampled_features = 53;
  facts.num_bins = 53 * 256;
  facts.native_prefix_eligible = false;
  const auto explicit_off = ControllerRequestFromSpec("auto,device_tree:off,device_levels:off");
  const auto explicit_whole = ControllerRequestFromSpec("auto,device_tree:on,device_levels:off");
  for (const int depth : {8, 10}) {
    facts.max_depth = depth;
    const int depth_leaves = 1 << depth;
    // D=8,L=255 and D=10,L=1023 admit the native final single-leaf bind.
    facts.num_leaves = depth_leaves - 1;
    facts.native_whole_eligible = true;
    ExpectLearnedController(facts);
    facts.native_whole_eligible = false;
    ExpectLearnedController(facts);
    // D=8,L=254 and D=10,L=1022 have native eligibility false. The actual
    // DeviceControllerLevelCount owns this predicate, not the learned wrapper.
    facts.num_leaves = depth_leaves - 2;
    facts.native_whole_eligible = false;
    ExpectLearnedController(facts);
    for (const int leaves : {depth_leaves - 2, depth_leaves - 1, depth_leaves}) {
      facts.num_leaves = leaves;
      facts.native_whole_eligible = true;
      EXPECT_EQ(Falcata::ResolveCUDAController(explicit_off, facts).resolved, Route::kHost);
      // Explicit intent is preserved; actual execution still checks native eligibility.
      EXPECT_EQ(Falcata::ResolveCUDAController(explicit_whole, facts).resolved, Route::kWhole);
    }
    facts.num_leaves = depth_leaves;
    ExpectLearnedController(facts);
  }
}

TEST(CUDAControllerPolicy, LearnedCoverageRetainsOldPrefixScenariosAndNativeEligibility) {
  auto facts = ControllerFacts();
  facts.num_leaves = 1023;
  facts.native_whole_eligible = false;
  facts.num_features = facts.sampled_features = 64;
  facts.num_bins = 64 * 256;
  ExpectLearnedController(facts);
  facts.num_features = facts.sampled_features = 65;
  ExpectLearnedController(facts);
  facts.num_features = 256;
  facts.sampled_features = 128;
  facts.feature_fraction = 0.5;
  facts.bagging_freq = 1;
  facts.bagging_fraction = 0.5;
  facts.max_feature_bins = 4;
  facts.num_bins = 1024;
  ExpectLearnedController(facts);
  facts.num_features = 258;
  facts.sampled_features = 129;
  ExpectLearnedController(facts);
  facts.num_features = 256;
  facts.sampled_features = 128;
  facts.max_feature_bins = 5;
  ExpectLearnedController(facts);
  facts.max_feature_bins = 4;
  facts.feature_fraction = 0.499;
  ExpectLearnedController(facts);
  facts.feature_fraction = 0.5;
  facts.bagging_fraction = 0.499;
  ExpectLearnedController(facts);
  facts.bagging_fraction = 0.5;
  facts.native_prefix_eligible = false;
  ExpectLearnedController(facts);
  facts = ControllerFacts();
  facts.num_features = facts.sampled_features = 28;
  facts.max_feature_bins = 4;
  facts.num_bins = 112;
  facts.native_whole_eligible = false;
  ExpectLearnedController(facts);
  facts.native_prefix_eligible = false;
  ExpectLearnedController(facts);
}

TEST(CUDAControllerPolicy, MissingMetadataAndHardwareScopeAreIndependentOfBinHistogram) {
  using Route = Falcata::CUDAControllerRoute;
  auto facts = ControllerFacts();
  auto resolve = [&]() { return Falcata::ResolveCUDAController({}, facts).resolved; };
  facts.max_feature_bins = 257;
  ExpectLearnedController(facts);
  facts.max_feature_bins = 9;
  facts.num_bins = 104 * 9;
  ExpectLearnedController(facts);
  facts.missing_feature_count = 1;  // same bins, a valid feature has None -> NaN or Zero mode
  ExpectLearnedController(facts);
  facts.metadata_known = false;
  EXPECT_EQ(resolve(), Route::kHost);
  facts = ControllerFacts();
  facts.device_sm_count = 169;
  EXPECT_EQ(resolve(), Route::kHost);
  facts.device_sm_count = 170;
  facts.device_memory_bus_bits = 384;
  EXPECT_EQ(resolve(), Route::kHost);
  facts.device_memory_bus_bits = 512;
  facts.device_minor = 1;
  EXPECT_EQ(resolve(), Route::kHost);
  facts.device_minor = 0;
  facts.scope_supported = false;
  EXPECT_EQ(resolve(), Route::kLegacy);
  facts.scope_supported = true;
  facts.quantized = false;
  EXPECT_EQ(resolve(), Route::kLegacy);
}

TEST(CUDAControllerPolicy, UnqualifiedGraphAndModifiersUseHostButDoNotOverrideExplicitIntent) {
  using Route = Falcata::CUDAControllerRoute;
  auto facts = ControllerFacts();
  facts.graph_cache_enabled = false;
  auto decision = Falcata::ResolveCUDAController({}, facts);
  EXPECT_EQ(decision.resolved, Route::kHost);
  EXPECT_STREQ(decision.reason, "v2_graph_disabled");
  facts.graph_cache_enabled = true;
  facts.performance_modifiers_supported = false;
  decision = Falcata::ResolveCUDAController({}, facts);
  EXPECT_EQ(decision.resolved, Route::kHost);
  EXPECT_STREQ(decision.reason, "v2_unqualified_modifiers");
  // Explicit intent bypasses performance admission on any hardware, including
  // when metadata is unavailable. Actual execution still uses native gates.
  facts.device_major = 8;
  facts.metadata_known = false;
  facts.graph_cache_enabled = false;
  EXPECT_EQ(Falcata::ResolveCUDAController({}, facts).resolved, Route::kHost);
  EXPECT_EQ(Falcata::ResolveCUDAController(ControllerRequestFromSpec("auto,device_levels:on"), facts).resolved,
            Route::kPrefix);
  EXPECT_EQ(Falcata::ResolveCUDAController(ControllerRequestFromSpec("auto,device_tree:on,device_levels:off"), facts).resolved,
            Route::kWhole);
  facts = ControllerFacts();
  facts.metadata_known = false;
  decision = Falcata::ResolveCUDAController({}, facts);
  EXPECT_EQ(decision.resolved, Route::kHost);
  EXPECT_STREQ(decision.reason, "unavailable_metadata");
}

TEST(CUDAControllerPolicy, LearnedCoverageRetainsOldLowBinWidthAndMissingScenarios) {
  using Route = Falcata::CUDAControllerRoute;
  auto facts = ControllerFacts();
  facts.num_features = 2048;
  facts.sampled_features = 512;
  facts.feature_fraction = 0.25;
  facts.max_feature_bins = 8;
  facts.num_bins = 8192;
  facts.min_data_in_leaf = 10000;
  ExpectLearnedController(facts);
  facts.missing_feature_count = 1;
  ExpectLearnedController(facts);
  facts.sampled_features = 513;
  facts.feature_fraction = 513.0 / facts.num_features;
  ExpectLearnedController(facts);
  facts.sampled_features = 512;
  facts.feature_fraction = 0.25;
  facts.num_bins = 8193;
  ExpectLearnedController(facts);
  facts.num_bins = 8192;
  facts.max_feature_bins = 9;
  ExpectLearnedController(facts);
  facts.missing_feature_count = 0;
  facts.sampled_features = 256;
  facts.feature_fraction = 0.125;
  ExpectLearnedController(facts);
  facts.min_data_in_leaf = 20;
  ExpectLearnedController(facts);
  facts.missing_feature_count = 1;
  ExpectLearnedController(facts);
  facts.max_feature_bins = 8;
  ExpectLearnedController(facts);
  facts.num_bins = 0;
  const auto unavailable = Falcata::ResolveCUDAController({}, facts);
  EXPECT_EQ(unavailable.resolved, Route::kHost);
  EXPECT_STREQ(unavailable.reason, "unavailable_metadata");
  facts.num_bins = 8192;
  facts.native_whole_eligible = false;
  ExpectLearnedController(facts);
}

TEST(CUDAControllerPolicy, TotalHistogramFootprintCannotBeReplacedBySampledWidth) {
  auto facts = ControllerFacts();
  facts.num_features = 3555;
  facts.sampled_features = 356;
  facts.feature_fraction = 0.1;
  facts.max_feature_bins = 6;
  facts.num_bins = 15000;
  facts.min_data_in_leaf = 10000;
  ExpectLearnedController(facts);
  facts.missing_feature_count = 1;
  ExpectLearnedController(facts);
  // Preserve the independent total-footprint input at the same sampled width;
  // the generated model, rather than a manual cap, supplies its prediction.
  facts.num_bins = 8192;
  ExpectLearnedController(facts);
}

TEST(CUDAControllerPolicy, LearnedWinnerUsesOnlyItsOwnNativeCapability) {
  auto facts = ControllerFacts();
  const int shapes[][2] = {{1, 2}, {1, 512}, {2, 4}, {8, 254}, {8, 255}, {10, 1022}, {10, 1023},
                          {11, 2048}, {-1, 1024}};
  for (const int bins : {4, 256}) {
    facts.max_feature_bins = bins;
    facts.num_bins = facts.num_features * bins;
    for (const auto& shape : shapes) {
      facts.max_depth = shape[0];
      facts.num_leaves = shape[1];
      for (const bool prefix : {false, true}) {
        for (const bool whole : {false, true}) {
          // Inject the initial capability interface, not a second implementation
          // of the native level-count predicate or an alternate-route ranking.
          facts.native_prefix_eligible = prefix;
          facts.native_whole_eligible = whole;
          ExpectLearnedController(facts);
        }
      }
    }
  }
}

TEST(CUDAControllerPolicy, HardwareAndBasicMetadataPrecedeLegacyScope) {
  using Route = Falcata::CUDAControllerRoute;
  auto facts = ControllerFacts();
  facts.quantized = false;
  facts.scope_supported = false;
  facts.device_major = 0;
  auto decision = Falcata::ResolveCUDAController({}, facts);
  EXPECT_EQ(decision.resolved, Route::kHost);
  EXPECT_STREQ(decision.reason, "unqualified_hardware");
  facts.device_major = 12;
  facts.metadata_known = false;
  decision = Falcata::ResolveCUDAController({}, facts);
  EXPECT_EQ(decision.resolved, Route::kHost);
  EXPECT_STREQ(decision.reason, "unavailable_metadata");
  facts.metadata_known = true;
  facts.graph_cache_enabled = false;
  facts.performance_modifiers_supported = false;
  decision = Falcata::ResolveCUDAController({}, facts);
  EXPECT_EQ(decision.resolved, Route::kLegacy);
  EXPECT_STREQ(decision.reason, "legacy_scope");
}
