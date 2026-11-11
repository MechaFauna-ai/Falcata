/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/bin.h>
#include <Falcata/utils/common.h>
#include <Falcata/utils/json11.h>

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

// CHECK used to expand to a bare `if (...) Fatal(...);`, so with the
// caller's semicolon `if (a) CHECK(b); else ...` did not compile (or, without
// that semicolon, bound the else to the macro's if). It is a single
// statement now.
namespace Falcata {
static int CheckInIfElse(bool take_if, bool check_ok) {
  if (take_if)
    CHECK(check_ok);
  else
    return 2;
  return 1;
}
}  // namespace Falcata

TEST(CheckMacro, IsASingleStatement) {
  EXPECT_EQ(Falcata::CheckInIfElse(true, true), 1);
  EXPECT_EQ(Falcata::CheckInIfElse(false, false), 2);
  EXPECT_THROW(Falcata::CheckInIfElse(true, false), std::exception);
}

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

// CommonC::StringToArrayFast (used to parse tree arrays from model text)
// had no end-of-string check: a short string silently produced 0 for the
// missing integers and NaN for the missing floats.
TEST(CommonCStringToArrayFast, ShortStringFails) {
  EXPECT_EQ(Falcata::CommonC::StringToArrayFast<int>("1 2 3", 3), (std::vector<int>{1, 2, 3}));
  EXPECT_THROW(Falcata::CommonC::StringToArrayFast<int>("1 2", 3), std::exception);
  EXPECT_THROW(Falcata::CommonC::StringToArrayFast<double>("0.5 ", 2), std::exception);
  EXPECT_THROW(Falcata::CommonC::StringToArrayFast<int>("", 1), std::exception);
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

// Inputs that end inside a \u escape or a literal used to be read 4 (or
// len("false")) bytes past the end of the buffer before the length check;
// under AddressSanitizer that was a heap-buffer-overflow.
TEST(Json11, TruncatedInputFailsWithoutOverread) {
  using json11_internal_lightgbm::Json;
  // padded past the small-string buffer, so each text sits in an exact-size
  // heap block and an overread is visible to sanitizers
  const std::string pad(32, ' ');
  for (const char* text : {"\"\\u12", "\"\\u", "t", "fa", "nul"}) {
    const std::string padded = pad + text;
    const std::string in(padded.data(), padded.size());
    std::string err;
    Json::parse(in, &err);
    EXPECT_FALSE(err.empty()) << "accepted truncated input: " << text;
  }
}

// int_value() of a number outside int's range was an undefined
// double-to-int cast (x86 gives INT_MIN for both signs); it saturates now.
TEST(Json11, IntValueSaturates) {
  using json11_internal_lightgbm::Json;
  std::string err;
  EXPECT_EQ(Json::parse("1e30", &err).int_value(), std::numeric_limits<int>::max());
  EXPECT_EQ(Json::parse("-1e30", &err).int_value(), std::numeric_limits<int>::min());
  EXPECT_EQ(Json::parse("-7.9", &err).int_value(), -7);
  EXPECT_EQ(Json(std::nan("")).int_value(), 0);
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

// Categorical ValueToBin used to cast the double to int before any range
// check, which is undefined for +-inf and values outside int's range (UBSan:
// "1e+20 is outside the range of representable values of type 'int'").
// Values must still truncate as before: (-1, 0) is category 0, <= -1 and
// out-of-range values go to bin 0.
TEST(BinMapper, CategoricalValueToBinOutOfRange) {
  std::vector<double> values;
  for (int i = 0; i < 300; ++i) {
    values.push_back(i % 3);
  }
  Falcata::BinMapper mapper;
  mapper.FindBin(values.data(), static_cast<int>(values.size()), values.size(), 255, 3, 1, false,
                 Falcata::BinType::CategoricalBin, true, false, {});
  const uint32_t bin_of_zero = mapper.ValueToBin(0.0);
  EXPECT_EQ(mapper.ValueToBin(-0.5), bin_of_zero);
  EXPECT_EQ(mapper.ValueToBin(2.7), mapper.ValueToBin(2.0));
  for (double v : {-1.0, -1e20, 1e20, 2147483648.0, std::numeric_limits<double>::infinity(),
                   -std::numeric_limits<double>::infinity()}) {
    EXPECT_EQ(mapper.ValueToBin(v), 0u) << v;
  }
}
