/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <Falcata/meta.h>
#include <Falcata/utils/common.h>
#include <Falcata/utils/pipeline_reader.h>
#include <Falcata/utils/text_reader.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
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

// Join(strs, start, end) tested `end - start <= 0` on size_t, which wraps
// for end < start, so an empty range still joined one element.
TEST(CommonJoin, EmptyOrInvertedRangeIsEmpty) {
  const std::vector<int> v = {1, 2, 3};
  EXPECT_EQ(Falcata::Common::Join(v, 1, 3, ","), "2,3");
  EXPECT_EQ(Falcata::Common::Join(v, 2, 2, ","), "");
  EXPECT_EQ(Falcata::Common::Join(v, 2, 1, ","), "");
  EXPECT_EQ(Falcata::Common::Join(std::vector<int>(), 0, 1, ","), "");
}

// SIZE_ALIGNED was not parenthesized as a whole, so inside an expression
// the trailing "* kAlignedSize" bound to the surrounding operator.
TEST(SizeAligned, IsASingleOperand) {
  using Falcata::kAlignedSize;
  EXPECT_EQ(SIZE_ALIGNED(1), 32);
  EXPECT_EQ(64 / SIZE_ALIGNED(1), 2);
  EXPECT_EQ(SIZE_ALIGNED(33) % 32, 0);
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

// TextReader's end-of-line skip read buffer_process[i] before checking
// i < read_cnt, so a line terminator in the last byte of a full 16 MiB block
// read one byte past the block (visible under AddressSanitizer).
TEST(TextReader, LineEndAtBlockEndDoesNotOverread) {
  const std::string path = ::testing::TempDir() + "falcata_text_reader_block_end.txt";
  const size_t block = 16 * 1024 * 1024;
  const size_t line_len = 1024;  // 1023 'x' + '\n'; block / line_len lines fill the block exactly
  {
    std::ofstream out(path, std::ios::binary);
    const std::string line = std::string(line_len - 1, 'x') + "\n";
    for (size_t i = 0; i < block / line_len; ++i) out << line;
    out << "tail\n";
  }
  Falcata::TextReader<int> reader(path.c_str(), false);
  std::vector<size_t> lengths;
  const int n = reader.ReadAllAndProcess([&lengths](int, const char*, size_t len) { lengths.push_back(len); });
  std::remove(path.c_str());
  ASSERT_EQ(n, static_cast<int>(block / line_len + 1));
  EXPECT_EQ(lengths.front(), line_len - 1);
  EXPECT_EQ(lengths.back(), 4u);
}

// PipelineReader::Read read skip_bytes bytes into its 16 MiB scratch buffer in
// one call, so skipping more than a block wrote past it (visible under
// AddressSanitizer).
TEST(PipelineReader, SkipsMoreThanOneBlock) {
  const std::string path = ::testing::TempDir() + "falcata_pipeline_reader_skip.bin";
  const size_t skip = 20 * 1024 * 1024;
  {
    std::ofstream out(path, std::ios::binary);
    const std::string zeros(1024 * 1024, '\0');
    for (size_t i = 0; i < skip / zeros.size(); ++i) out << zeros;
    out << "hello";
  }
  std::string seen;
  const size_t cnt = Falcata::PipelineReader::Read(path.c_str(), static_cast<int>(skip),
                                                   [&seen](const char* buf, size_t len) {
                                                     seen.append(buf, len);
                                                     return len;
                                                   });
  std::remove(path.c_str());
  EXPECT_EQ(cnt, 5u);
  EXPECT_EQ(seen, "hello");
}
