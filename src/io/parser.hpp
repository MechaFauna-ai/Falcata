/*!
 * Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_IO_PARSER_HPP_
#define FALCATA_SRC_IO_PARSER_HPP_

#include <Falcata/dataset.h>
#include <Falcata/utils/common.h>
#include <Falcata/utils/log.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Falcata {

/*!
 * \brief parses the feature index of a LibSVM "index:value" token; a negative
 *        index or one past the int range is a format error (both used to index
 *        the per-feature arrays out of bounds)
 */
inline const char* AtoiLibSVMIndex(const char* p, int* out) {
  while (*p == ' ') {
    ++p;
  }
  const char* start = p;
  bool negative = false;
  if (*p == '-') {
    negative = true;
    ++p;
  } else if (*p == '+') {
    ++p;
  }
  int64_t value = 0;
  for (; *p >= '0' && *p <= '9'; ++p) {
    value = value * 10 + (*p - '0');
    if (value >= std::numeric_limits<int>::max()) {
      Log::Fatal("LibSVM feature index %s is too large", std::string(start, p - start + 1).c_str());
    }
  }
  if (negative && value != 0) {
    Log::Fatal("LibSVM feature index -%s is negative", std::to_string(value).c_str());
  }
  *out = static_cast<int>(value);
  while (*p == ' ') {
    ++p;
  }
  return p;
}

class CSVParser: public Parser {
 public:
  explicit CSVParser(int label_idx, int total_columns, AtofFunc atof)
    :label_idx_(label_idx), total_columns_(total_columns), atof_(atof) {
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    int idx = 0;
    double val = 0.0f;
    int offset = 0;
    *out_label = 0.0f;
    while (*str != '\0') {
      str = atof_(str, &val);
      if (idx == label_idx_) {
        *out_label = val;
        offset = -1;
      } else if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
        out_features->emplace_back(idx + offset, val);
      }
      ++idx;
      if (*str == ',') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("Input format error when parsing as CSV");
      }
    }
  }

  inline int NumFeatures() const override {
    return total_columns_ - (label_idx_ >= 0);
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
  AtofFunc atof_;
};

class TSVParser: public Parser {
 public:
  explicit TSVParser(int label_idx, int total_columns, AtofFunc atof)
    :label_idx_(label_idx), total_columns_(total_columns), atof_(atof) {
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    int idx = 0;
    double val = 0.0f;
    int offset = 0;
    // like CSVParser: a line without a label column must not keep the
    // caller's previous label
    *out_label = 0.0f;
    while (*str != '\0') {
      str = atof_(str, &val);
      if (idx == label_idx_) {
        *out_label = val;
        offset = -1;
      } else if (std::fabs(val) > kZeroThreshold || std::isnan(val)) {
        out_features->emplace_back(idx + offset, val);
      }
      ++idx;
      if (*str == '\t') {
        ++str;
      } else if (*str != '\0') {
        Log::Fatal("Input format error when parsing as TSV");
      }
    }
  }

  inline int NumFeatures() const override {
    return total_columns_ - (label_idx_ >= 0);
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
  AtofFunc atof_;
};

class LibSVMParser: public Parser {
 public:
  explicit LibSVMParser(int label_idx, int total_columns, AtofFunc atof)
    :label_idx_(label_idx), total_columns_(total_columns), atof_(atof) {
    if (label_idx > 0) {
      Log::Fatal("Label should be the first column in a LibSVM file");
    }
  }
  inline void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features, double* out_label) const override {
    int idx = 0;
    double val = 0.0f;
    *out_label = 0.0f;  // label_idx_ == -1: no label column (see CSVParser)
    if (label_idx_ == 0) {
      str = atof_(str, &val);
      *out_label = val;
      str = Common::SkipSpaceAndTab(str);
    }
    while (*str != '\0') {
      str = AtoiLibSVMIndex(str, &idx);
      str = Common::SkipSpaceAndTab(str);
      if (*str == ':') {
        ++str;
        str = Common::Atof(str, &val);
        out_features->emplace_back(idx, val);
      } else {
        Log::Fatal("Input format error when parsing as LibSVM");
      }
      str = Common::SkipSpaceAndTab(str);
    }
  }

  inline int NumFeatures() const override {
    return total_columns_;
  }

 private:
  int label_idx_ = 0;
  int total_columns_ = -1;
  AtofFunc atof_;
};

}  // namespace Falcata
#endif   // FALCATA_SRC_IO_PARSER_HPP_
