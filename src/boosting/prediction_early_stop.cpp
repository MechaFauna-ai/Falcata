/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <Falcata/prediction_early_stop.h>

#include <Falcata/utils/log.h>

#include <utility>
#include <cmath>
#include <functional>
#include <limits>
#include <string>

namespace Falcata {

PredictionEarlyStopInstance CreateNone(const PredictionEarlyStopConfig&) {
  return PredictionEarlyStopInstance{
    [](const double*, int) {
    return false;
  },
    std::numeric_limits<int>::max()  // make sure the lambda is almost never called
  };
}

PredictionEarlyStopInstance CreateMulticlass(const PredictionEarlyStopConfig& config) {
  // margin_threshold will be captured by value
  const double margin_threshold = config.margin_threshold;

  return PredictionEarlyStopInstance{
    [margin_threshold](const double* pred, int sz) {
    if (sz < 2) {
      Log::Fatal("Multiclass early stopping needs predictions to be of length two or larger");
    }

    double largest = pred[0];
    double second = pred[1];
    if (second > largest) std::swap(largest, second);
    for (int i = 2; i < sz; ++i) {
      if (pred[i] > largest) {
        second = largest;
        largest = pred[i];
      } else if (pred[i] > second) {
        second = pred[i];
      }
    }
    const auto margin = largest - second;

    if (margin > margin_threshold) {
      return true;
    }

    return false;
  },
    config.round_period
  };
}

PredictionEarlyStopInstance CreateBinary(const PredictionEarlyStopConfig& config) {
  // margin_threshold will be captured by value
  const double margin_threshold = config.margin_threshold;

  return PredictionEarlyStopInstance{
    [margin_threshold](const double* pred, int sz) {
    if (sz != 1) {
      Log::Fatal("Binary early stopping needs predictions to be of length one");
    }
    const auto margin = 2.0 * fabs(pred[0]);

    if (margin > margin_threshold) {
      return true;
    }

    return false;
  },
    config.round_period
  };
}

PredictionEarlyStopInstance CreatePredictionEarlyStopInstance(const std::string& type,
                                                              const PredictionEarlyStopConfig& config) {
  if (type == "none") {
    return CreateNone(config);
  } else if (type == "multiclass") {
    return CreateMulticlass(config);
  } else if (type == "binary") {
    return CreateBinary(config);
  } else {
    Log::Fatal("Unknown early stopping type: %s", type.c_str());
  }

  // Fix for compiler warnings about reaching end of control
  return CreateNone(config);
}

}  // namespace Falcata
