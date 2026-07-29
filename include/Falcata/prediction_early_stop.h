/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_INCLUDE_FALCATA_PREDICTION_EARLY_STOP_H_
#define FALCATA_INCLUDE_FALCATA_PREDICTION_EARLY_STOP_H_

#include <Falcata/export.h>

#include <string>
#include <functional>

namespace Falcata {

struct PredictionEarlyStopInstance {
  /// Callback function type for early stopping.
  /// Takes current prediction and number of elements in prediction
  /// @returns true if prediction should stop according to criterion
  using FunctionType = std::function<bool(const double*, int)>;

  FunctionType callback_function;  // callback function itself
  int          round_period;       // call callback_function every `runPeriod` iterations
};

struct PredictionEarlyStopConfig {
  int round_period;
  double margin_threshold;
};

/// Create an early stopping algorithm of type `type`, with given round_period and margin threshold
FALCATA_EXPORT PredictionEarlyStopInstance CreatePredictionEarlyStopInstance(const std::string& type,
                                                                              const PredictionEarlyStopConfig& config);

}   // namespace Falcata

#endif  // FALCATA_INCLUDE_FALCATA_PREDICTION_EARLY_STOP_H_
