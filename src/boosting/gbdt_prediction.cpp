/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <Falcata/objective_function.h>
#include <Falcata/prediction_early_stop.h>
#include <Falcata/utils/openmp_wrapper.h>

#include <unordered_map>

#include "gbdt.h"

namespace Falcata {

void GBDT::PredictRawInRange(const double* features, double* output, const PredictionEarlyStopInstance* early_stop,
                             const PredictRange& range) const {
  int early_stop_round_counter = 0;
  const int num_output = NumOutputPerIteration();
  // set zero
  std::memset(output, 0, sizeof(double) * num_output);
  const int end_iteration_for_pred = range.start_iteration + range.num_iteration;
  for (int i = range.start_iteration; i < end_iteration_for_pred; ++i) {
    if (vector_leaf_mode_) {
      const double* tree_output = models_[i]->PredictVector(features);
      for (int target = 0; target < num_output; ++target) {
        output[target] += tree_output[target];
      }
    } else {
      // predict all the trees for one iteration
      for (int k = 0; k < num_tree_per_iteration_; ++k) {
        output[k] += models_[i * num_tree_per_iteration_ + k]->Predict(features);
      }
    }
    // check early stopping
    ++early_stop_round_counter;
    if (early_stop->round_period == early_stop_round_counter) {
      if (early_stop->callback_function(output, num_output)) {
        return;
      }
      early_stop_round_counter = 0;
    }
  }
}

void GBDT::PredictRawByMapInRange(const std::unordered_map<int, double>& features, double* output,
                                  const PredictionEarlyStopInstance* early_stop, const PredictRange& range) const {
  int early_stop_round_counter = 0;
  const int num_output = NumOutputPerIteration();
  // set zero
  std::memset(output, 0, sizeof(double) * num_output);
  const int end_iteration_for_pred = range.start_iteration + range.num_iteration;
  for (int i = range.start_iteration; i < end_iteration_for_pred; ++i) {
    if (vector_leaf_mode_) {
      const double* tree_output = models_[i]->PredictVectorByMap(features);
      for (int target = 0; target < num_output; ++target) {
        output[target] += tree_output[target];
      }
    } else {
      // predict all the trees for one iteration
      for (int k = 0; k < num_tree_per_iteration_; ++k) {
        output[k] += models_[i * num_tree_per_iteration_ + k]->PredictByMap(features);
      }
    }
    // check early stopping
    ++early_stop_round_counter;
    if (early_stop->round_period == early_stop_round_counter) {
      if (early_stop->callback_function(output, num_output)) {
        return;
      }
      early_stop_round_counter = 0;
    }
  }
}

void GBDT::PredictInRange(const double* features, double* output, const PredictionEarlyStopInstance* early_stop,
                          const PredictRange& range) const {
  PredictRawInRange(features, output, early_stop, range);
  if (average_output_ && range.num_iteration > 0) {
    for (int k = 0; k < NumOutputPerIteration(); ++k) {
      output[k] /= range.num_iteration;
    }
  }
  if (objective_function_ != nullptr) {
    objective_function_->ConvertOutput(output, output);
  }
}

void GBDT::PredictByMapInRange(const std::unordered_map<int, double>& features, double* output,
                               const PredictionEarlyStopInstance* early_stop, const PredictRange& range) const {
  PredictRawByMapInRange(features, output, early_stop, range);
  if (average_output_ && range.num_iteration > 0) {
    for (int k = 0; k < NumOutputPerIteration(); ++k) {
      output[k] /= range.num_iteration;
    }
  }
  if (objective_function_ != nullptr) {
    objective_function_->ConvertOutput(output, output);
  }
}

void GBDT::PredictLeafIndexInRange(const double* features, double* output, const PredictRange& range) const {
  int start_tree = range.start_iteration * num_tree_per_iteration_;
  int num_trees = range.num_iteration * num_tree_per_iteration_;
  const auto* models_ptr = models_.data() + start_tree;
  for (int i = 0; i < num_trees; ++i) {
    output[i] = models_ptr[i]->PredictLeafIndex(features);
  }
}

void GBDT::PredictLeafIndexByMapInRange(const std::unordered_map<int, double>& features, double* output,
                                        const PredictRange& range) const {
  int start_tree = range.start_iteration * num_tree_per_iteration_;
  int num_trees = range.num_iteration * num_tree_per_iteration_;
  const auto* models_ptr = models_.data() + start_tree;
  for (int i = 0; i < num_trees; ++i) {
    output[i] = models_ptr[i]->PredictLeafIndexByMap(features);
  }
}


// the legacy entry points predict over the range set by the last InitPredict
void GBDT::PredictRaw(const double* features, double* output, const PredictionEarlyStopInstance* early_stop) const {
  PredictRawInRange(features, output, early_stop, InitializedPredictRange());
}

void GBDT::PredictRawByMap(const std::unordered_map<int, double>& features, double* output,
                           const PredictionEarlyStopInstance* early_stop) const {
  PredictRawByMapInRange(features, output, early_stop, InitializedPredictRange());
}

void GBDT::Predict(const double* features, double* output, const PredictionEarlyStopInstance* early_stop) const {
  PredictInRange(features, output, early_stop, InitializedPredictRange());
}

void GBDT::PredictByMap(const std::unordered_map<int, double>& features, double* output,
                        const PredictionEarlyStopInstance* early_stop) const {
  PredictByMapInRange(features, output, early_stop, InitializedPredictRange());
}

void GBDT::PredictLeafIndex(const double* features, double* output) const {
  PredictLeafIndexInRange(features, output, InitializedPredictRange());
}

void GBDT::PredictLeafIndexByMap(const std::unordered_map<int, double>& features, double* output) const {
  PredictLeafIndexByMapInRange(features, output, InitializedPredictRange());
}

}  // namespace Falcata
