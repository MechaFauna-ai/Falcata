/*!
 * Copyright (c) 2026 Falcata contributors. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "feature_histogram.hpp"

namespace Falcata {

void FeatureHistogram::FuncForNumrical() {
  if (meta_->config->extra_trees) {
    if (meta_->config->monotone_constraints.empty()) {
      FuncForNumricalL1<true, false>();
    } else {
      FuncForNumricalL1<true, true>();
    }
  } else {
    if (meta_->config->monotone_constraints.empty()) {
      FuncForNumricalL1<false, false>();
    } else {
      FuncForNumricalL1<false, true>();
    }
  }
}

template <bool USE_RAND, bool USE_MC>
void FeatureHistogram::FuncForNumricalL1() {
  if (meta_->config->lambda_l1 > 0) {
    if (meta_->config->max_delta_step > 0) {
      FuncForNumricalL2<USE_RAND, USE_MC, true, true>();
    } else {
      FuncForNumricalL2<USE_RAND, USE_MC, true, false>();
    }
  } else {
    if (meta_->config->max_delta_step > 0) {
      FuncForNumricalL2<USE_RAND, USE_MC, false, true>();
    } else {
      FuncForNumricalL2<USE_RAND, USE_MC, false, false>();
    }
  }
}

template <bool USE_RAND, bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT>
void FeatureHistogram::FuncForNumricalL2() {
  if (meta_->config->path_smooth > kEpsilon) {
    FuncForNumricalL3<USE_RAND, USE_MC, USE_L1, USE_MAX_OUTPUT, true>();
  } else {
    FuncForNumricalL3<USE_RAND, USE_MC, USE_L1, USE_MAX_OUTPUT, false>();
  }
}

template <bool USE_RAND, bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
void FeatureHistogram::FuncForNumricalL3() {
#define TEMPLATE_PREFIX USE_RAND, USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING
#define LAMBDA_ARGUMENTS                                         \
double sum_gradient, double sum_hessian, data_size_t num_data, \
  const FeatureConstraint* constraints, double parent_output, SplitInfo *output
#define BEFORE_ARGUMENTS sum_gradient, sum_hessian, parent_output, num_data, output, &rand_threshold
#define FUNC_ARGUMENTS                                                      \
sum_gradient, sum_hessian, num_data, constraints, min_gain_shift, \
  output, rand_threshold, parent_output

  if (meta_->num_bin > 2 && meta_->missing_type != MissingType::None) {
    if (meta_->missing_type == MissingType::Zero) {
      find_best_threshold_fun_ = [=](LAMBDA_ARGUMENTS) {
        int rand_threshold = 0;
        double min_gain_shift =
            BeforeNumerical<USE_RAND, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                BEFORE_ARGUMENTS);
        FindBestThresholdSequentially<TEMPLATE_PREFIX, true, true, false>(
            FUNC_ARGUMENTS);
        FindBestThresholdSequentially<TEMPLATE_PREFIX, false, true, false>(
            FUNC_ARGUMENTS);
      };
    } else {
      find_best_threshold_fun_ = [=](LAMBDA_ARGUMENTS) {
        int rand_threshold = 0;
        double min_gain_shift =
            BeforeNumerical<USE_RAND, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                BEFORE_ARGUMENTS);
        FindBestThresholdSequentially<TEMPLATE_PREFIX, true, false, true>(
            FUNC_ARGUMENTS);
        FindBestThresholdSequentially<TEMPLATE_PREFIX, false, false, true>(
            FUNC_ARGUMENTS);
      };
    }
  } else {
    if (meta_->missing_type != MissingType::NaN) {
      find_best_threshold_fun_ = [=](LAMBDA_ARGUMENTS) {
        int rand_threshold = 0;
        double min_gain_shift =
            BeforeNumerical<USE_RAND, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                BEFORE_ARGUMENTS);
        FindBestThresholdSequentially<TEMPLATE_PREFIX, true, false, false>(
            FUNC_ARGUMENTS);
      };
    } else {
      find_best_threshold_fun_ = [=](LAMBDA_ARGUMENTS) {
        int rand_threshold = 0;
        double min_gain_shift =
            BeforeNumerical<USE_RAND, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                BEFORE_ARGUMENTS);
        FindBestThresholdSequentially<TEMPLATE_PREFIX, true, false, false>(
            FUNC_ARGUMENTS);
        output->default_left = false;
      };
    }
  }
#undef TEMPLATE_PREFIX
#undef LAMBDA_ARGUMENTS
#undef BEFORE_ARGUMENTS
#undef FUNC_ARGUMENTS
}

}  // namespace Falcata
