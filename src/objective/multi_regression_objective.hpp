/*!
 * Copyright (c) 2026 The Falcata developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef FALCATA_SRC_OBJECTIVE_MULTI_REGRESSION_OBJECTIVE_HPP_
#define FALCATA_SRC_OBJECTIVE_MULTI_REGRESSION_OBJECTIVE_HPP_

#include <Falcata/network.h>
#include <Falcata/objective_function.h>
#include <Falcata/utils/common.h>

#include <string>
#include <sstream>
#include <vector>

namespace Falcata {

/*!
 * \brief Round-robin multi-target squared-error regression: one tree per
 * target per iteration (the multiclass machinery minus the softmax coupling),
 * so every iteration produces num_target independent single-target trees and
 * the models are identical to training each target separately -- while paying
 * dataset construction, binning and the boosting loop once.
 *
 * num_class carries the target count, reusing the multiclass plumbing end to
 * end: the [target * num_data + i] score layout, per-class boost-from-average,
 * the interleaved tree storage and the [n, num_class] predict reshape.
 *
 * Labels are column-major [target * num_data + i]; the Python package
 * flattens a 2D y of shape [num_data, num_target] accordingly (Fortran order)
 * and the Metadata layer accepts label vectors whose length is an exact
 * multiple of num_data.
 */
class MultiRegressionL2 : public ObjectiveFunction {
 public:
  explicit MultiRegressionL2(const Config& config) {
    num_target_ = config.num_class;
    if (num_target_ <= 1) {
      Log::Fatal("multi_regression requires num_class > 1 "
                 "(num_class carries the number of targets)");
    }
  }

  explicit MultiRegressionL2(const std::vector<std::string>& strs) {
    num_target_ = -1;
    for (auto str : strs) {
      auto tokens = Common::Split(str.c_str(), ':');
      if (tokens.size() == 2 && tokens[0] == std::string("num_class")) {
        Common::Atoi(tokens[1].c_str(), &num_target_);
      }
    }
    if (num_target_ < 0) {
      Log::Fatal("Objective multi_regression should contain num_class field");
    }
  }

  ~MultiRegressionL2() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
    if (metadata.label_size() !=
        static_cast<size_t>(num_data_) * static_cast<size_t>(num_target_)) {
      Log::Fatal("multi_regression with num_class=%d expects %zu label values "
                 "(num_data * num_class, column-major), but got %zu. Pass y as "
                 "a 2D array of shape [num_data, num_class].",
                 num_target_,
                 static_cast<size_t>(num_data_) * num_target_,
                 metadata.label_size());
    }
    target_means_.resize(num_target_, 0.0);
    double sum_weights = 0.0;
    if (weights_ == nullptr) {
      sum_weights = static_cast<double>(num_data_);
    } else {
      for (data_size_t i = 0; i < num_data_; ++i) {
        sum_weights += weights_[i];
      }
    }
    for (int k = 0; k < num_target_; ++k) {
      const label_t* target_label = label_ + static_cast<size_t>(k) * num_data_;
      double sum = 0.0;
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum)
        for (data_size_t i = 0; i < num_data_; ++i) {
          sum += target_label[i];
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:sum)
        for (data_size_t i = 0; i < num_data_; ++i) {
          sum += target_label[i] * weights_[i];
        }
      }
      if (Network::num_machines() > 1) {
        sum = Network::GlobalSyncUpBySum(sum);
      }
      target_means_[k] = sum / sum_weights;
    }
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    for (int k = 0; k < num_target_; ++k) {
      const size_t offset = static_cast<size_t>(k) * num_data_;
      const label_t* target_label = label_ + offset;
      const double* target_score = score + offset;
      score_t* target_grad = gradients + offset;
      score_t* target_hess = hessians + offset;
      if (weights_ == nullptr) {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < num_data_; ++i) {
          target_grad[i] = static_cast<score_t>(target_score[i] - target_label[i]);
          target_hess[i] = 1.0f;
        }
      } else {
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (data_size_t i = 0; i < num_data_; ++i) {
          target_grad[i] = static_cast<score_t>(
            (target_score[i] - target_label[i]) * weights_[i]);
          target_hess[i] = static_cast<score_t>(weights_[i]);
        }
      }
    }
  }

  void ConvertOutput(const double* input, double* output) const override {
    for (int k = 0; k < num_target_; ++k) {
      output[k] = input[k];
    }
  }

  const char* GetName() const override { return "multi_regression"; }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName() << " ";
    str_buf << "num_class:" << num_target_;
    return str_buf.str();
  }

  int NumModelPerIteration() const override { return num_target_; }

  int NumPredictOneRow() const override { return num_target_; }

  bool IsConstantHessian() const override { return weights_ == nullptr; }

  double BoostFromScore(int class_id) const override {
    return target_means_[class_id];
  }

 protected:
  /*! \brief Number of targets (carried by num_class) */
  int num_target_;
  /*! \brief Number of data rows */
  data_size_t num_data_;
  /*! \brief Column-major [target * num_data + i] label pointer */
  const label_t* label_;
  /*! \brief Per-row weights (broadcast over targets) */
  const label_t* weights_;
  /*! \brief Per-target weighted label means (boost-from-average seeds) */
  std::vector<double> target_means_;
};

}  // namespace Falcata

#endif  // FALCATA_SRC_OBJECTIVE_MULTI_REGRESSION_OBJECTIVE_HPP_
