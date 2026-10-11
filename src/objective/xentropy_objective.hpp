/*!
 * Copyright (c) 2017-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2017-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef FALCATA_SRC_OBJECTIVE_XENTROPY_OBJECTIVE_HPP_
#define FALCATA_SRC_OBJECTIVE_XENTROPY_OBJECTIVE_HPP_

#include <Falcata/meta.h>
#include <Falcata/objective_function.h>
#include <Falcata/utils/common.h>

#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

/*
 * Implements gradients and Hessians for the following point losses.
 * Target y is anything in interval [0, 1].
 *
 * (1) CrossEntropy; "xentropy";
 *
 * loss(y, p, w) = { -(1-y)*log(1-p)-y*log(p) }*w,
 * with probability p = 1/(1+exp(-f)), where f is being boosted
 *
 * ConvertToOutput: f -> p
 *
 * (2) CrossEntropyLambda; "xentlambda"
 *
 * loss(y, p, w) = -(1-y)*log(1-p)-y*log(p),
 * with p = 1-exp(-lambda*w), lambda = log(1+exp(f)), f being boosted, and w > 0
 *
 * ConvertToOutput: f -> lambda
 *
 * (1) and (2) are the same if w=1; but outputs still differ.
 *
 */

namespace Falcata {
/*!
* \brief Objective function for cross-entropy (with optional linear weights)
*/
class CrossEntropy: public ObjectiveFunction {
 public:
  explicit CrossEntropy(const Config& config)
      : deterministic_(config.deterministic) {}

  explicit CrossEntropy(const std::vector<std::string>&)
      : deterministic_(false) {
  }

  ~CrossEntropy() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();

    CHECK_NOTNULL(label_);
    Common::CheckElementsIntervalClosed<label_t>(label_, 0.0f, 1.0f, num_data_, GetName());
    Log::Info("[%s:%s]: (objective) labels passed interval [0, 1] check",  GetName(), __func__);

    if (weights_ != nullptr) {
      label_t minw;
      double sumw;
      Common::ObtainMinMaxSum(weights_, num_data_, &minw, static_cast<label_t*>(nullptr), &sumw);
      if (minw < 0.0f) {
        Log::Fatal("[%s]: at least one weight is negative", GetName());
      }
      if (sumw == 0.0f) {
        Log::Fatal("[%s]: sum of weights is zero", GetName());
      }
    }
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    // z = expit(score) = 1 / (1 + exp(-score))
    // gradient = z - label = expit(score) - label
    // Numerically more stable, see http://fa.bianp.net/blog/2019/evaluate_logistic/
    //     if score < 0:
    //         exp_tmp = exp(score)
    //         return ((1 - label) * exp_tmp - label) / (1 + exp_tmp)
    //     else:
    //         exp_tmp = exp(-score)
    //         return ((1 - label) - label * exp_tmp) / (1 + exp_tmp)
    // Note that optimal speed would be achieved, at the cost of precision, by
    //     return expit(score) - y_true
    // i.e. no "if else" and an own inline implementation of expit.
    // The case distinction score < 0 in the stable implementation does not
    // provide significant better precision apart from protecting overflow of exp(..).
    // The branch (if else), however, can incur runtime costs of up to 30%.
    // Instead, we help branch prediction by almost always ending in the first if clause
    // and making the second branch (else) a bit simpler. This has the exact same
    // precision but is faster than the stable implementation.
    // As branching criteria, we use the same cutoff as in log1pexp, see link above.
    // Note that the maximal value to get gradient = -1 with label = 1 is -37.439198610162731
    // (based on mpmath), and scipy.special.logit(np.finfo(float).eps) ~ -36.04365.
    if (weights_ == nullptr) {
      // compute pointwise gradients and Hessians with implied unit weights
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        if (score[i] > -37.0) {
          const double exp_tmp = std::exp(-score[i]);
          gradients[i] = static_cast<score_t>(((1.0f - label_[i]) - label_[i] * exp_tmp) / (1.0f + exp_tmp));
          hessians[i] = static_cast<score_t>(exp_tmp / ((1 + exp_tmp) * (1 + exp_tmp)));
        } else {
          const double exp_tmp = std::exp(score[i]);
          gradients[i] = static_cast<score_t>(exp_tmp - label_[i]);
          hessians[i] = static_cast<score_t>(exp_tmp);
        }
      }
    } else {
      // compute pointwise gradients and Hessians with given weights
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        if (score[i] > -37.0) {
          const double exp_tmp = std::exp(-score[i]);
          gradients[i] = static_cast<score_t>(((1.0f - label_[i]) - label_[i] * exp_tmp) / (1.0f + exp_tmp) * weights_[i]);
          hessians[i] = static_cast<score_t>(exp_tmp / ((1 + exp_tmp) * (1 + exp_tmp)) * weights_[i]);
        } else {
          const double exp_tmp = std::exp(score[i]);
          gradients[i] = static_cast<score_t>((exp_tmp - label_[i]) * weights_[i]);
          hessians[i] = static_cast<score_t>(exp_tmp * weights_[i]);
        }
      }
    }
  }

  const char* GetName() const override {
    return "cross_entropy";
  }

  // convert score to a probability
  void ConvertOutput(const double* input, double* output) const override {
    output[0] = 1.0f / (1.0f + std::exp(-input[0]));
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  // implement custom average to boost from (if enabled among options)
  double BoostFromScore(int) const override {
    double suml = 0.0f;
    double sumw = 0.0f;
    if (weights_ != nullptr) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:suml, sumw) if (!deterministic_)

      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += static_cast<double>(label_[i]) * weights_[i];
        sumw += weights_[i];
      }
    } else {
      sumw = static_cast<double>(num_data_);
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:suml) if (!deterministic_)

      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += label_[i];
      }
    }
    double pavg = suml / sumw;
    pavg = std::min(pavg, 1.0 - kEpsilon);
    pavg = std::max<double>(pavg, kEpsilon);
    double initscore = std::log(pavg / (1.0f - pavg));
    Log::Info("[%s:%s]: pavg = %f -> initscore = %f",  GetName(), __func__, pavg, initscore);
    return initscore;
  }

 private:
  /*! \brief Number of data points */
  data_size_t num_data_;
  /*! \brief Pointer for label */
  const label_t* label_;
  /*! \brief Weights for data */
  const label_t* weights_;
  const bool deterministic_;
};

/*!
* \brief Objective function for alternative parameterization of cross-entropy (see top of file for explanation)
*/
class CrossEntropyLambda: public ObjectiveFunction {
 public:
  explicit CrossEntropyLambda(const Config& config)
      : deterministic_(config.deterministic) {
    min_weight_ = max_weight_ = 0.0f;
  }

  explicit CrossEntropyLambda(const std::vector<std::string>&)
      : deterministic_(false) {}

  ~CrossEntropyLambda() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();

    CHECK_NOTNULL(label_);
    Common::CheckElementsIntervalClosed<label_t>(label_, 0.0f, 1.0f, num_data_, GetName());
    Log::Info("[%s:%s]: (objective) labels passed interval [0, 1] check",  GetName(), __func__);

    if (weights_ != nullptr) {
      Common::ObtainMinMaxSum(weights_, num_data_, &min_weight_, &max_weight_, static_cast<label_t*>(nullptr));
      if (min_weight_ <= 0.0f) {
        Log::Fatal("[%s]: at least one weight is non-positive", GetName());
      }

      // Issue an info statement about this ratio
      double weight_ratio = max_weight_ / min_weight_;
      Log::Info("[%s:%s]: min, max weights = %f, %f; ratio = %f",
                GetName(), __func__,
                min_weight_, max_weight_,
                weight_ratio);
    } else {
      // all weights are implied to be unity; no need to do anything
    }
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (weights_ == nullptr) {
      // compute pointwise gradients and Hessians with implied unit weights; exactly equivalent to CrossEntropy with unit weights
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double z = 1.0f / (1.0f + std::exp(-score[i]));
        gradients[i] = static_cast<score_t>(z - label_[i]);
        hessians[i] = static_cast<score_t>(z * (1.0f - z));
      }
    } else {
      // compute pointwise gradients and Hessians with given weights
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double w = weights_[i];
        const double y = label_[i];
        const double epf = std::exp(score[i]);
        const double hhat = std::log1p(epf);
        const double u = w * hhat;
        if (u < kMinReliableU || u > kMaxReliableU || score[i] > kMaxReliableScore) {
          double grad, hess;
          StableWeightedGradHess(score[i], w, y, &grad, &hess);
          gradients[i] = static_cast<score_t>(grad);
          hessians[i] = static_cast<score_t>(hess);
          continue;
        }
        const double z = 1.0f - std::exp(-w*hhat);
        const double enf = 1.0f / epf;  // = std::exp(-score[i]);
        gradients[i] = static_cast<score_t>((1.0f - y / z) * w / (1.0f + enf));
        const double c = 1.0f / (1.0f - z);
        double d = 1.0f + epf;
        const double a = w * epf / (d * d);
        d = c - 1.0f;
        const double b = (c / (d * d) ) * (1.0f + w * epf - c);
        hessians[i] = static_cast<score_t>(a * (1.0f + y * b));
      }
    }
  }

  const char* GetName() const override {
    return "cross_entropy_lambda";
  }

  //
  // ATTENTION: the function output is the "normalized exponential parameter" lambda > 0, not the probability
  //
  // If this code would read: output[0] = 1.0f / (1.0f + std::exp(-input[0]));
  // The output would still not be the probability unless the weights are unity.
  //
  // Let z = 1 / (1 + exp(-f)), then prob(z) = 1-(1-z)^w, where w is the weight for the specific point.
  //

  void ConvertOutput(const double* input, double* output) const override {
    output[0] = std::log1p(std::exp(input[0]));
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  double BoostFromScore(int) const override {
    double suml = 0.0f;
    double sumw = 0.0f;
    if (weights_ != nullptr) {
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:suml, sumw) if (!deterministic_)

      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += static_cast<double>(label_[i]) * weights_[i];
        sumw += weights_[i];
      }
    } else {
      sumw = static_cast<double>(num_data_);
      #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static) reduction(+:suml) if (!deterministic_)

      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += label_[i];
      }
    }
    double havg = suml / sumw;
    double initscore = std::log(std::expm1(havg));
    Log::Info("[%s:%s]: havg = %f -> initscore = %f",  GetName(), __func__, havg, initscore);
    return initscore;
  }

 private:
  // z = 1 - exp(-u) with u = w * log1p(exp(s)) loses about -log10(u) digits as
  // u -> 0 and log10(e) * u digits as u grows, until it rounds to 0 or 1; the
  // expression in GetGradients is only used where u keeps it accurate.
  // (1 + exp(s))^2 also overflows for large s.
  static constexpr double kMinReliableU = 1e-3;
  static constexpr double kMaxReliableU = 20.0;
  static constexpr double kMaxReliableScore = 50.0;

  // Cancellation-free form of the weighted gradient and Hessian.
  static void StableWeightedGradHess(double s, double w, double y, double* grad, double* hess) {
    const double p = s >= 0.0 ? 1.0 / (1.0 + std::exp(-s)) : std::exp(s) / (1.0 + std::exp(s));
    const double one_minus_p = s >= 0.0 ? std::exp(-s) / (1.0 + std::exp(-s)) : 1.0 / (1.0 + std::exp(s));
    const double hhat = s > 0.0 ? s + std::log1p(std::exp(-s)) : std::log1p(std::exp(s));
    const double u = w * hhat;
    const double z = -std::expm1(-u);
    const double q = std::exp(-u);  // 1 - z
    const double wp = w * p;
    const double a = wp * one_minus_p;
    if (z < std::numeric_limits<double>::min()) {
      // limits as u -> 0, where w * p / z -> 1
      *grad = wp - y;
      *hess = a * (1.0 + y * (1.0 - w) / (2.0 * w));
      return;
    }
    // 1 - y / z, with z - y formed without cancellation
    const double one_minus_y_over_z = (u < 1.0 ? z - y : (1.0 - y) - q) / z;
    *grad = wp * one_minus_y_over_z;
    if (u < 1e-8) {
      // leading term of the series in u; the closed form below cancels here
      *hess = a * (1.0 + y * (1.0 - w) / (2.0 * w));
    } else {
      *hess = a * one_minus_y_over_z + y * wp * wp * q / (z * z);
    }
  }

  /*! \brief Number of data points */
  data_size_t num_data_;
  /*! \brief Pointer for label */
  const label_t* label_;
  /*! \brief Weights for data */
  const label_t* weights_;
  /*! \brief Minimum weight found during init */
  label_t min_weight_;
  /*! \brief Maximum weight found during init */
  label_t max_weight_;
  const bool deterministic_;
};

}  // end namespace Falcata

#endif  // FALCATA_SRC_OBJECTIVE_XENTROPY_OBJECTIVE_HPP_
