// Generated candidate only; existing overrides/native/support/hardware gates remain outside.
// model_sha256=33eceaeae7a242febfe450aca655a0d993b4934d6c02a17841fa4204b2e0bd28
#pragma once
#include <cmath>
#include <Falcata/cuda_controller_policy.h>
namespace Falcata {
inline CUDAControllerRoute LearnedCUDAControllerCandidate(const CUDAControllerFacts& facts) {
  if (!facts.metadata_known) return CUDAControllerRoute::kHost;
  if (facts.num_rows < -9007199254740992LL || facts.num_rows >= 9007199254740992LL) return CUDAControllerRoute::kHost;
  const double v0 = static_cast<double>(facts.num_rows);
  if (!std::isfinite(v0)) return CUDAControllerRoute::kHost;
  if (facts.num_features < -2147483648LL || facts.num_features >= 2147483648LL) return CUDAControllerRoute::kHost;
  const double v1 = static_cast<double>(facts.num_features);
  if (!std::isfinite(v1)) return CUDAControllerRoute::kHost;
  if (facts.sampled_features < -2147483648LL || facts.sampled_features >= 2147483648LL) return CUDAControllerRoute::kHost;
  const double v2 = static_cast<double>(facts.sampled_features);
  if (!std::isfinite(v2)) return CUDAControllerRoute::kHost;
  if (facts.num_bins < -9007199254740992LL || facts.num_bins >= 9007199254740992LL) return CUDAControllerRoute::kHost;
  const double v3 = static_cast<double>(facts.num_bins);
  if (!std::isfinite(v3)) return CUDAControllerRoute::kHost;
  if (facts.max_feature_bins < -2147483648LL || facts.max_feature_bins >= 2147483648LL) return CUDAControllerRoute::kHost;
  const double v4 = static_cast<double>(facts.max_feature_bins);
  if (!std::isfinite(v4)) return CUDAControllerRoute::kHost;
  if (facts.missing_feature_count < -2147483648LL || facts.missing_feature_count >= 2147483648LL) return CUDAControllerRoute::kHost;
  const double v5 = static_cast<double>(facts.missing_feature_count);
  if (!std::isfinite(v5)) return CUDAControllerRoute::kHost;
  if (facts.num_leaves < -2147483648LL || facts.num_leaves >= 2147483648LL) return CUDAControllerRoute::kHost;
  const double v6 = static_cast<double>(facts.num_leaves);
  if (!std::isfinite(v6)) return CUDAControllerRoute::kHost;
  if (facts.max_depth < -2147483648LL || facts.max_depth >= 2147483648LL) return CUDAControllerRoute::kHost;
  const double v7 = static_cast<double>(facts.max_depth);
  if (!std::isfinite(v7)) return CUDAControllerRoute::kHost;
  if (facts.min_data_in_leaf < -2147483648LL || facts.min_data_in_leaf >= 2147483648LL) return CUDAControllerRoute::kHost;
  const double v8 = static_cast<double>(facts.min_data_in_leaf);
  if (!std::isfinite(v8)) return CUDAControllerRoute::kHost;
  const double v9 = static_cast<double>(facts.feature_fraction);
  if (!std::isfinite(v9)) return CUDAControllerRoute::kHost;
  const double v10 = static_cast<double>(facts.bagging_fraction);
  if (!std::isfinite(v10)) return CUDAControllerRoute::kHost;
  if (facts.bagging_freq < -2147483648LL || facts.bagging_freq >= 2147483648LL) return CUDAControllerRoute::kHost;
  const double v11 = static_cast<double>(facts.bagging_freq);
  if (!std::isfinite(v11)) return CUDAControllerRoute::kHost;
  if (v4 <= 4.5) {
    if (v6 <= 1279.5) {
      return CUDAControllerRoute::kWhole;
    } else {
      return CUDAControllerRoute::kPrefix;
    }
  } else {
    if (v4 <= 6.5) {
      if (v0 <= 244001.5) {
        return CUDAControllerRoute::kWhole;
      } else {
        return CUDAControllerRoute::kHost;
      }
    } else {
      if (v8 <= 32.5) {
        if (v1 <= 48.0) {
          return CUDAControllerRoute::kWhole;
        } else {
          return CUDAControllerRoute::kHost;
        }
      } else {
        if (v3 <= 57344.0) {
          return CUDAControllerRoute::kWhole;
        } else {
          return CUDAControllerRoute::kHost;
        }
      }
    }
  }
}
}  // namespace Falcata
