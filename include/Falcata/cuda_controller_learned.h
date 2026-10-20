// Generated candidate only; existing overrides/native/support/hardware gates remain outside.
// model_sha256=bb6c23f7df3f5b1ae4031fb86f41e33ae9da36a7b5d2dc59b98d90d910735ea9
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
  if (v6 <= 1536.0) {
    return CUDAControllerRoute::kWhole;
  } else {
    return CUDAControllerRoute::kPrefix;
  }
}
}  // namespace Falcata
