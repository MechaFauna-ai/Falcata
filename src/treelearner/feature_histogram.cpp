/*!
 * Copyright (c) 2024-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2024-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "feature_histogram.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace Falcata {

namespace {

// Draws for the RANDOM categorical search come from a SplitMix64 stream: the
// per-feature Random is an LCG whose low bits carry short periods, so its raw
// output cannot supply independent subset draws.
inline uint64_t SplitMix64(uint64_t* state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Uniform index in [0, n); n is a category count, so modulo bias is below 2^-48.
inline int NextIndex(uint64_t* state, int n) {
  return static_cast<int>(SplitMix64(state) % static_cast<uint64_t>(n));
}

// Seeds a trial stream from the feature's Random, keeping masks reproducible
// given extra_seed, the feature, and the sequence of node evaluations.
inline uint64_t SeedFromFeatureRandom(Random* rand) {
  return (static_cast<uint64_t>(rand->NextShort(0, 32768)) << 15) |
         static_cast<uint64_t>(rand->NextShort(0, 32768));
}

// Moves `size` uniformly-chosen candidates to the front of `pool` (partial
// Fisher-Yates; the residual order carries into the next trial, which stays
// uniform).
inline void DrawSubset(uint64_t* state, int size, std::vector<int>* pool) {
  const int num_cand = static_cast<int>(pool->size());
  for (int j = 0; j < size; ++j) {
    const int r = j + NextIndex(state, num_cand - j);
    std::swap((*pool)[j], (*pool)[r]);
  }
}

}  // namespace

void FeatureHistogram::FuncForCategorical() {
  if (meta_->config->extra_trees) {
    if (meta_->config->monotone_constraints.empty()) {
      FuncForCategoricalL1<true, false>();
    } else {
      FuncForCategoricalL1<true, true>();
    }
  } else {
    if (meta_->config->monotone_constraints.empty()) {
      FuncForCategoricalL1<false, false>();
    } else {
      FuncForCategoricalL1<false, true>();
    }
  }
}

template <bool USE_RAND, bool USE_MC>
void FeatureHistogram::FuncForCategoricalL1() {
  if (meta_->config->path_smooth > kEpsilon) {
    FuncForCategoricalL2<USE_RAND, USE_MC, true>();
  } else {
    FuncForCategoricalL2<USE_RAND, USE_MC, false>();
  }
}

template <bool USE_RAND, bool USE_MC, bool USE_SMOOTHING>
void FeatureHistogram::FuncForCategoricalL2() {
#define ARGUMENTS                                                      \
std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, \
    std::placeholders::_4, std::placeholders::_5, std::placeholders::_6
  if (meta_->config->lambda_l1 > 0) {
    if (meta_->config->max_delta_step > 0) {
      find_best_threshold_fun_ =
          std::bind(&FeatureHistogram::FindBestThresholdCategoricalInner<
                        USE_RAND, USE_MC, true, true, USE_SMOOTHING>,
                    this, ARGUMENTS);
    } else {
      find_best_threshold_fun_ =
          std::bind(&FeatureHistogram::FindBestThresholdCategoricalInner<
                        USE_RAND, USE_MC, true, false, USE_SMOOTHING>,
                    this, ARGUMENTS);
    }
  } else {
    if (meta_->config->max_delta_step > 0) {
      find_best_threshold_fun_ =
          std::bind(&FeatureHistogram::FindBestThresholdCategoricalInner<
                        USE_RAND, USE_MC, false, true, USE_SMOOTHING>,
                    this, ARGUMENTS);
    } else {
      find_best_threshold_fun_ =
          std::bind(&FeatureHistogram::FindBestThresholdCategoricalInner<
                        USE_RAND, USE_MC, false, false, USE_SMOOTHING>,
                    this, ARGUMENTS);
    }
  }
#undef ARGUMENTS
}

template <bool USE_RAND, bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
void FeatureHistogram::FindBestThresholdCategoricalInner(double sum_gradient,
                                        double sum_hessian,
                                        data_size_t num_data,
                                        const FeatureConstraint* constraints,
                                        double parent_output,
                                        SplitInfo* output) {
  is_splittable_ = false;
  output->default_left = false;
  double best_gain = kMinScore;
  data_size_t best_left_count = 0;
  double best_sum_left_gradient = 0;
  double best_sum_left_hessian = 0;
  double gain_shift;
  if (USE_MC) {
    constraints->InitCumulativeConstraints(true);
  }
  if (USE_SMOOTHING) {
    gain_shift = GetLeafGainGivenOutput<USE_L1>(
        sum_gradient, sum_hessian, meta_->config->lambda_l1, meta_->config->lambda_l2, parent_output);
  } else {
    // Need special case for no smoothing to preserve existing behaviour. If no smoothing, the parent output is calculated
    // with the larger categorical l2, whereas min_split_gain uses the original l2.
    gain_shift = GetLeafGain<USE_L1, USE_MAX_OUTPUT, false>(sum_gradient, sum_hessian,
        meta_->config->lambda_l1, meta_->config->lambda_l2, meta_->config->max_delta_step, 0,
        num_data, 0);
  }

  double min_gain_shift = gain_shift + meta_->config->min_gain_to_split;
  const int8_t offset = meta_->offset;
  const int bin_start = 1 - offset;
  const int bin_end = meta_->num_bin - offset;
  int used_bin = -1;

  std::vector<int> sorted_idx;
  double l2 = meta_->config->lambda_l2;
  bool use_onehot = meta_->num_bin <= meta_->config->max_cat_to_onehot;
  int best_threshold = -1;
  int best_dir = 1;
  const double cnt_factor = num_data / sum_hessian;
  int rand_threshold = 0;
  if (use_onehot) {
    if (USE_RAND) {
      if (bin_end - bin_start > 0) {
        rand_threshold = meta_->rand.NextInt(bin_start, bin_end);
      }
    }
    for (int t = bin_start; t < bin_end; ++t) {
      const auto grad = GET_GRAD(data_, t);
      const auto hess = GET_HESS(data_, t);
      data_size_t cnt =
          static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));
      // if data not enough, or sum hessian too small
      if (cnt < meta_->config->min_data_in_leaf ||
          hess < meta_->config->min_sum_hessian_in_leaf) {
        continue;
      }
      data_size_t other_count = num_data - cnt;
      // if data not enough
      if (other_count < meta_->config->min_data_in_leaf) {
        continue;
      }

      double sum_other_hessian = sum_hessian - hess - kEpsilon;
      // if sum hessian too small
      if (sum_other_hessian < meta_->config->min_sum_hessian_in_leaf) {
        continue;
      }

      double sum_other_gradient = sum_gradient - grad;
      if (USE_RAND) {
        if (t != rand_threshold) {
          continue;
        }
      }
      // current split gain
      double current_gain = GetSplitGains<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
          sum_other_gradient, sum_other_hessian, grad, hess + kEpsilon,
          meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
          constraints, 0, meta_->config->path_smooth, other_count, cnt, parent_output);
      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) {
        continue;
      }

      // mark as able to be split
      is_splittable_ = true;
      // better split point
      if (current_gain > best_gain) {
        best_threshold = t;
        best_sum_left_gradient = grad;
        best_sum_left_hessian = hess + kEpsilon;
        best_left_count = cnt;
        best_gain = current_gain;
      }
    }
  } else if (meta_->config->cat_random_search > 0) {
    // RANDOM categorical search (YDF-style): each trial draws a random subset
    // of the present categories and evaluates subset-vs-rest with the exact
    // search's gain formula and constraints; the best subset wins. Subset
    // sizes are uniform over [1, max_num_cat], the same cap the exact search
    // puts on its prefix length -- drawing each category with probability 1/2
    // instead would put every subset of a high-cardinality feature over that
    // cap, so the feature would never split at all.
    for (int i = bin_start; i < bin_end; ++i) {
      if (Common::RoundInt(GET_HESS(data_, i) * cnt_factor) >=
          meta_->config->cat_smooth) {
        sorted_idx.push_back(i);
      }
    }
    const int num_cand = static_cast<int>(sorted_idx.size());
    const int max_num_cat =
        std::min(meta_->config->max_cat_threshold, (num_cand + 1) / 2);

    l2 += meta_->config->cat_l2;

    std::vector<int> pool(sorted_idx);
    std::vector<int> best_subset;
    uint64_t rand_state = SeedFromFeatureRandom(&meta_->rand);
    for (int trial = 0; trial < meta_->config->cat_random_search && max_num_cat > 0;
         ++trial) {
      const int subset_size = 1 + NextIndex(&rand_state, max_num_cat);
      DrawSubset(&rand_state, subset_size, &pool);

      double sum_left_gradient = 0.0;
      double sum_left_hessian = kEpsilon;
      data_size_t left_count = 0;
      for (int j = 0; j < subset_size; ++j) {
        const int t = pool[j];
        const auto hess = GET_HESS(data_, t);
        sum_left_gradient += GET_GRAD(data_, t);
        sum_left_hessian += hess;
        left_count +=
            static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));
      }
      // categories filtered by cat_smooth, and categories unseen in this node,
      // always go right
      const data_size_t right_count = num_data - left_count;
      if (left_count < meta_->config->min_data_in_leaf ||
          left_count < meta_->config->min_data_per_group ||
          sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) {
        continue;
      }
      if (right_count < meta_->config->min_data_in_leaf ||
          right_count < meta_->config->min_data_per_group) {
        continue;
      }
      const double sum_right_hessian = sum_hessian - sum_left_hessian;
      if (sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) {
        continue;
      }
      const double sum_right_gradient = sum_gradient - sum_left_gradient;
      double current_gain = GetSplitGains<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, meta_->config->lambda_l1, l2,
          meta_->config->max_delta_step, constraints, 0, meta_->config->path_smooth,
          left_count, right_count, parent_output);
      if (current_gain <= min_gain_shift) {
        continue;
      }
      is_splittable_ = true;
      if (current_gain > best_gain) {
        best_subset.assign(pool.begin(), pool.begin() + subset_size);
        best_left_count = left_count;
        best_sum_left_gradient = sum_left_gradient;
        best_sum_left_hessian = sum_left_hessian;
        best_gain = current_gain;
      }
    }
    if (is_splittable_) {
      // the shared output block below emits sorted_idx's prefix as the left
      // set; the bin order within a split is a bitset downstream, so the
      // ascending order is for reproducible model text only
      std::sort(best_subset.begin(), best_subset.end());
      sorted_idx = best_subset;
      used_bin = static_cast<int>(sorted_idx.size());
      best_threshold = used_bin - 1;
      best_dir = 1;
    }
  } else {
    for (int i = bin_start; i < bin_end; ++i) {
      if (Common::RoundInt(GET_HESS(data_, i) * cnt_factor) >=
          meta_->config->cat_smooth) {
        sorted_idx.push_back(i);
      }
    }
    used_bin = static_cast<int>(sorted_idx.size());

    l2 += meta_->config->cat_l2;

    auto ctr_fun = [this](double sum_grad, double sum_hess) {
      return (sum_grad) / (sum_hess + meta_->config->cat_smooth);
    };
    std::stable_sort(
        sorted_idx.begin(), sorted_idx.end(), [this, &ctr_fun](int i, int j) {
          return ctr_fun(GET_GRAD(data_, i), GET_HESS(data_, i)) <
                  ctr_fun(GET_GRAD(data_, j), GET_HESS(data_, j));
        });

    std::vector<int> find_direction(1, 1);
    std::vector<int> start_position(1, 0);
    find_direction.push_back(-1);
    start_position.push_back(used_bin - 1);
    const int max_num_cat =
        std::min(meta_->config->max_cat_threshold, (used_bin + 1) / 2);
    int max_threshold = std::max(std::min(max_num_cat, used_bin) - 1, 0);
    if (USE_RAND) {
      if (max_threshold > 0) {
        rand_threshold = meta_->rand.NextInt(0, max_threshold);
      }
    }

    is_splittable_ = false;
    for (size_t out_i = 0; out_i < find_direction.size(); ++out_i) {
      auto dir = find_direction[out_i];
      auto start_pos = start_position[out_i];
      data_size_t min_data_per_group = meta_->config->min_data_per_group;
      data_size_t cnt_cur_group = 0;
      double sum_left_gradient = 0.0f;
      double sum_left_hessian = kEpsilon;
      data_size_t left_count = 0;
      for (int i = 0; i < used_bin && i < max_num_cat; ++i) {
        auto t = sorted_idx[start_pos];
        start_pos += dir;
        const auto grad = GET_GRAD(data_, t);
        const auto hess = GET_HESS(data_, t);
        data_size_t cnt =
            static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));

        sum_left_gradient += grad;
        sum_left_hessian += hess;
        left_count += cnt;
        cnt_cur_group += cnt;

        if (left_count < meta_->config->min_data_in_leaf ||
            sum_left_hessian < meta_->config->min_sum_hessian_in_leaf) {
          continue;
        }
        data_size_t right_count = num_data - left_count;
        if (right_count < meta_->config->min_data_in_leaf ||
            right_count < min_data_per_group) {
          break;
        }

        double sum_right_hessian = sum_hessian - sum_left_hessian;
        if (sum_right_hessian < meta_->config->min_sum_hessian_in_leaf) {
          break;
        }

        if (cnt_cur_group < min_data_per_group) {
          continue;
        }

        cnt_cur_group = 0;

        double sum_right_gradient = sum_gradient - sum_left_gradient;
        if (USE_RAND) {
          if (i != rand_threshold) {
            continue;
          }
        }
        double current_gain = GetSplitGains<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, meta_->config->lambda_l1, l2,
            meta_->config->max_delta_step, constraints, 0, meta_->config->path_smooth,
            left_count, right_count, parent_output);
        if (current_gain <= min_gain_shift) {
          continue;
        }
        is_splittable_ = true;
        if (current_gain > best_gain) {
          best_left_count = left_count;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
          best_threshold = i;
          best_gain = current_gain;
          best_dir = dir;
        }
      }
    }
  }

  if (is_splittable_) {
    output->left_output = CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
        best_sum_left_gradient, best_sum_left_hessian,
        meta_->config->lambda_l1, l2, meta_->config->max_delta_step,
        constraints->LeftToBasicConstraint(), meta_->config->path_smooth, best_left_count, parent_output);
    output->left_count = best_left_count;
    output->left_sum_gradient = best_sum_left_gradient;
    output->left_sum_hessian = best_sum_left_hessian - kEpsilon;
    output->right_output = CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
        sum_gradient - best_sum_left_gradient,
        sum_hessian - best_sum_left_hessian, meta_->config->lambda_l1, l2,
        meta_->config->max_delta_step, constraints->RightToBasicConstraint(), meta_->config->path_smooth,
        num_data - best_left_count, parent_output);
    output->right_count = num_data - best_left_count;
    output->right_sum_gradient = sum_gradient - best_sum_left_gradient;
    output->right_sum_hessian =
        sum_hessian - best_sum_left_hessian - kEpsilon;
    output->gain = best_gain - min_gain_shift;
    if (use_onehot) {
      output->num_cat_threshold = 1;
      output->cat_threshold =
          std::vector<uint32_t>(1, static_cast<uint32_t>(best_threshold + offset));
    } else {
      output->num_cat_threshold = best_threshold + 1;
      output->cat_threshold =
          std::vector<uint32_t>(output->num_cat_threshold);
      if (best_dir == 1) {
        for (int i = 0; i < output->num_cat_threshold; ++i) {
          auto t = sorted_idx[i] + offset;
          output->cat_threshold[i] = t;
        }
      } else {
        for (int i = 0; i < output->num_cat_threshold; ++i) {
          auto t = sorted_idx[used_bin - 1 - i] + offset;
          output->cat_threshold[i] = t;
        }
      }
    }
    output->monotone_type = 0;
  }
}

}  // namespace Falcata
