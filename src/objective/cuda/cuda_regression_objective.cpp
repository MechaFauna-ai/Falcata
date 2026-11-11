/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_regression_objective.hpp"

#include <string>
#include <vector>

namespace Falcata {

CUDARegressionL2loss::CUDARegressionL2loss(const Config& config):
CUDARegressionObjectiveInterface<RegressionL2loss>(config) {}

CUDARegressionL2loss::CUDARegressionL2loss(const std::vector<std::string>& strs):
CUDARegressionObjectiveInterface<RegressionL2loss>(strs) {}

CUDARegressionL2loss::~CUDARegressionL2loss() {}

void CUDARegressionL2loss::Init(const Metadata& metadata, data_size_t num_data) {
  CUDARegressionObjectiveInterface<RegressionL2loss>::Init(metadata, num_data);
}

CUDAMultiRegressionL2::CUDAMultiRegressionL2(const Config& config): CUDAObjectiveInterface<MultiRegressionL2>(config) {}

CUDAMultiRegressionL2::CUDAMultiRegressionL2(const std::vector<std::string>& strs): CUDAObjectiveInterface<MultiRegressionL2>(strs) {}

CUDAMultiRegressionL2::~CUDAMultiRegressionL2() {}

CUDARegressionL1loss::CUDARegressionL1loss(const Config& config):
CUDAPercentileRegressionObjective<RegressionL1loss>(config) {}

CUDARegressionL1loss::CUDARegressionL1loss(const std::vector<std::string>& strs):
CUDAPercentileRegressionObjective<RegressionL1loss>(strs) {}

CUDARegressionL1loss::~CUDARegressionL1loss() {}



CUDARegressionHuberLoss::CUDARegressionHuberLoss(const Config& config):
CUDARegressionObjectiveInterface<RegressionHuberLoss>(config) {}

CUDARegressionHuberLoss::CUDARegressionHuberLoss(const std::vector<std::string>& strs):
CUDARegressionObjectiveInterface<RegressionHuberLoss>(strs) {}

CUDARegressionHuberLoss::~CUDARegressionHuberLoss() {}


CUDARegressionFairLoss::CUDARegressionFairLoss(const Config& config):
CUDARegressionObjectiveInterface<RegressionFairLoss>(config) {}

CUDARegressionFairLoss::CUDARegressionFairLoss(const std::vector<std::string>& strs):
CUDARegressionObjectiveInterface<RegressionFairLoss>(strs) {}

CUDARegressionFairLoss::~CUDARegressionFairLoss() {}


CUDARegressionPoissonLoss::CUDARegressionPoissonLoss(const Config& config):
CUDARegressionObjectiveInterface<RegressionPoissonLoss>(config) {}

CUDARegressionPoissonLoss::CUDARegressionPoissonLoss(const std::vector<std::string>& strs):
CUDARegressionObjectiveInterface<RegressionPoissonLoss>(strs) {}

CUDARegressionPoissonLoss::~CUDARegressionPoissonLoss() {}

void CUDARegressionPoissonLoss::Init(const Metadata& metadata, data_size_t num_data) {
  CUDARegressionObjectiveInterface<RegressionPoissonLoss>::Init(metadata, num_data);
  LaunchCheckLabelKernel();
}

double CUDARegressionPoissonLoss::LaunchCalcInitScoreKernel(const int class_id) const {
  return Common::SafeLog(CUDARegressionObjectiveInterface<RegressionPoissonLoss>::LaunchCalcInitScoreKernel(class_id));
}


CUDARegressionQuantileloss::CUDARegressionQuantileloss(const Config& config):
CUDAPercentileRegressionObjective<RegressionQuantileloss>(config) {}

CUDARegressionQuantileloss::CUDARegressionQuantileloss(const std::vector<std::string>& strs):
CUDAPercentileRegressionObjective<RegressionQuantileloss>(strs) {}

CUDARegressionQuantileloss::~CUDARegressionQuantileloss() {}


}  // namespace Falcata

#endif  // USE_CUDA
