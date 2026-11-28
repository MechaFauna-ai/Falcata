/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 *
 * Device-agnostic core of the linear-tree leaf fit. The per-leaf normal
 * equations (XtHX, Xtg) are accumulated per backend -- an OMP loop on CPU, a
 * CUDA kernel on GPU -- but the ridge-regularized solve, coefficient pruning,
 * and storage onto the Tree are identical, so they live here once and are
 * shared by LinearTreeLearner (CPU/OpenCL) and the CUDA tree learner.
 */
#ifndef FALCATA_TREELEARNER_LINEAR_LEAF_SOLVER_H_
#define FALCATA_TREELEARNER_LINEAR_LEAF_SOLVER_H_

#include <Falcata/dataset.h>
#include <Falcata/meta.h>
#include <Falcata/tree.h>

#include <vector>

namespace Falcata {

namespace LinearLeafSolver {

// Solve coeffs = -(XtHX + lambda*I_features)^-1 * Xtg for one leaf, prune
// near-zero coefficients (non-refit) or blend with the old model (refit), and
// store the resulting linear model onto the tree.
//   xthx: upper-triangle (row-major) of the (num_feat+1)x(num_feat+1) gram,
//         length (num_feat+1)*(num_feat+2)/2
//   xtg:  length num_feat+1 (last entry is the constant term)
//   features_inner: num_feat inner feature indices, paired with the first
//                   num_feat coefficients
void SolveAndStore(Tree* tree, int leaf, int num_feat,
                   const double* xthx, const double* xtg,
                   const std::vector<int>& features_inner,
                   const Dataset* train_data, double lambda,
                   bool is_refit, double decay_rate, double shrinkage,
                   const std::vector<int>& old_coeff_idx);

}  // namespace LinearLeafSolver

}  // namespace Falcata

#endif  // FALCATA_TREELEARNER_LINEAR_LEAF_SOLVER_H_
