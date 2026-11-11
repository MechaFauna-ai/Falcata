"""Models written by stock LightGBM 3.x must load.

LightGBM 3.x writes a constant (1-leaf) tree -- e.g. every tree of a class
that never occurs in the labels -- with EMPTY ``leaf_count=`` and
``leaf_weight=`` lines. Falcata accepted the empty leaf_weight but parsed
leaf_count as one required value and refused the whole model with "Malformed
model file: not enough values in string for array of size 1". In fuzzing,
143 of 269 random LightGBM 3.3.5 models failed to load this way.
"""

import numpy as np

import falcata as flc

# lightgbm==3.3.5: multiclass, num_class=3, class 1 absent, 2 rounds (parameters block trimmed)
_LGB3_MODEL = "tree\nversion=v3\nnum_class=3\nnum_tree_per_iteration=3\nlabel_index=0\nmax_feature_idx=1\nobjective=multiclass num_class:3\nfeature_names=Column_0 Column_1\nfeature_infos=[-3.1063368012832915:2.7558075582759511] [-3.7722751561227339:3.0660367390488967]\ntree_sizes=456 239 456 447 221 449\n\nTree=0\nnum_leaves=3\nnum_cat=0\nsplit_feature=0 0\nsplit_gain=133.333 1.42109e-14\nthreshold=1.0000000180025095e-35 0.95212359741877872\ndecision_type=2 2\nleft_child=-1 -2\nright_child=1 -3\nleaf_value=-0.49272934347430941 -0.76111368562481663 -0.76111368562481663\nleaf_weight=40.240798830986023 23.47379931807518 10.805399686098097\nleaf_count=108 63 29\ninternal_value=-0.616186 -0.761114\ninternal_weight=0 34.2792\ninternal_count=200 92\nis_linear=0\nshrinkage=1\n\n\nTree=1\nnum_leaves=1\nnum_cat=0\nsplit_feature=\nsplit_gain=\nthreshold=\ndecision_type=\nleft_child=\nright_child=\nleaf_value=-34.538776391283193\nleaf_weight=\nleaf_count=\ninternal_value=\ninternal_weight=\ninternal_count=\nis_linear=0\nshrinkage=1\n\n\nTree=2\nnum_leaves=3\nnum_cat=0\nsplit_feature=0 0\nsplit_gain=133.333 1.42109e-14\nthreshold=1.0000000180025095e-35 0.95212359741877872\ndecision_type=2 2\nleft_child=-1 -2\nright_child=1 -3\nleaf_value=-0.89998558544850382 -0.63160124329799661 -0.63160124329799661\nleaf_weight=40.240798830986023 23.47379931807518 10.805399686098097\nleaf_count=108 63 29\ninternal_value=-0.776529 -0.631601\ninternal_weight=0 34.2792\ninternal_count=200 92\nis_linear=0\nshrinkage=1\n\n\nTree=3\nnum_leaves=3\nnum_cat=0\nsplit_feature=0 0\nsplit_gain=101.797 7.10543e-15\nthreshold=1.0000000180025095e-35 -0.34593722783176895\ndecision_type=2 2\nleft_child=1 -1\nright_child=-2 -3\nleaf_value=0.11103157199099647 -0.12523491467957576 0.11103157199099647\nleaf_weight=30.229146480560303 34.355731248855591 8.636898994445799\nleaf_count=84 92 24\ninternal_value=0 0.111032\ninternal_weight=0 38.866\ninternal_count=200 108\nis_linear=0\nshrinkage=0.1\n\n\nTree=4\nnum_leaves=1\nnum_cat=0\nsplit_feature=\nsplit_gain=\nthreshold=\ndecision_type=\nleft_child=\nright_child=\nleaf_value=0\nleaf_weight=\nleaf_count=\ninternal_value=\ninternal_weight=\ninternal_count=\nis_linear=0\nshrinkage=1\n\n\nTree=5\nnum_leaves=3\nnum_cat=0\nsplit_feature=0 0\nsplit_gain=101.797 7.10543e-15\nthreshold=1.0000000180025095e-35 -0.34593722783176895\ndecision_type=2 2\nleft_child=1 -1\nright_child=-2 -3\nleaf_value=-0.11103157199099647 0.12523491467957576 -0.11103157199099647\nleaf_weight=30.229146480560303 34.355731248855591 8.636898994445799\nleaf_count=84 92 24\ninternal_value=0 -0.111032\ninternal_weight=0 38.866\ninternal_count=200 108\nis_linear=0\nshrinkage=0.1\n\n\nend of trees\n\nfeature_importances:\nColumn_0=8\n\n"

# lightgbm 3.3.5 predict(raw_score=True) on the rows below
_LGB3_RAW = np.array(
    [
        [-0.8863486, -34.53877639, -0.50636633],
        [-0.38169777, -34.53877639, -1.01101716],
        [-0.38169777, -34.53877639, -1.01101716],
    ]
)


def test_lightgbm3_constant_trees_load_and_predict():
    booster = flc.Booster(model_str=_LGB3_MODEL)
    x = np.array([[1.0, 0.0], [-1.0, 0.5], [np.nan, 2.0]])
    np.testing.assert_allclose(booster.predict(x, raw_score=True), _LGB3_RAW, rtol=1e-7)
    # and it survives Falcata's own formats
    np.testing.assert_array_equal(
        flc.Booster(model_bin=booster.model_to_binary(with_stats=True)).predict(x, raw_score=True),
        booster.predict(x, raw_score=True),
    )
