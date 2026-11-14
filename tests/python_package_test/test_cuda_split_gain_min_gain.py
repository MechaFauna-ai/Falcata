"""CUDA trees store each split's gain as the CPU learner does: with min_gain_to_split added back.

SerialTreeLearner::Split stores best_split_info.gain + min_gain_to_split (the gain over the parent's gain). The CUDA
split finders report the gain over parent gain + min_gain_to_split, and the CUDA learner stored that, so with
min_gain_to_split > 0 the model text's split_gain and feature_importance(importance_type="gain") were lower than the
CPU model's by min_gain_to_split per split.
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only regression test")


@pytest.mark.parametrize("min_gain_to_split", [0.0, 0.5])
@pytest.mark.parametrize(
    "extra",
    [{}, {"cuda_plan": "auto,hybrid:off"}, {"cuda_plan": "auto,device_tree:off"}, {"linear_tree": True}],
    ids=["default", "hybrid_off", "device_tree_off", "linear"],
)
def test_cuda_split_gain_matches_cpu(min_gain_to_split, extra):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5000, 6))
    y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.3 * rng.standard_normal(5000)
    boosters = {}
    for device in ("cpu", "cuda"):
        params = {
            "objective": "regression",
            "device_type": device,
            "num_leaves": 31,
            "min_gain_to_split": min_gain_to_split,
            "deterministic": True,
            "verbose": -1,
            **extra,
        }
        if device == "cuda":
            params["quant_mode"] = "none"
        else:
            params.pop("cuda_plan", None)
        boosters[device] = flc.train(params, flc.Dataset(X, y, params=params), num_boost_round=5)
    # (31 leaves: deeper trees can part ways with min_gain_to_split > 0, because the CPU learner stops searching a
    # feature in a leaf whose parent found no split on it above the threshold, and the CUDA learner does not)
    cpu, cuda = (boosters[d].trees_to_dataframe() for d in ("cpu", "cuda"))
    np.testing.assert_array_equal(cuda["split_feature"].fillna(""), cpu["split_feature"].fillna(""))
    np.testing.assert_allclose(cuda["split_gain"], cpu["split_gain"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        boosters["cuda"].feature_importance(importance_type="gain"),
        boosters["cpu"].feature_importance(importance_type="gain"),
        rtol=1e-6,
    )
