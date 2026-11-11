"""Scoring a model that holds stumps (single-leaf trees) on CUDA.

GBDT rebuilds a training score from the model's trees when the training data is reset (Booster.update(train_set=...))
and a validation score when a validation set is added. CUDATree's scoring kernel traversed every tree from node 0, so a
stump read split_feature_inner[0] of an empty tree, a null device pointer ("an illegal memory access was encountered",
which also aborted the process when the booster was freed).
"""

from os import getenv

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(getenv("TASK", "") != "cuda", reason="CUDA-only regression test")

# min_gain_to_split makes the middle class's one-vs-all trees stumps once its gradients shrink, after a first
# iteration in which every class splits
_PARAMS = {
    "objective": "multiclassova",
    "num_class": 3,
    "device_type": "cuda",
    "num_leaves": 7,
    "learning_rate": 0.3,
    "min_gain_to_split": 50.0,
    "deterministic": True,
    "quant_mode": "none",
    "verbose": -1,
    "seed": 7,
}


def _trained_booster():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((1000, 2))
    y = np.digitize(X[:, 0], [-0.4, 0.4]).astype(float)
    train = flc.Dataset(X[:600], y[:600], params=_PARAMS, free_raw_data=False)
    booster = flc.Booster(_PARAMS, train)
    for _ in range(6):
        booster.update()
    num_leaves = [tree["num_leaves"] for tree in booster.dump_model()["tree_info"]]
    assert min(num_leaves[:3]) > 1 and min(num_leaves) == 1, f"the fixture must produce later stumps: {num_leaves}"
    return booster, train, X, y


def test_cuda_reset_training_data_scores_stumps():
    booster, train, X, y = _trained_booster()
    new_train = flc.Dataset(X[600:], y[600:], params=_PARAMS, reference=train)
    booster.update(train_set=new_train)  # rescored every tree, stumps included, on the new data
    scores = np.asarray(booster._Booster__inner_predict(data_idx=0)).reshape(400, -1)
    np.testing.assert_allclose(scores, booster.predict(X[600:]).reshape(400, -1), rtol=1e-9)
    booster.update()
    assert np.all(np.isfinite(booster.predict(X)))


def test_cuda_validation_set_added_after_stumps():
    booster, train, X, y = _trained_booster()
    booster.add_valid(flc.Dataset(X[600:], y[600:], params=_PARAMS, reference=train), "valid")
    scores = np.asarray(booster._Booster__inner_predict(data_idx=1)).reshape(400, -1)
    np.testing.assert_allclose(scores, booster.predict(X[600:]).reshape(400, -1), rtol=1e-9)
