# coding: utf-8
"""CUDA categorical split record: the learner-owned growth buffers (CUDATreeCatArena) and what a tree keeps.

Each tree takes the learner's categorical record buffers (both bitsets and boundary arrays) over, appends one
bitset per categorical split, reads the record back in ToHost and hands the buffers back to the learner for the next
tree, keeping exact-size device copies of the inner bitset and boundaries. Those copies are what
AddPredictionToScore traverses for the tree's whole lifetime: the validation scores, the out-of-bag pass under
bagging and DART's re-scoring of dropped trees.

Per shape, the test checks that
- two trainings give the same model (md5 of the model text without its parameters),
- the model survives a text round trip (same text, same predictions),
- the scores the booster maintained on the device (validation set; the training set under bagging) equal the host
  prediction of the final model, i.e. every retained tree's device record matches its host record.
The shapes cover one-hot and many-vs-many splits, a high-cardinality feature whose per-tree record exceeds the
1 MiB presize (so the buffers grow while a tree records), multiclass, DART, bagging and a random forest grown on
concurrent forest slots (one learner, so one arena, per slot and host thread).
"""

import hashlib
import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

#: CUDATreeCatArena presize cap in words (cuda_single_gpu_tree_learner.cpp AllocateBitset)
_PRESIZE_CAP_WORDS = 1 << 18


def _data(rows, seed, cards=(12, 40, 200), high=None, num_class=2):
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, c, rows).astype(np.float64) for c in cards]
    if high is not None:
        # 200 distinct category values spread up to `high`: the real-value bitset of a split holding the largest
        # ones is ~high/32 words long
        values = np.random.default_rng(99).choice(high, 200, replace=False)
        cols.append(rng.choice(values, rows).astype(np.float64))
    num = rng.standard_normal((rows, 4))
    effect = num[:, 0] - 0.5 * num[:, 1]
    for j, c in enumerate(cols):
        _, codes = np.unique(c, return_inverse=True)
        effect += np.random.default_rng(10 + j).standard_normal(codes.max() + 1)[codes]
    if num_class > 2:
        y = np.digitize(effect, np.quantile(effect, np.linspace(0, 1, num_class + 1)[1:-1])).astype(np.float64)
    else:
        y = (effect + 0.5 * rng.standard_normal(rows) > 0).astype(np.float64)
    return np.column_stack([*cols, num]), y, list(range(len(cols)))


def _strip_parameters(model_str):
    out, skip = [], False
    for line in model_str.splitlines():
        if line.startswith("parameters:"):
            skip = True
        if not skip:
            out.append(line)
        if line.startswith("end of parameters"):
            skip = False
    return "\n".join(out)


def _md5(text):
    return hashlib.md5(_strip_parameters(text).encode()).hexdigest()


BASE = {
    "device_type": "cuda",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "max_depth": 5,
    "max_bin": 63,
    "min_data_in_leaf": 20,
    "learning_rate": 0.1,
    "seed": 7,
    "verbose": -1,
}
SHAPES = {
    "one_hot": ({"objective": "regression", "metric": "l2", "max_cat_to_onehot": 256}, {}, 20),
    "many_vs_many": ({}, {}, 20),
    "high_cardinality": (
        {"num_leaves": 63, "max_depth": 6, "max_cat_threshold": 128, "min_data_per_group": 5, "cat_smooth": 1},
        {"high": 2_000_000},
        3,
    ),
    "multiclass": ({"objective": "multiclass", "num_class": 3, "metric": "multi_logloss"}, {"num_class": 3}, 15),
    "dart": ({"boosting": "dart", "drop_rate": 0.3}, {}, 20),
    "bagging": ({"bagging_fraction": 0.7, "bagging_freq": 1, "metric": "auc"}, {}, 20),
    "rf_forest_slots": (
        {"boosting": "rf", "bagging_fraction": 0.632, "bagging_freq": 1, "cuda_plan": "auto,forest_group:4"},
        {},
        12,
    ),
}


def _train(params, data_kw, rounds):
    x, y, cats = _data(20000, 3, **data_kw)
    xv, yv, _ = _data(5000, 4, **data_kw)
    ds = flc.Dataset(x, y, categorical_feature=cats, params=params, free_raw_data=False)
    dv = flc.Dataset(xv, yv, reference=ds, categorical_feature=cats)
    bst = flc.train(params, ds, num_boost_round=rounds, valid_sets=[dv], valid_names=["v"], keep_training_booster=True)
    return bst, x, xv


def _device_scores(bst, data_idx, rows, num_class):
    """The scores the booster maintained on the device, in output space, shaped like Booster.predict."""
    score = bst._Booster__inner_predict(data_idx=data_idx)
    return score.reshape(rows, num_class) if num_class > 1 else score


def _per_tree_words(text, key):
    return [len(line.split("=", 1)[1].split()) for line in text.splitlines() if line.startswith(key + "=")]


@_REQUIRES_CUDA
@pytest.mark.parametrize("shape", list(SHAPES))
def test_cuda_categorical_record(shape):
    extra, data_kw, rounds = SHAPES[shape]
    params = {**BASE, **extra}
    bst, x, xv = _train(params, data_kw, rounds)
    text = bst.model_to_string()
    assert sum(int(line[8:]) for line in text.splitlines() if line.startswith("num_cat=")) > 0, "no categorical split"

    # repeated training: the reused buffers carry no state from one training (or tree) into the model
    again, _, _ = _train(params, data_kw, rounds)
    assert _md5(again.model_to_string()) == _md5(text)

    # text round trip
    loaded = flc.Booster(model_str=text)
    assert _md5(loaded.model_to_string()) == _md5(text)
    np.testing.assert_array_equal(loaded.predict(xv), bst.predict(xv))

    # device-maintained scores (retained inner bitsets) == host prediction of the final model
    num_class = params.get("num_class", 1)
    # (both in output space: the booster's inner prediction applies the objective's output transform)
    np.testing.assert_allclose(_device_scores(bst, 1, xv.shape[0], num_class), bst.predict(xv), rtol=1e-9, atol=1e-12)
    if params.get("bagging_freq", 0) > 0 and params.get("boosting") != "rf":
        # the out-of-bag rows are scored through AddPredictionToScore on the finished tree
        np.testing.assert_allclose(_device_scores(bst, 0, x.shape[0], num_class), bst.predict(x), rtol=1e-9, atol=1e-12)

    if shape == "high_cardinality":
        # one tree's real-value record is longer than the presize, so the buffer grew while that tree recorded
        assert max(_per_tree_words(text, "cat_threshold")) > _PRESIZE_CAP_WORDS
    if shape == "rf_forest_slots":
        # concurrent slots grow the same forest as sequential training
        seq, _, _ = _train({**params, "cuda_plan": "auto,forest_batched:off"}, data_kw, rounds)
        assert _md5(seq.model_to_string()) == _md5(text)
