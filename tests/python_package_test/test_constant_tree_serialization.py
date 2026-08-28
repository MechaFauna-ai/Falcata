# coding: utf-8
"""Models containing constant (1-leaf) trees survive the FALB binary format.

A boosting round that finds no split worth making emits a CONSTANT tree: one
leaf, no internal nodes. It is an ordinary event -- in multiclass every class
that is already well fit produces them, and a real TabArena `anneal` model has
567 of them out of 1500 trees.

FALB lays its stats and diagnostics sections out BY COUNT: the loader owns no
per-array offsets for them, it slices each tree's block out of running totals
it derives from the records' num_leaves. The writer used to size those blocks
from the source vectors' `.size()` instead, which is an allocation artifact and
not the model:

  * a text-loaded 1-leaf tree has an EMPTY leaf_weight_ (the Tree text
    constructor returns before it sizes it), so the writer emitted 8 bytes per
    constant tree too FEW and the deficit accumulated until some later tree's
    read ran off the end -- `pickle.dumps(booster)` then failed to reload at
    all, since FALB is the default pickle payload;
  * a CPU-trained tree keeps its arrays at the max_leaves the learner reserved,
    so a tree that stopped short wrote too MANY, and every following tree's
    statistics silently decoded from the wrong offset.

Both directions are covered here, for every writer option combination.
"""

import os
import pickle

import numpy as np
import pytest

import falcata as lgb

_DEVICES = ["cpu"] + (["cuda"] if os.environ.get("TASK", "") == "cuda" else [])


def _leaf_counts(booster):
    chunks = booster.model_to_string().split("\nTree=")[1:]
    return [int(c.split("num_leaves=")[1].split("\n")[0]) for c in chunks]


def _all_constant_multiclass(device):
    """min_data_in_leaf above what any split can satisfy: every tree is constant."""
    rng = np.random.default_rng(0)
    x = rng.random((300, 6))
    y = rng.integers(0, 3, 300)
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "device_type": device,
        "min_data_in_leaf": 200,
        "verbosity": -1,
    }
    return x, lgb.train(params, lgb.Dataset(x, label=y, params=params), num_boost_round=10)


def _mixed_multiclass(device):
    """A gain floor some classes clear and others do not: constant and grown trees."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3000, 6))
    y = np.zeros(3000, dtype=int)
    y[x[:, 0] > 1.2] = 2
    rest = x[:, 0] <= 1.2
    y[rest] = (x[rest, 1] + 0.5 * rng.standard_normal(int(rest.sum())) > 0).astype(int)
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "device_type": device,
        "num_leaves": 15,
        "min_gain_to_split": 1.0,
        "verbosity": -1,
        "seed": 1,
    }
    return x, lgb.train(params, lgb.Dataset(x, label=y, params=params), num_boost_round=60)


def _constant_binary(device):
    rng = np.random.default_rng(11)
    x = rng.random((300, 5))
    y = rng.integers(0, 2, 300)
    params = {
        "objective": "binary",
        "device_type": device,
        "min_data_in_leaf": 200,
        "verbosity": -1,
    }
    return x, lgb.train(params, lgb.Dataset(x, label=y, params=params), num_boost_round=10)


@pytest.mark.parametrize("device", _DEVICES)
def test_constant_tree_model_survives_pickle(device):
    x, booster = _all_constant_multiclass(device)
    leaves = _leaf_counts(booster)
    assert leaves
    assert all(n == 1 for n in leaves), leaves

    reloaded = pickle.loads(pickle.dumps(booster))
    np.testing.assert_array_equal(reloaded.predict(x), booster.predict(x))
    assert reloaded.model_to_string() == booster.model_to_string()


@pytest.mark.parametrize("device", _DEVICES)
def test_mixed_constant_and_grown_trees_survive_pickle(device):
    x, booster = _mixed_multiclass(device)
    leaves = _leaf_counts(booster)
    assert min(leaves) == 1, leaves
    assert max(leaves) > 1, leaves

    reloaded = pickle.loads(pickle.dumps(booster))
    np.testing.assert_array_equal(reloaded.predict(x), booster.predict(x))
    assert reloaded.model_to_string() == booster.model_to_string()


@pytest.mark.parametrize("device", _DEVICES)
def test_constant_binary_model_survives_pickle(device):
    x, booster = _constant_binary(device)
    assert all(n == 1 for n in _leaf_counts(booster))

    reloaded = pickle.loads(pickle.dumps(booster))
    np.testing.assert_array_equal(reloaded.predict(x), booster.predict(x))


@pytest.mark.parametrize("with_stats", [False, True])
@pytest.mark.parametrize("with_diagnostics", [False, True])
@pytest.mark.parametrize("model", [_all_constant_multiclass, _mixed_multiclass])
def test_constant_trees_roundtrip_under_every_writer_option(model, with_stats, with_diagnostics):
    x, booster = model("cpu")
    blob = booster.model_to_binary(with_stats=with_stats, with_diagnostics=with_diagnostics)
    reloaded = lgb.Booster(model_bin=blob)
    np.testing.assert_array_equal(reloaded.predict(x), booster.predict(x))
    if with_stats and with_diagnostics:
        # the full section set is the one that must reproduce the text exactly
        assert reloaded.model_to_string() == booster.model_to_string()


def test_constant_tree_stats_are_not_shifted_between_trees():
    """The failure mode a size check alone would miss: readable, but shifted.

    Every tree's leaf_count must still sum to the training row count, which it
    cannot if a tree's block is read from another tree's offset.
    """
    x, booster = _mixed_multiclass("cpu")
    dumped = lgb.Booster(model_bin=booster.model_to_binary(with_stats=True, with_diagnostics=True)).dump_model()
    for tree in dumped["tree_info"]:
        counts = []
        stack = [tree["tree_structure"]]
        while stack:
            node = stack.pop()
            if "leaf_count" in node:
                counts.append(node["leaf_count"])
            else:
                stack += [node["left_child"], node["right_child"]]
        assert sum(counts) == len(x)


def _short_trained_booster(**train_kwargs):
    """A CPU booster still holding the learner's trees, every tree short of num_leaves.

    Such a tree keeps its stats arrays at the max_leaves the learner reserved.
    Sizing the sections from those widths put each tree's block at an offset
    the loader does not read from -- no error, just another tree's numbers.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2000, 8))
    y = x @ rng.standard_normal(8)
    params = {
        "objective": "regression",
        "device_type": "cpu",
        "num_leaves": 31,
        "min_data_in_leaf": 300,
        "verbosity": -1,
    }
    booster = lgb.train(
        params, lgb.Dataset(x, label=y, params=params), num_boost_round=10, keep_training_booster=True, **train_kwargs
    )
    assert max(_leaf_counts(booster)) < 31  # every tree stopped short
    return x, booster


def test_short_trained_trees_keep_their_own_statistics():
    x, booster = _short_trained_booster()
    blob = booster.model_to_binary(with_stats=True, with_diagnostics=True)
    reloaded = lgb.Booster(model_bin=blob)
    assert reloaded.model_to_string() == booster.model_to_string()
    np.testing.assert_array_equal(reloaded.predict(x), booster.predict(x))


def test_short_trained_trees_keep_what_the_statistics_are_for():
    """The consumers, not just the bytes.

    Predictions never depended on these sections, so a shifted block is
    invisible until something reads them: gain importance is computed from
    split_gain, split importance only counts splits whose gain is positive,
    and TreeSHAP divides by each node's data count -- which is zero in a block
    that belongs to another tree, so contributions come back non-finite.
    """
    x, booster = _short_trained_booster()
    reloaded = lgb.Booster(model_bin=booster.model_to_binary(with_stats=True, with_diagnostics=True))

    np.testing.assert_array_equal(reloaded.feature_importance("gain"), booster.feature_importance("gain"))
    np.testing.assert_array_equal(reloaded.feature_importance("split"), booster.feature_importance("split"))

    want = booster.predict(x[:20], pred_contrib=True)
    got = reloaded.predict(x[:20], pred_contrib=True)
    assert np.isfinite(got).all()
    np.testing.assert_array_equal(got, want)
