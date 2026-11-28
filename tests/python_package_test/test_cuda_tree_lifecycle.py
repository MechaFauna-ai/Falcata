# coding: utf-8
"""Regression tests for CUDATree lifecycle optimizations (W4-B2).

Covers:
1. Retained device arenas outliving learner resets, training-data replacement, and teardown.
2. Chunked device slab rollover for both numerical leaf values and categorical inner bitsets/boundaries.
3. Deferred root initialization on quantized and non-quantized stumps plus device_tree/one_sync fallbacks.
4. Feature-source upload cache invalidation across feature_fraction, bagging subsets, and ResetTrainingData.
5. Full-capacity (num_leaves == 2^max_depth) and 1-leaf trees exercising leaf_parent_ and serialization.
"""

import gc
import hashlib
import os

import numpy as np
import pytest

import falcata as flc

_REQUIRES_CUDA = pytest.mark.skipif(os.environ.get("TASK", "") != "cuda", reason="requires CUDA-enabled Falcata")


def _tree_md5(booster: flc.Booster) -> str:
    """Return an MD5 digest over the serialized tree section of a model."""
    tree_section = booster.model_to_string().split("\nparameters:")[0]
    return hashlib.md5(tree_section.encode("utf-8")).hexdigest()


def _make_dataset(n: int = 3000, f: int = 16, seed: int = 2026, with_cat: bool = False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float32)
    cats = []
    if with_cat:
        X[:, 0] = rng.integers(0, 12, size=n)
        X[:, 1] = rng.integers(0, 32, size=n)
        cats = [0, 1]
    y_reg = (
        1.4 * X[:, 0] - 0.9 * X[:, 1] + 0.35 * (X[:, 2] ** 2) - 0.5 * X[:, 3] * X[:, 4] + rng.standard_normal(n) * 0.15
    ).astype(np.float32)
    y_bin = (y_reg > np.median(y_reg)).astype(np.float32)
    y_mc = np.clip(np.floor((y_reg - y_reg.min()) / (y_reg.max() - y_reg.min() + 1e-6) * 4), 0, 3).astype(np.int32)
    return X, y_reg, y_bin, y_mc, cats


@_REQUIRES_CUDA
def test_tree_outlives_learner_reset_and_teardown():
    """Trees backed by refcounted device slabs must survive learner reset, ResetTrainingData, and dataset free."""
    X1, y1, _, _, cats = _make_dataset(n=2500, f=16, seed=11, with_cat=True)
    X2, y2, _, _, _ = _make_dataset(n=1200, f=16, seed=12, with_cat=True)
    Xv, yv, _, _, _ = _make_dataset(n=600, f=16, seed=13, with_cat=True)

    dtr1 = flc.Dataset(X1, label=y1, categorical_feature=cats, free_raw_data=False)
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "quant_mode": "fixedpoint",
        "max_depth": 6,
        "num_leaves": 63,
        "learning_rate": 0.1,
        "deterministic": True,
        "seed": 7,
        "verbose": -1,
    }
    bst = flc.train(params, dtr1, num_boost_round=12, keep_training_booster=True)
    pred_before = bst.predict(Xv)
    assert np.all(np.isfinite(pred_before))

    # Reset learner config (changes num_leaves), update, and rollback.
    bst.reset_parameter({"num_leaves": 15, "max_depth": 4})
    bst.update()
    bst.rollback_one_iter()
    np.testing.assert_allclose(bst.predict(Xv), pred_before, rtol=1e-12, atol=1e-12)

    # Replace training data (ResetTrainingData) and drop original Dataset.
    dtr2 = flc.Dataset(X2, label=y2, reference=dtr1, categorical_feature=cats, free_raw_data=False)
    dval = flc.Dataset(Xv, label=yv, reference=dtr1, categorical_feature=cats, free_raw_data=False)
    del dtr1
    gc.collect()

    bst.update(train_set=dtr2)
    bst.add_valid(dval, "val")
    eval_res = bst.eval_valid()
    assert len(eval_res) == 1
    assert np.isfinite(eval_res[0][2])

    # Refit against new data and verify round-trip serialization.
    refit_bst = bst.refit(Xv, yv, decay_rate=0.5)
    pred_refit = refit_bst.predict(Xv)
    assert np.all(np.isfinite(pred_refit))

    bst.rollback_one_iter()
    roundtrip = flc.Booster(model_str=bst.model_to_string())
    np.testing.assert_allclose(roundtrip.predict(Xv), pred_before, rtol=1e-12, atol=1e-12)


@_REQUIRES_CUDA
@pytest.mark.parametrize("with_cat", [False, True])
def test_arena_slab_rollover_and_truncation(with_cat):
    """Multi-slab allocation for retained leaf values (and categorical bitsets) must match baseline and rollback."""
    X, _, y_bin, y_mc, cats = _make_dataset(n=4096, f=16, seed=23, with_cat=with_cat)
    Xv, _, _, _, _ = _make_dataset(n=512, f=16, seed=24, with_cat=with_cat)

    # 36 rounds * 4 classes = 144 trees of up to 255 leaves (~290 KiB leaf values, crossing 64 KiB + 128 KiB slabs)
    # or 64 binary categorical trees crossing categorical arena slabs as well.
    if with_cat:
        dtr = flc.Dataset(X, label=y_bin, categorical_feature=cats, free_raw_data=False)
        base_params = {
            "device_type": "cuda",
            "objective": "binary",
            "quant_mode": "fixedpoint",
            "max_depth": 8,
            "num_leaves": 255,
            "min_data_in_leaf": 5,
            "learning_rate": 0.08,
            "deterministic": True,
            "seed": 19,
            "verbose": -1,
        }
        rounds = 64
    else:
        dtr = flc.Dataset(X, label=y_mc, free_raw_data=False)
        base_params = {
            "device_type": "cuda",
            "objective": "multiclass",
            "num_class": 4,
            "quant_mode": "fixedpoint",
            "max_depth": 8,
            "num_leaves": 255,
            "min_data_in_leaf": 5,
            "learning_rate": 0.08,
            "deterministic": True,
            "seed": 19,
            "verbose": -1,
        }
        rounds = 36

    bst_auto = flc.train({**base_params, "cuda_plan": "auto"}, dtr, num_boost_round=rounds, keep_training_booster=True)
    bst_off = flc.train({**base_params, "cuda_plan": "auto,device_levels:off"}, dtr, num_boost_round=rounds)

    assert _tree_md5(bst_auto) == _tree_md5(bst_off)
    np.testing.assert_allclose(bst_auto.predict(Xv), bst_off.predict(Xv), rtol=0.0, atol=0.0)

    # Truncate via rollback_one_iter and verify predictions match prefix of reference model.
    keep_rounds = rounds - 5
    for _ in range(5):
        bst_auto.rollback_one_iter()
    np.testing.assert_allclose(
        bst_auto.predict(Xv),
        bst_off.predict(Xv, num_iteration=keep_rounds),
        rtol=1e-12,
        atol=1e-12,
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("quant_mode", "plan"),
    [
        ("fixedpoint", "auto"),
        ("none", "auto"),
        ("fixedpoint", "auto,device_tree:off"),
        ("none", "auto,one_sync:off"),
    ],
)
def test_deferred_root_init_stumps_and_fallbacks(quant_mode, plan):
    """Deferred root init must produce bit-identical models on stumps and non-device-tree fallbacks."""
    X, y_reg, _, _, _ = _make_dataset(n=1000, f=12, seed=31, with_cat=False)
    dtr = flc.Dataset(X, label=y_reg, free_raw_data=False)
    common = {
        "device_type": "cuda",
        "objective": "regression",
        "quant_mode": quant_mode,
        "max_depth": 5,
        "num_leaves": 31,
        "deterministic": True,
        "seed": 5,
        "verbose": -1,
    }

    # Case 1: forced 1-leaf stumps (0 splits).
    stump_params = {**common, "min_data_in_leaf": 1000}
    bst_stump = flc.train({**stump_params, "cuda_plan": plan}, dtr, num_boost_round=4)
    bst_stump_ref = flc.train({**stump_params, "cuda_plan": "auto,device_levels:off"}, dtr, num_boost_round=4)
    assert _tree_md5(bst_stump) == _tree_md5(bst_stump_ref)
    assert bst_stump.num_trees() == 1
    preds = bst_stump.predict(X[:32])
    assert np.all(np.isfinite(preds))
    np.testing.assert_allclose(preds, bst_stump_ref.predict(X[:32]), rtol=0.0, atol=0.0)

    # Case 2: normal splits under the same plan/fallback.
    bst_norm = flc.train({**common, "cuda_plan": plan}, dtr, num_boost_round=8)
    bst_norm_ref = flc.train({**common, "cuda_plan": "auto,device_levels:off"}, dtr, num_boost_round=8)
    assert _tree_md5(bst_norm) == _tree_md5(bst_norm_ref)


@_REQUIRES_CUDA
def test_feature_source_cache_invalidation_paths():
    """Feature-source cache must stay bit-identical across feature_fraction, bagging subsets, and ResetTrainingData."""
    X, y_reg, _, _, _ = _make_dataset(n=2400, f=20, seed=41, with_cat=False)
    dtr_full = flc.Dataset(X, label=y_reg, free_raw_data=False)
    dtr_sub1 = dtr_full.subset(np.arange(0, 1200, dtype=np.int32))
    dtr_sub2 = dtr_full.subset(np.arange(400, 2200, dtype=np.int32))

    base = {
        "device_type": "cuda",
        "objective": "regression",
        "quant_mode": "fixedpoint",
        "max_depth": 6,
        "num_leaves": 63,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "deterministic": True,
        "seed": 13,
        "verbose": -1,
    }

    def _run_sequence(plan: str) -> str:
        bst = flc.train({**base, "cuda_plan": plan}, dtr_sub1, num_boost_round=6, keep_training_booster=True)
        for _ in range(4):
            bst.update(train_set=dtr_sub2)
        for _ in range(4):
            bst.update(train_set=dtr_full)
        return _tree_md5(bst)

    assert _run_sequence("auto") == _run_sequence("auto,device_levels:off")


@_REQUIRES_CUDA
@pytest.mark.parametrize(("max_depth", "num_leaves"), [(4, 16), (5, 32)])
def test_full_capacity_trees_leaf_parent_and_refit(max_depth, num_leaves):
    """Full-capacity trees (num_leaves == 2^max_depth) must serialize, predict, and refit bit-identically."""
    X, y_reg, _, _, _ = _make_dataset(n=2500, f=16, seed=53, with_cat=False)
    Xv, yv, _, _, _ = _make_dataset(n=500, f=16, seed=54, with_cat=False)
    dtr = flc.Dataset(X, label=y_reg, free_raw_data=False)

    base = {
        "device_type": "cuda",
        "objective": "regression",
        "quant_mode": "fixedpoint",
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_data_in_leaf": 5,
        "learning_rate": 0.1,
        "deterministic": True,
        "seed": 29,
        "verbose": -1,
    }
    bst_auto = flc.train({**base, "cuda_plan": "auto"}, dtr, num_boost_round=10, keep_training_booster=True)
    bst_off = flc.train(
        {**base, "cuda_plan": "auto,device_levels:off"}, dtr, num_boost_round=10, keep_training_booster=True
    )

    assert _tree_md5(bst_auto) == _tree_md5(bst_off)
    dump = bst_auto.dump_model()
    assert any(t["num_leaves"] == num_leaves for t in dump["tree_info"])

    refit_auto = bst_auto.refit(Xv, yv, decay_rate=0.5)
    refit_off = bst_off.refit(Xv, yv, decay_rate=0.5)
    assert _tree_md5(refit_auto) == _tree_md5(refit_off)
    np.testing.assert_allclose(refit_auto.predict(Xv), refit_off.predict(Xv), rtol=0.0, atol=0.0)
