# coding: utf-8
"""CUDA vector-leaf multi-target training: one shared-structure tree per
iteration whose leaves carry one output per target, split gain summed over
targets.

Everything here needs a CUDA build (TASK=cuda). Correctness is checked
through invariants rather than a golden model, because non-quantized CUDA
training is run-to-run nondeterministic (fp64 atomic histograms):

* duplicated targets: y2 == y1 doubles every candidate's gain, so the greedy
  structure must match scalar training on y1, and both targets' outputs must
  match the scalar model's;
* negated targets: y2 == -y1 leaves every gain unchanged (gains are even in
  the gradient), so predictions must be antisymmetric across targets;
* the loss on each target must drop well below the label variance; and
* the trained model must round-trip through text and FALB serialization.
"""

import os
import tempfile

import numpy as np
import pytest

import falcata as lgb

_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

N_ROWS = 4000
N_FEATURES = 10


def _make_data(seed=20260827):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_ROWS, N_FEATURES))
    y0 = 2.0 * X[:, 0] - 1.5 * X[:, 3] + np.where(X[:, 5] > 0.25, 1.0, -1.0) + 0.05 * rng.normal(size=N_ROWS)
    y1 = -1.0 * X[:, 0] + 0.5 * X[:, 7] + np.where(X[:, 2] > -0.5, 0.5, -0.5) + 0.05 * rng.normal(size=N_ROWS)
    return X, y0, y1


def _base_params():
    return {
        "device_type": "cuda",
        "quant_mode": "none",
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "seed": 7,
        "verbosity": -1,
    }


def _train_vector(X, labels, num_boost_round=20, extra_params=None):
    params = _base_params()
    params.update(
        {
            "objective": "multi_regression",
            "metric": "multi_rmse",
            "num_class": labels.shape[1],
            "tree_mode": "vector_leaf",
        }
    )
    if extra_params:
        params.update(extra_params)
    return lgb.train(params, lgb.Dataset(X, label=labels), num_boost_round=num_boost_round)


def _train_scalar(X, y, num_boost_round=20, extra_params=None):
    params = _base_params()
    params.update({"objective": "regression", "boost_from_average": False})
    if extra_params:
        params.update(extra_params)
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=num_boost_round)


@_REQUIRES_CUDA
def test_vector_leaf_cuda_train_predict_save_load_roundtrip():
    X, y0, y1 = _make_data()
    labels = np.column_stack((y0, y1))
    booster = _train_vector(X, labels)

    prediction = booster.predict(X)
    assert prediction.shape == (N_ROWS, 2)
    assert np.all(np.isfinite(prediction))

    # each target's fit must beat the trivial (mean) predictor by a wide margin
    for target in range(2):
        residual = prediction[:, target] - labels[:, target]
        assert np.mean(residual**2) < 0.25 * np.var(labels[:, target])

    # the model text carries per-tree vector leaves and reloads identically
    model_text = booster.model_to_string()
    assert model_text.count("leaf_value_dim=2") == booster.num_trees()
    assert "num_tree_per_iteration=1" in model_text
    reloaded = lgb.Booster(model_str=model_text)
    np.testing.assert_array_equal(reloaded.predict(X), prediction)

    # FALB binary round trip
    blob = booster.model_to_binary()
    from_binary = lgb.Booster(model_str=model_text).model_from_binary(blob)
    np.testing.assert_array_equal(from_binary.predict(X), prediction)

    # file round trip
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "vector_leaf.txt")
        booster.save_model(path)
        from_file = lgb.Booster(model_file=path)
        np.testing.assert_array_equal(from_file.predict(X), prediction)


@_REQUIRES_CUDA
def test_vector_leaf_cuda_duplicated_target_matches_scalar():
    X, y0, _ = _make_data()
    labels = np.column_stack((y0, y0))
    vector = _train_vector(X, labels)
    scalar = _train_scalar(X, y0)

    # duplicating the target doubles every candidate gain: the greedy
    # structure equals scalar training's, so rows land in the same leaves
    vector_leaves = vector.predict(X, pred_leaf=True)
    scalar_leaves = scalar.predict(X, pred_leaf=True)
    np.testing.assert_array_equal(vector_leaves, scalar_leaves)

    scalar_prediction = scalar.predict(X)
    vector_prediction = vector.predict(X)
    for target in range(2):
        np.testing.assert_allclose(vector_prediction[:, target], scalar_prediction, rtol=1e-9, atol=1e-9)


@_REQUIRES_CUDA
def test_vector_leaf_cuda_negated_target_is_antisymmetric():
    X, y0, _ = _make_data()
    labels = np.column_stack((y0, -y0))
    booster = _train_vector(X, labels)
    prediction = booster.predict(X)
    np.testing.assert_allclose(prediction[:, 1], -prediction[:, 0], rtol=1e-9, atol=1e-9)


@_REQUIRES_CUDA
def test_vector_leaf_cuda_validation_metric_and_early_rounds():
    X, y0, y1 = _make_data()
    labels = np.column_stack((y0, y1))
    train_set = lgb.Dataset(X[: N_ROWS // 2], label=labels[: N_ROWS // 2])
    valid_set = train_set.create_valid(X[N_ROWS // 2 :], label=labels[N_ROWS // 2 :])
    params = _base_params()
    params.update(
        {
            "objective": "multi_regression",
            "metric": "multi_rmse",
            "num_class": 2,
            "tree_mode": "vector_leaf",
        }
    )
    evals = {}
    lgb.train(
        params,
        train_set,
        num_boost_round=20,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.record_evaluation(evals)],
    )
    curve = evals["valid"]["multi_rmse"]
    assert len(curve) == 20
    assert curve[-1] < curve[0]


@_REQUIRES_CUDA
def test_vector_leaf_cuda_rejects_goss_and_query_bagging():
    X, y0, y1 = _make_data()
    labels = np.column_stack((y0, y1))
    with pytest.raises(lgb.basic.FalcataError, match="GOSS"):
        _train_vector(
            X,
            labels,
            num_boost_round=1,
            extra_params={"data_sample_strategy": "goss"},
        )
    with pytest.raises(lgb.basic.FalcataError, match="bagging"):
        _train_vector(
            X,
            labels,
            num_boost_round=1,
            extra_params={
                "bagging_fraction": 0.5,
                "bagging_freq": 1,
                "bagging_by_query": True,
            },
        )


_BAGGING_PARAMS = {"bagging_fraction": 0.7, "bagging_freq": 1}


def _make_cat_data(seed=20260828):
    """Numerical features plus two declared categoricals: one low-cardinality
    (one-hot split path) and one 12-category (sorted many-vs-many path)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_ROWS, N_FEATURES))
    cat_small = rng.integers(0, 3, size=N_ROWS)
    cat_large = rng.integers(0, 12, size=N_ROWS)
    small_effect = np.array([1.5, -0.5, -1.0])[cat_small]
    large_effect = np.linspace(-2.0, 2.0, 12)[cat_large]
    y0 = 1.5 * X[:, 0] + small_effect + large_effect + 0.05 * rng.normal(size=N_ROWS)
    y1 = -0.8 * X[:, 2] + 0.5 * small_effect - large_effect + 0.05 * rng.normal(size=N_ROWS)
    X_full = np.column_stack((X, cat_small.astype(np.float64), cat_large.astype(np.float64)))
    cat_indices = [N_FEATURES, N_FEATURES + 1]
    return X_full, y0, y1, cat_indices


@_REQUIRES_CUDA
def test_vector_leaf_cuda_categorical_trains_and_roundtrips():
    X, y0, y1, cat_indices = _make_cat_data()
    labels = np.column_stack((y0, y1))
    params = _base_params()
    params.update(
        {
            "objective": "multi_regression",
            "metric": "multi_rmse",
            "num_class": 2,
            "tree_mode": "vector_leaf",
        }
    )
    booster = lgb.train(
        params,
        lgb.Dataset(X, label=labels, categorical_feature=cat_indices),
        num_boost_round=20,
    )
    prediction = booster.predict(X)
    assert prediction.shape == (N_ROWS, 2)
    assert np.all(np.isfinite(prediction))
    # the categorical effects dominate the targets: a fit that never splits on
    # them cannot reach this margin
    for target in range(2):
        residual = prediction[:, target] - labels[:, target]
        assert np.mean(residual**2) < 0.25 * np.var(labels[:, target])
    # at least one tree must actually use a categorical split
    model_text = booster.model_to_string()
    num_cat_counts = [int(line.split("=", 1)[1]) for line in model_text.splitlines() if line.startswith("num_cat=")]
    assert sum(num_cat_counts) > 0
    reloaded = lgb.Booster(model_str=model_text)
    np.testing.assert_array_equal(reloaded.predict(X), prediction)


@_REQUIRES_CUDA
def test_vector_leaf_cuda_categorical_duplicated_target_matches_scalar():
    X, y0, _, cat_indices = _make_cat_data()
    labels = np.column_stack((y0, y0))
    vec_params = _base_params()
    vec_params.update(
        {
            "objective": "multi_regression",
            "metric": "multi_rmse",
            "num_class": 2,
            "tree_mode": "vector_leaf",
        }
    )
    vector = lgb.train(
        vec_params,
        lgb.Dataset(X, label=labels, categorical_feature=cat_indices),
        num_boost_round=20,
    )
    scal_params = _base_params()
    scal_params.update({"objective": "regression", "boost_from_average": False})
    scalar = lgb.train(
        scal_params,
        lgb.Dataset(X, label=y0, categorical_feature=cat_indices),
        num_boost_round=20,
    )

    vector_leaves = vector.predict(X, pred_leaf=True)
    scalar_leaves = scalar.predict(X, pred_leaf=True)
    np.testing.assert_array_equal(vector_leaves, scalar_leaves)

    scalar_prediction = scalar.predict(X)
    vector_prediction = vector.predict(X)
    for target in range(2):
        np.testing.assert_allclose(vector_prediction[:, target], scalar_prediction, rtol=1e-9, atol=1e-9)


@_REQUIRES_CUDA
def test_vector_leaf_cuda_bagging_trains_and_loss_decreases():
    X, y0, y1 = _make_data()
    labels = np.column_stack((y0, y1))
    train_set = lgb.Dataset(X[: N_ROWS // 2], label=labels[: N_ROWS // 2])
    valid_set = train_set.create_valid(X[N_ROWS // 2 :], label=labels[N_ROWS // 2 :])
    params = _base_params()
    params.update(
        {
            "objective": "multi_regression",
            "metric": "multi_rmse",
            "num_class": 2,
            "tree_mode": "vector_leaf",
        }
    )
    params.update(_BAGGING_PARAMS)
    evals = {}
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=20,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.record_evaluation(evals)],
    )
    curve = evals["valid"]["multi_rmse"]
    assert len(curve) == 20
    assert curve[-1] < curve[0]
    prediction = booster.predict(X)
    assert prediction.shape == (N_ROWS, 2)
    assert np.all(np.isfinite(prediction))


@_REQUIRES_CUDA
def test_vector_leaf_cuda_bagging_duplicated_target_matches_scalar():
    # the bag is drawn per iteration from (bagging_seed, iter) only, so the
    # vector and scalar runs subset identical rows; duplicated targets then
    # reproduce the scalar bagged model exactly
    X, y0, _ = _make_data()
    labels = np.column_stack((y0, y0))
    vector = _train_vector(X, labels, extra_params=_BAGGING_PARAMS)
    scalar = _train_scalar(X, y0, extra_params=_BAGGING_PARAMS)

    vector_leaves = vector.predict(X, pred_leaf=True)
    scalar_leaves = scalar.predict(X, pred_leaf=True)
    np.testing.assert_array_equal(vector_leaves, scalar_leaves)

    scalar_prediction = scalar.predict(X)
    vector_prediction = vector.predict(X)
    for target in range(2):
        np.testing.assert_allclose(vector_prediction[:, target], scalar_prediction, rtol=1e-9, atol=1e-9)


# depth-limited config (2^max_depth <= num_leaves + 1): the regime in which
# plain level batching is exactly leaf-wise-equivalent, and therefore the regime
# vector-leaf training takes the hybrid level-batched prefix in
_HYBRID_PARAMS = {"num_leaves": 15, "max_depth": 4}


def _assert_same_leaf_partition(hybrid_leaves, classic_leaves):
    # Level-batched growth applies a whole level at once, so right children take
    # leaf indices in level order where the per-split loop numbers them in
    # best-gain order: the two labelings of the SAME partition differ. Assert the
    # partitions match by requiring the (hybrid leaf, classic leaf) pairs to form
    # a bijection within every tree.
    assert hybrid_leaves.shape == classic_leaves.shape
    for tree_index in range(hybrid_leaves.shape[1]):
        left = hybrid_leaves[:, tree_index]
        right = classic_leaves[:, tree_index]
        pairs = np.unique(np.stack([left, right], axis=1), axis=0)
        assert len(pairs) == len(np.unique(left)) == len(np.unique(right))


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_targets", [2, 4])
def test_vector_leaf_cuda_hybrid_level_matches_classic(num_targets):
    # the hybrid level-batched path is an execution-strategy change: a level's
    # splits are searched in batched launches instead of one at a time, and the
    # tree it grows is the one the classic per-split loop grows
    rng = np.random.default_rng(20260901)
    X = rng.normal(size=(N_ROWS, N_FEATURES))
    columns = []
    for target in range(num_targets):
        columns.append(
            (2.0 + target) * X[:, target % N_FEATURES]
            - 1.5 * X[:, (target + 3) % N_FEATURES]
            + np.where(X[:, (target + 5) % N_FEATURES] > 0.25, 1.0, -1.0)
            + 0.05 * rng.normal(size=N_ROWS)
        )
    labels = np.column_stack(columns)

    hybrid = _train_vector(X, labels, extra_params=dict(_HYBRID_PARAMS, cuda_plan="auto"))
    classic = _train_vector(X, labels, extra_params=dict(_HYBRID_PARAMS, cuda_plan="auto,hybrid:off"))

    _assert_same_leaf_partition(hybrid.predict(X, pred_leaf=True), classic.predict(X, pred_leaf=True))
    np.testing.assert_allclose(hybrid.predict(X), classic.predict(X), rtol=1e-9, atol=1e-9)

    hybrid_trees = hybrid.dump_model()["tree_info"]
    classic_trees = classic.dump_model()["tree_info"]
    assert [t["num_leaves"] for t in hybrid_trees] == [t["num_leaves"] for t in classic_trees]
