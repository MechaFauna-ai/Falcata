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


@_REQUIRES_CUDA
def test_vector_leaf_cuda_hybrid_duplicated_target_matches_scalar():
    # Ground truth for the hybrid level prefix, not just self-consistency with
    # the classic loop: both vector paths construct planes 1..T-1 gradient-only,
    # so a hybrid-vs-classic comparison alone cannot catch a plane whose
    # hessians are wrong. Duplicating the target makes the scalar model the
    # answer -- doubling every candidate gain leaves the greedy structure
    # unchanged -- and the scalar path never takes a gradient-only construct.
    X, y0, _ = _make_data()
    labels = np.column_stack((y0, y0))
    vector = _train_vector(X, labels, extra_params=dict(_HYBRID_PARAMS, cuda_plan="auto"))
    scalar = _train_scalar(X, y0, extra_params=dict(_HYBRID_PARAMS, cuda_plan="auto"))

    _assert_same_leaf_partition(
        vector.predict(X, pred_leaf=True), scalar.predict(X, pred_leaf=True).reshape(N_ROWS, -1)
    )
    # The structure assertion above is the exact one. Leaf VALUES are compared
    # loosely because the reference is itself nondeterministic: the scalar
    # hybrid path's batched level construct reduces with fp64 atomics, so two
    # runs of the same scalar config differ by ~2e-8 absolute. The classic
    # duplicated-target test above can assert 1e-9 only because both of its
    # paths take the same deterministic per-leaf construct.
    scalar_prediction = scalar.predict(X)
    vector_prediction = vector.predict(X)
    for target in range(2):
        np.testing.assert_allclose(vector_prediction[:, target], scalar_prediction, rtol=1e-5, atol=1e-6)


# ---- quantized vector-leaf training -----------------------------------------
#
# One discretized gradient plane per target, each at its OWN gradient scale
# (targets differ in gradient magnitude, and a shared scale quantizes the small
# ones away), all sharing plane 0's quantized hessian stream. Every histogram
# scan is exact packed-integer arithmetic, which makes the quantized vector
# path run-to-run deterministic where the fp64 one is not.

_QUANT_PARAMS = {"quant_mode": "fixedpoint", "num_grad_quant_bins": 16}
# 64 bins over these 4000 rows drives the root and the big leaves past the
# 16-bit packed-histogram range (num_data * bins > 65535) while small leaves
# stay 16-bit: the only setting that exercises the 32-bit accumulator arm and
# the mixed-width parent/child subtract.
_QUANT_PARAMS_32BIT = {"quant_mode": "fixedpoint", "num_grad_quant_bins": 64}


def _multi_target_labels(num_targets, seed=20260902):
    """Targets that SHARE their driving features but live on deliberately
    different magnitude scales: target t is 3^t times target 0.

    Both properties are load-bearing, in opposite directions:

    * the 3^t spread is what makes the per-target gradient scale observable.
      Under one scale set by the largest target, target 0's gradients sit
      3^(T-1) below the quantum, round to zero, and its leaf values stay ~0 --
      so every target fitting equally well is the assertion that the scales are
      per-target.
    * the SHARED drivers are what make a fit possible at all. Vector-leaf gain
      is summed over targets, so the split search is dominated by the
      largest-magnitude target; with disjoint driving features the small
      targets are simply not fit (measured: nmse 0.61/0.82 on targets 0/1),
      and identically so at fp64 -- an objective property, not a quantization
      one. Data whose targets do not share structure is the wrong shape for a
      shared-structure tree, and a quantization test must not confuse the two.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_ROWS, N_FEATURES))
    base = 2.0 * X[:, 0] - 1.5 * X[:, 3] + np.where(X[:, 5] > 0.25, 1.0, -1.0)
    columns = []
    for target in range(num_targets):
        signal = base + 0.4 * X[:, (target + 1) % N_FEATURES] + 0.05 * rng.normal(size=N_ROWS)
        columns.append(signal * (3.0**target))
    return X, np.column_stack(columns)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("num_targets", "quant_params"), [(2, _QUANT_PARAMS), (4, _QUANT_PARAMS), (4, _QUANT_PARAMS_32BIT)]
)
def test_vector_leaf_cuda_quantized_trains_and_roundtrips(num_targets, quant_params):
    X, labels = _multi_target_labels(num_targets)
    train_set = lgb.Dataset(X, label=labels)
    params = _base_params()
    params.update(
        {
            "objective": "multi_regression",
            "metric": "multi_rmse",
            "num_class": num_targets,
            "tree_mode": "vector_leaf",
        }
    )
    params.update(quant_params)
    evals = {}
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=30,
        valid_sets=[train_set],
        valid_names=["train"],
        callbacks=[lgb.record_evaluation(evals)],
    )

    curve = evals["train"]["multi_rmse"]
    assert len(curve) == 30
    assert curve[-1] < 0.5 * curve[0]

    prediction = booster.predict(X)
    assert prediction.shape == (N_ROWS, num_targets)
    assert np.all(np.isfinite(prediction))
    # EVERY target must fit, including target 0 at 3^-(T-1) of target T-1's
    # magnitude: that is the per-target gradient scale under test. One shared
    # scale rounds target 0's gradients to zero and leaves it at nmse ~1.
    nmse = np.mean((prediction - labels) ** 2, axis=0) / np.var(labels, axis=0)
    assert np.all(nmse < 0.15), nmse

    model_text = booster.model_to_string()
    assert model_text.count(f"leaf_value_dim={num_targets}") == booster.num_trees()
    reloaded = lgb.Booster(model_str=model_text)
    np.testing.assert_array_equal(reloaded.predict(X), prediction)

    blob = booster.model_to_binary()
    from_binary = lgb.Booster(model_str=model_text).model_from_binary(blob)
    np.testing.assert_array_equal(from_binary.predict(X), prediction)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "vector_leaf_quant.txt")
        booster.save_model(path)
        np.testing.assert_array_equal(lgb.Booster(model_file=path).predict(X), prediction)


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_targets", [2, 4])
def test_vector_leaf_cuda_quantized_quality_tracks_fp64(num_targets):
    # Quantization trades histogram resolution for speed; it must stay close to
    # the fp64 vector model, not collapse. The per-target scale is what makes
    # this hold across targets whose gradients differ by orders of magnitude:
    # under one shared scale the small-magnitude targets would barely move.
    X, labels = _multi_target_labels(num_targets)
    quantized = _train_vector(X, labels, num_boost_round=30, extra_params=_QUANT_PARAMS)
    reference = _train_vector(X, labels, num_boost_round=30)

    quantized_mse = np.mean((quantized.predict(X) - labels) ** 2, axis=0)
    reference_mse = np.mean((reference.predict(X) - labels) ** 2, axis=0)
    # measured 1.00-1.02x per target at T=2 and T=4, on 16 and 64 quant bins
    assert np.all(quantized_mse < 1.25 * reference_mse + 1e-12), quantized_mse / reference_mse


@_REQUIRES_CUDA
def test_vector_leaf_cuda_quantized_duplicated_target_matches_scalar():
    # Ground truth rather than self-consistency: duplicating the target doubles
    # every candidate gain, so the greedy structure is scalar quantized
    # training's, and both target outputs must equal the scalar model's. This is
    # what pins the per-plane packed leaf totals -- a plane whose integer parent
    # sum drifted would split elsewhere.
    X, y0, _ = _make_data()
    labels = np.column_stack((y0, y0))
    vector = _train_vector(X, labels, extra_params=_QUANT_PARAMS)
    scalar = _train_scalar(X, y0, extra_params=_QUANT_PARAMS)

    np.testing.assert_array_equal(vector.predict(X, pred_leaf=True), scalar.predict(X, pred_leaf=True))
    scalar_prediction = scalar.predict(X)
    vector_prediction = vector.predict(X)
    for target in range(2):
        np.testing.assert_allclose(vector_prediction[:, target], scalar_prediction, rtol=1e-9, atol=1e-9)


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant_params", [_QUANT_PARAMS, _QUANT_PARAMS_32BIT])
def test_vector_leaf_cuda_quantized_hybrid_matches_classic(quant_params):
    # Integer histogram accumulation is order-invariant, so unlike the fp64
    # vector path the quantized one owes bit-identical models across execution
    # strategies -- the batched level prefix and the per-split loop must agree
    # exactly, including leaf values.
    X, labels = _multi_target_labels(3)
    hybrid = _train_vector(X, labels, extra_params=dict(_HYBRID_PARAMS, **quant_params, cuda_plan="auto"))
    classic = _train_vector(X, labels, extra_params=dict(_HYBRID_PARAMS, **quant_params, cuda_plan="auto,hybrid:off"))
    _assert_same_leaf_partition(hybrid.predict(X, pred_leaf=True), classic.predict(X, pred_leaf=True))
    np.testing.assert_allclose(hybrid.predict(X), classic.predict(X), rtol=1e-12, atol=1e-12)


@_REQUIRES_CUDA
def test_vector_leaf_cuda_quantized_bagging_does_not_collapse():
    # Bagged quantized training is where the discretized split search is most
    # exposed: bagging redraws the rounding noise every iteration and the gain
    # argmax harvests children whose noisy |G| spiked while noisy H landed near
    # zero, which in scalar multiclass training compounds into a quality
    # collapse. Vector mode sums T such gains, so the find kernels carry the
    # same one-hessian-quantum l2 ridge under bagging. Deep trees over many
    # rounds are what expose the pathology; this pins bagged quantized quality
    # against both the unbagged quantized and the bagged fp64 model.
    X, labels = _multi_target_labels(4)
    bagged = _train_vector(X, labels, num_boost_round=60, extra_params=dict(_QUANT_PARAMS, **_BAGGING_PARAMS))
    unbagged = _train_vector(X, labels, num_boost_round=60, extra_params=_QUANT_PARAMS)
    bagged_fp64 = _train_vector(X, labels, num_boost_round=60, extra_params=_BAGGING_PARAMS)

    variance = np.var(labels, axis=0)
    bagged_mse = np.mean((bagged.predict(X) - labels) ** 2, axis=0)
    unbagged_mse = np.mean((unbagged.predict(X) - labels) ** 2, axis=0)
    bagged_fp64_mse = np.mean((bagged_fp64.predict(X) - labels) ** 2, axis=0)
    # measured 1.02-1.07x of unbagged quantized and 0.98-1.04x of bagged fp64
    assert np.all(bagged_mse < 1.5 * unbagged_mse + 1e-12), bagged_mse / unbagged_mse
    assert np.all(bagged_mse < 1.5 * bagged_fp64_mse + 1e-12), bagged_mse / bagged_fp64_mse
    assert np.all(bagged_mse < 0.15 * variance), bagged_mse / variance


@_REQUIRES_CUDA
def test_vector_leaf_cuda_quantized_rejects_categorical():
    # the quantized vector search covers numerical tasks only; the fence is
    # explicit rather than a silently wrong categorical split
    X, y0, y1, cat_indices = _make_cat_data()
    params = _base_params()
    params.update(
        {
            "objective": "multi_regression",
            "metric": "multi_rmse",
            "num_class": 2,
            "tree_mode": "vector_leaf",
        }
    )
    params.update(_QUANT_PARAMS)
    with pytest.raises(lgb.basic.FalcataError, match="categorical"):
        lgb.train(
            params,
            lgb.Dataset(X, label=np.column_stack((y0, y1)), categorical_feature=cat_indices),
            num_boost_round=1,
        )
