# coding: utf-8
"""CUDA vector-leaf multi-target training: one shared-structure tree per
iteration whose leaves carry one output per target, split gain summed over
targets.

Everything here needs a CUDA build (TASK=cuda). Correctness is checked
through invariants rather than a golden model, because non-quantized CUDA
training is run-to-run nondeterministic (fp64 atomic histograms). The scalar
references and the vector arms are compared leaf for leaf, so every
non-quantized training here asks for the deterministic histograms (cuda_plan
key nonquant_deterministic, opt-in since 1.2; _with_det):

* duplicated targets: y2 == y1 doubles every candidate's gain, so the greedy
  structure must match scalar training on y1, and both targets' outputs must
  match the scalar model's;
* negated targets: y2 == -y1 leaves every gain unchanged (gains are even in
  the gradient), so predictions must be antisymmetric across targets;
* the loss on each target must drop well below the label variance; and
* the trained model must round-trip through text and FALB serialization.
"""

import os
import subprocess
import sys
import tempfile
import textwrap

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
        "cuda_plan": "auto,nonquant_deterministic:on",
    }


def _with_det(params):
    """the deterministic non-quantized histograms (opt-in since 1.2) appended to a plan the test set itself"""
    plan = params.get("cuda_plan", "auto")
    if params.get("quant_mode") == "none" and "nonquant_deterministic" not in plan:
        params["cuda_plan"] = plan + ",nonquant_deterministic:on"
    return params


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
    return lgb.train(_with_det(params), lgb.Dataset(X, label=labels), num_boost_round=num_boost_round)


def _train_scalar(X, y, num_boost_round=20, extra_params=None):
    params = _base_params()
    params.update({"objective": "regression", "boost_from_average": False})
    if extra_params:
        params.update(extra_params)
    return lgb.train(_with_det(params), lgb.Dataset(X, label=y), num_boost_round=num_boost_round)


@_REQUIRES_CUDA
def test_vector_leaf_cuda_fp32_request_is_rejected():
    # Multi-target vector leaves keep fp64 histogram planes: the configuration
    # refuses cuda_precision=fp32, so the CUDA histogram planner never sees a
    # multi-plane training with the fp32 layout (DecideNonquantHistograms).
    X, y0, y1 = _make_data()
    labels = np.column_stack((y0, y1))
    params = {
        **_base_params(),
        "objective": "multi_regression",
        "num_class": 2,
        "tree_mode": "vector_leaf",
        "cuda_precision": "fp32",
    }
    with pytest.raises(lgb.basic.FalcataError, match="requires cuda_precision=fp64"):
        lgb.train(params, lgb.Dataset(X, label=labels), num_boost_round=1)


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

    scalar_prediction = scalar.predict(X, device="cpu")
    vector_prediction = vector.predict(X, device="cpu")
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

    scalar_prediction = scalar.predict(X, device="cpu")
    vector_prediction = vector.predict(X, device="cpu")
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

    scalar_prediction = scalar.predict(X, device="cpu")
    vector_prediction = vector.predict(X, device="cpu")
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


# budget-limited config (2^max_depth > num_leaves + 1): the regime plain level
# batching is only approximate in, and therefore the regime vector-leaf training
# takes the SELECTIVE (grow-then-prune) level prefix in
_SELECTIVE_PARAMS = {"num_leaves": 31, "max_depth": 10}


def _assert_vector_selective_engaged(extra_params=None):
    """A vacuous equivalence test is worse than none: the two arms below are the
    same code path unless vector mode really reaches selective growth, so probe
    for the growth phase's own stderr line in a subprocess (FALCATA_DEBUG is read
    once per process)."""
    probe = textwrap.dedent(f"""
        import numpy as np, falcata as lgb
        rng = np.random.default_rng(0)
        X = rng.normal(size=({N_ROWS}, {N_FEATURES}))
        y = X[:, 0] + 0.5 * X[:, 3]
        labels = np.column_stack((y, -y))
        p = {{"device_type": "cuda", "quant_mode": "none", "objective": "multi_regression",
              "num_class": 2, "tree_mode": "vector_leaf", "min_data_in_leaf": 20,
              "learning_rate": 0.1, "seed": 7, "verbosity": -1,
              "num_leaves": {_SELECTIVE_PARAMS["num_leaves"]},
              "max_depth": {_SELECTIVE_PARAMS["max_depth"]}}}
        p.update({extra_params or {}!r})
        lgb.train(p, lgb.Dataset(X, label=labels), num_boost_round=5)
    """)
    env = dict(os.environ, FALCATA_DEBUG="debug")
    out = subprocess.run([sys.executable, "-c", probe], check=False, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-4000:]
    assert "[selective]" in out.stderr, (
        "vector-leaf training never reached selective growth; the equivalence "
        "assertions below would compare the classic loop against itself"
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_targets", [2, 4])
def test_vector_leaf_cuda_selective_matches_classic(num_targets):
    _assert_vector_selective_engaged()
    # the selective prefix grows whole levels and prunes the splits the classic
    # greedy selection would never have made, so the tree it ends with is the
    # one the classic per-split loop grows -- with classic leaf numbering, since
    # the final structure is replayed host-side in greedy order
    rng = np.random.default_rng(20260902)
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

    selective = _train_vector(X, labels, extra_params=dict(_SELECTIVE_PARAMS, cuda_plan="auto"))
    classic = _train_vector(X, labels, extra_params=dict(_SELECTIVE_PARAMS, cuda_plan="auto,hybrid:off"))

    _assert_same_leaf_partition(selective.predict(X, pred_leaf=True), classic.predict(X, pred_leaf=True))
    np.testing.assert_allclose(selective.predict(X), classic.predict(X), rtol=1e-9, atol=1e-9)

    selective_trees = selective.dump_model()["tree_info"]
    classic_trees = classic.dump_model()["tree_info"]
    assert [t["num_leaves"] for t in selective_trees] == [t["num_leaves"] for t in classic_trees]


@_REQUIRES_CUDA
def test_vector_leaf_cuda_selective_duplicated_target_matches_scalar():
    # Ground truth for the selective prefix: the per-target leaf outputs of the
    # rebuilt (pruned) tree come from a host snapshot of each split's vector
    # payload, which a vector-vs-vector comparison alone cannot validate.
    # Duplicating the target makes the scalar model the answer.
    X, y0, _ = _make_data()
    labels = np.column_stack((y0, y0))
    vector = _train_vector(X, labels, extra_params=dict(_SELECTIVE_PARAMS, cuda_plan="auto"))
    scalar = _train_scalar(X, y0, extra_params=dict(_SELECTIVE_PARAMS, cuda_plan="auto"))

    _assert_same_leaf_partition(
        vector.predict(X, pred_leaf=True), scalar.predict(X, pred_leaf=True).reshape(N_ROWS, -1)
    )
    scalar_prediction = scalar.predict(X, device="cpu")
    vector_prediction = vector.predict(X, device="cpu")
    for target in range(2):
        np.testing.assert_allclose(vector_prediction[:, target], scalar_prediction, rtol=1e-5, atol=1e-6)


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
    scalar_prediction = scalar.predict(X, device="cpu")
    vector_prediction = vector.predict(X, device="cpu")
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
    scalar_prediction = scalar.predict(X, device="cpu")
    vector_prediction = vector.predict(X, device="cpu")
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
@pytest.mark.parametrize("quant_params", [_QUANT_PARAMS, _QUANT_PARAMS_32BIT])
def test_vector_leaf_cuda_quantized_selective_matches_classic(quant_params):
    # The budget-limited regime (num_leaves << 2^max_depth) routes vector mode
    # to the SELECTIVE grow-then-prune prefix, and quantization is orthogonal to
    # it: the prefix reuses the same batched level histograms and vector find as
    # the exact-fit one, so a quantized selective tree is the quantized classic
    # tree. Integer accumulation is order-invariant, so this is bit-identity, not
    # a tolerance -- including the leaf values the selective prefix replays from
    # its host payload snapshot.
    _assert_vector_selective_engaged(quant_params)
    X, labels = _multi_target_labels(3)
    selective = _train_vector(X, labels, extra_params=dict(_SELECTIVE_PARAMS, **quant_params, cuda_plan="auto"))
    classic = _train_vector(
        X, labels, extra_params=dict(_SELECTIVE_PARAMS, **quant_params, cuda_plan="auto,hybrid:off")
    )
    _assert_same_leaf_partition(selective.predict(X, pred_leaf=True), classic.predict(X, pred_leaf=True))
    np.testing.assert_allclose(selective.predict(X), classic.predict(X), rtol=1e-12, atol=1e-12)
    assert [t["num_leaves"] for t in selective.dump_model()["tree_info"]] == [
        t["num_leaves"] for t in classic.dump_model()["tree_info"]
    ]


@_REQUIRES_CUDA
def test_vector_leaf_cuda_quantized_selective_duplicated_target_matches_scalar():
    # Ground truth for quantized selective growth: the rebuilt (pruned) tree's
    # per-target leaf outputs come from a host snapshot of each split's vector
    # payload, whose gradient half the quantized plane fan-out re-packs against
    # plane 0's shared integer hessians. Duplicating the target makes scalar
    # quantized training the answer, exactly and to the bit.
    X, y0, _ = _make_data()
    labels = np.column_stack((y0, y0))
    vector = _train_vector(X, labels, extra_params=dict(_SELECTIVE_PARAMS, **_QUANT_PARAMS, cuda_plan="auto"))
    scalar = _train_scalar(X, y0, extra_params=dict(_SELECTIVE_PARAMS, **_QUANT_PARAMS, cuda_plan="auto"))

    np.testing.assert_array_equal(
        vector.predict(X, pred_leaf=True), scalar.predict(X, pred_leaf=True).reshape(N_ROWS, -1)
    )
    scalar_prediction = scalar.predict(X, device="cpu")
    vector_prediction = vector.predict(X, device="cpu")
    for target in range(2):
        np.testing.assert_allclose(vector_prediction[:, target], scalar_prediction, rtol=1e-9, atol=1e-9)


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


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_targets", [2, 4, 5, 8, 16])
@pytest.mark.parametrize("mode", ["fp64", "quant_16bit", "quant_32bit"])
def test_vector_leaf_cuda_targets_sweep_hybrid_matches_classic(num_targets, mode):
    """Exercise all vector split-finder kernels across T in {2, 4, 5, 8, 16}
    (non-power-of-2 and max capacity 16) in both fp64 and quantized (16-bit &
    32-bit accumulator) paths, verifying convergence and exact hybrid/classic
    agreement."""
    rng = np.random.default_rng(20261128 + num_targets)
    n_rows = 4000
    n_feat = 16
    X = rng.normal(size=(n_rows, n_feat))
    base = 2.0 * X[:, 0] - 1.5 * X[:, 3] + np.where(X[:, 5] > 0.2, 1.0, -1.0)
    labels = np.column_stack(
        [
            (base + 0.5 * X[:, (t + 1) % n_feat] + 0.05 * rng.normal(size=n_rows)) * (1.0 + 0.25 * t)
            for t in range(num_targets)
        ]
    )

    extra = dict(_HYBRID_PARAMS, lambda_l2=1.0)
    if mode == "quant_16bit":
        extra.update(_QUANT_PARAMS)
    elif mode == "quant_32bit":
        extra.update(_QUANT_PARAMS_32BIT)

    hybrid = _train_vector(X, labels, num_boost_round=12, extra_params=dict(extra, cuda_plan="auto"))
    classic = _train_vector(X, labels, num_boost_round=12, extra_params=dict(extra, cuda_plan="auto,hybrid:off"))

    pred_h = hybrid.predict(X)
    pred_c = classic.predict(X)
    assert pred_h.shape == (n_rows, num_targets)
    assert np.all(np.isfinite(pred_h))

    nmse = np.mean((pred_h - labels) ** 2, axis=0) / np.var(labels, axis=0)
    assert np.all(nmse < 0.35), nmse

    _assert_same_leaf_partition(hybrid.predict(X, pred_leaf=True), classic.predict(X, pred_leaf=True))
    tol = 1e-12 if mode != "fp64" else 1e-9
    np.testing.assert_allclose(pred_h, pred_c, rtol=tol, atol=tol)


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_targets", [5, 16])
@pytest.mark.parametrize("mode", ["fp64", "quant_16bit", "quant_32bit"])
def test_vector_leaf_cuda_nan_and_zero_missing_match_classic(num_targets, mode):
    """Exercise na_missing_head (NaN with mfb_offset==1) and skip_default_bin
    (zero_as_missing) across both fp64 and quantized vector split finders."""
    rng = np.random.default_rng(20261129 + num_targets)
    n_rows = 4000
    n_feat = 12
    X = rng.normal(size=(n_rows, n_feat))
    # Make columns 0..5 heavy-zero (mfb=0 -> mfb_offset=1) with NaNs (na_missing_head)
    zero_mask = rng.random((n_rows, 6)) < 0.6
    X[:, :6][zero_mask] = 0.0
    nan_mask = rng.random((n_rows, n_feat)) < 0.1
    X[nan_mask] = np.nan

    clean0 = np.nan_to_num(X[:, 0], nan=-2.0)
    clean6 = np.nan_to_num(X[:, 6], nan=1.5)
    base = 1.8 * clean0 - 1.2 * clean6 + np.where(np.isnan(X[:, 1]), 1.2, -0.4)
    labels = np.column_stack(
        [
            (base + 0.3 * np.nan_to_num(X[:, (t + 2) % n_feat]) + 0.05 * rng.normal(size=n_rows)) * (1.0 + 0.2 * t)
            for t in range(num_targets)
        ]
    )

    for zero_as_missing in (False, True):
        extra = dict(_HYBRID_PARAMS, zero_as_missing=zero_as_missing, lambda_l2=0.5)
        if mode == "quant_16bit":
            extra.update(_QUANT_PARAMS)
        elif mode == "quant_32bit":
            extra.update(_QUANT_PARAMS_32BIT)

        hybrid = _train_vector(X, labels, num_boost_round=10, extra_params=dict(extra, cuda_plan="auto"))
        classic = _train_vector(X, labels, num_boost_round=10, extra_params=dict(extra, cuda_plan="auto,hybrid:off"))

        _assert_same_leaf_partition(hybrid.predict(X, pred_leaf=True), classic.predict(X, pred_leaf=True))
        tol = 1e-12 if mode != "fp64" else 1e-9
        np.testing.assert_allclose(hybrid.predict(X), classic.predict(X), rtol=tol, atol=tol)


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_targets", [5, 16])
@pytest.mark.parametrize("lambda_l2", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("mode", ["fp64", "quant_16bit"])
def test_vector_leaf_cuda_ties_and_lambda_l2_match_scalar(num_targets, lambda_l2, mode):
    """Verify exact tie-breaking on duplicated/constant features and L2
    regularization across duplicated-target vector vs scalar training."""
    rng = np.random.default_rng(20261130 + num_targets)
    n_rows = 3000
    f0 = rng.normal(size=n_rows)
    f1 = rng.normal(size=n_rows)
    # Exact duplicate columns (plateau ties) + constant column (zero-gain bins)
    X = np.column_stack((f0, f0, f1, f1, np.zeros(n_rows), rng.normal(size=(n_rows, 4))))
    y0 = 2.0 * f0 - 1.5 * f1 + np.where(f0 > 0.0, 0.8, -0.8) + 0.05 * rng.normal(size=n_rows)
    labels = np.column_stack([y0] * num_targets)

    extra = dict(_HYBRID_PARAMS, lambda_l2=lambda_l2, cuda_plan="auto,hybrid:off")
    if mode == "quant_16bit":
        extra.update(_QUANT_PARAMS)

    vector = _train_vector(X, labels, num_boost_round=12, extra_params=extra)
    scalar = _train_scalar(X, y0, num_boost_round=12, extra_params=extra)

    np.testing.assert_array_equal(vector.predict(X, pred_leaf=True), scalar.predict(X, pred_leaf=True))
    vec_pred = vector.predict(X, device="cpu")
    scal_pred = scalar.predict(X, device="cpu")
    for t in range(num_targets):
        np.testing.assert_allclose(vec_pred[:, t], scal_pred, rtol=1e-9, atol=1e-9)


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_targets", [5, 16])
def test_vector_leaf_cuda_categorical_many_targets_matches_classic_and_scalar(num_targets):
    """Exercise categorical one-hot and many-vs-many vector split search at
    T=5 and T=16 (parallel prefix + parallel winner replay)."""
    X, y0, _, cat_indices = _make_cat_data(seed=20261131 + num_targets)
    labels = np.column_stack([y0] * num_targets)

    vec_params = dict(
        _base_params(),
        objective="multi_regression",
        metric="multi_rmse",
        num_class=num_targets,
        tree_mode="vector_leaf",
        **_HYBRID_PARAMS,
        lambda_l2=1.0,
    )
    hybrid = lgb.train(
        dict(vec_params, cuda_plan="auto,nonquant_deterministic:on"),
        lgb.Dataset(X, label=labels, categorical_feature=cat_indices),
        num_boost_round=12,
    )
    classic = lgb.train(
        dict(vec_params, cuda_plan="auto,hybrid:off,nonquant_deterministic:on"),
        lgb.Dataset(X, label=labels, categorical_feature=cat_indices),
        num_boost_round=12,
    )
    scal_params = dict(
        _base_params(),
        objective="regression",
        boost_from_average=False,
        **_HYBRID_PARAMS,
        lambda_l2=1.0,
        cuda_plan="auto,hybrid:off,nonquant_deterministic:on",
    )
    scalar = lgb.train(
        scal_params,
        lgb.Dataset(X, label=y0, categorical_feature=cat_indices),
        num_boost_round=12,
    )

    _assert_same_leaf_partition(hybrid.predict(X, pred_leaf=True), classic.predict(X, pred_leaf=True))
    np.testing.assert_allclose(hybrid.predict(X), classic.predict(X), rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(classic.predict(X, pred_leaf=True), scalar.predict(X, pred_leaf=True))
    model_text = classic.model_to_string()
    num_cat_counts = [int(line.split("=", 1)[1]) for line in model_text.splitlines() if line.startswith("num_cat=")]
    assert sum(num_cat_counts) > 0


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", ["fp64", "quant_16bit"])
def test_vector_leaf_cuda_feature_fraction_compact_tasks(mode):
    """Exercise used_task_indices compact-task launch path in batched level
    vector split finders."""
    X, labels = _multi_target_labels(8, seed=20261201)
    extra = dict(_HYBRID_PARAMS, feature_fraction=0.5, feature_fraction_seed=19)
    if mode == "quant_16bit":
        extra.update(_QUANT_PARAMS)

    hybrid = _train_vector(X, labels, num_boost_round=12, extra_params=dict(extra, cuda_plan="auto"))
    classic = _train_vector(X, labels, num_boost_round=12, extra_params=dict(extra, cuda_plan="auto,hybrid:off"))

    _assert_same_leaf_partition(hybrid.predict(X, pred_leaf=True), classic.predict(X, pred_leaf=True))
    tol = 1e-12 if mode != "fp64" else 1e-9
    np.testing.assert_allclose(hybrid.predict(X), classic.predict(X), rtol=tol, atol=tol)


@_REQUIRES_CUDA
@pytest.mark.parametrize("quant_params", [_QUANT_PARAMS, _QUANT_PARAMS_32BIT])
def test_vector_leaf_cuda_quantized_257_bin_na_missing_head(quant_params):
    """Regression test for a 257-bin feature (mfb_offset==1 + 255 stored normal
    bins + 1 NaN bin, so num_bin==257 while num_bin_hist==256 stays on the
    256-thread shared-memory quantized kernel). Verifies that the half-warp
    replay for targets t>=1 bounds sum_all to the exact 256 scanned lanes and
    matches plane 0's direct scan (scan_gh0) and scalar quantized training."""
    rng = np.random.default_rng(20261202)
    # 0.0 is the most-frequent bin (bin 0, mfb_offset=1), 1..255 are 255
    # distinct positive bins, and NaN is bin 256 -> total 257 bins!
    # Give the final bin (255) large non-zero target mass so including it in
    # replay sum_all would visibly corrupt target t>=1 leaf outputs.
    x_zero = np.zeros(1200, dtype=np.float64)
    x_pos = np.repeat(np.arange(1, 256, dtype=np.float64), 6)
    x_nan = np.full(300, np.nan, dtype=np.float64)
    x_feat = np.concatenate([x_zero, x_pos, x_nan])
    perm = rng.permutation(len(x_feat))
    x_feat = x_feat[perm]
    X = x_feat.reshape(-1, 1)

    # Strong gradient in bin 0 vs high bins, plus very large mass on bin 255
    # (the stored bin index 255 outside the 256-thread na_missing_head scan).
    # Round y0 to float32 first and use positive power-of-2 scales so every
    # target's float32 labels, grad_scale[t], and fixedpoint subtractive dither
    # produce identical integer quantized gradients across all 4 targets.
    y0 = np.where(np.isnan(x_feat), -1.5, np.where(x_feat == 0.0, 2.0, np.where(x_feat == 255.0, 25.0, -0.5)))
    y0 = (y0 + 0.02 * rng.normal(size=len(x_feat))).astype(np.float32).astype(np.float64)
    scales = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    labels = np.column_stack([s * y0 for s in scales])

    common = {
        "max_bin": 257,
        "min_data_in_bin": 1,
        "bin_construct_sample_cnt": 100000,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 20,
        "learning_rate": 1.0,
        "lambda_l2": 0.0,
        **quant_params,
    }
    vec_hybrid = _train_vector(X, labels, num_boost_round=1, extra_params=dict(common, cuda_plan="auto"))
    vec_classic = _train_vector(X, labels, num_boost_round=1, extra_params=dict(common, cuda_plan="auto,hybrid:off"))
    # robust_scale only engages for num_class<=1; disable it on scalar so the
    # 25.0 mass in bin 255 uses the same global-max fixedpoint scale as vector.
    scalar = _train_scalar(
        X, y0, num_boost_round=1, extra_params=dict(common, cuda_plan="auto,hybrid:off,robust_scale:off")
    )

    # Confirm the feature actually produced 257 bins (so num_bin == 257 and
    # num_bin_hist == 256 with na_missing_head active).
    ds = lgb.Dataset(X, label=y0, params={"max_bin": 257, "min_data_in_bin": 1, "bin_construct_sample_cnt": 100000})
    ds.construct()
    assert ds.feature_num_bin(0) == 257

    # Both classic and hybrid must match scalar on target 0 AND scale exactly
    # across targets t=1..3 (direct scan_gh0 vs half-warp replay).
    scal_pred = scalar.predict(X, device="cpu")
    for vec in (vec_hybrid, vec_classic):
        vec_pred = vec.predict(X, device="cpu")
        np.testing.assert_allclose(vec_pred[:, 0], scal_pred, rtol=1e-9, atol=1e-9)
        for t, s in enumerate(scales):
            np.testing.assert_allclose(vec_pred[:, t], s * vec_pred[:, 0], rtol=1e-9, atol=1e-9)


@_REQUIRES_CUDA
@pytest.mark.parametrize("mode", ["fp64", "quant_16bit", "quant_32bit"])
def test_vector_leaf_cuda_distinct_targets_exact_leaf_outputs(mode):
    """Independent ground-truth check for distinct (non-identical) targets
    across T=8: for a single tree with learning_rate=1.0 and lambda_l2=0.0,
    each leaf's output vector must match the empirical mean of each target's
    labels over the rows landing in that leaf (exact for fp64; close for
    quantized), and reversing the target column order must reproduce the
    corresponding per-target predictions without cross-target mix-up."""
    rng = np.random.default_rng(20261203)
    n_rows = 4000
    n_feat = 12
    num_targets = 8
    X = rng.normal(size=(n_rows, n_feat))
    # Distinct targets driven by different features, signs, and scales
    cols = []
    for t in range(num_targets):
        sign = 1.0 if (t % 2 == 0) else -1.0
        scale = 0.5 + 1.5 * t
        sig = sign * (
            1.8 * X[:, t % n_feat] - 1.1 * X[:, (t + 3) % n_feat] + np.where(X[:, (t + 5) % n_feat] > 0.0, 1.0, -1.0)
        ) + 0.05 * rng.normal(size=n_rows)
        cols.append(scale * sig)
    labels = np.column_stack(cols)

    extra = {
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 20,
        "learning_rate": 1.0,
        "lambda_l2": 0.0,
        "cuda_plan": "auto",
    }
    if mode == "quant_16bit":
        extra.update(_QUANT_PARAMS)
    elif mode == "quant_32bit":
        extra.update(_QUANT_PARAMS_32BIT)

    booster = _train_vector(X, labels, num_boost_round=1, extra_params=extra)
    leaf_ids = booster.predict(X, pred_leaf=True).reshape(-1)
    preds = booster.predict(X, device="cpu")
    assert preds.shape == (n_rows, num_targets)

    # Independent CPU ground truth: compare each leaf's predicted vector
    # against the exact mean of labels[mask, t] (stored as float32 in Dataset)
    # for every leaf and every target.
    labels_f32 = labels.astype(np.float32).astype(np.float64)
    for leaf in np.unique(leaf_ids):
        mask = leaf_ids == leaf
        expected_mean = np.mean(labels_f32[mask], axis=0)
        leaf_pred = preds[mask][0]
        if mode == "fp64":
            np.testing.assert_allclose(leaf_pred, expected_mean, rtol=1e-10, atol=1e-10)
        else:
            # Quantized leaf outputs match empirical means within quantization resolution
            assert np.all(np.abs(leaf_pred - expected_mean) < 0.08 * np.std(labels, axis=0) + 1e-6), (
                leaf_pred,
                expected_mean,
            )

    # Verify target-order permutation consistency: swapping target 0 and target 5
    # swaps their predicted columns without mixing up any target slot.
    perm_idx = [5, 1, 2, 3, 4, 0, 6, 7]
    swapped = _train_vector(X, labels[:, perm_idx], num_boost_round=1, extra_params=extra)
    swapped_preds = swapped.predict(X, device="cpu")
    _assert_same_leaf_partition(booster.predict(X, pred_leaf=True), swapped.predict(X, pred_leaf=True))
    tol = 1e-12 if mode != "fp64" else 1e-10
    np.testing.assert_allclose(swapped_preds, preds[:, perm_idx], rtol=tol, atol=tol)
