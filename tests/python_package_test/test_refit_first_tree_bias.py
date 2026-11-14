"""Refit rebuilds the first iteration's trees from the average score, as training built them.

Training adds the boost_from_average score to the first iteration's trees (Tree::AddBias, which sets their shrinkage
to 1). Refit started from score 0 and gave those trees a full, unshrunk Newton step from there. For log-link
objectives that step is far off (a leaf of ~mean(label) on the log scale), and the next iteration's exp() gradients
overflowed: refit with tweedie_variance_power=1.0 or poisson returned NaN, and gamma / tweedie(1.5) refits were
several times worse than the model they refit.
"""

import numpy as np
import pytest

import falcata as flc

OBJECTIVES = [
    {"objective": "regression"},
    {"objective": "poisson"},
    {"objective": "gamma"},
    {"objective": "tweedie", "tweedie_variance_power": 1.0},
    {"objective": "tweedie", "tweedie_variance_power": 1.5},
    {"objective": "binary"},
    {"objective": "multiclass", "num_class": 3},
]


def _data(objective):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    z = X[:, 0] + 0.5 * X[:, 1]
    if objective["objective"] == "binary":
        y = (z + rng.normal(size=600) > 0).astype(float)
    elif objective["objective"] == "multiclass":
        y = np.digitize(z + rng.normal(size=600), [-0.5, 0.5]).astype(float)
    else:
        # positive labels with a large mean: the log-link objectives' first step from score 0 is huge
        y = 150.0 * np.exp(0.3 * z) + rng.uniform(1.0, 5.0, size=600)
    return X, y


def _train(objective, X, y, **extra):
    params = {**objective, "verbose": -1, "num_leaves": 7, "learning_rate": 0.1, "device_type": "cpu", **extra}
    return flc.train(params, flc.Dataset(X, y), num_boost_round=10)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_refit_on_training_data_reproduces_the_model(objective):
    X, y = _data(objective)
    booster = _train(objective, X, y)
    refit = booster.refit(X, y, decay_rate=0.0)
    np.testing.assert_allclose(
        refit.predict(X, raw_score=True), booster.predict(X, raw_score=True), rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("decay_rate", [0.0, 0.5])
def test_refit_on_other_data_stays_finite_and_close(objective, decay_rate):
    X, y = _data(objective)
    booster = _train(objective, X[:400], y[:400])
    refit = booster.refit(X[200:], y[200:], decay_rate=decay_rate)
    raw, original = refit.predict(X, raw_score=True), booster.predict(X, raw_score=True)
    assert np.isfinite(raw).all()
    # refitting the leaves on overlapping data of the same distribution moves the raw scores only a little (on master,
    # the log-link refits moved them by ~mean(label) or returned NaN)
    assert np.max(np.abs(raw - original)) < 0.5 * max(1.0, float(np.std(original)))


def test_refit_without_boost_from_average_is_unchanged():
    # no bias in the first tree (its shrinkage is the learning rate): refit starts from 0, as training did
    objective = {"objective": "regression"}
    X, y = _data(objective)
    booster = _train(objective, X, y, boost_from_average=False)
    assert booster.dump_model()["tree_info"][0]["shrinkage"] == pytest.approx(0.1)
    refit = booster.refit(X, y, decay_rate=0.0)
    np.testing.assert_allclose(refit.predict(X), booster.predict(X), rtol=1e-9, atol=1e-9)


def test_refit_onto_zero_average_labels_shrinks_the_first_tree():
    # the first tree carries the bias of labels with mean ~150; refit labels whose average is exactly 0 (integers that
    # cancel) still need its leaves rebuilt as 0 + learning_rate * step, not a full step
    objective = {"objective": "regression"}
    X, y = _data(objective)
    booster = _train(objective, X, y)
    rng = np.random.default_rng(1)
    half = rng.integers(-50, 50, size=300).astype(float)
    y0 = np.concatenate([half, -half])
    rng.shuffle(y0)
    assert np.sum(y0) == 0.0
    refit = booster.refit(X, y0, decay_rate=0.0)
    leaf = booster.predict(X, pred_leaf=True)[:, 0]
    for i in np.unique(leaf):
        # L2 at score 0: the Newton step is the leaf's mean label
        np.testing.assert_allclose(refit.get_leaf_output(0, int(i)), 0.1 * np.mean(y0[leaf == i]), rtol=1e-6, atol=1e-9)
