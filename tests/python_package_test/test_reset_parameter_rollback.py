import numpy as np
import pytest

import falcata as flc

# GBDT::ResetConfig reset the tree learner (and the sample strategy) with a
# pointer to the new config before the sample strategy validated it. When that
# failed ("Cannot use bagging in GOSS"), the new config was freed while they
# still pointed at it, and the next update read freed memory (ASan:
# heap-use-after-free in GOSSStrategy::Bagging). The booster now keeps training
# on its previous parameters.


@pytest.mark.parametrize("goss", [{"data_sample_strategy": "goss"}, {"boosting": "goss"}])
def test_failed_reset_keeps_training_on_the_previous_config(goss):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 5))
    y = X[:, 0] + 0.1 * rng.standard_normal(500)
    params = {
        "objective": "regression",
        "device_type": "cpu",
        "verbose": -1,
        "deterministic": True,
        "num_threads": 1,
        **goss,
    }
    bst = flc.train(params, flc.Dataset(X, y), 2, keep_training_booster=True)
    with pytest.raises(flc.basic.FalcataError, match="Cannot use bagging in GOSS"):
        bst.reset_parameter({"bagging_fraction": 0.5, "bagging_freq": 1, "lambda_l2": 3.0})
    for _ in range(13):  # GOSS starts sampling after 1 / learning_rate rounds
        bst.update()
    assert np.all(np.isfinite(bst.predict(X)))


@pytest.mark.parametrize("first_reset_fails", [False, True])
def test_goss_resets_after_sampling_started(first_reset_fails):
    # Once GOSS samples (after 1 / learning_rate rounds) the tree learner trains
    # on GOSS's subset Dataset. Every ResetSampleConfig replaced that subset, so
    # the learner pointed at a freed Dataset until the next Bagging call, and a
    # second reset in between read it (ASan: heap-use-after-free in
    # SerialTreeLearner::ResetConfig / Dataset::num_features).
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 5))
    y = X[:, 0] + 0.1 * rng.standard_normal(500)
    params = {"objective": "regression", "data_sample_strategy": "goss", "device_type": "cpu", "verbose": -1}
    bst = flc.train(params, flc.Dataset(X, y), 15, keep_training_booster=True)
    if first_reset_fails:
        with pytest.raises(flc.basic.FalcataError, match="Cannot use bagging in GOSS"):
            bst.reset_parameter({"bagging_fraction": 0.5, "bagging_freq": 1})
    else:
        bst.reset_parameter({"lambda_l2": 1.0})
    bst.reset_parameter({"lambda_l2": 2.0, "top_rate": 0.3, "other_rate": 0.1})
    for _ in range(3):
        bst.update()
    assert np.all(np.isfinite(bst.predict(X)))
