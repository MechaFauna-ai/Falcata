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
    params = {"objective": "regression", "verbose": -1, "deterministic": True, "num_threads": 1, **goss}
    bst = flc.train(params, flc.Dataset(X, y), 2, keep_training_booster=True)
    with pytest.raises(flc.basic.FalcataError, match="Cannot use bagging in GOSS"):
        bst.reset_parameter({"bagging_fraction": 0.5, "bagging_freq": 1, "lambda_l2": 3.0})
    for _ in range(13):  # GOSS starts sampling after 1 / learning_rate rounds
        bst.update()
    assert np.all(np.isfinite(bst.predict(X)))
