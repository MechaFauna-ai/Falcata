import numpy as np
import pytest

import falcata as flc

# When the first iteration can only grow stumps, GBDT stops training but keeps
# that iteration's constant trees; it did not count them in iter_, which
# rescoring a new training set (update(train_set=...)) or a new validation set
# relies on. After a parameter reset let later iterations split, the rescored
# scores missed the last iteration's trees (0.2 away from predict()).
# Found by the fz-cuda fuzzer (CPU code).


def _data(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((600, 4))
    return X, X[:, 0] + 0.1 * rng.standard_normal(600)


@pytest.mark.parametrize("objective", ["regression", "multiclassova"])
def test_rescoring_after_a_stump_first_iteration(objective):
    X, y = _data(0)
    X2, y2 = _data(1)
    if objective == "multiclassova":
        y, y2 = np.digitize(y, [-0.5, 0.5]).astype(float), np.digitize(y2, [-0.5, 0.5]).astype(float)
    params = {
        "objective": objective,
        "min_data_in_leaf": 700,
        "feature_pre_filter": False,
        "verbose": -1,
        "device_type": "cpu",
    }
    if objective == "multiclassova":
        params["num_class"] = 3
    train = flc.Dataset(X, y, params=params, free_raw_data=False)
    bst = flc.Booster(params, train)
    assert bst.update()  # stumps only: finished
    assert bst.current_iteration() == 1
    bst.reset_parameter({"min_data_in_leaf": 20})
    bst.update()
    bst.update()
    new = flc.Dataset(X2, y2, reference=train, free_raw_data=False)
    bst.update(train_set=new)
    # the inner (rescored) predictions must equal predicting with the model
    np.testing.assert_allclose(bst._Booster__inner_predict(data_idx=0), bst.predict(X2), atol=1e-10)
    valid = flc.Dataset(X, y, reference=train)
    bst.add_valid(valid, "v")
    np.testing.assert_allclose(bst._Booster__inner_predict(data_idx=1), bst.predict(X), atol=1e-10)


@pytest.mark.parametrize("uniform_drop", [False, True])
def test_dart_continues_after_a_stump_first_iteration(uniform_drop):
    # DART keeps one weight per counted iteration; the kept stump iteration needs one too,
    # or the next dropping step reads tree_weight_[0] of an empty vector
    X, y = _data(0)
    params = {
        "objective": "regression",
        "boosting": "dart",
        "drop_rate": 0.9,
        "skip_drop": 0.0,
        "uniform_drop": uniform_drop,
        "min_data_in_leaf": 700,
        "feature_pre_filter": False,
        "verbose": -1,
        "device_type": "cpu",
    }
    bst = flc.Booster(params, flc.Dataset(X, y, params=params))
    assert bst.update()
    bst.reset_parameter({"min_data_in_leaf": 20})
    for _ in range(5):
        bst.update()
    assert np.all(np.isfinite(bst.predict(X)))
