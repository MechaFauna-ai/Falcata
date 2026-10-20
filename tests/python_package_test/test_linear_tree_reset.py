"""Linear learner resets must preserve the complete fitted model and its values."""

import numpy as np
import pytest

import falcata as lgb
from falcata.basic import _LIB, _safe_call


def _data(nrows=192, *, missing=True):
    rows = np.arange(nrows)
    x = np.column_stack((rows % 31 / 31.0, rows % 5, rows % 7 * 0.2))
    if missing:
        x[rows % 13 == 0, 2] = np.nan
    y = (2 * x[:, 0] + (rows % 5 == 1)).astype(np.float32)
    return x, y


def _params(**overrides):
    return {
        "device_type": "cpu",
        "quant_mode": "none",
        "objective": "regression",
        "linear_tree": True,
        "num_leaves": 5,
        "min_data_in_leaf": 3,
        "min_data_in_bin": 1,
        "num_threads": 2,
        "verbosity": -1,
        **overrides,
    }


def _pair(x, y, params):
    data = lgb.Dataset(x, y, params=params, free_raw_data=False)
    small = lgb.Booster(params, train_set=data)
    # Before the fix, ResetConfig shrinks only the base learner's storage.
    # This control therefore has valid linear workspace throughout the test.
    preallocated = lgb.Booster(dict(params, num_leaves=32, num_threads=4), train_set=data)
    for model in (small, preallocated):
        # Equal reset counts also preserve the feature-sampling RNG stream.
        model.reset_parameter({"num_leaves": params["num_leaves"], "num_threads": params["num_threads"]})
    assert small.params == preallocated.params
    return small, preallocated, data


def _assert_equal(actual, expected, x):
    # Includes every split, leaf constant, linear coefficient and feature list.
    assert actual.dump_model() == expected.dump_model()
    assert actual.model_to_string() == expected.model_to_string()
    # Prediction shares process-wide thread state with subsequent updates.
    # Keep ordinary comparisons on each booster's configured training threads.
    predictions = actual.predict(x, num_threads=actual.params["num_threads"])
    assert np.isfinite(predictions).all()
    assert np.ptp(predictions) > 0
    np.testing.assert_array_equal(predictions, expected.predict(x, num_threads=expected.params["num_threads"]))


@pytest.mark.parametrize("sample_features", [False, True])
@pytest.mark.parametrize("leaves", [(5,), (6,), (3, 9, 4, 12, 5, 6)])
def test_linear_leaf_capacity_reset(leaves, sample_features):
    x, y = _data()
    params = _params(categorical_feature=[1])
    if sample_features:
        params.update(feature_fraction=0.8, feature_fraction_bynode=0.8, seed=19)
    small, control, data = _pair(x, y, params)
    for _ in range(3):
        for model in (small, control):
            assert not model.update()
        # Validate the preallocation/config/RNG assumptions before leaf growth.
        _assert_equal(small, control, x)
    assert small.dump_model()["feature_infos"]["Column_1"]["values"] == [-1, 0, 1, 2, 3, 4]
    for reset_index, count in enumerate(leaves):
        for model in (small, control):
            with pytest.raises(lgb.basic.FalcataError, match="quant"):
                model.reset_parameter({"quant_mode": "stochastic"})
            model.reset_parameter({"num_leaves": count})
            # Exercise the exact same-data C API call (Python update skips it).
            _safe_call(_LIB.FLC_BoosterResetTrainingData(model._handle, data._handle))
            assert not model.update()
        assert small.current_iteration() == 4 + reset_index
        _assert_equal(small, control, x)
        fitted_leaves = small.dump_model()["tree_info"][-1]["num_leaves"]
        assert 1 < fitted_leaves <= count
        if not sample_features:
            assert fitted_leaves == count


def test_linear_reset_grows_feature_and_thread_workspaces():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(1536, 8))
    y = x @ np.arange(1, 9, dtype=float)
    small, control, _ = _pair(x, y, _params(num_leaves=2))
    for _ in range(3):
        for model in (small, control):
            assert not model.update()
        _assert_equal(small, control, x)
    for count, threads in ((2, 4), (16, 4), (3, 1), (24, 2), (5, 4)):
        for model in (small, control):
            model.reset_parameter({"num_leaves": count, "num_threads": threads})
            assert not model.update()
        _assert_equal(small, control, x)
    assert any(
        len(node["leaf_features"]) > 2
        for tree in small.dump_model()["tree_info"]
        for node in _leaves(tree["tree_structure"])
    )


@pytest.mark.parametrize("reset_kind", ["data", "leaves"])
@pytest.mark.parametrize("nrows", [192, 1536])
def test_linear_reset_retains_prediction_thread_capacity(nrows, reset_kind):
    rows = np.arange(nrows)
    x = np.column_stack((rows % 31 / 31.0, rows % 7 * 0.2))
    y = (2 * x[:, 0] + 3 * x[:, 1]).astype(np.float32)
    small, control, original = _pair(x, y, _params(num_threads=4))
    datasets = [original]
    for _ in range(3):
        for model in (small, control):
            assert not model.update()
        _assert_equal(small, control, x)
    for reset_index, low_threads in enumerate((1, 2, 1)):
        count = 6 + 3 * reset_index
        if reset_kind == "data":
            target = lgb.Dataset(x.copy(), y.copy(), reference=original, free_raw_data=False).construct()
            datasets.append(target)
            assert small.params["num_threads"] == 4
        else:
            for model in (small, control):
                # Reinitialize leaf storage at the lower effective thread count.
                model.reset_parameter({"num_leaves": count, "num_threads": low_threads})
        for model in (small, control):
            model.predict(x, num_threads=low_threads)
            # The control keeps the identical original data and has enough leaf
            # capacity to avoid InitLinear on both the base and revised builds.
            train_set = target if reset_kind == "data" and model is small else None
            assert not model.update(train_set=train_set)
        _assert_equal(small, control, x)
        for model in (small, control):
            # Restore the earlier allocation's thread count without ResetConfig.
            model.predict(x, num_threads=4)
            assert not model.update()
        assert small.current_iteration() == 5 + 2 * reset_index
        _assert_equal(small, control, x)
        assert small.dump_model()["tree_info"][-1]["num_leaves"] == (count if reset_kind == "leaves" else 5)


def _leaves(node):
    if "leaf_index" in node:
        yield node
    else:
        yield from _leaves(node["left_child"])
        yield from _leaves(node["right_child"])


def test_linear_parameter_only_reset():
    x, y = _data()
    small, control, _ = _pair(x, y, _params(categorical_feature=[1]))
    for _ in range(3):
        for model in (small, control):
            assert not model.update()
    for model in (small, control):
        model.reset_parameter({"learning_rate": 0.07, "linear_lambda": 0.3})
        assert not model.update()
    _assert_equal(small, control, x)


def test_linear_reset_training_data():
    x, y = _data(missing=False)
    # Match the Dataset's stored float32 raw values in the fresh continuation's
    # initial predictions, so exact equality also checks the reset's arithmetic.
    x = x.astype(np.float32)
    y += 3 * x[:, 2]
    params = _params(categorical_feature=[1])
    original = lgb.Dataset(x, y, params=params, free_raw_data=False)
    model = lgb.Booster(params, train_set=original)
    # The native reset validates against the previous Dataset before replacing
    # its pointer. Keep those wrappers alive across repeated Python updates.
    datasets = [original]
    for _ in range(3):
        assert not model.update()
    for nrows, missing in ((1536, True), (96, False), (384, True)):
        x_new, y_new = _data(nrows, missing=missing)
        x_new = x_new.astype(np.float32)
        y_new += 3 * np.nan_to_num(x_new[:, 2])
        # Reference Dataset parameters omit num_threads; bind construction too.
        target = lgb.Dataset(x_new, y_new, params=params, reference=original, free_raw_data=False)
        oracle_data = lgb.Dataset(x_new, y_new, params=params, reference=original, free_raw_data=False)
        datasets.append(target)
        # A fresh learner on the new raw data cannot inherit stale row maps or
        # missing flags; continue the complete previous model, with no sampling.
        oracle = lgb.train(
            dict(params, num_iterations=1),
            oracle_data,
            num_boost_round=1,
            init_model=model,
            keep_training_booster=True,
        )
        assert not model.update(train_set=target)
        assert model.dump_model() == oracle.dump_model()
        # train() records num_iterations; compare the entire serialized model
        # before that parameter metadata, as well as the full dump above.
        assert model.model_to_string().split("parameters:")[0] == oracle.model_to_string().split("parameters:")[0]
        prediction = model.predict(x_new, num_threads=model.params["num_threads"])
        assert np.isfinite(prediction).all()
        np.testing.assert_array_equal(prediction, oracle.predict(x_new, num_threads=oracle.params["num_threads"]))
    assert any(
        2 in node["leaf_features"]
        for tree in model.dump_model()["tree_info"]
        for node in _leaves(tree["tree_structure"])
    )
