"""from_xgboost must apply DART tree weights.

A DART model predicts sum(weight_drop[i] * tree_i). XGBoost >= 3.4 saves it as
a "gbtree" whose model carries "weight_drop"; the importer ignored that list,
so a dart-trained model converted without error and predicted something else
(relative error up to ~1 in fuzzing). Older XGBoost saves the
{"name": "dart", "gbtree": ..., "weight_drop": ...} layout, which the importer
refused; it converts now too.
"""

import json
import warnings

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from falcata.importers import from_xgboost  # noqa: E402


def _dart(objective, **extra):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 3)).astype(np.float32)
    lin = x[:, 0] + x[:, 1]
    if objective == "binary:logistic":
        y = (lin > 0).astype(float)
    elif objective == "multi:softprob":
        y = np.clip(np.floor(lin + 1), 0, 2)
        extra["num_class"] = 3
    else:
        y = lin
    params = {"booster": "dart", "objective": objective, "rate_drop": 0.5, "max_depth": 3, "nthread": 1, "seed": 1}
    params.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # booster=dart is deprecated in xgboost >= 3.4
        bst = xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=10)
    return bst, x


def _weights(raw):
    gb = raw["learner"]["gradient_booster"]
    model = gb.get("gbtree", gb)["model"]
    return model, model.get("weight_drop") or gb.get("weight_drop")


@pytest.mark.parametrize(
    ("objective", "extra"),
    [
        ("reg:squarederror", {}),
        ("binary:logistic", {}),
        ("multi:softprob", {}),
        ("reg:squarederror", {"num_parallel_tree": 2, "subsample": 0.8}),
    ],
)
def test_dart_weights_are_applied(objective, extra):
    bst, x = _dart(objective, **extra)
    raw = json.loads(bst.save_raw("json"))
    _, weights = _weights(raw)
    assert weights is not None
    assert any(abs(w - 1.0) > 1e-6 for w in weights)
    ref = bst.predict(xgb.DMatrix(x))
    got = np.asarray(from_xgboost(bst).predict(x)).reshape(ref.shape)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)

    # the pre-3.4 layout of the same model
    model, weights = _weights(raw)
    model = dict(model)
    model.pop("weight_drop", None)
    raw["learner"]["gradient_booster"] = {
        "name": "dart",
        "gbtree": {"name": "gbtree", "model": model},
        "weight_drop": weights,
    }
    got = np.asarray(from_xgboost(raw).predict(x)).reshape(ref.shape)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_dart_weight_count_mismatch_is_refused():
    bst, _ = _dart("reg:squarederror")
    raw = json.loads(bst.save_raw("json"))
    model, weights = _weights(raw)
    model["weight_drop"] = list(weights)[:-1]
    raw["learner"]["gradient_booster"].pop("weight_drop", None)
    with pytest.raises(ValueError, match="weight_drop"):
        from_xgboost(raw)
