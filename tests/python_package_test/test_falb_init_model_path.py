"""A FALB model file must work wherever a model file path is accepted.

Booster(model_file=...) sniffed the FALB magic in Python, but every path that
hands the file to the C++ loader -- train(init_model="model.falb"),
cv(init_model=...), FLC_BoosterCreateFromModelfile, the CLI's input_model --
read it as text and failed with "Unknown model format or submodel type".
"""

import numpy as np
import pytest

import falcata as flc


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 4))
    x[:, 3] = rng.integers(0, 5, size=300)
    y = x[:, 0] + (x[:, 3] == 2)
    params = {"verbose": -1, "num_threads": 1, "num_leaves": 7, "deterministic": True}
    ds = flc.Dataset(x, label=y, categorical_feature=[3], free_raw_data=False)
    booster = flc.train(params, ds, num_boost_round=3)
    d = tmp_path_factory.mktemp("models")
    booster.save_model(d / "m.txt")
    booster.save_model(d / "m.falb", with_stats=True, with_diagnostics=True)
    return x, y, params, d


def _continued(x, y, params, init):
    ds = flc.Dataset(x, label=y, categorical_feature=[3])
    out = flc.train(params, ds, num_boost_round=2, init_model=init).model_to_string()
    return out[out.index("Tree=0") : out.index("end of trees")]


def test_train_init_model_accepts_falb_path(setup):
    x, y, params, d = setup
    assert _continued(x, y, params, d / "m.falb") == _continued(x, y, params, d / "m.txt")
    assert _continued(x, y, params, str(d / "m.falb")) == _continued(x, y, params, str(d / "m.txt"))


def test_cv_init_model_accepts_falb_path(setup):
    x, y, params, d = setup
    res = flc.cv(params, flc.Dataset(x, label=y), num_boost_round=2, nfold=2, stratified=False, init_model=d / "m.falb")
    assert res


def test_inner_predictor_from_falb_file_predicts_like_text(setup):
    x, _, _, d = setup
    falb = flc.basic._InnerPredictor.from_model_file(d / "m.falb", pred_parameter={})
    text = flc.basic._InnerPredictor.from_model_file(d / "m.txt", pred_parameter={})
    np.testing.assert_array_equal(falb.predict(x), text.predict(x))
