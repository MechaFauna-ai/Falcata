"""from_xgboost must map a logistic base_score the way the writing XGBoost did.

XGBoost 3.2+ clamps a logistic base_score to [1e-6, 1 - 1e-6] before taking
the margin; older versions do not clamp inside (0, 1). Both compute
-log(1/p - 1) in float32. The importer clamped to 1e-12 in float64, so a model
whose base_score is 0 or 1 -- e.g. fitted to all-negative labels -- converted
~14 logits off, and base_score 1 - 1e-6 was 0.07 off on every XGBoost version.
"""

import json

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from falcata.importers import from_xgboost  # noqa: E402


@pytest.mark.parametrize("objective", ["binary:logistic", "reg:logistic"])
@pytest.mark.parametrize(
    ("label", "base_score"),
    [(0.0, None), (0.0, 1.0), (1.0, None), (1.0, 0.0), (0.0, 1e-8), (1.0, 1 - 1e-8), (0.0, 0.3)],
)
def test_logistic_base_score_matches_xgboost(objective, label, base_score):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 2)).astype(np.float32)
    y = np.full(200, label)
    params = {"objective": objective, "nthread": 1, "max_depth": 3}
    if base_score is not None:
        params["base_score"] = base_score
    bst = xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=3)
    # margins: XGBoost sums them in float32, so compare there, to float32 rounding
    ref = bst.predict(xgb.DMatrix(x), output_margin=True)
    got = from_xgboost(bst).predict(x, raw_score=True)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize(
    ("version", "base_score", "margin"),
    [
        # XGBoost < 3.2 does not clamp inside (0, 1) ...
        ([2, 1, 4], "[1E-8]", -18.420681),
        ([3, 1, 0], "[1E-8]", -18.420681),
        # ... 3.2+ clamps to [1e-6, 1 - 1e-6]
        ([3, 2, 0], "[1E-8]", -13.81551),
        ([3, 2, 0], "[0E0]", -13.81551),
        # both compute -log(1/p - 1) in float32, which is not the float64 logit near 1
        ([2, 1, 4], "[9.99999E-1]", 13.74516),
        ([3, 4, 1], "[9.99999E-1]", 13.74516),
    ],
)
def test_logistic_base_score_follows_writer_version(version, base_score, margin):
    x = np.zeros((4, 2), dtype=np.float32)
    bst = xgb.train(
        {"objective": "binary:logistic", "nthread": 1, "eta": 0.0},
        xgb.DMatrix(x, label=np.array([0.0, 1.0, 0.0, 1.0])),
        num_boost_round=1,
    )
    raw = json.loads(bst.save_raw("json"))
    raw["version"] = version
    raw["learner"]["learner_model_param"]["base_score"] = base_score
    got = from_xgboost(raw).predict(x, raw_score=True)
    np.testing.assert_allclose(got, margin, rtol=1e-6)
