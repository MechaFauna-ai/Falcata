"""from_xgboost must map a degenerate logistic base_score the way XGBoost does.

XGBoost clamps a logistic base_score to [1e-6, 1 - 1e-6] before taking the
margin (in float32). The importer clamped to 1e-12, so a model whose
base_score is 0 or 1 -- e.g. fitted to all-negative labels -- converted to a
model ~14 logits off: XGBoost predicts 1e-6 where Falcata predicted 1e-12,
and with base_score=1 and negative leaves 1e-7 against 0.086.
"""

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
