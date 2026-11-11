"""An importer must identify the source objective before choosing a link."""

import pytest

from falcata.importers import _catboost_objective


@pytest.mark.parametrize("params", [None, {}, {"loss_function": ""}, "invalid JSON"])
def test_missing_catboost_loss_is_rejected(params):
    with pytest.raises(ValueError, match="loss_function"):
        _catboost_objective({"model_info": {"params": params}})


@pytest.mark.parametrize("loss", ["RMSE", "MAE", "Quantile"])
def test_known_catboost_regression_loss(loss):
    assert _catboost_objective({"model_info": {"params": {"loss_function": loss}}}) == ("regression", None)
