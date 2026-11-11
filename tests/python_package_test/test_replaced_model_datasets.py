"""Replacing a model detaches Python state for native training datasets."""

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("format", ["text", "binary"])
def test_replacing_training_model_detaches_datasets(format):
    x = np.arange(120, dtype=float).reshape(60, 2)
    ds = flc.Dataset(x, label=x[:, 0])
    model = flc.train(
        {"objective": "regression", "device_type": "cpu", "num_threads": 2, "verbosity": -1},
        ds,
        num_boost_round=2,
        valid_sets=[flc.Dataset(x, label=x[:, 0], reference=ds)],
        keep_training_booster=True,
    )
    expected = model.predict(x)
    if format == "text":
        model.model_from_string(model.model_to_string())
    else:
        model.model_from_binary(model.model_to_binary())
    np.testing.assert_array_equal(model.predict(x), expected)
    with pytest.raises(ValueError, match="number of dataset"):
        model._Booster__inner_predict(data_idx=0)
    assert not hasattr(model, "train_set")
    assert not hasattr(model, "valid_sets")
    assert model.eval_valid() == []
    with pytest.raises(ValueError, match="number of dataset"):
        model.eval_train()
