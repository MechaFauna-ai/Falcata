"""Loaded models have no attached datasets to evaluate."""

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("source", ["text", "binary", "file"])
def test_loaded_model_evaluation_without_datasets(source, tmp_path):
    x = np.arange(80, dtype=float).reshape(40, 2)
    model = flc.train(
        {"objective": "regression", "device_type": "cpu", "num_threads": 2, "verbosity": -1},
        flc.Dataset(x, label=x[:, 0]),
        num_boost_round=2,
    )
    if source == "text":
        loaded = flc.Booster(model_str=model.model_to_string())
    elif source == "binary":
        loaded = flc.Booster(model_bin=model.model_to_binary())
    else:
        path = tmp_path / "model.txt"
        model.save_model(path)
        loaded = flc.Booster(model_file=path)
    assert loaded.eval_valid() == []
    with pytest.raises(ValueError, match="number of dataset"):
        loaded.eval_train()
