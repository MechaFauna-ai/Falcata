"""Loaded models have no attached datasets to evaluate."""

import pickle

import numpy as np
import pytest

import falcata as flc


@pytest.mark.parametrize("source", ["text", "binary", "file", "pickle", "legacy-pickle"])
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
    elif source in ("pickle", "legacy-pickle"):
        ds = flc.Dataset(x, label=x[:, 0])
        model = flc.train(
            {"objective": "regression", "device_type": "cpu", "num_threads": 2, "verbosity": -1},
            ds,
            num_boost_round=2,
            valid_sets=[flc.Dataset(x, label=x[:, 0], reference=ds)],
            keep_training_booster=True,
        )
        if source == "pickle":
            loaded = pickle.loads(pickle.dumps(model))
        else:
            state = model.__getstate__()
            state.pop("_Booster__num_dataset", None)
            loaded = flc.Booster.__new__(flc.Booster)
            loaded.__setstate__(state)
    else:
        path = tmp_path / "model.txt"
        model.save_model(path)
        loaded = flc.Booster(model_file=path)
    assert loaded.eval_valid() == []
    with pytest.raises(ValueError, match="number of dataset"):
        loaded.eval_train()
