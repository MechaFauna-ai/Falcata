"""Training prediction arrays are ephemeral caches, not model pickle state."""

import pickle

import numpy as np

import falcata as flc


def test_pickle_omits_training_prediction_buffers():
    data = np.arange(160, dtype=np.float64).reshape(80, 2)
    booster = flc.train(
        {"num_threads": 2, "verbosity": -1, "min_data_in_leaf": 1},
        flc.Dataset(data, label=np.arange(80)),
        num_boost_round=2,
        keep_training_booster=True,
    )
    booster.eval_train(feval=lambda prediction, dataset: ("custom", 0.0, False))
    original = booster.__dict__["_Booster__inner_predict_buffer"]
    assert original[0] is not None
    state = booster.__getstate__()
    assert state["_Booster__inner_predict_buffer"] == []
    assert booster.__dict__["_Booster__inner_predict_buffer"] is original
    restored = pickle.loads(pickle.dumps(booster))
    np.testing.assert_array_equal(restored.predict(data), booster.predict(data))
