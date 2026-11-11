"""Training prediction arrays are ephemeral caches, not model pickle state."""

import pickle

import numpy as np

import falcata as flc


def test_pickle_omits_training_prediction_buffers():
    data = np.arange(160, dtype=np.float64).reshape(80, 2)
    train = flc.Dataset(data, label=np.arange(80))
    valid = train.create_valid(data + 0.5, label=np.arange(80))
    booster = flc.train(
        {"num_threads": 2, "verbosity": -1, "min_data_in_leaf": 1},
        train,
        valid_sets=[valid],
        num_boost_round=2,
        keep_training_booster=True,
    )
    booster.eval_train(feval=lambda prediction, dataset: ("custom", 0.0, False))
    booster.eval_valid(feval=lambda prediction, dataset: ("custom", 0.0, False))
    original = booster.__dict__["_Booster__inner_predict_buffer"]
    assert len(original) == 2
    assert all(prediction is not None for prediction in original)
    state = booster.__getstate__()
    assert state["_Booster__inner_predict_buffer"] == []
    assert state["_Booster__is_predicted_cur_iter"] == []
    assert state["_Booster__num_dataset"] == 0
    assert booster.__dict__["_Booster__inner_predict_buffer"] is original
    restored = pickle.loads(pickle.dumps(booster))
    np.testing.assert_array_equal(restored.predict(data), booster.predict(data))
