"""Importing parsed CatBoost data must not mutate the caller's object."""

from copy import deepcopy

import falcata as flc


def test_catboost_import_preserves_input_dict():
    raw = {
        "features_info": {"float_features": [{"feature_index": 0, "flat_feature_index": 0}]},
        "model_info": {"params": {"loss_function": "RMSE"}},
        "oblivious_trees": [
            {
                "splits": [{"split_type": "FloatFeature", "float_feature_index": 0, "border": 0.5}],
                "leaf_values": [0.0, 1.0],
                "leaf_weights": [1, 1],
            }
        ],
        "scale_and_bias": [1.0, [0.0]],
    }
    before = deepcopy(raw)
    flc.from_catboost(raw)
    assert raw == before
