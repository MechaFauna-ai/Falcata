"""reset_parameter of an objective or metric parameter takes effect.

The objective and the metrics copy their parameters (alpha, sigmoid, scale_pos_weight, eval_at, ...) when they are
built, and a reset rebuilt them only when it named ``objective``. A reset of only such a parameter was silently
ignored: training went on with the old value.
"""

import numpy as np
import pytest

import falcata as flc

# device_type=cpu: the reset semantics are device independent; pinned so the exact comparison holds
BASE = {
    "device_type": "cpu",
    "deterministic": True,
    "num_threads": 1,
    "num_leaves": 7,
    "min_data_in_leaf": 5,
    "verbose": -1,
}


def _data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1500, 5))
    y = x[:, 0] + np.sin(x[:, 1]) + rng.standard_t(2, size=1500) * 0.5
    counts = rng.poisson(np.exp(x[:, 0] * 0.5)).astype(float)
    return x, y, counts


@pytest.mark.parametrize(
    ("objective", "label", "before", "after"),
    [
        ("quantile", "y", {"alpha": 0.9}, {"alpha": 0.2}),
        ("huber", "y", {"alpha": 0.5}, {"alpha": 3.0}),
        ("fair", "y", {"fair_c": 1.0}, {"fair_c": 0.1}),
        ("regression", "y", {"reg_sqrt": False}, {"reg_sqrt": True}),
        ("tweedie", "counts", {"tweedie_variance_power": 1.2}, {"tweedie_variance_power": 1.8}),
        ("poisson", "counts", {"poisson_max_delta_step": 0.7}, {"poisson_max_delta_step": 0.1}),
        ("binary", "binary", {"scale_pos_weight": 1.0}, {"scale_pos_weight": 5.0}),
        ("binary", "binary", {"sigmoid": 1.0}, {"sigmoid": 3.0}),
        ("binary", "binary", {"is_unbalance": False}, {"is_unbalance": True}),
    ],
)
def test_reset_objective_parameter_matches_continued_training(objective, label, before, after):
    x, y, counts = _data()
    target = {"y": y, "counts": counts, "binary": (y > 0.5).astype(float)}[label]
    booster = flc.train(
        {**BASE, "objective": objective, **before},
        flc.Dataset(x, target, free_raw_data=False),
        num_boost_round=3,
        keep_training_booster=True,
    )
    initial = booster.model_to_string()
    booster.reset_parameter(after)
    for _ in range(4):
        booster.update()
    expected = flc.train(
        {**BASE, "objective": objective, **after},
        flc.Dataset(x, target),
        num_boost_round=4,
        init_model=flc.Booster(model_str=initial),
    )
    np.testing.assert_allclose(
        booster.predict(x, raw_score=True), expected.predict(x, raw_score=True), rtol=0, atol=1e-12
    )


def test_reset_eval_at_rebuilds_metrics():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(600, 4))
    y = np.clip(np.round(x[:, 0] + 1), 0, 3)
    group = [20] * 30
    train = flc.Dataset(x, y, group=group, free_raw_data=False)
    valid = flc.Dataset(x, y, group=group, reference=train)
    params = {**BASE, "objective": "lambdarank", "metric": "ndcg", "eval_at": [1, 3]}
    booster = flc.Booster(params, train)
    booster.add_valid(valid, "valid")
    booster.update()
    assert [name for _, name, _, _ in booster.eval_valid()] == ["ndcg@1", "ndcg@3"]
    booster.reset_parameter({"eval_at": [2]})
    assert [name for _, name, _, _ in booster.eval_valid()] == ["ndcg@2"]
    assert [name for _, name, _, _ in booster.eval_train()] == ["ndcg@2"]


def test_reset_objective_parameter_on_loaded_model():
    x, y, _ = _data()
    booster = flc.train({**BASE, "objective": "quantile", "alpha": 0.9}, flc.Dataset(x, y), num_boost_round=2)
    loaded = flc.Booster(model_str=booster.model_to_string())
    # a model-loaded booster has no objective to rebuild; the reset only updates its config, as before
    loaded.reset_parameter({"alpha": 0.5})
    np.testing.assert_array_equal(loaded.predict(x), booster.predict(x))


@pytest.mark.parametrize("objective", ["lambdarank", "rank_xendcg"])
def test_reset_metric_parameter_keeps_objective_state(objective):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(600, 4))
    y = np.clip(np.round(x[:, 0] + 1), 0, 3)
    position = np.tile(np.arange(20), 30)
    params = {**BASE, "objective": objective, "metric": "ndcg", "eval_at": [1, 3], "objective_seed": 4}

    def boosted(reset):
        booster = flc.Booster(params, flc.Dataset(x, y, group=[20] * 30, position=position))
        for i in range(6):
            if reset and i == 3:
                # evaluation only: the objective (position biases, rank_xendcg generators) must carry on as is
                booster.reset_parameter({"eval_at": [2]})
            booster.update()
        return booster.predict(x, raw_score=True)

    np.testing.assert_array_equal(boosted(reset=True), boosted(reset=False))
