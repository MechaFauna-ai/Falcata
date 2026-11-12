"""A query with zero documents (a 0 in ``group`` or in a ``.query`` file) is skipped.

lambdarank read the best-ranked document of every query, so an empty query read
``sorted_idx[0]`` of an empty vector (segfault). Its per-query weight was 0 / 0, so the
weighted ranking metrics came out NaN.
"""

import numpy as np
import pytest

import falcata as flc


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = rng.integers(0, 4, 40).astype(float)
    w = rng.uniform(0.5, 2.0, 40)
    return X, y, w


@pytest.mark.parametrize("weighted", [False, True])
def test_empty_queries_do_not_change_lambdarank(weighted):
    X, y, w = _data()
    params = {
        "objective": "lambdarank",
        "metric": ["ndcg", "map"],
        "eval_at": [1, 3],
        "min_data_in_leaf": 1,
        "num_threads": 1,
        "deterministic": True,
        "verbose": -1,
    }
    results = []
    for group in ([10, 20, 10], [0, 10, 0, 20, 10, 0]):
        ds = flc.Dataset(X, y, group=group, weight=w if weighted else None, params={"min_data_in_bin": 1})
        evals = {}
        booster = flc.train(params, ds, num_boost_round=3, valid_sets=[ds], callbacks=[flc.record_evaluation(evals)])
        results.append((booster.predict(X), evals["training"]))
    (pred, evals), (pred_empty, evals_empty) = results
    np.testing.assert_allclose(pred_empty, pred)
    for name, values in evals_empty.items():
        assert np.all(np.isfinite(values)), name
        if weighted and name.startswith("map"):
            # an empty query has weight 0, so it does not move the weighted average
            np.testing.assert_allclose(values, evals[name])


def test_empty_query_in_query_file(tmp_path):
    X, y, _ = _data()
    data = tmp_path / "t.csv"
    data.write_text("".join(f"{int(y[i])},{X[i, 0]},{X[i, 1]},{X[i, 2]}\n" for i in range(40)))
    (tmp_path / "t.csv.query").write_text("0\n15\n0\n25\n")
    ds = flc.Dataset(str(data), params={"verbose": -1, "min_data_in_bin": 1})
    booster = flc.train({"objective": "lambdarank", "min_data_in_leaf": 1, "verbose": -1}, ds, num_boost_round=2)
    assert np.all(np.isfinite(booster.predict(X)))
