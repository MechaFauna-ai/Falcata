# coding: utf-8
"""Row-wise sparse datasets whose rows hold more than 504 non-default elements in one feature partition.
CalcConstructHistogramKernelDim then halves blockDim.x (each thread covers two columns of a wide partition), and
the sparse construct kernels must stride over the row's elements: with one element per thread, the elements past
blockDim.x were dropped from every histogram, so the split finder's sums disagreed with the data partition and
quantized training produced leaf values far outside the label range (gbdt diverged, rf predictions with std >> 1
for a [0, 1] target; found on Numerai v5.3 at 200k rows). The data mimics that: K features uniform on {0..4} plus
200 nearly constant ones, which make the data sparse-ish (row-wise CSR bin) with ~0.8 K non-default values per row.
Builds with the cuda_plan key dense_rowdata train quantized sparse-ish data that fits on the dense row-wise bin
instead, so every test also runs with that key off there (skipped on builds without it)."""

import os

import numpy as np
import pytest

import falcata as flc

pytestmark = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled Falcata build (set TASK=cuda)",
)

PLANS = ["", "dense_rowdata:off"]


def _train(params, X, y, rounds):
    try:
        return flc.train(params, flc.Dataset(X, label=y, params=params), num_boost_round=rounds)
    except flc.basic.FalcataError as e:
        if "unknown key" in str(e) and params.get("cuda_plan"):
            pytest.skip(f"this build has no cuda_plan key {params['cuda_plan']}")
        raise


def _wide_sparse_rows(num_uniform, num_rows, num_sparse=200, seed=0):
    rng = np.random.default_rng(seed)
    X = np.empty((num_rows, num_uniform + num_sparse), np.float32)
    X[:, :num_uniform] = rng.integers(0, 5, size=(num_rows, num_uniform))
    X[:, num_uniform:] = np.where(rng.random((num_rows, num_sparse)) < 0.95, 2,
                                  rng.choice(np.array([0, 1, 3, 4]), size=(num_rows, num_sparse)))
    y = 0.5 + 0.05 * (X[:, 0] - 2) + 0.05 * (X[:, num_uniform - 1] - 2) + 0.1 * rng.standard_normal(num_rows)
    return X, np.clip(y, 0, 1).astype(np.float32)


def _node_errors(bst, X, y, learning_rate):
    """|stored value - value recomputed from the node's training rows| for every node of tree 0"""
    tree = bst.dump_model()["tree_info"][0]["tree_structure"]
    leaf = bst.predict(X, pred_leaf=True)
    leaf = (leaf[:, 0] if leaf.ndim == 2 else leaf).astype(int)
    init = float(y.astype(np.float64).mean())
    errors = []

    def walk(node):
        if "leaf_index" in node:
            leaves, value, count = [node["leaf_index"]], node["leaf_value"], node["leaf_count"]
        else:
            leaves = walk(node["left_child"]) + walk(node["right_child"])
            value, count = node["internal_value"], node["internal_count"]
        rows = np.isin(leaf, leaves)
        assert int(rows.sum()) == count
        errors.append(abs(value - (init + learning_rate * (y[rows].astype(np.float64) - init).mean())))
        return leaves

    walk(tree)
    return errors


# quantization noise only: fixed builds stay below 0.0015 (fixedpoint) / 0.04 (stochastic) on these shapes, the
# dropped row elements moved node values by 0.006 .. 18
TOLERANCE = {"fixedpoint": 0.005, "stochastic": 0.1}


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("quant_mode", ["stochastic", "fixedpoint"])
@pytest.mark.parametrize("shape", [(700, 5_000), (600, 20_000)])
def test_quantized_wide_sparse_rows_match_data(shape, quant_mode, plan):
    X, y = _wide_sparse_rows(*shape)
    params = {"objective": "regression", "device_type": "cuda", "quant_mode": quant_mode, "num_leaves": 15,
              "min_data_in_leaf": 100, "learning_rate": 1.0, "verbosity": -1, "seed": 7, "cuda_plan": plan}
    errors = _node_errors(_train(params, X, y, 1), X, y, 1.0)
    assert max(errors) < TOLERANCE[quant_mode], sorted(errors)[-5:]


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("quant_mode", ["stochastic", "fixedpoint"])
def test_quantized_wide_sparse_rows_bounded_predictions(quant_mode, plan):
    X, y = _wide_sparse_rows(700, 5_000)
    params = {"objective": "regression", "device_type": "cuda", "quant_mode": quant_mode, "num_leaves": 31,
              "min_data_in_leaf": 20, "learning_rate": 0.1, "verbosity": -1, "seed": 7, "cuda_plan": plan}
    for boosting in ["gbdt", "rf"]:
        p = dict(params, boosting=boosting)
        if boosting == "rf":
            p.update(bagging_fraction=0.632, bagging_freq=1)
        bst = _train(p, X, y, 20)
        assert bst.num_trees() == 20
        pred = bst.predict(X)
        assert -0.5 < pred.min() and pred.max() < 1.5, (boosting, pred.min(), pred.max())


def test_non_quantized_wide_sparse_rows_match_data():
    X, y = _wide_sparse_rows(700, 5_000)
    params = {"objective": "regression", "device_type": "cuda", "quant_mode": "none", "num_leaves": 15,
              "min_data_in_leaf": 100, "learning_rate": 1.0, "verbosity": -1, "seed": 7}
    assert max(_node_errors(_train(params, X, y, 1), X, y, 1.0)) < 1e-6
