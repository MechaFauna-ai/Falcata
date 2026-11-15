# coding: utf-8
"""A huge max_cat_threshold must not break distributed training.

The data- and voting-parallel learners sized the SyncUpGlobalBestSplit slots with
SplitInfo::Size(max_cat_threshold), an int. max_cat_threshold is only checked > 0, so
2**30 + 1 wrapped the size to 98 bytes: the larger leaf's split was written past the
end of its slot and the collective buffer (both workers segfaulted). A split carries
at most min(max_cat_threshold, (bins + 1) / 2) categories, so on this data any
max_cat_threshold >= 6 must give the same model.
"""

import multiprocessing as mp
import socket

import numpy as np
import pytest

import falcata as flc

from .utils import BuildInfo


def _free_ports(n):
    socks = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def _worker(rank, ports, tree_learner, max_cat_threshold, queue):
    rng = np.random.default_rng(0)
    n = 4000
    cat = rng.integers(0, 12, size=n)
    X = np.column_stack([cat, rng.normal(size=n)])
    # four groups of three categories: the best splits send several categories left
    y = np.repeat([0.0, 3.0, -2.0, 5.0], 3)[cat] + 0.3 * X[:, 1] + 0.1 * rng.normal(size=n)
    part = slice(rank * n // 2, (rank + 1) * n // 2)
    params = {
        "objective": "regression",
        "tree_learner": tree_learner,
        "num_machines": 2,
        "machines": ",".join(f"127.0.0.1:{p}" for p in ports),
        "local_listen_port": ports[rank],
        "pre_partition": True,
        "num_leaves": 7,
        "num_threads": 1,
        "deterministic": True,
        "verbose": -1,
        "max_cat_threshold": max_cat_threshold,
        "max_cat_to_onehot": 2,
        "min_data_per_group": 5,
        "cat_smooth": 1,
        "cat_l2": 1,
    }
    ds = flc.Dataset(X[part], label=y[part], categorical_feature=[0], params=params)
    booster = flc.train(params, ds, 3)
    queue.put((rank, booster.model_to_string().split("parameters:")[0]))


def _train(tree_learner, max_cat_threshold):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    ports = _free_ports(2)
    procs = [ctx.Process(target=_worker, args=(r, ports, tree_learner, max_cat_threshold, queue)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    assert [p.exitcode for p in procs] == [0, 0]
    models = dict(queue.get(timeout=10) for _ in procs)
    return models[0]


@pytest.mark.skipif(BuildInfo.has_cuda, reason="data/voting-parallel training is not supported by the CUDA version")
@pytest.mark.parametrize("tree_learner", ["data", "voting"])
def test_huge_max_cat_threshold_matches_a_small_one(tree_learner):
    reference = _train(tree_learner, 32)
    assert "num_cat=" in reference
    assert "num_cat=0" not in reference, "the test needs categorical splits"
    assert _train(tree_learner, 2**30 + 1) == reference
