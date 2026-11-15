# coding: utf-8
"""A random-forest tree without any split takes the global root output on every machine.

SerialTreeLearner::Train sets the root output from smaller_leaf_splits_. The voting-parallel learner keeps
this machine's sums there (the global ones apart), so a tree that found no split took each machine's local
root output, and the machines ended up with different models; the root's row count for path smoothing was
also this machine's in both parallel learners.
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


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    y = np.where(np.arange(600) < 300, 1.0, 5.0) + rng.normal(size=600)  # the machines' label means differ
    return X, y


PARAMS = {
    "objective": "regression",
    # rf keeps a tree without a split as a forest member with its root output (gbdt replaces it by a constant)
    "boosting": "rf",
    "feature_fraction": 0.8,
    "min_data_in_leaf": 1000,  # no split anywhere: every tree is a root
    "feature_pre_filter": False,
    "num_threads": 1,
    "verbose": -1,
}


def _worker(rank, ports, tree_learner, extra, queue):
    X, y = _data()
    part = slice(rank * 300, (rank + 1) * 300)
    params = {
        **PARAMS,
        **extra,
        "tree_learner": tree_learner,
        "num_machines": 2,
        "machines": ",".join(f"127.0.0.1:{p}" for p in ports),
        "local_listen_port": ports[rank],
        "pre_partition": True,
    }
    booster = flc.train(params, flc.Dataset(X[part], label=y[part], params=params), 3)
    queue.put((rank, booster.predict(X)))


@pytest.mark.skipif(BuildInfo.has_cuda, reason="Parallel tree learners are CPU-only")
@pytest.mark.parametrize("tree_learner", ["data", "voting"])
@pytest.mark.parametrize("extra", [{}, {"path_smooth": 50.0}])
def test_root_only_trees_match_serial(tree_learner, extra):
    X, y = _data()
    expected = flc.train({**PARAMS, **extra}, flc.Dataset(X, label=y), 3).predict(X)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    ports = _free_ports(2)
    procs = [ctx.Process(target=_worker, args=(r, ports, tree_learner, extra, queue)) for r in range(2)]
    for p in procs:
        p.start()
    try:
        results = dict(queue.get(timeout=120) for _ in procs)
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                p.join()
    assert [p.exitcode for p in procs] == [0, 0]
    for rank in (0, 1):
        np.testing.assert_allclose(results[rank], expected, rtol=0, atol=1e-10)
