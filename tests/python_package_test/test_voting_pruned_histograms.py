# coding: utf-8
"""Voting-parallel training builds the local histograms of every feature it may aggregate.

A feature the parent could not split on (locally) gets no local split candidate, but the global voting can
still select it from another machine's vote, and its histograms are then summed across machines. The voting
learner skipped building them, so stale histograms were summed: with lambda_l2 the trees took splits whose
gains exceeded the root's, with out-of-range leaf values. With top_k at least the number of features, voting
aggregates every voted feature and must match serial training here (constant hessian, same bins, even split).
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
    X = rng.integers(0, 8, size=(2000, 5)).astype(float)
    return X, X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=2000)


PARAMS = {
    "objective": "regression",
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "lambda_l2": 5.0,
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def _worker(rank, ports, queue):
    X, y = _data()
    part = np.arange(len(y)) % 2 == rank
    params = {
        **PARAMS,
        "tree_learner": "voting",
        "top_k": 20,
        "num_machines": 2,
        "machines": ",".join(f"127.0.0.1:{p}" for p in ports),
        "local_listen_port": ports[rank],
        "pre_partition": True,
    }
    booster = flc.train(params, flc.Dataset(X[part], label=y[part], params=params), 3)
    queue.put((rank, booster.predict(X)))


@pytest.mark.skipif(BuildInfo.has_cuda, reason="Parallel tree learners are CPU-only")
def test_voting_aggregates_fresh_histograms():
    X, y = _data()
    expected = flc.train(PARAMS, flc.Dataset(X, label=y, params=PARAMS), 3).predict(X)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    ports = _free_ports(2)
    procs = [ctx.Process(target=_worker, args=(r, ports, queue)) for r in range(2)]
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
        np.testing.assert_allclose(results[rank], expected, rtol=0, atol=1e-12)
