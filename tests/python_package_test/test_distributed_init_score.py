# coding: utf-8
"""Distributed boost_from_average starts from the global average label.

Each machine computed the regression / cross-entropy average of its own labels and the boosting averaged
those init scores, which is not the global average when the machines hold different numbers of rows (the
binary objective already summed the counts across machines). Data-parallel training on an uneven split
then started from a different score than serial training on all rows.
"""

import multiprocessing as mp
import socket

import numpy as np
import pytest

import falcata as flc

from .utils import BuildInfo

SPLIT = 700  # rank 0 gets the first 700 rows, rank 1 the other 1300


def _free_ports(n):
    socks = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def _data(objective):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 8, size=(2000, 4)).astype(float)  # every machine finds the serial bins
    y = X[:, 0] - X[:, 1] + 0.5 * rng.normal(size=2000) + np.where(np.arange(2000) < SPLIT, 3.0, 0.0)
    if objective == "cross_entropy":
        y = 1.0 / (1.0 + np.exp(-y))
    if objective == "poisson":
        y = np.exp(y / 4.0)
    return X, y


def _params(objective):
    return {"objective": objective, "num_leaves": 7, "num_threads": 1, "deterministic": True, "verbose": -1}


def _worker(rank, ports, objective, queue):
    X, y = _data(objective)
    part = slice(0, SPLIT) if rank == 0 else slice(SPLIT, None)
    params = {
        **_params(objective),
        "tree_learner": "data",
        "num_machines": 2,
        "machines": ",".join(f"127.0.0.1:{p}" for p in ports),
        "local_listen_port": ports[rank],
        "pre_partition": True,
    }
    booster = flc.train(params, flc.Dataset(X[part], label=y[part], params=params), 5)
    queue.put((rank, booster.predict(X, raw_score=True)))


@pytest.mark.skipif(BuildInfo.has_cuda, reason="Parallel tree learners are CPU-only")
@pytest.mark.parametrize("objective", ["regression", "cross_entropy", "poisson"])
def test_distributed_init_score_is_global(objective):
    X, y = _data(objective)
    expected = flc.train(_params(objective), flc.Dataset(X, label=y), 5).predict(X, raw_score=True)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    ports = _free_ports(2)
    procs = [ctx.Process(target=_worker, args=(r, ports, objective, queue)) for r in range(2)]
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
