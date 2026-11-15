# coding: utf-8
"""Data-parallel training skips the features the parent could not split on, as serial training does.

The serial learner does not evaluate, in a leaf's children, a feature on which the leaf itself had no valid
split. The data-parallel learner evaluated every feature in every leaf, so a child that found a positive gain on
such a feature split differently from serial training (with lambda_l2 or max_delta_step, for instance), even
with a constant hessian, the same bins and an even row split.
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
    X = rng.integers(0, 8, size=(2000, 5)).astype(float)  # every machine finds the serial bins
    return X, X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=2000)


def _params(extra):
    return {
        "objective": "regression",
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "num_threads": 1,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
        **extra,
    }


def _worker(rank, ports, extra, queue):
    X, y = _data()
    part = np.arange(len(y)) % 2 == rank
    params = {
        **_params(extra),
        "tree_learner": "data",
        "num_machines": 2,
        "machines": ",".join(f"127.0.0.1:{p}" for p in ports),
        "local_listen_port": ports[rank],
        "pre_partition": True,
    }
    booster = flc.train(params, flc.Dataset(X[part], label=y[part], params=params), 10)
    queue.put((rank, booster.predict(X)))


@pytest.mark.skipif(BuildInfo.has_cuda, reason="Parallel tree learners are CPU-only")
@pytest.mark.parametrize("extra", [{"lambda_l2": 50.0}, {"max_delta_step": 0.1}])
def test_data_parallel_matches_serial_with_regularization(extra):
    X, y = _data()
    expected = flc.train(_params(extra), flc.Dataset(X, label=y, params=_params(extra)), 10).predict(X)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    ports = _free_ports(2)
    procs = [ctx.Process(target=_worker, args=(r, ports, extra, queue)) for r in range(2)]
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
