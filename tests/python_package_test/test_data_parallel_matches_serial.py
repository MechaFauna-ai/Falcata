# coding: utf-8
"""Data-parallel training on a row split matches serial training on all rows.

The data-parallel learner kept the children counts inferred from the histograms (sum of hessians times
the leaf's rows per hessian), which are exact only for a constant hessian, where serial training uses
the partition's actual counts. With a multiclass (or any non-constant hessian) objective the
min_data_in_leaf decisions then differed from serial training after a few iterations. The features take
8 values, so every machine finds the same bins as serial training.
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


def _data(objective):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 8, size=(2000, 5)).astype(float)
    y = X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=2000)
    if objective == "binary":
        y = (y > 0).astype(float)
    else:
        y = np.digitize(y, [-2, 2]).astype(float)
    return X, y


def _params(objective):
    params = {
        "objective": objective,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "num_threads": 1,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    if objective != "binary":
        params["num_class"] = 3
    return params


def _worker(rank, ports, objective, queue):
    X, y = _data(objective)
    part = np.arange(len(y)) % 2 == rank
    params = {
        **_params(objective),
        "tree_learner": "data",
        "num_machines": 2,
        "machines": ",".join(f"127.0.0.1:{p}" for p in ports),
        "local_listen_port": ports[rank],
        "pre_partition": True,
    }
    booster = flc.train(params, flc.Dataset(X[part], label=y[part], params=params), 10)
    queue.put((rank, booster.predict(X)))


@pytest.mark.skipif(BuildInfo.has_cuda, reason="Parallel tree learners are CPU-only")
@pytest.mark.parametrize("objective", ["binary", "multiclass", "multiclassova"])
def test_data_parallel_matches_serial(objective):
    X, y = _data(objective)
    expected = flc.train(_params(objective), flc.Dataset(X, label=y, params=_params(objective)), 10).predict(X)
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
        np.testing.assert_allclose(results[rank], expected, rtol=0, atol=1e-12)
