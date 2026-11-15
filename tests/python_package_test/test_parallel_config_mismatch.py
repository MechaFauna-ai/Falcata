# coding: utf-8
"""Parallel training with a structural parameter that differs between machines is a clear error.

The data- and voting-parallel learners size their exchange buffers from num_leaves, max_cat_threshold and
top_k and split the leaves the machines agree on. A machine with a smaller num_leaves than its peer split a
leaf past its arrays (SEGV in SerialTreeLearner::SplitInner); now every machine stops with the same error.
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


def _worker(rank, ports, tree_learner, overrides, queue):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    part = slice(rank * 300, (rank + 1) * 300)
    params = {
        "objective": "binary",
        "tree_learner": tree_learner,
        "num_machines": 2,
        "machines": ",".join(f"127.0.0.1:{p}" for p in ports),
        "local_listen_port": ports[rank],
        "num_leaves": 4,
        "min_data_in_leaf": 5,
        "num_threads": 1,
        "verbose": -1,
        **overrides[rank],
    }
    try:
        flc.train(params, flc.Dataset(X[part], label=y[part], params=params), 3)
        queue.put((rank, "trained"))
    except flc.basic.FalcataError as e:
        queue.put((rank, str(e)))


@pytest.mark.skipif(BuildInfo.has_cuda, reason="Parallel tree learners are CPU-only")
@pytest.mark.parametrize(
    ("tree_learner", "overrides", "name"),
    [
        ("data", [{}, {"num_leaves": 63}], "num_leaves"),
        ("data", [{"max_depth": 2}, {}], "max_depth"),
        ("voting", [{"top_k": 1}, {"top_k": 5}], "top_k"),
    ],
)
def test_parallel_parameter_mismatch_is_an_error(tree_learner, overrides, name):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    ports = _free_ports(2)
    procs = [ctx.Process(target=_worker, args=(r, ports, tree_learner, overrides, queue)) for r in range(2)]
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
        assert f"Parameter {name} differs across machines" in results[rank]
