# coding: utf-8
"""reset_parameter(top_k=...) takes effect in the voting-parallel learner.

VotingParallelTreeLearner set top_k_ (and sized its collective buffers from
top_k and max_cat_threshold) only at construction/Init; ResetConfig kept the
stale value, so a top_k reset was silently ignored. Two local processes
train with tree_learner=voting: starting at top_k=20 must give the same model
as starting at top_k=1 and resetting to 20 before the first iteration.
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


def _worker(rank, ports, initial_top_k, reset_top_k, queue):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 30))
    y = X[:, :10] @ rng.normal(size=10) + 0.1 * rng.normal(size=4000)
    part = slice(rank * 2000, (rank + 1) * 2000)
    params = {
        "objective": "regression",
        "tree_learner": "voting",
        "top_k": initial_top_k,
        "num_machines": 2,
        "machines": ",".join(f"127.0.0.1:{p}" for p in ports),
        "local_listen_port": ports[rank],
        "pre_partition": True,
        "num_leaves": 15,
        "num_threads": 1,
        "deterministic": True,
        "verbose": -1,
    }
    callbacks = [] if reset_top_k is None else [flc.reset_parameter(top_k=[reset_top_k] * 5)]
    booster = flc.train(params, flc.Dataset(X[part], label=y[part], params=params), 5, callbacks=callbacks)
    queue.put((rank, booster.model_to_string()))


def _train(initial_top_k, reset_top_k=None):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    ports = _free_ports(2)
    procs = [ctx.Process(target=_worker, args=(r, ports, initial_top_k, reset_top_k, queue)) for r in range(2)]
    for p in procs:
        p.start()
    models = dict(queue.get(timeout=120) for _ in procs)
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    return models[0].split("parameters:")[0]


@pytest.mark.skipif(BuildInfo.has_cuda, reason="Voting-parallel training is not supported by CUDA version")
def test_reset_top_k_takes_effect():
    small = _train(1)
    large = _train(20)
    assert small != large, "top_k must matter for this data, or the test proves nothing"
    assert _train(1, reset_top_k=20) == large
