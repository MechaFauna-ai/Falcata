# coding: utf-8
"""Data- and voting-parallel training with interaction_constraints.

At the root there is no larger leaf (its index is -1), and the parallel learners still asked
ColSampler::GetByNode for its features, which reads the tree's branch features at index -1 when
interaction constraints are set (heap-buffer-overflow under ASan, std::bad_array_new_length in a
plain build). Two local processes must train, agree, and keep each branch within one constraint.
"""

import multiprocessing as mp
import re
import socket

import numpy as np
import pytest

import falcata as flc

from .utils import BuildInfo

CONSTRAINTS = [[0, 1], [2, 3, 4, 5]]


def _free_ports(n):
    socks = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def _worker(rank, ports, tree_learner, queue):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    y = X[:, 0] * X[:, 1] + X[:, 2] + X[:, 3]
    part = slice(rank * 300, (rank + 1) * 300)
    params = {
        "objective": "regression",
        "tree_learner": tree_learner,
        "num_machines": 2,
        "machines": ",".join(f"127.0.0.1:{p}" for p in ports),
        "local_listen_port": ports[rank],
        "num_leaves": 7,
        "interaction_constraints": CONSTRAINTS,
        "top_k": 2,
        "num_threads": 1,
        "verbose": -1,
    }
    booster = flc.train(params, flc.Dataset(X[part], label=y[part], params=params), 5)
    queue.put((rank, booster.model_to_string(), booster.dump_model()["tree_info"]))


def _branch_feature_sets(node, path=()):
    if "split_feature" not in node:
        yield set(path)
        return
    path = (*path, node["split_feature"])
    yield from _branch_feature_sets(node["left_child"], path)
    yield from _branch_feature_sets(node["right_child"], path)


@pytest.mark.skipif(BuildInfo.has_cuda, reason="Parallel tree learners are CPU-only")
@pytest.mark.parametrize("tree_learner", ["data", "voting"])
def test_parallel_learners_with_interaction_constraints(tree_learner):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    ports = _free_ports(2)
    procs = [ctx.Process(target=_worker, args=(r, ports, tree_learner, queue)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in procs:
            rank, model, trees = queue.get(timeout=120)
            results[rank] = (model, trees)
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():  # a hung worker would keep the test process from exiting
                p.terminate()
                p.join()
    assert [p.exitcode for p in procs] == [0, 0]

    def structure(model):
        # voting-parallel internal node statistics are local to each machine
        model = model.split("parameters:")[0]
        return re.sub(r"(internal_value|internal_weight|internal_count|tree_sizes)=[^\n]*", "", model)

    assert structure(results[0][0]) == structure(results[1][0])
    num_splits = 0
    for tree in results[0][1]:
        for features in _branch_feature_sets(tree["tree_structure"]):
            num_splits += len(features)
            assert any(features <= set(group) for group in CONSTRAINTS), features
    assert num_splits > 0
