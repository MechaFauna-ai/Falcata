# coding: utf-8
"""Distributed bin finding applies each feature's own forced bins.

With more than one machine, each machine finds the bins of a slice of the features
[start[rank], start[rank] + len[rank]) and the bin mappers are gathered. The slice loop passed
forced_bin_bounds[i] (and, from a text file, max_bin_by_feature[i]) with the local index i, so
every machine but rank 0 binned its features with the forced bins of the first features. Two
local processes that hold the full data must bin it exactly like a serial Dataset.
"""

import json
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
    X = rng.normal(size=(500, 4))
    return X, X[:, 2] + X[:, 3]


def _binned_rows(dataset, path):
    dataset._dump_text(path)
    return path.read_text().split("forced_bins:")[1]


def _params(tmp_path):
    forced = tmp_path / "forced.json"
    # features 2 and 3 are binned by rank 1; features 0 and 1 have no forced bins
    forced.write_text(
        json.dumps(
            [
                {"feature": 2, "bin_upper_bound": [-1.5, -1.0, -0.5, 0.25, 0.3, 0.35]},
                {"feature": 3, "bin_upper_bound": [0.1, 0.2]},
            ]
        )
    )
    return {"forcedbins_filename": str(forced), "max_bin": 7, "num_threads": 1, "verbose": -1}


def _worker(rank, ports, params, data_file, out_path, queue):
    params = {
        **params,
        "tree_learner": "data",
        "num_machines": 2,
        "machines": ",".join(f"127.0.0.1:{p}" for p in ports),
        "local_listen_port": ports[rank],
        "pre_partition": True,  # each process loads the full file
    }
    if data_file is None:
        X, y = _data()
        dataset = flc.Dataset(X, label=y, params=params)
    else:
        dataset = flc.Dataset(data_file, params=params)
    # the Booster initialises the network before it constructs the Dataset
    flc.train({**params, "objective": "regression", "num_leaves": 4}, dataset, 1)
    queue.put((rank, _binned_rows(dataset, out_path)))


def _distributed_bins(tmp_path, params, data_file=None):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    ports = _free_ports(2)
    procs = [
        ctx.Process(target=_worker, args=(r, ports, params, data_file, tmp_path / f"rank{r}.txt", queue))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    bins = dict(queue.get(timeout=120) for _ in procs)
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    return bins


@pytest.mark.skipif(BuildInfo.has_cuda, reason="data-parallel training is CPU-only")
@pytest.mark.parametrize("source", ["matrix", "text_file", "text_file_max_bin_by_feature"])
def test_distributed_bins_use_each_features_forced_bins(tmp_path, source):
    params = _params(tmp_path)
    if source == "text_file_max_bin_by_feature":
        params["max_bin_by_feature"] = [7, 7, 7, 3]
    X, y = _data()
    data_file = None
    if source.startswith("text_file"):
        data_file = tmp_path / "train.csv"
        np.savetxt(data_file, np.column_stack([y, X]), delimiter=",")
        data_file = str(data_file)
        serial = flc.Dataset(data_file, params=params).construct()
    else:
        serial = flc.Dataset(X, label=y, params=params).construct()
    expected = _binned_rows(serial, tmp_path / "serial.txt")

    bins = _distributed_bins(tmp_path, params, data_file)
    assert bins[0] == expected
    assert bins[1] == expected
