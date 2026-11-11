# coding: utf-8
"""A machine list entry that is not a numeric IPv4 address is rejected.

TcpSocket::GetAddress ignored inet_pton's result, so an unparsable address
left the zeroed sockaddr (0.0.0.0) in place and Connect silently targeted
the local host, retrying for minutes before giving up.
"""

import socket

import numpy as np
import pytest

import falcata as flc
from falcata.basic import FalcataError


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_invalid_machine_address_raises():
    port = _free_port()
    X = np.random.default_rng(0).normal(size=(100, 2))
    params = {
        "objective": "regression",
        "tree_learner": "data",
        "num_machines": 2,
        "machines": f"127.0.0.1:{port},not-an-ip.invalid:{port + 1}",
        "local_listen_port": port,
        "verbose": -1,
    }
    try:
        with pytest.raises(FalcataError, match="Invalid machine address 'not-an-ip.invalid'"):
            flc.train(params, flc.Dataset(X, label=X[:, 0]), num_boost_round=1)
    finally:
        flc.basic._safe_call(flc.basic._LIB.FLC_NetworkFree())
