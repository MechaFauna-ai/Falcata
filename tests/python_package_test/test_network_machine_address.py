# coding: utf-8
"""A machine list entry that is not a numeric IPv4 address is rejected.

TcpSocket::GetAddress ignored inet_pton's result, so an unparsable address
left the zeroed sockaddr (0.0.0.0) in place and Connect silently targeted
the local host, retrying for minutes before giving up.

Linkers::ListenThread also spun forever on a peer that closed before sending
its rank, and counted a duplicate rank as a new connection.
"""

import socket
import struct
import subprocess
import sys
import textwrap
import time

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


# Linkers::ListenThread handshake: the process under test is the highest rank,
# so every other rank connects in to it and sends its rank; the test plays
# those peers over plain sockets.
_NETWORK_CHILD = textwrap.dedent(
    """
    import ctypes
    import sys

    from falcata.basic import _LIB, _c_str, _safe_call

    machines, port, num_machines = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    _safe_call(_LIB.FLC_NetworkInit(_c_str(machines), ctypes.c_int(port), ctypes.c_int(1), ctypes.c_int(num_machines)))
    print("network ready", flush=True)
    _safe_call(_LIB.FLC_NetworkFree())
    """
)


def _start_top_rank(num_machines):
    ports = [_free_port() for _ in range(num_machines)]
    machines = ",".join(f"127.0.0.1:{p}" for p in ports)
    proc = subprocess.Popen(
        [sys.executable, "-c", _NETWORK_CHILD, machines, str(ports[-1]), str(num_machines)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, ports[-1]


def _connect(port):
    deadline = time.monotonic() + 30
    while True:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=10)
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)


def _send_rank(port, rank):
    s = _connect(port)
    s.sendall(struct.pack("=i", rank))
    return s


def _finish(proc, timeout):
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        pytest.fail(f"network init did not finish within {timeout} s:\n{out[-2000:]}")
    return out


def test_peer_closing_before_its_rank_is_dropped():
    # recv() == 0 used to leave read_cnt unchanged, so the listener spun on the
    # closed connection forever and the real rank 0 was never accepted
    proc, port = _start_top_rank(2)
    peers = []
    try:
        _connect(port).close()
        peers.append(_send_rank(port, 0))
        out = _finish(proc, timeout=30)
    finally:
        for s in peers:
            s.close()
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, out
    assert "network ready" in out


def test_duplicate_rank_does_not_complete_the_handshake():
    # a second connection claiming rank 0 used to replace the first linker and
    # be counted, so initialization finished with rank 1 never connected
    proc, port = _start_top_rank(3)
    peers = []
    try:
        peers.append(_send_rank(port, 0))
        peers.append(_send_rank(port, 0))
        time.sleep(2)
        assert proc.poll() is None, f"network init finished without rank 1:\n{proc.communicate()[0][-2000:]}"
        peers.append(_send_rank(port, 1))
        out = _finish(proc, timeout=30)
    finally:
        for s in peers:
            s.close()
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, out
    assert "network ready" in out
