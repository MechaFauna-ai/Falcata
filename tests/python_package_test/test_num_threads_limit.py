# coding: utf-8
"""An absurd num_threads is an error, not a segfault.

libgomp lays out a parallel team's start-up state on the calling thread's stack, so
num_threads=100000 overflowed the stack in the first parallel region (Dataset
construction) and killed the process. Large but sane values keep working.
"""

import subprocess
import sys
import textwrap

import numpy as np

import falcata as flc

_CHILD = textwrap.dedent(
    """
    import numpy as np
    import falcata as flc

    X = np.random.default_rng(0).random((500, 4))
    try:
        flc.train({"verbose": -1, "num_threads": 100000}, flc.Dataset(X, X[:, 0], params={"num_threads": 100000}), 1)
        print("TRAINED")
    except flc.basic.FalcataError as err:
        print("RAISED", err)
    """
)


def test_huge_num_threads_raises_instead_of_crashing():
    proc = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "RAISED num_threads=100000 is more than the supported" in proc.stdout, proc.stdout


def test_oversubscribed_num_threads_still_train():
    X = np.random.default_rng(0).random((500, 4))
    flc.train({"verbose": -1, "num_threads": 64}, flc.Dataset(X, X[:, 0], params={"num_threads": 64}), 1)
