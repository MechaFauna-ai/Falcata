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


_PREDICT_CHILD = textwrap.dedent(
    """
    import os
    import numpy as np
    import falcata as flc

    X = np.random.default_rng(0).random((500, 4))
    booster = flc.train({"verbose": -1, "num_threads": 4, "device_type": "cpu"}, flc.Dataset(X, X[:, 0].copy()), 3)
    limit = max(1024, 4 * os.cpu_count())
    reg = flc.FalcataRegressor(n_estimators=3, n_jobs=4, device_type="cpu", verbose=-1).fit(X, X[:, 0])
    cases = {
        "num_threads": lambda: booster.predict(X, num_threads=100000),
        "n_jobs": lambda: booster.predict(X, n_jobs=100000),
        "device_cpu": lambda: booster.predict(X, device="cpu", num_threads=100000),
        "over_limit": lambda: booster.predict(X, num_threads=limit + 1),
        "sklearn": lambda: reg.set_params(n_jobs=100000).predict(X),
    }
    for name, call in cases.items():
        try:
            call()
            print("PREDICTED", name)
        except flc.basic.FalcataError as err:
            print("RAISED", name, err)
    ref = booster.predict(X, num_threads=1)
    print("LIMIT_OK", bool(np.array_equal(booster.predict(X, num_threads=limit), ref)))
    """
)


def test_huge_num_threads_on_predict_raises_instead_of_crashing():
    """Predictions set the OpenMP thread count from their own parameters: the same limit as training applies, and a
    value at the limit still predicts (the same bits as one thread)."""
    proc = subprocess.run(
        [sys.executable, "-c", _PREDICT_CHILD], capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in ("num_threads", "n_jobs", "device_cpu", "over_limit", "sklearn"):
        assert f"RAISED {name} num_threads=" in proc.stdout, proc.stdout
    assert "is more than the supported" in proc.stdout, proc.stdout
    assert "LIMIT_OK True" in proc.stdout, proc.stdout
