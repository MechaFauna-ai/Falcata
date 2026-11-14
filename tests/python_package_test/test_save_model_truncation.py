"""A model serialized under memory pressure must fail, never come back truncated.

std::stringstream swallows an allocation failure inside operator<<: it sets
badbit and drops every later insertion. SaveModelToString then returned the
truncated text as a success, so model_to_string()/save_model() handed out a
model that no longer loads ("invalid tree_sizes entry") -- or, had the cut
fallen inside a tree, one that predicts differently.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("resource")

_CHILD = textwrap.dedent(
    """
    import os, resource
    import numpy as np
    import falcata as flc

    rng = np.random.default_rng(0)
    x = np.zeros((400, 2))
    x[:, 0] = rng.choice([0, 1, 2**27], size=400)
    x[:, 1] = rng.standard_normal(400)
    y = (x[:, 0] == 2**27).astype(float) + 0.01 * x[:, 1]
    params = {"verbose": -1, "num_threads": 1, "min_data_in_leaf": 1, "max_cat_to_onehot": 1, "num_leaves": 2}
    # each tree holds a 4M-word categorical bitset, ~8 MB of text
    booster = flc.train(params, flc.Dataset(x, label=y, categorical_feature=[0]), num_boost_round=3)
    vm = int(open("/proc/self/statm").read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    for headroom_mb in range(4, 122, 4):
        resource.setrlimit(resource.RLIMIT_AS, (vm + (headroom_mb << 20), resource.RLIM_INFINITY))
        try:
            text = booster.model_to_string()
        except (flc.basic.FalcataError, MemoryError):
            continue
        finally:
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        try:
            flc.Booster(model_str=text)
        except flc.basic.FalcataError as err:
            print("CORRUPT", headroom_mb, len(text), err)
    print("DONE")
    """
)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc and RLIMIT_AS")
def test_model_to_string_never_returns_truncated_text():
    # a fixed mmap threshold keeps the saves allocating fresh mappings (glibc
    # otherwise raises it after training and reuses freed heap)
    env = dict(os.environ, MALLOC_MMAP_THRESHOLD_="131072")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=600, check=False, env=env
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "DONE" in proc.stdout, proc.stdout + proc.stderr[-2000:]
    assert "CORRUPT" not in proc.stdout, proc.stdout
