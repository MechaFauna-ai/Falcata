"""An allocation failure while saving a model must be an error, not an abort.

SaveModelToString builds the tree strings in an OpenMP loop without the
exception guards, so a std::bad_alloc there (tree strings get large: a
categorical split over large category ids writes the whole bitset) escaped
the parallel region and called std::terminate, killing the process.
"""

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
    x = np.zeros((400, 1))
    x[:, 0] = rng.choice([0, 1, 2**27], size=400)
    y = (x[:, 0] == 2**27).astype(float)
    params = {"verbose": -1, "num_threads": 2, "min_data_in_leaf": 1, "max_cat_to_onehot": 1, "num_leaves": 2}
    booster = flc.train(params, flc.Dataset(x, label=y, categorical_feature=[0]), num_boost_round=1)
    # No warm-up save: its freed buffers would let later saves fit. The model
    # text is ~8 MB, built in several copies, so sweeping the headroom puts
    # the failing allocation inside the per-tree loop for some of the steps.
    vm = int(open("/proc/self/statm").read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    for headroom_mb in range(2, 34, 2):
        resource.setrlimit(resource.RLIMIT_AS, (vm + (headroom_mb << 20), resource.RLIM_INFINITY))
        try:
            booster.model_to_string()
        except flc.basic.FalcataError as err:
            print("CLEAN", headroom_mb, err)
        else:
            print("SAVED", headroom_mb)
    """
)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc and RLIMIT_AS")
def test_model_to_string_bad_alloc_is_a_clean_error():
    proc = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "CLEAN" in proc.stdout, proc.stdout + proc.stderr[-2000:]
