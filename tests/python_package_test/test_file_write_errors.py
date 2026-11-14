# coding: utf-8
"""A native file write that fails must raise, not leave a silently incomplete file.

LocalFile::Write only checked fwrite, and fwrite of a small payload just fills the
stdio buffer: on a full disk the bytes are lost in the unchecked fclose. So
Dataset.save_binary, FLC_BoosterSaveModel (Booster.save_model text, the R package,
the CLI) and the CLI prediction output reported success for a truncated file.

The child caps its file size with RLIMIT_FSIZE (SIGXFSZ ignored, so a write past the
cap fails with EFBIG), which fails writes the way a full disk does.
"""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("resource")

_CHILD = textwrap.dedent(
    """
    import os, resource, signal, sys
    import numpy as np
    import falcata as flc

    out_dir = sys.argv[1]
    rng = np.random.default_rng(0)
    X = rng.random((50, 2))
    y = X[:, 0] + 0.1 * rng.random(50)
    ds = flc.Dataset(X, y, params={"verbose": -1}).construct()
    booster = flc.train({"verbose": -1, "num_leaves": 2}, flc.Dataset(X, y), 1)

    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    # every payload is a few KB: it fits in the stdio buffer, so only the close can see the failure
    resource.setrlimit(resource.RLIMIT_FSIZE, (512, resource.RLIM_INFINITY))

    def native_model_save(path):
        # the native text save alone: Booster.save_model would also append pandas_categorical from Python
        flc.basic._safe_call(flc.basic._LIB.FLC_BoosterSaveModel(
            booster._handle, 0, -1, 0, flc.basic._c_str(path)))

    for name, save in [("binary", ds.save_binary), ("model", native_model_save)]:
        path = os.path.join(out_dir, name)
        try:
            save(path)
            print(name, "SILENT", os.path.getsize(path))
        except flc.basic.FalcataError as err:
            print(name, "RAISED", str(err).splitlines()[0])
    """
)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs RLIMIT_FSIZE semantics of Linux")
def test_failed_native_file_writes_raise(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(tmp_path)], capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = dict(line.split(" ", 1) for line in proc.stdout.splitlines())
    assert lines["binary"].startswith("RAISED Cannot write binary data"), proc.stdout
    assert lines["model"].startswith("RAISED Cannot write model"), proc.stdout
