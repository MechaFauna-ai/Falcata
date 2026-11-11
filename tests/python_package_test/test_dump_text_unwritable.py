# coding: utf-8
"""Dumping a Dataset to a path that cannot be opened raises instead of crashing.

Dataset::DumpTextFile used fopen's result unchecked, so an unwritable path
reached fprintf(NULL, ...) and killed the process. The dump runs in a child
process so that the crash on an unfixed library fails this test instead of
taking pytest down with it.
"""

import subprocess
import sys
import textwrap

_CHILD = textwrap.dedent(
    """
    import sys

    import numpy as np

    import falcata as flc

    X = np.random.default_rng(0).normal(size=(50, 3))
    ds = flc.Dataset(X, label=X[:, 0], params={"verbose": -1}).construct()
    try:
        ds._dump_text(sys.argv[1])
    except flc.basic.FalcataError as err:
        print("raised:", err)
        sys.exit(0)
    print("no error")
    sys.exit(3)
    """
)


def test_dump_text_to_unopenable_path_raises(tmp_path):
    target = tmp_path / "missing_dir" / "dump.txt"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"
    assert "Could not open" in proc.stdout
