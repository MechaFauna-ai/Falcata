# coding: utf-8
"""Test collection must not depend on PYTHONHASHSEED.

pytest-xdist workers each collect the suite in their own process, with their own string-hash seed, and refuse to
run when the collected lists differ. Parametrizing over a set of strings gives a seed-dependent order.
"""

import os
import subprocess
import sys
from pathlib import Path


def _collected(hash_seed):
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed)}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(Path(__file__).parent)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if "::" in line]


def test_collection_order_does_not_depend_on_hash_seed():
    first = _collected(1)
    assert first
    assert _collected(2) == first
