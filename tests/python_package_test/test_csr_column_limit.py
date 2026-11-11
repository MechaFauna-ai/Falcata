"""Oversized sparse columns are rejected even in optimized Python."""

import subprocess
import sys


def test_csr_column_limit_under_optimized_python():
    code = """
import numpy as np
from scipy.sparse import csr_matrix
import falcata as flc
X = np.arange(80, dtype=np.float64).reshape(40, 2)
bst = flc.train({"device_type": "cpu", "num_threads": 2, "verbosity": -1}, flc.Dataset(X, label=np.arange(40)), num_boost_round=2)
sparse = csr_matrix(([1.0], ([0], [2**31])), shape=(1, 2**31 + 1))
try:
    bst.predict(sparse, use_fil=False)
except ValueError as error:
    if "columns" not in str(error):
        raise
else:
    raise AssertionError("oversized sparse columns accepted")
"""
    result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
