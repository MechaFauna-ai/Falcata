"""Sparse dimension bounds survive optimized Python."""

import subprocess
import sys


def test_csr_column_limit_under_optimized_python():
    code = """
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import falcata as flc
X = np.arange(80, dtype=np.float64).reshape(40, 2)
bst = flc.train({"device_type": "cpu", "num_threads": 2, "verbosity": -1}, flc.Dataset(X, label=np.arange(40)), num_boost_round=2)
sparse = csr_matrix(([1.0], ([0], [2**31])), shape=(1, 2**31 + 1))
def rejects(operation, message):
    try:
        operation()
    except ValueError as error:
        if message not in str(error):
            raise
    else:
        raise AssertionError("oversized sparse dimension accepted")
for contrib in [False, True]:
    rejects(lambda: bst.predict(sparse, use_fil=False, pred_contrib=contrib), "more than 2147483647 columns")
rejects(lambda: flc.Dataset(sparse, params={"device_type": "cpu"}).construct(), "more than 2147483647 columns")
tall = csc_matrix(([1.0], ([2**31], [0])), shape=(2**31 + 1, 1))
rejects(lambda: flc.Dataset(tall, params={"device_type": "cpu"}).construct(), "more than 2147483647 rows")
"""
    result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
