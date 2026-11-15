# coding: utf-8
"""FLC_BoosterPredictSparseOutput with zero rows on a multiclass model.

Booster::PredictSparseCSR computed where each class's matrix starts from the previous
matrix's last row, read at index row_vector_cnt - 1. With no rows that is index -1 of
an empty vector: the process segfaulted (UBSan: offset 2**64 - 8 applied to a null
pointer). The Python package returns early on empty input, so this is reached
through the C API (and bindings that pass empty batches through).
"""

import subprocess
import sys
import textwrap

_CHILD = textwrap.dedent(
    """
    import ctypes
    import numpy as np
    import falcata as flc
    from falcata.basic import _LIB, _c_str

    rng = np.random.default_rng(0)
    X = rng.random((300, 4))
    y = rng.integers(0, 3, 300)
    booster = flc.train({"objective": "multiclass", "num_class": 3, "verbose": -1}, flc.Dataset(X, y), 3)

    # CSR: 0 rows x 4 columns (indptr [0]); CSC: 4 columns of a 0-row matrix (indptr [0, 0, 0, 0, 0])
    for matrix_type, nindptr, num_col_or_row in [(0, 1, 4), (1, 5, 0)]:
        indptr = np.zeros(nindptr, dtype=np.int32)
        indices = np.zeros(1, dtype=np.int32)
        data = np.zeros(1, dtype=np.float64)
        out_len = (ctypes.c_int64 * 2)()
        out_indptr = ctypes.c_void_p()
        out_indices = ctypes.POINTER(ctypes.c_int32)()
        out_data = ctypes.c_void_p()
        rc = _LIB.FLC_BoosterPredictSparseOutput(
            booster._handle,
            indptr.ctypes.data_as(ctypes.c_void_p),
            2,  # C_API_DTYPE_INT32
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            data.ctypes.data_as(ctypes.c_void_p),
            1,  # C_API_DTYPE_FLOAT64
            ctypes.c_int64(nindptr),
            ctypes.c_int64(0),
            ctypes.c_int64(num_col_or_row),
            3,  # C_API_PREDICT_CONTRIB
            0,
            -1,
            _c_str(""),
            matrix_type,
            out_len,
            ctypes.byref(out_indptr),
            ctypes.byref(out_indices),
            ctypes.byref(out_data),
        )
        if rc != 0:
            print(matrix_type, "ERROR", _LIB.FLC_GetLastError().decode())
            continue
        print(matrix_type, "OK", out_len[0], out_len[1])
        _LIB.FLC_BoosterFreePredictSparse(out_indptr, out_indices, out_data, 2, 1)
    """
)


def test_sparse_contrib_with_zero_rows_multiclass():
    proc = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = proc.stdout.splitlines()
    # CSR: no elements, one empty indptr (a single 0) per class
    assert "0 OK 0 3" in lines, proc.stdout
    # CSC: no elements, indptr of length num_feature + 2 per class
    assert "1 OK 0 18" in lines, proc.stdout
