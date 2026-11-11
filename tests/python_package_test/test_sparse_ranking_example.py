"""Keep the example's CSR feature partitions sparse."""

import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix, isspmatrix_csr


def test_ranking_example_partitions_stay_sparse():
    da = pytest.importorskip("dask.array")
    path = Path(__file__).resolve().parents[2] / "examples" / "python-guide" / "dask" / "ranking.py"
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "dX" for t in n.targets)
    )
    x = csr_matrix(np.eye(8, 4))
    ns = {"da": da, "X": x, "rows_in_part1": 3, "rows_in_part2": 5, "num_features": 4}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    first = ns["dX"].blocks[0, 0].compute(scheduler="synchronous")
    assert isspmatrix_csr(first)
    np.testing.assert_array_equal(first.toarray(), x[:3].toarray())
