# coding: utf-8
"""A Dataset built against a reference must have the reference's column count.

With a reference the new Dataset takes the reference's bin mappers, one per column. A CSC
matrix with more columns wrote past them (ASan: heap-buffer-overflow in
FLC_DatasetCreateFromCSC), and dense and CSR input with more or fewer columns was accepted
silently, putting values on the wrong features or dropping them. The Arrow path has the
same check (#539).
"""

import numpy as np
import pytest
import scipy.sparse as sp

import falcata as flc


@pytest.fixture
def reference():
    rng = np.random.default_rng(0)
    X = rng.random((200, 3))
    return flc.Dataset(X, X[:, 0], params={"verbose": -1}).construct()


@pytest.mark.parametrize("ncol", [2, 4])
@pytest.mark.parametrize(
    ("kind", "make", "what"),
    [
        ("dense", np.asarray, "The matrix"),
        ("fortran", np.asfortranarray, "The matrix"),
        ("csr", sp.csr_matrix, "The CSR matrix"),
        ("csc", sp.csc_matrix, "The CSC matrix"),
    ],
)
def test_column_count_must_match_reference(reference, kind, make, what, ncol):
    data = make(np.random.default_rng(1).random((50, ncol)))
    with pytest.raises(flc.basic.FalcataError, match=f"{what} has {ncol} columns, but its reference Dataset has 3"):
        flc.Dataset(data, np.zeros(50), reference=reference).construct()


def test_matching_column_count_still_works(reference):
    for make in (np.asarray, sp.csr_matrix, sp.csc_matrix):
        data = make(np.random.default_rng(1).random((50, 3)))
        assert flc.Dataset(data, np.zeros(50), reference=reference).construct().num_data() == 50
