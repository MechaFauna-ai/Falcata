"""Sparse multiclass contributions must include all feature chunks."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import falcata as flc
from falcata.dask import _predict


@pytest.mark.parametrize("feature_chunk", [1, 4])
def test_multiclass_sparse_contrib_column_chunks(feature_chunk):
    da = pytest.importorskip("dask.array")
    rng = np.random.default_rng(42)
    data = rng.normal(size=(60, 4))
    labels = np.arange(60) % 3
    model = flc.FalcataClassifier(n_estimators=2, n_jobs=2, min_child_samples=1, verbosity=-1).fit(data, labels)
    sparse = csr_matrix(data)
    chunks = da.from_array(sparse, chunks=(30, feature_chunk), asarray=False)
    predicted = _predict(model=model, data=chunks, client=None, pred_contrib=True)
    expected = model.predict(sparse, pred_contrib=True)
    for actual, reference in zip(predicted, expected, strict=True):
        np.testing.assert_array_equal(actual.compute(scheduler="synchronous").toarray(), reference.toarray())
