"""Empty batches preserve prediction output dimensions."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import falcata as flc
from falcata.basic import _fil_run_host_slabs


@pytest.mark.parametrize("kind", ["normal", "raw", "leaf", "contrib"])
@pytest.mark.parametrize("multiclass", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_empty_predictions(kind, sparse, multiclass):
    data = np.arange(80, dtype=np.float64).reshape(40, 2)
    params = {"num_threads": 2, "verbosity": -1, "min_data_in_leaf": 1}
    if multiclass:
        params.update(objective="multiclass", num_class=3)
    booster = flc.train(
        params,
        flc.Dataset(data, label=np.arange(40) % 3 if multiclass else np.arange(40)),
        num_boost_round=2,
    )
    options = {
        "pred_leaf": kind == "leaf",
        "pred_contrib": kind == "contrib",
        "raw_score": kind == "raw",
        "use_fil": False,
    }
    batch = csr_matrix(data) if sparse else data
    prediction = booster.predict(batch[:0], **options)
    expected = booster.predict(batch[:1], **options)
    if isinstance(prediction, list):
        assert len(prediction) == len(expected)
        for empty, one_row in zip(prediction, expected, strict=True):
            assert empty.shape == (0, *one_row.shape[1:])
    else:
        assert prediction.shape == (0, *expected.shape[1:])


def test_empty_fil_host_input_falls_back():
    assert _fil_run_host_slabs(object(), np.empty((0, 2))) is None
