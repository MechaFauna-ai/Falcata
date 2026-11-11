"""Checking eval-set placement must not gather training partitions."""

import numpy as np
import pytest

import falcata as flc


def test_dask_eval_placement_gathers_only_metadata(monkeypatch):
    da = pytest.importorskip("dask.array")
    distributed = pytest.importorskip("distributed")
    original = distributed.Future.result

    def result(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        assert not (isinstance(value, dict) and "data" in value), "training partition gathered to client"
        return value

    monkeypatch.setattr(distributed.Future, "result", result)
    data = da.from_array(np.arange(160, dtype=np.float64).reshape(80, 2), chunks=(40, 2))
    labels = da.from_array(np.arange(80, dtype=np.float64), chunks=40)
    with distributed.Client(processes=False, n_workers=1, threads_per_worker=1, dashboard_address=None) as client:
        model = flc.DaskFalcataRegressor(client=client, n_estimators=2, n_jobs=1, verbosity=-1)
        model.fit(data, labels, eval_set=[(data, labels)])
