"""DataFrame weights follow sklearn's sample-weight validation contract."""

import numpy as np
import pytest

import falcata as flc

pd = pytest.importorskip("pandas")


@pytest.mark.parametrize("dataframe", [False, True])
def test_scalar_sample_weight_is_broadcast(dataframe):
    data = np.arange(80, dtype=np.float64).reshape(40, 2)
    X = pd.DataFrame(data) if dataframe else data
    model = flc.FalcataRegressor(n_estimators=2, n_jobs=2, verbosity=-1).fit(X, np.arange(40), sample_weight=2.0)
    vector = flc.FalcataRegressor(n_estimators=2, n_jobs=2, verbosity=-1).fit(
        X, np.arange(40), sample_weight=np.full(40, 2.0)
    )
    np.testing.assert_array_equal(model.predict(X), vector.predict(X))
    with pytest.raises(ValueError, match="sample_weight"):
        flc.FalcataRegressor(n_estimators=2, n_jobs=2, verbosity=-1).fit(X, np.arange(40), sample_weight=np.ones(3))
