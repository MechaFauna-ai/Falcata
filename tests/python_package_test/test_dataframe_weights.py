"""DataFrame weights follow sklearn's sample-weight validation contract."""

import numpy as np
import pandas as pd
import pytest

import falcata as flc


@pytest.mark.parametrize("dataframe", [False, True])
def test_scalar_sample_weight_is_broadcast(dataframe):
    data = np.arange(80, dtype=np.float64).reshape(40, 2)
    X = pd.DataFrame(data) if dataframe else data
    model = flc.FalcataRegressor(n_estimators=2, n_jobs=2, verbosity=-1).fit(X, np.arange(40), sample_weight=2.0)
    assert model.predict(X).shape == (40,)
