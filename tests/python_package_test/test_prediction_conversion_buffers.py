"""Training and validation score conversion must agree with raw-data prediction."""

import ctypes

import numpy as np
import pytest

import falcata as lgb
from falcata.basic import _LIB, _safe_call


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("objective", ["regression", "binary", "multiclass", "multiclassova"])
def test_prediction_conversion_buffers(objective, threads):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(128, 4))
    classes = 10 if objective.startswith("multiclass") else 1
    y = np.arange(len(x)) % (classes if classes > 1 else 2)
    training = lgb.Dataset(x, label=y)
    validation = lgb.Dataset(x[::2], label=y[::2], reference=training)
    params = {"objective": objective, "num_threads": threads, "verbosity": -1, "num_leaves": 3}
    if classes > 1:
        params["num_class"] = classes
    model = lgb.train(
        params,
        training,
        num_boost_round=3,
        valid_sets=[training, validation],
        keep_training_booster=True,
    )
    for data_index, values in enumerate([x, x[::2]]):
        output = np.empty(len(values) * classes)
        length = ctypes.c_int64()
        for _ in range(2):
            _safe_call(
                _LIB.FLC_BoosterGetPredict(
                    model._handle,
                    data_index,
                    ctypes.byref(length),
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                )
            )
            assert length.value == output.size
            actual = output.reshape(classes, -1).T if classes > 1 else output
            np.testing.assert_array_equal(actual, model.predict(values, num_threads=threads))
