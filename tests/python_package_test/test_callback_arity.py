"""Signature inspection belongs to callback setup, outside the boosting loop."""

from unittest.mock import Mock

import numpy as np
import pytest

import falcata.sklearn as sk


@pytest.mark.parametrize("wrapper", [sk._ObjectiveFunctionWrapper, sk._EvalFunctionWrapper])
@pytest.mark.parametrize("argc", [2, 3, 4])
def test_callback_arity_is_inspected_once(wrapper, argc, monkeypatch):
    callbacks = {
        2: lambda y, p: (y, p),
        3: lambda y, p, w: (y, p),
        4: lambda y, p, w, g: (y, p),
    }
    signature = Mock(wraps=sk.signature)
    monkeypatch.setattr(sk, "signature", signature)
    wrapped = wrapper(callbacks[argc])
    ds = Mock()
    labels = np.arange(4, dtype=float)
    preds = labels + 1
    ds.get_label.return_value = labels
    ds.get_weight.return_value = None
    ds.get_group.return_value = None
    expected = callbacks[argc](labels, preds, *([None] * (argc - 2)))
    for _ in range(3):
        got = wrapped(preds, ds)
        assert all(a is b for a, b in zip(got, expected, strict=True))
    signature.assert_called_once_with(callbacks[argc])
